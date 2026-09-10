#!/usr/bin/env python3
"""Render the whole SplatWorld3 address space as an operator atlas.

The interactive app shows one query plus four nearby queries. This script asks a
more global question: what entire visual world is exposed by one persistent
operator plate g across a grid of addresses?

Outputs:
  atlas_faces.png              - 2-D address map of decoded faces
  atlas_metrics.png            - dM(origin), cond(A), and |c| maps
  atlas_data.npz               - exact numeric atlas data

With --mutate:
  atlas_before.png
  atlas_after.png
  atlas_difference.png         - amplified absolute pixel change
  atlas_mutation_metrics.png   - per-address coefficient displacement |dc|
  atlas_mutation_summary.json

The mutation is one conservative edge->edge transfer chosen only to move the
operator family across the displayed atlas. It never looks at pixels, and total
material sum(g) remains constant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from operator_plate import OperatorPlate
from splatworld2 import OnnxDecoder, MockDecoder, find_model, probe, random_identity, rgb
from splatworld3 import operator_match_scale, operator_coeff, local_latent, visible_mutation

EPS = 1e-9
LATENT = 128


def cv2():
    import cv2 as cv
    return cv


def decode_batches(dec, latents, batch=64):
    outs = []
    for i in range(0, len(latents), batch):
        outs.append(dec(np.asarray(latents[i:i + batch], np.float32)))
    return np.concatenate(outs, axis=0)


def make_basis(dec, z0, args, verbose=True):
    if verbose:
        print(f"identity |z|={np.linalg.norm(z0):.3f} — probing {args.operator_dim} local modes")
    B, chosen, metrics = probe(
        dec,
        z0,
        n=args.probe_dirs,
        k=args.operator_dim,
        eps=args.probe_eps,
        seed=args.seed,
        verbose=verbose,
    )
    if verbose:
        print("selected transport axes:", chosen)
    return B


def address_grid(span, grid, z_value):
    xs = np.linspace(-span, span, grid, dtype=np.float32)
    # Positive y is the top row so the saved image reads like a map.
    ys = np.linspace(span, -span, grid, dtype=np.float32)
    addrs = np.asarray([[x, y, z_value] for y in ys for x in xs], np.float32)
    return xs, ys, addrs


def sample_state(dec, z0, B, plate, op_scale, args):
    xs, ys, addrs = address_grid(args.address_span, args.grid, args.z)
    M0 = plate.matrix((0.0, 0.0, 0.0))
    M0n = np.linalg.norm(M0) + EPS

    coeffs = []
    latents = []
    dm = []
    cond = []
    cnorm = []

    for a in addrs:
        c = operator_coeff(plate, a, op_scale)
        M = plate.matrix(a)
        coeffs.append(c)
        cnorm.append(float(np.linalg.norm(c)))
        dm.append(float(np.linalg.norm(M - M0) / M0n))
        cond.append(float(np.linalg.cond(plate.system_matrix(a))))
        latents.append(local_latent(z0, B, c, args.motion_gain))

    outs = decode_batches(dec, np.asarray(latents, np.float32), batch=args.batch)
    images = []
    cv = cv2()
    for out in outs:
        im = (255.0 * np.clip(rgb(out), 0.0, 1.0)).astype(np.uint8)
        images.append(cv.cvtColor(im, cv.COLOR_RGB2BGR))

    g = args.grid
    return {
        "xs": xs,
        "ys": ys,
        "addresses": addrs.reshape(g, g, 3),
        "coeffs": np.asarray(coeffs, np.float32).reshape(g, g, -1),
        "dm": np.asarray(dm, np.float64).reshape(g, g),
        "cond": np.asarray(cond, np.float64).reshape(g, g),
        "cnorm": np.asarray(cnorm, np.float64).reshape(g, g),
        "images": np.asarray(images, np.uint8).reshape(g, g, *images[0].shape),
    }


def put_text(img, text, xy, scale=0.45, color=(235, 235, 235), thick=1):
    cv = cv2()
    cv.putText(img, text, xy, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)


def face_atlas(state, tile=112, title="ONE g -> WHOLE ADDRESS WORLD"):
    cv = cv2()
    g = len(state["xs"])
    left = 72
    top = 70
    cell = tile
    canvas = np.full((top + g * cell + 28, left + g * cell + 18, 3), 24, np.uint8)

    put_text(canvas, title, (18, 28), 0.68, (80, 255, 80), 1)
    put_text(canvas, "columns = x address    rows = y address", (18, 51), 0.42, (190, 190, 190), 1)

    for j, x in enumerate(state["xs"]):
        put_text(canvas, f"{x:+.1f}", (left + j * cell + 5, top - 14), 0.33, (190, 190, 190), 1)
    for i, y in enumerate(state["ys"]):
        put_text(canvas, f"{y:+.1f}", (8, top + i * cell + 19), 0.33, (190, 190, 190), 1)

    for i in range(g):
        for j in range(g):
            im = cv.resize(state["images"][i, j], (tile, tile), interpolation=cv.INTER_CUBIC)
            y0 = top + i * cell
            x0 = left + j * cell
            canvas[y0:y0 + tile, x0:x0 + tile] = im
            # Tiny unobtrusive dM number: lets us see where visual transitions sit in operator space.
            put_text(canvas, f"{state['dm'][i,j]:.1f}", (x0 + 4, y0 + tile - 6), 0.30, (255, 255, 255), 1)

    put_text(canvas, "small white number in each tile = relative dM from origin", (18, canvas.shape[0] - 8),
             0.36, (180, 180, 180), 1)
    return canvas


def metric_panel(values, title, cell=58, log=False):
    cv = cv2()
    v = np.asarray(values, np.float64)
    shown = np.log10(np.maximum(v, EPS)) if log else v.copy()
    lo = float(np.nanmin(shown))
    hi = float(np.nanmax(shown))
    q = (shown - lo) / max(hi - lo, EPS)
    q = np.clip(q, 0.0, 1.0)
    raw = (255.0 * q).astype(np.uint8)
    heat = cv.applyColorMap(raw, cv.COLORMAP_TURBO)
    heat = cv.resize(heat, (v.shape[1] * cell, v.shape[0] * cell), interpolation=cv.INTER_NEAREST)

    top = 48
    panel = np.full((top + heat.shape[0], heat.shape[1], 3), 24, np.uint8)
    panel[top:] = heat
    put_text(panel, title, (10, 21), 0.49, (80, 255, 80), 1)
    unit = "log10 " if log else ""
    put_text(panel, f"{unit}range {lo:.3g} .. {hi:.3g}", (10, 41), 0.34, (205, 205, 205), 1)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            text = f"{v[i,j]:.1f}" if abs(v[i,j]) >= 10 else f"{v[i,j]:.2f}"
            put_text(panel, text, (j * cell + 4, top + i * cell + cell // 2 + 4),
                     0.29, (255, 255, 255), 1)
    return panel


def metrics_atlas(state, cell=58):
    cv = cv2()
    p1 = metric_panel(state["dm"], "relative ||M(a)-M(0)||", cell=cell)
    p2 = metric_panel(state["cond"], "condition number cond(A)", cell=cell, log=True)
    p3 = metric_panel(state["cnorm"], "operator coefficient norm |c|", cell=cell)
    gap = 12
    h = max(p.shape[0] for p in (p1, p2, p3))
    w = sum(p.shape[1] for p in (p1, p2, p3)) + 2 * gap
    out = np.full((h, w, 3), 18, np.uint8)
    x = 0
    for p in (p1, p2, p3):
        out[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    return out


def difference_state(before, after, gain=4.0):
    cv = cv2()
    b = before["images"].astype(np.int16)
    a = after["images"].astype(np.int16)
    diff = np.mean(np.abs(a - b), axis=-1)
    out_images = np.zeros_like(before["images"])
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            g = np.clip(diff[i, j] * float(gain), 0, 255).astype(np.uint8)
            out_images[i, j] = cv.applyColorMap(g, cv.COLORMAP_TURBO)
    st = dict(before)
    st["images"] = out_images
    st["dm"] = np.mean(np.abs(after["coeffs"] - before["coeffs"]), axis=-1)
    return st, diff


def save_state_data(path, state, z0, plate, op_scale):
    np.savez_compressed(
        path,
        xs=state["xs"], ys=state["ys"], addresses=state["addresses"],
        coeffs=state["coeffs"], dm=state["dm"], cond=state["cond"], cnorm=state["cnorm"],
        z0=np.asarray(z0, np.float32), material=np.asarray(plate.g, np.float64),
        op_scale=np.float64(op_scale),
    )


def render_and_save(dec, z0, B, plate, op_scale, args, outdir, stem="atlas"):
    cv = cv2()
    state = sample_state(dec, z0, B, plate, op_scale, args)
    faces = face_atlas(state, tile=args.tile, title=f"{stem.upper()} — ONE g / {args.grid}x{args.grid} ADDRESSES")
    metrics = metrics_atlas(state, cell=args.metric_cell)
    cv.imwrite(str(outdir / f"{stem}_faces.png"), faces)
    cv.imwrite(str(outdir / f"{stem}_metrics.png"), metrics)
    save_state_data(outdir / f"{stem}_data.npz", state, z0, plate, op_scale)
    return state, faces, metrics


def build_decoder(args):
    if args.mock:
        print("decoder: mock")
        return MockDecoder()
    path = find_model(args.model)
    dec = OnnxDecoder(path, cpu=args.cpu)
    print("decoder:", path, "backend:", dec.backend)
    return dec


def run(args):
    cv = cv2()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dec = build_decoder(args)
    rng = np.random.default_rng(args.seed)
    z0 = random_identity(rng, args.anchor_std)
    B = make_basis(dec, z0, args, verbose=True)

    if args.plate and Path(args.plate).is_file():
        plate = OperatorPlate.load(args.plate)
        if plate.dim != len(B):
            raise ValueError(f"plate dim {plate.dim} != local rig dim {len(B)}")
        print("loaded operator plate", args.plate)
    else:
        plate = OperatorPlate(dim=len(B), seed=args.seed + 4001)

    op_scale = operator_match_scale(plate, len(B), min(args.address_span, 1.25))
    print(f"operator visual scale: {op_scale:.3f}x")
    print(f"atlas: {args.grid}x{args.grid}, span +/-{args.address_span}, z={args.z:+.2f}, motion_gain={args.motion_gain}")

    before, before_img, before_metrics = render_and_save(
        dec, z0, B, plate, op_scale, args, outdir, stem="atlas_before" if args.mutate else "atlas"
    )

    print("before dM range:", f"{before['dm'].min():.3f} .. {before['dm'].max():.3f}")
    print("before cond range:", f"{before['cond'].min():.3f} .. {before['cond'].max():.3f}")
    print("before |c| range:", f"{before['cnorm'].min():.3f} .. {before['cnorm'].max():.3f}")

    show_items = [("atlas faces", before_img), ("atlas metrics", before_metrics)]

    if args.mutate:
        total_before = plate.material_sum
        addresses = before["addresses"].reshape(-1, 3)
        info = visible_mutation(plate, addresses, op_scale, delta=args.mutate_delta)
        if info is None:
            raise RuntimeError("no conservative mutation could be applied")

        after, after_img, after_metrics = render_and_save(
            dec, z0, B, plate, op_scale, args, outdir, stem="atlas_after"
        )
        diff_state, pixel_diff = difference_state(before, after, gain=args.diff_gain)
        diff_img = face_atlas(diff_state, tile=args.tile, title="ABS PIXEL CHANGE AFTER ONE g MUTATION")
        coeff_delta = np.linalg.norm(after["coeffs"] - before["coeffs"], axis=-1)
        coeff_panel = metric_panel(coeff_delta, "per-address coefficient displacement |dc|", cell=args.metric_cell)
        cv.imwrite(str(outdir / "atlas_difference.png"), diff_img)
        cv.imwrite(str(outdir / "atlas_mutation_metrics.png"), coeff_panel)

        summary = {
            "mutation": {
                "src": int(info["src"]),
                "dst": int(info["dst"]),
                "amount": float(info["amount"]),
                "mean_operator_response_change": float(info["score"]),
                "material_sum_before": float(total_before),
                "material_sum_after": float(plate.material_sum),
            },
            "atlas": {
                "grid": int(args.grid),
                "address_span": float(args.address_span),
                "z": float(args.z),
                "motion_gain": float(args.motion_gain),
                "mean_coeff_delta": float(np.mean(coeff_delta)),
                "max_coeff_delta": float(np.max(coeff_delta)),
                "mean_pixel_absdiff": float(np.mean(pixel_diff)),
                "max_tile_pixel_absdiff": float(np.max(np.mean(pixel_diff, axis=(-2, -1)))),
            },
        }
        with open(outdir / "atlas_mutation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"mutation e{info['src']}->e{info['dst']} dg={info['amount']:.3f} "
            f"mean|dc|={np.mean(coeff_delta):.3f} max|dc|={np.max(coeff_delta):.3f} "
            f"sum(g) {total_before:.6f}->{plate.material_sum:.6f}"
        )
        print("mean absolute pixel change:", f"{np.mean(pixel_diff):.3f}")
        show_items.extend([
            ("atlas after", after_img),
            ("atlas difference", diff_img),
            ("mutation |dc|", coeff_panel),
        ])

    print("saved to", outdir.resolve())
    for p in sorted(outdir.glob("*.png")):
        print("  ", p.name)

    if args.show:
        for name, im in show_items:
            # Fit giant atlases to screen while preserving the files at full resolution.
            scale = min(1.0, 1400 / im.shape[1], 900 / im.shape[0])
            shown = cv.resize(im, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else im
            cv.imshow(name, shown)
        print("press any key in an image window to close")
        cv.waitKey(0)
        cv.destroyAllWindows()

    return 0


def selftest():
    dec = MockDecoder()
    z0 = np.zeros(LATENT, np.float32)
    D = np.zeros((8, LATENT), np.float32)
    for i in range(8):
        D[i, i] = 1.0
    B, _, _ = probe(dec, z0, n=8, k=4, eps=0.5, seed=3, directions=D, verbose=False)
    plate = OperatorPlate(dim=4, seed=3)

    class A:
        address_span = 2.0
        grid = 4
        z = 0.0
        motion_gain = 2.0
        batch = 16
        tile = 48
        metric_cell = 32

    args = A()
    scale = operator_match_scale(plate, len(B), 1.25)
    before = sample_state(dec, z0, B, plate, scale, args)
    total = plate.material_sum
    info = visible_mutation(plate, before["addresses"].reshape(-1, 3), scale, delta=0.08)
    after = sample_state(dec, z0, B, plate, scale, args)

    checks = [
        ("atlas shape", before["images"].shape[:2] == (4, 4)),
        ("address changes operator", float(before["dm"].max()) > 0.05),
        ("atlas images finite", np.isfinite(before["images"]).all()),
        ("mutation found", info is not None),
        ("mutation conserves g", abs(plate.material_sum - total) < 1e-12),
        ("mutation moves atlas", float(np.mean(np.linalg.norm(after["coeffs"] - before["coeffs"], axis=-1))) > 1e-3),
    ]

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        cv = cv2()
        cv.imwrite(str(out / "faces.png"), face_atlas(before, tile=48))
        cv.imwrite(str(out / "metrics.png"), metrics_atlas(before, cell=32))
        checks.append(("PNG renderer", (out / "faces.png").stat().st_size > 100 and (out / "metrics.png").stat().st_size > 100))

    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    print("operator_atlas selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--anchor_std", type=float, default=0.60)
    p.add_argument("--probe_dirs", type=int, default=40)
    p.add_argument("--probe_eps", type=float, default=0.35)
    p.add_argument("--operator_dim", type=int, default=6)
    p.add_argument("--address_span", type=float, default=5.0)
    p.add_argument("--z", type=float, default=0.0)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--motion_gain", type=float, default=9.0)
    p.add_argument("--tile", type=int, default=112)
    p.add_argument("--metric_cell", type=int, default=58)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--plate", default=None)
    p.add_argument("--outdir", default="operator_atlas")
    p.add_argument("--mutate", action="store_true", help="render before/after one conservative visible mutation")
    p.add_argument("--mutate_delta", type=float, default=0.12)
    p.add_argument("--diff_gain", type=float, default=4.0)
    p.add_argument("--show", action="store_true", help="open the generated PNGs after saving")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        return selftest()
    if args.grid < 2:
        raise SystemExit("--grid must be >=2")
    if args.operator_dim < 2:
        raise SystemExit("--operator_dim must be >=2")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
