#!/usr/bin/env python3
"""SplatWorld3 — one shared operator plate, many visible local face transformations.

One persistent material vector g generates an address-conditioned dense operator

    M_g(a) = [K(g) + D(a) + i gamma I]^-1

The operator response is mapped into the local transport-like rig B(z0) measured
from the existing SplatWorld decoder. The operator is not a 3-D face model; x/y/z
are abstract query coordinates.

v1.1 fixes an important visualization failure in the first operator build:
the mathematical operator changed, but its response occupied only a tiny fraction
of the coefficient range used by SplatWorld2. The app now calibrates operator and
direct controls to the same local-motion budget, previews a neighborhood around
the current address, and makes M choose a single conservative material transfer
that maximizes visible operator-family displacement in that neighborhood.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import numpy as np

from operator_plate import OperatorPlate, selftest as plate_selftest
from splatworld2 import (
    OnnxDecoder, MockDecoder, IdentityLock, find_model, probe,
    random_identity, rgb,
)

EPS = 1e-9
LATENT = 128


def cv2():
    import cv2 as cv
    return cv


class AddressControl:
    def __init__(self, span=1.25):
        self.span = float(span)
        self.a = np.zeros(3, np.float32)
        self.ta = self.a.copy()

    def drag(self, dx, dy, fine=False):
        gain = 0.006 * (0.2 if fine else 1.0)
        self.ta[0] += gain * dx
        self.ta[1] -= gain * dy
        self.ta[:2] = np.clip(self.ta[:2], -self.span, self.span)

    def depth(self, delta):
        self.ta[2] = float(np.clip(self.ta[2] + delta, -self.span, self.span))

    def tick(self):
        self.a += 0.24 * (self.ta - self.a)

    def reset(self):
        self.ta[:] = 0.0


def direct_coeff(address, dim):
    """Non-operator A/B control: coordinates map directly to local rig axes."""
    a = np.asarray(address, np.float32)
    c = np.zeros(dim, np.float32)
    if dim > 0:
        c[0] = 0.75 * a[0]
    if dim > 1:
        c[1] = 0.75 * a[1]
    if dim > 2:
        c[2] = 0.55 * a[2]
    return c


def local_latent(z0, B, coeff, gain=1.0):
    c = np.asarray(coeff, np.float32) * float(gain)
    return (np.asarray(z0, np.float32) + c @ np.asarray(B, np.float32)).astype(np.float32)


def make_identity(dec, args, z0, probe_seed, verbose=True):
    anchor = rgb(dec(z0[None])[0])
    if verbose:
        print(f"identity |z|={np.linalg.norm(z0):.3f} — probing {args.operator_dim} local modes")
    B, chosen, metrics = probe(
        dec, z0, n=args.probe_dirs, k=args.operator_dim,
        eps=args.probe_eps, seed=probe_seed, verbose=verbose,
    )
    lock = IdentityLock(
        anchor, size=args.size, anchor_image=args.anchor_image,
        sigma=args.detail_sigma, gain=args.detail_gain,
        conf_sigma=args.confidence_sigma,
    )
    return anchor, B, chosen, metrics, lock


def neighborhood(center, span, offset=0.62):
    """Four nearby addresses centered on the current query."""
    c = np.asarray(center, np.float32)
    pts = []
    for dx, dy in ((-offset, -offset), (+offset, -offset),
                   (-offset, +offset), (+offset, +offset)):
        q = c.copy()
        q[0] = np.clip(q[0] + dx, -span, span)
        q[1] = np.clip(q[1] + dy, -span, span)
        pts.append(q.astype(np.float32))
    return pts


def operator_match_scale(plate, dim, span):
    """Match the operator's coefficient norm to the direct-control budget.

    This is a fixed scale for one plate state at startup/identity change. It is
    not recomputed after material mutation, so mutation effects are not hidden.
    """
    probes = [
        np.array([-0.85, -0.85, 0.0], np.float32),
        np.array([+0.85, -0.85, 0.0], np.float32),
        np.array([-0.85, +0.85, 0.0], np.float32),
        np.array([+0.85, +0.85, 0.0], np.float32),
        np.array([0.0, 0.0, +0.65], np.float32),
        np.array([0.0, 0.0, -0.65], np.float32),
    ]
    probes = [np.clip(a, -span, span) for a in probes]
    op = np.median([np.linalg.norm(plate.response(a)) for a in probes])
    direct = np.median([np.linalg.norm(direct_coeff(a, dim)) for a in probes])
    return float(np.clip(direct / max(op, 1e-6), 1.0, 24.0))


def operator_coeff(plate, address, scale):
    return np.asarray(plate.response(address), np.float32) * float(scale)


def visible_mutation(plate, addresses, scale, delta=0.12):
    """Choose ONE conservative edge->edge transfer with large family effect.

    This is not learning and does not inspect pixels. It searches only the
    operator response at the displayed addresses, then applies the single local
    material transfer that moves that response family most.
    """
    addresses = [np.asarray(a, np.float32) for a in addresses]
    base_g = plate.g.copy()
    base = [operator_coeff(plate, a, scale) for a in addresses]
    best = None

    for src in range(len(base_g)):
        amount = min(float(delta), max(0.0, float(base_g[src]) - 0.035))
        if amount <= 0:
            continue
        for dst in range(len(base_g)):
            if src == dst:
                continue
            candidate = base_g.copy()
            candidate[src] -= amount
            candidate[dst] += amount
            plate.g = candidate
            changes = [
                np.linalg.norm(operator_coeff(plate, a, scale) - b)
                for a, b in zip(addresses, base)
            ]
            score = float(np.mean(changes))
            if best is None or score > best[0]:
                best = (score, src, dst, amount, candidate.copy(), changes)

    plate.g = base_g
    if best is None:
        return None

    score, src, dst, amount, candidate, changes = best
    plate._undo.append(base_g.copy())
    plate.g = candidate
    return {
        "score": score, "src": src, "dst": dst, "amount": amount,
        "changes": np.asarray(changes, np.float64),
    }


def _text(img, text, xy, scale=0.48, color=(80, 255, 80), thick=1):
    cv = cv2()
    cv.putText(img, text, xy, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)


def _delta_heatmap(plate, address, side=160):
    """Show address-induced operator deformation, not raw-M frame rescaling."""
    cv = cv2()
    M = plate.matrix(address)
    M0 = plate.matrix((0.0, 0.0, 0.0))
    x = np.real(M - M0)
    s = np.percentile(np.abs(x), 95) + EPS
    q = np.clip(0.5 + 0.5 * x / s, 0.0, 1.0)
    q = (255.0 * q).astype(np.uint8)
    heat = cv.applyColorMap(q, cv.COLORMAP_TURBO)
    return cv.resize(heat, (side, side), interpolation=cv.INTER_NEAREST)


def _preview_batch(dec, z0, B, plate, scale, gain, center, span, side=104):
    cv = cv2()
    addresses = neighborhood(center, span)
    zs, coeffs = [], []
    for a in addresses:
        coeff = operator_coeff(plate, a, scale)
        coeffs.append(coeff)
        zs.append(local_latent(z0, B, coeff, gain))
    outs = dec(np.asarray(zs, np.float32))
    thumbs = []
    for out in outs:
        im = (255.0 * rgb(out)).astype(np.uint8)
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        thumbs.append(cv.resize(im, (side, side), interpolation=cv.INTER_CUBIC))
    return addresses, thumbs, coeffs


def draw_ui(face_rgb, plate, address, coeff, args, mode, locked, fps,
            previews, op_scale, material_note=""):
    cv = cv2()
    size, panel = args.size, 300
    canvas = np.full((size, size + panel, 3), 28, np.uint8)
    face = cv.cvtColor((255.0 * np.clip(face_rgb, 0, 1)).astype(np.uint8), cv.COLOR_RGB2BGR)
    canvas[:, :size] = cv.resize(face, (size, size), interpolation=cv.INTER_CUBIC)

    cx = int(size * (0.5 + 0.34 * float(address[0]) / max(args.address_span, EPS)))
    cy = int(size * (0.5 - 0.34 * float(address[1]) / max(args.address_span, EPS)))
    cv.drawMarker(canvas, (cx, cy), (90, 255, 90), cv.MARKER_CROSS, 18, 1, cv.LINE_AA)

    status = (f"{mode.upper()} a=({address[0]:+.2f},{address[1]:+.2f},{address[2]:+.2f}) "
              f"|c|={np.linalg.norm(coeff):.2f} lock={int(locked)} fps={fps:4.1f}")
    cv.rectangle(canvas, (0, size - 34), (size, size), (0, 0, 0), -1)
    _text(canvas, status, (12, size - 11), 0.48)

    x0 = size + 14
    _text(canvas, "ONE g -> MANY M_g(a)", (x0, 25), 0.50)
    _text(canvas, f"dim {plate.dim} edges {len(plate.g)}  vis-scale {op_scale:.2f}x",
          (x0, 47), 0.36, (210, 210, 210))
    stats = plate.stats(address)
    _text(canvas, f"dM(origin) {stats.matrix_delta_from_origin:.3f}   cond(A) {stats.condition:.2f}",
          (x0, 66), 0.36, (210, 210, 210))

    if previews:
        addrs, thumbs, coeffs = previews
        y0, side, gap = 90, thumbs[0].shape[0], 7
        for i, th in enumerate(thumbs):
            xx = x0 + (i % 2) * (side + gap)
            yy = y0 + (i // 2) * (side + 30)
            canvas[yy:yy + side, xx:xx + side] = th
            _text(canvas, f"|c| {np.linalg.norm(coeffs[i]):.2f}",
                  (xx, yy + side + 15), 0.31, (190, 190, 190))
        _text(canvas, "4 nearby queries / same plate", (x0, y0 + 2 * (side + 30) + 5),
              0.34, (190, 190, 190))

    hy = 390
    heat_side = min(175, panel - 28)
    if hy + heat_side < size - 70:
        heat = _delta_heatmap(plate, address, heat_side)
        canvas[hy:hy + heat_side, x0:x0 + heat_side] = heat
        _text(canvas, "Re [M_g(a)-M_g(0)]", (x0, hy - 8), 0.36, (210, 210, 210))

    by = size - 58
    bw = max(1, int((panel - 30) / max(len(plate.g), 1)))
    gmax = max(float(np.max(plate.g)), EPS)
    for i, val in enumerate(plate.g):
        h = int(30 * float(val) / gmax)
        xx = x0 + i * bw
        if xx >= size + panel - 5:
            break
        cv.rectangle(canvas, (xx, by + 30 - h), (xx + max(1, bw - 1), by + 30),
                     (70, 180, 255), -1)
    if material_note:
        _text(canvas, material_note, (x0, size - 7), 0.31, (90, 255, 255))
    return canvas


def run_selftest():
    ok = plate_selftest(True)
    dec = MockDecoder()
    z0 = np.zeros(LATENT, np.float32)
    D = np.zeros((8, LATENT), np.float32)
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    for i in range(2, 8):
        D[i, i] = 1.0
    B, _, _ = probe(dec, z0, n=8, k=4, eps=0.5, seed=3, directions=D, verbose=False)
    plate = OperatorPlate(dim=4, seed=3)
    scale = operator_match_scale(plate, len(B), 1.25)
    c0 = operator_coeff(plate, (0, 0, 0), scale)
    c1 = operator_coeff(plate, (0.8, -0.6, 0.2), scale)
    z1 = local_latent(z0, B, c1)
    out = dec(z1[None])

    pts = neighborhood((0, 0, 0), 1.25)
    before = [operator_coeff(plate, a, scale) for a in pts]
    total = plate.material_sum
    info = visible_mutation(plate, pts, scale, delta=0.10)
    after = [operator_coeff(plate, a, scale) for a in pts]
    family_move = np.mean([np.linalg.norm(a-b) for a, b in zip(after, before)])

    checks = [
        ("origin response centered", np.linalg.norm(c0) < 1e-6),
        ("matched operator scale finite", np.isfinite(scale) and scale >= 1.0),
        ("address visibly moves local latent", np.linalg.norm(z1-z0) > 0.02),
        ("mock render finite", np.isfinite(out).all()),
        ("visible mutation found", info is not None),
        ("visible mutation conserves material", abs(plate.material_sum-total) < 1e-12),
        ("visible mutation moves family", family_move > 1e-3),
    ]
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    print("SplatWorld3 selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def live(dec, args):
    cv = cv2()
    rng = np.random.default_rng(args.seed)
    z0 = random_identity(rng, args.anchor_std)
    history, hist_i, probe_seed = [z0.copy()], 0, args.seed
    anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)

    if args.plate and Path(args.plate).is_file():
        plate = OperatorPlate.load(args.plate)
        if plate.dim != len(B):
            raise ValueError(f"saved plate dim {plate.dim} != local rig dim {len(B)}")
        print("loaded operator plate", args.plate)
    else:
        plate = OperatorPlate(dim=len(B), seed=args.seed + 4001)

    op_scale = operator_match_scale(plate, len(B), args.address_span)
    print(f"operator visual scale: {op_scale:.3f}x (matched to direct-control coefficient budget)")

    address = AddressControl(args.address_span)
    operator_mode, locked, auto = True, not args.raw, bool(args.auto)
    previews, preview_dirty, material_note = None, True, ""
    frame_index, fps_frames = 0, 0
    t_last, fps = time.time(), 0.0

    win = ("SplatWorld3  O=operator/direct drag=address [ ]=depth "
           "M=visible-mutate U=undo G=reset ENTER=commit N/P=faces "
           "L=lock A=auto K=save S=shot Q=quit")
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    mouse = {"b": 0, "x": 0, "y": 0}

    def cb(ev, x, y, flags, _):
        if ev in (cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN):
            mouse.update(b=1 if ev == cv.EVENT_LBUTTONDOWN else 2, x=x, y=y)
        elif ev in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
            mouse["b"] = 0
        elif ev == cv.EVENT_MOUSEMOVE and mouse["b"]:
            dx, dy = x-mouse["x"], y-mouse["y"]
            address.drag(dx, dy, fine=(mouse["b"] == 2))
            mouse.update(x=x, y=y)
        elif ev == cv.EVENT_MOUSEWHEEL:
            try:
                d = cv.getMouseWheelDelta(flags)
            except Exception:
                d = 1 if flags > 0 else -1
            address.depth(0.08*np.sign(d))

    cv.setMouseCallback(win, cb)

    while True:
        now = time.time()
        if auto:
            t = now * args.auto_speed
            address.ta[0] = args.address_span*0.78*math.sin(t)
            address.ta[1] = args.address_span*0.78*math.sin(0.73*t + 0.9)
            address.ta[2] = args.address_span*0.42*math.sin(0.41*t - 0.4)
        address.tick()

        if operator_mode:
            coeff = operator_coeff(plate, address.a, op_scale)
            mode = "operator"
        else:
            coeff = direct_coeff(address.a, len(B))
            mode = "direct"

        z = local_latent(z0, B, coeff, args.motion_gain)
        current = rgb(dec(z[None])[0])
        shown = lock(current) if locked else cv.resize(
            current, (args.size, args.size), interpolation=cv.INTER_CUBIC
        )

        if args.previews and (preview_dirty or previews is None or frame_index % args.preview_every == 0):
            previews = _preview_batch(
                dec, z0, B, plate, op_scale, args.motion_gain,
                address.a, args.address_span, side=args.preview_size,
            )
            preview_dirty = False

        frame_index += 1
        fps_frames += 1
        dt = now - t_last
        if dt >= 0.5:
            fps = fps_frames/dt
            fps_frames = 0
            t_last = now

        ui = draw_ui(
            shown, plate, address.a, coeff, args, mode, locked,
            fps, previews, op_scale, material_note,
        )
        cv.imshow(win, ui)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("o"):
            operator_mode = not operator_mode
        elif key == ord("l"):
            locked = not locked
        elif key == ord("a"):
            auto = not auto
        elif key == ord("r"):
            address.reset()
        elif key == ord("["):
            address.depth(-0.12)
        elif key == ord("]"):
            address.depth(+0.12)
        elif key == ord("m"):
            pts = neighborhood(address.a, args.address_span)
            info = visible_mutation(plate, pts, op_scale, delta=args.mutate_delta)
            if info:
                d = info["changes"]
                material_note = (
                    f"M e{info['src']}->e{info['dst']} dg={info['amount']:.3f} "
                    f"mean|dc|={d.mean():.3f} sum={plate.material_sum:.4f}"
                )
                print(material_note, "per-view", np.array2string(d, precision=3))
                preview_dirty = True
        elif key == ord("u"):
            if plate.undo():
                material_note = "undo: restored previous g"
                preview_dirty = True
        elif key == ord("g"):
            plate.reset()
            material_note = "reset: startup g restored"
            preview_dirty = True
        elif key == ord("k"):
            p = plate.save(args.save_plate)
            material_note = f"saved {p.name}"
        elif key == ord("s"):
            out = Path(args.screenshot)
            cv.imwrite(str(out), ui)
            material_note = f"saved {out.name}"
        elif key in (13, 10):
            z0 = z.copy()
            history = history[:hist_i+1] + [z0.copy()]
            hist_i = len(history)-1
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
            if len(B) != plate.dim:
                plate = OperatorPlate(dim=len(B), seed=args.seed+4001)
            op_scale = operator_match_scale(plate, len(B), args.address_span)
            address.reset()
            preview_dirty = True
            material_note = "committed identity; same plate, new local rig"
        elif key == ord("n"):
            z0 = random_identity(rng, args.anchor_std)
            history = history[:hist_i+1] + [z0.copy()]
            hist_i = len(history)-1
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
            op_scale = operator_match_scale(plate, len(B), args.address_span)
            address.reset()
            preview_dirty = True
            material_note = "new identity; plate persisted"
        elif key == ord("p") and hist_i > 0:
            hist_i -= 1
            z0 = history[hist_i].copy()
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
            op_scale = operator_match_scale(plate, len(B), args.address_span)
            address.reset()
            preview_dirty = True
            material_note = "previous identity; plate persisted"

    cv.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--mock", action="store_true", help="use the synthetic decoder")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--anchor_std", type=float, default=0.60)
    p.add_argument("--probe_dirs", type=int, default=40)
    p.add_argument("--probe_eps", type=float, default=0.35)
    p.add_argument("--operator_dim", type=int, default=6)
    p.add_argument("--motion_gain", type=float, default=1.0,
                   help="shared final coefficient gain after operator/direct budget matching")
    p.add_argument("--address_span", type=float, default=1.25)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--auto_speed", type=float, default=0.45)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--anchor_image", default=None)
    p.add_argument("--detail_sigma", type=float, default=1.2)
    p.add_argument("--detail_gain", type=float, default=1.0)
    p.add_argument("--confidence_sigma", type=float, default=0.10)
    p.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preview_every", type=int, default=8)
    p.add_argument("--preview_size", type=int, default=104)
    p.add_argument("--mutate_delta", type=float, default=0.12,
                   help="material moved by M; total g remains conserved")
    p.add_argument("--plate", default=None, help="load a saved operator plate .npz")
    p.add_argument("--save_plate", default="operator_plate.npz")
    p.add_argument("--screenshot", default="splatworld3.png")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        return run_selftest()
    if args.operator_dim < 2:
        raise SystemExit("--operator_dim must be >=2")
    if args.mock:
        dec = MockDecoder()
        print("decoder: mock")
    else:
        path = find_model(args.model)
        dec = OnnxDecoder(path, cpu=args.cpu)
        print("decoder:", path, "backend:", dec.backend)
    live(dec, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
