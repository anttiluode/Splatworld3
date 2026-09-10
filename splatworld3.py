#!/usr/bin/env python3
"""SplatWorld3 — one shared operator plate, many local face transformations.

SplatWorld2 discovered that each committed identity exposes a different local
transport-like rig. SplatWorld3 inserts one extra object between the user's
address and that rig:

    persistent material g
          -> address-conditioned dense matrix M_g(a)
          -> coefficients in the local rig B(z0)
          -> decoder F(z0 + c B)

The important UI fact is that M_g(a) is *not stored per address*. One small
material vector g is stored; moving the address regenerates a different matrix.
The four corner previews are simultaneous samples of that one matrix family.

This is an explorer, not a claim that the CelebA decoder contains a 3-D world.
The purpose is to make the operator-family idea visible on a model we can
already inspect by eye.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import numpy as np

from operator_plate import OperatorPlate, selftest as plate_selftest
from splatworld2 import (
    OnnxDecoder,
    MockDecoder,
    IdentityLock,
    find_model,
    probe,
    random_identity,
    rgb,
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
    """Simple non-operator A/B control: coordinates map directly to rig axes."""
    a = np.asarray(address, np.float32)
    c = np.zeros(dim, np.float32)
    if dim > 0:
        c[0] = 0.75 * a[0]
    if dim > 1:
        c[1] = 0.75 * a[1]
    if dim > 2:
        c[2] = 0.55 * a[2]
    return c


def local_latent(z0, B, coeff, gain):
    c = np.asarray(coeff, np.float32) * float(gain)
    return (np.asarray(z0, np.float32) + c @ np.asarray(B, np.float32)).astype(np.float32)


def make_identity(dec, args, z0, probe_seed, verbose=True):
    anchor = rgb(dec(z0[None])[0])
    if verbose:
        print(f"identity |z|={np.linalg.norm(z0):.3f} — probing {args.operator_dim} local modes")
    B, chosen, metrics = probe(
        dec,
        z0,
        n=args.probe_dirs,
        k=args.operator_dim,
        eps=args.probe_eps,
        seed=probe_seed,
        verbose=verbose,
    )
    lock = IdentityLock(
        anchor,
        size=args.size,
        anchor_image=args.anchor_image,
        sigma=args.detail_sigma,
        gain=args.detail_gain,
        conf_sigma=args.confidence_sigma,
    )
    return anchor, B, chosen, metrics, lock


def _text(img, text, xy, scale=0.48, color=(80, 255, 80), thick=1):
    cv = cv2()
    cv.putText(img, text, xy, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)


def _matrix_heatmap(M, side=210):
    cv = cv2()
    x = np.real(np.asarray(M))
    s = np.max(np.abs(x)) + EPS
    q = np.clip(0.5 + 0.5 * x / s, 0.0, 1.0)
    q = (255.0 * q).astype(np.uint8)
    heat = cv.applyColorMap(q, cv.COLORMAP_TURBO)
    return cv.resize(heat, (side, side), interpolation=cv.INTER_NEAREST)


def _preview_batch(dec, z0, B, plate, gain, side=104):
    cv = cv2()
    addresses = [
        np.array([-0.9, -0.9, 0.0], np.float32),
        np.array([+0.9, -0.9, 0.0], np.float32),
        np.array([-0.9, +0.9, 0.0], np.float32),
        np.array([+0.9, +0.9, 0.0], np.float32),
    ]
    zs = []
    for a in addresses:
        coeff = plate.response(a)
        zs.append(local_latent(z0, B, coeff, gain))
    outs = dec(np.asarray(zs, np.float32))
    thumbs = []
    for out in outs:
        im = (255.0 * rgb(out)).astype(np.uint8)
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        thumbs.append(cv.resize(im, (side, side), interpolation=cv.INTER_CUBIC))
    return addresses, thumbs


def draw_ui(face_rgb, plate, address, coeff, args, mode, locked, fps, previews, material_note=""):
    cv = cv2()
    size = args.size
    panel = 250
    canvas = np.full((size, size + panel, 3), 28, np.uint8)
    face = np.clip(face_rgb, 0, 1)
    face = (255.0 * face).astype(np.uint8)
    face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
    face = cv.resize(face, (size, size), interpolation=cv.INTER_CUBIC)
    canvas[:, :size] = face

    cx = int(size * (0.5 + 0.34 * float(address[0]) / max(args.address_span, EPS)))
    cy = int(size * (0.5 - 0.34 * float(address[1]) / max(args.address_span, EPS)))
    cv.drawMarker(canvas, (cx, cy), (90, 255, 90), cv.MARKER_CROSS, 18, 1, cv.LINE_AA)

    status = f"{mode.upper()}  a=({address[0]:+.2f},{address[1]:+.2f},{address[2]:+.2f})  lock={int(locked)}  fps={fps:4.1f}"
    cv.rectangle(canvas, (0, size - 34), (size, size), (0, 0, 0), -1)
    _text(canvas, status, (12, size - 11), 0.50)

    x0 = size + 14
    _text(canvas, "ONE g -> MANY M_g(a)", (x0, 25), 0.50)
    _text(canvas, f"dim {plate.dim}  edges {len(plate.g)}", (x0, 47), 0.42, (210, 210, 210))
    stats = plate.stats(address)
    _text(canvas, f"dM(origin) {stats.matrix_delta_from_origin:.3f}", (x0, 66), 0.40, (210, 210, 210))
    _text(canvas, f"cond(A) {stats.condition:.2f}", (x0, 84), 0.40, (210, 210, 210))

    if previews:
        _, thumbs = previews
        y0 = 100
        side = thumbs[0].shape[0]
        gap = 7
        for i, th in enumerate(thumbs):
            xx = x0 + (i % 2) * (side + gap)
            yy = y0 + (i // 2) * (side + gap)
            canvas[yy:yy + side, xx:xx + side] = th
        _text(canvas, "same plate / four addresses", (x0, y0 + 2 * (side + gap) + 12), 0.36, (190, 190, 190))

    heat_side = min(160, panel - 28)
    heat = _matrix_heatmap(plate.matrix(address), heat_side)
    hy = 365
    if hy + heat_side < size - 80:
        canvas[hy:hy + heat_side, x0:x0 + heat_side] = heat
        _text(canvas, "Re M_g(a)", (x0, hy - 8), 0.40, (210, 210, 210))

    by = size - 64
    bw = max(1, int((panel - 30) / max(len(plate.g), 1)))
    gmax = max(float(np.max(plate.g)), EPS)
    for i, val in enumerate(plate.g):
        h = int(34 * float(val) / gmax)
        xx = x0 + i * bw
        if xx >= size + panel - 5:
            break
        cv.rectangle(canvas, (xx, by + 34 - h), (xx + max(1, bw - 1), by + 34), (70, 180, 255), -1)
    if material_note:
        _text(canvas, material_note, (x0, size - 9), 0.36, (90, 255, 255))

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
    c0 = plate.response((0.0, 0.0, 0.0))
    c1 = plate.response((0.8, -0.6, 0.2))
    z1 = local_latent(z0, B, c1, 1.2)
    out = dec(z1[None])
    checks = [
        ("origin response centered", np.linalg.norm(c0) < 1e-6),
        ("address moves local latent", np.linalg.norm(z1 - z0) > 1e-5),
        ("mock render finite", np.isfinite(out).all()),
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
    history = [z0.copy()]
    hist_i = 0
    probe_seed = args.seed
    anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)

    if args.plate and Path(args.plate).is_file():
        plate = OperatorPlate.load(args.plate)
        if plate.dim != len(B):
            raise ValueError(f"saved plate dim {plate.dim} != local rig dim {len(B)}")
        print("loaded operator plate", args.plate)
    else:
        plate = OperatorPlate(dim=len(B), seed=args.seed + 4001)

    address = AddressControl(args.address_span)
    operator_mode = True
    locked = not args.raw
    auto = bool(args.auto)
    previews = None
    preview_dirty = True
    material_note = ""

    win = "SplatWorld3  O=operator/direct  drag=address  [ ]=depth  M=mutate U=undo G=reset  ENTER=commit N/P=faces L=lock A=auto K=save S=shot Q=quit"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    mouse = {"b": 0, "x": 0, "y": 0}

    def cb(ev, x, y, flags, _):
        if ev in (cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN):
            mouse.update(b=1 if ev == cv.EVENT_LBUTTONDOWN else 2, x=x, y=y)
        elif ev in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
            mouse["b"] = 0
        elif ev == cv.EVENT_MOUSEMOVE and mouse["b"]:
            dx, dy = x - mouse["x"], y - mouse["y"]
            address.drag(dx, dy, fine=(mouse["b"] == 2))
            mouse.update(x=x, y=y)
        elif ev == cv.EVENT_MOUSEWHEEL:
            try:
                d = cv.getMouseWheelDelta(flags)
            except Exception:
                d = 1 if flags > 0 else -1
            address.depth(0.08 * np.sign(d))

    cv.setMouseCallback(win, cb)
    t_last = time.time()
    fps = 0.0
    frames = 0

    while True:
        now = time.time()
        if auto:
            t = now * args.auto_speed
            address.ta[0] = args.address_span * 0.78 * math.sin(t)
            address.ta[1] = args.address_span * 0.78 * math.sin(0.73 * t + 0.9)
            address.ta[2] = args.address_span * 0.42 * math.sin(0.41 * t - 0.4)
        address.tick()

        if operator_mode:
            coeff = plate.response(address.a)
            mode = "operator"
        else:
            coeff = direct_coeff(address.a, len(B))
            mode = "direct"

        z = local_latent(z0, B, coeff, args.operator_gain)
        current = rgb(dec(z[None])[0])
        shown = lock(current) if locked else cv.resize(current, (args.size, args.size), interpolation=cv.INTER_CUBIC)

        if args.previews and (preview_dirty or previews is None or frames % args.preview_every == 0):
            previews = _preview_batch(dec, z0, B, plate, args.operator_gain, side=args.preview_size)
            preview_dirty = False

        frames += 1
        dt = now - t_last
        if dt >= 0.5:
            fps = frames / dt
            frames = 0
            t_last = now

        ui = draw_ui(shown, plate, address.a, coeff, args, mode, locked, fps, previews, material_note)
        cv.imshow(win, ui)
        key = cv.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('o'):
            operator_mode = not operator_mode
        elif key == ord('l'):
            locked = not locked
        elif key == ord('a'):
            auto = not auto
        elif key == ord('r'):
            address.reset()
        elif key == ord('['):
            address.depth(-0.12)
        elif key == ord(']'):
            address.depth(+0.12)
        elif key == ord('m'):
            before = plate.material_sum
            if plate.mutate(rng, delta=args.mutate_delta):
                material_note = f"mutated g; sum={plate.material_sum:.4f} (was {before:.4f})"
                preview_dirty = True
        elif key == ord('u'):
            if plate.undo():
                material_note = "undo material mutation"
                preview_dirty = True
        elif key == ord('g'):
            plate.reset()
            material_note = "reset material g"
            preview_dirty = True
        elif key == ord('k'):
            p = plate.save(args.save_plate)
            material_note = f"saved {p.name}"
        elif key == ord('s'):
            out = Path(args.screenshot)
            cv.imwrite(str(out), ui)
            material_note = f"saved {out.name}"
        elif key in (13, 10):
            z0 = z.copy()
            history = history[:hist_i + 1] + [z0.copy()]
            hist_i = len(history) - 1
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
            if len(B) != plate.dim:
                plate = OperatorPlate(dim=len(B), seed=args.seed + 4001)
            address.reset()
            preview_dirty = True
            material_note = "committed identity; same plate, new local rig"
        elif key == ord('n'):
            z0 = random_identity(rng, args.anchor_std)
            history = history[:hist_i + 1] + [z0.copy()]
            hist_i = len(history) - 1
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
            address.reset()
            preview_dirty = True
            material_note = "new identity; plate persisted"
        elif key == ord('p') and hist_i > 0:
            hist_i -= 1
            z0 = history[hist_i].copy()
            probe_seed += 1
            anchor, B, chosen, metrics, lock = make_identity(dec, args, z0, probe_seed)
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
    p.add_argument("--operator_gain", type=float, default=2.0)
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
    p.add_argument("--preview_every", type=int, default=24)
    p.add_argument("--preview_size", type=int, default=104)
    p.add_argument("--mutate_delta", type=float, default=0.025)
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
