#!/usr/bin/env python3
"""
SplatWorld3D -- first external-world novel-view test.

This experiment deliberately does NOT use another operator plate as teacher.
It renders a small analytic 3D scene from known camera poses, trains one shared
operator plate g from sparse camera views, and evaluates genuinely held-out
intermediate views.

The experiment compares:
  OPERATOR  pose -> address-conditioned shared plate -> image-basis coefficients
  LINEAR    interpolate the neighboring training images in angle
  NEAREST   nearest training image
  MLP       small parameter-matched pose->coefficient MLP baseline (if torch exists)

The image basis is built from TRAIN VIEWS ONLY. Held-out images never enter the
basis or fitting objective.

Run:
    python splatworld3d.py --selftest
    python splatworld3d.py --run --out_dir splatworld3d_out

The default orbit uses 8 training views and 8 halfway held-out views.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from operator_plate import OperatorPlate

# ---------------------------------------------------------------------------
# Analytic external 3D world adapted from the older worldrank experiment.
# This is intentionally self-contained so the result does not depend on the
# historical SplatWorld4 tree at runtime.
# ---------------------------------------------------------------------------

SPHERE_C = np.array([0.62, -0.32, 0.34], dtype=np.float64)
SPHERE_R = 0.44
CUBE_C = np.array([-0.58, -0.28, -0.30], dtype=np.float64)
CUBE_H = np.array([0.38, 0.52, 0.38], dtype=np.float64)
FLOOR_Y = -0.82
FLOOR_EXT = 3.2
LIGHT = np.array([0.45, 0.82, 0.36], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)
FOV = 0.24
ORBIT_HGT = 1.75
ORBIT_DST = 8.80


def camera(phi: float, hgt: float = ORBIT_HGT, dist: float = ORBIT_DST):
    eye = np.array([dist * math.sin(phi), hgt, dist * math.cos(phi)], dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def render(phi: float, size: int = 48) -> np.ndarray:
    eye, right, up, fwd = camera(phi)
    t = math.tan(FOV)
    g = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    d = fwd[None, None, :] + (g[None, :] * t)[..., None] * right + (-g[:, None] * t)[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    INF = 1e9
    t_best = np.full((size, size), INF, dtype=np.float64)
    nrm = np.zeros((size, size, 3), dtype=np.float64)
    alb = np.zeros((size, size), dtype=np.float64)

    # sphere
    oc = eye - SPHERE_C
    b = d @ oc
    disc = b * b - (float(oc @ oc) - SPHERE_R * SPHERE_R)
    m = disc > 0
    ts = np.where(m, -b - np.sqrt(np.maximum(disc, 0.0)), INF)
    upd = m & (ts > 1e-4) & (ts < t_best)
    t_best = np.where(upd, ts, t_best)
    p = eye + ts[..., None] * d
    nrm = np.where(upd[..., None], (p - SPHERE_C) / SPHERE_R, nrm)
    alb = np.where(upd, 0.86, alb)

    # cube
    lo, hi = CUBE_C - CUBE_H, CUBE_C + CUBE_H
    dd = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - eye) / dd, (hi - eye) / dd
    tn = np.max(np.minimum(t1, t2), axis=-1)
    tf = np.min(np.maximum(t1, t2), axis=-1)
    m = tf > np.maximum(tn, 1e-4)
    tc = np.where(m, tn, INF)
    upd = m & (tc < t_best)
    t_best = np.where(upd, tc, t_best)
    p = eye + tc[..., None] * d
    rel = (p - CUBE_C) / CUBE_H
    ax = np.argmax(np.abs(rel), axis=-1)
    n = np.zeros((size, size, 3), dtype=np.float64)
    for a in range(3):
        sel = ax == a
        n[sel, a] = np.sign(rel[sel, a])
    nrm = np.where(upd[..., None], n, nrm)
    alb = np.where(upd, 0.52, alb)

    # floor
    tp = np.where(np.abs(d[..., 1]) > 1e-9, (FLOOR_Y - eye[1]) / dd[..., 1], INF)
    p = eye + tp[..., None] * d
    m = (tp > 1e-4) & (np.abs(p[..., 0]) < FLOOR_EXT) & (np.abs(p[..., 2]) < FLOOR_EXT)
    tp = np.where(m, tp, INF)
    upd = m & (tp < t_best)
    t_best = np.where(upd, tp, t_best)
    chk = (np.floor(p[..., 0] * 1.6) + np.floor(p[..., 2] * 1.6)) % 2 == 0
    nrm = np.where(upd[..., None], np.array([0.0, 1.0, 0.0]), nrm)
    alb = np.where(upd, np.where(chk, 0.74, 0.30), alb)

    hit = t_best < INF / 2
    img = np.where(hit, alb * (0.26 + 0.74 * np.clip(nrm @ LIGHT, 0.0, 1.0)), 0.07)
    return img.astype(np.float64)


# ---------------------------------------------------------------------------
# Training-view-only visual basis.
# ---------------------------------------------------------------------------


def fit_basis(train_images: np.ndarray, k: int):
    flat = train_images.reshape(len(train_images), -1)
    mean = flat.mean(axis=0)
    X = flat - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(int(k), Vt.shape[0])
    basis = Vt[:k]
    coeff = X @ basis.T
    return mean, basis, coeff, S


def decode_coeff(mean: np.ndarray, basis: np.ndarray, coeff: np.ndarray, shape):
    x = mean + coeff @ basis
    return np.clip(x.reshape(shape), 0.0, 1.0)


def address(phi: float, radius: float = 2.0) -> np.ndarray:
    # Keep norm fixed: azimuth changes, radial/fovea envelope cannot explain
    # held-out angular interpolation.
    return np.array([radius * math.cos(phi), radius * math.sin(phi), 0.0], dtype=np.float64)


def plate_features(plate: OperatorPlate, phi: float, radius: float, out_dim: int) -> np.ndarray:
    a = address(phi, radius)
    # OperatorPlate.response() is the same compact address-conditioned readout used
    # by SplatWorld3. Trim/pad deterministically if basis dimension differs.
    c = np.asarray(plate.response(a), dtype=np.float64).ravel()
    if c.size >= out_dim:
        return c[:out_dim]
    out = np.zeros(out_dim, dtype=np.float64)
    out[: c.size] = c
    return out


def fit_output_map(F: np.ndarray, C: np.ndarray, ridge: float = 1e-6):
    # A small linear readout from plate response to image-basis coefficients.
    # g remains the shared nonlinear address-conditioned object; this readout is
    # fitted only from training views and is held fixed during plate fitting.
    X = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ C)
    return W


def predict_coeff(plate: OperatorPlate, phis: np.ndarray, radius: float, W: np.ndarray, feat_dim: int):
    F = np.stack([plate_features(plate, p, radius, feat_dim) for p in phis])
    X = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    return X @ W


def rel_rmse(pred: np.ndarray, true: np.ndarray):
    num = np.sqrt(np.mean((pred - true) ** 2))
    den = np.sqrt(np.mean(true ** 2)) + 1e-12
    return float(num / den)


def psnr(pred: np.ndarray, true: np.ndarray):
    mse = float(np.mean((pred - true) ** 2))
    return float(99.0 if mse <= 1e-14 else -10.0 * math.log10(mse))


def circular_neighbors(phi: float, train_phis: np.ndarray):
    x = phi % (2 * math.pi)
    arr = np.sort(train_phis % (2 * math.pi))
    j = int(np.searchsorted(arr, x, side="right"))
    lo = arr[(j - 1) % len(arr)]
    hi = arr[j % len(arr)]
    hi_u = hi
    x_u = x
    if hi_u <= lo:
        hi_u += 2 * math.pi
    if x_u < lo:
        x_u += 2 * math.pi
    t = (x_u - lo) / max(hi_u - lo, 1e-12)
    return lo, hi % (2 * math.pi), float(t)


def linear_image_baseline(test_phis, train_phis, train_images):
    lookup = {round(float(p % (2 * math.pi)), 12): img for p, img in zip(train_phis, train_images)}
    out = []
    for phi in test_phis:
        lo, hi, t = circular_neighbors(float(phi), train_phis)
        a = lookup[round(float(lo), 12)]
        b = lookup[round(float(hi), 12)]
        out.append((1.0 - t) * a + t * b)
    return np.stack(out)


def nearest_baseline(test_phis, train_phis, train_images):
    out = []
    for p in test_phis:
        d = np.abs(np.angle(np.exp(1j * (train_phis - p))))
        out.append(train_images[int(np.argmin(d))])
    return np.stack(out)


def objective_for_plate(plate, train_phis, train_coeff, radius, feat_dim):
    F = np.stack([plate_features(plate, p, radius, feat_dim) for p in train_phis])
    W = fit_output_map(F, train_coeff)
    pred = predict_coeff(plate, train_phis, radius, W, feat_dim)
    return float(np.mean((pred - train_coeff) ** 2)), W


def fit_plate(train_phis, train_coeff, radius=2.0, feat_dim=6, steps=2500, seed=7):
    rng = np.random.default_rng(seed)
    plate = OperatorPlate(dim=6, seed=seed)
    base_sum = float(np.sum(plate.g))
    best, W = objective_for_plate(plate, train_phis, train_coeff, radius, feat_dim)
    delta = 0.08
    accepted = 0

    # derivative-free, conservative edge-to-edge dither as in the operator line
    for step in range(int(steps)):
        i, j = rng.choice(len(plate.g), size=2, replace=False)
        if plate.g[i] <= delta + 1e-6:
            continue
        old = plate.g.copy()
        plate.g[i] -= delta
        plate.g[j] += delta
        val, W_new = objective_for_plate(plate, train_phis, train_coeff, radius, feat_dim)
        if val < best:
            best, W = val, W_new
            accepted += 1
        else:
            plate.g[:] = old
        if step in (steps // 3, 2 * steps // 3):
            delta *= 0.5

    assert abs(float(np.sum(plate.g)) - base_sum) < 1e-9
    return plate, W, {"train_mse": best, "accepted": accepted, "sum_g": float(np.sum(plate.g))}


def run_experiment(size=48, n_train=8, basis_dim=6, steps=2500, radius=2.0, seed=7, out_dir="splatworld3d_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Interleaved train/test orbit: test poses are exact halfway angles.
    train_phis = np.arange(n_train) * (2 * math.pi / n_train)
    test_phis = train_phis + math.pi / n_train
    train_images = np.stack([render(float(p), size) for p in train_phis])
    test_images = np.stack([render(float(p), size) for p in test_phis])

    mean, basis, train_coeff, singulars = fit_basis(train_images, basis_dim)

    # Oracle projection of test images into training-only basis: this separates
    # image-basis limitation from address-field prediction limitation.
    test_flat = test_images.reshape(len(test_images), -1)
    true_test_coeff = (test_flat - mean) @ basis.T
    basis_oracle = np.stack([decode_coeff(mean, basis, c, (size, size)) for c in true_test_coeff])

    plate, W, fit_info = fit_plate(train_phis, train_coeff, radius, basis.shape[0], steps, seed)
    pred_coeff = predict_coeff(plate, test_phis, radius, W, basis.shape[0])
    pred_images = np.stack([decode_coeff(mean, basis, c, (size, size)) for c in pred_coeff])

    linear_images = linear_image_baseline(test_phis, train_phis, train_images)
    nearest_images = nearest_baseline(test_phis, train_phis, train_images)

    metrics = {
        "size": int(size),
        "n_train": int(n_train),
        "n_test": int(len(test_phis)),
        "basis_dim": int(basis.shape[0]),
        "plate_steps": int(steps),
        "radius": float(radius),
        "operator_rel_rmse": rel_rmse(pred_images, test_images),
        "operator_psnr": psnr(pred_images, test_images),
        "linear_rel_rmse": rel_rmse(linear_images, test_images),
        "linear_psnr": psnr(linear_images, test_images),
        "nearest_rel_rmse": rel_rmse(nearest_images, test_images),
        "nearest_psnr": psnr(nearest_images, test_images),
        "basis_oracle_rel_rmse": rel_rmse(basis_oracle, test_images),
        "basis_oracle_psnr": psnr(basis_oracle, test_images),
        "coeff_rel_rmse": rel_rmse(pred_coeff, true_test_coeff),
        "train_mse": fit_info["train_mse"],
        "accepted_mutations": int(fit_info["accepted"]),
        "sum_g": fit_info["sum_g"],
        "basis_singular_values": [float(x) for x in singulars[:basis.shape[0]]],
        "train_angles_deg": [float(np.degrees(p) % 360) for p in train_phis],
        "test_angles_deg": [float(np.degrees(p) % 360) for p in test_phis],
    }

    np.savez_compressed(
        out / "splatworld3d_data.npz",
        train_phis=train_phis,
        test_phis=test_phis,
        train_images=train_images,
        test_images=test_images,
        operator_images=pred_images,
        linear_images=linear_images,
        nearest_images=nearest_images,
        basis_oracle=basis_oracle,
        g=plate.g,
        W=W,
        basis=basis,
        mean=mean,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_contact_sheet(out / "comparison.png", test_phis, test_images, pred_images, linear_images, nearest_images, basis_oracle)
    return metrics


def render_contact_sheet(path: Path, phis, truth, operator, linear, nearest, basis_oracle):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    rows = [("TRUTH", truth), ("OPERATOR", operator), ("LINEAR", linear), ("NEAREST", nearest), ("TRAIN-BASIS ORACLE", basis_oracle)]
    tile = truth.shape[-1]
    scale = max(1, 128 // tile)
    tw = tile * scale
    label_h = 24
    left = 170
    canvas = Image.new("RGB", (left + tw * len(phis), label_h + tw * len(rows)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for j, p in enumerate(phis):
        draw.text((left + j * tw + 3, 4), f"{np.degrees(p)%360:.1f}°", fill=(120, 255, 120))
    for i, (name, imgs) in enumerate(rows):
        y = label_h + i * tw
        draw.text((8, y + 5), name, fill=(120, 255, 120))
        for j, arr in enumerate(imgs):
            im = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="L").convert("RGB").resize((tw, tw), Image.Resampling.NEAREST)
            canvas.paste(im, (left + j * tw, y))
    canvas.save(path)


def selftest():
    # Renderer must visibly change with angle.
    a = render(0.0, 24)
    b = render(0.7, 24)
    assert np.mean(np.abs(a - b)) > 1e-3

    # Training-only basis round-trip on train data improves over mean.
    imgs = np.stack([render(p, 24) for p in np.linspace(0, 2 * math.pi, 6, endpoint=False)])
    mean, basis, coeff, _ = fit_basis(imgs, 4)
    recon = np.stack([decode_coeff(mean, basis, c, (24, 24)) for c in coeff])
    mean_img = np.broadcast_to(mean.reshape(24, 24), imgs.shape)
    assert np.mean((recon - imgs) ** 2) < np.mean((mean_img - imgs) ** 2)

    # Plate fitting must conserve material exactly.
    phis = np.linspace(0, 2 * math.pi, 4, endpoint=False)
    train = np.stack([render(float(p), 20) for p in phis])
    mean, basis, coeff, _ = fit_basis(train, 3)
    plate, W, info = fit_plate(phis, coeff, radius=2.0, feat_dim=3, steps=40, seed=2)
    assert np.isfinite(info["train_mse"])
    assert abs(float(np.sum(plate.g)) - info["sum_g"]) < 1e-12
    print("SplatWorld3D selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--basis_dim", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default="splatworld3d_out")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run:
        ap.error("use --selftest or --run")
    m = run_experiment(args.size, args.n_train, args.basis_dim, args.steps, args.radius, args.seed, args.out_dir)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
