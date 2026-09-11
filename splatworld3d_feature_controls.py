#!/usr/bin/env python3
"""Matched feature controls for SplatWorld3D.

The regularized, UNTRAINED operator feature map interpolated held-out external
3D views better than plain pixel interpolation. This script asks whether that
is special to the resolvent/operator geometry or just what any five smooth
nonlinear features of camera angle would do.

All models:
  * see the same 8 training images;
  * use the same training-only PCA image basis;
  * have a tiny feature map from camera angle;
  * choose ridge from {0.01, 0.1, 1.0} by TRAINING leave-one-out only;
  * are evaluated once on the untouched halfway views.

Controls:
  ADDRESS2    [cos(phi), sin(phi)]
  HARMONIC5   [cos phi, sin phi, cos 2phi, sin 2phi, cos 3phi]
  RANDOM5     tanh(B [cos phi,sin phi] + b), 5 features, 32 fixed seeds
  OPERATOR5   first 5 bounded responses of an untrained OperatorPlate,
              32 fixed seeds

The seed distributions are diagnostics. No seed is selected using test error.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from operator_plate import OperatorPlate
from splatworld3d import render, fit_basis, decode_coeff, plate_features, rel_rmse, psnr, linear_image_baseline
from splatworld3d_loo import RIDGES, loo_score_from_features, fit_map, predict_map


def decode_many(mean, basis, coeff, size):
    return np.stack([decode_coeff(mean, basis, c, (size, size)) for c in coeff])


def address2(phis):
    phis = np.asarray(phis, dtype=np.float64)
    return np.stack([np.cos(phis), np.sin(phis)], axis=1)


def harmonic5(phis):
    phis = np.asarray(phis, dtype=np.float64)
    return np.stack([
        np.cos(phis), np.sin(phis),
        np.cos(2 * phis), np.sin(2 * phis),
        np.cos(3 * phis),
    ], axis=1)


def random5(phis, seed):
    u = address2(phis)
    rng = np.random.default_rng(int(seed))
    B = rng.normal(0.0, 1.0, size=(5, 2))
    b = rng.uniform(-math.pi, math.pi, size=5)
    return np.tanh(u @ B.T + b[None, :])


def operator5(phis, seed, radius=2.0):
    p = OperatorPlate(dim=6, seed=int(seed))
    return np.stack([plate_features(p, float(phi), radius, 5) for phi in phis])


def fit_eval(Ftr, Fte, train_coeff, test_images, mean, basis, size):
    loo, ridge = loo_score_from_features(Ftr, train_coeff, RIDGES)
    W = fit_map(Ftr, train_coeff, ridge)
    coeff = predict_map(Fte, W)
    imgs = decode_many(mean, basis, coeff, size)
    return {
        "rel_rmse": rel_rmse(imgs, test_images),
        "psnr": psnr(imgs, test_images),
        "train_loo_mse": float(loo),
        "ridge": float(ridge),
    }


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "min_diagnostic_only": float(np.min(a)),
        "max": float(np.max(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def run(size=48, n_train=8, basis_dim=6, radius=2.0, seeds=32, out_dir="splatworld3d_feature_controls"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_phis = np.arange(n_train) * (2 * math.pi / n_train)
    test_phis = train_phis + math.pi / n_train
    train_images = np.stack([render(float(p), size) for p in train_phis])
    test_images = np.stack([render(float(p), size) for p in test_phis])
    mean, basis, train_coeff, _ = fit_basis(train_images, basis_dim)

    A = fit_eval(address2(train_phis), address2(test_phis), train_coeff, test_images, mean, basis, size)
    H = fit_eval(harmonic5(train_phis), harmonic5(test_phis), train_coeff, test_images, mean, basis, size)

    random_rows = []
    operator_rows = []
    for seed in range(int(seeds)):
        random_rows.append({"seed": seed, **fit_eval(random5(train_phis, seed), random5(test_phis, seed), train_coeff, test_images, mean, basis, size)})
        operator_rows.append({"seed": seed, **fit_eval(operator5(train_phis, seed, radius), operator5(test_phis, seed, radius), train_coeff, test_images, mean, basis, size)})

    linear = linear_image_baseline(test_phis, train_phis, train_images)
    result = {
        "address2": A,
        "harmonic5": H,
        "random5_rel_rmse_distribution": summarize([r["rel_rmse"] for r in random_rows]),
        "operator5_rel_rmse_distribution": summarize([r["rel_rmse"] for r in operator_rows]),
        "random5_rows": random_rows,
        "operator5_rows": operator_rows,
        "linear_image_rel_rmse": rel_rmse(linear, test_images),
        "linear_image_psnr": psnr(linear, test_images),
        "operator_seed7_rel_rmse": float(operator_rows[7]["rel_rmse"]) if len(operator_rows) > 7 else None,
        "random_seed7_rel_rmse": float(random_rows[7]["rel_rmse"]) if len(random_rows) > 7 else None,
        "seeds": int(seeds),
        "note": "seed minima are diagnostic only; no test-selected seed is a valid model choice",
    }
    op_med = result["operator5_rel_rmse_distribution"]["median"]
    rnd_med = result["random5_rel_rmse_distribution"]["median"]
    result["operator_median_beats_random_median"] = bool(op_med < rnd_med)
    result["operator_median_beats_harmonic5"] = bool(op_med < H["rel_rmse"])
    result["operator_median_beats_linear_image"] = bool(op_med < result["linear_image_rel_rmse"])

    (out / "feature_controls.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def selftest():
    p = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    assert address2(p).shape == (8, 2)
    assert harmonic5(p).shape == (8, 5)
    assert random5(p, 0).shape == (8, 5)
    assert operator5(p, 0).shape == (8, 5)
    print("SplatWorld3D feature controls selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--basis_dim", type=int, default=6)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--out_dir", default="splatworld3d_feature_controls")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.size, args.n_train, args.basis_dim, args.radius, args.seeds, args.out_dir)


if __name__ == "__main__":
    main()
