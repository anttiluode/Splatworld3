#!/usr/bin/env python3
"""SplatWorld3D second attack: learn g for interpolation, not memorization.

The first external-world run optimized in-sample coefficient error while a
6-feature+bias readout was fitted on only 8 views. Its design matrix had
condition number >1000 and learned g slightly HURT true unseen views.

This preregistered correction changes one thing:
  - every candidate g is scored by leave-one-view-out prediction across the
    eight TRAINING cameras;
  - readout ridge is chosen only by that training-only LOO score from the fixed
    set {0.01, 0.1, 1.0};
  - the eight halfway TEST cameras are never touched until fitting is over.

No test image, test coefficient, test angle, 3D hit point, depth, or renderer
internal is used to choose g.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from operator_plate import OperatorPlate
from splatworld3d import (
    render,
    fit_basis,
    decode_coeff,
    plate_features,
    rel_rmse,
    psnr,
    linear_image_baseline,
    nearest_baseline,
)
from splatworld3d_diagnostic import fourier_design, ridge_fit

RIDGES = (0.01, 0.1, 1.0)


def feature_matrix(plate, phis, radius, feat_dim):
    return np.stack([plate_features(plate, p, radius, feat_dim) for p in phis])


def fit_map(F, C, ridge):
    X = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    return np.linalg.solve(X.T @ X + float(ridge) * np.eye(X.shape[1]), X.T @ C)


def predict_map(F, W):
    X = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    return X @ W


def loo_score_from_features(F, C, ridges=RIDGES):
    best = None
    for ridge in ridges:
        err = 0.0
        for i in range(len(F)):
            mask = np.ones(len(F), dtype=bool)
            mask[i] = False
            W = fit_map(F[mask], C[mask], ridge)
            p = predict_map(F[i:i+1], W)[0]
            err += float(np.mean((p - C[i]) ** 2))
        err /= len(F)
        rec = (err, float(ridge))
        if best is None or rec[0] < best[0]:
            best = rec
    return best


def loo_score(plate, phis, coeff, radius, feat_dim):
    F = feature_matrix(plate, phis, radius, feat_dim)
    err, ridge = loo_score_from_features(F, coeff)
    return err, ridge, F


def fit_plate_loo(train_phis, train_coeff, radius=2.0, feat_dim=5, steps=3000, seed=7):
    rng = np.random.default_rng(seed)
    plate = OperatorPlate(dim=6, seed=seed)
    base = plate.g.copy()
    initial_score, initial_ridge, _ = loo_score(plate, train_phis, train_coeff, radius, feat_dim)
    best = initial_score
    best_ridge = initial_ridge
    delta = 0.08
    accepted = 0

    for step in range(int(steps)):
        i, j = rng.choice(len(plate.g), 2, replace=False)
        if plate.g[i] <= delta + 1e-6:
            continue
        old = plate.g.copy()
        plate.g[i] -= delta
        plate.g[j] += delta
        val, ridge, _ = loo_score(plate, train_phis, train_coeff, radius, feat_dim)
        if val < best:
            best = val
            best_ridge = ridge
            accepted += 1
        else:
            plate.g[:] = old
        if step in (steps // 3, 2 * steps // 3):
            delta *= 0.5

    final_F = feature_matrix(plate, train_phis, radius, feat_dim)
    # Re-select ridge on final g from training LOO only, then fit all train views.
    final_loo, final_ridge = loo_score_from_features(final_F, train_coeff)
    W = fit_map(final_F, train_coeff, final_ridge)
    return plate, W, {
        "initial_loo_mse": float(initial_score),
        "initial_ridge": float(initial_ridge),
        "final_loo_mse": float(final_loo),
        "final_ridge": float(final_ridge),
        "accepted": int(accepted),
        "material_l2_change": float(np.linalg.norm(plate.g - base)),
        "sum_g": float(np.sum(plate.g)),
        "sum_g_base": float(np.sum(base)),
    }


def decode_many(mean, basis, coeff, size):
    return np.stack([decode_coeff(mean, basis, c, (size, size)) for c in coeff])


def run(size=48, n_train=8, basis_dim=6, feat_dim=5, steps=3000, radius=2.0, seed=7, out_dir="splatworld3d_loo"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_phis = np.arange(n_train) * (2 * math.pi / n_train)
    test_phis = train_phis + math.pi / n_train
    train_images = np.stack([render(float(p), size) for p in train_phis])
    test_images = np.stack([render(float(p), size) for p in test_phis])
    mean, basis, train_coeff, _ = fit_basis(train_images, basis_dim)
    test_coeff = (test_images.reshape(len(test_images), -1) - mean) @ basis.T
    basis_oracle = decode_many(mean, basis, test_coeff, size)

    # Untrained g with the SAME training-only LOO ridge selection.
    base = OperatorPlate(dim=6, seed=seed)
    F0 = feature_matrix(base, train_phis, radius, feat_dim)
    base_loo, base_ridge = loo_score_from_features(F0, train_coeff)
    W0 = fit_map(F0, train_coeff, base_ridge)
    base_test_F = feature_matrix(base, test_phis, radius, feat_dim)
    base_coeff = predict_map(base_test_F, W0)
    base_images = decode_many(mean, basis, base_coeff, size)

    plate, W, info = fit_plate_loo(train_phis, train_coeff, radius, feat_dim, steps, seed)
    Ft = feature_matrix(plate, test_phis, radius, feat_dim)
    pred_coeff = predict_map(Ft, W)
    pred_images = decode_many(mean, basis, pred_coeff, size)

    linear_images = linear_image_baseline(test_phis, train_phis, train_images)
    nearest_images = nearest_baseline(test_phis, train_phis, train_images)

    # Periodic Fourier baseline, ridge chosen by TRAINING LOO only.
    XF = fourier_design(train_phis, harmonics=3)
    ferr, fridge = loo_score_from_features(XF[:, 1:], train_coeff, RIDGES)  # intercept added by helper
    WF = fit_map(XF[:, 1:], train_coeff, fridge)
    fourier_coeff = predict_map(fourier_design(test_phis, harmonics=3)[:, 1:], WF)
    fourier_images = decode_many(mean, basis, fourier_coeff, size)

    result = {
        "verdict_question": "does LOO-trained g beat untrained g and plain image interpolation on untouched halfway views?",
        "operator_loo_rel_rmse": rel_rmse(pred_images, test_images),
        "operator_loo_psnr": psnr(pred_images, test_images),
        "untrained_operator_rel_rmse": rel_rmse(base_images, test_images),
        "untrained_operator_psnr": psnr(base_images, test_images),
        "linear_rel_rmse": rel_rmse(linear_images, test_images),
        "linear_psnr": psnr(linear_images, test_images),
        "nearest_rel_rmse": rel_rmse(nearest_images, test_images),
        "nearest_psnr": psnr(nearest_images, test_images),
        "fourier_rel_rmse": rel_rmse(fourier_images, test_images),
        "fourier_psnr": psnr(fourier_images, test_images),
        "basis_oracle_rel_rmse": rel_rmse(basis_oracle, test_images),
        "basis_oracle_psnr": psnr(basis_oracle, test_images),
        "operator_coeff_rel_rmse": rel_rmse(pred_coeff, test_coeff),
        "untrained_coeff_rel_rmse": rel_rmse(base_coeff, test_coeff),
        "fourier_coeff_rel_rmse": rel_rmse(fourier_coeff, test_coeff),
        "untrained_loo_mse": float(base_loo),
        "untrained_ridge": float(base_ridge),
        "fourier_train_loo_mse": float(ferr),
        "fourier_ridge": float(fridge),
        "feat_dim": int(feat_dim),
        "basis_dim": int(basis.shape[0]),
        **info,
    }
    result["g_beats_untrained"] = bool(result["operator_loo_rel_rmse"] < result["untrained_operator_rel_rmse"])
    result["g_beats_linear"] = bool(result["operator_loo_rel_rmse"] < result["linear_rel_rmse"])
    result["g_beats_fourier"] = bool(result["operator_loo_rel_rmse"] < result["fourier_rel_rmse"])
    result["verdict"] = (
        "EXTERNAL_WORLD_OPERATOR_INTERPOLATION_SURVIVES"
        if result["g_beats_untrained"] and result["g_beats_linear"]
        else "EXTERNAL_WORLD_OPERATOR_INTERPOLATION_NOT_YET_SHOWN"
    )

    (out / "loo_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "loo_data.npz",
        truth=test_images,
        operator=pred_images,
        untrained=base_images,
        linear=linear_images,
        nearest=nearest_images,
        fourier=fourier_images,
        basis_oracle=basis_oracle,
        g=plate.g,
    )
    print(json.dumps(result, indent=2))
    return result


def selftest():
    phis = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    C = np.stack([np.sin(phis), np.cos(phis)], axis=1)
    p = OperatorPlate(dim=6, seed=2)
    F = feature_matrix(p, phis, 2.0, 5)
    e, r = loo_score_from_features(F, C)
    assert np.isfinite(e) and r in RIDGES
    p2, W, info = fit_plate_loo(phis, C, 2.0, 5, 30, 2)
    assert abs(info["sum_g"] - info["sum_g_base"]) < 1e-10
    print("SplatWorld3D LOO selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--basis_dim", type=int, default=6)
    ap.add_argument("--feat_dim", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default="splatworld3d_loo")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.size, args.n_train, args.basis_dim, args.feat_dim, args.steps, args.radius, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
