#!/usr/bin/env python3
"""Diagnostic for the first SplatWorld3D external-world run.

The first external-world result beat nearest-frame only slightly and lost to
plain image interpolation. Before changing the architecture, this script asks:

1. Did conservative learning of g improve held-out prediction over the same
   operator readout with the initial/untrained g?
2. How ill-conditioned is the fitted operator->image coefficient readout?
3. How does a simple periodic Fourier pose baseline compare?

No test view participates in any fit.
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
    fit_output_map,
    predict_coeff,
    fit_plate,
    linear_image_baseline,
    nearest_baseline,
    rel_rmse,
    psnr,
)


def fourier_design(phis: np.ndarray, harmonics: int = 3) -> np.ndarray:
    cols = [np.ones(len(phis), dtype=np.float64)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(k * phis), np.sin(k * phis)]
    return np.stack(cols, axis=1)


def ridge_fit(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def decode_many(mean, basis, coeff, size):
    return np.stack([decode_coeff(mean, basis, c, (size, size)) for c in coeff])


def evaluate(size=48, n_train=8, basis_dim=6, steps=2500, radius=2.0, seed=7, out_dir="splatworld3d_diag"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_phis = np.arange(n_train) * (2 * math.pi / n_train)
    test_phis = train_phis + math.pi / n_train
    train_images = np.stack([render(float(p), size) for p in train_phis])
    test_images = np.stack([render(float(p), size) for p in test_phis])
    mean, basis, train_coeff, _ = fit_basis(train_images, basis_dim)
    test_coeff = (test_images.reshape(len(test_images), -1) - mean) @ basis.T
    basis_oracle = decode_many(mean, basis, test_coeff, size)

    # Same readout architecture, but g is untouched.
    base = OperatorPlate(dim=6, seed=seed)
    F0 = np.stack([plate_features(base, p, radius, basis.shape[0]) for p in train_phis])
    W0 = fit_output_map(F0, train_coeff)
    base_coeff = predict_coeff(base, test_phis, radius, W0, basis.shape[0])
    base_images = decode_many(mean, basis, base_coeff, size)
    X0 = np.concatenate([F0, np.ones((len(F0), 1))], axis=1)

    # Learned material, same output-map fitting rule as the first run.
    plate, W, info = fit_plate(train_phis, train_coeff, radius, basis.shape[0], steps, seed)
    F1 = np.stack([plate_features(plate, p, radius, basis.shape[0]) for p in train_phis])
    learned_coeff = predict_coeff(plate, test_phis, radius, W, basis.shape[0])
    learned_images = decode_many(mean, basis, learned_coeff, size)
    X1 = np.concatenate([F1, np.ones((len(F1), 1))], axis=1)

    # Strong, explicit periodic smoothness baseline. This baseline knows camera
    # angle but no 3D geometry and sees only the same training coefficients.
    XF = fourier_design(train_phis, harmonics=3)
    WF = ridge_fit(XF, train_coeff, ridge=1e-3)
    fourier_coeff = fourier_design(test_phis, harmonics=3) @ WF
    fourier_images = decode_many(mean, basis, fourier_coeff, size)

    linear_images = linear_image_baseline(test_phis, train_phis, train_images)
    nearest_images = nearest_baseline(test_phis, train_phis, train_images)

    result = {
        "operator_untrained_rel_rmse": rel_rmse(base_images, test_images),
        "operator_untrained_psnr": psnr(base_images, test_images),
        "operator_learned_rel_rmse": rel_rmse(learned_images, test_images),
        "operator_learned_psnr": psnr(learned_images, test_images),
        "g_learning_relative_improvement": float((rel_rmse(base_images, test_images) - rel_rmse(learned_images, test_images)) / (rel_rmse(base_images, test_images) + 1e-12)),
        "fourier_rel_rmse": rel_rmse(fourier_images, test_images),
        "fourier_psnr": psnr(fourier_images, test_images),
        "linear_rel_rmse": rel_rmse(linear_images, test_images),
        "linear_psnr": psnr(linear_images, test_images),
        "nearest_rel_rmse": rel_rmse(nearest_images, test_images),
        "nearest_psnr": psnr(nearest_images, test_images),
        "basis_oracle_rel_rmse": rel_rmse(basis_oracle, test_images),
        "basis_oracle_psnr": psnr(basis_oracle, test_images),
        "base_coeff_rel_rmse": rel_rmse(base_coeff, test_coeff),
        "learned_coeff_rel_rmse": rel_rmse(learned_coeff, test_coeff),
        "fourier_coeff_rel_rmse": rel_rmse(fourier_coeff, test_coeff),
        "base_readout_design_condition": float(np.linalg.cond(X0)),
        "learned_readout_design_condition": float(np.linalg.cond(X1)),
        "material_l2_change": float(np.linalg.norm(plate.g - base.g)),
        "accepted_mutations": int(info["accepted"]),
        "sum_g_base": float(np.sum(base.g)),
        "sum_g_learned": float(np.sum(plate.g)),
    }
    (out / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "diagnostic_data.npz",
        test_images=test_images,
        untrained_operator=base_images,
        learned_operator=learned_images,
        fourier=fourier_images,
        linear=linear_images,
        nearest=nearest_images,
        basis_oracle=basis_oracle,
        g_base=base.g,
        g_learned=plate.g,
    )
    print(json.dumps(result, indent=2))
    return result


def selftest():
    p = np.array([0.0, 0.4, 1.2])
    X = fourier_design(p, 3)
    assert X.shape == (3, 7)
    assert np.isfinite(X).all()
    print("SplatWorld3D diagnostic selftest: PASS")


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
    ap.add_argument("--out_dir", default="splatworld3d_diag")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run:
        ap.error("use --selftest or --run")
    evaluate(args.size, args.n_train, args.basis_dim, args.steps, args.radius, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
