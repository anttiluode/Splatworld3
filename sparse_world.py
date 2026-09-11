#!/usr/bin/env python3
"""Sparse-world interpolation experiment for SplatWorld3.

Question
--------
If one persistent plate g defines an address-conditioned operator family, can a
student plate infer the *untaught* parts of that family from only a few taught
addresses?

The test is designed to avoid the obvious fovea/radial confound. Four taught
addresses all lie at the same radius. Held-out evaluation includes intermediate
angles on exactly that same ring, so center-to-periphery falloff cannot explain
success.

A hidden teacher plate is made from the same starting plate by many conservative
edge->edge material transfers. The student never sees teacher material. It sees
only responses at four cardinal addresses.

Three arms are compared:
  FULL     all 6 response components at each of 4 addresses (positive control)
  SPARSE   2 fixed projections per address = 8 scalar observations total
           (fewer than the 11 independent material DOF after sum(g) conservation)
  SHUFFLED same sparse observations assigned to the wrong addresses (kill control)

Training is derivative-free. At each round every legal conservative one-edge
transfer is tried, the best improving transfer is kept, and the step size is
reduced over a fixed schedule.

The decoder is used only for final visualization. All learning and verdicts are
computed in the small operator response space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from operator_plate import OperatorPlate
from splatworld2 import OnnxDecoder, MockDecoder, find_model, random_identity
from splatworld3 import operator_match_scale
from operator_atlas import make_basis, sample_state, face_atlas, metric_panel

EPS = 1e-12


def responses(plate: OperatorPlate, addresses) -> np.ndarray:
    """Centered bounded response vectors, computing origin only once."""
    r0 = plate.raw_response((0.0, 0.0, 0.0))
    out = []
    for a in addresses:
        r = plate.raw_response(a) - r0
        out.append(np.tanh(3.5 * np.real(r)))
    return np.asarray(out, np.float64)


def projected_objective(plate, addresses, target_measurements, projector) -> float:
    pred = responses(plate, addresses) @ projector.T
    return float(np.mean((pred - target_measurements) ** 2))


def best_conservative_dither(
    plate: OperatorPlate,
    addresses,
    target_measurements,
    projector,
    deltas=(0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125),
    rounds_per_delta=12,
):
    """Derivative-free best local edge->edge transfer search.

    Every candidate conserves sum(g). No gradient, teacher material, decoder
    pixel, or held-out address is consulted.
    """
    best = projected_objective(plate, addresses, target_measurements, projector)
    before = best
    history = []

    for delta in deltas:
        for _ in range(int(rounds_per_delta)):
            g0 = plate.g.copy()
            winner = None
            for src in range(len(g0)):
                amount = min(float(delta), max(0.0, float(g0[src]) - 0.035))
                if amount <= 0:
                    continue
                for dst in range(len(g0)):
                    if src == dst:
                        continue
                    candidate = g0.copy()
                    candidate[src] -= amount
                    candidate[dst] += amount
                    plate.g = candidate
                    score = projected_objective(
                        plate, addresses, target_measurements, projector
                    )
                    if winner is None or score < winner[0]:
                        winner = (score, candidate.copy(), src, dst, amount)

            plate.g = g0
            if winner is None or winner[0] >= best - 1e-15:
                break

            best, candidate, src, dst, amount = winner
            plate.g = candidate
            history.append(
                {
                    "delta": float(amount),
                    "objective": float(best),
                    "src": int(src),
                    "dst": int(dst),
                }
            )

    return {
        "before": float(before),
        "after": float(best),
        "accepted": len(history),
        "history": history,
    }


def hidden_teacher(dim, seed, steps, delta, mutation_seed):
    p = OperatorPlate(dim=dim, seed=seed)
    rng = np.random.default_rng(mutation_seed)
    for _ in range(int(steps)):
        if not p.mutate(rng, delta=float(delta)):
            raise RuntimeError("teacher mutation failed")
        # Teacher construction history is irrelevant; keep memory bounded.
        if p._undo:
            p._undo.pop()
    return p


def cardinal_anchors(radius):
    r = float(radius)
    return np.asarray(
        [[r, 0.0, 0.0], [0.0, r, 0.0], [-r, 0.0, 0.0], [0.0, -r, 0.0]],
        np.float64,
    )


def heldout_ring(radius, n=64):
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    pts = np.asarray([[radius * np.cos(t), radius * np.sin(t), 0.0] for t in th])
    # Cardinal anchors are indices 0, n/4, n/2, 3n/4 for n divisible by four.
    mask = np.ones(len(pts), dtype=bool)
    if len(pts) % 4 == 0:
        mask[[0, len(pts)//4, len(pts)//2, 3*len(pts)//4]] = False
    return pts[mask]


def square_grid(span, n):
    xs = np.linspace(-span, span, int(n))
    return np.asarray([[x, y, 0.0] for y in xs for x in xs], np.float64)


def relative_field_error(student, teacher, addresses):
    s = responses(student, addresses)
    t = responses(teacher, addresses)
    return float(np.linalg.norm(s - t) / (np.linalg.norm(t) + EPS))


def field_cosines(student, teacher, addresses):
    s = responses(student, addresses)
    t = responses(teacher, addresses)
    c = np.sum(s * t, axis=1) / (
        np.linalg.norm(s, axis=1) * np.linalg.norm(t, axis=1) + EPS
    )
    return {
        "median": float(np.median(c)),
        "mean": float(np.mean(c)),
        "p10": float(np.quantile(c, 0.10)),
    }


def radial_only_error(teacher, ring_addresses):
    """Same-radius constant predictor: strongest possible pure-radius baseline."""
    t = responses(teacher, ring_addresses)
    mu = np.mean(t, axis=0, keepdims=True)
    return float(np.linalg.norm(t - mu) / (np.linalg.norm(t) + EPS))


def per_address_error(student, teacher, addresses):
    s = responses(student, addresses)
    t = responses(teacher, addresses)
    den = np.linalg.norm(t, axis=1) + 1e-9
    return np.linalg.norm(s - t, axis=1) / den


def build_decoder(args):
    if args.mock:
        print("decoder: mock")
        return MockDecoder()
    path = find_model(args.model)
    dec = OnnxDecoder(path, cpu=args.cpu)
    print("decoder:", path, "backend:", dec.backend)
    return dec


def render_plate(dec, z0, B, plate, op_scale, args, title, path):
    state = sample_state(dec, z0, B, plate, op_scale, args)
    img = face_atlas(state, tile=args.tile, title=title)
    import cv2 as cv
    cv.imwrite(str(path), img)
    return state


def save_error_panels(outdir, args, teacher, plates):
    import cv2 as cv
    addrs = square_grid(args.address_span, args.grid)
    panels = []
    for name, plate in plates:
        err = per_address_error(plate, teacher, addrs).reshape(args.grid, args.grid)
        panels.append(metric_panel(err, f"held-out response error: {name}", cell=args.metric_cell))
    gap = 10
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((h, w, 3), 20, np.uint8)
    x = 0
    for p in panels:
        canvas[:p.shape[0], x:x+p.shape[1]] = p
        x += p.shape[1] + gap
    cv.imwrite(str(outdir / "sparse_world_error_maps.png"), canvas)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    anchors = cardinal_anchors(args.anchor_radius)
    ring = heldout_ring(args.anchor_radius, args.ring_points)
    grid_eval = square_grid(args.eval_span, args.eval_grid)

    base = OperatorPlate(dim=args.operator_dim, seed=args.plate_seed)
    teacher = hidden_teacher(
        args.operator_dim, args.plate_seed,
        args.teacher_mutations, args.teacher_delta, args.teacher_seed,
    )
    material_sum = base.material_sum

    target_response = responses(teacher, anchors)

    rng = np.random.default_rng(args.projector_seed)
    projector = rng.normal(size=(args.measurement_dim, args.operator_dim))
    projector /= np.linalg.norm(projector, axis=1, keepdims=True) + EPS
    sparse_targets = target_response @ projector.T

    sparse = OperatorPlate(dim=args.operator_dim, seed=args.plate_seed)
    sparse_fit = best_conservative_dither(
        sparse, anchors, sparse_targets, projector,
        deltas=args.fit_deltas, rounds_per_delta=args.rounds_per_delta,
    )

    full = OperatorPlate(dim=args.operator_dim, seed=args.plate_seed)
    full_projector = np.eye(args.operator_dim, dtype=np.float64)
    full_fit = best_conservative_dither(
        full, anchors, target_response, full_projector,
        deltas=args.fit_deltas, rounds_per_delta=args.rounds_per_delta,
    )

    shuffled = OperatorPlate(dim=args.operator_dim, seed=args.plate_seed)
    shuffled_targets = np.roll(sparse_targets, 1, axis=0)
    shuffled_fit = best_conservative_dither(
        shuffled, anchors, shuffled_targets, projector,
        deltas=args.fit_deltas, rounds_per_delta=args.rounds_per_delta,
    )

    def errors(plate):
        return {
            "anchors": relative_field_error(plate, teacher, anchors),
            "heldout_same_radius_ring": relative_field_error(plate, teacher, ring),
            "heldout_grid": relative_field_error(plate, teacher, grid_eval),
            "ring_cosine": field_cosines(plate, teacher, ring),
        }

    result = {
        "design": {
            "operator_dim": int(args.operator_dim),
            "edges": int(len(base.g)),
            "independent_material_dof_after_conservation": int(len(base.g) - 1),
            "anchor_radius": float(args.anchor_radius),
            "anchors": anchors.tolist(),
            "measurement_dim_per_anchor": int(args.measurement_dim),
            "sparse_scalar_observations": int(len(anchors) * args.measurement_dim),
            "full_scalar_observations": int(len(anchors) * args.operator_dim),
            "teacher_mutations": int(args.teacher_mutations),
            "teacher_delta": float(args.teacher_delta),
        },
        "fit": {
            "sparse": {k: v for k, v in sparse_fit.items() if k != "history"},
            "full": {k: v for k, v in full_fit.items() if k != "history"},
            "shuffled": {k: v for k, v in shuffled_fit.items() if k != "history"},
        },
        "errors": {
            "base": errors(base),
            "sparse": errors(sparse),
            "full": errors(full),
            "shuffled": errors(shuffled),
            "radial_only_same_radius_ring": radial_only_error(teacher, ring),
        },
        "material": {
            "sum_base": float(material_sum),
            "sum_teacher": float(teacher.material_sum),
            "sum_sparse": float(sparse.material_sum),
            "sum_full": float(full.material_sum),
            "sum_shuffled": float(shuffled.material_sum),
            "teacher_distance_from_base": float(np.linalg.norm(teacher.g - base.g)),
            "sparse_distance_from_teacher": float(np.linalg.norm(sparse.g - teacher.g)),
            "full_distance_from_teacher": float(np.linalg.norm(full.g - teacher.g)),
        },
    }

    b_ring = result["errors"]["base"]["heldout_same_radius_ring"]
    s_ring = result["errors"]["sparse"]["heldout_same_radius_ring"]
    f_ring = result["errors"]["full"]["heldout_same_radius_ring"]
    x_ring = result["errors"]["shuffled"]["heldout_same_radius_ring"]
    radial = result["errors"]["radial_only_same_radius_ring"]

    checks = {
        "full_positive_control": bool(f_ring < 0.02),
        "sparse_beats_base_by_2x": bool(s_ring < 0.5 * b_ring),
        "sparse_beats_shuffled_by_2x": bool(s_ring < 0.5 * x_ring),
        "sparse_beats_radial_only_by_3x": bool(s_ring < radial / 3.0),
        "all_material_conserved": bool(
            max(abs(result["material"][k] - material_sum) for k in
                ("sum_teacher", "sum_sparse", "sum_full", "sum_shuffled")) < 1e-10
        ),
        "sparse_is_underdetermined": bool(
            result["design"]["sparse_scalar_observations"]
            < result["design"]["independent_material_dof_after_conservation"]
        ),
    }
    result["checks"] = checks
    result["verdict"] = (
        "SPARSE_SAME_RADIUS_INTERPOLATION_SURVIVES_FOVEA_CONTROL"
        if all(checks.values()) else "CRITERIA_NOT_ALL_MET"
    )

    with open(outdir / "sparse_world_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    np.savez_compressed(
        outdir / "sparse_world_numeric.npz",
        anchors=anchors, projector=projector,
        base_g=base.g, teacher_g=teacher.g, sparse_g=sparse.g,
        full_g=full.g, shuffled_g=shuffled.g,
    )
    teacher.save(outdir / "teacher_plate.npz")
    sparse.save(outdir / "sparse_student_plate.npz")
    full.save(outdir / "full_student_plate.npz")

    print("\nSPARSE WORLD INTERPOLATION")
    print("  material DOF after conservation:", len(base.g) - 1)
    print("  sparse observations:", len(anchors) * args.measurement_dim)
    print(f"  base held-out same-radius ring error:     {b_ring:.6f}")
    print(f"  sparse held-out same-radius ring error:   {s_ring:.6f}")
    print(f"  full held-out same-radius ring error:     {f_ring:.6f}")
    print(f"  shuffled held-out same-radius ring error: {x_ring:.6f}")
    print(f"  radius-only predictor error on same ring: {radial:.6f}")
    print("  verdict:", result["verdict"])
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    if not args.no_render:
        dec = build_decoder(args)
        rng_id = np.random.default_rng(args.seed)
        z0 = random_identity(rng_id, args.anchor_std)
        B = make_basis(dec, z0, args, verbose=True)
        op_scale = operator_match_scale(base, len(B), min(args.address_span, 1.25))

        render_plate(dec, z0, B, teacher, op_scale, args,
                     "HIDDEN TEACHER WORLD", outdir / "teacher_faces.png")
        render_plate(dec, z0, B, base, op_scale, args,
                     "STUDENT BEFORE / NO TEACHING", outdir / "student_before_faces.png")
        render_plate(dec, z0, B, sparse, op_scale, args,
                     "STUDENT AFTER 8 SCALAR OBSERVATIONS", outdir / "student_sparse_faces.png")
        render_plate(dec, z0, B, full, op_scale, args,
                     "FULL POSITIVE CONTROL / 24 SCALARS", outdir / "student_full_faces.png")
        render_plate(dec, z0, B, shuffled, op_scale, args,
                     "SHUFFLED-ADDRESS CONTROL", outdir / "student_shuffled_faces.png")
        save_error_panels(
            outdir, args, teacher,
            [("base", base), ("sparse", sparse), ("full", full), ("shuffled", shuffled)],
        )

    return result


def selftest():
    class A:
        pass
    args = A()
    args.operator_dim = 6
    args.plate_seed = 17
    args.teacher_mutations = 24
    args.teacher_delta = 0.02
    args.teacher_seed = 777
    args.anchor_radius = 1.6
    args.ring_points = 32
    args.eval_span = 1.8
    args.eval_grid = 7
    args.projector_seed = 2026
    args.measurement_dim = 2
    args.fit_deltas = (0.02, 0.01, 0.005, 0.0025)
    args.rounds_per_delta = 5
    args.outdir = str(Path("_sparse_world_selftest"))
    args.no_render = True
    result = run(args)
    ok = (
        np.isfinite(result["errors"]["sparse"]["heldout_grid"])
        and abs(result["material"]["sum_sparse"] - result["material"]["sum_base"]) < 1e-10
        and result["design"]["sparse_scalar_observations"] == 8
    )
    print("sparse_world selftest:", "PASS" if ok else "FAIL")
    return bool(ok)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default="sparse_world_results")
    ap.add_argument("--operator_dim", type=int, default=6)
    ap.add_argument("--plate_seed", type=int, default=17)
    ap.add_argument("--teacher_seed", type=int, default=777)
    ap.add_argument("--teacher_mutations", type=int, default=80)
    ap.add_argument("--teacher_delta", type=float, default=0.04)
    ap.add_argument("--anchor_radius", type=float, default=2.0)
    ap.add_argument("--ring_points", type=int, default=64)
    ap.add_argument("--measurement_dim", type=int, default=2)
    ap.add_argument("--projector_seed", type=int, default=2026)
    ap.add_argument("--rounds_per_delta", type=int, default=12)
    ap.add_argument("--fit_deltas", type=float, nargs="+",
                    default=[0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125])
    ap.add_argument("--eval_span", type=float, default=2.2)
    ap.add_argument("--eval_grid", type=int, default=17)

    # Rendering options. Defaults stay inside the central usable chart instead
    # of deliberately driving into the old +/-5, gain=9 fire horizon.
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--anchor_std", type=float, default=0.55)
    ap.add_argument("--probe_dirs", type=int, default=40)
    ap.add_argument("--probe_eps", type=float, default=0.45)
    ap.add_argument("--address_span", type=float, default=2.4)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--motion_gain", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tile", type=int, default=92)
    ap.add_argument("--metric_cell", type=int, default=48)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    run(args)
