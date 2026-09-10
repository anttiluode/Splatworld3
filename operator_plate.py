#!/usr/bin/env python3
"""Small address-conditioned operator substrate for SplatWorld3.

This is deliberately tiny and inspectable. One persistent material vector g
controls a reciprocal graph. A query address a=(x,y,z) changes the diagonal
loading of that graph, and the inverse response is a full dense operator:

    M_g(a) = [K(g) + D(a) + i*gamma I]^-1

The visual app does not store a separate matrix for every address. It stores g
and regenerates M_g(a) when the address changes.

The conservative mutation rule mirrors the spirit of ObjektiYksi: a small
amount of coupling is moved from one edge to another, so total material stays
constant. The module is standalone NumPy and has its own self-test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    return v / (np.linalg.norm(v) + EPS)


def _node_coords(n: int) -> np.ndarray:
    """Deterministic 3-D coordinates used only to make address loading smooth."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = 2.0 * math.pi * i / phi
        pts.append([r * math.cos(th), r * math.sin(th), z])
    return np.asarray(pts, np.float64)


def _edges(n: int) -> np.ndarray:
    """Sparse connected graph: ring plus deterministic chord families."""
    e = set()
    for i in range(n):
        for step in (1, 2):
            j = (i + step) % n
            a, b = sorted((i, j))
            if a != b:
                e.add((a, b))
    return np.asarray(sorted(e), np.int32)


@dataclass
class PlateStats:
    address: np.ndarray
    response_norm: float
    condition: float
    matrix_delta_from_origin: float


class OperatorPlate:
    """One persistent material vector -> an address-conditioned matrix family."""

    def __init__(
        self,
        dim: int = 8,
        seed: int = 17,
        material: np.ndarray | None = None,
        damping: float = 0.35,
        address_gain: float = 0.55,
        onsite: float = 2.75,
        carrier: float = 1.15,
    ):
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = int(dim)
        self.seed = int(seed)
        self.damping = float(damping)
        self.address_gain = float(address_gain)
        self.onsite = float(onsite)
        self.carrier = float(carrier)
        self.coords = _node_coords(self.dim)
        self.edges = _edges(self.dim)
        rng = np.random.default_rng(self.seed)
        if material is None:
            g = 0.28 + 0.08 * rng.random(len(self.edges))
        else:
            g = np.asarray(material, np.float64).copy()
            if g.shape != (len(self.edges),):
                raise ValueError(f"material must have shape {(len(self.edges),)}")
        self.g = g
        self._initial_g = g.copy()
        self._undo: list[np.ndarray] = []

        phase = np.linspace(0.0, 2.0 * math.pi, self.dim, endpoint=False)
        amp = 0.75 + 0.25 * rng.random(self.dim)
        self.drive = _unit(amp * np.exp(1j * phase))

    @property
    def material_sum(self) -> float:
        return float(self.g.sum())

    def stiffness(self) -> np.ndarray:
        k = np.eye(self.dim, dtype=np.float64) * self.onsite
        for gij, (i, j) in zip(self.g, self.edges):
            k[i, i] += gij
            k[j, j] += gij
            k[i, j] -= gij
            k[j, i] -= gij
        return k

    def system_matrix(self, address) -> np.ndarray:
        a = np.asarray(address, np.float64).reshape(3)
        detune = self.address_gain * (self.coords @ a)
        omega = self.carrier * (1.0 + 0.16 * np.tanh(a[2]))
        diag = detune - omega * omega
        A = self.stiffness().astype(np.complex128)
        A += np.diag(diag + 1j * self.damping * omega)
        return A

    def matrix(self, address) -> np.ndarray:
        """Dense complex operator exposed by this address."""
        return np.linalg.inv(self.system_matrix(address))

    def raw_response(self, address) -> np.ndarray:
        return self.matrix(address) @ self.drive

    def response(self, address, center=True) -> np.ndarray:
        """Bounded real coefficients used to drive SplatWorld's local basis."""
        r = self.raw_response(address)
        if center:
            r = r - self.raw_response((0.0, 0.0, 0.0))
        return np.tanh(3.5 * np.real(r)).astype(np.float32)

    def stats(self, address) -> PlateStats:
        a = np.asarray(address, np.float64).reshape(3)
        M = self.matrix(a)
        M0 = self.matrix((0.0, 0.0, 0.0))
        return PlateStats(
            address=a,
            response_norm=float(np.linalg.norm(self.response(a))),
            condition=float(np.linalg.cond(self.system_matrix(a))),
            matrix_delta_from_origin=float(
                np.linalg.norm(M - M0) / (np.linalg.norm(M0) + EPS)
            ),
        )

    def mutate(self, rng: np.random.Generator | None = None, delta=0.025) -> bool:
        """Move coupling between two edges while conserving total material."""
        if rng is None:
            rng = np.random.default_rng()
        if len(self.g) < 2:
            return False
        for _ in range(64):
            src, dst = rng.choice(len(self.g), 2, replace=False)
            amount = min(float(delta), max(0.0, self.g[src] - 0.035))
            if amount <= 0:
                continue
            self._undo.append(self.g.copy())
            self.g[src] -= amount
            self.g[dst] += amount
            return True
        return False

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.g = self._undo.pop()
        return True

    def reset(self) -> None:
        self._undo.append(self.g.copy())
        self.g = self._initial_g.copy()

    def objective(self, examples) -> float:
        """Mean matrix-family error over [(address, target_matrix), ...]."""
        if not examples:
            return 0.0
        vals = []
        for address, target in examples:
            M = self.matrix(address)
            T = np.asarray(target, np.complex128)
            vals.append(np.linalg.norm(M - T) ** 2 / (np.linalg.norm(T) ** 2 + EPS))
        return float(np.mean(vals))

    def dither_fit(self, examples, steps=1000, seed=0, delta=0.02) -> dict:
        """Derivative-free conservative fitting for small demos/tests."""
        rng = np.random.default_rng(seed)
        before = self.objective(examples)
        best = before
        accepted = 0
        for _ in range(int(steps)):
            old = self.g.copy()
            n0 = len(self._undo)
            if not self.mutate(rng, delta=delta):
                continue
            if len(self._undo) > n0:
                self._undo.pop()
            score = self.objective(examples)
            if score <= best:
                best = score
                accepted += 1
            else:
                self.g = old
        return {"before": before, "after": best, "accepted": accepted}

    def save(self, path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path,
            dim=np.int32(self.dim),
            seed=np.int64(self.seed),
            g=self.g.astype(np.float64),
            damping=np.float64(self.damping),
            address_gain=np.float64(self.address_gain),
            onsite=np.float64(self.onsite),
            carrier=np.float64(self.carrier),
        )
        return path

    @classmethod
    def load(cls, path) -> "OperatorPlate":
        d = np.load(path)
        return cls(
            dim=int(d["dim"]),
            seed=int(d["seed"]),
            material=d["g"],
            damping=float(d["damping"]),
            address_gain=float(d["address_gain"]),
            onsite=float(d["onsite"]),
            carrier=float(d["carrier"]),
        )


def selftest(verbose=True) -> bool:
    ok = True

    def check(name, cond, note=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")

    p = OperatorPlate(dim=6, seed=1)
    m0 = p.matrix((0, 0, 0))
    m1 = p.matrix((0.8, -0.4, 0.2))
    check("matrix finite", np.isfinite(m0).all() and np.isfinite(m1).all())
    check("address changes matrix", np.linalg.norm(m1 - m0) > 1e-3)
    check("same address deterministic", np.allclose(m1, p.matrix((0.8, -0.4, 0.2))))

    total = p.material_sum
    r_before = p.response((0.6, 0.2, 0.0)).copy()
    p.mutate(np.random.default_rng(2), delta=0.03)
    r_after = p.response((0.6, 0.2, 0.0)).copy()
    check("mutation conserves material", abs(p.material_sum - total) < 1e-12)
    check("one material edit changes exposed response", np.linalg.norm(r_after - r_before) > 1e-6)
    p.undo()
    check("undo restores material", abs(p.material_sum - total) < 1e-12)

    teacher = OperatorPlate(dim=6, seed=7)
    student = OperatorPlate(dim=6, seed=7)
    rr = np.random.default_rng(99)
    for _ in range(18):
        teacher.mutate(rr, delta=0.018)
    addresses = [
        (-0.8, -0.5, -0.2),
        (0.7, -0.3, 0.1),
        (-0.2, 0.8, 0.3),
        (0.65, 0.65, -0.1),
    ]
    examples = [(a, teacher.matrix(a)) for a in addresses]
    fit = student.dither_fit(examples, steps=2200, seed=123, delta=0.006)
    check(
        "dither fit improves matrix family",
        fit["after"] < 0.55 * fit["before"],
        f"{fit['before']:.4g}->{fit['after']:.4g}",
    )
    check(
        "fit keeps conserved material",
        abs(student.material_sum - OperatorPlate(dim=6, seed=7).material_sum) < 1e-10,
    )

    if verbose:
        print("operator_plate selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if selftest(True) else 1)
