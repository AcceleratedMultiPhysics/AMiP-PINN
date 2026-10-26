"""Invariant checks for the DG package esdg (Qu, Narayan & Chan 2026).

Run: python tests/test_dg_invariants.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esdg import hyperreduction as hr  # noqa: E402
from esdg import rom as R  # noqa: E402
from esdg.fom import DGFOM1D, gauss_lobatto  # noqa: E402
from esdg.physics import Advection, Burgers, Euler1D  # noqa: E402

rng = np.random.default_rng(0)


def random_euler_states(n):
    phys = Euler1D()
    return phys, phys.prim_to_cons(1 + rng.random(n), rng.standard_normal(n), 1 + rng.random(n))


def test_fluxes():
    for phys, uL, uR in [
        (Advection(0.7), rng.standard_normal((1, 50)), rng.standard_normal((1, 50))),
        (Burgers(), rng.standard_normal((1, 50)), rng.standard_normal((1, 50))),
        (Euler1D(), random_euler_states(50)[1], random_euler_states(50)[1]),
    ]:
        assert np.allclose(phys.flux_ec(uL, uL), phys.flux(uL), atol=1e-13)
        assert np.allclose(phys.flux_ec(uL, uR), phys.flux_ec(uR, uL), atol=1e-14)
        shuffle = np.sum((phys.entropy_vars(uL) - phys.entropy_vars(uR)) * phys.flux_ec(uL, uR), 0)
        assert np.max(np.abs(shuffle - (phys.psi(uL) - phys.psi(uR)))) < 1e-12


def test_gll():
    for p in [1, 3, 7]:
        r, w, D = gauss_lobatto(p)
        for k in range(2 * p):                  # GLL is exact to degree 2p-1
            assert abs(w @ r ** k - (1 - (-1) ** (k + 1)) / (k + 1)) < 1e-13
        for k in range(p + 1):                  # D differentiates degree <= p exactly
            assert np.allclose(D @ r ** k, k * r ** max(k - 1, 0), atol=1e-11)
        Q = w[:, None] * D                      # SBP: Q + Q^T = B
        B = np.zeros((p + 1, p + 1))
        B[0, 0], B[-1, -1] = -1, 1
        assert np.allclose(Q + Q.T, B, atol=1e-13)


def test_global_operators():
    for p in [0, 3, 7]:
        f = DGFOM1D(Euler1D(), 16, p)
        Q = f.Q.toarray()
        assert np.allclose(Q @ np.ones(f.n), 0, atol=1e-13)               # Theorem 1
        assert np.allclose(Q + Q.T, 0, atol=1e-13)
        assert abs(f.mass.sum() - 2) < 1e-13
        _, u = random_euler_states(f.n)
        assert abs(f.entropy_residual(u)) < 1e-11                          # Theorem 2
    # p = 0 is the FV scheme: M = h I, Q = 1/2 central difference
    f = DGFOM1D(Euler1D(), 8, 0)
    Qfv = 0.5 * (np.eye(8, k=1) - np.eye(8, k=-1))
    Qfv[0, -1], Qfv[-1, 0] = -0.5, 0.5
    assert np.allclose(f.Q.toarray(), Qfv) and np.allclose(f.mass, 0.25)


def _setup():
    phys = Euler1D()
    f = DGFOM1D(phys, 50, 3, eps=1e-3)
    snaps = []
    for k in range(10):
        rho = 2 + 0.3 * np.exp(-50 * (f.x - 0.1 * k + 0.3) ** 2)
        snaps.append(phys.prim_to_cons(rho, 0.1 * np.sin(np.pi * (k + 1) * f.x / 3), rho ** 1.4))
    snaps = np.array(snaps)
    modes, s = R.snapshot_svd(snaps, phys, f.mass)
    return f, modes[:, :12], snaps


def test_rom_structure():
    f, V, snaps = _setup()
    assert np.allclose(V.T @ (f.mass[:, None] * V), np.eye(12), atol=1e-12)
    gal = R.GalerkinROM(f, V)
    uN = gal.project(snaps[3])
    assert abs(gal.entropy_residual(uN)) < 1e-11                           # Theorem 3
    for stab in ["stacked", "union"]:
        rom, info = R.build_hyper_rom(f, V, 1e-4, stab=stab)
        assert np.allclose(rom.Qbar, -rom.Qbar.T, atol=1e-12)
        assert np.allclose(rom.Qbar @ np.ones(len(rom.I)), 0, atol=1e-10)
        assert np.allclose(rom.Pt @ rom.Vt[rom.I], np.eye(rom.Vt.shape[1]), atol=1e-8)
        assert np.all(rom.w > 0)
        assert abs(rom.entropy_residual(uN)) < 1e-10
    rom, _ = R.build_hyper_rom(f, V, 1e-4, visc="es")
    assert rom.viscous_dissipation(uN) >= 0                                # eq. (43)


def test_theorem4_ideal_hyperreduction():
    """V^T (Q - Qbar) = 0 with all nodes iff M^{-1} Q^T V is in the test space."""
    f, V, _ = _setup()
    for enrich, zero in [("MinvQtV", True), ("QV", False), (None, False)]:
        rom, _ = R.build_hyper_rom(f, V, 1e-4, enrich=enrich, selection="all")
        res = np.linalg.norm(V.T @ (f.Q.toarray() - rom.Qbar)) / np.linalg.norm(V.T @ f.Q.toarray())
        assert (res < 1e-10) == zero, (enrich, res)


def test_caratheodory():
    A = np.linalg.qr(rng.standard_normal((300, 40)))[0]
    w = rng.random(300)
    I, wI = hr.caratheodory_prune(A, w)
    assert len(I) == 40 and np.all(wI > 0)
    assert np.linalg.norm(A[I].T @ wI - A.T @ w) < 1e-12 * np.linalg.norm(A.T @ w)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
