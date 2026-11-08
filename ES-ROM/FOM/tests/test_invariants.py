"""Invariant checks from the notes (Sec. 10). Run: python tests/test_invariants.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esrom import fom as fom_mod  # noqa: E402
from esrom import rom as rom_mod  # noqa: E402
from esrom.physics import Burgers, Euler1D  # noqa: E402

rng = np.random.default_rng(0)


def random_euler_states(n):
    phys = Euler1D()
    rho = 1 + rng.random(n)
    vel = rng.standard_normal(n)
    p = 1 + rng.random(n)
    return phys, phys.prim_to_cons(rho, vel, p)


def test_flux_properties():
    for phys, uL, uR in [
        (Burgers(), rng.standard_normal((1, 100)), rng.standard_normal((1, 100))),
        (Euler1D(), random_euler_states(100)[1], random_euler_states(100)[1]),
    ]:
        assert np.allclose(phys.flux_ec(uL, uL), phys.flux(uL), rtol=1e-13, atol=1e-13)
        assert np.allclose(phys.flux_ec(uL, uR), phys.flux_ec(uR, uL), rtol=1e-14, atol=1e-14)
        shuffle = np.sum((phys.entropy_vars(uL) - phys.entropy_vars(uR)) * phys.flux_ec(uL, uR), 0)
        assert np.max(np.abs(shuffle - (phys.psi(uL) - phys.psi(uR)))) < 1e-12


def test_flux_nearly_equal_states():
    phys, u = random_euler_states(50)
    du = 1e-9 * rng.standard_normal(u.shape)
    shuffle = np.sum((phys.entropy_vars(u) - phys.entropy_vars(u + du)) * phys.flux_ec(u, u + du), 0)
    assert np.max(np.abs(shuffle - (phys.psi(u) - phys.psi(u + du)))) < 1e-14


def test_entropy_variables_and_jacobian():
    phys, u = random_euler_states(20)
    assert np.allclose(phys.cons_vars(phys.entropy_vars(u)), u, rtol=1e-12)
    h = 1e-6
    for k in range(3):
        e = np.zeros((3, 1)); e[k] = h
        dS = (phys.entropy(u + e) - phys.entropy(u - e)) / (2 * h)
        assert np.allclose(dS, phys.entropy_vars(u)[k], rtol=1e-7, atol=1e-8)
        v = phys.entropy_vars(u)
        du = (phys.cons_vars(v + e) - phys.cons_vars(v - e)) / (2 * h)
        assert np.allclose(du, phys.dudv(u)[:, k], rtol=1e-6, atol=1e-7)


def test_fom_operators():
    for bc in ["periodic", "wall"]:
        f = fom_mod.FOM1D(Euler1D(), 64, bc=bc)
        Q = f.Q.toarray()
        assert np.allclose(Q @ np.ones(64), 0)
        B = Q + Q.T
        if bc == "periodic":
            assert np.allclose(B, 0)
        else:
            assert np.allclose(B, np.diag(np.r_[-1, np.zeros(62), 1]))


def test_fom_entropy_conservation():
    for bc in ["periodic", "wall"]:
        phys = Euler1D()
        f = fom_mod.FOM1D(phys, 64, bc=bc)
        _, u = random_euler_states(64)
        assert abs(f.entropy_residual(u)) < 1e-12
    f = fom_mod.FOM1D(Burgers(), 64)
    assert abs(f.entropy_residual(rng.standard_normal((1, 64)))) < 1e-12


def _euler_rom_setup(bc):
    phys = Euler1D()
    f = fom_mod.FOM1D(phys, 200, bc=bc, eps=1e-3)
    x = f.x
    snaps = []
    for k in range(10):
        rho = 2 + 0.3 * np.exp(-50 * (x - 0.1 * k + 0.3) ** 2)
        vel = 0.1 * np.sin(np.pi * (k + 1) * x / 3)
        snaps.append(phys.prim_to_cons(rho, vel, rho ** 1.4))
    snaps = np.array(snaps)
    U, s = rom_mod.snapshot_svd(snaps, phys)
    V = rom_mod.pod_basis(U, 12)
    return f, V, snaps, s


def test_rom_entropy_conservation():
    for bc in ["periodic", "wall"]:
        f, V, snaps, s = _euler_rom_setup(bc)
        uN = rom_mod.GalerkinROM(f, V).project(snaps[3])
        g = rom_mod.GalerkinROM(f, V)
        assert abs(g.entropy_residual(uN)) < 1e-11
        tol = max(rom_mod.hr.energy_tol(s, 12), 1e-6)
        hrom, info = rom_mod.build_hyper_rom(f, V, tol, visc="22")
        Qt = hrom.Qt
        if bc == "periodic":
            assert np.allclose(Qt, -Qt.T, atol=1e-12)
        else:  # generalized SBP: Qt + Qt^T = E^T B E
            Vt = rom_mod.test_basis(V, f.Q, f.dx)
            E = Vt[[0, -1]] @ hrom.Pt
            assert np.allclose(Qt + Qt.T, E.T @ np.diag([-1.0, 1.0]) @ E, atol=1e-12)
            Qh = hrom.Qh
            assert np.allclose(Qh + Qh.T, np.diag(np.r_[np.zeros(len(hrom.I)), -1, 1]), atol=1e-12)
        assert np.allclose(Qt @ np.ones(len(hrom.I)), 0, atol=1e-11)
        nh = hrom.Qh.shape[0]
        assert np.allclose(hrom.Qh @ np.ones(nh), 0, atol=1e-11)
        assert np.all(hrom.w > 0)
        VtI = rom_mod.test_basis(V, f.Q, f.dx)[hrom.I]
        assert np.allclose(hrom.Pt @ VtI, np.eye(VtI.shape[1]), atol=1e-8)
        assert abs(hrom.entropy_residual(uN)) < 1e-11
        assert hrom.entropy_dissipation(uN) >= 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
