"""FOM invariants (plan Sec. 8). Run: python tests/test_fom.py  (or pytest tests/)"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from utils.fom import FOM1D, integrate  # noqa: E402
from utils.physics import Burgers, Euler1D  # noqa: E402

torch.set_default_dtype(torch.float64)


def euler_pulse(K=200, eps=0.0):
    phys = Euler1D()
    fom = FOM1D(phys, K, eps=eps)
    g = torch.exp(-100 * (fom.x - 0.5) ** 2)
    rho = 2 + 0.5 * g
    return fom, phys.prim_to_cons(rho, 0.1 * g, rho ** phys.gamma)


def burgers_wave(K=200, eps=0.0):
    fom = FOM1D(Burgers(), K, eps=eps)
    return fom, -torch.sin(np.pi * fom.x)[None, :]


def test_conservation_of_totals():
    """Rows of 2(Q o F)1 and K u telescope: sum over cells is zero."""
    for fom, u in (euler_pulse(eps=1e-3), burgers_wave(eps=1e-3)):
        assert fom.rhs(u).sum(-1).abs().max() < 1e-10


def test_entropy_conservative_convection():
    for fom, u in (euler_pulse(), burgers_wave()):
        res = fom.entropy_residual(u).item()
        scale = (fom.phys.entropy_vars(u).abs() * fom.conv(u).abs()).sum().item()
        assert abs(res) < 1e-13 * max(scale, 1.0), res


def test_viscous_dissipation_nonnegative():
    for fom, u in (euler_pulse(eps=1e-3), burgers_wave(eps=1e-3)):
        assert fom.entropy_dissipation(u).item() >= 0.0
        rate = fom.dx * (fom.phys.entropy_vars(u) * fom.rhs(u)).sum()
        assert torch.allclose(rate, -fom.entropy_dissipation(u), rtol=1e-10, atol=1e-11)


def test_batched_rhs_matches_loop():
    fom, u = euler_pulse(eps=1e-3)
    U = torch.stack([u, 1.01 * u, 0.99 * u])
    R = fom.rhs(U)
    for k in range(3):
        assert torch.allclose(R[k], fom.rhs(U[k]), rtol=1e-14, atol=1e-14)


def test_matches_numpy_fv_code():
    """Cross-check against the committed FV package FOM/esrom (skipped if absent)."""
    fv_root = os.path.join(HERE, "..", "..", "FOM")
    if not os.path.isdir(os.path.join(fv_root, "esrom")):
        print("skip: FOM/esrom not found")
        return
    sys.path.insert(0, fv_root)
    from esrom.fom import FOM1D as NpFOM1D
    from esrom.physics import Euler1D as NpEuler1D
    fom, u = euler_pulse(eps=2e-4)
    ref = NpFOM1D(NpEuler1D(), fom.K, bc="periodic", eps=2e-4)
    assert np.allclose(fom.rhs(u).numpy(), ref.rhs(u.numpy()), rtol=1e-12, atol=1e-12)


def test_entropy_drift_converges_with_dt():
    """With eps = 0 the semi-discrete entropy is conserved; RK4 drift must shrink fast with dt."""
    fom, u0 = euler_pulse(K=100)
    S0 = fom.total_entropy(u0)
    drift = []
    for nsteps in (100, 200):
        _, U = integrate(fom.rhs, u0, 0.2 / nsteps, nsteps, save_every=nsteps)
        drift.append((fom.total_entropy(U[-1]) - S0).abs().item())
    assert drift[1] < drift[0] / 8 or drift[1] < 1e-13, drift


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
