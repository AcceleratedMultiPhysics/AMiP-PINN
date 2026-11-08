"""Physics invariants (plan Sec. 8). Run: python tests/test_physics.py  (or pytest tests/)"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.physics import Burgers, Euler1D, logmean  # noqa: E402

torch.set_default_dtype(torch.float64)
gen = torch.Generator().manual_seed(0)


def random_euler_states(n):
    phys = Euler1D()
    rho = 1 + torch.rand(n, generator=gen)
    vel = torch.randn(n, generator=gen)
    p = 1 + torch.rand(n, generator=gen)
    return phys, phys.prim_to_cons(rho, vel, p)


def cases(n=100):
    _, uE = random_euler_states(n)
    _, uE2 = random_euler_states(n)
    return [(Burgers(), torch.randn(1, n, generator=gen), torch.randn(1, n, generator=gen)),
            (Euler1D(), uE, uE2)]


def test_entropy_vars_are_gradient_of_entropy():
    for phys, u, _ in cases():
        u = u.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(phys.entropy(u).sum(), u)
        assert torch.allclose(g, phys.entropy_vars(u), rtol=1e-12, atol=1e-12)


def test_round_trip():
    for phys, u, _ in cases():
        assert torch.allclose(phys.cons_vars(phys.entropy_vars(u)), u, rtol=1e-12, atol=1e-12)


def test_flux_properties():
    """Consistency, symmetry and Tadmor's shuffle condition."""
    for phys, uL, uR in cases():
        assert torch.allclose(phys.flux_ec(uL, uL), phys.flux(uL), rtol=1e-13, atol=1e-13)
        assert torch.allclose(phys.flux_ec(uL, uR), phys.flux_ec(uR, uL), rtol=1e-14, atol=1e-14)
        shuffle = ((phys.entropy_vars(uL) - phys.entropy_vars(uR)) * phys.flux_ec(uL, uR)).sum(0)
        assert (shuffle - (phys.psi(uL) - phys.psi(uR))).abs().max() < 1e-12


def test_flux_nearly_equal_states():
    """Shuffle condition where the log mean switches to its series branch."""
    phys, u = random_euler_states(50)
    w = u + 1e-9 * torch.randn(u.shape, generator=gen)
    shuffle = ((phys.entropy_vars(u) - phys.entropy_vars(w)) * phys.flux_ec(u, w)).sum(0)
    assert (shuffle - (phys.psi(u) - phys.psi(w))).abs().max() < 1e-14


def test_psi_is_potential():
    """psi = v^T f - F with F the entropy flux (checks the scaling of psi)."""
    for phys, u, _ in cases():
        psi = (phys.entropy_vars(u) * phys.flux(u)).sum(0) - phys.entropy_flux(u)
        assert torch.allclose(psi, phys.psi(u), rtol=1e-12, atol=1e-12)


def test_logmean():
    a = 1 + torch.rand(200, generator=gen)
    b = a * (1 + torch.cat([1e-1 * torch.rand(100, generator=gen), 1e-6 * torch.rand(100, generator=gen)]))
    # log a - log b = -log1p((b - a)/a): no cancellation when a ~ b, unlike the naive
    # difference of logs (which is only good to ~1e-9 for |b/a - 1| ~ 1e-6)
    exact = (a - b) / -torch.log1p((b - a) / a)
    assert torch.allclose(logmean(a, b), exact, rtol=1e-13)
    assert torch.allclose(logmean(a, a), a, rtol=1e-15)


def test_logmean_gradient_is_finite():
    """The double-where keeps gradients finite at equal states (plan Sec. 7, NaN trap)."""
    a = (1 + torch.rand(10, generator=gen)).requires_grad_(True)
    b = a.detach().clone().requires_grad_(True)
    (ga, gb) = torch.autograd.grad(logmean(a, b).sum(), (a, b))
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    assert torch.allclose(ga, torch.full_like(ga, 0.5)) and torch.allclose(gb, torch.full_like(gb, 0.5))


def test_euler_admissibility():
    phys, u = random_euler_states(5)
    assert phys.is_admissible(u)
    bad = u.clone()
    bad[2, 3] = 0.5 * bad[1, 3] ** 2 / bad[0, 3] - 1.0     # negative pressure
    assert not phys.is_admissible(bad)
    assert phys.is_admissible(torch.stack([u, bad])).tolist() == [True, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
