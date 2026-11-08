"""Basis and ES-ROM teacher invariants (plan Sec. 8). Run: python tests/test_pod_esrom.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from helpers import small_problem, small_run  # noqa: E402
from utils.esrom import ESROM  # noqa: E402
from utils.pod import build_basis, conserved_indices, project, reconstruct  # noqa: E402


def test_basis_orthonormal_with_constant_mode():
    fom, data = small_run("euler")
    V, sigma = build_basis(data["U"], 8, fom.phys)
    K = fom.K
    assert torch.allclose(V.T @ V, torch.eye(8), atol=1e-12)
    assert torch.allclose(V[:, 0], torch.full((K,), K ** -0.5), atol=1e-15)
    assert V[:, 1:].sum(0).abs().max() < 1e-12
    assert (sigma[1:] <= sigma[:-1] + 1e-12).all()


def test_project_reconstruct():
    prob = small_problem("euler")
    a = torch.randn(4, prob.n)
    assert torch.allclose(project(reconstruct(a, prob.V, prob.nvar), prob.V), a, atol=1e-12)


def test_totals_live_in_constant_coefficients():
    prob = small_problem("euler")
    a = prob.A[:5]
    totals = prob.fom.totals(prob.reconstruct(a))
    c = conserved_indices(prob.nvar, prob.r)
    assert torch.allclose(totals, prob.dx * prob.fom.K ** 0.5 * a[:, c], rtol=1e-12)


def test_teacher_entropy_conservative_convection():
    """|v_N^T V^T 2(Q o F)1| ~ round-off at every snapshot (Chan Fig. 4)."""
    prob = small_problem("euler")
    a = prob.A
    res = prob.teacher.entropy_residual(a).abs()
    scale = (prob.teacher.v_N(a).abs() * prob.teacher.project(
        prob.fom.conv(prob.teacher.states(a))).abs()).sum(-1)
    assert (res <= 1e-12 * scale.clamp_min(1.0)).all(), res.max()


def test_teacher_entropy_stable_and_conservative():
    prob = small_problem("euler")
    a = prob.A
    assert (prob.teacher.entropy_rate(a) <= 1e-12).all()
    assert prob.teacher.rhs(a)[:, prob.conserved].abs().max() < 1e-10


def test_teacher_entropy_conservative_without_viscosity():
    prob = small_problem("euler")
    teacher = ESROM(prob.fom.__class__(prob.phys, prob.fom.K, eps=0.0), prob.V)
    rate = teacher.entropy_rate(prob.A).abs()
    scale = prob.dx * (teacher.v_N(prob.A).abs() * teacher.rhs(prob.A).abs()).sum(-1)
    assert (rate <= 1e-12 * scale.clamp_min(1.0)).all(), rate.max()


def test_burgers_entropy_projection_is_trivial():
    """v = u for Burgers, so the ES-ROM equals POD-Galerkin."""
    prob = small_problem("burgers")
    assert torch.allclose(prob.teacher.rhs(prob.A), prob.galerkin.rhs(prob.A), rtol=1e-11, atol=1e-11)


def test_projected_data_is_consistent():
    prob = small_problem("euler")
    assert torch.allclose(prob.Adot, project(prob.fom.rhs(prob.U), prob.V), rtol=1e-12, atol=1e-12)
    perr = prob.projection_error()
    assert perr.shape == (len(prob.t),) and (perr < 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
