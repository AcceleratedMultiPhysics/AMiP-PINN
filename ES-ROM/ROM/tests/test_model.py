"""Structure of the neural ROM with random (untrained) weights, plus a trainer smoke run.

Plan Sec. 8: v_N^T J v_N ~ round-off, v_N^T (J - R) v_N <= 0, conserved rates exactly
zero, S(t) non-increasing along rollouts. Run: python tests/test_model.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from helpers import small_problem  # noqa: E402
from esnn_model import (ESNNROM, StructuredNet, UnconstrainedNet, build_net,  # noqa: E402
                        history, net_rhs, rollout, rollout_memory)
from esnn_config import TRAIN, CAUSAL  # noqa: E402

NET_SMALL = dict(width=32, depth=2, act="tanh", rank=None, l_init=1.0, n_mu=0)
TRAIN_SMALL = {**TRAIN, "nepochs": 4, "n_pretrain": 2, "n_teacher": 20, "print_step": 1,
               "compile": False}
CAUSAL_SMALL = {**CAUSAL, "nepochs": 3, "print_step": 1, "substeps": 2}
MEM_SMALL = {**NET_SMALL, "n_delay": 3, "delay_every": 2}


def random_states(prob, n=16, noise=0.05):
    """Admissible states near the snapshots."""
    a = prob.A[torch.randint(len(prob.A), (n,))]
    std = prob.A.std(0)
    a = a + noise * std * torch.randn(a.shape)
    return a[prob.phys.is_admissible(prob.reconstruct(a))]


def check_structure(net, prob, a):
    vN = net.v_N(a)
    adot = net(a)
    vf = vN[:, net.free].unsqueeze(-1)
    J, R = net.operators(a)

    # skew J: exact entropy conservation of the J part, up to round-off
    assert torch.allclose(J, -J.mT, atol=0)
    vJv = (vf.mT @ J @ vf).squeeze()
    assert (vJv.abs() <= 1e-13 * vf.squeeze(-1).pow(2).sum(-1) * J.flatten(1).norm(dim=-1)).all()

    # PSD R, and dS/dt = dx v_N^T da/dt = -v^T R v <= 0
    assert torch.linalg.eigvalsh(R).min() >= -1e-12 * R.abs().max()
    rate = prob.dx * (vN * adot).sum(-1)
    vRv = (vf.mT @ R @ vf).squeeze()
    assert (rate <= 1e-13 * vRv.abs().max().clamp_min(1e-300)).all()
    assert torch.allclose(rate, -vRv, rtol=1e-10, atol=1e-13 * vRv.abs().max().item())

    # conserved coefficients exactly frozen
    assert (adot[:, prob.conserved] == 0).all()


def test_structured_random_weights_euler():
    torch.manual_seed(0)
    prob = small_problem("euler")
    net = build_net("structured", prob, NET_SMALL)
    net.set_normalization(prob.A)
    check_structure(net, prob, random_states(prob))


def test_structured_low_rank():
    torch.manual_seed(1)
    prob = small_problem("euler")
    net = build_net("structured", prob, {**NET_SMALL, "rank": 3})
    assert isinstance(net, StructuredNet) and net.nL == net.nf * 3
    check_structure(net, prob, random_states(prob))


def test_output_sizes():
    prob = small_problem("euler", r=6)
    net = build_net("structured", prob, NET_SMALL)
    nf = prob.n - prob.nvar
    assert net.nf == nf and net.nJ == nf * (nf - 1) // 2 and net.nL == nf * (nf + 1) // 2
    unc = build_net("unconstrained", prob, NET_SMALL)
    assert isinstance(unc, UnconstrainedNet) and unc(prob.A[:3]).shape == (3, prob.n)


def test_precomputed_vN_matches():
    torch.manual_seed(2)
    prob = small_problem("euler")
    net = build_net("structured", prob, NET_SMALL)
    a = random_states(prob)
    ref = net(a)
    assert (ref - net(a, vN=prob.teacher.v_N(a))).abs().max() <= 1e-13 * ref.abs().max()


def test_batch_consistency():
    torch.manual_seed(3)
    prob = small_problem("euler")
    net = build_net("structured", prob, NET_SMALL)
    a = random_states(prob, n=4)
    rhs = net_rhs(net)
    ref = net(a)
    for k in range(len(a)):
        assert (ref[k] - rhs(a[k])).abs().max() <= 1e-12 * ref[k].abs().max()


def test_rollout_entropy_nonincreasing_burgers():
    """Fully discrete: S may only rise by RK time error, which is tiny at small dt."""
    torch.manual_seed(4)
    prob = small_problem("burgers", K=100, eps=1e-2)
    trainer = ESNNROM(prob, "structured", None, NET_SMALL, TRAIN_SMALL)   # sets data scales
    net = trainer.dnn.eval()
    with torch.no_grad():
        t, A, _ = rollout(net_rhs(net), prob.A[0], t_final=0.2, dt=1e-4, save_dt=1e-3)
        S = prob.teacher.total_entropy(A)
        tot = prob.fom.totals(prob.reconstruct(A))
    assert torch.isfinite(A).all() and len(t) == 201
    assert (S[1:] - S[:-1]).max() <= 1e-12 * S[0].abs()
    assert (tot - tot[0]).abs().max() < 1e-12


def test_trainer_smoke():
    """A few epochs for both networks: finite losses, structure intact after updates."""
    for model_type in ("structured", "unconstrained"):
        torch.manual_seed(5)
        prob = small_problem("euler")
        trainer = ESNNROM(prob, model_type, None, NET_SMALL, TRAIN_SMALL)
        trainer.train()
        assert len(trainer.losses) == TRAIN_SMALL["nepochs"]
        assert all(torch.isfinite(torch.tensor(trainer.losses)))
        assert len(trainer.a_teach) > len(trainer.a_data)
        if model_type == "structured":
            check_structure(trainer.dnn, prob, random_states(prob))


def closure_net(prob, seed, z0=1.0, model_type="closure"):
    """Random closure network with nu of the order of eps (not the tiny initial value)."""
    torch.manual_seed(seed)
    net = build_net(model_type, prob, NET_SMALL)
    net.set_normalization(prob.A)
    with torch.no_grad():
        net.mlp[-1].weight.mul_(100.0)           # undo the small init: a varying nu(x; a)
        net.mlp[-1].bias[0] = z0
    return net


def test_closure_entropy_and_conservation():
    """Eddy-viscosity closure: nu >= 0, closure dS/dt <= 0, totals conserved (Euler)."""
    prob = small_problem("euler")
    net = closure_net(prob, 6)
    a = random_states(prob)
    with torch.no_grad():
        base, gu = net.precompute(a)
        nu = net.viscosity(a)
        C = net.closure(nu, gu)
        vN = net.v_N(a)
        rate_c = prob.dx * (vN * C).sum(-1)
        rate = prob.dx * (vN * net(a)).sum(-1)
    assert (nu >= 0).all() and nu.std() > 0
    scale = prob.dx * (vN.abs() * C.abs()).sum(-1)
    assert (rate_c <= 1e-13 * scale).all() and (rate_c < 0).all()
    assert (rate <= 1e-12 * scale).all()
    assert C[:, prob.conserved].abs().max() <= 1e-12 * C.abs().max()


def test_closure_reduces_to_esrom_and_keeps_uniform_states():
    prob = small_problem("euler")
    net = closure_net(prob, 7)
    a = random_states(prob)
    with torch.no_grad():
        base, gu = net.precompute(a)
        assert torch.equal(net.closure(torch.zeros(len(a), prob.fom.K), gu),
                           torch.zeros_like(base))                  # nu = 0: exactly the ES-ROM
        assert torch.allclose(net(a), net(a, cache=(base, gu)), rtol=1e-14, atol=1e-14)
        K = prob.fom.K
        u_c = prob.phys.prim_to_cons(torch.full((K,), 1.7), torch.full((K,), 0.3),
                                     torch.full((K,), 2.1))
        a_c = prob.teacher.project(u_c)[None]
        _, gu_c = net.precompute(a_c)
        C_c = net.closure(net.viscosity(a_c), gu_c)
    assert C_c.abs().max() < 1e-12 * net.viscosity(a_c).max() / prob.dx


def test_closure_rollout_entropy_nonincreasing_burgers():
    prob = small_problem("burgers", K=100, eps=1e-2)
    net = closure_net(prob, 8).eval()
    with torch.no_grad():
        t, A, _ = rollout(net_rhs(net), prob.A[0], t_final=0.2, dt=1e-4, save_dt=1e-3)
        S = prob.teacher.total_entropy(A)
        tot = prob.fom.totals(prob.reconstruct(A))
    assert torch.isfinite(A).all() and len(t) == 201
    assert (S[1:] - S[:-1]).max() <= 1e-12 * S[0].abs()
    assert (tot - tot[0]).abs().max() < 1e-12


def test_closure_trainer_smoke():
    """A few epochs: finite losses, no teacher samples, nu >= 0 after updates."""
    torch.manual_seed(9)
    prob = small_problem("burgers")
    trainer = ESNNROM(prob, "closure", None, NET_SMALL, TRAIN_SMALL)
    trainer.train()
    assert len(trainer.losses) == TRAIN_SMALL["nepochs"]
    assert all(torch.isfinite(torch.tensor(trainer.losses)))
    assert len(trainer.a_teach) == 0
    with torch.no_grad():
        assert (trainer.dnn.viscosity(prob.A) >= 0).all()


def test_backscatter_entropy_and_conservation():
    """nu >= -eps: the closure alone may produce entropy, ES-ROM + closure may not (Euler)."""
    prob = small_problem("euler")
    eps = prob.fom.eps
    net = closure_net(prob, 12, z0=0.5, model_type="backscatter")    # nu of both signs
    a = random_states(prob)
    with torch.no_grad():
        base, gu = net.precompute(a)
        nu = net.viscosity(a)
        vN = net.v_N(a)
        C = net.closure(nu, gu)
        rate_c = prob.dx * (vN * C).sum(-1)
        rate = prob.dx * (vN * net(a)).sum(-1)
        # the closure has the discrete form of the ES-ROM viscosity: nu = -eps removes it exactly
        ut = prob.teacher.states(a)
        conv = prob.teacher.project(-prob.fom.conv(ut) / prob.dx)
        no_visc = base + net.closure(torch.full_like(nu, -eps), gu)
    assert (nu >= -eps).all() and (nu < 0).any() and (nu > 0).any()
    assert (rate_c > 0).any(), "with nu < 0 the closure should remove some ES-ROM dissipation"
    scale = prob.dx * (vN.abs() * (base.abs() + C.abs())).sum(-1)
    assert (rate <= 1e-12 * scale).all()
    assert torch.allclose(no_visc, conv, rtol=1e-12, atol=1e-12 * conv.abs().max().item())
    assert C[:, prob.conserved].abs().max() <= 1e-12 * C.abs().max()


def test_backscatter_starts_as_esrom():
    """Untrained backscatter net: nu = 0 up to the small last-layer init, rhs = ES-ROM."""
    prob = small_problem("burgers", K=100, eps=1e-2)
    net = build_net("backscatter", prob, NET_SMALL)
    net.set_normalization(prob.A)
    with torch.no_grad():
        nu = net.viscosity(prob.A)
        base, _ = net.precompute(prob.A)
        out = net(prob.A)
    assert nu.abs().max() < 0.05 * prob.fom.eps
    assert (out - base).abs().max() < 0.05 * base.abs().max()
    raised, eps = False, prob.fom.eps
    prob.fom.eps = 0.0
    try:
        build_net("backscatter", prob, NET_SMALL)
    except ValueError:
        raised = True
    prob.fom.eps = eps
    assert raised, "backscatter must refuse eps = 0 (no floor to borrow from)"


def test_backscatter_rollout_entropy_nonincreasing_burgers():
    prob = small_problem("burgers", K=100, eps=1e-2)
    net = closure_net(prob, 13, z0=-1.0, model_type="backscatter").eval()
    with torch.no_grad():
        assert (net.viscosity(prob.A) < 0).any()
        t, A, _ = rollout(net_rhs(net), prob.A[0], t_final=0.2, dt=1e-4, save_dt=1e-3)
        S = prob.teacher.total_entropy(A)
        tot = prob.fom.totals(prob.reconstruct(A))
    assert torch.isfinite(A).all() and len(t) == 201
    assert (S[1:] - S[:-1]).max() <= 1e-12 * S[0].abs()
    assert (tot - tot[0]).abs().max() < 1e-12


def test_backscatter_trainer_smoke():
    """A few epochs: finite losses, nu >= -eps after updates."""
    torch.manual_seed(14)
    prob = small_problem("burgers")
    trainer = ESNNROM(prob, "backscatter", None, NET_SMALL, TRAIN_SMALL)
    trainer.train()
    assert len(trainer.losses) == TRAIN_SMALL["nepochs"]
    assert all(torch.isfinite(torch.tensor(trainer.losses)))
    with torch.no_grad():
        assert (trainer.dnn.viscosity(prob.A) >= -prob.fom.eps).all()


def memory_net(prob, seed, n_delay=3, delay_every=2):
    """Random memory closure with nu of the order of eps."""
    torch.manual_seed(seed)
    net = build_net("memory", prob, {**NET_SMALL, "n_delay": n_delay, "delay_every": delay_every})
    net.set_normalization(prob.A)
    with torch.no_grad():
        net.mlp[-1].weight.mul_(100.0)
        net.mlp[-1].bias[0] = 1.0
    return net


def test_history_indices():
    """Past states A[k - j lag], initial state repeated before t = 0."""
    A = torch.arange(10.0)[:, None].repeat(1, 2)
    h = history(A, torch.tensor([0, 3, 9]), n_delay=3, lag=2)
    assert h.shape == (3, 3, 2)
    assert h[:, :, 0].tolist() == [[0, 0, 0], [1, 0, 0], [7, 5, 3]]


def test_memory_entropy_and_conservation():
    """Any past states: nu >= 0, closure dS/dt <= 0, totals conserved (Euler)."""
    prob = small_problem("euler")
    net = memory_net(prob, 10)
    a = random_states(prob)
    hist = random_states(prob, n=len(a) * net.n_delay)[: len(a) * net.n_delay]
    hist = hist.reshape(-1, net.n_delay, prob.n)[: len(a)]
    a = a[: len(hist)]
    with torch.no_grad():
        _, gu = net.precompute(a)
        nu = net.viscosity(a, hist=hist)
        C = net.closure(nu, gu)
        vN = net.v_N(a)
        rate_c = prob.dx * (vN * C).sum(-1)
    assert (nu >= 0).all() and nu.std() > 0
    assert (rate_c <= 1e-13 * prob.dx * (vN.abs() * C.abs()).sum(-1)).all()
    assert C[:, prob.conserved].abs().max() <= 1e-12 * C.abs().max()
    raised = False
    try:
        net(a)
    except ValueError:
        raised = True
    assert raised, "memory closure must refuse to run without past states"


def test_memory_rollout_entropy_nonincreasing_burgers():
    prob = small_problem("burgers", K=100, eps=1e-2)
    net = memory_net(prob, 11).eval()
    t, A, _ = rollout_memory(net, prob.A[0], t_final=0.2, dt=1e-4, save_dt=1e-3)
    with torch.no_grad():
        S = prob.teacher.total_entropy(A)
        tot = prob.fom.totals(prob.reconstruct(A))
    assert torch.isfinite(A).all() and len(t) == 201
    assert (S[1:] - S[:-1]).max() <= 1e-12 * S[0].abs()
    assert (tot - tot[0]).abs().max() < 1e-12


def test_memory_trainer_smoke():
    """A few epochs: finite losses, past states taken from the FOM data."""
    torch.manual_seed(12)
    prob = small_problem("burgers")
    trainer = ESNNROM(prob, "memory", None, {**NET_SMALL, "n_delay": 3, "delay_every": 2}, TRAIN_SMALL)
    trainer.train()
    assert all(torch.isfinite(torch.tensor(trainer.losses)))
    hist = trainer.kw_data["hist"]
    assert hist.shape == (len(trainer.a_data), 3, prob.n)
    assert torch.equal(hist[10, 0], prob.A[8]) and torch.equal(hist[1, 2], prob.A[0])



# ---------------------------------------------------------------- causal rollout training
def causal_trainer(prob, seed, causal_kw=CAUSAL_SMALL):
    """Causal trainer of a memory closure with a varying nu (weights as in memory_net)."""
    torch.manual_seed(seed)
    trainer = ESNNROM(prob, "memory", None, MEM_SMALL, TRAIN_SMALL, mode="causal", causal_kw=causal_kw)
    with torch.no_grad():
        trainer.dnn.mlp[-1].weight.mul_(100.0)
        trainer.dnn.mlp[-1].bias[0] = 1.0
    return trainer


def test_closure_differentiable_in_state():
    """d(da/dt)/da through the ES-ROM part and the closure: autograd = finite differences (Euler)."""
    prob = small_problem("euler")
    net = memory_net(prob, 15)
    a = random_states(prob, n=4)
    hist = prob.A[:len(a) * 3].reshape(len(a), 3, -1)
    da = 1e-3 * prob.A.std(0) * torch.randn(a.shape)
    da[:, prob.conserved] = 0.0
    a_req = a.clone().requires_grad_(True)
    out = net(a_req, hist=hist)
    jvp = torch.autograd.grad(out, a_req, grad_outputs=torch.ones_like(out))[0]
    h = 1e-4                                         # smaller h: round-off cancellation dominates
    with torch.no_grad():
        fd = ((net(a + h * da, hist=hist) - net(a - h * da, hist=hist)).sum(-1) / (2 * h))
    assert torch.allclose((jvp * da).sum(-1), fd, rtol=1e-6, atol=1e-8 * fd.abs().max().item())


def test_causal_rollout_matches_rollout_memory():
    """Training rollout (own past states, RK4 substeps) = the post-processing rollout."""
    prob = small_problem("burgers")
    trainer = causal_trainer(prob, 16, {**CAUSAL_SMALL, "w_cut": 0.0})
    with torch.no_grad():
        out, L, W = trainer.causal_rollout()
        t, A, _ = rollout_memory(trainer.dnn, prob.A[0], t_final=float(prob.t[len(out)]),
                                 dt=trainer.dt_roll, save_dt=trainer.dt_snap, snap_dt=trainer.dt_snap)
    assert len(out) == len(trainer.a_data) - 1 == len(A) - 1
    assert torch.allclose(torch.stack(out), A[1:], rtol=1e-12, atol=1e-13)


def test_causal_weights():
    """w_1 = 1, w_i = exp(-tol sum_{k<i} L_k) non-increasing; a large tol cuts the rollout."""
    prob = small_problem("burgers")
    trainer = causal_trainer(prob, 17, {**CAUSAL_SMALL, "w_cut": 0.0})
    with torch.no_grad():
        _, L, W = trainer.causal_rollout()
    L = torch.stack(L)
    assert W[0] == 1.0 and (W[1:] <= W[:-1]).all()
    cum = torch.cat([torch.zeros(1), torch.cumsum(L, 0)[:-1]])
    assert torch.allclose(W, torch.exp(-trainer.ccfg["tol"] * cum), rtol=1e-12)
    tol = 100.0 / L.sum().item()                     # w near the end ~ e^-100 << w_cut = 1e-4
    trainer.ccfg.update(tol=tol, w_cut=1e-4)
    with torch.no_grad():
        _, L2, W2 = trainer.causal_rollout()
    assert 0 < len(L2) < len(L) and (W2 >= 1e-4).all()


def test_causal_trainer_smoke():
    """A few causal epochs: finite loss, gradients reach the network, nu >= 0 kept."""
    prob = small_problem("burgers")
    trainer = causal_trainer(prob, 18)
    w0 = [p.detach().clone() for p in trainer.dnn.parameters()]
    trainer.train()
    assert len(trainer.losses) == CAUSAL_SMALL["nepochs"]
    assert all(torch.isfinite(torch.tensor(trainer.losses)))
    assert any((p - q).abs().max() > 0 for p, q in zip(trainer.dnn.parameters(), w0))
    k = torch.arange(len(prob.A))
    with torch.no_grad():
        assert (trainer.dnn.viscosity(prob.A, hist=history(prob.A, k, 3, 2)) >= 0).all()
    raised = False
    try:
        ESNNROM(prob, "structured", None, NET_SMALL, TRAIN_SMALL, mode="causal")
    except ValueError:
        raised = True
    assert raised, "causal training is only for the closure models"

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
