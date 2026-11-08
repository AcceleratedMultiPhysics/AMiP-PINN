"""Measured values of the invariants checked in tests/ (the tests only check thresholds).

Run from ES-ROM/ROM/:  python report/measure_invariants.py
Writes report/logs/measurements.json and figures for the report (report/figs/).
Uses the same small problems as the tests, plus the cached FOM data in outputs/
for the data figures (no new FOM run of the full problem).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from helpers import small_problem, small_run  # noqa: E402  (sets ESNN_DEVICE=cpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.physics import Burgers, Euler1D, logmean  # noqa: E402
from utils.fom import FOM1D, integrate  # noqa: E402
from utils.esrom import ESROM  # noqa: E402
from esnn_model import ESNNROM, build_net, net_rhs, rollout  # noqa: E402
from esnn_config import TRAIN  # noqa: E402

torch.manual_seed(0)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
m = {}
mx = lambda t: float(t.abs().max())  # noqa: E731

# ---------------------------------------------------------------- physics
phys_E = Euler1D()
rho, vel, p = 1 + torch.rand(1000), torch.randn(1000), 1 + torch.rand(1000)
uE = phys_E.prim_to_cons(rho, vel, p)
uE2 = phys_E.prim_to_cons(1 + torch.rand(1000), torch.randn(1000), 1 + torch.rand(1000))
uB, uB2 = torch.randn(1, 1000), torch.randn(1, 1000)
for name, phys, u, w in (("burgers", Burgers(), uB, uB2), ("euler", phys_E, uE, uE2)):
    ug = u.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(phys.entropy(ug).sum(), ug)
    shuffle = ((phys.entropy_vars(u) - phys.entropy_vars(w)) * phys.flux_ec(u, w)).sum(0) \
        - (phys.psi(u) - phys.psi(w))
    m[f"physics/{name}"] = dict(
        grad_entropy_minus_v=mx(g - phys.entropy_vars(u)),
        round_trip=mx(phys.cons_vars(phys.entropy_vars(u)) - u),
        consistency=mx(phys.flux_ec(u, u) - phys.flux(u)),
        symmetry=mx(phys.flux_ec(u, w) - phys.flux_ec(w, u)),
        shuffle=mx(shuffle),
        psi=mx((phys.entropy_vars(u) * phys.flux(u)).sum(0) - phys.entropy_flux(u) - phys.psi(u)))
w = uE + 1e-9 * torch.randn(uE.shape)
m["physics/euler"]["shuffle_nearly_equal"] = mx(
    ((phys_E.entropy_vars(uE) - phys_E.entropy_vars(w)) * phys_E.flux_ec(uE, w)).sum(0)
    - (phys_E.psi(uE) - phys_E.psi(w)))
a = 1 + torch.rand(1000)
b = a * (1 + torch.logspace(-12, -1, 1000))
m["physics/logmean_rel_err"] = float(((logmean(a, b) - (a - b) / -torch.log1p((b - a) / a)) / logmean(a, b)).abs().max())

# ---------------------------------------------------------------- FOM
fom, data = small_run("euler", K=200, eps=0.0, T=0.05, nsteps=10, save_every=10)
u = data["U"][-1]
conv = fom.conv(u)
m["fom/euler_K200"] = dict(
    sum_rhs=mx(FOM1D(fom.phys, 200, eps=1e-3).rhs(u).sum(-1)),
    entropy_residual=float(fom.entropy_residual(u)),
    entropy_residual_scale=float((fom.phys.entropy_vars(u).abs() * conv.abs()).sum()))
fv = FOM1D(fom.phys, 200, eps=1e-3)
m["fom/euler_K200"]["dissipation"] = float(fv.entropy_dissipation(u))
m["fom/euler_K200"]["rate_plus_dissipation"] = float(
    fv.dx * (fv.phys.entropy_vars(u) * fv.rhs(u)).sum() + fv.entropy_dissipation(u))
sys.path.insert(0, os.path.join(ROOT, "..", "FOM"))
from esrom.fom import FOM1D as NpFOM1D  # noqa: E402
from esrom.physics import Euler1D as NpEuler1D  # noqa: E402
ref = NpFOM1D(NpEuler1D(), 200, bc="periodic", eps=1e-3)
m["fom/euler_K200"]["vs_FOM_esrom_rhs"] = float(np.abs(fv.rhs(u).numpy() - ref.rhs(u.numpy())).max())
fom0, data0 = small_run("euler", K=100, eps=0.0, T=0.0, nsteps=1, save_every=1)
u0 = data0["U"][0]
drift = {}
for nsteps in (50, 100, 200, 400):
    _, U = integrate(fom0.rhs, u0, 0.2 / nsteps, nsteps, save_every=nsteps)
    drift[nsteps] = float((fom0.total_entropy(U[-1]) - fom0.total_entropy(u0)).abs())
m["fom/entropy_drift_eps0_T0.2"] = drift

# ---------------------------------------------------------------- basis and teacher
prob = small_problem("euler")
V, A = prob.V, prob.A
res = prob.teacher.entropy_residual(A).abs()
scale = (prob.teacher.v_N(A).abs() * prob.teacher.project(prob.fom.conv(prob.teacher.states(A))).abs()).sum(-1)
teach0 = ESROM(FOM1D(prob.phys, prob.fom.K, eps=0.0), V)
m["pod_teacher/euler_K100_r6"] = dict(
    VtV_minus_I=mx(V.T @ V - torch.eye(V.shape[1])),
    nonconstant_column_sums=mx(V[:, 1:].sum(0)),
    teacher_entropy_residual_rel=float((res / scale).max()),
    teacher_max_dSdt=float(prob.teacher.entropy_rate(A).max()),
    teacher_min_dissipation=float((-prob.teacher.entropy_rate(A)).min()),
    teacher_conserved_rates=mx(prob.teacher.rhs(A)[:, prob.conserved]),
    teacher_eps0_max_abs_dSdt=mx(teach0.entropy_rate(A)),
    projection_error_train_end=float(prob.projection_error()[int(prob.train_mask.sum()) - 1]),
    projection_error_final=float(prob.projection_error()[-1]))
probB = small_problem("burgers")
m["pod_teacher/burgers_teacher_minus_galerkin"] = mx(probB.teacher.rhs(probB.A) - probB.galerkin.rhs(probB.A))

# ---------------------------------------------------------------- network (random weights)
NET_SMALL = dict(width=32, depth=2, act="tanh", rank=None, l_init=1.0, n_mu=0)
std = A.std(0)
a = A[torch.randint(len(A), (64,))]
a = a + 0.05 * std * torch.randn(a.shape)
a = a[prob.phys.is_admissible(prob.reconstruct(a))]
for label, kw in (("full_L", NET_SMALL), ("rank3_L", {**NET_SMALL, "rank": 3})):
    net = build_net("structured", prob, kw)
    net.set_normalization(A)
    with torch.no_grad():
        vN = net.v_N(a)
        adot = net(a)
        vf = vN[:, net.free].unsqueeze(-1)
        J, R = net.operators(a)
        vJv = (vf.mT @ J @ vf).squeeze()
        vRv = (vf.mT @ R @ vf).squeeze()
        rate = prob.dx * (vN * adot).sum(-1)
    m[f"model/random_{label}"] = dict(
        states=len(a),
        max_rel_vJv=float((vJv.abs() / (vf.squeeze(-1).pow(2).sum(-1) * J.flatten(1).norm(dim=-1))).max()),
        min_eig_R_rel=float(torch.linalg.eigvalsh(R).min() / R.abs().max()),
        max_dSdt=float(rate.max()), min_vRv=float(vRv.min()),
        rate_plus_vRv_rel=float(((rate + vRv).abs() / vRv.abs().max()).max()),
        conserved_rates=mx(adot[:, prob.conserved]))

# rollout of a random structured net and a random unconstrained net (Burgers)
TRAIN_SMALL = {**TRAIN, "nepochs": 0, "n_pretrain": 0, "n_teacher": 20, "print_step": 1}
probB = small_problem("burgers", K=100, eps=1e-2)
roll = {}
for mt in ("structured", "unconstrained"):
    torch.manual_seed(4)
    tr = ESNNROM(probB, mt, None, NET_SMALL, TRAIN_SMALL)
    with torch.no_grad():
        t, Ar, _ = rollout(net_rhs(tr.dnn.eval()), probB.A[0], t_final=0.2, dt=1e-4, save_dt=1e-3)
        S = probB.teacher.total_entropy(Ar)
        tot = probB.fom.totals(probB.reconstruct(Ar))
    roll[mt] = dict(t=t, S=S.numpy(), max_step_increase=float((S[1:] - S[:-1]).max()),
                    S0=float(S[0]), S_end=float(S[-1]), totals_drift=mx(tot - tot[0]))
m["model/random_rollout_burgers"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("t", "S")}
                                     for k, v in roll.items()}

with open(os.path.join(HERE, "logs", "measurements.json"), "w") as f:
    json.dump(m, f, indent=2)
print(json.dumps(m, indent=2))

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for mt, st in (("structured", "C0-"), ("unconstrained", "C3--")):
    ax.plot(roll[mt]["t"], roll[mt]["S"] - roll[mt]["S"][0], st, label=f"random {mt} net")
ax.set_xlabel("t"); ax.set_ylabel(r"$\mathcal{S}(t)-\mathcal{S}(0)$")
ax.set_title("Untrained networks, Burgers rollout"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "random_rollout_entropy.png"), dpi=200); plt.close(fig)

# FOM data figures from the caches written by esnn_train.py (if present)
for case, var, title in (("burgers", 0, "Burgers, K=1000"), ("euler_quick", 0, "Euler (quick), K=500"),
                         ("euler", 0, "Euler, K=2500")):
    path = os.path.join(ROOT, "outputs", case, "fom.npz")
    if not os.path.exists(path):
        continue
    f = np.load(path)
    t, U, sig = f["t"], f["U"], f["sigma"]
    x = -1 + (np.arange(U.shape[-1]) + 0.5) * 2 / U.shape[-1]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for k in np.linspace(0, len(t) - 1, 6).astype(int):
        axs[0].plot(x, U[k, var], lw=1, label=f"t={t[k]:.2f}")
    axs[0].axhline(np.nan); axs[0].legend(fontsize=7); axs[0].set_xlabel("x")
    axs[0].set_title(f"{title}: FOM snapshots of {['u', 'rho'][U.shape[1] == 3]}")
    axs[1].plot(t, f["S"] - f["S"][0], "C2-")
    axs[1].axvline(float(f["T_TRAIN"]), color="k", lw=0.6)
    axs[1].set_xlabel("t"); axs[1].set_title(r"FOM entropy $\mathcal{S}(t)-\mathcal{S}(0)$")
    axs[2].semilogy(sig / sig[0], ".", ms=3)
    axs[2].axvline(int(f["R_MODES"]) - 1, color="k", lw=0.6)
    axs[2].set_xlabel("index"); axs[2].set_title(r"singular values $\sigma_j/\sigma_1$ (train window)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"fom_{case}.png"), dpi=200); plt.close(fig)
    print("figure", case)
