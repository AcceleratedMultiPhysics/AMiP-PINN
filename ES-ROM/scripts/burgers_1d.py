"""Step 1: periodic Burgers, u0 = -sin(pi x)  (Chan 2020, Sec. 4.1.1 / Fig. 1).

Shows that the hyper-reduced ROM is badly wrong with test basis [1, V] and is
fixed by [1, V, QV]; checks entropy conservation (eps = 0) and dissipation.
Burgers has v = u, so the entropy projection is trivial here.

Run from ES-ROM/:  python scripts/burgers_1d.py
"""
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # small dense ops: threading only hurts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esrom import hyperreduction as hr  # noqa: E402
from esrom import rom as R  # noqa: E402
from esrom.fom import FOM1D, integrate  # noqa: E402
from esrom.physics import Burgers  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "burgers")
os.makedirs(OUT, exist_ok=True)

K, EPS, T, NSTEPS, SAVE = 1000, 2e-3, 1.0, 2000, 5
phys = Burgers()
fom = FOM1D(phys, K, bc="periodic", eps=EPS)
x, dx = fom.x, fom.dx
dt = T / NSTEPS
u0 = -np.sin(np.pi * x)[None, :]

# ---------------------------------------------------------------- FOM
t, U_fom = integrate(fom.rhs, u0, dt, NSTEPS, SAVE)
print(f"FOM: K={K}, dt={dt:g}, {len(t)} snapshots, "
      f"S(0)={fom.total_entropy(U_fom[0]):.6f}, S(T)={fom.total_entropy(U_fom[-1]):.6f}")

Usvd, sigma = R.snapshot_svd(U_fom, phys, entropy_enrich=False)

# ---------------------------------------------------------------- Fig. 1
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k in np.linspace(0, len(t) - 1, 6).astype(int):
    ax[0].plot(x, U_fom[k, 0], label=f"t={t[k]:.2f}")
ax[0].set_title("FOM snapshots"); ax[0].legend(fontsize=7)
for j in range(3):
    ax[1].plot(x, Usvd[:, j], label=f"mode {j+1}")
    ax[2].plot(x, (fom.Q @ Usvd[:, j]) / dx, label=f"Q V(:,{j+1}) / dx")
ax[1].set_title("POD modes (antisymmetric)"); ax[1].legend(fontsize=7)
ax[2].set_title("mode derivatives (nearly symmetric)"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_modes.png"), dpi=150)

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.semilogy(sigma / sigma[0], ".")
ax.set_xlabel("index"); ax.set_ylabel(r"$\sigma_j/\sigma_1$"); ax.set_title("Burgers snapshot singular values")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "singular_values.png"), dpi=150)


def rel_err(rom, traj):
    U = np.array([rom.reconstruct(uN) for uN in traj])
    return np.linalg.norm((U - U_fom).reshape(len(t), -1), axis=1) / np.linalg.norm(
        U_fom.reshape(len(t), -1), axis=1)


def run(rom):
    uN0 = R.GalerkinROM(fom, rom.V).project(u0)
    return integrate(rom.rhs, uN0, dt, NSTEPS, SAVE)[1]


# ---------------------------------------------------------------- ROMs
for N in [4, 8, 12]:
    V = R.pod_basis(Usvd, N)
    tol = hr.energy_tol(sigma, N)
    VtQV_small = np.linalg.norm(V.T @ (fom.Q @ V)) / np.linalg.norm(fom.Q @ V)

    gal = R.GalerkinROM(fom, V)
    roms = {"HR, Vt=[1,V]": R.build_hyper_rom(fom, V, tol, enrich=False),
            "HR, Vt=[1,V,QV], target VoV (paper)": R.build_hyper_rom(fom, V, tol),
            "HR, Vt=[1,V,QV], target VtoVt": R.build_hyper_rom(fom, V, tol, target="test_mass")}
    trajs = {"Galerkin (Thm 2, no HR)": run(gal)}
    trajs.update({k: run(r) for k, (r, _) in roms.items()})
    errs = {k: rel_err(gal, v) for k, v in trajs.items()}
    print(f"\nN={N}: tol={tol:.2e}, ||V^T Q V||/||QV|| = {VtQV_small:.2e}")
    for k, (_, info) in roms.items():
        print(f"  {k:38s}: {info['n_points']} pts ({info['n_stabilizing']} stab), "
              f"{info['n_viscous']} visc pts, dim Vt={info['n_test']}, cond(Mt)={info['cond_test_mass']:.1e}")
    for k, e in errs.items():
        print(f"  {k:38s} rel. L2 error at T: {e[-1]:.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x, U_fom[-1, 0], "k", lw=2.5, label="FOM")
    for (k, tr), ls in zip(trajs.items(), ["-", ":", "--", "-."]):
        ax[0].plot(x, gal.reconstruct(tr[-1])[0], ls, label=k)
        ax[1].semilogy(t, errs[k], ls, label=k)
    ax[0].set_title(f"Burgers, N={N}, T={T}"); ax[0].legend(fontsize=7)
    ax[1].set_title("relative L2 error"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"rom_N{N}.png"), dpi=150)

# ---------------------------------------------------------------- entropy checks
N = 8
V = R.pod_basis(Usvd, N)
tol = hr.energy_tol(sigma, N)
rom, info = R.build_hyper_rom(fom, V, tol)
uN0 = R.GalerkinROM(fom, V).project(u0)

res, diss, ent = [], [], []
fom.eps = 0.0  # entropy conservative run
traj0 = integrate(rom.rhs, uN0, dt, NSTEPS, 1,
                  callback=lambda k, uN: res.append(rom.entropy_residual(uN)))[1]
ent0 = [rom.sampled_entropy(uN) for uN in traj0]
fom.eps = EPS
integrate(rom.rhs, uN0, dt, NSTEPS, 1, callback=lambda k, uN: diss.append(rom.entropy_dissipation(uN)))
print(f"\nEntropy (N={N}, eps=0): max |v_N^T conv| = {np.max(np.abs(res)):.2e}, "
      f"sampled entropy drift = {abs(ent0[-1] - ent0[0]):.2e} (time integration error)")
print(f"Entropy dissipation (eps={EPS}): min v_N^T d = {np.min(diss):.3e} (must be >= 0)")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
tt = dt * np.arange(1, NSTEPS + 1)
ax[0].semilogy(tt, np.abs(res)); ax[0].set_title(r"$|v_N^T\,2V(I)^T(Q_t\circ F)1|$, $\epsilon=0$")
ax[1].plot(dt * np.arange(NSTEPS + 1), ent0); ax[1].set_title(r"sampled entropy $1^TW S$, $\epsilon=0$")
ax[2].plot(tt, diss); ax[2].set_title(r"$v_N^T d(u_N)$, $\epsilon$=%g" % EPS)
for a in ax:
    a.set_xlabel("t")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "entropy.png"), dpi=150)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(x, V @ traj0[-1][0], label="HR-ROM, eps=0")
ax.plot(x, U_fom[-1, 0], "k", label="FOM, eps=%g" % EPS)
ax.legend(); ax.set_title(f"N={N}, no viscosity: oscillatory but bounded")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rom_eps0.png"), dpi=150)
print(f"\nFigures written to {os.path.abspath(OUT)}")
