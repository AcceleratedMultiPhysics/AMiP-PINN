"""Step 2: 1D compressible Euler with reflective walls (Chan 2020, Sec. 7.1).

Reproduces Fig. 2 (snapshot SVD with/without entropy-variable enrichment),
Fig. 3 (ROM density), Table 1 (point counts), Fig. 4 (entropy conservation
with eps = 0), Fig. 5 (viscous entropy dissipation (20)/(22)/(23)) and
Fig. 6 (error and entropy vs time, with/without hyper-reduction).

Run from ES-ROM/:  python scripts/euler_1d.py  [--modes 25 75 125] [--quick]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # small dense ops: threading only hurts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esrom import hyperreduction as hr  # noqa: E402
from esrom import rom as R  # noqa: E402
from esrom.fom import FOM1D, integrate  # noqa: E402
from esrom.physics import Euler1D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--modes", type=int, nargs="+", default=[25, 75, 125])
p.add_argument("--quick", action="store_true", help="K=500 grid for a fast smoke test")
p.add_argument("--t-snap", type=float, default=0.75,
               help="snapshot window end; the paper text says 0.7 but plots ROMs at 0.75")
args = p.parse_args()

OUT = os.path.join(os.path.dirname(__file__), "..", "results",
                   "euler" + ("_quick" if args.quick else "") + ("" if args.t_snap == 0.75 else f"_tsnap{args.t_snap:g}"))
os.makedirs(OUT, exist_ok=True)

# Sec. 7.1 setup. 2801 stored steps to T=0.7 -> dt = 0.7/2800 (CFL ~0.4 by |u|+c;
# the paper quotes CFL .75 with its own wave-speed estimate).
K = 500 if args.quick else 2500
EPS, T_SNAP, T_END = 2e-4, args.t_snap, 0.75
dt = 0.7 / 2800 * (2500 / K)
NSNAP, NEND = round(T_SNAP / dt), round(T_END / dt)
SUB = 10

phys = Euler1D(gamma=1.4)
fom = FOM1D(phys, K, bc="wall", eps=EPS, lxf=True)
x, dx = fom.x, fom.dx
g = np.exp(-100 * (x - 0.5) ** 2)
rho0 = 2 + 0.5 * g
u0 = phys.prim_to_cons(rho0, 0.1 * g, rho0 ** phys.gamma)


def rel_err(Urom, Ufom):
    """Relative discrete L2 error over all conservative variables, per time."""
    d = (Urom - Ufom).reshape(len(Ufom), -1)
    return np.linalg.norm(d, axis=1) / np.linalg.norm(Ufom.reshape(len(Ufom), -1), axis=1)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


# ================================================================ FOM
cache = os.path.join(OUT, "fom.npz")
if os.path.exists(cache):
    d = np.load(cache)
    t_all, U_all = d["t"], d["U"]
else:
    tic = time.time()
    t_all, U_all = integrate(fom.rhs, u0, dt, NEND, 1)
    np.savez(cache, t=t_all, U=U_all)
    print(f"FOM: {NEND} steps in {time.time() - tic:.1f}s")
snaps = U_all[:NSNAP + 1:SUB]
print(f"K={K}, dt={dt:.3e}, {NSNAP + 1} stored steps to T={T_SNAP:g}, {len(snaps)} SVD snapshots")
print(f"max CFL (|u|+c) dt/dx = {max(phys.wavespeed(u).max() for u in U_all[::100]) * dt / dx:.3f}")

# Times compared against: every 10th step up to T_END
idx = np.arange(0, NEND + 1, SUB)
t_cmp, U_cmp = t_all[idx], U_all[idx]

# ================================================================ Fig. 2
Uen, s_en = R.snapshot_svd(snaps, phys, entropy_enrich=True)
Upl, s_pl = R.snapshot_svd(snaps, phys, entropy_enrich=False)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].semilogy(s_pl / s_pl[0], ".", ms=3, label="u only")
ax[0].semilogy(s_en / s_en[0], ".", ms=3, label="u and v(u)")
ax[0].set_ylim(1e-16, 2); ax[0].set_title("snapshot singular values"); ax[0].legend()
for j in range(3):
    ax[1].plot(x, Upl[:, j], "--", lw=1, color=f"C{j}")
    ax[1].plot(x, Uen[:, j], lw=1, color=f"C{j}", label=f"mode {j+1}")
ax[1].set_title("POD modes (solid: enriched, dashed: u only)"); ax[1].legend(fontsize=7)
for lab, Ub in [("u only", Upl), ("u and v(u)", Uen)]:
    V = R.pod_basis(Ub, 25)
    Pj = V @ np.linalg.pinv(V)
    e = [np.linalg.norm(u - u @ Pj.T) / np.linalg.norm(u) for u in snaps]
    ax[2].semilogy(t_all[:NSNAP + 1:SUB], e, label=lab)
ax[2].set_title("projection error, N=25"); ax[2].set_xlabel("t"); ax[2].legend()
savefig(fig, "fig2_svd.png")

# ================================================================ ROMs
results, table = {}, {}
for N in args.modes:
    V = R.pod_basis(Uen, N)
    tol = hr.energy_tol(s_en, N)
    uN0 = R.GalerkinROM(fom, V).project(u0)
    res = {}

    tic = time.time()
    gal = R.GalerkinROM(fom, V)
    _, tr = integrate(gal.rhs, uN0, dt, NEND, SUB)
    res["no HR (12)"] = np.array([gal.reconstruct(u) for u in tr])
    print(f"\nN={N} (tol={tol:.2e}) Galerkin ROM: {time.time() - tic:.1f}s")

    for target in ["mass", "test_mass"]:
        tic = time.time()
        rom, info = R.build_hyper_rom(fom, V, tol, visc="22", target=target)
        toff = time.time() - tic
        tic = time.time()
        try:
            _, tr = integrate(rom.rhs, uN0, dt, NEND, SUB)
            Ur = np.array([rom.reconstruct(u) for u in tr])
            if not np.all(np.isfinite(Ur)):
                raise FloatingPointError("non-finite state")
        except (FloatingPointError, ValueError) as err:
            print(f"  HR target={target}: FAILED ({err})")
            Ur = np.full_like(res["no HR (12)"], np.nan)
        res[f"HR, target {target}"] = Ur
        info["offline_s"], info["online_s"] = round(toff, 1), round(time.time() - tic, 1)
        table[(N, target)] = info
        print(f"  HR target={target}: {info}")
        if target == "mass":
            rom_paper = rom

    errs = {k: rel_err(U, U_cmp) for k, U in res.items()}
    for k, e in errs.items():
        print(f"  {k:24s} rel. L2 error: T=0.25 {e[np.argmin(abs(t_cmp - .25))]:.3e}, T=0.75 {e[-1]:.3e}")
    results[N] = dict(res=res, errs=errs, I=rom_paper.I)

# ---------------------------------------------------------------- Fig. 3
fig, ax = plt.subplots(len(args.modes), 2, figsize=(12, 3.2 * len(args.modes)), squeeze=False)
for r, N in enumerate(args.modes):
    for c, tt in enumerate([0.25, 0.75]):
        k = np.argmin(abs(t_cmp - tt))
        a = ax[r, c]
        a.plot(x, U_cmp[k, 0], "k", lw=2, label="FOM")
        Ur = results[N]["res"]["HR, target mass"][k, 0]
        a.plot(x, Ur, "C0", lw=1, label="HR-ROM (paper target)")
        I = results[N]["I"]
        a.plot(x[I], Ur[I], "r.", ms=3, label="HR points")
        a.plot(x, results[N]["res"]["HR, target test_mass"][k, 0], "C2--", lw=1, label="HR-ROM (Vt o Vt target)")
        a.set_title(f"density, N={N}, T={tt}")
        if r == 0 and c == 0:
            a.legend(fontsize=7)
savefig(fig, "fig3_density.png")

# ---------------------------------------------------------------- Table 1
with open(os.path.join(OUT, "table1.json"), "w") as f:
    json.dump({f"N={N},{tg}": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                               for k, v in info.items()} for (N, tg), info in table.items()}, f, indent=1)
print("\nTable 1 (paper target):  N | cubature | stabilizing | viscous")
for N in args.modes:
    i = table[(N, "mass")]
    print(f"   {N:4d} | {i['n_cubature']:8d} | {i['n_stabilizing']:11d} | {i['n_viscous']:7d}")

# ---------------------------------------------------------------- Fig. 6
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for j, N in enumerate(args.modes):
    for k, ls in zip(results[N]["errs"], ["-", "--", ":"]):
        ax[0].semilogy(t_cmp, results[N]["errs"][k], ls, color=f"C{j}", label=f"N={N}, {k}")
S_fom = [fom.total_entropy(u) for u in U_cmp]
ax[1].plot(t_cmp, S_fom, "k", lw=2, label="FOM")
for j, N in enumerate(args.modes):
    ax[1].plot(t_cmp, [fom.total_entropy(u) for u in results[N]["res"]["HR, target mass"]],
               color=f"C{j}", label=f"HR-ROM N={N}")
ax[0].set_title("relative L2 error"); ax[0].legend(fontsize=6); ax[0].set_xlabel("t")
ax[1].set_title(r"entropy $\Delta x\, 1^T S(u)$ on FOM grid"); ax[1].legend(fontsize=7); ax[1].set_xlabel("t")
savefig(fig, "fig6_error_entropy.png")

# ================================================================ Fig. 4: eps = 0
fom.eps = 0.0
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for j, N in enumerate(args.modes):
    V = R.pod_basis(Uen, N)
    rom, _ = R.build_hyper_rom(fom, V, hr.energy_tol(s_en, N), visc=None)
    resid = []
    _, tr = integrate(rom.rhs, R.GalerkinROM(fom, V).project(u0), dt, NEND, NEND,
                      callback=lambda k, uN: resid.append(abs(rom.entropy_residual(uN))))
    print(f"eps=0, N={N}: max |convective entropy residual| = {max(resid):.2e}")
    ax[0].semilogy(dt * np.arange(1, NEND + 1), resid, lw=0.5, label=f"N={N}")
    if N == args.modes[-1]:
        ax[1].plot(x, U_all[-1, 0], "k", lw=2, label=r"FOM ($\epsilon$=2e-4)")
        ax[1].plot(x, rom.reconstruct(tr[-1])[0], lw=1, label=rf"HR-ROM N={N}, $\epsilon$=0")
ax[0].set_title(r"$|v_N^T[2V_h^T(Q_h\circ F)1 + V_b^TB(f^*_{EC}-f)]|$"); ax[0].legend(); ax[0].set_xlabel("t")
ax[1].set_title("density at T=0.75 without viscosity"); ax[1].legend()
savefig(fig, "fig4_entropy_eps0.png")
fom.eps = EPS

# ================================================================ Fig. 5: viscous dissipation
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), squeeze=False)
for j, N in enumerate(args.modes[:2]):
    V = R.pod_basis(Uen, N)
    tol = hr.energy_tol(s_en, N)
    uN0 = R.GalerkinROM(fom, V).project(u0)
    for visc, ls in zip(["20", "22", "23"], ["-", "--", ":"]):
        rom, _ = R.build_hyper_rom(fom, V, tol, visc=visc)
        diss = []
        integrate(rom.rhs, uN0, dt, NEND, NEND,
                  callback=lambda k, uN: diss.append(rom.entropy_dissipation(uN)) if k % SUB == 0 else None)
        diss = np.array(diss)
        print(f"N={N}, viscosity ({visc}): min v_N^T d = {diss.min():.3e}")
        ax[0, j].plot(t_cmp[1:], diss, ls, label=f"({visc})")
    ax[0, j].set_title(rf"$v_N^T d(u_N)$, N={N}"); ax[0, j].legend(); ax[0, j].set_xlabel("t")
savefig(fig, "fig5_dissipation.png")
print(f"\nFigures written to {os.path.abspath(OUT)}")
