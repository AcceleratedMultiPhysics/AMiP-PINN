"""DG experiment 1: FVM vs DG ROMs on periodic [-1, 1] (Qu, Narayan & Chan 2026, Sec. 6.2).

Total FOM dimension 1024: p = 0 (1024 cells, the FV limit), p = 3 (256 elements),
p = 7 (128 elements). Reproduces Tables 1-3 (HR-ROM error vs the FOM at T and the
number of hyper-reduced nodes) and Figs. 2-4 (snapshot singular values), and adds
solutions.png (FOM / ROM / HR-ROM at T with the HR nodes, for every p and N) and
error_nodes.png (errors and HR node counts vs N, with the paper's values).
Settings follow the authors' code: M-weighted POD, test basis [1, V, M^{-1}Q^T V],
target V * V (+1), stacked stabilization, naive viscosity K_N.

Run from ES-ROM/:
    python scripts/dg_compare_1d.py advection|burgers|euler [--trim abs] [--fom-dt]
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
from esdg import hyperreduction as hr  # noqa: E402
from esdg import rom as R  # noqa: E402
from esdg.fom import DGFOM1D, integrate, integrate_converged  # noqa: E402
from esdg.physics import Advection, Burgers, Euler1D  # noqa: E402


def euler_ic(ph, x):
    rho = 1 + 0.1 * np.exp(-25 * x ** 2)
    return ph.prim_to_cons(rho, 0.1 * np.sin(np.pi * x), rho ** ph.gamma)


CASES = {  # name: (physics, eps, T, modes, initial condition)
    "advection": (Advection(), 0.0, 1.0, [15, 20, 25], lambda ph, x: np.exp(-50 * x ** 2)[None]),
    "burgers": (Burgers(), 1e-2, 1.0, [30, 40, 50], lambda ph, x: (0.5 - np.sin(np.pi * x))[None]),
    "euler": (Euler1D(), 5e-4, 1.0, [20, 30, 40], euler_ic),
}
PAPER = {  # (error, nodes) from Tables 1-3, keyed by (p, N)
    "advection": {(0, 15): (1.28e-3, 31), (3, 15): (1.29e-3, 32), (7, 15): (1.29e-3, 31),
                  (0, 20): (1.52e-5, 44), (3, 20): (1.55e-5, 81), (7, 20): (1.55e-5, 68),
                  (0, 25): (9.12e-8, 58), (3, 25): (5.55e-7, 164), (7, 25): (1.02e-7, 155)},
    "burgers": {(0, 30): (1.38e-3, 68), (3, 30): (4.35e-4, 125), (7, 30): (6.18e-4, 69),
                (0, 40): (3.35e-5, 94), (3, 40): (6.93e-5, 214), (7, 40): (5.16e-5, 145),
                (0, 50): (2.43e-5, 128), (3, 50): (8.64e-6, 283), (7, 50): (1.81e-5, 271)},
    "euler": {(0, 20): (1.00e-4, 53), (3, 20): (1.00e-4, 77), (7, 20): (1.01e-4, 53),
              (0, 30): (4.98e-6, 93), (3, 30): (5.77e-6, 174), (7, 30): (5.10e-6, 80),
              (0, 40): (1.09e-7, 285), (3, 40): (1.14e-7, 415), (7, 40): (1.10e-7, 281)},
}
NFRAMES = 400

ap = argparse.ArgumentParser()
ap.add_argument("case", nargs="?", default="burgers", choices=list(CASES))
ap.add_argument("--trim", default="energy", choices=["energy", "abs"])
ap.add_argument("--fom-dt", action="store_true", help="integrate ROMs with the FOM time step")
args = ap.parse_args()
name = args.case
phys, EPS, T, MODES, ic = CASES[name]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dg", "compare", name)
os.makedirs(OUT, exist_ok=True)
print(f"{name}: target trimming = {args.trim}, ROM time step = {'FOM' if args.fom_dt else 'own'}")


def rel_err(fom, u, uref):
    return np.sqrt(np.sum(fom.mass * (u - uref) ** 2) / np.sum(fom.mass * uref ** 2))


def run_rom(rom, fom, uN0, U, nsteps_fom):
    """Integrate to T; returns (error at T, nodal state at T, steps, seconds incl. step refinement)."""
    tic = time.time()
    if args.fom_dt:
        n = nsteps_fom
        uT = integrate(rom.rhs, uN0, T / n, n, n)[1][-1]
    else:   # start at the ROM's stability limit, refine until converged in time
        samples = [rom.project(u) for u in U[::NFRAMES // 8]]
        n, _, tr = integrate_converged(rom.rhs, uN0, T, T / R.rom_stable_dt(rom, samples), rtol=1e-8)
        uT = tr[-1]
    uT = rom.reconstruct(uT)
    return rel_err(fom, uT, U[-1]), uT, n, time.time() - tic


rows, svals, sols = [], {}, {}
for p, n_el in [(0, 1024), (3, 256), (7, 128)]:
    fom = DGFOM1D(phys, n_el, p, eps=EPS)
    u0 = ic(phys, fom.x)
    nsteps = NFRAMES * int(np.ceil(T / fom.stable_dt(u0) / NFRAMES))
    tic = time.time()
    U = integrate(fom.rhs, u0, T / nsteps, nsteps, nsteps // NFRAMES)[1]
    print(f"\np={p}, {n_el} elements: FOM {nsteps} steps in {time.time() - tic:.1f}s")
    modes, sig = R.snapshot_svd(U, phys, fom.mass, entropy_enrich=phys.nvar > 1)
    svals[p] = sig / sig[0]

    for N in MODES:
        V = modes[:, :N]
        tol = hr.energy_tol(sig, N)
        gal = R.GalerkinROM(fom, V)
        uN0 = gal.project(u0)
        e_rom, u_rom, n_rom, t_rom = run_rom(gal, fom, uN0, U, nsteps)
        rom, info = R.build_hyper_rom(fom, V, tol, trim=args.trim)
        e_hr, u_hr, n_hr, t_hr = run_rom(rom, fom, uN0, U, nsteps)
        sols[(p, N)] = dict(x=fom.x, fom=U[-1, 0], rom=u_rom[0], hr=u_hr[0], I=rom.I,
                            e_rom=e_rom, e_hr=e_hr)
        pe, pn = PAPER[name][(p, N)]
        print(f"  N={N}: ROM {e_rom:.2e} | HR-ROM {e_hr:.2e} with {info['n_nodes']} nodes "
              f"(paper {pe:.2e} / {pn}), cond(Mt)={info['cond_Mt']:.1e}, "
              f"steps {n_rom}/{n_hr}, {t_rom:.0f}s/{t_hr:.0f}s")
        rows.append(dict(trim=args.trim, fom_dt=args.fom_dt, p=p, N=N, err_rom=e_rom, err_hr=e_hr,
                         nodes=info["n_nodes"], paper_err=pe, paper_nodes=pn,
                         cond_Mt=info["cond_Mt"], steps_fom=nsteps, steps_rom=n_rom, steps_hr=n_hr))

with open(os.path.join(OUT, "table.json"), "w") as f:
    json.dump(rows, f, indent=1)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for p, s in svals.items():
    ax.semilogy(s, ".", ms=3, label=f"p={p}")
ax.set_ylim(1e-16, 2)
ax.set_title(f"{name}: snapshot singular values")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "singular_values.png"), dpi=150)

print(f"\n{name}: HR-ROM error / nodes [paper]")
for N in MODES:
    cells = [next(r for r in rows if r["p"] == p and r["N"] == N) for p in (0, 3, 7)]
    print(f"N={N:3d} | " + " | ".join(
        f"p={c['p']}: {c['err_hr']:.2e}/{c['nodes']:3d} [{c['paper_err']:.2e}/{c['paper_nodes']:3d}]"
        for c in cells))

# ---------------------------------------------------------------- solutions at T
var = phys.names[0]
fig, ax = plt.subplots(3, len(MODES), figsize=(4.6 * len(MODES), 9.5), squeeze=False)
for i, p in enumerate((0, 3, 7)):
    for j, N in enumerate(MODES):
        d, a = sols[(p, N)], ax[i, j]
        a.plot(d["x"], d["fom"], "k", lw=2.2, label="FOM")
        a.plot(d["x"], d["rom"], "C0--", lw=1.2, label="ROM (no HR)")
        a.plot(d["x"], d["hr"], "C3:", lw=1.5, label="HR-ROM")
        a.plot(d["x"][d["I"]], d["hr"][d["I"]], "C3.", ms=3, label=f"HR nodes ({len(d['I'])})")
        a.set_title(f"p={p}, N={N}: ROM {d['e_rom']:.1e}, HR {d['e_hr']:.1e}", fontsize=9)
        a.legend(fontsize=6, loc="best")
        if j == 0:
            a.set_ylabel(var)
fig.suptitle(f"{name}: solution at T={T} (relative L2 errors vs FOM)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "solutions.png"), dpi=130)

# ---------------------------------------------------------------- error / nodes vs N
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, p in enumerate((0, 3, 7)):
    rr = [r for r in rows if r["p"] == p]
    Ns = [r["N"] for r in rr]
    ax[0].semilogy(Ns, [r["err_rom"] for r in rr], "o-", color=f"C{i}", label=f"p={p} ROM")
    ax[0].semilogy(Ns, [r["err_hr"] for r in rr], "s--", color=f"C{i}", label=f"p={p} HR-ROM")
    ax[0].semilogy(Ns, [r["paper_err"] for r in rr], "x", color=f"C{i}", ms=9, label=f"p={p} paper")
    ax[1].plot(Ns, [r["nodes"] for r in rr], "s--", color=f"C{i}", label=f"p={p}")
    ax[1].plot(Ns, [r["paper_nodes"] for r in rr], "x", color=f"C{i}", ms=9, label=f"p={p} paper")
ax[0].set_xlabel("N"); ax[0].set_title("relative L2 error at T"); ax[0].legend(fontsize=7, ncol=3)
ax[1].set_xlabel("N"); ax[1].set_title("HR nodes"); ax[1].legend(fontsize=7, ncol=2)
fig.suptitle(f"{name}: trim={args.trim}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "error_nodes.png"), dpi=150)
