"""DG experiment 3: test basis and hyper-reduction choices.

Case: viscous Burgers, DG p=3 on 256 periodic elements, eps=1e-2,
u0 = 0.5 - sin(pi x), T = 1 (the setting of Fig. 1 in Qu, Narayan & Chan 2026).

A. Test basis (Sec. 3.5.1, Theorem 4): enrichment none / QV / M^{-1}Q^T V / both.
   "ideal" = all FOM nodes (offline check of V^T (Q - Qbar) only); "HR" = real
   hyper-reduction, time-integrated (reproduces Fig. 1).
B. Hyper-reduction choices at fixed N: target V*V vs Vt*Vt, trimming, stabilization
   (stacked = authors' code, union = our FV variant, none), greedy vs Caratheodory.
C. Error vs number of HR nodes (sweeping the cubature tolerance).

Diagnostics per case (offline): cond of the HR test mass matrix, error of the HR
mass matrix, Theorem-4 residual under the actual quadrature, and the error of the
HR convective term against the unreduced ROM over the snapshots.

Run from ES-ROM/:  python scripts/dg_play_1d.py
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # small dense ops: threading only hurts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esdg import hyperreduction as hr  # noqa: E402
from esdg import rom as R  # noqa: E402
from esdg.fom import DGFOM1D, integrate, integrate_converged  # noqa: E402
from esdg.physics import Burgers  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dg", "play")
os.makedirs(OUT, exist_ok=True)
NFRAMES, T = 400, 1.0

phys = Burgers()
fom = DGFOM1D(phys, 256, 3, eps=1e-2)
u0 = (0.5 - np.sin(np.pi * fom.x))[None]
NSTEPS = NFRAMES * int(np.ceil(T / fom.stable_dt(u0) / NFRAMES))
U = integrate(fom.rhs, u0, T / NSTEPS, NSTEPS, NSTEPS // NFRAMES)[1]
MODES, SIG = R.snapshot_svd(U, phys, fom.mass, entropy_enrich=False)
QD = fom.Q.toarray()


def rel_err(u, uref):
    return np.sqrt(np.sum(fom.mass * (u - uref) ** 2) / np.sum(fom.mass * uref ** 2))


def time_run(rom):
    """Converged-in-time solution at T; returns (error, final nodal state, steps)."""
    uN0 = rom.project(u0)
    samples = [rom.project(u) for u in U[::NFRAMES // 8]]
    n, _, tr = integrate_converged(rom.rhs, uN0, T, T / R.rom_stable_dt(rom, samples), rtol=1e-8)
    uT = tr[-1]
    uT = rom.reconstruct(uT)
    err = rel_err(uT, U[-1])
    return (err, uT[0], n) if np.isfinite(err) else (np.nan, None, n)


def diagnostics(rom, V):
    G = V.T @ (fom.mass[:, None] * V)
    VtQ = V.T @ QD
    X = np.hstack([V, U[::20, 0].T])                 # functions in range(V) and snapshots
    exact = VtQ @ X
    hrv = rom.V[rom.I].T @ rom.Qbar @ X[rom.I]
    gal = R.GalerkinROM(fom, V)
    num = den = 0.0
    for u in U[::20]:
        uN = gal.project(u)
        rg = (fom.conv(gal._states(uN)[1]) @ V) @ gal.MNinv
        rh = rom._conv(rom._states(uN)[1]) @ rom.MNinv
        num += np.sum((rh - rg) ** 2)
        den += np.sum(rg ** 2)
    return dict(thm4=np.linalg.norm(hrv - exact) / np.linalg.norm(exact),
                conv_err=np.sqrt(num / den),
                mass_err=np.linalg.norm(rom.MN - G) / np.linalg.norm(G))


def run(job):
    """Never raises: a failed case is reported instead of killing the whole sweep."""
    label, N, kw, tol_fac = job
    try:
        V = MODES[:, :N]
        tic = time.time()
        rom, info = R.build_hyper_rom(fom, V, hr.energy_tol(SIG, N) * tol_fac, **kw)
        t_off = time.time() - tic
        d = diagnostics(rom, V)
        tic = time.time()
        if kw.get("selection") == "all":             # ideal HR: offline check only
            err, uT, n = np.nan, None, 0
        else:
            err, uT, n = time_run(rom)
        return dict(label=label, N=N, tol_fac=tol_fac, err=err, nodes=info["n_nodes"],
                    n_test=info["n_test"], n_target=info["n_target"], cond_Mt=info["cond_Mt"],
                    steps=n, t_off=t_off, t_on=time.time() - tic, I=rom.I.tolist(),
                    u=None if uT is None else uT.tolist(), **d)
    except Exception as exc:  # noqa: BLE001
        return dict(label=label, N=N, tol_fac=tol_fac, failed=f"{type(exc).__name__}: {exc}")


def galerkin(N):
    gal = R.GalerkinROM(fom, MODES[:, :N])
    err, uT, n = time_run(gal)
    return dict(N=N, err=err, u=uT.tolist(), steps=n)


PAPER = dict(stab="stacked")                          # authors' configuration
JOBS = []
for N in [20, 30]:                                    # A. test basis
    for enr in [None, "QV", "MinvQtV", "both"]:
        JOBS.append((f"A ideal enrich={enr}", N, dict(enrich=enr, selection="all"), 1.0))
        JOBS.append((f"A HR enrich={enr}", N, dict(enrich=enr, **PAPER), 1.0))
for N in [20, 30, 40]:                                # B. hyper-reduction choices
    JOBS += [
        ("B paper: VV, energy trim, stacked", N, dict(PAPER), 1.0),
        ("B VV, abs trim, stacked", N, dict(PAPER, trim="abs"), 1.0),
        ("B VV, stab union", N, dict(stab="union"), 1.0),
        ("B VV, no stab", N, dict(stab=None), 1.0),
        ("B VtVt target", N, dict(target="test_mass", **PAPER), 1.0),
        ("B Caratheodory VV + stacked", N, dict(selection="caratheodory", **PAPER), 1.0),
        ("B Caratheodory VtVt", N, dict(selection="caratheodory", target="test_mass", **PAPER), 1.0),
    ]
for fac in [10.0, 1.0, 0.1, 0.01, 1e-3]:             # C. error vs nodes
    JOBS.append(("C VV", 30, dict(PAPER), fac))
    JOBS.append(("C VtVt", 30, dict(target="test_mass", **PAPER), fac))

if __name__ == "__main__":
    print(f"FOM: 1024 nodes, {NSTEPS} steps; {len(JOBS)} hyper-reduced ROMs")
    tic = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        gal = {g["N"]: g for g in ex.map(galerkin, [20, 30, 40])}
        res = list(ex.map(run, JOBS))
    print(f"all runs done in {time.time() - tic:.0f}s")
    for r in res:
        if "failed" in r:
            print(f"FAILED {r['label']} N={r['N']}: {r['failed']}")
    res = [r for r in res if "failed" not in r]
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(galerkin={k: dict(err=v["err"], steps=v["steps"]) for k, v in gal.items()},
                       runs=[{k: v for k, v in r.items() if k not in ("u", "I")} for r in res]),
                  f, indent=1)

    hdr = "nodes  cond(Mt)  mass_err  thm4_res  conv_err    HR err  (ROM err)"
    for part in "ABC":
        print(f"\n=== Part {part} ===")
        print(f"{'case':36s} {'N':>3s} {'tolx':>5s} " + hdr)
        for r in res:
            if r["label"].startswith(part):
                print(f"{r['label'][2:]:36s} {r['N']:3d} {r['tol_fac']:5g} {r['nodes']:5d} "
                      f"{r['cond_Mt']:9.1e} {r['mass_err']:9.1e} {r['thm4']:9.1e} {r['conv_err']:9.1e} "
                      f"{r['err']:9.2e}  ({gal[r['N']]['err']:.2e})")

    # Fig. 1 of the paper: test basis QV vs M^{-1} Q^T V (N = 30, where every variant runs;
    # at N = 20 the V*V target gives 40 nodes for a 41-dimensional test basis: singular M_t)
    panels = [("A HR enrich=QV", "[1, V, QV], target V*V"),
              ("A HR enrich=MinvQtV", "[1, V, M^-1 Q^T V], target V*V (paper)"),
              ("B VtVt target", "[1, V, M^-1 Q^T V], target Vt*Vt")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for a, (lab, title) in zip(ax, panels):
        r = next(r for r in res if r["label"] == lab and r["N"] == 30)
        a.plot(fom.x, U[-1, 0], lw=2, label="FOM")
        a.plot(fom.x, gal[30]["u"], "--", label="ROM (no HR)")
        if r["u"] is not None:
            u = np.array(r["u"])
            a.plot(fom.x, u, label="HR-ROM")
            a.plot(fom.x[r["I"]], u[r["I"]], "k.", ms=3, label=f"HR nodes ({r['nodes']})")
        a.set_title(f"{title}\nHR err {r['err']:.1e} (ROM {gal[30]['err']:.1e})", fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle("Burgers, DG p=3, N=30, T=1")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_test_basis.png"), dpi=150)

    # Part C: error vs nodes
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, mk in [("C VV", "o-"), ("C VtVt", "s-")]:
        rr = sorted([r for r in res if r["label"] == lab and np.isfinite(r["err"])], key=lambda r: r["nodes"])
        ax.loglog([r["nodes"] for r in rr], [r["err"] for r in rr], mk, label=lab[2:] + " target")
    ax.axhline(gal[30]["err"], color="k", ls=":", label="ROM without HR")
    ax.set_xlabel("number of HR nodes"); ax.set_ylabel("relative L2 error at T"); ax.set_title("N = 30")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "error_vs_nodes.png"), dpi=150)

    # node locations, N = 30
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(fom.x, U[-1, 0], "k", lw=1, label="FOM at T")
    for k, (lab, c) in enumerate([("B paper: VV, energy trim, stacked", "C0"), ("B VtVt target", "C1"),
                                  ("B Caratheodory VV + stacked", "C2")]):
        r = next((r for r in res if r["label"] == lab and r["N"] == 30), None)
        if r is not None:
            ax.plot(fom.x[r["I"]], np.full(len(r["I"]), -2.0 - 0.25 * k), "|", color=c, ms=8,
                    label=f"{lab[2:]} ({r['nodes']} nodes)")
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("HR node locations, N = 30")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "node_locations.png"), dpi=150)
    print(f"\nFigures written to {os.path.abspath(OUT)}")
