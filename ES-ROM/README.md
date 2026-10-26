# ES-ROM: entropy stable reduced order models (step 1: Chan 2020, 1D)

A Python/NumPy reproduction of

> J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws*, JCP 423 (2020) 109789

for 1D problems. It is the first step toward combining Chan's ES-ROMs with our VMS/PINN closures (see `papers/es_rom_chan2020_notes.md`).

**[docs/code_map.md](docs/code_map.md)** maps every equation of the paper to the code that implements it. It also explains each change to the method and each choice made where the paper is ambiguous.

## Layout

| file | contents | paper |
|---|---|---|
| `esrom/physics.py` | Burgers and 1D Euler: entropy, entropy variables and their inverse, EC fluxes (Chandrashekar for Euler, stable log-mean), `du/dv` | Sec. 2, 7 |
| `esrom/operators.py` | `Q` (periodic / SBP), difference `D`, viscosity `K = dx DᵀD` | eqs. (5), (8), (24) |
| `esrom/fom.py` | flux-differencing FV FOM, mirror-state wall flux + LxF, 5-stage RK4 (Carpenter–Kennedy) | Secs. 2, 5 |
| `esrom/hyperreduction.py` | greedy empirical cubature (Alg. 1), target compression (eq. 18), stabilizing points | Secs. 4.2–4.3 |
| `esrom/rom.py` | POD with entropy-variable enrichment, test basis `[1, V, QV]`, `GalerkinROM` (Thm 2), `HyperROM` (Thms 3/4, hybridized SBP), viscosity (20)/(22)/(23), offline builder | Secs. 3–5 |
| `tests/test_invariants.py` | the invariant checklist from the notes | |
| `scripts/burgers_1d.py` | step 1: periodic Burgers, Fig. 1 test-basis failure and fix | Sec. 4.1.1 |
| `scripts/euler_1d.py` | step 2: Euler with walls, Figs. 2–6 and Table 1 | Sec. 7.1 |

## Running

```bash
python tests/test_invariants.py
python scripts/burgers_1d.py          # ~15 s
python scripts/euler_1d.py            # K=2500, N=25,75,125
python scripts/euler_1d.py --quick    # K=500 smoke test, ~1 min
```

Figures go to `results/`. The scripts set `OPENBLAS_NUM_THREADS=1`. The online work is many small dense products, and multithreaded BLAS made it more than 100× slower on this 20-core machine.

## Results (step 1 complete)

`tests/test_invariants.py` passes. It checks flux consistency, symmetry and the Tadmor shuffle condition; `v(u)` and `du/dv` against finite differences; `Q1=0` and skew/SBP structure for `Q`, `Q_t` and the hybridized `Q_h`; `P_t V_t(I) = I`; positive weights; entropy residuals ~1e-14.

### Burgers, periodic, `u0 = -sin(pi x)` (`results/burgers/`, `log.txt`)
K=1000, ε=2e-3, T=1. Relative L2 error at T:

| N | unreduced (Thm 2) | HR, `V_t=[1,V]` | HR, `[1,V,QV]`, paper target `V∘V` | HR, `[1,V,QV]`, target `V_t∘V_t` |
|---|---|---|---|---|
| 4 | 5.4e-2 | 1.03 | 1.3e-1 | 5.4e-2 |
| 8 | 1.8e-3 | 1.03 | 2.3e-1 | 1.8e-3 |
| 12 | 6.9e-5 | 1.03 | 7.8e-3 | 6.9e-5 |

- Fig. 1 is reproduced: the POD modes are antisymmetric and their derivatives `QV` are nearly symmetric. So `VᵀQV ≈ 1e-16`, and the `[1,V]` ROM freezes at its initial condition.
- With ε=0 the entropy residual is ~1e-12, against terms of order 10². `v_Nᵀd ≥ 0` throughout.

### 1D Euler, reflective walls (Sec. 7.1; `results/euler/`, `log.txt`, `table1.json`)
K=2500, ε=2e-4, dt=2.5e-4, 301 snapshots of `u` and `v(u)` on [0, 0.75], viscosity (22). Relative L2 error over all conservative variables at T=0.75:

| N | unreduced (12) | HR, paper target | HR, `V_t∘V_t` target | cubature / stabilizing / viscous pts | paper Table 1 |
|---|---|---|---|---|---|
| 25 | 1.17e-2 | 1.16e-2 | 1.16e-2 | 59 / 1 / 66 | 54 / 3 / 54 |
| 75 | 1.19e-3 | 1.18e-3 | 1.18e-3 | 191 / 7 / 211 | 158 / 21 / 159 |
| 125 | 2.72e-4 | 2.73e-4 | 2.72e-4 | 354 / 0 / 389 | 259 / 36 / 259 |

- **Accuracy (Figs. 3, 6):** the hyper-reduced error equals the unreduced error, and both converge with N, as in the paper. There are Gibbs oscillations at N=25. One small difference: the N=75 HR error has a bump around t≈0.5–0.65 (1.6e-3 vs 7e-4 unreduced) that recovers by T=0.75.
- **Entropy (Fig. 4):** with ε=0, `|v_Nᵀ[2V_hᵀ(Q_h∘F)1 + V_bᵀB(f*_EC − f)]|` is ≤ 5e-13 at every step, and the ROM stays bounded.
- **Dissipation (Fig. 5):** (20) and (22) coincide and are positive; (23) differs slightly.
- **Snapshots (Fig. 2):** entropy-variable enrichment slows the singular-value decay only below about 1e-8. The N=25 projection errors with and without enrichment are indistinguishable.
- **Point counts:** ours are about 20–50% higher than the paper's. Likely reasons: the snapshot window (0.75 vs 0.7), and the compression of the product space (all (i,j) pairs, via the Gram matrix).
- **Cost:** with the `V_t∘V_t` target the online cost is 2–7× higher (1.9–2.4× the points, and O(N_s²) flux evaluations). At this resolution the HR-ROM is no cheaper than the FOM under explicit time stepping, as the paper notes in Sec. 7.3.

### Departures from the paper, found while reproducing it
1. **Cubature target.** The paper targets `V∘V` (eq. 17), which makes the reduced mass matrix accurate. The hyper-reduced *test* mass matrix on the `QV` directions is then left poorly integrated.
   - For Burgers this is the dominant error (table above).
   - Targeting `V_t∘V_t` (`build_hyper_rom(..., target="test_mass")`) makes HR match the unreduced ROM, at about 1.5–2.5× the points.
   - For Euler both targets work, as the paper reports.
2. **Stabilizing points (Sec. 4.3).**
   - Re-solving the joint weights with NNLS (α_Z=1e-2) drops points, because NNLS solutions are sparse. At N=75 that gave cond(M_t) ≈ 1e9 and a blow-up.
   - The fixes:
     - stabilizing points are chosen outside `I`;
     - the weights come from min-norm least squares, with bounded LS (`w ≥ 0.1 dx`) as the fallback;
     - the step repeats until M_t has no eigenvalue below `1e-3`;
     - ties in Algorithm 1 go to the largest row. With a one-column target all normalized rows tie, and the weak direction at N=25 is a one-cell spike at the wall.
   - The defaults `eig_tol=1e-3, alpha_Z=1e-4` were tuned at N=75. With α_Z=1e-2, 87 points get added and the error is 8e-2. With the new defaults, 7 points get added and the error is 1.2e-3.
3. **Snapshot window.** The paper says snapshots to T=0.7 but plots at T=0.75. Past 0.7 the ROM is extrapolating and the error jumps (`results/euler_quick_tsnap0.7`). The default is now 0.75 (`--t-snap 0.7` to compare).

## Step 2: DG version (Qu, Narayan & Chan 2026), 1D

A separate package, `esdg/`, implements the entropy stable DG ROM of Qu, Narayan & Chan, JCP 557 (2026) 114830, for 1D periodic problems (Secs. 3.1–3.5 and 5; boundary conditions left out). It does not import or modify the FV package `esrom/`.

**[docs/dg_notes.md](docs/dg_notes.md)** has the full write-up:
- what changes from FV to DG;
- why the test basis must be `M⁻¹QᵀV` (Theorem 4);
- a paper → code map and the reproduced tables;
- experiments with the test basis and hyper-reduction;
- ideas for combining with our closures.

| file | contents |
|---|---|
| `esdg/physics.py` | copy of the FV physics, plus linear advection |
| `esdg/fom.py` | GLL/SBP operators, global periodic `Q_Ω`, nodal ES-DG FOM, BR-1 viscosity, RK4, time-step tools |
| `esdg/hyperreduction.py` | V∗V target, greedy cubature, stabilization (authors' "stacked" and our "union"), Carathéodory pruning |
| `esdg/rom.py` | M-weighted POD, test basis, Galerkin ROM (Thm 3), hyper-reduced ROM, offline builder, ROM time step |
| `tests/test_dg_invariants.py` | SBP, skew-symmetry, entropy residuals, Theorem 4, Carathéodory |
| `scripts/dg_compare_1d.py` | Sec. 6.2: FV (p=0) vs DG (p=3, 7) on advection, Burgers, Euler (Tables 1–3) |
| `scripts/dg_play_1d.py` | experiments: test basis, target, stabilization, Carathéodory, error vs nodes |

```bash
python tests/test_dg_invariants.py
python scripts/dg_compare_1d.py advection     # also: burgers, euler --trim abs
python scripts/dg_play_1d.py
```
Results go to `results/dg/`.
