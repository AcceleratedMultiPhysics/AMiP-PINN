# Entropy stable DG ROMs in 1D: notes, code map and experiments

Reference: R. Qu, A. Narayan, J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws using discontinuous Galerkin methods*, JCP 557 (2026) 114830 (`papers/Qu2026.pdf`). Authors' code: `github.com/rayqu1126/ES_DG_ROM`, read for details the paper leaves out.

**Scope.** 1D periodic problems only: Secs. 3.1–3.5 (DG FOM, ROM, hyper-reduction, test basis) and Sec. 5 (viscosity). Boundary conditions (Sec. 3.6) are left out on purpose. From Sec. 4 only Carathéodory pruning (4.5) is used, applied to the 1D volume quadrature. Experiments are the periodic 1D ones of Sec. 6.2.

**Code.** The DG work lives in its own package, `esdg/`, and imports nothing from the FV package `esrom/`, which is unchanged. `esdg/physics.py` is a copy of the FV physics plus linear advection.

---

## 1. FV → DG: what actually changes

The DG FOM has *the same algebraic structure* as Chan's FV FOM:

| | FV (Chan 2020) | DG (Qu et al. 2026) |
|---|---|---|
| unknowns | cell averages | values at (p+1) Gauss–Lobatto nodes per element |
| mass matrix M | Δx I | diagonal, **non-uniform**: J·(GLL weights) per element |
| Q | ½ central difference (tridiagonal) | ½ [Q−Qᵀ blocks inside elements, ±1 couplings between neighbouring end nodes] (eq. 10) |
| properties needed for entropy | Q = −Qᵀ, Q1 = 0 | the same (Theorem 1), plus M diagonal |
| flux differencing | each cell talks to 2 neighbours | each node talks to **every node of its element** + the neighbour's end node |
| viscosity | ε (1/Δx) tridiag(−1, 2, −1) | ε Qᵀ M⁻¹ Q (BR-1, eq. 39) |

With p = 0 the DG scheme *is* the FV scheme. `tests/test_dg_invariants.py` checks that M = h I and that Q is the ½ central difference.

The single new ingredient is the **non-uniform M**. Examples of GLL weights on [−1, 1]:

| p | weights |
|---|---|
| 3 | 0.167, 0.833, 0.833, 0.167 |
| 7 | 0.036 … 0.413 |

The end nodes of each element carry weights 5–10× smaller than interior nodes. Every place where FV silently used "Δx × identity" must now use M. That is the whole story of the paper's main contribution, the new test basis.

### M-weighting throughout the ROM
| Step | FV | DG |
|---|---|---|
| POD | SVD of X | SVD of √M X, modes √M⁻¹U (Sec. 3.4.2) |
| projection | V† = (VᵀV)⁻¹Vᵀ | V† = (VᵀMV)⁻¹VᵀM |
| entropy projection | ũ = u(VV† v(Vu_N)) | same, with the M-projection (eq. 13) |
| reference weights for the cubature | Δx 1 | diag(M) |
| test basis | [1, V, QV] | **[1, V, M⁻¹QᵀV]** (eq. 15) |

---

## 2. Why the test basis must be M⁻¹QᵀV (Theorem 4)

**What the ROM needs.** The ROM only ever sees the flux term through Vᵀ(Q∘F)1, i.e. through the rows of VᵀQ. Write one row as an M-inner product:

  VᵀQ x = (M⁻¹QᵀV)ᵀ M x = ⟨M⁻¹QᵀV, x⟩_M.

**What hyper-reduction does.** It replaces x by its projection onto the test space, V_t P_t x. In the ideal case (all nodes, weights M), P_t is the M-orthogonal projector. So VᵀQ is reproduced exactly **iff M⁻¹QᵀV ∈ range(V_t)**.

**Why QV is the wrong object.** QV is a column of *weighted integrals* ∫φ_i ∂_x ψ, a "dual" object (moments), not nodal values of a function. Dividing by M turns it into a function, the strong derivative M⁻¹QV. With M = Δx I (FV) the division is a harmless constant, so QV and M⁻¹QᵀV span the same space; the FV enrichment was right for that reason. With GLL weights, QV is distorted node by node: the end-node values get the wrong size relative to interior ones.

**Checked in the code:**
- `test_theorem4_ideal_hyperreduction` asserts ‖Vᵀ(Q − Q̄)‖ ≈ 1e-15 for M⁻¹QᵀV and not for QV or no enrichment.
- Experiment A below shows the same on Burgers. With ideal hyper-reduction the convective term is exact to 1e-14 with M⁻¹QᵀV, but off by 3.5% (QV) or 9% (none).

**Periodic case.** Here Qᵀ = −Q, so M⁻¹QᵀV and M⁻¹QV span the same space; the authors' periodic code uses `M \ (Q*V)`.

---

## 3. Paper → code map (`esdg/`)

| Paper | Code |
|---|---|
| GLL nodes, weights, D; SBP Q + Qᵀ = B (eq. 4) | [`gauss_lobatto`](../esdg/fom.py#L47) |
| Global periodic Q_Ω (eq. 10), M_Ω, K = QᵀM⁻¹Q (eq. 39) | [`DGFOM1D.__init__`](../esdg/fom.py#L72) |
| FOM M du/dt + 2(Q∘F)1 + εKu = 0 (eq. 9) | [`DGFOM1D.conv`](../esdg/fom.py#L99), [`rhs`](../esdg/fom.py#L110) |
| Theorem 2 check | [`DGFOM1D.entropy_residual`](../esdg/fom.py#L119) |
| M-weighted POD with u and v(u) snapshots (Sec. 3.4.2) | [`snapshot_svd`](../esdg/rom.py#L20) |
| Test basis (14) / (15) | [`test_basis`](../esdg/rom.py#L42) |
| ROM without HR, eq. (13), Theorem 3 | [`GalerkinROM`](../esdg/rom.py#L85) |
| HR ROM: M̄_N, P_N, P_t, Q̄ = P_tᵀ V_tᵀQV_t P_t (Sec. 3.5) | [`HyperROM.__init__`](../esdg/rom.py#L119) |
| Hyper-reduced flux term 2V(I)ᵀ(Q̄∘F)1 | [`HyperROM._conv`](../esdg/rom.py#L150) |
| Entropy residual of the HR ROM | [`HyperROM.entropy_residual`](../esdg/rom.py#L167) |
| Viscosity K_N (eq. 40); provably dissipative variant (eq. 43) | [`_Viscosity`](../esdg/rom.py#L57) |
| Target V * V (Sec. 3.5) | [`target_basis`](../esdg/hyperreduction.py#L22) |
| Greedy cubature (Alg. 1 of Chan 2020) | [`greedy_cubature`](../esdg/hyperreduction.py#L47) |
| Stabilizing nodes, authors' code | [`stabilize_stacked`](../esdg/hyperreduction.py#L86) |
| Stabilizing nodes, our FV variant | [`stabilize_union`](../esdg/hyperreduction.py#L97) |
| Carathéodory pruning (Sec. 4.5, Algorithm 1) | [`caratheodory_prune`](../esdg/hyperreduction.py#L123) |
| Offline recipe, all options | [`build_hyper_rom`](../esdg/rom.py#L180) |
| ROM time step (CFL relaxation, Sec. 6.3.3) | [`rom_stable_dt`](../esdg/rom.py#L213), [`integrate_converged`](../esdg/fom.py#L139) |

**Choices where the paper is silent**, following the authors' code unless noted:
- **POD:** no forced constant mode.
- **Target:** 1 is added to the V∘V target.
- **Stabilization:** "stacked" stabilization (eigenvalue threshold 1e-8, α = 1e-2).
- **Viscosity:** naive K_N with the hyper-reduced mass matrix.
- **Euler flux:** Chandrashekar, where the code uses Ranocha's. Both are entropy conservative.
- **Target trimming:** relative energy E_k ≤ tol by default, as in Chan 2020. The authors' code uses an absolute threshold (`--trim abs`), used here for Euler.
- **Time stepping** (our own choice): the authors use adaptive RK. Here the FOM uses a fixed step from the spectral radius. The ROMs use their own step, starting from the stability limit of the ROM Jacobian and halving until the solution at T changes by less than `rtol` relative (1e-8 in these scripts). Choose `rtol` well below the ROM error being measured: too strict a value costs steps for nothing.
  - Stability alone was not enough: for advection, the step at the stability limit gave a time error of 1.5e-3, which swamped ROM errors of 1e-7.

---

## 4. Reproduction of the paper's 1D results

### 4.1 FV vs DG on periodic problems (Sec. 6.2; `results/dg/compare/`)
Total FOM dimension is 1024: p = 0 (1024 cells), p = 3 (256 elements), p = 7 (128 elements). Each figure folder has `solutions.png`, `error_nodes.png` and `singular_values.png`. Entries are HR-ROM error / HR nodes, with the paper's values in brackets.

**Advection** (Table 1; target trimmed by energy): matches the paper to 2–3 digits.

| N | p = 0 | p = 3 | p = 7 |
|---|---|---|---|
| 15 | 1.28e-3 / 29 [1.28e-3 / 31] | 1.29e-3 / 30 [1.29e-3 / 32] | 1.29e-3 / 30 [1.29e-3 / 31] |
| 20 | 1.52e-5 / 42 [1.52e-5 / 44] | 1.56e-5 / 72 [1.55e-5 / 81] | 1.55e-5 / 42 [1.55e-5 / 68] |
| 25 | 9.16e-8 / 56 [9.12e-8 / 58] | 5.66e-7 / 147 [5.55e-7 / 164] | 1.16e-7 / 56 [1.02e-7 / 155] |

**Euler** (Table 3; target trimmed absolutely, as in the authors' code): the HR-ROM error equals the unreduced ROM error everywhere. Our errors are uniformly about 3.6× below the paper's, likely from time-integration tolerance or snapshot details.

| N | p = 0 | p = 3 | p = 7 |
|---|---|---|---|
| 20 | 2.82e-5 / 51 [1.00e-4 / 53] | 2.83e-5 / 75 [1.00e-4 / 77] | 2.83e-5 / 51 [1.01e-4 / 53] |
| 30 | 1.30e-6 / 78 [4.98e-6 / 93] | 1.28e-6 / 161 [5.77e-6 / 174] | 1.31e-6 / 78 [5.10e-6 / 80] |
| 40 | 2.69e-8 / 108 [1.09e-7 / 285] | 2.89e-8 / 300 [1.14e-7 / 415] | 2.83e-8 / 151 [1.10e-7 / 281] |

**Burgers** (Table 2; energy trimming). Format: HR-ROM error / nodes [paper]; unreduced ROM error in parentheses.

| N | p = 0 | p = 3 | p = 7 |
|---|---|---|---|
| 30 | 4.46e-4 / 68 [1.38e-3 / 68] (4.4e-4) | **3.73e-3** / 92 [4.35e-4 / 125] (4.2e-4) | **2.04e-2** / 63 [6.18e-4 / 69] (4.2e-4) |
| 40 | 7.59e-5 / 87 [3.35e-5 / 94] (3.4e-5) | 1.13e-4 / 170 [6.93e-5 / 214] (3.1e-5) | 1.78e-4 / 109 [5.16e-5 / 145] (3.1e-5) |
| 50 | 4.88e-5 / 110 [2.43e-5 / 128] (2.5e-6) | 6.28e-5 / 233 [8.64e-6 / 283] (2.2e-6) | 8.48e-6 / 202 [1.81e-5 / 271] (2.2e-6) |

- **The DG paper's own claim reproduces:** FV and DG ROMs reach similar accuracy, but DG needs more hyper-reduced nodes.
- **Burgers is where the paper's hyper-reduction setting loses accuracy:** the hyper-reduced error is 3–50× the unreduced error. The paper's Table 2 errors are close to the unreduced ones, but it uses fewer nodes than we do in several cells, so its setting must differ somewhere we can't see. Experiment B (§5) identifies the cause.

---

## 5. Experiments: test basis and hyper-reduction (`results/dg/play/`)

Case: Burgers, DG p = 3, 256 elements, ε = 1e-2, u₀ = 0.5 − sin πx, T = 1 (the setting of the paper's Fig. 1). Columns in `log.txt`:
- **cond(M_t):** condition number of the hyper-reduced test mass matrix;
- **mass_err:** error of the hyper-reduced mass matrix;
- **thm4:** relative ‖V(I)ᵀQ̄x(I) − VᵀQx‖ over x in range(V) and snapshots;
- **conv_err:** error of the hyper-reduced convective term vs the unreduced ROM, over the snapshots;
- **HR err / ROM err:** final-time errors with and without hyper-reduction.

### A. Test basis (paper Fig. 1 → `fig1_test_basis.png`)

**Ideal hyper-reduction** (all 1024 nodes; offline check only):

| enrichment | thm4 | conv_err |
|---|---|---|
| none | 1.8e-3 | 9.1e-2 |
| QV | 3.2e-4 | 3.5e-2 |
| **M⁻¹QᵀV** | **1e-15** | **8e-15** |

Theorem 4 holds exactly, and only for the DG enrichment.

**Real hyper-reduction** with the paper's V∘V target, N = 30 (unreduced ROM error 4.2e-4):

| enrichment | HR error | cond(M_t) |
|---|---|---|
| none | 2.4e-2 | 1.6 |
| QV | 2.7e-1 | 1.5e6 |
| **M⁻¹QᵀV** | **3.7e-3** | 4.3e2 |
| both | 9.4e-2 | 1.2e6 |

The paper's improvement from QV to M⁻¹QᵀV reproduces (70× better). The result is still about 9× off the unreduced ROM; see B.

- At **N = 20** the paper's setting fails outright. The test basis has dimension 41 = 1 + 20 + 20, but the V∘V target selects only 40 nodes, so M_t is singular (cond 3e16) and the run blows up. The authors' stacked stabilization does not repair it.

### B. Hyper-reduction choices. HR error, with node counts in parentheses

| | N = 20 (ROM 5.27e-3) | N = 30 (ROM 4.23e-4) | N = 40 (ROM 3.14e-5) |
|---|---|---|---|
| paper: V∘V, energy trim, stacked | **fails** (40) | 3.7e-3 (92) | 1.1e-4 (170) |
| V∘V, abs trim, stacked | 3.4e-2 (49) | 4.7e-4 (126) | 4.3e-5 (213) |
| V∘V, union stabilization | 1.0e-2 (56) | 3.7e-3 (92) | 1.1e-4 (170) |
| V∘V, no stabilization | 8.7e-1 (41) | 3.7e-3 (92) | 1.1e-4 (170) |
| **V_t∘V_t target** | **5.28e-3 (151)** | **4.23e-4 (251)** | **3.14e-5 (298)** |
| Carathéodory, V∘V + stacked | 1.1e-2 (41) | 6.6e-4 (92) | 6.9e-5 (170) |
| **Carathéodory, V_t∘V_t** | **5.28e-3 (159)** | **4.23e-4 (251)** | **3.14e-5 (298)** |

### C. Error vs number of nodes, N = 30 (`error_vs_nodes.png`)
I swept the cubature tolerance from 10× to 0.001× the paper's tol:
- **V∘V target:** 64 → 175 nodes, error 7e-3 → 4.24e-4. It reaches the unreduced ROM only at about 175 nodes.
- **V_t∘V_t target:** already exact (4.26e-4) at its smallest set, 230 nodes.

### What the experiments say
1. **The paper's test basis is right, and it is necessary.** Under ideal hyper-reduction it makes the hyper-reduced flux term exact (Theorem 4). QV is wrong for DG for a structural reason (moments vs nodal values, §2).
2. **Theorem 4 assumes the quadrature reproduces the M-inner product on V_t.** The V∘V target only guarantees it on V, so in practice the remaining hyper-reduction error is the test-mass-matrix error. This is the same mechanism as in our FV Burgers finding.
   - Evidence: cond(M_t) and thm4 move together with the error, and the V_t∘V_t target, where cond(M_t) = 1, gives HR error = ROM error to 3–4 digits at every N.
3. **Two ways to make hyper-reduction as accurate as the unreduced ROM:**
   - target V_t∘V_t, which costs 1.5–4× the nodes but is robust with no stabilization needed;
   - keep V∘V and tighten its tolerance (~0.001× tol). That gives similar node counts.
   
   The paper's default tolerance (the POD energy) is simply too loose for the V∘V target on this problem.
4. **Stabilization patches symptoms.** "Stacked" (eigenvalue threshold 1e-8) misses moderately ill-conditioned cases, e.g. p = 7, N = 30 with cond 3e7 and error 2e-2. "Union" fixes the singular N = 20 case but doesn't improve accuracy otherwise.
5. **Carathéodory pruning:**
   - It gives exact moments with exactly dim(target) nodes, the same count as the greedy algorithm.
   - With V∘V it is noticeably more accurate than the greedy at the same node count: at N = 30, 6.6e-4 vs 3.7e-3. The greedy only matches moments to tol; Carathéodory matches them to round-off.
   - With V_t∘V_t both are equivalent.
   - Cost: offline only, about seconds for 1024 nodes.
6. **Node locations** (`node_locations.png`) cluster around the shock and the ramp. The V_t∘V_t sets are denser everywhere.

---

## 6. Ideas for combining with our VMS / PINN closures

These are hypotheses, not results.

1. **A learned closure that is entropy stable by construction.** Eq. (43) shows that any term of the form
   - d = Vᵀ QᵀM⁻¹ diag(ν) H Q V v_N, with ν ≥ 0 pointwise and H = ∂u/∂v(ũ),

   satisfies v_Nᵀ d ≥ 0. So a PINN that outputs a *nonnegative eddy-viscosity field ν(x; u_N)*, e.g. through softplus, yields an entropy-dissipative closure regardless of training, with no penalty term needed. This is a VMS-style eddy viscosity built on the paper's symmetrized viscosity. Hyper-reduction could reuse the same nodes. `_Viscosity("es")` is the ν ≡ 1 version.
2. **The test basis as a closure space.** The enrichment M⁻¹QᵀV is exactly "the derivative directions of the resolved modes that the flux term needs". In VMS language it is a natural candidate for the *unresolved-but-coupled* scales: closure terms could be sought in span(V_t) ⊖ span(V).
3. **Learning the hyper-reduction error.** Experiments B and C show the HR error is a structured, offline-computable defect: conv_err against the unreduced ROM over the snapshots. A closure trained on it would let a cheap V∘V quadrature (90 nodes) reach the accuracy of 250 nodes. Since the closure enters the entropy balance only through v_Nᵀ M_N C, the admissibility condition from the FV notes applies unchanged.
4. **Entropy-projection error as an indicator.** u − ũ is computable online at the HR nodes. It is small when the basis represents v(u) well (Fig. 5 of the paper) and could serve as a closure input or trust signal.
5. **Cost.** O(N_s²) flux evaluations make the HR-ROM slower per step than the FOM at large N_s. A closure that allows smaller N (or N_s) is worth more here than in standard ROMs. The relaxed ROM time step (CFL relaxation, the paper's Table 4) is the other lever.

---

## 7. Status
- **Done:**
  - DG FOM and ROMs, hyper-reduction with all the options above, and tests (`tests/test_dg_invariants.py`);
  - Tables 1–3 of Sec. 6.2;
  - the test-basis and hyper-reduction experiments.
- **Left out on purpose:** wall boundary conditions (Sec. 3.6, experiments 6.3.1–6.3.2).
