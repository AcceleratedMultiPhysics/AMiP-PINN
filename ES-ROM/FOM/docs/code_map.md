# Chan (2020) ES-ROM: paper → code map, and where we differ

Reference: J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws*, JCP 423 (2020) 109789 (`papers/Chan2020.pdf`). Section, equation and theorem numbers below are the paper's. Line numbers refer to the code as of 2026-10-24.

**Contents**
1. [Pipeline overview](#1-pipeline-overview)
2. [Paper → code map](#2-paper--code-map)
3. [Changes to the method](#3-changes-to-the-method) (these affect results)
4. [Choices where the paper is ambiguous or silent](#4-choices-where-the-paper-is-ambiguous-or-silent)
5. [What is unchanged](#5-what-is-unchanged)
6. [Questions for Jesse](#6-questions-for-jesse)

---

## 1. Pipeline overview

```
FOM (Sec. 2, 5)  →  snapshots of u and v(u)  →  POD basis V (Sec. 3)
      →  test basis Vt, range = [1, V, QV]   (Sec. 4.1.1)
      →  points I and weights w               (Sec. 4.2–4.3, Algorithm 1)
      →  Q_t = P_tᵀ (VtᵀQVt) P_t               (Sec. 4.1, eqs. 13–16)
      →  hyper-reduced ROM, eq. (19)/(27), viscosity (20)/(22)/(23)
```

The **offline** steps work on FOM-sized objects (K cells). The **online** ROM only touches N modes and N_s sampled points.

| File | Role |
|---|---|
| `esrom/physics.py` | PDE-specific pieces: entropy, entropy variables, EC flux, ∂u/∂v |
| `esrom/operators.py` | FV matrices Q, D, K |
| `esrom/fom.py` | full order model, wall flux, time stepping |
| `esrom/rom.py` | bases, unreduced ROM (Theorem 2), hyper-reduced ROM (Theorems 3/4), offline builder |
| `esrom/hyperreduction.py` | empirical cubature (Algorithm 1), target compression, stabilizing points |
| `scripts/*.py` | experiment set-up and figures only, no method code |
| `tests/test_invariants.py` | the structural properties the proofs rely on |

---

## 2. Paper → code map

### 2.1 Full order model (Sec. 2, 5, 7)

| Paper | Math | Code |
|---|---|---|
| Entropy (Sec. 7) | S = −ρs, s = log(p/ρ^γ) | [`Euler1D.entropy`](../esrom/physics.py#L76) |
| Entropy variables and inverse (Sec. 7) | v(u), u(v) | [`entropy_vars`](../esrom/physics.py#L81), [`cons_vars`](../esrom/physics.py#L88) |
| Entropy potential | ψ = (γ−1)ρu (derived; not printed in the paper) | [`psi`](../esrom/physics.py#L95) |
| EC flux, eq. (3); Chandrashekar (Sec. 7) | f_S with {{ρ}}_log, p_avg, E_avg | [`Euler1D.flux_ec`](../esrom/physics.py#L105), [`logmean`](../esrom/physics.py#L14) |
| Burgers EC flux | (u_L² + u_L u_R + u_R²)/6 | [`Burgers.flux_ec`](../esrom/physics.py#L45) |
| ∂u/∂v (Sec. 4.6, for H in (22)) | Euler symmetrizer | [`dudv`](../esrom/physics.py#L128) |
| Q, eq. (5) | periodic, ½ central difference, Q = −Qᵀ, Q1 = 0 | [`Q_periodic`](../esrom/operators.py#L11) |
| Q, eq. (24) | SBP, Q + Qᵀ = B = diag(−1, 0, …, 0, 1) | [`Q_sbp`](../esrom/operators.py#L19) |
| D (Sec. 4.6), K (eq. 8) | edge differences; Laplacian | [`D_matrix`](../esrom/operators.py#L34), [`K_matrix`](../esrom/operators.py#L43) |
| FOM, eqs. (6), (25), (8) | Δx du/dt + 2(Q∘F)1 + B(f* − f(u)) + εKu = 0 | [`FOM1D.conv`](../esrom/fom.py#L72), [`FOM1D.rhs`](../esrom/fom.py#L87) |
| Wall BC (Sec. 7.1) | mirror state ρ⁺ = ρ, u⁺ = −u, p⁺ = p; f* = f_S(u⁺, u) + LxF | [`wall_fluxes`](../esrom/fom.py#L47), [`mirror`](../esrom/physics.py#L71) |
| Theorem 1 check | vᵀ[2(Q∘F)1 + B(f*_EC − f)] = 0 | [`FOM1D.entropy_residual`](../esrom/fom.py#L96) |
| Time integrator (Sec. 7) | 5-stage, 4th-order low-storage RK (Carpenter–Kennedy) | [`rk45_step`](../esrom/fom.py#L25) |

**Note on the FOM operator.** The code never forms the dense F. Q is tridiagonal, so row i of 2(Q∘F)1 is f_S(u_i, u_{i+1}) − f_S(u_{i−1}, u_i). At a wall, the SBP corner entry −½f(u₀) and the term B(f* − f) combine to F₀₁ − f*_L. That is why `conv` simply places f* on the two end faces. It is algebraically identical to eq. (25).

### 2.2 Entropy conservative ROM, no hyper-reduction (Sec. 3, Theorem 2)

| Paper | Math | Code |
|---|---|---|
| Snapshots of u **and** v(u) (Sec. 3 remark) | SVD of [u snapshots, v(u) snapshots]; all components side by side, one shared V | [`snapshot_svd`](../esrom/rom.py#L19), [`pod_basis`](../esrom/rom.py#L32) |
| Projected entropy variables | v_N = V†v(Vu_N), V† = (VᵀV)⁻¹Vᵀ | [`GalerkinROM._states`](../esrom/rom.py#L79) |
| Entropy projection | ũ = u(V v_N) | [`GalerkinROM._states`](../esrom/rom.py#L80) |
| Eq. (12) | Δx VᵀV du_N/dt + 2Vᵀ(Q∘F(ũ))1 = 0 | [`GalerkinROM.rhs`](../esrom/rom.py#L82) |

`rhs` evaluates the FOM operator at ũ and then applies Galerkin projection. The only difference from a plain POD–Galerkin ROM is that the flux sees ũ instead of Vu_N. Passing `entropy_projection=False` turns it into the plain ROM.

### 2.3 Hyper-reduced ROM (Sec. 4.1, 4.4, 5; Theorems 3, 4)

| Paper | Math | Code |
|---|---|---|
| Test basis (Sec. 4.1.1) | range(V_t) = range([1, V, QV]) | [`test_basis`](../esrom/rom.py#L50), [`orth`](../esrom/rom.py#L38) |
| Eq. (13) | Q̂_t = V_tᵀ Q V_t | [rom.py:109](../esrom/rom.py#L109) |
| Eq. (14) | P_t = (V_t(I)ᵀW V_t(I))⁻¹ V_t(I)ᵀW | [rom.py:108](../esrom/rom.py#L108) |
| Eq. (16), Lemma 1 | Q_t = P_tᵀ Q̂_t P_t | [rom.py:110](../esrom/rom.py#L110) |
| Eq. (19) | M_N = V(I)ᵀWV(I), P = M_N⁻¹V(I)ᵀW | [rom.py:102](../esrom/rom.py#L102), [rom.py:104](../esrom/rom.py#L104) |
| Eq. (19) | v_N = P v(V(I)u_N), ũ = u(V_h v_N) | [`HyperROM._states`](../esrom/rom.py#L139) |
| Eq. (19) | F_ij = f_S(ũ_i, ũ_j) (dense N_s×N_s); 2V(I)ᵀ(Q_t∘F)1 | [`HyperROM._conv`](../esrom/rom.py#L144) |
| Sec. 5, hybridized SBP | E = V_bt P_t; Q_h = ½[[Q_t − Q_tᵀ, EᵀB], [−BE, B]]; V_h = [V(I); V_b] | [rom.py:114–121](../esrom/rom.py#L114) |
| Eq. (27) boundary term | V_bᵀ B (f* − f(ũ_b)), with ũ_b = u(V_b v_N) | [rom.py:147–151](../esrom/rom.py#L147) |
| Offline recipe (Sec. 4.4) | steps 1–4 | [`build_hyper_rom`](../esrom/rom.py#L195) |

### 2.4 Points and weights (Sec. 4.2, 4.3)

| Paper | Math | Code |
|---|---|---|
| tol (Sec. 4.2) | √(Σ_{j>N} σ_j² / Σ σ_j²) | [`energy_tol`](../esrom/hyperreduction.py#L6) |
| Target (17) and compression (18) | span{V(:,i)∘V(:,j)}, keep k modes with E_k ≤ tol | [`compress_products`](../esrom/hyperreduction.py#L18), [`_truncate`](../esrom/hyperreduction.py#L36) |
| Algorithm 1 | greedy: most positively parallel row → LS → NNLS if any w ≤ 0 | [`greedy_cubature`](../esrom/hyperreduction.py#L42) |
| Stabilizing points (Sec. 4.3) | small eigenvectors z_j → Z = V_t z → target Z∘Z → joint weights | [`stabilize`](../esrom/hyperreduction.py#L73) |

### 2.5 Entropy dissipation (Sec. 4.6)

| Paper | Math | Code |
|---|---|---|
| Viscous points | basis V_D of DV (SVD), then Algorithm 1 on products of V_D, target weights Δx per edge | [`build_hyper_rom`](../esrom/rom.py#L217) |
| Eq. (22) | VᵀD(I_D)ᵀ W_D H D(I_D) V v_N, H = ∂u/∂v at the average of ũ_i, ũ_{i+1} | [`dissipation_term`](../esrom/rom.py#L156) |
| Eq. (20) | VᵀD(I_D)ᵀ W_D D(I_D) ũ | [rom.py:162](../esrom/rom.py#L162) |
| Eq. (23) | VᵀKV P ũ (not provably dissipative) | [rom.py:166](../esrom/rom.py#L166) |

### 2.6 The quantities plotted in the paper

| Paper | Quantity | Code |
|---|---|---|
| Fig. 4 | v_Nᵀ[2V_hᵀ(Q_h∘F)1 + V_bᵀB(f*_EC − f)] (should be ~1e-14) | [`HyperROM.entropy_residual`](../esrom/rom.py#L178) |
| Fig. 5 | v_Nᵀd(u_N) ≥ 0 | [`HyperROM.entropy_dissipation`](../esrom/rom.py#L183) |
| Theorem 3 | sampled entropy 1ᵀW S(V(I)u_N) | [`HyperROM.sampled_entropy`](../esrom/rom.py#L188) |
| Fig. 6 | entropy and relative L2 error on the FOM grid | [`scripts/euler_1d.py`](../scripts/euler_1d.py) |

---

## 3. Changes to the method

These change how the method behaves. For each one: what the paper says, what went wrong when implemented literally, what the code does now, and how to switch back where that is possible.

### 3.1 Stabilizing points (Sec. 4.3): the largest change

**Paper.** After the greedy cubature for V∘V, look at the hyper-reduced test mass matrix M_t = V_t(I)ᵀWV_t(I). If it is ill-conditioned, take the eigenvectors z_j of its small eigenvalues and form Z = [V_t z_1, …]. Run Algorithm 1 on the target Z∘Z to get points I_Z. Then recompute all weights on I ∪ I_Z by NNLS on

  ½‖V_target(I)ᵀc − b‖² + α_Z‖Z_target(I)ᵀc − d‖², with α_Z = 1e-2.

The paper does not give the condition-number tolerance or define which eigenvalues count as "small".

**What happened when implemented literally (Euler, K = 2500):**
- **NNLS drops points.** An NNLS solution has at most as many nonzero weights as there are equations, so the re-solve discards points. That includes points just added for stability and original ones.
  - N = 75: 191 cubature + 21 stabilizing points became 160 points after the NNLS re-solve. cond(M_t) ≈ 9e8, and the ROM produced non-finite values (blow-up).
  - N = 25: the stabilizing point chosen by Algorithm 1 was index 0. That cell was useless because of the tie problem in §3.2. cond(M_t) stayed at 6.7e6, and the entropy residual was only 1.5e-10 instead of ~1e-14.
- **α_Z = 1e-2 spoils the mass matrix.** After fixing the pruning (below) and keeping α_Z = 1e-2 with an eigenvalue threshold of 1e-2, N = 75 added 87 points. It was stable (cond 226), but the mass-matrix error rose to 6e-3 and the ROM error was **8.3e-2**, against **1.2e-3** without hyper-reduction.

**What the code does now** ([`stabilize`](../esrom/hyperreduction.py#L73)):
1. Stabilizing points are chosen only among points **not already in I** (the `exclude` argument of `greedy_cubature`).
2. Joint weights use **min-norm least squares**. If any weight ends up ≤ w_min, the code switches to **bounded least squares** (w ≥ w_min = 0.1Δx, `scipy.optimize.lsq_linear`). The point set is never pruned.
3. The step **repeats** (up to 10 times) until M_t has no eigenvalue below `eig_tol`. V_t is orthonormal in the Δx inner product, so the exact test mass matrix is I and `eig_tol` is an absolute threshold.
4. **Defaults:** `eig_tol = 1e-3`, `alpha_Z = 1e-4`, `w_min_frac = 0.1`.

**How the defaults were chosen.** I ran the N = 75 Euler ROM with four settings (error at T = 0.75; unreduced ROM = 1.19e-3):

| α_Z | eig_tol | w_min/Δx | points (added) | cond(M_t) | error |
|---|---|---|---|---|---|
| 1e-2 | 1e-2 | 0.01 | 278 (+87) | 2.3e2 | 8.4e-2 |
| 1e-4 | 1e-2 | 0.1 | 235 (+44) | 3.1e2 | 3.3e-2 |
| 1e-3 | 1e-3 | 0.01 | 195 (+4) | 1.8e3 | 1.7e-3 |
| **1e-4** | **1e-3** | **0.1** | **198 (+7)** | **2.7e3** | **1.2e-3** |

The pattern: aggressive stabilization (many points, heavily weighted Z) makes M_t look better but degrades the mass matrix M_N, and accuracy follows the mass matrix. A few lightly weighted points are enough.

**Caveat.** These defaults were tuned on one case (Euler, N = 75). At N = 25 the setting adds 1 point; at N = 125 it adds none. Other problems may need other values.

**Reverting.** `build_hyper_rom(..., eig_tol=1e-2, alpha_Z=1e-2)` restores the paper's parameters. The NNLS re-solve and the "points may already be in I" behaviour are not switchable. Restoring them means changing `stabilize`.

### 3.2 Tie-breaking in Algorithm 1

**Paper.** Pick the row of V_target most positively parallel to the residual, using rows normalized to unit length.

**Problem.** When the target has a single column (one small eigenvector, so Z∘Z is one column), every normalized row is ±1. All candidates then tie, and `argmax` returns the first index, 0. At N = 25 this mattered: the weak direction of M_t is a spike concentrated in the **single last cell** next to the right wall (|Z| = 1/√Δx there). The algorithm was choosing the left-most cell instead.

**Now.** Ties (within 1e-12) go to the row with the largest norm, i.e. where the target function actually lives ([hyperreduction.py:59–60](../esrom/hyperreduction.py#L59)). At N = 25 this took cond(M_t) from 6.1e6 to 349, and the entropy residual from 4e-11 to 3e-15. With multi-column targets ties essentially never occur, so the normal case is unchanged.

### 3.3 Optional cubature target V_t∘V_t

This is an addition; the paper's target remains the default.

**Paper.** Target the products V∘V (eq. 17), so the reduced mass matrix ΔxVᵀV is accurate. Sec. 4.3 reports that targeting ΔxVᵀV_t or ΔxV_tᵀV_t gave many more points "without any significant improvement in accuracy".

**What we found.**
- **Burgers.** The paper target leaves M_t badly integrated on the QV directions: eigenvalues range from 0.02 to 1.95 when the exact value is 1. That error dominates the ROM. Targeting V_t∘V_t (`target="test_mass"`) removes it.

  Relative L2 error at T = 1:

  | N | unreduced | paper target | V_t∘V_t target | points (paper / V_t∘V_t) |
  |---|---|---|---|---|
  | 4 | 5.4e-2 | 1.3e-1 | 5.4e-2 | 9 / 14 |
  | 8 | 1.8e-3 | 2.3e-1 | 1.8e-3 | 23 / 35 |
  | 12 | 6.9e-5 | 7.8e-3 | 6.9e-5 | 36 / 54 |

  The paper only uses Burgers to motivate [1, V, QV]; it never runs the fixed Burgers ROM. So this does not contradict any reported result.
- **Euler.** Both targets give the same error once stabilization works, confirming the paper's statement. V_t∘V_t costs 1.9–2.4× the points and 2–7× the online time (11 → 23 s, 57 → 410 s, 211 → 772 s at N = 25, 75, 125; the N = 75 paper-target run is cheap because it needs only 7 stabilizing points).

**Default.** `target="mass"` (the paper's choice).

---

## 4. Choices where the paper is ambiguous or silent

### 4.1 Snapshot window (Sec. 7.1)
- **Paper:** "run the full order model until final time T = .7 and store the solution at 2801 time-steps". Figs. 3–6 then show ROM results up to **T = 0.75**.
- **Problem:** with snapshots to 0.7, the ROM extrapolates between 0.7 and 0.75, and the error jumps right at t = 0.7. On the K = 500 grid at N = 30, the error at T = 0.75 is 3.1e-2 with snapshots to 0.7, against 7.6e-3 with snapshots to 0.75 (`results/euler_quick_tsnap0.7/fig6_error_entropy.png`). The paper's Fig. 6 shows no such jump, which suggests its snapshots covered [0, 0.75].
- **Chosen:** snapshots on [0, 0.75], every 10th step (301 snapshots). `--t-snap 0.7` gives the literal reading.
- This may partly explain why our point counts are 20–50% above Table 1.

### 4.2 Time step and CFL (Sec. 7.1)
- **Paper:** "CFL of .75" and 2801 stored steps to T = 0.7, which implies dt = 0.7/2800 = 2.5e-4.
- **Chosen:** dt = 2.5e-4 exactly, so the step count matches. With the wave speed |u| + c this is CFL ≈ 0.50, so the paper must define CFL with a different wave-speed estimate or a different h. The ROMs use the same dt as the FOM, as the paper does.

### 4.3 Making 1 part of the basis (Theorem 2)
- **Paper:** conservation of the global averages needs 1 ∈ range(V), but it does not say how to ensure that.
- **Chosen:** V = orthonormalize([1, first N−1 POD modes]), so N counts the constant ([`pod_basis`](../esrom/rom.py#L32)).

### 4.4 One basis for all components
- **Paper:** "a reduced basis for each component", and Table 1 is consistent with a single N-column basis.
- **Chosen:** one shared V. It comes from the SVD of all components of u and v(u) placed side by side (6 blocks for Euler, 2 for Burgers), without per-component scaling.

### 4.5 Compressing the product target (eq. 18)
- **Paper:** SVD of the matrix of products V(:,i)∘V(:,j).
- **Chosen:** eigendecomposition of (VVᵀ)∘(VVᵀ). Its eigenvectors are the left singular vectors of the matrix of all (i,j) products. It spans the same space, and it avoids building a 2500 × 31,000 matrix at N = 125 (seconds instead of minutes). Cross pairs (i ≠ j) count twice compared with an i ≤ j list, which shifts the singular values slightly and so can move k by a few modes.

### 4.6 Viscosity scaling (eq. 8 vs Sec. 4.6)
- **Paper:** eq. (8) has K = (1/Δx)·tridiag(−1, 2, −1). Sec. 4.6 defines D with a factor 1/Δx and says K = DᵀD, which would give 1/Δx².
- **Chosen:** K = Δx·DᵀD = (1/Δx)·tridiag, consistent with eq. (8), with D as in Sec. 4.6. The viscous hyper-reduction then targets weights Δx per edge, so W_D approximates Δx·I.

### 4.7 Test basis construction
- **Paper:** says nothing about how V_t is built numerically.
- **Chosen:** columns of [1, V, QV] are normalized and then orthonormalized in the Δx inner product (SVD with rank truncation 1e-10), so the exact test mass matrix is I.
- **Dropping Q·1.** Because 1 is a basis vector, QV contains Q·1 ≈ 1e-17, i.e. pure round-off. Normalizing it added a random noise direction to V_t and made M_t singular: cond 1.5e16 on Burgers at N = 20. Columns with norm below 1e-12 of the largest are dropped first ([`orth`](../esrom/rom.py#L38)).

### 4.8 Which quantity Fig. 4 plots
- **Paper:** Fig. 4 shows |v_NᵀV_hᵀ(Q_h∘F)1| ≈ 1e-14.
- **Issue:** with walls, the volume term alone is not zero. It equals −1ᵀBψ(ũ_b), which balances the entropy-conservative boundary flux.
- **Chosen:** the plotted residual includes V_bᵀB(f*_EC − f(ũ_b)), i.e. everything the proof of Theorem 4 shows vanishes (LxF penalty excluded). For periodic problems the boundary term is absent, and this is exactly the paper's quantity.

### 4.9 Log-mean evaluation
- **Paper:** use the stable algorithm of Ismail & Roe.
- **Chosen:** the same algorithm, but the series branch is used only when u = ((a−b)/(a+b))² < 1e-4, not < 1e-2. With the usual 1e-2 the truncated series is accurate only to about 1e-9, which would limit entropy conservation to about 1e-9. With 1e-4 the Tadmor shuffle condition holds to 1e-14, including for nearly equal states (tested).

### 4.10 Wall flux details
- **Chosen:** the LxF wave speed is |u| + c at the boundary cell; the mirror state has the same value. The mirror flux f_S(u⁺, u) reduces exactly to (0, p, 0), which is entropy conservative.

### 4.11 ROM initial condition
- **Paper:** not specified.
- **Chosen:** full-grid L2 projection, u_N(0) = V†u₀.

### 4.12 Burgers set-up
- **Paper:** gives only the initial condition u₀ = −sin(πx) on a periodic domain.
- **Chosen:** [−1, 1], K = 1000, ε = 2e-3, T = 1, dt = 5e-4, snapshots every 5 steps. The POD singular values decay quickly, so the paper's tol rule gives tol ≈ 2e-15 at N = 40, which selects every point. We therefore use N = 4, 8, 12.

### 4.13 Product-space target for the viscous points
- **Paper:** "computed using Algorithm 1 with dimensionality reduction".
- **Chosen:** V_D = left singular vectors of DV (rank N−1, since D1 = 0), target V_D∘V_D compressed with the same tol, target weights Δx per edge.

---

## 5. What is unchanged

These are implemented exactly as in the paper, and the property the proofs need is checked in `tests/test_invariants.py` where possible:
- **Flux differencing:** the FV FOM (eqs. 6, 8, 25), the Chandrashekar flux including E_avg (checked: consistency, symmetry, and the shuffle condition to 1e-12).
- **ROM structure:**
  - entropy projection and Theorem 2;
  - compress-and-project Q_t (eqs. 13–16; checked: Q_t = −Q_tᵀ periodic, generalized SBP with walls, Q_t1 = 0, P_tV_t(I) = I);
  - hybridized Q_h (checked: Q_h + Q_hᵀ = B_h, Q_h1 = 0);
  - test basis [1, V, QV].
- **Hyper-reduction:** Algorithm 1 (apart from the tie-break and the `exclude` option), the tol formula, targets (17)/(18), the hyper-reduced mass matrix in (19).
- **Dissipation and boundaries:** viscosity treatments (20), (22), (23); mirror-state wall flux with LxF; Euler entropy variables and the inverse map.

## 6. Questions for Jesse

1. Snapshots: [0, 0.7] or [0, 0.75] in Sec. 7.1? And which wave speed defines CFL 0.75?
2. Stabilizing points: what condition-number tolerance was used, and did the NNLS re-solve drop points for him? Our literal implementation pruned 212 → 160 at N = 75 and blew up.
3. Was V∘V vs V_t∘V_t ever compared on Burgers? For us the paper target is the dominant error there.
4. How was 1 ∈ range(V) enforced, and how was the product-space compression done (i ≤ j or all pairs)?
