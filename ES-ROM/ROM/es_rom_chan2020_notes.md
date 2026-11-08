# Entropy Stable ROM (Chan 2020) — Paper Summary & Discussion Notes

Context file for a Claude Code session. Goal: implement the entropy stable reduced order model (ES-ROM) from

> J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws*, J. Comput. Phys. 423 (2020) 109789.

and later extend it with a learned closure. Equations below are transcribed from the paper; verify against the PDF before relying on a formula in code.

---

## 0. Research direction (why this code exists)

- Long-term direction: **ES-ROMs + closures.** Chan's ES-ROM guarantees stability but explicitly does **not** address accuracy for transport-dominated problems (slow Kolmogorov $n$-width). A closure model targets that accuracy gap.
- Prior work of mine: **C-PINN-ROM** (Kaya, Koc, Aygun, Ata, Karakus, arXiv:2603.01998) — VMS-based ROM closure learned with a PINN (JAX), for incompressible NS / 2D Burgers on a Scott–Vogelius FE FOM. That FOM does **not** have the flux-differencing structure needed for Chan's guarantee, so the ES-ROM needs its own FOM.
- Follow-up papers in this line:
  - Qu, Narayan, Chan, *Entropy stable ROM ... using DG methods*, JCP 557 (2026) 114830 — generalizes this paper from FV to DG FOMs. Code: `github.com/rayqu1126/ES_DG_ROM`.
  - Chan, *An artificial viscosity approach to high order entropy stable DG methods*, JCP 543 (2025) 114380 — parameter-free entropy correction artificial viscosity.

---

## 1. One-paragraph summary

Projection-based ROMs of nonlinear conservation laws do not inherit stability from the FOM. This paper builds a hyper-reduced ROM that is globally conservative and satisfies a **semi-discrete entropy inequality, independent of the basis and parameters**. Three ingredients: (1) an entropy conservative finite volume FOM written in **flux-differencing** form, whose entropy proof depends only on two matrix properties of $Q$; (2) a modified Galerkin projection using the **entropy projection** so the reduced flux term still conserves entropy; (3) a two-step **"compress and project" hyper-reduction** that keeps both required matrix properties. Entropy dissipation is added via provably dissipative hyper-reduced viscosity.

---

## 2. Notation and the full order model (Sec. 2)

Conservation law with convex entropy $S(u)$, entropy variables $v = \partial S/\partial u$, entropy potential $\psi$.

**Entropy conservative two-point flux** $f_S(u_L,u_R)$:
- consistency: $f_S(u,u) = f(u)$
- symmetry: $f_S(u_L,u_R) = f_S(u_R,u_L)$
- entropy conservation (Tadmor shuffle): $(v_L - v_R)^T f_S(u_L,u_R) = \psi(u_L) - \psi(u_R)$

**FOM (1D periodic FV, $K$ cells):**
$$\Delta x \frac{du_h}{dt} + 2(Q\circ F)\mathbf{1} = 0, \qquad F_{ij} = f_S\big((u_h)_i,(u_h)_j\big)$$
- $Q$ = periodic central difference with factor $1/2$: tridiagonal-periodic, $+\tfrac12$ on superdiagonal, $-\tfrac12$ on subdiagonal, wrap-around corners.
- $\circ$ = Hadamard product. Systems handled in Kronecker fashion (one $F$ block per flux component).

**The two properties everything depends on:**
$$\boxed{Q = -Q^T, \qquad Q\mathbf{1} = 0}$$
Theorem 1: with these, $\Delta x\,\mathbf{1}^T \tfrac{d}{dt}S(u_h) = 0$.

**Viscosity (Sec. 2.1):** add $\epsilon K u_h$, $K$ = Laplacian (Neumann in the paper, periodic also fine), $K = D^TD$ with $D$ the $(K-1)\times K$ difference matrix. Then $\Delta x\,\mathbf{1}^T \tfrac{dS}{dt} = -\epsilon\, v_h^T K u_h \le 0$, by the mean value theorem and positive definiteness of $\partial u/\partial v$.

---

## 3. Section 3 — entropy conservative ROM (plain-language version)

**Setup.** Approximate $u_h \approx V u_N$ ($V$: $K \times N$, columns are POD modes = FOM grid functions). Galerkin projection gives
$$\Delta x\, V^TV \frac{du_N}{dt} + 2V^T(Q\circ F)\mathbf{1} = 0.$$
This standard POD–Galerkin ROM is **not** entropy conservative.

**Why it breaks.** The entropy proof tests with $v$. In the ROM you may only test with vectors in $\text{range}(V)$, but $v(Vu_N)$ is a nonlinear function of $Vu_N$ and generally lies outside that span.

**Fix, step 1 — test with the projection.** Use $v_N = V^\dagger v(Vu_N)$, $V^\dagger = (V^TV)^{-1}V^T$, grid values $\tilde v = VV^\dagger v(Vu_N)$. For the time derivative this works exactly (Eq. 11), because $du_N/dt$ lives in the span.

**Remaining problem.** The shuffle condition needs $\tilde v_i$ to be exactly the entropy variables of the states fed into $f_S$. With $F$ evaluated at $Vu_N$ they don't match.

**Fix, step 2 — entropy projection.** Feed the flux the conservative variables recovered from the projected entropy variables:
$$\tilde u = u\big(VV^\dagger v(Vu_N)\big), \qquad F_{ij} = f_S(\tilde u_i, \tilde u_j).$$

**The chain:**
$$u_N \xrightarrow{V} Vu_N \xrightarrow{v(\cdot)} v(Vu_N) \xrightarrow{VV^\dagger} \tilde v \xrightarrow{u(\cdot)} \tilde u \to f_S(\tilde u_i,\tilde u_j)$$

**Theorem 2:** $\Delta x\,\mathbf{1}^T \tfrac{d}{dt}S(Vu_N) = 0$. If $\mathbf{1} \in \text{range}(V)$, global averages of conserved variables are also conserved.

**Practical remark:** accuracy needs $\tilde u \approx u$, i.e. the basis must represent the entropy variables well. **Build POD from snapshots of both $u$ and $v(u)$.**

This ROM is not yet cheap: $F$ still lives on all $K$ points. Section 4 fixes that.

---

## 4. Section 4.1 — two-step hyper-reduction (plain-language version)

**Goal.** Evaluate nonlinear terms only at sampled points $\mathcal{I}$ ($N_s \ll K$) with positive weights $W = \text{diag}(w)$ — a reduced quadrature. Need a small $Q_s$ that is **both** skew-symmetric **and** has zero row sums. Standard hyper-reduction gives one or the other, never both.

**Step 1 — compress.** With a test basis $V_t$ (contains $V$ and $\mathbf{1}$):
$$\hat Q_t = V_t^T Q V_t.$$
Small "modal" operator (acts on coefficients). Skew-symmetric by the sandwich; kills constants because $\mathbf{1} \in \text{range}(V_t)$ and $Q\mathbf{1}=0$.

**Step 2 — project.** Map point values at $\mathcal{I}$ to test-basis coefficients with weighted least squares:
$$P_t = \big(V_t(\mathcal{I},:)^T W V_t(\mathcal{I},:)\big)^{-1} V_t(\mathcal{I},:)^T W,$$
then
$$Q_t = P_t^T \hat Q_t P_t.$$
Read right to left: points → coefficients → differentiate → back to points.

**Lemma 1:** if $\mathbf{1} \in \text{range}(V_t)$ and Assumption 1 holds, then $Q_t = -Q_t^T$ and $Q_t\mathbf{1} = 0$.
- Key property: $P_t$ reproduces test-basis functions exactly, $P_t V_t(\mathcal{I},:) c = c$.
- **Assumption 1:** the hyper-reduced test mass matrix $V_t(\mathcal{I},:)^T W V_t(\mathcal{I},:)$ is nonsingular.

**Price:** $Q_t$ is **dense** ($N_s \times N_s$) → $O(N_s^2)$ flux evaluations per RHS.

## 5. Section 4.1.1 — choice of test basis (plain-language version)

$V$ and $\mathbf{1}$ in $V_t$ are necessary but **not sufficient**.

**Burgers counterexample:** $u(x,0) = -\sin(\pi x)$ → stationary antisymmetric shock → antisymmetric POD modes, nearly symmetric derivatives $QV$ (Fig. 1). Then $V^TQV \approx 0$ (odd × even integrates to ~0) and the ROM is badly wrong.

**Intuition:** $\hat Q_t = V_t^TQV_t$ only measures the output of $Q$ along directions in $V_t$. If $QV$ points outside $V_t$, it's invisible.

**Fix:**
$$\text{range}(V_t) = \text{range}\big([\mathbf{1},\ V,\ QV]\big), \qquad \dim V_t \le 2N+1.$$
Without hyper-reduction this gives $Q_tV = QV$ exactly. Idea borrowed from LSPG ROMs.

**DG note (2026 paper):** for DG the mass matrix is not a scalar multiple of $I$, and the correct enrichment becomes $M_\Omega^{-1}Q_\Omega^T V$. For FV ($M = \Delta x\, I$) it reduces to $QV$.

---

## 6. Remaining Section 4 pieces (implementation-relevant)

**4.2 Empirical cubature (Algorithm 1, greedy, from Hernández et al. / Yano).**
- Input $V_{\text{target}}$, $w_{\text{target}}$, tol. Approximate $V_{\text{target}}^T w_{\text{target}} \approx V_{\text{target}}(\mathcal{I},:)^T w$.
- Loop while $\|r\|/\|b\| > \text{tol}$: pick index most positively parallel to residual (normalized rows), add to $\mathcal{I}$, solve least squares for $w$; if any $w \le 0$, redo with NNLS; update residual.
- $w_{\text{target}} = \Delta x\,\mathbf{1}$.
- Target space: products of modes, $\text{span}\{V(:,i)\circ V(:,j)\}$ — so the reduced mass matrix $\Delta x V^TV$ is accurate. Compress $V_{\text{target}}$ by SVD, keep leading $k$ vectors with energy residual $E_k \le \text{tol}$.
- $\text{tol} = \sqrt{\sum_{j>N}\sigma_j^2 / \sum_j \sigma_j^2}$ from snapshot singular values.
- Single fixed point set for the whole simulation.

**4.3 Conditioning — stabilizing points.** Targeting only $V\ast V$ can make the test mass matrix singular. Take eigenvectors $z_j$ of small eigenvalues, $Z = [V_tz_1, \ldots]$, run Algorithm 1 on $Z_{\text{target}}$ (products, no dimensionality reduction), take $\mathcal{I} = \mathcal{I}_V \cup \mathcal{I}_Z$, recompute weights by NNLS on the combined objective with $\alpha_Z = 10^{-2}$.

**4.4 Theorem 3 — hyper-reduced ROM:**
$$M_N\frac{du_N}{dt} + 2V(\mathcal{I},:)^T(Q_t\circ F)\mathbf{1} = 0,$$
$$M_N = V(\mathcal{I},:)^TWV(\mathcal{I},:),\quad P = M_N^{-1}V(\mathcal{I},:)^TW,\quad v_N = P\,v(V(\mathcal{I},:)u_N),$$
$$\tilde v = V(\mathcal{I},:)v_N,\quad \tilde u = u(\tilde v),\quad F_{ij} = f_S(\tilde u_i,\tilde u_j).$$
Conserves the sampled, weighted entropy $\mathbf{1}^TW\tfrac{d}{dt}S(V(\mathcal{I},:)u_N) = 0$. Paper reports noticeably larger errors if the exact mass matrix is used instead of the hyper-reduced one.

**Offline recipe (periodic):**
1. Basis $V$ from snapshots of conservative **and** entropy variables.
2. Test basis $V_t$ with $\text{range}(V_t) = \text{range}([\mathbf{1}, V, QV])$; form $\hat Q_t = V_t^TQV_t$.
3. Hyper-reduced quadrature by Algorithm 1, plus stabilizing points as needed.
4. $Q_t = P_t^T\hat Q_tP_t$.

**4.5 Cost.** $O(N_s^2)$ flux evaluations vs $O(N_s)$ for standard hyper-reduction. Observed $N_s \approx 2N$ in 1D, $\approx 10N$ in 2D.

**4.6 Entropy dissipation (most relevant for closures).** With a viscous term $\epsilon\,d(u_N)$:
$$\Delta x\,\mathbf{1}^T\frac{dS}{dt} = -\epsilon\,v_N^Td(u_N).$$
Design requirement: $v_N^Td(u_N) \ge 0$. Options:
- (20) sample rows of $D$: $d(u_N) = V^TD(\mathcal{I}_D,:)^TW_DD(\mathcal{I}_D,:)\tilde u$ — provable, FV-specific.
- (22) $d(u_N) = V^TD(\mathcal{I}_D,:)^TW_DHD(\mathcal{I}_D,:)Vv_N$, $H$ = $\partial u/\partial v$ at averages $(\tilde u_i + \tilde u_{i+1})/2$ — provable for any $D$.
- (23) naive $V^TKVP\tilde u$ — not provable, observed dissipative.
- Both provable options need a separate hyper-reduction for $DV$ (auxiliary basis $V_D$).

---

## 7. Sections 5–7 (brief; not discussed in depth yet)

- **Sec. 5:** weak BCs via numerical flux and a "hybridized" SBP operator. Wall BC via mirror state $\rho^+ = \rho$, $u^+ = -u$, $p^+ = p$, with $f^* = f_S(u^+,u)$ + local Lax–Friedrichs.
- **Sec. 6:** extension to higher dimensions (periodic and non-periodic).
- **Sec. 7 experiments:** compressible Euler, 5-stage 4th-order low-storage RK.
  - **1D Euler:** 2500 cells on $[-1,1]$, CFL 0.75, $\epsilon = 2\times10^{-4}$, reflective walls. IC: $\rho = 2 + \tfrac12 e^{-100(x-1/2)^2}$, $u = \tfrac{1}{10}e^{-100(x-1/2)^2}$, $p = \rho^\gamma$. Run to $T=0.7$, 2801 steps, SVD on every 10th snapshot. Modes $N = 25, 75, 125, 175$. Table 1 point counts: e.g. $N=25$: 54 cubature, 3 stabilizing, 54 viscous.
  - **Key check (Fig. 4):** with $\epsilon = 0$, $|v_N^TV^T(Q\circ F)\mathbf{1}| \sim 10^{-14}$ every step; ROM stays bounded despite Gibbs oscillations.
  - **2D Euler:** periodic $[-1,1]^2$, smoothed Kelvin–Helmholtz.
- **Sec. 7.3 cost caveat:** KH ROM needs ~781k flux evaluations vs ~40k for FOM (884 hyper-reduced points, 75 modes). **The ROM is more expensive per explicit step than the FOM**; the expected payoff is under implicit time stepping.

**Euler ingredients used (Sec. 7):**
- $S = -\rho s$, $s = \log(p/\rho^\gamma)$.
- Entropy variables: $v_1 = \dfrac{\rho e(\gamma+1-s) - E}{\rho e}$, $v_{1+i} = \dfrac{\rho u_i}{\rho e}$, $v_{d+2} = -\dfrac{\rho}{\rho e}$.
- Inverse: $\rho = -(\rho e)v_{d+2}$, $\rho u_i = (\rho e)v_{1+i}$, $E = (\rho e)\Big(1 - \dfrac{\sum_j v_{1+j}^2}{2v_{d+2}}\Big)$, with $\rho e = \Big(\dfrac{\gamma-1}{(-v_{d+2})^\gamma}\Big)^{1/(\gamma-1)}e^{-s/(\gamma-1)}$, $s = \gamma - v_1 + \dfrac{\sum_j v_{1+j}^2}{2v_{d+2}}$.
- Flux: Chandrashekar EC flux with logarithmic mean (stable evaluation per Ismail–Roe). 1D: $f_S^1 = \{\{\rho\}\}_{\log}\{\{u\}\}$, $f_S^2 = f_S^1\{\{u\}\} + p_{\text{avg}}$, $f_S^3 = (E_{\text{avg}} + p_{\text{avg}})\{\{u\}\}$, $\beta = \rho/(2p)$, $p_{\text{avg}} = \{\{\rho\}\}/(2\{\{\beta\}\})$. **Check $E_{\text{avg}}$ against the paper / Chandrashekar before coding.**
- Assumes positive density/pressure; the entropy projection does not enforce positivity (open issue).

---

## 8. Key conceptual point from our discussion: the FOM matters

"Discretize, then reduce." The ROM never sees the PDE — it compresses the FOM's discrete equations and **inherits their structure**. The entropy guarantee exists only because the FOM operator has $Q = -Q^T$, $Q\mathbf{1}=0$, and the ROM construction preserves them.

Where the FOM appears even though online work is small:
- $V$ columns are FOM grid functions (length $K$).
- $V^\dagger$ uses the FOM inner product ($\Delta x$; $M_\Omega$ for DG).
- $\hat Q_t = V_t^TQV_t$ uses the FOM $Q$; enrichment $QV$ is the FOM derivative of the modes.
- $\mathcal{I}$ ⊂ FOM cells; $W$ approximates FOM quadrature.
- $f_S$ is the FOM flux.
- Entropy statement is the FOM's discrete entropy with sampled quadrature.

**Offline** = FOM-sized (basis, $\hat Q_t$, points, $Q_t$). **Online** = only $N$/$N_s$-sized objects.

**Subtlety:** snapshots can come from any solver (they only determine $V$), but the **operator** compressed must be the entropy stable flux-differencing one.

---

## 9. Hook for closures (planned extension)

Any extra term added to the ROM enters the entropy balance only through its contraction with the projected entropy variables — same mechanism as Sec. 4.6. For a learned closure:
$$M_N\frac{du_N}{dt} + 2V(\mathcal{I},:)^T(Q_t\circ F)\mathbf{1} = M_N\,\mathbb{C}(u_N)$$
$$\Rightarrow\quad \mathbf{1}^TW\frac{dS}{dt} = v_N^TM_N\,\mathbb{C}(u_N).$$
**Admissibility condition:** $v_N^TM_N\,\mathbb{C}(u_N) \le 0$ — a single scalar; constrains only net entropy production, so backscatter in individual modes is allowed.

Ideas discussed (not yet validated):
- Hinge penalty in the PINN loss: $\mathcal{L}_{\text{entropy}} = \tfrac{1}{N_t}\sum \max(0,\ v_N^TM_N\mathbb{C})^2$ (JAX, autodiff).
- Runtime backstop: Chan 2025 entropy correction AV with $\Pi_N \to$ ROM projector; parameter-free $\epsilon$ from the entropy residual $\delta$.
- Chan 2025 Lemma 4: $\delta$ is a product of two projection errors → possible online, computable surrogate for ROM projection error (error indicator / extrapolation trust signal).
- Open questions: aggregate vs localized admissibility; whether $\delta$'s superconvergence survives hyper-reduction; learning the closure on top of the hyper-reduced ROM to absorb hyper-reduction error.

---

## 10. Suggested implementation plan

1. **1D periodic FV FOM, Burgers first.** $S = u^2/2$, $v = u$, $f_S = (u_L^2 + u_Lu_R + u_R^2)/6$, $\psi = u^3/6$.
   - Note: since $v = u$, the entropy projection is trivial ($\tilde u = Vu_N$) — Burgers tests flux differencing and hyper-reduction, **not** the entropy projection.
   - Reproduce the Fig. 1 failure ($V_t = [\mathbf{1}, V]$) and the fix ($[\mathbf{1}, V, QV]$) with $u_0 = -\sin(\pi x)$.
2. **1D Euler FOM** (Sec. 7.1 setup) — exercises the entropy projection.
3. Snapshots of $u$ and $v(u)$ → weighted POD.
4. Unreduced ROM (Theorem 2), then hyper-reduced ROM (Theorem 3).
5. Viscosity via (22), then (23) for comparison.
6. Boundary conditions (Sec. 5) last.

**Unit tests / invariants to assert:**
- $\|Q + Q^T\| \approx 0$, $\|Q\mathbf{1}\| \approx 0$ (FOM) and same for $Q_t$.
- $P_tV_t(\mathcal{I},:) \approx I$.
- Condition number of test mass matrix $V_t(\mathcal{I},:)^TWV_t(\mathcal{I},:)$.
- Flux: consistency, symmetry, shuffle condition to machine precision on random states.
- Convective entropy residual $|v_N^TV(\mathcal{I},:)^T(Q_t\circ F)\mathbf{1}| \sim 10^{-14}$ at every step ($\epsilon = 0$).
- Viscous dissipation $v_N^Td(u_N) \ge 0$.
- With $\mathbf{1}\in\text{range}(V)$: conserved totals constant to round-off.
- Weights $w > 0$.
