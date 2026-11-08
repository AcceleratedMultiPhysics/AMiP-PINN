# Entropy Stable Neural ROM — 1D Uniform FV Plan

Context file for a Claude Code session. Goal: build a neural-network reduced order model (ROM) for 1D conservation laws (target: compressible Euler) that is **entropy stable by construction**, using Chan's entropy stable ROM (ES-ROM) as the physics reference.

Companion file: `es_rom_chan2020_notes.md` (paper summary of Chan 2020; formulas, hyper-reduction details).

Main reference: J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws*, JCP 423 (2020) 109789. Parenthetical notes "(Chan: …)" map our notation to the paper.

**Scope decisions**
- 1D, uniform finite volume (FV) grid, periodic boundaries first.
- No VMS / closure. The network *is* the reduced model.
- PyTorch, float64 everywhere.
- DG deferred (maybe never). Hyper-reduction deferred to a later phase.
- First runs: single FOM run, no parameter $\mu$ (temporal extrapolation only).

---

## 1. Notation

| Ours | Meaning | Chan 2020 |
|---|---|---|
| $K$ | number of FV cells | $K$ |
| $u \in \mathbb{R}^K$ (per variable) | FOM state | $u_h$ |
| $V \in \mathbb{R}^{K\times r}$ | orthonormal POD basis, $V^TV = I$ | $V$ |
| $a$ | reduced coefficients, $u \approx Va$ | $u_N$ |
| $n$ | total reduced dimension; Euler: $n = 3r$ | — |
| $M$ | reduced mass matrix; here $M = \Delta x\,I$ | $\Delta x V^TV$, or $M_N$ hyper-reduced |
| $v(u)$ | entropy variables, $v = \partial S/\partial u$ | $v$ |
| $v_N(a) = V^Tv(Va)$ | projected entropy variables | $V^\dagger v(Vu_N)$ |
| $\mathcal{S}(a) = \Delta x\sum_i S((Va)_i)$ | total entropy of reconstructed state | $\Delta x\,\mathbf{1}^TS(Vu_N)$ |
| $\mu$ | parameters varying between FOM runs | — |

Same basis $V$ for every conserved variable (Chan's Kronecker convention): for Euler, $a = [a_\rho,\ a_{\rho u},\ a_E]$, each of length $r$.

---

## 2. The core identity (why this works)

For any reduced model,
$$\frac{d\mathcal{S}}{dt} = \Delta x\, v(Va)^TV\dot a = v_N(a)^TM\,\dot a. \qquad \text{(Chan: Eq. 11)}$$
Second equality: $\dot a$ only lives in the reduced space, so only the projection of $v$ matters.

**Entropy stability ⇔ $v_N(a)^TM\dot a \le 0$.** One scalar.

Chan's ES-ROM gets this through flux differencing + entropy projection + structure-preserving operators. The network gets it from its output structure.

**Requirement:** $M$ and $v_N$ must come from the **same inner product**. For uniform FV + plain SVD: $M = \Delta x I$, $v_N = V^Tv(Va)$. Never mix exact and hyper-reduced versions.

---

## 3. Model architecture

$$\boxed{M\dot a = \big(J_\theta(a,\mu) - R_\theta(a,\mu)\big)\,v_N(a)}, \qquad J_\theta = A - A^T,\quad R_\theta = LL^T.$$

Then for **any** weights:
$$\frac{d\mathcal S}{dt} = v_N^TJ_\theta v_N - v_N^TR_\theta v_N = 0 - \|L^Tv_N\|^2 \le 0.$$

**Pieces:**
- **Inputs:** $a$ (current reduced state) and $\mu$ (parameters; omit if single run). **Not** time $t$ — dynamics are autonomous; this helps temporal extrapolation.
- **Network outputs:** unconstrained numbers that fill $A$ and $L$. Never output $J$, $R$ directly (no guarantee).
  - Compact version (preferred): $n(n-1)/2$ strict-upper-triangle entries → $J$ (fill $J_{ji} = -J_{ij}$, zero diagonal); $n(n+1)/2$ entries → lower-triangular $L$, $R = LL^T$.
  - Large $n$: thin $L \in \mathbb{R}^{n\times k}$, low-rank dissipation.
- **$v_N(a)$: computed by physics, NOT learned.** $a \to Va \to v(Va)$ pointwise $\to V^T(\cdot)$. A learned $v_N$ would make the guarantee about a fake entropy.
- **Roles:** $J$ ↔ convective term (Chan: $2V^T(Q\circ F)\mathbf{1}$, entropy conservative); $R$ ↔ viscosity (Chan: $\epsilon\,d(a)$, Sec. 4.6 requires $v_N^Td \ge 0$).
- $R = 0$ ⇒ exactly entropy conservative. Initialize the last layer so $L$ starts small.
- **Generality:** any entropy-conserving field $f$ (with $v^Tf = 0$) equals $Jv$ for skew $J = (fv^T - vf^T)/\|v\|^2$. Chan's ES-ROM is itself one $(J, R)$ pair. No expressiveness lost.

**Conservation (exact, free).** Put the normalized constant vector $\mathbf{1}/\sqrt{K}$ as the first column of $V$ (orthogonalize snapshots against it). Then total mass/momentum/energy are proportional to the first coefficient of each variable block, and conservation ⇔ those 3 entries of $\dot a$ are zero. Enforce by **zeroing the corresponding rows and columns of $J$ and $R$** (= projector $\Pi_c J\Pi_c$, $\Pi_c R\Pi_c$). Skewness and PSD are preserved. (Chan: Theorem 2 assumption $\mathbf{1}\in\text{range}(V)$.)

**Boundaries.** Periodic: nothing to enforce. The network acts on coefficients; periodicity lives in the basis and FOM operator. Reflective walls (mirror states) also fine: no boundary entropy flux. Inflow/outflow (later, if ever): $\mathcal S$ can increase via the boundary, so split $M\dot a = (J - R)v_N + b(a,t)$ and keep the structure on the interior part only.

### Forward pass sketch (PyTorch)

```python
def forward(self, a, mu=None):                      # a: (B, n)
    x = a if mu is None else torch.cat([a, mu], -1)
    z = self.mlp(x)
    J = a.new_zeros(a.shape[0], n, n); J[:, iu0, iu1] = z[:, :nJ]
    J = J - J.transpose(-1, -2)                     # skew
    L = a.new_zeros(a.shape[0], n, n); L[:, il0, il1] = z[:, nJ:]
    R = L @ L.transpose(-1, -2)                     # PSD
    J, R = J * self.cmask, R * self.cmask           # zero conserved rows/cols
    vN = v_N(a)                                     # physics, not learned
    return ((J - R) @ vN.unsqueeze(-1)).squeeze(-1) / dx
```
`iu0, iu1` = `torch.triu_indices(n, n, 1)`; `il0, il1` = `torch.tril_indices(n, n)`; `cmask` = outer product of a 0/1 vector that is 0 at conserved indices.

---

## 4. Full order model (FOM) — 1D periodic FV, Chan Sec. 2

$$\Delta x\frac{du}{dt} + 2(Q\circ F)\mathbf{1} + \epsilon K_{\text{lap}}u = 0, \qquad F_{ij} = f_S(u_i, u_j).$$
- $Q$: periodic central difference, $+\tfrac12$ superdiagonal, $-\tfrac12$ subdiagonal, wrap corners. Sparse, so $(Q\circ F)\mathbf{1}$ only needs neighbor fluxes: $\big((Q\circ F)\mathbf{1}\big)_i = \tfrac12\big(f_S(u_i,u_{i+1}) - f_S(u_i,u_{i-1})\big)$. $O(K)$.
- $K_{\text{lap}} = D^TD$, periodic. Entropy dissipation: $-\epsilon\,v^TK_{\text{lap}}u \le 0$.
- Time stepping: low-storage RK4 (Carpenter–Kennedy, used in the papers) or classical RK4.

### Burgers (sanity only)
$S = u^2/2$, $v = u$, $f_S = (u_L^2 + u_Lu_R + u_R^2)/6$, $\psi = u^3/6$. **Caveat:** $v = u$ ⇒ entropy projection is trivial. Burgers tests plumbing only.

### Euler (real target)
Entropy $S = -\rho s$, $s = \log(p/\rho^\gamma)$, $\rho e = E - \tfrac12\rho u^2$, $p = (\gamma-1)\rho e$.
- $v_1 = \dfrac{\rho e(\gamma+1-s) - E}{\rho e}$, $v_2 = \dfrac{\rho u}{\rho e}$, $v_3 = -\dfrac{\rho}{\rho e}$.
- Inverse (for the teacher's entropy projection): $\rho = -(\rho e)v_3$, $\rho u = (\rho e)v_2$, $E = (\rho e)\big(1 - \tfrac{v_2^2}{2v_3}\big)$, with $s = \gamma - v_1 + \tfrac{v_2^2}{2v_3}$, $\rho e = \Big(\tfrac{\gamma-1}{(-v_3)^\gamma}\Big)^{1/(\gamma-1)}e^{-s/(\gamma-1)}$.
- EC flux (Chandrashekar, as in Chan 2018 Eq. 106), $\beta = \rho/(2p)$:
  - $f_1 = \{\{\rho\}\}_{\log}\{\{u\}\}$
  - $f_2 = \dfrac{\{\{\rho\}\}}{2\{\{\beta\}\}} + \{\{u\}\}f_1$
  - $f_3 = f_1\Big(\dfrac{1}{2(\gamma-1)\{\{\beta\}\}_{\log}} - \tfrac12\{\{u^2\}\}\Big) + \{\{u\}\}f_2$
  - $\{\{\cdot\}\}$ arithmetic mean, $\{\{u^2\}\}$ mean of squares, $\{\{\cdot\}\}_{\log}$ logarithmic mean.
- Stable log mean (Ismail–Roe): $\zeta = a_L/a_R$, $f = (\zeta-1)/(\zeta+1)$, $w = f^2$, $\mathcal{F} = \log(\zeta)/(2f)$ if $w \ge \text{tol}$, else $1 + w/3 + w^2/5 + w^3/7$; $\{\{a\}\}_{\log} = (a_L + a_R)/(2\mathcal F)$. Chan 2018 reports tol $10^{-2}$ gives entropy residual $\sim10^{-10}$, tol $10^{-4}$ gives $\sim10^{-15}$.
- **Formulas above are transcribed — verify with the unit tests in §8 (autodiff $v = \partial S/\partial u$, round trip, shuffle condition) rather than trusting them.**
- Positivity: $v$ undefined if $\rho \le 0$ or $p \le 0$.

**Test cases.** Start periodic. Periodic version of Chan's Gaussian pulse on $[-1,1]$: $\rho = 2 + \tfrac12 e^{-100(x-x_0)^2}$, $u = \tfrac1{10}e^{-100(x-x_0)^2}$, $p = \rho^\gamma$, $\gamma = 1.4$. Chan's 1D setup used 2500 cells, CFL 0.75, $\epsilon = 2\times10^{-4}$, $T = 0.7$, walls. Reflective walls (mirror state $\rho^+ = \rho$, $u^+ = -u$, $p^+ = p$) as phase 2.

---

## 5. POD basis (Chan end of Sec. 3)

- Snapshots of **both** conserved variables $u$ **and** entropy variables $v(u)$, all components as columns of one $K$-row matrix (one shared basis).
- Orthogonalize snapshots against $\mathbf{1}/\sqrt K$, SVD the rest, $V = [\mathbf{1}/\sqrt K,\ \text{first } r-1 \text{ left singular vectors}]$.
- Plain (unweighted) SVD is correct for uniform FV. (DG would need weighted POD.)
- Check $\|V^TV - I\|$ and that projection error of $v$ is small; the entropy projection is only accurate if $V$ represents $v$ well.

---

## 6. Teacher: unreduced ES-ROM (Chan Theorem 2) — no hyper-reduction in phase 1

$$\tilde v = VV^Tv(Va), \qquad \tilde u = u(\tilde v), \qquad F_{ij} = f_S(\tilde u_i, \tilde u_j),$$
$$\Delta x\,\dot a_{\text{ES}} = -V^T\Big[2(Q\circ F)\mathbf{1} + \epsilon K_{\text{lap}}\tilde u\Big].$$
- Cost $O(K)$ per RHS since $Q$ is sparse. Fine for 1D.
- Convective part conserves $\mathcal S$; viscous part $\tilde v^TK_{\text{lap}}u(\tilde v) \ge 0$ (mean value argument, like the FOM) ⇒ dissipative.
- Teacher is also a baseline (§9). Verify its convective entropy residual $|v_N^TV^T(Q\circ F)\mathbf{1}| \sim 10^{-14}$ (Chan Fig. 4).

---

## 7. Training

**Data.** Project snapshots $a_k = V^Tu(t_k)$. Exact projected time derivative from the FOM RHS (no finite differences): $\dot a_k = V^T\dot u_{\text{FOM}}(t_k)$.

**Losses** (stability comes from architecture; losses deliver accuracy):
1. **Data:** $\|\dot a_k - \text{model}(a_k)\|^2$ (normalize per variable block).
2. **Teacher/physics:** $\|\text{model}(a) - \dot a_{\text{ES}}(a)\|^2$ at sampled states, including perturbed $a_k + \delta$ (reject non-admissible states: $\rho, p \le 0$ in $Va$). Targets precomputed offline, no gradient through the flux.
3. **Entropy rate:** match dissipation so the model can't cheat by damping everything: $\big(v_N^TR_\theta v_N/\Delta x - \text{target}\big)^2$, target = $-\tfrac{d\mathcal S}{dt}$ from data ($\epsilon\,v^TK_{\text{lap}}u$ from FOM) or from teacher.
4. Optional later: multi-step rollout loss.

**Optional pretraining:** fit $(J_\theta, R_\theta)$ to the teacher first (model starts as the ES-ROM), then fine-tune on FOM data.

**Soft-constraint alternative (not primary):** trajectory network $a_\theta(t)$ with residual $\|\Delta x\,\partial_ta_\theta - \Delta x\,\dot a_{\text{ES}}(a_\theta)\|^2$ + penalty $\max(0, v_N^TM\partial_ta_\theta)^2$; $\partial_ta_\theta$ via `torch.func.jacfwd` + `vmap`. Only stable where trained.

**PyTorch settings**
- `torch.set_default_dtype(torch.float64)` (structure check $v^TJv = 0$ needs FP64).
- MLP: width 128–256, tanh or GELU, Xavier normal; input normalization of $a$.
- Adam, lr $10^{-3}$, `StepLR(step_size=5000, gamma=0.9)` (Kaya2026 settings).
- Constant tensors as `register_buffer`. Batch all samples; `.item()` only occasionally; try `torch.compile(mode="reduce-overhead")`.
- **NaN-gradient trap:** the log-mean `torch.where` evaluates both branches; sanitize inputs to the unused branch ("double where") if the flux is ever differentiated.

**Online integration.** $a_0 = V^Tu_0$; RK4 on $\dot a = \text{model}(a)$. Semi-discrete guarantee only; add relaxation RK (scalar root solve per step to enforce the entropy balance) in a later phase. Monitor $\mathcal S(t)$, conserved totals, admissibility of $Va$.

---

## 8. Unit tests / invariants

Physics functions:
- $v(u) = \nabla_uS$ via autodiff on random admissible states.
- Round trip $u(v(u)) = u$.
- EC flux: consistency $f_S(u,u) = f(u)$, symmetry, shuffle $(v_L - v_R)^Tf_S = \psi_L - \psi_R$ (Euler $\psi = (\gamma-1)\rho u$ for $S = -\rho s$ — verify scaling numerically via $\psi = v^Tf - F$, $F = Su$).
- FOM: $Q = -Q^T$, $Q\mathbf 1 = 0$; entropy conserved to round-off with $\epsilon = 0$ (semi-discrete residual).

Basis: $V^TV = I$; first column constant.

Teacher: convective entropy residual $\sim10^{-14}$; viscous part dissipative; conserved totals constant.

Model:
- **Random weights** (untrained): $v_N^T(J - R)v_N \le 0$ and $v_N^TJv_N \approx 10^{-14}$ on random admissible $a$.
- Conserved indices of $\dot a$ exactly zero.
- $\mathcal S(t)$ non-increasing along RK rollouts (up to time-discretization error, shrinking with $\Delta t$).

---

## 9. Experiments

1. Burgers plumbing check (periodic, $u_0 = -\sin(\pi x)$ or smooth wave).
2. 1D periodic Euler Gaussian pulse: train on $t \in [0, T_{\text{train}}]$, roll out to $\gg T_{\text{train}}$.
3. Compare:
   - **Structured network** (this plan),
   - **Unconstrained neural ODE** (same MLP, outputs $\dot a$ directly),
   - **Teacher ES-ROM** (Chan Theorem 2),
   - Plain POD–Galerkin ROM (non-ES) as a reference.
4. Metrics: relative $L^2$ error of $Va$ vs FOM, $\mathcal S(t)$, conserved totals, wall time per step, behaviour far past training window. The headline plot: unconstrained model drifts/blows up, structured model stays bounded and accurate.
5. Phase 2: reflective walls. Phase 3: parameters $\mu$ (e.g. pulse amplitude, $\epsilon$).

---

## 10. Phase 3 (later): hyper-reduction

Network removes Chan's $O(N_s^2)$ flux matrix, but $v_N(a) = V^Tv(Va)$ still costs $O(Kr)$. Hyper-reduce **only the entropy variables**:
$$M = V(\mathcal I,:)^TWV(\mathcal I,:), \qquad v_N = M^{-1}V(\mathcal I,:)^TW\,v(V(\mathcal I,:)a).$$
- Identity stays exact for the sampled entropy $\sum_{i\in\mathcal I}w_iS(u_i)$ (Chan Theorem 3).
- Much simpler than Chan Sec. 4: no $Q_t$, no test basis $[\mathbf 1, V, QV]$, no compress-and-project. Need only a positive-weight quadrature accurate for products $V(:,i)\circ V(:,j)$ with nonsingular $M$ — Chan Algorithm 1 (greedy empirical cubature + NNLS).
- Conservation: re-derive $c$ with $M$ (no longer $\Delta xI$).
- **Train with exactly the $M$, $v_N$ used online.**

---

## 11. Known limitations / open points

- Guarantee is semi-discrete; positivity not guaranteed (shared with ES-ROM papers).
- Stability ≠ accuracy: transport problems with slow POD decay (Kolmogorov $n$-width) remain hard.
- Over-dissipation risk → entropy-rate loss.
- Output size $O(n^2)$: fine for $n \lesssim 150$; else low-rank $L$, structured $A$.
- Later research direction (separate): ES-ROM + learned closures (C-PINN-ROM, Kaya et al. arXiv:2603.01998), entropy admissibility $v_N^TM\,\mathbb C \le 0$.

## 12. Suggested code layout

```
euler1d.py     # S, v(u), u(v), flux f, EC flux f_S, log mean, psi
fom.py         # periodic FV FOM RHS, RK4, snapshot generation
pod.py         # basis with constant mode, projection utilities
esrom.py       # teacher: unreduced ES-ROM RHS (Chan Thm 2)
model.py       # structured network (J - R) v_N, unconstrained baseline
train.py       # data/teacher/entropy-rate losses, training loop
rollout.py     # online integration, metrics, plots
tests/         # invariants from §8
```
