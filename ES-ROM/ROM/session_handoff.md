# Entropy stable ROMs with learning: session handoff (Nov 2026)

Context file for a Claude project. It summarizes one long working session: what was built, what was tried, the numbers, what we learned, and the next idea (an entropy stable, data-driven VMS ROM). Read this first, then `es_nn_rom_fv1d_plan.md` (the original plan), `es_rom_chan2020_notes.md` (paper notes) and `report/esnn_report.pdf` (methods and tests in detail).

Code: repository `AMiP-PINN`, branch `ES-ROM`, folder `ES-ROM/ROM/`. The FV/DG reproductions of Chan's papers are in `ES-ROM/FOM/` and are not used by the ROM code.

---

## 1. Goal and context

- **Research goal:** a reduced order model (ROM) for 1D conservation laws (Burgers now, compressible Euler later) that is **entropy stable by construction** and more accurate than Chan's entropy stable ROM (ES-ROM) at the same reduced dimension.
- **Reference:** J. Chan, *Entropy stable reduced order modeling of nonlinear conservation laws*, JCP 423 (2020) 109789. Collaboration with Jesse Chan; the user's own background is VMS-ROM closures with PINNs.
- **Original plan** (`es_nn_rom_fv1d_plan.md`): no VMS and no closure; a structured network *is* the reduced model, $M\dot a=(J_\theta-R_\theta)v_N$. Section 4 below explains why we moved away from that, and section 7 proposes the next step.

## 2. Working agreements (from the user)

- Reproductive regime only for now: train and evaluate on the same time window $[0,T]$. No time extrapolation.
- Never run two trainings or heavy jobs at the same time (weak CPUs). Run cases one after another.
- Launch Python with `python3 -u` when output goes to a log file (otherwise the log fills in 8 KB blocks and looks frozen).
- `report/esnn_report.pdf`: add content only when the user says what goes in.
- The unconstrained network baseline adds no contribution; do not retrain it.
- No paper PDFs in git. Keep `ES-ROM/FOM/` untouched. Commit only when asked.
- Code layout follows the user's `bnsPinn` style: `esnn_config.py`, `esnn_model.py`, `esnn_train.py`, `esnn_post.py`, plus `utils/` and `tests/`. PyTorch, float64.

## 3. Setup

**Full order model (FOM).** Periodic finite volume on $[-1,1]$:
$$\Delta x\,\dot{\mathbf u}=-2(\mathbf Q\circ\mathbf F(\mathbf u))\mathbf 1-\epsilon\mathbf K\mathbf u,\qquad \mathbf Q=-\mathbf Q^T,\ \mathbf Q\mathbf 1=\mathbf 0,\ \mathbf K=\Delta x\,\mathbf D^T\mathbf D,$$
with an entropy conservative two-point flux $\mathbf F_{ij}=\mathbf f_S(\mathbf u_i,\mathbf u_j)$ and low-storage RK4 (Carpenter–Kennedy).

| | Burgers | Euler |
|---|---|---|
| cells $K$ | 1000 | 2500 |
| $\epsilon$ | $2\times10^{-3}$ | $2\times10^{-4}$ |
| $\Delta t$ | $5\times10^{-4}$ | $2.5\times10^{-4}$ |
| initial condition | $-\sin(\pi x)$ | Gaussian pulse at $x_0=0.5$ |
| $T$ | 1.5 (shock forms near $t\approx0.32$) | 2.0 |
| snapshots | every 5 steps, 601 | every 10 steps |
| ROM results so far | yes | none yet |

**Reduced space.** $\mathbf u\approx\mathbf V\mathbf a$, $\mathbf V=[\mathbf 1/\sqrt K,\ \text{POD}]$ with $\mathbf V^T\mathbf V=\mathbf I$, entropy-variable snapshots added to the POD. $r$ modes per variable, $n=n_{var}\,r$. Projected entropy variables $\mathbf v_N=\mathbf V^T\mathbf v(\mathbf V\mathbf a)$; total entropy $\mathcal S(\mathbf a)=\Delta x\sum_iS((\mathbf V\mathbf a)_i)$ and
$$\dot{\mathcal S}=\Delta x\,\mathbf v_N^T\dot{\mathbf a}.$$

**Chan's ES-ROM (the baseline and "teacher").** $\tilde{\mathbf v}=\mathbf V\mathbf v_N$, $\tilde{\mathbf u}=\mathbf u(\tilde{\mathbf v})$,
$$\Delta x\,\dot{\mathbf a}=-\mathbf V^T\big[2(\mathbf Q\circ\mathbf F(\tilde{\mathbf u}))\mathbf 1+\epsilon\mathbf K\tilde{\mathbf u}\big].$$
For Burgers $\mathbf v=\mathbf u$, so $\mathbf v_N=\mathbf a$ and the ES-ROM equals the POD-Galerkin ROM with the entropy conservative flux.

## 4. What was tried, in order

### 4.1 Structured network for the whole dynamics (the original plan)

$\Delta x\,\dot{\mathbf a}=(\mathbf J_\theta(\mathbf a)-\mathbf R_\theta(\mathbf a))\mathbf v_N$, $\mathbf J_\theta=\mathbf D(\mathbf U-\mathbf U^T)\mathbf D$, $\mathbf R_\theta=\mathbf D\mathbf L\mathbf L^T\mathbf D$, with $\mathbf U,\mathbf L$ from an MLP of $\mathbf a$ and a fixed per-mode scaling $d_i^2=\Delta x\,\mathrm{rms}(\dot a_i)/\mathrm{rms}(v_i)$. Conserved coefficients are removed from $\mathbf J,\mathbf R$, so totals are conserved and uniform states preserved (this replaces Chan's $\mathbf Q\mathbf 1=\mathbf 0$). Losses: data ($\dot{\mathbf a}$ fit), teacher (ES-ROM labels at perturbed states), entropy rate.

Burgers, $r=16$, extrapolation regime (train $t\le0.75$, roll out to 1.5). The ES-ROM itself is essentially exact here (error $\sim10^{-6}$).

| network | error at $t=0.75$ | error at $t=1.5$ |
|---|---|---|
| structured, block-normalized loss | 45.9% | 64.9% |
| structured + per-mode $\mathbf D$ | 17.2% | 24.5% |
| per-coefficient loss variants | stalled, stopped | |
| unconstrained MLP (baseline, dropped) | 85.5% | 317% |

**Diagnosis: multiscale cancellation.** Amplitudes span 6 orders of magnitude, rates 4 orders, time scales 1.7 to 0.0065. For modes $\ge5$ the convective and viscous terms are each 20–68× the net rate and nearly cancel; the MLP must output both huge terms to get their small difference. The user's point: predicting bare numbers with no operators or physics is not physics-informed. Decision: keep Chan's ES-ROM and learn only a physically shaped correction.

### 4.2 Learned eddy-viscosity closure on top of the ES-ROM

$$\dot{\mathbf a}=\dot{\mathbf a}_{ES}(\mathbf a)-\mathbf V^T\mathbf D^T(\boldsymbol\nu_\theta\odot\mathbf D\tilde{\mathbf u}),\qquad \boldsymbol\nu=\nu_0\,\mathrm{softplus}\Big(z_0+\sum_kz_k(\mathbf a)\,\phi_k(x)\Big),$$
with $\phi_k$ the POD modes at cell edges and $z=\mathrm{MLP}(\mathbf a)$ (a DeepONet with a fixed trunk). Its entropy contribution is $-\Delta x\sum_e\nu_e(\mathbf D\tilde{\mathbf v})_e^T(\mathbf D\tilde{\mathbf u})_e\le0$ because $\mathbf u(\mathbf v)$ is monotone. Variants:

- `closure`: $\nu\ge0$, Markovian $\nu(\mathbf a(t))$.
- `memory`: $\nu$ also sees past states $\mathbf a(t-j\,\ell)$, $j=1..5$ (finite Mori–Zwanzig memory).
- `backscatter`: $\nu\ge-\epsilon$, so the total edge viscosity $\epsilon+\nu\ge0$; the closure may remove part of the ES-ROM dissipation, the model still cannot produce entropy (closure has the same discrete form as the ES-ROM viscosity; tested to round-off).

**Reproductive regime, relative error $\|\mathbf V\mathbf a-\mathbf u\|/\|\mathbf u\|$ at $t=1.5$:**

| $r$ | projection | ES-ROM | closure $\nu\ge0$ | backscatter $\nu\ge-\epsilon$ |
|---|---|---|---|---|
| 4 | 2.0% | 13.0% | **9.3%** | 15.2% |
| 6 | 0.23% | 1.32% | **0.84%** | 1.20% |

Extrapolation regime (earlier): $r=4$ closure 0.74% vs ES-ROM 3.4% at $t=0.75$ but 20% vs 17% at 1.5; $r=6$ closure 3.3% vs 7.4% at 1.5; memory $r=6$ about the same as Markovian (3.5%).

**Observations.**
- ES-ROM dissipation is mistimed: $-d\mathcal S/dt$ peaks at 0.78 near $t\approx0.57$ versus the FOM's 0.65 near $t\approx0.5$ ($r=4$), then under-dissipates. A $\nu\ge0$ closure can only add dissipation.
- $r=6$: closure is worse than ES-ROM during shock formation ($t<0.5$) and better in the decay phase. $r=4$: about 1% vs 8% for $0.5<t<1$, over-smoothing late.
- `backscatter` matches the FOM dissipation curve almost exactly, and its error is about 1.2% for $0.4<t<0.95$ ($r=4$), then grows to 15%.
- The derivative loss falls much more than the rollout error improves (e.g. 19× vs 1.6× for the closure).

**Derivative error is not rollout error** ($r=4$, relative $\dot{\mathbf a}$ error on the true FOM trajectory):

| $t$ | ES-ROM | closure | backscatter |
|---|---|---|---|
| 0.50 | 71% | 3.9% | 3.9% |
| 1.00 | 28% | 20% | 0.4% |
| 1.50 | 146% | 111% | 1.6% |

Backscatter fits the rates almost perfectly but its rollout drifts: it was trained only on states on the FOM trajectory, and it drives $\epsilon+\nu$ to $\sim10^{-5}\epsilon$ on half the domain, removing the damping that would pull deviations back. $\nu\ge0$ keeps at least the ES-ROM damping, which makes it robust rather than accurate.

### 4.3 Causal rollout training (memory closure, $\nu\ge0$)

Differentiable RK4 rollout from $\mathbf a_{FOM}(0)$ through all snapshot times, memory fed with the rollout's own past states, loss
$$L_i=\frac{|\mathbf a(t_i)-\mathbf a_{FOM}(t_i)|^2}{\overline{|\mathbf a_{FOM}|^2}},\qquad \mathcal L=\frac1N\sum_iw_iL_i,\qquad w_i=\exp\Big(-\mathrm{tol}\sum_{k<i}L_k\Big),\ \mathrm{tol}=100,$$
as in the user's causal PINN code; the rollout stops once $w_i<10^{-4}$.

Result ($r=4$): the causal horizon stuck at $t=0.40$ (shock formation) for 100 epochs; mean $L$ only 6.0e-4 → 5.7e-4. Reason: ES-ROM error at shock formation is over-dissipation, which a $\nu\ge0$ closure cannot remove, so later times never receive weight. Stopped. Cost: about 2 s/epoch at that horizon, 7.3 s for the full horizon (600 sequential RK4 steps, 5 stages each, plus backprop). Note: the printed causal "Loss" is constant ($\approx1/(\mathrm{tol}\,N)$) whenever the rollout is cut; watch horizon and mean $L$ instead.

### 4.4 Compute notes

- Laptop: i7-12800H, RTX A1000. FP64 on this GPU: 88 GFLOP/s measured (FP32: 2904), so float64 training runs faster on CPU.
- Batch-1 work (rollouts, causal training) is fastest with **one thread**: forward+backward 2.3 ms on 1 thread vs 27.7 ms on 20. Set `torch.set_num_threads(1)` (done automatically for `ESNN_TRAIN=causal`).

## 5. Lessons

1. Learning the whole operator fails on multiscale cancellation; keep exact physics and learn corrections.
2. A dissipation-only correction ($\mathbf R\succeq0$ on top of Chan) cannot fix Chan's over-dissipation during shock formation.
3. Allowing less dissipation ($\nu\ge-\epsilon$) fixes the dissipation curve but, with derivative-only training of a 200k-weight MLP on one trajectory, the rollout drifts off the training manifold.
4. Rollout (causal) training addresses the drift but is slow on CPU and, with $\nu\ge0$, gets stuck at the first error the model class cannot fix.
5. Markovian closures cannot anticipate shock formation; memory did not help in the extrapolation test.

## 6. Code map and how to run

| File | Contents |
|---|---|
| `esnn_config.py` | Case setup, `NET`, `CLOSURE`, `TRAIN`, `CAUSAL` settings, output folders |
| `esnn_model.py` | `load_problem`, `ReducedProblem`, `StructuredNet`, `UnconstrainedNet`, `ClosureNet` (closure / memory / backscatter), `ESNNROM` trainer (derivative and causal modes), `rollout`, `rollout_memory` |
| `esnn_train.py` | Sanity prints, training, writes weights and loss plot |
| `esnn_post.py` | Rolls out ES-ROM, Galerkin and every trained network; error, entropy, dissipation, conservation, profile and viscosity figures; `metrics.json` |
| `utils/` | `physics.py` (Burgers, Euler), `fom.py` (FOM, RK4), `pod.py`, `esrom.py` (Chan's ES-ROM) |
| `tests/` | 45 tests; run one file at a time with `python3 tests/run_tests.py tests/test_model.py` |
| `report/` | `esnn_report.tex/pdf` (methods, coupling, tests, closure results) |

`README.md` still describes an older scalar scaling $s$; the code uses the per-mode $\mathbf D$.

Environment switches (run from `ES-ROM/ROM/`):

| variable | values |
|---|---|
| `ESNN_PROBLEM` | `burgers` (default), `euler` |
| `ESNN_MODEL` | `structured`, `unconstrained`, `closure`, `memory`, `backscatter` |
| `ESNN_TRAIN` | `derivative` (default), `causal` (closure models; saves to `<model>_causal/`) |
| `ESNN_RMODES` | modes per variable (default 16) |
| `ESNN_REGIME` | `reproductive` (default), `extrapolation` |
| `ESNN_QUICK` | `1` for a coarse smoke run |
| `ESNN_DEVICE` | `cpu`, `cuda` |

```bash
ESNN_MODEL=closure ESNN_RMODES=4 ESNN_DEVICE=cpu python3 -u esnn_train.py > outputs/burgers_r4_repro/logs/train_closure.log 2>&1
ESNN_RMODES=4 ESNN_DEVICE=cpu python3 -u esnn_post.py
```

Outputs go to `outputs/<problem>[_r{R}][_repro]/<model>/` and `.../post/`. A derivative-trained closure takes about 10–12 min (30000 epochs) on CPU.

## 7. Next idea: entropy stable data-driven VMS ROM

**Lemma.** Any reduced model
$$\Delta x\,\dot{\mathbf a}=\big(\mathbf J(\mathbf a)-\mathbf R(\mathbf a)\big)\mathbf v_N(\mathbf a),\qquad\mathbf J=-\mathbf J^T,\quad\mathbf R\succeq0,$$
has $\dot{\mathcal S}=-\mathbf v_N^T\mathbf R\mathbf v_N\le0$. Zero rows and columns for the conserved modes give exact conservation and preserve uniform states.

Chan's viscous term fits this form: by the mean value theorem $\tilde{\mathbf u}_{i+1}-\tilde{\mathbf u}_i=\bar{\mathbf H}_e(\tilde{\mathbf v}_{i+1}-\tilde{\mathbf v}_i)$ with $\bar{\mathbf H}_e\succ0$, so
$$\mathbf R^\epsilon(\mathbf a)=\epsilon\,\Delta x\,\mathbf V^T\mathbf D^T\bar{\mathbf H}(\mathbf a)\mathbf D\mathbf V\succeq0\qquad(\text{Burgers: }\bar{\mathbf H}=\mathbf I,\ \mathbf R^\epsilon=\epsilon\mathbf V^T\mathbf K\mathbf V).$$

**Scale split.** $\mathbf V=[\mathbf V_L\ \mathbf V_S]$, $\mathbf a=(\mathbf a_L,\mathbf a_S)$, $\mathbf v_N=(\mathbf v_L,\mathbf v_S)$: large resolved scales (first $r_L$ modes, including the constant mode) and small resolved scales (modes $r_L+1..r$). Unresolved: $\mathbf u'=(\mathbf I-\mathbf V\mathbf V^T)\mathbf u$. VMS hypothesis: unresolved scales act directly only on $\mathbf a_S$.

**Model.**
$$\Delta x\begin{pmatrix}\dot{\mathbf a}_L\\\dot{\mathbf a}_S\end{pmatrix}=\begin{pmatrix}\mathbf J_{LL}(\mathbf a)&\mathbf J_{LS}(\mathbf a)\\-\mathbf J_{LS}(\mathbf a)^T&\mathbf J_{SS}(\mathbf a)\end{pmatrix}\begin{pmatrix}\mathbf v_L\\\mathbf v_S\end{pmatrix}-\mathbf R^\epsilon(\mathbf a)\begin{pmatrix}\mathbf v_L\\\mathbf v_S\end{pmatrix}-\begin{pmatrix}\mathbf 0&\mathbf 0\\\mathbf 0&\mathbf R'_{SS}\end{pmatrix}\begin{pmatrix}\mathbf v_L\\\mathbf v_S\end{pmatrix}$$
with $\mathbf J_{LL},\mathbf J_{SS}$ skew and $\mathbf R'_{SS}\succeq0$. Entropy budget:
$$\dot{\mathcal S}=\underbrace{\mathbf v_L^T\mathbf J_{LS}\mathbf v_S-\mathbf v_S^T\mathbf J_{LS}^T\mathbf v_L}_{=0}-\mathbf v_N^T\mathbf R^\epsilon\mathbf v_N-\mathbf v_S^T\mathbf R'_{SS}\mathbf v_S\le0.$$
$\mathbf J_{LS}$ moves entropy between scales (forward cascade and backscatter, zero net production) and can change shock timing, which no dissipation can. $\mathbf R'_{SS}$ is the unresolved-scale dissipation, entering only through the small resolved scales. "Without a closure": all of $\mathbf J$ is learned from data, Galerkin convection included.

**Burgers parametrization** ($\mathbf v_N=\mathbf a$):
$$\mathbf J(\mathbf a)=\sum_{k=0}^n\phi_k(\mathbf a)(\mathbf U_k-\mathbf U_k^T),\quad\phi_0=1,\ \phi_k=a_k;\qquad\Delta x\,\dot{\mathbf a}=\mathbf J(\mathbf a)\mathbf a-\mathbf R^\epsilon\mathbf a-\mathbf P_S^T\mathbf R'_{SS}\mathbf P_S\mathbf a.$$
Chan's ES-ROM is a feasible point: its convective term $c_i=\sum_{jk}T_{ijk}a_ja_k$ ($T$ symmetric in $j,k$) satisfies $\mathbf a^T\mathbf c(\mathbf a)=0$, so the fully symmetrized $T$ vanishes, $c_i=-2\sum_{jk}T_{jik}a_ja_k$, and
$$\mathbf c(\mathbf a)=\mathbf J_G(\mathbf a)\mathbf a,\qquad(\mathbf J_G)_{ij}=\tfrac23\sum_k(T_{ijk}-T_{jik})a_k\ \ (\text{skew}).$$
Hence the fitted model's weighted derivative misfit is never worse than the ES-ROM's. Euler: $\mathbf v_N$ is nonlinear in $\mathbf a$; use features in $\mathbf a$ or $\mathbf v_N$ and Chan's viscous term directly.

**Fit** (data $\mathbf a^{(m)}=\mathbf V^T\mathbf u(t_m)$, $\dot{\mathbf a}^{(m)}=\mathbf V^T\dot{\mathbf u}_{FOM}(t_m)$):
$$\min_{\{\mathbf U_k\},\ \mathbf R'_{SS}\succeq0}\ \sum_m\Big\|\mathbf W\Big[\Delta x\,\dot{\mathbf a}^{(m)}-\big(\mathbf J(\mathbf a^{(m)})-\mathbf R^\epsilon-\mathbf P_S^T\mathbf R'_{SS}\mathbf P_S\big)\mathbf v_N^{(m)}\Big]\Big\|^2+\lambda\sum_k\|\mathbf U_k-\mathbf U_k^G\|^2,$$
$\mathbf W=\mathrm{diag}(1/\mathrm{rms}(\dot a_i))$ (the multiscale weighting), $\mathbf U_k^G$ the Galerkin operators. Convex (linear residual plus a semidefinite cone): one global minimum, no network training, seconds to solve. Size for $r=4$: 15 parameters in $\mathbf J$ plus $r_S(r_S+1)/2$ in $\mathbf R'_{SS}$, against $601\times3$ equations.

| | ES-VMS |
|---|---|
| entropy stability, conservation, uniform states | by construction |
| multiscale cancellation | handled by $\mathbf W$ in a linear fit |
| slow causal training | not needed |
| memory | still Markovian |
| accurate rollout | not guaranteed (derivative fit), but stable |

## 8. Open decisions

1. **Physical viscosity:** keep $\mathbf R^\epsilon$ fixed (dissipation cannot drop below Chan's at a given state, the $r=4$ shock-formation limit; only $\mathbf J_{LS}$ can compensate), or learn the full $\mathbf R\succeq0$ and use VMS only as a penalty on $\mathbf R_{LL}$.
2. Split $r_L/r_S$ (e.g. $r=4$: two and two).
3. Features in $\mathbf J$: linear in $\mathbf a$ (exact for Burgers) or in $\mathbf v_N$ (needed for Euler).
4. If derivative fitting drifts in rollout: short rollout windows in the least-squares fit, or causal weights.
5. Later: memory terms, the parametric regime, Euler, hyper-reduction.

## 9. References

- J. Chan (2020), *Entropy stable reduced order modeling of nonlinear conservation laws*, JCP 423, 109789.
- Qu, Narayan & Chan (2026), entropy stable DG ROM (reproduced in `ES-ROM/FOM/`).
- Iliescu & Wang (2014), *Variational multiscale proper orthogonal decomposition: Navier–Stokes equations*, Numer. Methods PDE 30.
- Mou, Koc, San, Rebholz & Iliescu (2021), *Data-driven variational multiscale reduced order models*, CMAME 373.
- Ahmed, Pawar, San, Rasheed, Iliescu & Noack (2021), *On closures for reduced order models — a spectrum of first-principle to machine-learned avenues*, Phys. Fluids 33.
- Kaptanoglu, Callaham, Aravkin, Hansen & Brunton (2021), *Promoting global stability in data-driven models of quadratic nonlinear dynamics*, Phys. Rev. Fluids 6.
- Wang, Sankaran & Perdikaris (2024), *Respecting causality for training physics-informed neural networks*, CMAME 421.
- Backscatter: Piomelli, Cabot, Moin & Lee (1991), Phys. Fluids A 3; Leith (1990), Phys. Fluids A 2; Germano, Piomelli, Moin & Cabot (1991), Phys. Fluids A 3; Carati, Ghosal & Moin (1995), Phys. Fluids 7.
