# Entropy stable neural ROM (1D periodic FV)

This is a PyTorch implementation of [es_nn_rom_fv1d_plan.md](es_nn_rom_fv1d_plan.md). The network is the reduced model:

$$M\dot a = \big(J_\theta(a) - R_\theta(a)\big)\,v_N(a),\qquad J_\theta = s(U-U^T),\quad R_\theta = s\,LL^T,\quad M = \Delta x I .$$

The entropy variables $v_N(a) = V^Tv(Va)$ come from the physics and are not learned. With this form, $d\mathcal S/dt = -s\|L^Tv_N\|^2 \le 0$ for any weights, and mass, momentum and energy are conserved exactly. The FV code in `../FOM/` is not used or changed. The only link is an optional cross-check test that compares against it.

The file layout follows `bnsPinn/tgv`, with config, model, train and post files:

| File | Contents |
|---|---|
| `esnn_config.py` | Case setup (FOM grid, ε, time step, training window, modes), `NET` and `TRAIN` settings, device, seed, `tonp` |
| `esnn_model.py` | `load_problem` (FOM snapshots and POD basis, cached), `ReducedProblem`, `StructuredNet`, `UnconstrainedNet`, `ESNNROM` trainer (losses, `train`, `report`, `plot_loss`), `rollout` |
| `esnn_train.py` | Basis and teacher sanity prints, then training. Writes `weights.pt`, `weights_best.pt`, `weights_losses.txt`, `loss.png`, `report.txt` |
| `esnn_post.py` | Rolls out all ROMs to `T_FINAL` and compares them with the FOM. Writes figures and `metrics.json` |
| `utils/physics.py` | Burgers and Euler: $S$, $v(u)$, $u(v)$, $f$, $f_S$ (Chandrashekar), $\psi$, log mean |
| `utils/fom.py` | Periodic FV FOM $\Delta x\,\dot u + 2(Q\circ F)\mathbf 1 + \epsilon Ku = 0$, low-storage RK4, snapshot generation |
| `utils/pod.py` | Basis $V = [\mathbf 1/\sqrt K, \text{POD}]$ with $V^TV = I$, `project` / `reconstruct` |
| `utils/esrom.py` | Teacher: unreduced ES-ROM (Chan Thm 2). Without the entropy projection it is the plain POD-Galerkin ROM |
| `tests/` | The invariants from plan Sec. 8 (see below) |

## Running

Run from `ES-ROM/ROM/`:

```bash
ESNN_PROBLEM=burgers ESNN_QUICK=1 python esnn_train.py
```
```bash
ESNN_PROBLEM=burgers ESNN_QUICK=1 python esnn_post.py
```
```bash
python esnn_train.py
```
```bash
ESNN_MODEL=unconstrained python esnn_train.py
```
```bash
python esnn_post.py
```

- The first two commands are a quick plumbing run on a coarse grid.
- The last three run the main Euler case: train the structured network, train the unconstrained baseline, then compare everything.
- Environment switches: `ESNN_PROBLEM` (burgers | euler), `ESNN_MODEL` (structured | unconstrained), `ESNN_QUICK=1`, and `ESNN_DEVICE` (cpu | cuda). Everything else is set in `esnn_config.py`.
- The first run of a case integrates the FOM and stores the snapshots and the basis in `outputs/<case>/fom.npz` (git-ignored). Later runs reuse that file, so training and post-processing always share the same $V$.
- `esnn_post.py` compares the structured network, the unconstrained network, the ES-ROM teacher and POD-Galerkin. It writes `error_vs_time.png`, `entropy_vs_time.png`, `dissipation_vs_time.png`, `conservation.png`, `profiles.png`, `singular_values.png` and `metrics.json` to `outputs/<case>/post/`.

## Tests

The test files cover:
- `test_physics.py`: $v = \nabla_u S$ (autograd), the round trip $u(v(u)) = u$, flux consistency, symmetry and shuffle, the scaling of $\psi$, and finite log-mean gradients.
- `test_fom.py`: conservation, the entropy conservative convection term, dissipation $\ge 0$, entropy drift of RK4 as $\Delta t \to 0$, and a cross-check against `../FOM/esrom`.
- `test_pod_esrom.py`: $V^TV = I$, the constant first mode, totals carried by the constant coefficients, the teacher's entropy residual at round-off, and the teacher being entropy stable and conservative.
- `test_model.py`: with random weights, $v^TJv \approx 0$, $v^T(J-R)v \le 0$ and conserved rates exactly zero. Also low-rank $L$, batch consistency, $\mathcal S(t)$ non-increasing along a rollout, and a few-epoch trainer smoke run.

Run them with either command:

```bash
python -m pytest tests -q
```
```bash
python tests/test_model.py
```

The root `.gitignore` has a `*test*` rule, so the files in `tests/` need `git add -f` before they can be committed.

## Choices not fixed by the plan

1. **Entropy-rate loss scaling.** With $M\dot a = (J-R)v_N$, the rate is $d\mathcal S/dt = \Delta x\,v_N^T\dot a = -v_N^TRv_N$, with no $1/\Delta x$. The loss compares $-\Delta x\,v_N^T\dot a_\theta$ with the target. For the structured network this equals $s\,v^TRv$, and for the baseline it is the same formula.
2. **Entropy-rate target.** `entropy_target="fom"` (the default) uses the FOM rate $\epsilon v^TKu \ge 0$. `"projected"` uses $-\Delta x\,v_N^TV^T\dot u$, the rate of the projected trajectory, which can have either sign.
3. **Conserved rows and columns.** These are removed by building $U$ and $L$ only on the $n_f = n - n_{var}$ free coefficients. The result equals the plan's mask $\Pi_cJ\Pi_c$, with $n_f^2$ instead of $n^2$ outputs.
4. **Output scale $s$.** It is fixed from data, $s = \overline{\Delta x\|\dot a_f\|}/(\overline{\|v_f\|}\sqrt{n_f})$, so outputs of size O(1) give the right magnitude of $\dot a$. The last-layer rows of $L$ are multiplied by `l_init` $=10^{-2}$. The baseline is scaled by the RMS of $\dot a$ in each variable block.
5. **Teacher samples.** These are the training snapshots plus `n_teacher` Gaussian perturbations, of size `teacher_noise` × the std of each coefficient. Conserved coefficients are not perturbed. States with $\rho \le 0$ or $p \le 0$ in $Va$, or with a non-finite teacher RHS, are dropped. Teacher targets are computed once, and so is $v_N$ at all fixed training inputs.
6. **Pretraining.** The first `n_pretrain` epochs use only the teacher loss, so the model starts close to the ES-ROM. After that, the loss is `w_data`·data + `w_teacher`·teacher + `w_entropy`·entropy. `weights_best.pt` only tracks the second stage.
7. **Basis.** It is built from training-window snapshots only, so rollouts past `T_TRAIN` are true extrapolation. $u$ and $v(u)$ snapshots are included (`ENTROPY_ENRICH`).
8. **Euler case.** This is the periodic Gaussian pulse at $x_0 = 0.5$ with K = 2500, ε = 2e-4 and Δt = 2.5e-4 (the same Δt as `FOM/scripts/euler_1d.py`). Training uses $[0, 0.7]$ and rollouts go to $T = 2$, with r = 16 modes per variable (n = 48, 2025 network outputs).
