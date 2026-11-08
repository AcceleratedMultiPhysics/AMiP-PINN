"""Configuration of the entropy stable neural ROM (1D periodic FV, es_nn_rom_fv1d_plan.md).

Select the case and network from the shell without editing this file:
    ESNN_PROBLEM = burgers | euler            (default euler)
    ESNN_MODEL   = structured | unconstrained | closure | memory | backscatter (default structured)
                   closure: ES-ROM + eddy viscosity nu >= 0; backscatter: same with nu >= -eps
    ESNN_RMODES  = modes per variable (default 16); other values get their own outputs folder
    ESNN_REGIME  = reproductive | extrapolation (default reproductive): train on the whole
                   run and evaluate on it, or train on t <= T_TRAIN and roll out to T_FINAL
    ESNN_TRAIN   = derivative | causal (default derivative): fit da/dt at the snapshots, or
                   roll the ROM out from a(0) and fit the trajectory with causal weights
                   (closure models only; weights go to outputs/<problem>/<model>_causal/)
    ESNN_QUICK   = 1  -> coarse grid and short training, for a smoke run
    ESNN_DEVICE  = cpu | cuda                 (default: cuda if available)
"""
import os
import sys
import time
import random
import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

torch.set_default_dtype(torch.float64)        # v^T J v = 0 must hold to round-off

PROBLEM    = os.environ.get("ESNN_PROBLEM", "burgers")  # burgers | euler
MODEL_TYPE = os.environ.get("ESNN_MODEL", "structured")
QUICK      = os.environ.get("ESNN_QUICK", "0") == "1"
REGIME     = os.environ.get("ESNN_REGIME", "reproductive")   # reproductive | extrapolation
TRAIN_MODE = os.environ.get("ESNN_TRAIN", "derivative")      # derivative | causal

# ---------------------------------------------------------------- FOM / data
if PROBLEM == "burgers":
    # plumbing check: v = u, so the entropy projection is trivial (plan Sec. 4)
    K          = 200 if QUICK else 1000
    EPS        = 2e-3
    DT         = 5e-4 * (1000 / K)
    T_TRAIN    = 0.75                  # snapshots used for basis and training
    T_FINAL    = 1.5                   # FOM/ROM horizon (extrapolation after T_TRAIN)
    SAVE_EVERY = 5
    R_MODES    = int(os.environ.get("ESNN_RMODES", 16))   # modes per variable, n = nvar * R_MODES
    GAMMA      = None

    def initial_condition(phys, x):
        return -torch.sin(np.pi * x)[None, :]

elif PROBLEM == "euler":
    # periodic version of Chan's Gaussian pulse (plan Sec. 4)
    K          = 500 if QUICK else 2500
    EPS        = 2e-4
    DT         = 2.5e-4 * (2500 / K)    # CFL ~0.4 by |u| + c, as in FOM/scripts/euler_1d.py
    T_TRAIN    = 0.7
    T_FINAL    = 2.0
    SAVE_EVERY = 10
    R_MODES    = int(os.environ.get("ESNN_RMODES", 16))
    GAMMA      = 1.4
    X0         = 0.5

    def initial_condition(phys, x):
        g = torch.exp(-100 * (x - X0) ** 2)
        rho = 2 + 0.5 * g
        return phys.prim_to_cons(rho, 0.1 * g, rho ** phys.gamma)

else:
    raise ValueError(f"unknown ESNN_PROBLEM={PROBLEM!r}")

if REGIME == "reproductive":
    T_TRAIN = T_FINAL                  # basis and training data from the whole FOM run
elif REGIME != "extrapolation":
    raise ValueError(f"unknown ESNN_REGIME={REGIME!r}")

DOMAIN         = (-1.0, 1.0)
ENTROPY_ENRICH = True                  # add v(u) snapshots to the POD (Chan Sec. 3)

# ---------------------------------------------------------------- network
NET = dict(
    width   = 256,
    depth   = 4,                       # hidden layers
    act     = "tanh",                  # tanh | gelu
    rank    = None,                    # None: full lower-triangular L; k: thin L (n_free x k)
    l_init  = 1e-2,                    # last-layer scale of the L head, so R = L L^T starts small
    n_mu    = 0,                       # parameter inputs (phase 3); 0 = single FOM run
)

# closure model: Chan's ES-ROM + learned eddy viscosity nu(x; a) >= 0
# (backscatter: nu >= -eps, so the total eps + nu >= 0; its initial nu is exactly 0)
CLOSURE = dict(
    nu_modes = None,                   # POD modes spanning log-viscosity; None: all r-1 non-constant
    nu_scale = None,                   # viscosity unit; None: the FOM eps
    z0_init  = -3.0,                   # initial nu = nu_scale * softplus(-3) ~ 0.05 nu_scale
    # memory model only: nu(x; a(t), a(t - lag), ..., a(t - n_delay lag))
    n_delay  = 5,                      # number of past states fed to the network
    delay_every = 4,                   # lag in snapshot intervals (DT * SAVE_EVERY each)
)

# ---------------------------------------------------------------- training
TRAIN = dict(
    nepochs        = 2000 if QUICK else 30000,
    n_pretrain     = 500 if QUICK else 5000,    # teacher-only epochs (model starts as the ES-ROM)
    lr             = 1e-3,
    step_size      = 5000,             # StepLR, Kaya2026 settings
    gamma          = 0.9,
    w_data         = 1.0,
    w_teacher      = 1.0,
    w_entropy      = 0.1,
    loss_norm      = "block",          # block: by variable block | coefficient: each da_i/dt by its own rms
    entropy_target = "fom",            # fom: eps v^T K u | projected: -dx v_N^T V^T du/dt
    n_teacher      = 500 if QUICK else 2000,   # perturbed states labelled by the ES-ROM
    teacher_noise  = 0.1,              # perturbation, fraction of each coefficient's std
    print_step     = 100 if QUICK else 500,
    compile        = False,            # torch.compile(mode="reduce-overhead")
)

# causal rollout training (ESNN_TRAIN=causal): from a(0), RK4 through the snapshot times t_i
# with the network in the loop (memory: its own past states), L_i = |a(t_i) - a_FOM(t_i)|^2
# / mean|a_FOM|^2, loss = mean_i w_i L_i with w_i = exp(-tol sum_{k<i} L_k) (Wang, Sankaran &
# Perdikaris 2024; causal/burgers). Later times count only once earlier ones are fitted.
CAUSAL = dict(
    nepochs    = 50 if QUICK else 1000,
    lr         = 1e-3,
    step_size  = 500,
    gamma      = 0.9,
    tol        = 100.0,                # causality parameter (tol in causal/burgers)
    w_cut      = 1e-4,                 # stop the rollout once w_i < w_cut (no weight left)
    substeps   = 1,                    # RK4 steps per snapshot interval DT * SAVE_EVERY
    clip       = 1.0,                  # gradient-norm clipping (backprop through the rollout)
    print_step = 10 if QUICK else 25,
    threads    = 1,                    # batch-1 RK4 steps: more threads only add overhead
)

# ---------------------------------------------------------------- rollout / post
DT_ROM   = DT                          # ROM time step; same grid as the stored FOM snapshots
USE_BEST = False                       # post: load weights_best.pt instead of weights.pt

save_folder = os.path.join(HERE, "outputs", PROBLEM + ("_quick" if QUICK else "")
                           + ("" if R_MODES == 16 else f"_r{R_MODES}")
                           + ("_repro" if REGIME == "reproductive" else ""), "")
model_folder = os.path.join(save_folder, MODEL_TYPE + ("_causal" if TRAIN_MODE == "causal" else ""), "")

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["figure.max_open_warning"] = 4

_dev = os.environ.get("ESNN_DEVICE")
if _dev:
    device = torch.device(_dev)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch(23)


def tonp(tensor):
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    raise TypeError(f"expected torch.Tensor or np.ndarray, got {type(tensor)}")
