"""Small FOM runs and reduced problems shared by the tests (CPU, a few seconds)."""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("ESNN_DEVICE", "cpu")      # trainer tensors live where the test data lives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.fom import FOM1D, generate_snapshots  # noqa: E402
from utils.physics import Burgers, Euler1D  # noqa: E402
from utils.pod import build_basis  # noqa: E402

torch.set_default_dtype(torch.float64)


def small_run(problem="euler", K=100, eps=1e-3, T=0.3, nsteps=150, save_every=5):
    """FOM snapshots of the periodic Euler pulse or the Burgers sine wave."""
    if problem == "euler":
        phys = Euler1D()
        fom = FOM1D(phys, K, eps=eps)
        g = torch.exp(-100 * (fom.x - 0.5) ** 2)
        rho = 2 + 0.5 * g
        u0 = phys.prim_to_cons(rho, 0.1 * g, rho ** phys.gamma)
    else:
        fom = FOM1D(Burgers(), K, eps=eps)
        u0 = -torch.sin(np.pi * fom.x)[None, :]
    data = generate_snapshots(fom, u0, T / nsteps, nsteps, save_every)
    return fom, data


def small_problem(problem="euler", r=6, t_train=0.2, **kw):
    """ReducedProblem on a small run; the basis uses snapshots with t <= t_train."""
    from esnn_model import ReducedProblem
    fom, data = small_run(problem, **kw)
    train = torch.as_tensor(data["t"] <= t_train + 1e-12)
    V, sigma = build_basis(data["U"][train], r, fom.phys, entropy_enrich=True)
    return ReducedProblem(fom, V, data, t_train, sigma=sigma)
