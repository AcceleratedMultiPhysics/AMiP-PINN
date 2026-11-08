"""Periodic entropy conservative/stable FV full order model (Chan 2020, Sec. 2; plan Sec. 4).

    dx du/dt + 2 (Q o F) 1 + eps K u = 0,   F_ij = f_S(u_i, u_j)

Q is the periodic 1/2-scaled central difference, so ((Q o F) 1)_i =
(f_S(u_i, u_{i+1}) - f_S(u_i, u_{i-1})) / 2 and nothing forms F densely.
K = (1/dx) tridiag(-1, 2, -1) is dx D^T D with D the edge difference (1/dx)[1 -1],
the same scaling as FOM/esrom/operators.py. All functions accept batched
states (..., nvar, K).
"""
import numpy as np
import torch

# Carpenter & Kennedy (1994) 5-stage, 4th order low-storage Runge-Kutta
RK4A = [0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0]
RK4B = [1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0]


def rk45_step(rhs, u, dt):
    res = torch.zeros_like(u)
    for a, b in zip(RK4A, RK4B):
        res = a * res + dt * rhs(u)
        u = u + b * res
    return u


@torch.no_grad()
def integrate(rhs, u0, dt, nsteps, save_every=1, stop_on_nonfinite=False):
    """Integrate du/dt = rhs(u).

    Returns (times as numpy array, saved states stacked along a new first axis).
    With stop_on_nonfinite the run ends at the first NaN/Inf state, which is
    not saved, so a blown-up ROM returns a shorter trajectory.
    """
    u = u0.clone()
    times, saved = [0.0], [u.clone()]
    for k in range(1, nsteps + 1):
        u = rk45_step(rhs, u, dt)
        if stop_on_nonfinite and not torch.isfinite(u).all():
            break
        if k % save_every == 0 or k == nsteps:
            times.append(k * dt)
            saved.append(u.clone())
    return np.array(times), torch.stack(saved)


class FOM1D:
    def __init__(self, phys, K, domain=(-1.0, 1.0), eps=0.0, device=None):
        self.phys, self.K, self.eps = phys, K, eps
        a, b = domain
        self.dx = (b - a) / K
        self.x = a + (torch.arange(K, dtype=torch.float64, device=device) + 0.5) * self.dx

    def conv(self, u):
        """2 (Q o F) 1, shape (..., nvar, K)."""
        fe = self.phys.flux_ec(u, torch.roll(u, -1, -1))     # flux on edge i+1/2
        return fe - torch.roll(fe, 1, -1)

    def visc(self, u):
        """K u with K = (1/dx) tridiag(-1, 2, -1), periodic."""
        return (2.0 * u - torch.roll(u, -1, -1) - torch.roll(u, 1, -1)) / self.dx

    def rhs(self, u):
        r = self.conv(u)
        if self.eps:
            r = r + self.eps * self.visc(u)
        return -r / self.dx

    def total_entropy(self, u):
        return self.dx * self.phys.entropy(u).sum(-1)

    def totals(self, u):
        """Total mass/momentum/energy dx * sum_i u_i, shape (..., nvar)."""
        return self.dx * u.sum(-1)

    def entropy_residual(self, u):
        """v^T 2(Q o F) 1: zero (round-off) for an entropy conservative flux."""
        return (self.phys.entropy_vars(u) * self.conv(u)).sum((-2, -1))

    def entropy_dissipation(self, u):
        """eps v^T K u = -dS/dt of the FOM (the convective part conserves S)."""
        return self.eps * (self.phys.entropy_vars(u) * self.visc(u)).sum((-2, -1))

    def stable_dt(self, u, cfl=0.75):
        """Explicit time step from the max wave speed, and dx^2/(2 eps) if viscous."""
        dt = cfl * self.dx / self.phys.wavespeed(u).max().item()
        if self.eps:
            dt = min(dt, cfl * self.dx ** 2 / (2.0 * self.eps))
        return dt


def generate_snapshots(fom, u0, dt, nsteps, save_every):
    """FOM trajectory plus the exact RHS and entropy dissipation at each snapshot (plan Sec. 7).

    Using rhs(U) instead of finite differences of U gives the exact time
    derivative that the ROM is trained to reproduce.
    """
    t, U = integrate(fom.rhs, u0, dt, nsteps, save_every)
    with torch.no_grad():
        dU = fom.rhs(U)
        D = fom.entropy_dissipation(U)
        S = fom.total_entropy(U)
    return dict(t=t, U=U, dU=dU, D=D, S=S)
