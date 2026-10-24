"""Entropy conservative/stable finite volume full order model (Chan 2020, Secs. 2, 5).

    dx du/dt + 2 (Q o F) 1 + B (f* - f(u)) + eps K u = 0

Q is tridiagonal, so 2(Q o F)1 reduces to differences of neighbouring two-point
fluxes; nothing here forms F densely.
"""
import numpy as np

from . import operators as ops

# Carpenter & Kennedy (1994) 5-stage, 4th order low-storage Runge-Kutta
RK4A = np.array([0.0,
                 -567301805773.0 / 1357537059087.0,
                 -2404267990393.0 / 2016746695238.0,
                 -3550918686646.0 / 2091501179385.0,
                 -1275806237668.0 / 842570457699.0])
RK4B = np.array([1432997174477.0 / 9575080441755.0,
                 5161836677717.0 / 13612068292357.0,
                 1720146321549.0 / 2090206949498.0,
                 3134564353537.0 / 4481467310338.0,
                 2277821191437.0 / 14882151754819.0])


def rk45_step(rhs, u, dt):
    res = np.zeros_like(u)
    for a, b in zip(RK4A, RK4B):
        res = a * res + dt * rhs(u)
        u = u + b * res
    return u


def integrate(rhs, u0, dt, nsteps, save_every=1, callback=None):
    """Integrate du/dt = rhs(u); returns (times, stacked saved states)."""
    u = u0.copy()
    times, saved = [0.0], [u.copy()]
    for k in range(1, nsteps + 1):
        u = rk45_step(rhs, u, dt)
        if callback is not None:
            callback(k, u)
        if k % save_every == 0 or k == nsteps:
            times.append(k * dt)
            saved.append(u.copy())
    return np.array(times), np.array(saved)


def wall_fluxes(phys, uL, uR, lxf=True):
    """Boundary fluxes f* at the left/right walls from the mirror state.

    f* = f_S(u+, u) (+ local Lax-Friedrichs penalty). Returns (fL, fR), each (nvar,).
    """
    mL, mR = phys.mirror(uL), phys.mirror(uR)
    fL = phys.flux_ec(mL, uL)
    fR = phys.flux_ec(uR, mR)
    if lxf:
        fL = fL - 0.5 * phys.wavespeed(uL[:, None])[0] * (uL - mL)
        fR = fR - 0.5 * phys.wavespeed(uR[:, None])[0] * (mR - uR)
    return fL, fR


class FOM1D:
    def __init__(self, phys, K, domain=(-1.0, 1.0), bc="periodic", eps=0.0, lxf=True):
        self.phys, self.K, self.bc, self.eps, self.lxf = phys, K, bc, eps, lxf
        self.periodic = bc == "periodic"
        a, b = domain
        self.dx = (b - a) / K
        self.x = a + (np.arange(K) + 0.5) * self.dx
        self.Q = ops.Q_periodic(K) if self.periodic else ops.Q_sbp(K)
        self.D = ops.D_matrix(K, self.dx, self.periodic)
        self.Kmat = ops.K_matrix(K, self.dx, self.periodic)

    def conv(self, u, lxf=None):
        """2(Q o F)1 + B(f* - f(u)), shape (nvar, K)."""
        lxf = self.lxf if lxf is None else lxf
        f = self.phys.flux_ec
        if self.periodic:
            fe = f(u, np.roll(u, -1, axis=1))          # flux on edge i+1/2
            return fe - np.roll(fe, 1, axis=1)
        fe = f(u[:, :-1], u[:, 1:])
        fL, fR = wall_fluxes(self.phys, u[:, 0], u[:, -1], lxf)
        fe = np.concatenate([fL[:, None], fe, fR[:, None]], axis=1)
        return fe[:, 1:] - fe[:, :-1]

    def visc(self, u):
        return (self.Kmat @ u.T).T

    def rhs(self, u):
        r = self.conv(u)
        if self.eps:
            r = r + self.eps * self.visc(u)
        return -r / self.dx

    def total_entropy(self, u):
        return self.dx * np.sum(self.phys.entropy(u))

    def entropy_residual(self, u):
        """v^T [2(Q o F)1 + B(f*_EC - f)]: zero for an EC flux and EC walls."""
        v = self.phys.entropy_vars(u)
        return np.sum(v * self.conv(u, lxf=False))
