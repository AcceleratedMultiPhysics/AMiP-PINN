"""Conservation laws with an entropy-conservative two-point flux (DG package).

Independent copy of esrom/physics.py (the FV package is left untouched), plus
linear advection, which Qu, Narayan & Chan (2026) use in Sec. 6.2.1.

Arrays are stored component-first: a state on n points has shape (nvar, n).
Two-point fluxes broadcast over trailing axes, so
``flux_ec(u[:, :, None], u[:, None, :])`` returns the (nvar, n, n) flux matrix F.

Entropy conventions follow Chan (2020):
  Burgers: S = u^2/2, v = u, psi = u^3/6.
  Euler:   S = -rho*s, s = log(p/rho^gamma), psi = (gamma-1)*rho*u.
"""
import numpy as np


def logmean(a, b):
    """Logarithmic mean (a-b)/(log a - log b), evaluated stably (Ismail & Roe)."""
    zeta = a / b
    f = (zeta - 1.0) / (zeta + 1.0)
    uu = f * f
    with np.errstate(divide="ignore", invalid="ignore"):
        F_big = np.log(zeta) / (2.0 * f)
    F_small = 1.0 + uu / 3.0 + uu * uu / 5.0 + uu * uu * uu / 7.0
    F = np.where(uu < 1e-4, F_small, F_big)
    return (a + b) / (2.0 * F)


class Burgers:
    nvar = 1
    names = ["u"]

    def entropy(self, u):
        return 0.5 * u[0] ** 2

    def entropy_vars(self, u):
        return u.copy()

    def cons_vars(self, v):
        return v.copy()

    def psi(self, u):
        return u[0] ** 3 / 6.0

    def flux(self, u):
        return 0.5 * u ** 2

    def flux_ec(self, uL, uR):
        return (uL * uL + uL * uR + uR * uR) / 6.0

    def dudv(self, u):
        """Jacobian du/dv, shape (nvar, nvar, n)."""
        return np.ones((1, 1) + u.shape[1:])

    def wavespeed(self, u):
        return np.abs(u[0])


class Advection(Burgers):
    """u_t + a u_x = 0 with S = u^2/2; the EC flux is the central flux."""

    def __init__(self, a=1.0):
        self.a = a

    def psi(self, u):
        return 0.5 * self.a * u[0] ** 2

    def flux(self, u):
        return self.a * u

    def flux_ec(self, uL, uR):
        return 0.5 * self.a * (uL + uR)

    def wavespeed(self, u):
        return np.abs(self.a) * np.ones_like(u[0])


class Euler1D:
    nvar = 3
    names = ["rho", "rho u", "E"]

    def __init__(self, gamma=1.4):
        self.gamma = gamma

    # --- primitive helpers -------------------------------------------------
    def pressure(self, u):
        rho, m, E = u
        return (self.gamma - 1.0) * (E - 0.5 * m * m / rho)

    def prim_to_cons(self, rho, vel, p):
        return np.array([rho, rho * vel, p / (self.gamma - 1.0) + 0.5 * rho * vel ** 2])

    # --- entropy -----------------------------------------------------------
    def entropy(self, u):
        rho = u[0]
        s = np.log(self.pressure(u)) - self.gamma * np.log(rho)
        return -rho * s

    def entropy_vars(self, u):
        g = self.gamma
        rho, m, E = u
        rhoe = E - 0.5 * m * m / rho
        s = np.log((g - 1.0) * rhoe) - g * np.log(rho)
        return np.array([(rhoe * (g + 1.0 - s) - E) / rhoe, m / rhoe, -rho / rhoe])

    def cons_vars(self, v):
        g = self.gamma
        v1, v2, v3 = v
        s = g - v1 + v2 * v2 / (2.0 * v3)
        rhoe = ((g - 1.0) / (-v3) ** g) ** (1.0 / (g - 1.0)) * np.exp(-s / (g - 1.0))
        return np.array([-rhoe * v3, rhoe * v2, rhoe * (1.0 - v2 * v2 / (2.0 * v3))])

    def psi(self, u):
        return (self.gamma - 1.0) * u[1]

    # --- fluxes ------------------------------------------------------------
    def flux(self, u):
        rho, m, E = u
        vel = m / rho
        p = self.pressure(u)
        return np.array([m, m * vel + p, vel * (E + p)])

    def flux_ec(self, uL, uR):
        """Chandrashekar entropy conservative flux."""
        g = self.gamma
        rhoL, rhoR = uL[0], uR[0]
        velL, velR = uL[1] / rhoL, uR[1] / rhoR
        betaL = rhoL / (2.0 * self.pressure(uL))
        betaR = rhoR / (2.0 * self.pressure(uR))

        rho_log = logmean(rhoL, rhoR)
        beta_log = logmean(betaL, betaR)
        rho_avg = 0.5 * (rhoL + rhoR)
        vel_avg = 0.5 * (velL + velR)
        beta_avg = 0.5 * (betaL + betaR)
        vel2_avg = 2.0 * vel_avg ** 2 - 0.5 * (velL ** 2 + velR ** 2)

        p_avg = rho_avg / (2.0 * beta_avg)
        E_avg = rho_log / (2.0 * beta_log * (g - 1.0)) + 0.5 * rho_log * vel2_avg

        f1 = rho_log * vel_avg
        f2 = f1 * vel_avg + p_avg
        f3 = (E_avg + p_avg) * vel_avg
        return np.array([f1, f2, f3])

    def dudv(self, u):
        """du/dv (symmetric positive definite), shape (3, 3, n).

        The standard symmetrizer for S = -rho*s/(gamma-1), divided by (gamma-1)
        because Chan's entropy is S = -rho*s.
        """
        g = self.gamma
        rho, m, E = u
        p = self.pressure(u)
        vel = m / rho
        H = E + p
        a2 = g * p / rho
        A = np.array([
            [rho, m, E],
            [m, m * vel + p, vel * H],
            [E, vel * H, H * H / rho - a2 * p / (g - 1.0)],
        ])
        return A / (g - 1.0)

    def wavespeed(self, u):
        rho = u[0]
        return np.abs(u[1] / rho) + np.sqrt(self.gamma * self.pressure(u) / rho)
