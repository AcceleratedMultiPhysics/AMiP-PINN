"""Conservation laws with an entropy conservative two-point flux (PyTorch).

States are stored component-first: shape (..., nvar, K). Every function
broadcasts over the leading axes, so a batch of reduced states reconstructed
as (B, nvar, K) goes through unchanged.

Entropy conventions follow Chan (2020) and FOM/esrom (plan Sec. 4):
  Burgers: S = u^2/2,  v = u,  psi = u^3/6.
  Euler:   S = -rho*s, s = log(p/rho^gamma), psi = (gamma-1)*rho*u.
"""
import torch

LOGMEAN_TOL = 1e-4


def logmean(a, b):
    """Logarithmic mean (a-b)/(log a - log b), evaluated stably (Ismail & Roe).

    torch.where evaluates both branches, so the log branch is fed a harmless
    zeta on the small-|f| entries ("double where"); otherwise its NaN gradient
    leaks through the unused branch when the flux is differentiated.
    """
    zeta = a / b
    f = (zeta - 1.0) / (zeta + 1.0)
    w = f * f
    small = w < LOGMEAN_TOL
    zeta_s = torch.where(small, torch.full_like(zeta, 2.0), zeta)
    f_s = (zeta_s - 1.0) / (zeta_s + 1.0)
    F_big = torch.log(zeta_s) / (2.0 * f_s)
    F_small = 1.0 + w / 3.0 + w * w / 5.0 + w * w * w / 7.0
    return (a + b) / (2.0 * torch.where(small, F_small, F_big))


class Burgers:
    nvar = 1
    names = ["u"]

    def entropy(self, u):
        return 0.5 * u[..., 0, :] ** 2

    def entropy_vars(self, u):
        return u.clone()

    def cons_vars(self, v):
        return v.clone()

    def psi(self, u):
        return u[..., 0, :] ** 3 / 6.0

    def entropy_flux(self, u):
        return u[..., 0, :] ** 3 / 3.0

    def flux(self, u):
        return 0.5 * u ** 2

    def flux_ec(self, uL, uR):
        return (uL * uL + uL * uR + uR * uR) / 6.0

    def wavespeed(self, u):
        return u[..., 0, :].abs()

    def is_admissible(self, u):
        """True for each leading index whose state is finite everywhere."""
        return torch.isfinite(u).all(-1).all(-1)


class Euler1D:
    nvar = 3
    names = ["rho", "rho u", "E"]

    def __init__(self, gamma=1.4):
        self.gamma = gamma

    # --- primitive helpers -------------------------------------------------
    def pressure(self, u):
        rho, m, E = u[..., 0, :], u[..., 1, :], u[..., 2, :]
        return (self.gamma - 1.0) * (E - 0.5 * m * m / rho)

    def prim_to_cons(self, rho, vel, p):
        return torch.stack([rho, rho * vel, p / (self.gamma - 1.0) + 0.5 * rho * vel ** 2], -2)

    # --- entropy -----------------------------------------------------------
    def entropy(self, u):
        rho = u[..., 0, :]
        s = torch.log(self.pressure(u)) - self.gamma * torch.log(rho)
        return -rho * s

    def entropy_vars(self, u):
        g = self.gamma
        rho, m, E = u[..., 0, :], u[..., 1, :], u[..., 2, :]
        rhoe = E - 0.5 * m * m / rho
        s = torch.log((g - 1.0) * rhoe) - g * torch.log(rho)
        return torch.stack([(rhoe * (g + 1.0 - s) - E) / rhoe, m / rhoe, -rho / rhoe], -2)

    def cons_vars(self, v):
        g = self.gamma
        v1, v2, v3 = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        s = g - v1 + v2 * v2 / (2.0 * v3)
        rhoe = ((g - 1.0) / (-v3) ** g) ** (1.0 / (g - 1.0)) * torch.exp(-s / (g - 1.0))
        return torch.stack([-rhoe * v3, rhoe * v2, rhoe * (1.0 - v2 * v2 / (2.0 * v3))], -2)

    def psi(self, u):
        return (self.gamma - 1.0) * u[..., 1, :]

    def entropy_flux(self, u):
        return self.entropy(u) * u[..., 1, :] / u[..., 0, :]

    # --- fluxes ------------------------------------------------------------
    def flux(self, u):
        rho, m, E = u[..., 0, :], u[..., 1, :], u[..., 2, :]
        vel = m / rho
        p = self.pressure(u)
        return torch.stack([m, m * vel + p, vel * (E + p)], -2)

    def flux_ec(self, uL, uR):
        """Chandrashekar entropy conservative flux (Chan 2018, eq. 106)."""
        g = self.gamma
        rhoL, rhoR = uL[..., 0, :], uR[..., 0, :]
        velL, velR = uL[..., 1, :] / rhoL, uR[..., 1, :] / rhoR
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
        return torch.stack([f1, f2, f3], -2)

    def wavespeed(self, u):
        rho = u[..., 0, :]
        return (u[..., 1, :] / rho).abs() + torch.sqrt(self.gamma * self.pressure(u) / rho)

    def is_admissible(self, u):
        """True for each leading index with rho > 0 and p > 0 everywhere."""
        ok = (u[..., 0, :] > 0) & (self.pressure(u) > 0)
        return ok.all(-1) & torch.isfinite(u).all(-1).all(-1)
