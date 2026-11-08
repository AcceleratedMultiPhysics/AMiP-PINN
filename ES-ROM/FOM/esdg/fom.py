"""Entropy conservative nodal DG full order model in 1D, periodic (Qu, Narayan & Chan 2026, Secs. 3.1-3.4, 5).

On each element: (p+1) Gauss-Lobatto nodes, diagonal mass M = J diag(w) and the
SBP operator Q = diag(w) D with Q + Q^T = B = diag(-1, 0, ..., 0, 1)  (eq. 4).
Global form (eqs. 9, 16):

    M_Omega du/dt + 2 (Q_Omega o F) 1 + eps K u = 0,

Q_Omega = 1/2 [S blocks on the diagonal, +-1 coupling between neighbouring end
nodes, including the periodic wrap-around], S = Q - Q^T (eq. 10), and
K = Q^T M^{-1} Q is the BR-1 Laplacian (eq. 39).
"""
import numpy as np
import scipy.sparse as sp
from numpy.polynomial import legendre as leg

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


def integrate(rhs, u0, dt, nsteps, save_every=1, callback=None):
    """Integrate du/dt = rhs(u) with low-storage RK4; returns (times, saved states)."""
    u = u0.copy()
    times, saved = [0.0], [u.copy()]
    for k in range(1, nsteps + 1):
        res = np.zeros_like(u)
        for a, b in zip(RK4A, RK4B):
            res = a * res + dt * rhs(u)
            u = u + b * res
        if callback is not None:
            callback(k, u)
        if k % save_every == 0 or k == nsteps:
            times.append(k * dt)
            saved.append(u.copy())
    return np.array(times), np.array(saved)


def gauss_lobatto(p):
    """Nodes, weights and differentiation matrix of the (p+1)-point GLL rule on [-1, 1].

    p = 0 gives one node with weight 2 and D = 0, i.e. the finite volume limit.
    """
    if p == 0:
        return np.zeros(1), np.array([2.0]), np.zeros((1, 1))
    cP = np.zeros(p + 1)
    cP[-1] = 1.0                                         # Legendre polynomial P_p
    r = np.concatenate([[-1.0], np.sort(leg.legroots(leg.legder(cP))), [1.0]])
    Pp = leg.legval(r, cP)
    w = 2.0 / (p * (p + 1) * Pp ** 2)
    D = np.zeros((p + 1, p + 1))                         # D_ij = l_j'(r_i)
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = Pp[i] / (Pp[j] * (r[i] - r[j]))
    D[0, 0] = -p * (p + 1) / 4.0
    D[p, p] = p * (p + 1) / 4.0
    return r, w, D


class DGFOM1D:
    """Periodic nodal ES-DG on n_el uniform elements of degree p; state shape (nvar, n_el*(p+1))."""

    def __init__(self, phys, n_el, p, domain=(-1.0, 1.0), eps=0.0):
        self.phys, self.n_el, self.p, self.eps = phys, n_el, p, eps
        a, b = domain
        h = (b - a) / n_el
        r, wq, Dr = gauss_lobatto(p)
        Np = p + 1
        self.Np, self.h, self.n = Np, h, n_el * Np
        J = h / 2.0
        centers = a + (np.arange(n_el) + 0.5) * h
        self.x = (centers[:, None] + J * r[None, :]).ravel()
        self.mass = np.tile(J * wq, n_el)                # diagonal of M_Omega

        Qref = wq[:, None] * Dr
        S = Qref - Qref.T
        Q = sp.kron(sp.identity(n_el), sp.csr_matrix(S), format="lil")
        last = np.arange(n_el) * Np + p                  # last node of each element
        for k in range(n_el - 1):
            Q[last[k], last[k] + 1] += 1.0
            Q[last[k] + 1, last[k]] -= 1.0
        Q[self.n - 1, 0] += 1.0                          # periodic wrap-around
        Q[0, self.n - 1] -= 1.0
        self.Q = (0.5 * Q).tocsr()                       # Q_Omega, eq. (10)
        self.Q.eliminate_zeros()
        Qc = self.Q.tocoo()
        self._rows, self._cols, self._vals = Qc.row, Qc.col, Qc.data
        self.Kmat = (self.Q.T @ sp.diags(1.0 / self.mass) @ self.Q).tocsr()   # eq. (39)

    def conv(self, u):
        """2(Q_Omega o F)1, evaluating f_EC only on the nonzeros of Q_Omega."""
        F = self.phys.flux_ec(u[:, self._rows], u[:, self._cols])
        r = np.zeros_like(u)
        for c in range(u.shape[0]):
            r[c] = np.bincount(self._rows, 2.0 * self._vals * F[c], minlength=self.n)
        return r

    def visc(self, u):
        return (self.Kmat @ u.T).T

    def rhs(self, u):
        r = self.conv(u)
        if self.eps:
            r = r + self.eps * self.visc(u)
        return -r / self.mass

    def total_entropy(self, u):
        return np.sum(self.mass * self.phys.entropy(u))

    def entropy_residual(self, u):
        """v^T 2(Q o F)1: zero by Theorem 2."""
        return np.sum(self.phys.entropy_vars(u) * self.conv(u))

    def stable_dt(self, u, cfl=1.5):
        """dt = cfl / (lambda_adv + lambda_visc), from
        lambda_adv <= 2 max|f'| ||M^{-1/2} Q M^{-1/2}||_2 and
        lambda_visc = eps rho(M^{-1/2} K M^{-1/2}) (symmetric eigensolver; ARPACK
        does not converge on the skew spectrum of M^{-1} Q)."""
        from scipy.sparse.linalg import eigsh
        Ms = sp.diags(1.0 / np.sqrt(self.mass))
        A = Ms @ self.Q @ Ms
        nrm = np.sqrt(eigsh(A.T @ A, k=1, which="LA", return_eigenvectors=False)[0])
        lam = 2.0 * self.phys.wavespeed(u).max() * nrm
        if self.eps:
            lam += self.eps * eigsh(Ms @ self.Kmat @ Ms, k=1, which="LA",
                                    return_eigenvectors=False)[0]
        return cfl / lam


def integrate_converged(rhs, u0, T, n0, rtol=1e-6, multiple=1, nsave=1, max_doublings=6):
    """Integrate to T, doubling the number of steps (from n0, rounded up to a
    multiple of `multiple`) until the final state changes by less than rtol
    relative; with 4th order RK the accepted run's time error is ~rtol/15.

    Choose rtol well below the ROM error you want to measure (1e-6 is plenty for
    errors >= 1e-5; advection at 1e-7 needs 1e-8). A stability-limited step alone
    is not enough for non-dissipative problems: the phase error at |lambda dt| ~ 1
    dominates. Stops after max_doublings with a warning.
    Returns (number of steps, times, states saved at nsave+1 equally spaced times).
    """
    n = multiple * int(np.ceil(n0 / multiple))
    n = nsave * int(np.ceil(n / nsave))
    tA, A = integrate(rhs, u0, T / n, n, n // nsave)
    for _ in range(max_doublings):
        tB, B = integrate(rhs, u0, T / (2 * n), 2 * n, 2 * n // nsave)
        n *= 2
        if np.linalg.norm(B[-1] - A[-1]) <= rtol * np.linalg.norm(B[-1]):
            return n, tB, B
        tA, A = tB, B
    print(f"integrate_converged: not converged to rtol={rtol:g} with {n} steps")
    return n, tA, A
