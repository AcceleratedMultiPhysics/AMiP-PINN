"""1D finite volume matrices from Chan (2020), Sections 2 and 5.

Q (eq. 5 / 24) is the 1/2-scaled central difference; D is the two-point
difference (1/dx)[1 -1] on each edge, and the viscosity matrix is
K = dx * D^T D = (1/dx) * tridiag(-1, 2, -1), the scaling used in eq. (8).
"""
import numpy as np
import scipy.sparse as sp


def Q_periodic(K):
    e = np.ones(K)
    Q = sp.diags([0.5 * e[:-1], -0.5 * e[:-1]], [1, -1], format="lil")
    Q[0, K - 1] = -0.5
    Q[K - 1, 0] = 0.5
    return Q.tocsr()


def Q_sbp(K):
    """Non-periodic Q with Q + Q^T = B = diag(-1, 0, ..., 0, 1) (eq. 24)."""
    e = np.ones(K)
    Q = sp.diags([0.5 * e[:-1], -0.5 * e[:-1]], [1, -1], format="lil")
    Q[0, 0] = -0.5
    Q[K - 1, K - 1] = 0.5
    return Q.tocsr()


def edges(K, periodic):
    """Left/right cell index of each edge used by D."""
    left = np.arange(K if periodic else K - 1)
    return left, (left + 1) % K


def D_matrix(K, dx, periodic):
    left, right = edges(K, periodic)
    n = len(left)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([left, right])
    vals = np.concatenate([np.ones(n), -np.ones(n)]) / dx
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, K))


def K_matrix(K, dx, periodic):
    D = D_matrix(K, dx, periodic)
    return (dx * (D.T @ D)).tocsr()
