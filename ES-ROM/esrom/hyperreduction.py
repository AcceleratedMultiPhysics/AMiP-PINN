"""Empirical cubature for hyper-reduction (Chan 2020, Secs. 4.2-4.3)."""
import numpy as np
from scipy.optimize import lsq_linear, nnls


def energy_tol(sigma, N):
    """tol = sqrt(sum_{j>N} s_j^2 / sum_j s_j^2) from snapshot singular values."""
    s2 = sigma ** 2
    return np.sqrt(s2[N:].sum() / s2.sum())


def products(V):
    """Columns V(:,i) o V(:,j), i <= j."""
    i, j = np.triu_indices(V.shape[1])
    return V[:, i] * V[:, j]


def compress_products(V, tol):
    """compress(products(V), tol) without forming the K x N^2 product matrix.

    The products of all pairs (i, j) have Gram matrix (V V^T) o (V V^T), whose
    eigenvectors are the left singular vectors of the product matrix.
    """
    G = V @ V.T
    m2, U = np.linalg.eigh(G * G)
    m2, U = np.maximum(m2[::-1], 0.0), U[:, ::-1]
    return _truncate(U, m2, tol)


def compress(Vtarget, tol):
    """Leading left singular vectors U[:, :k], k smallest with E_k <= tol (eq. 18)."""
    U, mu, _ = np.linalg.svd(Vtarget, full_matrices=False)
    return _truncate(U, mu ** 2, tol)


def _truncate(U, m2, tol):
    E = np.sqrt(np.maximum(m2.sum() - np.cumsum(m2), 0.0) / m2.sum())
    k = int(np.argmax(E <= tol)) + 1 if np.any(E <= tol) else len(m2)
    return U[:, :k]


def greedy_cubature(Vtarget, wtarget, tol, max_pts=None, exclude=()):
    """Algorithm 1: greedy points I and positive weights w with
    Vtarget^T wtarget ~= Vtarget[I]^T w. Indices in `exclude` are never picked.
    """
    n = Vtarget.shape[0]
    max_pts = n if max_pts is None else max_pts
    b = Vtarget.T @ wtarget
    nb = np.linalg.norm(b)
    rnorm = np.linalg.norm(Vtarget, axis=1)
    rows = Vtarget / np.maximum(rnorm, 1e-300)[:, None]
    I, w, r = [], np.zeros(0), b.copy()
    blocked = np.zeros(n, bool)
    blocked[list(exclude)] = True
    while np.linalg.norm(r) / nb > tol and len(I) < max_pts:
        score = rows @ (r / np.linalg.norm(r))
        score[blocked] = -np.inf
        # ties (always the case for a 1-column target) go to the largest row
        tied = np.flatnonzero(score >= score.max() - 1e-12)
        i = int(tied[np.argmax(rnorm[tied])])
        I.append(i)
        blocked[i] = True
        A = Vtarget[I].T
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        if w.min() <= 0:
            w = nnls(A, b, maxiter=100 * len(I))[0]
        r = b - A @ w
    I, w = np.array(I), np.asarray(w)
    keep = w > 0
    return I[keep], w[keep]


def stabilize(Vt, Vtarget, wtarget, I, w, tol, eig_tol=1e-3, alpha_Z=1e-4, max_iter=10,
              w_min_frac=1e-1):
    """Add stabilizing points so the test mass matrix Vt[I]^T W Vt[I] is
    well conditioned (Sec. 4.3).

    Vt is assumed orthonormal w.r.t. diag(wtarget), so the exact test mass matrix
    is the identity; eigenvectors with eigenvalue < eig_tol are treated as "small".

    Two departures from a literal reading of the paper, both needed in practice:
    stabilizing points are chosen among points not already in I, and the joint
    weights are the min-norm least squares solution, or bounded least squares
    with w >= w_min if that has a small weight. (NNLS, as in the paper, returns
    sparse weights and so drops most of the points just added.)
    The procedure is repeated until no small eigenvalues remain (or max_iter).
    Returns (I, w, number of stabilizing points added).
    """
    b = Vtarget.T @ wtarget
    sa = np.sqrt(alpha_Z)
    w_min = w_min_frac * wtarget.min()
    n0, Zcols = len(I), []
    for _ in range(max_iter):
        Mt = Vt[I].T @ (w[:, None] * Vt[I])
        lam, z = np.linalg.eigh(Mt)
        small = lam < eig_tol
        if not np.any(small):
            break
        Z = Vt @ z[:, small]
        Zcols.append(products(Z))
        IZ, _ = greedy_cubature(Zcols[-1], wtarget, tol, exclude=I)
        if len(IZ) == 0:
            break
        I = np.concatenate([I, IZ])
        Ztarget = np.hstack(Zcols)
        A = np.vstack([Vtarget[I].T, sa * Ztarget[I].T])
        rhs = np.concatenate([b, sa * (Ztarget.T @ wtarget)])
        w = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if w.min() <= w_min:
            w = lsq_linear(A, rhs, bounds=(w_min, np.inf), lsmr_tol="auto").x
    return I, w, len(I) - n0
