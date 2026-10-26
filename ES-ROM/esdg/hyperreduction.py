"""Hyper-reduced quadrature for DG ROMs (Qu, Narayan & Chan 2026, Secs. 3.5, 4.5).

Every quadrature here approximates integrals in the DG inner product: the
reference weights are the global GLL weights M_Omega (not dx * 1 as in FV).
"""
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


def target_basis(V, mass, tol, trim="energy", add_one=True):
    """Target "V * V" of Sec. 3.5: products of basis columns (plus 1), with linearly
    dependent directions removed by an M-weighted SVD.

    All pairs (i, j) are represented (i <= j, cross terms scaled by sqrt 2).
    trim="energy": keep k modes with sqrt(sum_{j>k} mu_j^2 / sum mu_j^2) <= tol;
    trim="abs":    keep singular values mu_j > tol (as in the authors' code).
    Returned columns are M-orthonormal.
    """
    n, N = V.shape
    i, j = np.triu_indices(N)
    P = V[:, i] * V[:, j] * np.where(i == j, 1.0, np.sqrt(2.0))
    if add_one:
        P = np.hstack([np.ones((n, 1)), P])
    sq = np.sqrt(mass)
    U, mu, _ = np.linalg.svd(sq[:, None] * P, full_matrices=False)
    if trim == "abs":
        k = int(np.sum(mu > tol))
    else:
        m2 = mu ** 2
        E = np.sqrt(np.maximum(m2.sum() - np.cumsum(m2), 0.0) / m2.sum())
        k = int(np.argmax(E <= tol)) + 1 if np.any(E <= tol) else len(mu)
    return U[:, :k] / sq[:, None]


def greedy_cubature(Vtarget, wtarget, tol, max_pts=None, exclude=()):
    """Greedy empirical cubature (Algorithm 1 of Chan 2020, as in the authors' code):
    points I and positive weights w with Vtarget^T wtarget ~= Vtarget[I]^T w.

    Ties in the selection score (e.g. a one-column target) go to the row of largest
    norm; indices in `exclude` are never picked.
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


def small_eigvecs(Vt, I, w, eig_tol):
    """Point values Z = Vt z_j of eigenvectors of the HR test mass matrix with eigenvalue < eig_tol."""
    lam, z = np.linalg.eigh(Vt[I].T @ (w[:, None] * Vt[I]))
    return Vt @ z[:, lam < eig_tol]


def stabilize_stacked(Vt, Vtarget, wtarget, I, w, tol, eig_tol=1e-8, alpha=1e-2):
    """Stabilizing nodes as in the authors' code: if the test
    mass matrix has eigenvalues < eig_tol, rerun the greedy algorithm from scratch on
    the stacked target [Vtarget, sqrt(alpha) Z o Z]."""
    Z = small_eigvecs(Vt, I, w, eig_tol)
    if Z.shape[1] == 0:
        return I, w
    A = np.hstack([Vtarget, np.sqrt(alpha) * products(Z)])
    return greedy_cubature(A, wtarget, tol)


def stabilize_union(Vt, Vtarget, wtarget, I, w, tol, eig_tol=1e-3, alpha=1e-4,
                    w_min_frac=0.1, max_iter=10):
    """Stabilizing nodes as in our FV reproduction (Chan 2020, Sec. 4.3, modified):
    add greedy nodes for Z o Z outside I, refit all weights by min-norm least squares
    (bounded LS with w >= w_min if needed, never NNLS, which drops nodes), and repeat
    until no eigenvalue of the test mass matrix is below eig_tol."""
    b = Vtarget.T @ wtarget
    sa, w_min, Zcols = np.sqrt(alpha), w_min_frac * wtarget.min(), []
    for _ in range(max_iter):
        Z = small_eigvecs(Vt, I, w, eig_tol)
        if Z.shape[1] == 0:
            break
        Zcols.append(products(Z))
        IZ, _ = greedy_cubature(Zcols[-1], wtarget, tol, exclude=I)
        if len(IZ) == 0:
            break
        I = np.concatenate([I, IZ])
        Zt = np.hstack(Zcols)
        A = np.vstack([Vtarget[I].T, sa * Zt[I].T])
        rhs = np.concatenate([b, sa * (Zt.T @ wtarget)])
        w = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if w.min() <= w_min:
            w = lsq_linear(A, rhs, bounds=(w_min, np.inf), lsmr_tol="auto").x
    return I, w


def caratheodory_prune(A, w0):
    """Caratheodory pruning (Sec. 4.5, Algorithm 1): from a positive rule (rows of A,
    weights w0) integrating the k columns of A, remove nodes along null vectors of A^T
    until k remain. Moments A^T w are preserved to round-off and weights stay >= 0.
    Each step uses a window of k+1 remaining nodes, as in the paper's algorithm."""
    n, k = A.shape
    idx = list(range(n))
    w = np.asarray(w0, dtype=float).copy()
    while len(idx) > k:
        sub = np.array(idx[:k + 1])
        c = np.linalg.svd(A[sub].T)[2][-1]              # null vector of the k x (k+1) block
        small = 1e-12 * np.abs(c).max()                 # ignore round-off entries: dividing
        if not np.any(c > small):                       # by them blows the weights up
            c = -c
        pos = np.flatnonzero(c > small)
        ratios = w[sub[pos]] / c[pos]
        j = pos[np.argmin(ratios)]
        w[sub] = np.maximum(w[sub] - ratios.min() * c, 0.0)
        w[sub[j]] = 0.0
        idx.remove(sub[j])
    I = np.array(idx)
    keep = w[I] > 0
    return I[keep], w[I][keep]
