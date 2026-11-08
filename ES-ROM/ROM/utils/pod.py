"""POD basis with the constant mode first (Chan 2020, end of Sec. 3; plan Sec. 5).

One basis V (K x r) is shared by every conserved variable (Kronecker
convention), so a reduced state is a = [a_1, ..., a_nvar], each of length r,
and n = nvar * r. Plain SVD gives V^T V = I, so the reduced mass matrix is
M = dx I and v_N(a) = V^T v(V a) (the inner product behind both must match).
"""
import torch


def build_basis(U, r, phys=None, entropy_enrich=True):
    """Orthonormal basis V = [1/sqrt(K), first r-1 POD modes of the rest].

    U: snapshots (Nt, nvar, K). With entropy_enrich the entropy variables
    v(U) are added as extra snapshot columns, so V also represents v well
    (needed for an accurate entropy projection). Returns (V, singular values).
    """
    K = U.shape[-1]
    X = U.reshape(-1, K).T
    if entropy_enrich:
        X = torch.cat([X, phys.entropy_vars(U).reshape(-1, K).T], 1)
    e = torch.full((K,), K ** -0.5, dtype=U.dtype, device=U.device)
    X = X - e[:, None] * (e @ X)[None, :]
    W, sigma, _ = torch.linalg.svd(X, full_matrices=False)
    W = W[:, :r - 1]
    W = W - e[:, None] * (e @ W)[None, :]            # remove round-off along 1
    V = torch.cat([e[:, None], torch.linalg.qr(W)[0]], 1)
    return V, sigma


def project(u, V):
    """(..., nvar, K) -> (..., nvar * r), a = V^T u per variable."""
    return (u @ V).flatten(-2)


def reconstruct(a, V, nvar):
    """(..., nvar * r) -> (..., nvar, K), u = V a per variable."""
    return a.unflatten(-1, (nvar, V.shape[1])) @ V.T


def conserved_indices(nvar, r):
    """Index of the constant-mode coefficient of each variable block.

    sum_i (V a)_{c,i} = sqrt(K) a_{c*r}, so total mass/momentum/energy are
    conserved iff these entries of da/dt vanish.
    """
    return [c * r for c in range(nvar)]
