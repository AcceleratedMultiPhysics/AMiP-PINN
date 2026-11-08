"""Entropy stable reduced order models (Chan 2020, Secs. 3-5).

Reduced states are stored as (nvar, N): one shared basis V (K x N) for every
component, as in the paper.

GalerkinROM  -- Theorem 2 (no hyper-reduction; cost scales with K)
HyperROM     -- Theorem 3 (periodic) / Theorem 4 (reflective walls)
"""
import numpy as np

from . import hyperreduction as hr
from .fom import wall_fluxes
from .operators import edges


# --------------------------------------------------------------------------
# Bases
# --------------------------------------------------------------------------
def snapshot_svd(snaps, phys, entropy_enrich=True):
    """SVD of the snapshot matrix with columns = components of u (and v(u)).

    snaps: (nsnap, nvar, K). Returns (U, sigma).
    """
    cols = [snaps[:, c, :].T for c in range(phys.nvar)]
    if entropy_enrich:
        vs = np.array([phys.entropy_vars(s) for s in snaps])
        cols += [vs[:, c, :].T for c in range(phys.nvar)]
    U, s, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
    return U, s


def pod_basis(U, N):
    """Orthonormal basis for span{1, U[:, :N-1]} so that 1 is in range(V)."""
    A = np.hstack([np.ones((U.shape[0], 1)), U[:, :N - 1]])
    return np.linalg.qr(A)[0]


def orth(A, dx, rtol=1e-10):
    """dx-orthonormal basis (dx * Vt^T Vt = I) for range(A), columns normalised first.

    Round-off columns (e.g. Q applied to the constant mode) are dropped before
    normalisation, otherwise they would inject noise directions into the basis.
    """
    nrm = np.linalg.norm(A, axis=0)
    A = A[:, nrm > 1e-12 * nrm.max()] / nrm[nrm > 1e-12 * nrm.max()]
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > rtol * s[0]] / np.sqrt(dx)


def test_basis(V, Q, dx, enrich=True):
    """range(Vt) = range([1, V, QV]) (Sec. 4.1.1), or [1, V] if enrich=False."""
    blocks = [np.ones((V.shape[0], 1)), V]
    if enrich:
        blocks.append(Q @ V)
    return orth(np.hstack(blocks), dx)


# --------------------------------------------------------------------------
# Theorem 2: entropy conservative Galerkin ROM without hyper-reduction
# --------------------------------------------------------------------------
class GalerkinROM:
    def __init__(self, fom, V, entropy_projection=True):
        self.fom, self.V, self.ep = fom, V, entropy_projection
        self.phys = fom.phys
        G = V.T @ V
        self.Minv = np.linalg.inv(fom.dx * G)
        self.Vdag = np.linalg.solve(G, V.T)

    def project(self, u):
        return u @ self.Vdag.T

    def reconstruct(self, uN):
        return uN @ self.V.T

    def _states(self, uN):
        u = uN @ self.V.T
        if not self.ep:
            return None, u
        vN = self.phys.entropy_vars(u) @ self.Vdag.T
        return vN, self.phys.cons_vars(vN @ self.V.T)

    def rhs(self, uN):
        _, ut = self._states(uN)
        r = self.fom.conv(ut)
        if self.fom.eps:
            r = r + self.fom.eps * self.fom.visc(ut)
        return -(r @ self.V) @ self.Minv

    def entropy_residual(self, uN):
        vN, ut = self._states(uN)
        return np.sum(vN * (self.fom.conv(ut, lxf=False) @ self.V))


# --------------------------------------------------------------------------
# Theorems 3/4: entropy conservative hyper-reduced ROM
# --------------------------------------------------------------------------
class HyperROM:
    def __init__(self, fom, V, Vt, I, w, visc="22", ID=None, wD=None):
        self.fom, self.phys, self.V = fom, fom.phys, V
        self.I, self.w, self.visc = I, w, visc
        VI = V[I]
        self.MN = VI.T @ (w[:, None] * VI)
        self.MNinv = np.linalg.inv(self.MN)
        self.P = self.MNinv @ (VI.T * w)

        VtI = Vt[I]
        self.Mt = VtI.T @ (w[:, None] * VtI)
        self.Pt = np.linalg.solve(self.Mt, VtI.T * w)
        Qhat = Vt.T @ (fom.Q @ Vt)
        self.Qt = self.Pt.T @ Qhat @ self.Pt

        if fom.periodic:
            self.Vh, self.Qh = VI, self.Qt
        else:
            bidx = [0, fom.K - 1]
            self.Vb = V[bidx]
            E = Vt[bidx] @ self.Pt
            B = np.diag([-1.0, 1.0])
            self.Bdiag = np.array([-1.0, 1.0])
            self.Qh = 0.5 * np.block([[self.Qt - self.Qt.T, E.T @ B],
                                      [-B @ E, B]])
            self.Vh = np.vstack([VI, self.Vb])

        if visc in ("20", "22"):
            left, right = edges(fom.K, fom.periodic)
            self.ID, self.wD = ID, wD
            self.VL, self.VR = V[left[ID]], V[right[ID]]
            self.DVs = (self.VL - self.VR) / fom.dx          # D(I_D,:) V
        elif visc == "23":
            self.VKV = V.T @ (fom.Kmat @ V)

    # ---- helpers ----------------------------------------------------------
    def project(self, u):
        return u @ np.linalg.pinv(self.V).T

    def reconstruct(self, uN):
        return uN @ self.V.T

    def _states(self, uN):
        """v_N = P v(V(I,:) u_N) and entropy projected states u~ at Vh points."""
        vN = self.phys.entropy_vars(uN @ self.V[self.I].T) @ self.P.T
        return vN, self.phys.cons_vars(vN @ self.Vh.T)

    def _conv(self, uh, lxf=True):
        F = self.phys.flux_ec(uh[:, :, None], uh[:, None, :])
        r = 2.0 * np.sum(self.Qh * F, axis=2) @ self.Vh
        if not self.fom.periodic:
            ub = uh[:, -2:]
            fL, fR = wall_fluxes(self.phys, ub[:, 0], ub[:, 1], lxf)
            fs = np.stack([fL, fR], axis=1)
            r = r + ((fs - self.phys.flux(ub)) * self.Bdiag) @ self.Vb
        return r

    def dissipation_term(self, vN, uh):
        """Hyper-reduced viscous term d(u_N), shape (nvar, N)."""
        if self.visc == "22":
            vL, vR = vN @ self.VL.T, vN @ self.VR.T
            ubar = 0.5 * (self.phys.cons_vars(vL) + self.phys.cons_vars(vR))
            H = self.phys.dudv(ubar)
            q = self.wD * np.einsum("abn,bn->an", H, (vL - vR) / self.fom.dx)
            return q @ self.DVs
        if self.visc == "20":
            uL = self.phys.cons_vars(vN @ self.VL.T)
            uR = self.phys.cons_vars(vN @ self.VR.T)
            return (self.wD * (uL - uR) / self.fom.dx) @ self.DVs
        if self.visc == "23":
            return (uh[:, :len(self.I)] @ self.P.T) @ self.VKV
        raise ValueError(self.visc)

    # ---- public -----------------------------------------------------------
    def rhs(self, uN):
        vN, uh = self._states(uN)
        r = self._conv(uh)
        if self.fom.eps and self.visc:
            r = r + self.fom.eps * self.dissipation_term(vN, uh)
        return -r @ self.MNinv

    def entropy_residual(self, uN):
        """v_N^T [2 Vh^T (Qh o F) 1 + Vb^T B (f*_EC - f)] -- zero by Theorems 3/4."""
        vN, uh = self._states(uN)
        return np.sum(vN * self._conv(uh, lxf=False))

    def entropy_dissipation(self, uN):
        """v_N^T d(u_N) >= 0 for (20) and (22)."""
        vN, uh = self._states(uN)
        return np.sum(vN * self.dissipation_term(vN, uh))

    def sampled_entropy(self, uN):
        return np.sum(self.w * self.phys.entropy(uN @ self.V[self.I].T))


# --------------------------------------------------------------------------
# Offline stage
# --------------------------------------------------------------------------
def build_hyper_rom(fom, V, tol, visc="22", enrich=True, eig_tol=1e-3, alpha_Z=1e-4,
                    w_min_frac=1e-1, stabilize=True, target="mass"):
    """Offline recipe of Sec. 4.4. Returns (rom, info).

    target="mass" integrates products V o V (paper, eq. 17); target="test_mass"
    integrates Vt o Vt, i.e. the hyper-reduced test mass matrix itself.
    Stabilization defaults (eig_tol, alpha_Z, w_min_frac) were tuned on 1D Euler,
    N=75: the paper's alpha_Z=1e-2 adds many points and spoils the mass matrix.
    """
    K, dx = fom.K, fom.dx
    Vt = test_basis(V, fom.Q, dx, enrich)

    Vtarget = hr.compress_products(V if target == "mass" else Vt, tol)
    wtarget = dx * np.ones(K)
    I, w = hr.greedy_cubature(Vtarget, wtarget, tol)
    n_cub = len(I)
    n_stab = 0
    if stabilize:
        I, w, n_stab = hr.stabilize(Vt, Vtarget, wtarget, I, w, tol, eig_tol, alpha_Z,
                                    w_min_frac=w_min_frac)

    ID = wD = None
    if visc in ("20", "22"):
        DV = fom.D @ V
        U, s, _ = np.linalg.svd(DV, full_matrices=False)
        VD = U[:, s > 1e-12 * s[0]]
        VDtarget = hr.compress_products(VD, tol)
        ID, wD = hr.greedy_cubature(VDtarget, dx * np.ones(DV.shape[0]), tol)

    rom = HyperROM(fom, V, Vt, I, w, visc, ID, wD)
    info = dict(n_modes=V.shape[1], n_test=Vt.shape[1], n_cubature=n_cub,
                n_stabilizing=n_stab, n_points=len(I),
                n_viscous=0 if ID is None else len(ID),
                cond_test_mass=np.linalg.cond(rom.Mt))
    return rom, info
