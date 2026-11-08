"""Entropy stable DG ROMs in 1D, periodic (Qu, Narayan & Chan 2026, Secs. 3.4-3.5, 5).

Everything is measured in the DG inner product <a, b>_M = a^T M_Omega b:
  POD modes        SVD of sqrt(M) X, modes sqrt(M)^{-1} U          (Sec. 3.4.2)
  projection       V^dagger = (V^T M V)^{-1} V^T M
  entropy proj.    u~ = u(V V^dagger v(V u_N))                       (eq. 13)
  test basis       range(Vt) = range([1, V, M^{-1} Q^T V])           (eq. 15)
  HR quadrature    targets V * V with reference weights M            (Sec. 3.5)
  viscosity        K_N = V^T Q^T M^{-1} Q V (BR-1)                   (eq. 40)
Reduced states are stored as (nvar, N); one basis V is shared by all components.
"""
import numpy as np

from . import hyperreduction as hr


# --------------------------------------------------------------------------
# Bases
# --------------------------------------------------------------------------
def snapshot_svd(snaps, phys, mass, entropy_enrich=True):
    """M-weighted POD of [u snapshots, v(u) snapshots]; snaps has shape (nsnap, nvar, n).
    Returns (modes, sigma) with modes M-orthonormal."""
    cols = [snaps[:, c, :].T for c in range(phys.nvar)]
    if entropy_enrich:
        vs = np.array([phys.entropy_vars(s) for s in snaps])
        cols += [vs[:, c, :].T for c in range(phys.nvar)]
    sq = np.sqrt(mass)
    U, s, _ = np.linalg.svd(sq[:, None] * np.hstack(cols), full_matrices=False)
    return U / sq[:, None], s


def m_orth(A, mass, rtol=1e-12):
    """M-orthonormal basis (B^T M B = I) of range(A). Columns are normalised first and
    round-off columns (e.g. Q applied to a constant) are dropped."""
    sq = np.sqrt(mass)
    nrm = np.linalg.norm(sq[:, None] * A, axis=0)
    keep = nrm > 1e-12 * nrm.max()
    U, s, _ = np.linalg.svd(sq[:, None] * A[:, keep] / nrm[keep], full_matrices=False)
    return U[:, s > rtol * s[0]] / sq[:, None]


def test_basis(V, Q, mass, enrich="MinvQtV"):
    """range(Vt) = range([1, V, X]) with
    X = M^{-1} Q^T V ("MinvQtV", Qu et al. eq. 15), Q V ("QV", Chan 2020 eq. 14),
    both ("both"), or nothing (None)."""
    blocks = [np.ones((V.shape[0], 1)), V]
    if enrich in ("QV", "both"):
        blocks.append(Q @ V)
    if enrich in ("MinvQtV", "both"):
        blocks.append((Q.T @ V) / mass[:, None])
    return m_orth(np.hstack(blocks), mass)


# --------------------------------------------------------------------------
# Viscosity in reduced form (Sec. 5)
# --------------------------------------------------------------------------
class _Viscosity:
    """visc="naive": K_N u_N (eq. 40, used by the authors).
    visc="es": V^T Q^T M^{-1} H Q V v_N with H = du/dv(u~) at the nodes (eq. 43),
    provably entropy dissipative. Not hyper-reduced: it costs O(n) per evaluation."""

    def __init__(self, fom, V, mode):
        self.fom, self.V, self.mode = fom, V, mode
        if mode == "naive":
            self.KN = V.T @ (fom.Kmat @ V)
        elif mode == "es":
            self.QV = (fom.Q @ V)

    def term(self, uN, vN):
        if self.mode == "naive":
            return uN @ self.KN
        ut = self.fom.phys.cons_vars(vN @ self.V.T)
        H = self.fom.phys.dudv(ut)
        g = (vN @ self.QV.T) / self.fom.mass                 # M^{-1} Q V v_N
        return np.einsum("abn,bn->an", H, g) @ self.QV

    def dissipation(self, uN, vN):
        """v_N^T d: >= 0 by construction for "es"; observed >= 0 for "naive"."""
        return np.sum(vN * self.term(uN, vN))


# --------------------------------------------------------------------------
# Theorem 3: entropy conservative ROM without hyper-reduction (eq. 13)
# --------------------------------------------------------------------------
class GalerkinROM:
    def __init__(self, fom, V, visc="naive"):
        self.fom, self.phys, self.V = fom, fom.phys, V
        MV = fom.mass[:, None] * V
        self.MNinv = np.linalg.inv(V.T @ MV)
        self.Vdag = self.MNinv @ MV.T
        self.visc = _Viscosity(fom, V, visc)

    def project(self, u):
        return u @ self.Vdag.T

    def reconstruct(self, uN):
        return uN @ self.V.T

    def _states(self, uN):
        vN = self.phys.entropy_vars(uN @ self.V.T) @ self.Vdag.T
        return vN, self.phys.cons_vars(vN @ self.V.T)

    def rhs(self, uN):
        vN, ut = self._states(uN)
        r = self.fom.conv(ut) @ self.V
        if self.fom.eps:
            r = r + self.fom.eps * self.visc.term(uN, vN)
        return -r @ self.MNinv

    def entropy_residual(self, uN):
        vN, ut = self._states(uN)
        return np.sum(vN * (self.fom.conv(ut) @ self.V))


# --------------------------------------------------------------------------
# Hyper-reduced ROM: eq. (13) + Sec. 3.5
# --------------------------------------------------------------------------
class HyperROM:
    def __init__(self, fom, V, Vt, I, w, visc="naive"):
        self.fom, self.phys, self.V, self.Vt, self.I, self.w = fom, fom.phys, V, Vt, I, w
        VI = V[I]
        self.MN = VI.T @ (w[:, None] * VI)                 # hyper-reduced mass matrix
        self.MNinv = np.linalg.inv(self.MN)
        self.P = self.MNinv @ (VI.T * w)                   # P_N
        VtI = Vt[I]
        self.Mt = VtI.T @ (w[:, None] * VtI)               # hyper-reduced test mass matrix
        self.Pt = np.linalg.solve(self.Mt, VtI.T * w)      # P_t
        self.Qhat = Vt.T @ (fom.Q @ Vt)                    # compressed Q_t^hat
        self.Qbar = self.Pt.T @ self.Qhat @ self.Pt        # Qbar = P_t^T Q^hat P_t
        self.Vh, self.Qh = VI, self.Qbar
        self.visc = _Viscosity(fom, V, visc)
        # F is symmetric: evaluate the flux only for pairs i < j (half the work)
        nh = self.Qh.shape[0]
        self._iu, self._ju = np.triu_indices(nh, k=1)
        self._qij, self._qji = self.Qh[self._iu, self._ju], self.Qh[self._ju, self._iu]
        self._qii = np.diag(self.Qh).copy()

    def project(self, u):
        MV = self.fom.mass[:, None] * self.V
        return u @ np.linalg.solve(self.V.T @ MV, MV.T).T

    def reconstruct(self, uN):
        return uN @ self.V.T

    def _states(self, uN):
        """v_N = P_N v(V(I) u_N), u~ = u(V_h v_N) at HR nodes (and boundary nodes)."""
        vN = self.phys.entropy_vars(uN @ self.V[self.I].T) @ self.P.T
        return vN, self.phys.cons_vars(vN @ self.Vh.T)

    def _conv(self, uh):
        """2 V(I)^T (Qbar o F) 1, F_ij = f_EC(u~_i, u~_j)."""
        nh = uh.shape[1]
        Fu = self.phys.flux_ec(uh[:, self._iu], uh[:, self._ju])      # pairs i < j
        rowsum = self._qii * self.phys.flux(uh)                       # F_ii = f(u~_i)
        for c in range(uh.shape[0]):
            rowsum[c] += (np.bincount(self._iu, self._qij * Fu[c], minlength=nh)
                          + np.bincount(self._ju, self._qji * Fu[c], minlength=nh))
        return 2.0 * rowsum @ self.Vh

    def rhs(self, uN):
        vN, uh = self._states(uN)
        r = self._conv(uh)
        if self.fom.eps:
            r = r + self.fom.eps * self.visc.term(uN, vN)
        return -r @ self.MNinv

    def entropy_residual(self, uN):
        """v_N^T 2 V(I)^T (Qbar o F) 1: zero because Qbar is skew with zero row sums."""
        vN, uh = self._states(uN)
        return np.sum(vN * self._conv(uh))

    def viscous_dissipation(self, uN):
        vN, _ = self._states(uN)
        return self.fom.eps * self.visc.dissipation(uN, vN)


# --------------------------------------------------------------------------
# Offline stage
# --------------------------------------------------------------------------
def build_hyper_rom(fom, V, tol, enrich="MinvQtV", target="mass", trim="energy",
                    selection="greedy", stab="stacked", visc="naive", stab_kwargs=None):
    """Offline stage of Sec. 3.5. Returns (rom, info).

    enrich:    test-basis enrichment (see test_basis).
    target:    "mass" = V * V (paper), "test_mass" = Vt * Vt.
    trim:      target compression, "energy" or "abs" (see hyperreduction.target_basis).
    selection: "greedy" (Algorithm 1 of Chan 2020), "caratheodory" (prune the full
               GLL rule to dim(target) nodes, Sec. 4.5), "all" (ideal hyper-reduction).
    stab:      "stacked" (authors' code), "union" (our FV variant) or None.
    """
    mass = fom.mass
    Vt = test_basis(V, fom.Q, mass, enrich)
    Vtarget = hr.target_basis(V if target == "mass" else Vt, mass, tol, trim)
    if selection == "greedy":
        I, w = hr.greedy_cubature(Vtarget, mass, tol)
    elif selection == "caratheodory":
        I, w = hr.caratheodory_prune(Vtarget, mass)
    elif selection == "all":
        I, w, stab = np.arange(fom.n), mass.copy(), None
    else:
        raise ValueError(selection)
    n_sel = len(I)
    if stab == "stacked":
        I, w = hr.stabilize_stacked(Vt, Vtarget, mass, I, w, tol, **(stab_kwargs or {}))
    elif stab == "union":
        I, w = hr.stabilize_union(Vt, Vtarget, mass, I, w, tol, **(stab_kwargs or {}))
    rom = HyperROM(fom, V, Vt, I, w, visc)
    info = dict(n_modes=V.shape[1], n_test=Vt.shape[1], n_target=Vtarget.shape[1],
                n_selected=n_sel, n_nodes=len(I), cond_Mt=np.linalg.cond(rom.Mt))
    return rom, info


def rom_stable_dt(rom, samples, cfl=1.5, h=1e-7):
    """Explicit time step from the ROM's own Jacobian (not the FOM's).

    The ROM drops the fine scales that limit the FOM step, so it can take much
    larger steps (Qu et al., Sec. 6.3.3). The Jacobian is formed by finite
    differences at each sample reduced state; dt = cfl / max spectral radius.
    """
    rho = 0.0
    for uN in samples:
        f0 = rom.rhs(uN).ravel()
        n = f0.size
        J = np.empty((n, n))
        flat = uN.ravel()
        for k in range(n):
            step = h * max(1.0, abs(flat[k]))
            du = np.zeros(n)
            du[k] = step
            J[:, k] = (rom.rhs(uN + du.reshape(uN.shape)).ravel() - f0) / step
        rho = max(rho, np.abs(np.linalg.eigvals(J)).max())
    return cfl / rho
