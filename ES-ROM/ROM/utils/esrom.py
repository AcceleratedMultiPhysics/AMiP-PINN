"""Unreduced entropy stable ROM: teacher and baseline (Chan 2020, Thm 2; plan Sec. 6).

    v~ = V V^T v(V a),  u~ = u(v~),
    dx da/dt = -V^T [ 2 (Q o F(u~)) 1 + eps K u~ ]   =   V^T rhs_FOM(u~) * dx

Since the entropy variables of u~ are exactly v~ = V v_N, the convective part
satisfies v_N^T V^T 2(Q o F)1 = v~^T 2(Q o F)1 = 0, and the viscous part gives
v~^T K u(v~) >= 0. With entropy_projection=False this is the plain POD-Galerkin
ROM (u~ = V a), which is not entropy stable.
"""
from .pod import project, reconstruct


class ESROM:
    def __init__(self, fom, V, entropy_projection=True):
        self.fom, self.V, self.phys = fom, V, fom.phys
        self.entropy_projection = entropy_projection
        self.nvar, self.r = self.phys.nvar, V.shape[1]
        self.n = self.nvar * self.r
        self.dx = fom.dx

    def project(self, u):
        return project(u, self.V)

    def reconstruct(self, a):
        return reconstruct(a, self.V, self.nvar)

    def v_N(self, a):
        """Projected entropy variables V^T v(V a)."""
        return self.project(self.phys.entropy_vars(self.reconstruct(a)))

    def states(self, a):
        """u~ = u(V V^T v(V a)), or V a without the entropy projection."""
        if not self.entropy_projection:
            return self.reconstruct(a)
        return self.phys.cons_vars(self.reconstruct(self.v_N(a)))

    def rhs(self, a):
        return self.project(self.fom.rhs(self.states(a)))

    def total_entropy(self, a):
        """S(a) = dx sum_i S((V a)_i)."""
        return self.fom.total_entropy(self.reconstruct(a))

    def entropy_residual(self, a):
        """v_N^T V^T 2(Q o F)1: round-off for the ES-ROM (Chan Fig. 4)."""
        return (self.v_N(a) * self.project(self.fom.conv(self.states(a)))).sum(-1)

    def entropy_rate(self, a):
        """dS/dt = dx v_N^T da/dt (plan eq. in Sec. 2)."""
        return self.dx * (self.v_N(a) * self.rhs(a)).sum(-1)
