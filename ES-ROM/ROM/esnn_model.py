"""Entropy stable neural ROM: data, networks and trainer (plan Secs. 3, 5, 7).

Structured model (entropy stable for any weights):

    M da/dt = (J(a) - R(a)) v_N(a),   J = D (U - U^T) D,  R = D L L^T D,  M = dx I,

with D = diag(d) > 0 a fixed per-mode scaling, so with w = D v_N:
dS/dt = v_N^T M da/dt = -|L^T w|^2 <= 0 and v_N^T J v_N = w^T (U - U^T) w = 0.
v_N(a) = V^T v(V a) is computed from the physics, never learned. The rows and
columns of the conserved (constant-mode) coefficients are removed, so total
mass/momentum/energy are exactly conserved.

Unconstrained model: the same MLP outputs da/dt directly (neural ODE baseline).

Atakan Aygun - AMiP
"""
from esnn_config import *
import torch.nn as nn

from utils.physics import Burgers, Euler1D
from utils.fom import FOM1D, generate_snapshots, integrate, rk45_step
from utils.pod import build_basis, project, reconstruct, conserved_indices
from utils.esrom import ESROM


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Data: FOM snapshots and POD basis
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def make_fom():
    phys = Burgers() if PROBLEM == "burgers" else Euler1D(GAMMA)
    return FOM1D(phys, K, DOMAIN, EPS, device=device)


class ReducedProblem:
    """FOM snapshots projected on V, plus the ES-ROM teacher and the POD-Galerkin ROM."""

    def __init__(self, fom, V, data, t_train, sigma=None):
        self.fom, self.phys, self.V, self.sigma = fom, fom.phys, V, sigma
        self.dx = fom.dx
        self.nvar, self.r = self.phys.nvar, V.shape[1]
        self.n = self.nvar * self.r
        self.conserved = conserved_indices(self.nvar, self.r)
        self.teacher = ESROM(fom, V)
        self.galerkin = ESROM(fom, V, entropy_projection=False)

        self.t = np.asarray(data["t"])
        self.U, self.dU = data["U"], data["dU"]          # (Nt, nvar, K)
        self.D_fom, self.S_fom = data["D"], data["S"]    # (Nt,)
        self.A = project(self.U, V)                       # a_k = V^T u(t_k)
        self.Adot = project(self.dU, V)                   # exact projected time derivative
        self.train_mask = torch.as_tensor(self.t <= t_train + 1e-12, device=self.U.device)

    def reconstruct(self, a):
        return reconstruct(a, self.V, self.nvar)

    def projection_error(self):
        """Best possible relative L2 error of any ROM in span(V), per snapshot."""
        Up = self.reconstruct(self.A)
        return tonp((Up - self.U).norm(dim=(-2, -1)) / self.U.norm(dim=(-2, -1)))


def load_problem(regenerate=False):
    """FOM data and basis, cached in save_folder so train and post use the same V."""
    fom = make_fom()
    os.makedirs(save_folder, exist_ok=True)
    path = save_folder + "fom.npz"
    meta = dict(K=K, EPS=EPS, DT=DT, T_FINAL=T_FINAL, T_TRAIN=T_TRAIN,
                SAVE_EVERY=SAVE_EVERY, R_MODES=R_MODES, ENRICH=ENTROPY_ENRICH)

    f = np.load(path) if os.path.exists(path) and not regenerate else None
    if f is None or not all(k in f and np.isclose(f[k], v) for k, v in meta.items()):
        u0 = initial_condition(fom.phys, fom.x)
        nsteps = round(T_FINAL / DT)
        start = time.time()
        data = generate_snapshots(fom, u0, DT, nsteps, SAVE_EVERY)
        print(f"FOM: K={K}, dt={DT:g}, {nsteps} steps, {len(data['t'])} snapshots, "
              f"{time.time() - start:.1f} s")
        train = torch.as_tensor(data["t"] <= T_TRAIN + 1e-12, device=data["U"].device)
        V, sigma = build_basis(data["U"][train], R_MODES, fom.phys, ENTROPY_ENRICH)
        np.savez(path, V=tonp(V), sigma=tonp(sigma),
                 **{k: tonp(v) for k, v in data.items()}, **meta)
        f = np.load(path)

    data = {k: torch.as_tensor(f[k], device=device) for k in ("U", "dU", "D", "S")}
    data["t"] = f["t"]
    V = torch.as_tensor(f["V"], device=device)
    return ReducedProblem(fom, V, data, T_TRAIN, sigma=f["sigma"])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Networks
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
ACTS = {"tanh": nn.Tanh, "gelu": nn.GELU}


def mlp(n_in, n_out, width, depth, act):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), ACTS[act]()]
        d = width
    layers.append(nn.Linear(d, n_out))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return net


class ReducedNet(nn.Module):
    """Holds V, the physics and the input normalization; computes v_N(a)."""

    def __init__(self, phys, V, dx, conserved, n_mu=0):
        super().__init__()
        self.phys, self.dx, self.nvar, self.n_mu = phys, dx, phys.nvar, n_mu
        self.r = V.shape[1]
        self.n = self.nvar * self.r
        conserved = set(conserved)
        free = [i for i in range(self.n) if i not in conserved]
        self.nf = len(free)
        self.register_buffer("V", V.clone())
        self.register_buffer("free", torch.tensor(free, dtype=torch.long, device=V.device))
        self.register_buffer("a_mean", torch.zeros(self.n, device=V.device))
        self.register_buffer("a_std", torch.ones(self.n, device=V.device))

    def set_normalization(self, a):
        std = a.std(0)
        std = torch.where(std > 1e-12 * std.max(), std, torch.ones_like(std))
        self.a_mean.copy_(a.mean(0))
        self.a_std.copy_(std)

    def v_N(self, a):
        """Projected entropy variables V^T v(V a) (physics, not learned)."""
        return project(self.phys.entropy_vars(reconstruct(a, self.V, self.nvar)), self.V)

    def inputs(self, a, mu=None):
        x = (a - self.a_mean) / self.a_std
        return x if mu is None else torch.cat([x, mu], -1)


class StructuredNet(ReducedNet):
    """da/dt = (J - R) v_N / dx on the free coefficients, 0 on the conserved ones.

    J = D (U - U^T) D and R = D L L^T D: the congruence with the fixed positive
    diagonal D keeps J skew and R PSD, so the entropy bound holds for any weights.
    D gives every mode its own size, so O(1) network outputs suit modes whose
    amplitudes differ by orders of magnitude.
    """

    def __init__(self, phys, V, dx, conserved, width, depth, act="tanh",
                 rank=None, l_init=1e-2, n_mu=0):
        super().__init__(phys, V, dx, conserved, n_mu)
        nf, dev = self.nf, V.device
        self.rank = rank
        iu = torch.triu_indices(nf, nf, 1, device=dev)
        self.register_buffer("iu0", iu[0])
        self.register_buffer("iu1", iu[1])
        self.nJ = iu.shape[1]                              # n_f (n_f - 1) / 2
        if rank is None:
            il = torch.tril_indices(nf, nf, device=dev)
            self.register_buffer("il0", il[0])
            self.register_buffer("il1", il[1])
            self.nL = il.shape[1]                          # n_f (n_f + 1) / 2
        else:
            self.nL = nf * rank
        self.register_buffer("d", torch.ones(nf, device=dev))   # per-mode scaling D

        self.mlp = mlp(self.n + n_mu, self.nJ + self.nL, width, depth, act)
        with torch.no_grad():
            self.mlp[-1].weight[self.nJ:] *= l_init       # R = L L^T starts small

    def factors(self, a, mu=None):
        """Unconstrained network outputs arranged as U (strict upper) and L, batched."""
        z = self.mlp(self.inputs(a, mu))
        B, nf = z.shape[0], self.nf
        Ju = z.new_zeros(B, nf, nf)
        Ju[:, self.iu0, self.iu1] = z[:, :self.nJ]
        if self.rank is None:
            L = z.new_zeros(B, nf, nf)
            L[:, self.il0, self.il1] = z[:, self.nJ:]
        else:
            L = z[:, self.nJ:].reshape(B, nf, self.rank)
        return Ju, L

    def operators(self, a, mu=None):
        """Dense J (skew) and R (PSD) on the free coefficients, for diagnostics."""
        Ju, L = self.factors(a, mu)
        DD = self.d[:, None] * self.d[None, :]
        return DD * (Ju - Ju.mT), DD * (L @ L.mT)

    def forward(self, a, mu=None, vN=None):
        if vN is None:
            vN = self.v_N(a)
        w = (self.d * vN[:, self.free]).unsqueeze(-1)      # w = D v_f
        Ju, L = self.factors(a, mu)
        Jw = Ju @ w - Ju.mT @ w                            # (U - U^T) w without forming it
        Rw = L @ (L.mT @ w)
        adot_f = (self.d / self.dx) * (Jw - Rw).squeeze(-1)
        return a.new_zeros(a.shape).index_copy(1, self.free, adot_f)


class UnconstrainedNet(ReducedNet):
    """Neural ODE baseline: da/dt = scale * MLP(a); no entropy or conservation structure."""

    def __init__(self, phys, V, dx, conserved, width, depth, act="tanh", n_mu=0, **unused):
        super().__init__(phys, V, dx, conserved, n_mu)
        self.register_buffer("scale", torch.ones(self.n, device=V.device))
        self.mlp = mlp(self.n + n_mu, self.n, width, depth, act)

    def forward(self, a, mu=None, vN=None):
        return self.scale * self.mlp(self.inputs(a, mu))


class ClosureNet(ReducedNet):
    """Chan's ES-ROM plus a learned eddy viscosity (entropy-admissible closure).

        da/dt = adot_ES(a) - V^T D^T (nu * D u~),   nu(x; a) >= 0,

    with D the edge difference (u_{i+1} - u_i)/dx and u~ = u(V v_N) the entropy-projected
    state used by the ES-ROM. The closure changes the entropy by
    -dx sum_e nu_e (D v~)_e^T (D u~)_e <= 0 (u(v) is monotone, as for the FOM viscosity),
    conserves the totals (difference of edge fluxes) and vanishes for a uniform state.
    nu = nu_scale * softplus(z_0 + sum_k z_k phi_k(x)), phi_k = non-constant POD modes at
    the edges, z = MLP(a): the network outputs only m + 1 numbers.

    Memory (n_delay > 0): the MLP also sees the past states a(t - j lag), j = 1..n_delay
    (hist, shape (B, n_delay, n)), a finite-memory (Mori-Zwanzig) closure for the
    truncated modes. The entropy argument holds for any nu >= 0, so it is unchanged.

    Backscatter (nu_floor = eps): nu = nu_scale * softplus(...) - eps >= -eps. The closure
    has the same discrete form as the ES-ROM viscosity (-eps V^T D^T D u~), so the two add
    to -V^T D^T ((eps + nu) * D u~) and the viscous entropy rate is
    -dx sum_e (eps + nu_e) (D v~)_e^T (D u~)_e <= 0 since eps + nu >= 0 edge-wise. The closure
    alone may now produce entropy (remove part of the ES-ROM dissipation); the model may not.
    The output bias starts at softplus^{-1}(eps / nu_scale), i.e. nu = 0: the ES-ROM.
    """

    def __init__(self, phys, V, dx, conserved, width, depth, act="tanh", esrom=None,
                 nu_modes=None, nu_scale=1.0, z0_init=-3.0, n_delay=0, delay_every=1,
                 nu_floor=0.0, n_mu=0, **unused):
        super().__init__(phys, V, dx, conserved, n_mu)
        self.esrom = esrom
        self.n_delay, self.delay_every = n_delay, delay_every
        m = self.r - 1 if nu_modes is None else nu_modes
        phi = 0.5 * (V[:, 1:m + 1] + torch.roll(V[:, 1:m + 1], -1, 0))   # at edge i+1/2
        self.register_buffer("phi", phi / phi.abs().max(0).values.clamp_min(1e-300))
        self.register_buffer("nu_scale", torch.tensor(float(nu_scale), device=V.device))
        self.register_buffer("nu_floor", torch.tensor(float(nu_floor), device=V.device),
                             persistent=False)          # set by build_net; old weights load
        self.mlp = mlp(self.n * (1 + n_delay) + n_mu, m + 1, width, depth, act)
        with torch.no_grad():                               # start as the ES-ROM (nu small)
            self.mlp[-1].weight.mul_(1e-2)
            self.mlp[-1].bias.zero_()
            # backscatter: softplus(z0) = nu_floor / nu_scale gives nu = 0 exactly
            self.mlp[-1].bias[0] = (np.log(np.expm1(nu_floor / nu_scale)) if nu_floor > 0
                                    else z0_init)

    def inputs(self, a, mu=None, hist=None):
        x = super().inputs(a, mu)
        if self.n_delay:
            if hist is None:
                raise ValueError("memory closure needs hist = past states, shape (B, n_delay, n)")
            x = torch.cat([x, ((hist - self.a_mean) / self.a_std).flatten(1)], -1)
        return x

    def viscosity(self, a, mu=None, hist=None):
        """nu at the K edges, shape (B, K); nu >= -nu_floor by construction."""
        z = self.mlp(self.inputs(a, mu, hist))
        return self.nu_scale * nn.functional.softplus(z[:, :1] + z[:, 1:] @ self.phi.T) - self.nu_floor

    def precompute(self, a):
        """ES-ROM rhs and edge differences D u~ (independent of the weights). Differentiable
        in a, which the causal rollout needs; wrap in no_grad to cache at fixed inputs."""
        ut = self.esrom.states(a)
        return self.esrom.rhs(a), (torch.roll(ut, -1, -1) - ut) / self.dx

    def closure(self, nu, gu):
        """-V^T D^T (nu * D u~): flux nu (D u~) at the edges, differenced per cell."""
        flux = nu[:, None, :] * gu
        return project((flux - torch.roll(flux, 1, -1)) / self.dx, self.V)

    def forward(self, a, mu=None, vN=None, cache=None, hist=None):
        base, gu = self.precompute(a) if cache is None else cache
        return base + self.closure(self.viscosity(a, mu, hist), gu)


def history(A, k, n_delay, lag):
    """Past states A[k - j lag], j = 1..n_delay, for samples k of a trajectory A saved at
    the snapshot times; before t = 0 the initial state is repeated. Shape (len(k), n_delay, n)."""
    idx = (k[:, None] - lag * torch.arange(1, n_delay + 1, device=k.device)).clamp_min(0)
    return A[idx]


CLOSURE_TYPES = ("closure", "memory", "backscatter")   # ES-ROM + learned viscosity


def build_net(model_type, prob, net_kw=NET):
    if model_type in CLOSURE_TYPES:
        kw = {**CLOSURE, **net_kw}
        if kw["nu_scale"] is None:
            kw["nu_scale"] = prob.fom.eps or 1.0
        if model_type in ("closure", "backscatter"):
            kw["n_delay"] = 0                          # Markovian: nu(x; a(t)) only
        if model_type == "backscatter":
            if not prob.fom.eps:
                raise ValueError("backscatter needs eps > 0: nu >= -eps")
            kw["nu_floor"] = prob.fom.eps              # eps + nu >= 0 edge-wise
        return ClosureNet(prob.phys, prob.V, prob.dx, prob.conserved, esrom=prob.teacher, **kw)
    cls = {"structured": StructuredNet, "unconstrained": UnconstrainedNet}[model_type]
    return cls(prob.phys, prob.V, prob.dx, prob.conserved, **net_kw)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trainer
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class ESNNROM():
    """ ES neural ROM trainer: data, teacher and entropy-rate losses (plan Sec. 7) """

    def __init__(self, prob, model_type=MODEL_TYPE, savept=None, net_kw=NET, train_kw=TRAIN,
                 mode=TRAIN_MODE, causal_kw=CAUSAL):

        # Initialization
        self.cfg = dict(train_kw)
        self.mode, self.ccfg = mode, dict(causal_kw)
        if mode == "causal" and model_type not in CLOSURE_TYPES:
            raise ValueError("causal rollout training is implemented for the closure models")
        self.prob, self.model_type, self.savept = prob, model_type, savept
        self.iter = 0
        self.exec_time = 0
        self.print_step = self.cfg["print_step"]
        self.Nepochs = self.cfg["nepochs"]
        self.it, self.ll = [], []
        self.loss, self.losses = None, []
        self.best_loss = np.inf
        self.dx = prob.dx
        teacher = prob.teacher

        # Data samples: projected FOM snapshots inside the training window
        m = prob.train_mask
        self.a_data = prob.A[m]
        self.adot_data = prob.Adot[m]
        self.vN_data = teacher.v_N(self.a_data)              # fixed inputs -> precompute
        if self.cfg["entropy_target"] == "fom":
            self.D_data = prob.D_fom[m]                       # eps v^T K u >= 0
        else:
            self.D_data = -self.dx * (self.vN_data * self.adot_data).sum(-1)

        # Teacher samples: snapshots and perturbed states labelled by the ES-ROM.
        # Not used by the closure model, whose base already is the ES-ROM.
        self.use_teacher = self.cfg["w_teacher"] > 0 and model_type not in CLOSURE_TYPES
        if self.use_teacher:
            self.a_teach, self.adot_teach = self.teacher_samples(self.cfg["n_teacher"],
                                                                 self.cfg["teacher_noise"])
        else:
            self.a_teach, self.adot_teach = self.a_data[:0], self.adot_data[:0]
        self.vN_teach = teacher.v_N(self.a_teach)

        # Network
        self.dnn = build_net(model_type, prob, net_kw).to(device)
        self.dnn.set_normalization(self.a_data)
        free = self.dnn.free
        if model_type == "structured":
            # d_i^2 = dx rms(da_i/dt) / rms(v_i): an O(1) diagonal entry of U or L L^T
            # then gives mode i its own typical rate of change
            rate = self.dx * self.adot_data[:, free].pow(2).mean(0).sqrt()
            self.dnn.d.copy_((rate / self.vN_data[:, free].pow(2).mean(0).sqrt()).sqrt())
        elif model_type == "unconstrained":
            rms = self.adot_data.unflatten(-1, (prob.nvar, prob.r)).pow(2).mean((0, 2)).sqrt()
            self.dnn.scale.copy_(rms.repeat_interleave(prob.r))
        self.forward = (torch.compile(self.dnn, mode="reduce-overhead")
                        if self.cfg["compile"] else self.dnn)
        # closure: ES-ROM rhs and D u~ at the fixed training inputs, computed once;
        # memory: also the past states of each training snapshot (from the FOM data)
        self.kw_data = {}
        if model_type in CLOSURE_TYPES:
            with torch.no_grad():
                self.kw_data["cache"] = self.dnn.precompute(self.a_data)
        if model_type == "memory":
            k = torch.arange(len(self.a_data), device=self.a_data.device)
            self.kw_data["hist"] = history(prob.A, k, self.dnn.n_delay, self.dnn.delay_every)

        # Loss normalization: per coefficient (each da_i/dt by its own rms, so the small
        # high-order modes count) or per variable block; and for the entropy rate
        blk = self.adot_data.unflatten(-1, (prob.nvar, prob.r)).pow(2).sum(-1).mean(0)
        self.block_scale = blk.clamp_min(1e-30)
        self.coef_scale = self.adot_data[:, free].pow(2).mean(0).clamp_min(1e-30)
        ref = (self.dx * self.vN_data[:, free].norm(dim=-1)
               * self.adot_data[:, free].norm(dim=-1)).pow(2).mean()
        self.D_scale = max(self.D_data.pow(2).mean().item(), 1e-12 * ref.item(), 1e-300)

        # Causal rollout: snapshot spacing, RK4 step, state scale of L_i
        if mode == "causal":
            c = self.ccfg
            self.Nepochs, self.print_step = c["nepochs"], c["print_step"]
            self.dt_snap = float(prob.t[1] - prob.t[0])
            self.dt_roll = self.dt_snap / c["substeps"]
            self.state_scale = self.a_data.pow(2).sum(-1).mean().item()
            self.best_horizon = 0

        # Optimizer (1st ord)
        opt = self.ccfg if mode == "causal" else self.cfg
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=opt["lr"])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt["step_size"],
                                             gamma=opt["gamma"])

        print(f"{model_type} network: {sum(p.numel() for p in self.dnn.parameters())} weights, "
              f"n={prob.n}, data samples={len(self.a_data)}, teacher samples={len(self.a_teach)}")
        if mode == "causal":
            print(f"causal rollout: {len(self.a_data) - 1} snapshot intervals, dt={self.dt_roll:.2e} "
                  f"({self.ccfg['substeps']} RK4 step(s) per interval), tol={self.ccfg['tol']}, "
                  f"w_cut={self.ccfg['w_cut']}")

    @torch.no_grad()
    def teacher_samples(self, n_samples, noise, chunk=256):
        """Snapshots plus Gaussian perturbations (conserved coefficients fixed).

        States with rho <= 0 or p <= 0 in V a, or a non-finite teacher RHS, are dropped.
        """
        prob = self.prob
        A = self.a_data
        std = A.std(0)
        std[prob.conserved] = 0.0
        idx = torch.randint(len(A), (n_samples,), device=A.device)
        a = torch.cat([A, A[idx] + noise * std * torch.randn(n_samples, prob.n, device=A.device)])
        a = a[prob.phys.is_admissible(prob.reconstruct(a))]
        adot = torch.cat([prob.teacher.rhs(c) for c in a.split(chunk)])
        ok = torch.isfinite(adot).all(-1)
        return a[ok], adot[ok]

    def block_loss(self, pred, target):
        if self.cfg.get("loss_norm", "block") == "coefficient":
            # conserved coefficients are left out: their targets are round-off
            err = (pred - target)[:, self.dnn.free].pow(2).mean(0)
            return (err / self.coef_scale).mean()
        err = (pred - target).unflatten(-1, (self.prob.nvar, self.prob.r)).pow(2).sum(-1).mean(0)
        return (err / self.block_scale).mean()

    def entropy_loss(self, adot_pred, vN, D_target):
        D_pred = -self.dx * (vN * adot_pred).sum(-1)       # = v^T R v for the structured net
        return (D_pred - D_target).pow(2).mean() / self.D_scale

    def loss_func(self):
        """ Loss function """

        self.optimizer.zero_grad()
        pretrain = self.iter < self.cfg["n_pretrain"] and self.use_teacher

        # Predictions
        adot_d = self.forward(self.a_data, vN=self.vN_data, **self.kw_data)
        self.loss_data = self.block_loss(adot_d, self.adot_data)
        if self.use_teacher:
            adot_t = self.forward(self.a_teach, vN=self.vN_teach)
            self.loss_teacher = self.block_loss(adot_t, self.adot_teach)
        else:
            self.loss_teacher = adot_d.new_zeros(())
        self.loss_entropy = self.entropy_loss(adot_d, self.vN_data, self.D_data)

        # Loss calculation
        if pretrain:
            self.loss = self.loss_teacher
        else:
            self.loss = (self.cfg["w_data"] * self.loss_data
                         + self.cfg["w_teacher"] * self.loss_teacher
                         + self.cfg["w_entropy"] * self.loss_entropy)

        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0 or self.iter == 1:
            print('Iter %d%s, Loss: %.3e, Data: %.3e, Teacher: %.3e, Entropy: %.3e, lr: %.1e, t/iter: %.1e' %
                  (self.iter, ' (pretrain)' if pretrain else '', self.loss.item(), self.loss_data.item(),
                   self.loss_teacher.item(), self.loss_entropy.item(),
                   self.scheduler.get_last_lr()[0], self.exec_time))
            self.it.append(self.iter)
            self.ll.append([self.loss.item(), self.loss_data.item(),
                            self.loss_teacher.item(), self.loss_entropy.item()])

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.losses.append(self.loss.item())

        if not pretrain and self.loss.item() < self.best_loss:
            self.best_loss = self.loss.item()
            if self.savept is not None:
                torch.save(self.dnn.state_dict(), str(self.savept) + "_best.pt")

    def causal_rollout(self, w_cut=None):
        """Differentiable RK4 rollout from a_FOM(0) through the snapshot times.

        The network sees its own past states (memory closure: a(t - j lag) from this
        rollout, held over the RK stages exactly as in rollout_memory). Stops when the
        causal weight w_i = exp(-tol sum_{k<i} L_k) of the next snapshot falls below
        w_cut, or at a non-finite state. Returns the states at the reached snapshots
        a(t_1..t_m), their errors L_1..L_m and the weights w_1..w_m (both detached).
        """
        A, net, c = self.a_data, self.dnn, self.ccfg
        w_cut = c["w_cut"] if w_cut is None else w_cut
        m = c["substeps"]
        lag = net.delay_every * m
        j = lag * torch.arange(1, net.n_delay + 1)
        a, states = A[0], [A[0]]
        out, L, W, cum = [], [], [], 0.0
        for i in range(1, len(A)):
            w = float(np.exp(-c["tol"] * cum))
            if w < w_cut:
                break
            for _ in range(m):
                k = len(states)                              # current state = states[k - 1]
                h = (torch.stack([states[q] for q in (k - 1 - j).clamp_min(0).tolist()])[None]
                     if net.n_delay else None)
                a = rk45_step(lambda b: net(b[None], hist=h)[0], a, self.dt_roll)
                states.append(a)
            if not torch.isfinite(a).all():
                break
            Li = (a - A[i]).pow(2).sum() / self.state_scale
            out.append(a); L.append(Li); W.append(w)
            cum += Li.item()
        return out, L, torch.tensor(W)

    def causal_loss_func(self):
        """ Causal rollout loss: mean_i w_i L_i over all snapshot intervals """

        self.optimizer.zero_grad()
        _, L, W = self.causal_rollout()
        n_int = len(self.a_data) - 1
        if not L:
            raise RuntimeError("causal rollout produced no finite state")
        Ls = torch.stack(L)
        self.loss = (W.to(Ls) * Ls).sum() / n_int
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dnn.parameters(), self.ccfg["clip"])
        self.iter += 1

        horizon = len(L) * self.dt_snap                   # time reached by the rollout
        err_end = Ls[-1].sqrt().item()                     # relative coefficient error there
        Lmean = Ls.detach().mean().item()
        if self.iter % self.print_step == 0 or self.iter == 1:
            print('Iter %d, Loss: %.3e, horizon t=%.3f (w_end %.1e), mean L: %.3e, '
                  'err(horizon): %.3e, lr: %.1e, t/iter: %.1e' %
                  (self.iter, self.loss.item(), horizon, W[-1].item(), Lmean, err_end,
                   self.scheduler.get_last_lr()[0], self.exec_time))
            self.it.append(self.iter)
            self.ll.append([self.loss.item(), horizon, Lmean, err_end])

        self.optimizer.step()
        self.scheduler.step()
        self.losses.append(self.loss.item())

        # best = longest horizon, then smallest loss (the loss alone favours short rollouts)
        key = (len(L), -self.loss.item())
        if key > (self.best_horizon, -self.best_loss):
            self.best_horizon, self.best_loss = len(L), self.loss.item()
            if self.savept is not None:
                torch.save(self.dnn.state_dict(), str(self.savept) + "_best.pt")

    def train(self):
        """ Train model """

        self.dnn.train()
        step = self.causal_loss_func if self.mode == "causal" else self.loss_func
        for epoch in range(self.Nepochs):
            start_time = time.time()
            step()
            self.exec_time = time.time() - start_time

        if self.savept is not None:
            d = np.column_stack((np.array(self.it), np.array(self.ll)))
            header = ('iter total horizon mean_L err_horizon' if self.mode == "causal"
                      else 'iter total data teacher entropy')
            np.savetxt(str(self.savept) + '_losses.txt', d, fmt='%.6e', header=header)
            torch.save(self.dnn.state_dict(), str(self.savept) + ".pt")

    @torch.no_grad()
    def predict(self, a):
        self.dnn.eval()
        return self.dnn(a)

    def plot_loss(self, savePath):
        ll = np.array(self.ll)
        if self.mode == "causal":
            fig, axs = plt.subplots(1, 2, figsize=(11, 4))
            axs[0].semilogy(np.arange(1, len(self.losses) + 1), self.losses, lw=0.6, label='loss')
            axs[0].semilogy(self.it, ll[:, 2], 'o-', ms=2, label='mean $L_i$ (reached)')
            axs[0].semilogy(self.it, ll[:, 3], 'o-', ms=2, label='error at horizon')
            axs[0].set_xlabel('epoch'); axs[0].legend()
            axs[1].plot(self.it, ll[:, 1], 'o-', ms=2)
            axs[1].axhline(self.dt_snap * (len(self.a_data) - 1), color='k', lw=0.6)
            axs[1].set_xlabel('epoch'); axs[1].set_ylabel('causal horizon t')
            fig.tight_layout(); fig.savefig(savePath, dpi=150); plt.close(fig)
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.arange(1, len(self.losses) + 1), self.losses, lw=0.6, label='total')
        for j, name in enumerate(['data', 'teacher', 'entropy'], start=1):
            ax.semilogy(self.it, ll[:, j], 'o-', ms=2, label=name)
        if self.cfg["n_pretrain"] and self.use_teacher:
            ax.axvline(self.cfg["n_pretrain"], color='k', ls=':', lw=0.8)
        ax.set_xlabel('epoch')
        ax.legend()
        fig.tight_layout()
        fig.savefig(savePath, dpi=150)
        plt.close(fig)

    def report(self, time, fileName):
        with open(fileName, "w+") as f:
            f.write(f'Problem            : {PROBLEM}, K={K}, eps={EPS}, r={self.prob.r}, n={self.prob.n}\n')
            f.write(f'Model              : {self.model_type} ({self.mode} training)\n')
            f.write(f'Total training time: {(time / 60):4.3f} mins.\n')
            f.write(f'Total epochs       : {self.Nepochs:6d}.\n')
            f.write(f'Best Loss          : {self.best_loss:1.6e}\n')
            f.write(f'Train settings     : {self.ccfg if self.mode == "causal" else self.cfg}\n')
            f.write(f'NN Architecture    : {self.dnn}\n')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Online integration
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def net_rhs(net):
    """Wrap a batched network as rhs(a) for a single state a of shape (n,)."""
    return lambda a: net(a[None])[0]


def rollout_memory(net, a0, t_final=T_FINAL, dt=DT_ROM, save_dt=DT * SAVE_EVERY, snap_dt=DT * SAVE_EVERY):
    """RK4 rollout of the memory closure, a delay equation in a.

    Every step state is kept; the past states a(t_n - j lag) fed to the network are
    taken at the start of the step and held over the RK stages (an O(dt) shift of the
    delayed argument, small for lag >> dt). lag = delay_every snapshot intervals, the
    same delay as in training (snap_dt = snapshot spacing of the training data).
    Returns (times, saved states, wall time per step).
    """
    nsteps, save_every = round(t_final / dt), max(1, round(save_dt / dt))
    lag = round(net.delay_every * snap_dt / dt)
    j = lag * torch.arange(1, net.n_delay + 1)
    start = time.time()
    with torch.no_grad():
        a, states = a0.clone(), [a0.clone()]
        times, saved = [0.0], [a0.clone()]
        for k in range(1, nsteps + 1):
            h = torch.stack([states[i] for i in (k - 1 - j).clamp_min(0).tolist()])[None]
            a = rk45_step(lambda b: net(b[None], hist=h)[0], a, dt)
            if not torch.isfinite(a).all():
                break
            states.append(a.clone())
            if k % save_every == 0 or k == nsteps:
                times.append(k * dt)
                saved.append(a.clone())
    steps_done = max(1, round(times[-1] / dt))
    return np.array(times), torch.stack(saved), (time.time() - start) / steps_done


def rollout(rhs, a0, t_final=T_FINAL, dt=DT_ROM, save_dt=DT * SAVE_EVERY):
    """RK4 rollout of da/dt = rhs(a); stops at the first non-finite state.

    Returns (times, states, wall time per step). Saved times match the FOM
    snapshots when save_dt is a multiple of dt.
    """
    nsteps, save_every = round(t_final / dt), max(1, round(save_dt / dt))
    start = time.time()
    t, A = integrate(rhs, a0, dt, nsteps, save_every, stop_on_nonfinite=True)
    steps_done = max(1, round(t[-1] / dt))
    return t, A, (time.time() - start) / steps_done
