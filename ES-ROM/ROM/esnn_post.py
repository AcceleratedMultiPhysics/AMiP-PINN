"""Post-process: roll out every ROM to T_FINAL and compare with the FOM (plan Sec. 9).

ROMs: structured, unconstrained, closure, memory and backscatter networks (whichever weights
exist in outputs/<problem>/<model>/ or, causal rollout training, <model>_causal/), the ES-ROM teacher (Chan Thm 2) and plain
POD-Galerkin. Run from ES-ROM/ROM/ after esnn_train.py:
    python esnn_post.py
    ESNN_PROBLEM=burgers ESNN_QUICK=1 python esnn_post.py

Writes to outputs/<problem>/post/: error, entropy, dissipation, conservation
and profile figures, plus metrics.json.
"""
import json

from esnn_model import *

post_folder = os.path.join(save_folder, 'post', '')
os.makedirs(post_folder, exist_ok=True)

prob = load_problem()
fom, phys = prob.fom, prob.phys
t_fom = prob.t
k_train = int(prob.train_mask.sum()) - 1
a0 = prob.A[0]

# ---------------------------------------------------------------- ROMs
roms = {'ES-ROM (teacher)': prob.teacher.rhs, 'POD-Galerkin': prob.galerkin.rhs}
nets = {}
for mt in ('structured', 'unconstrained', 'closure', 'backscatter', 'memory'):
    for label in (mt, mt + '_causal'):
        file_name = os.path.join(save_folder, label, 'weights_best.pt' if USE_BEST else 'weights.pt')
        if not os.path.exists(file_name):
            if label == mt:
                print(f'no {mt} weights at {file_name}, skipping')
            continue
        net = build_net(mt, prob, NET).to(device)
        net.load_state_dict(torch.load(file_name, map_location=device))
        net.eval()
        assert torch.allclose(net.V, prob.V), 'weights were trained with a different basis'
        nets[label] = net
        roms[f'NN {label}'] = net if getattr(net, 'n_delay', 0) else net_rhs(net)

# causal models are rolled out with the RK4 step they were trained with
dt_causal = DT * SAVE_EVERY / CAUSAL['substeps']


def is_memory(name):
    return name.startswith('NN ') and getattr(nets[name[3:]], 'n_delay', 0) > 0


def batched_rate(name, A):
    """da/dt at the saved states A of a rollout (memory: past states from the same rollout)."""
    if is_memory(name):
        net = nets[name[3:]]
        hist = history(A, torch.arange(len(A), device=A.device), net.n_delay, net.delay_every)
        return net(A, hist=hist)
    if name.startswith('NN '):
        return nets[name[3:]](A)
    return roms[name](A)

# ---------------------------------------------------------------- rollouts
results = {}
for name, rhs in roms.items():
    dt = dt_causal if name.endswith('_causal') else DT_ROM
    t, A, wall = rollout_memory(rhs, a0, dt=dt) if is_memory(name) else rollout(rhs, a0, dt=dt)
    nt = len(t)
    assert np.allclose(t, t_fom[:nt]), 'ROM save times must match the FOM snapshots'
    with torch.no_grad():
        U = prob.reconstruct(A)
        err = (U - prob.U[:nt]).norm(dim=(-2, -1)) / prob.U[:nt].norm(dim=(-2, -1))
        S = fom.total_entropy(U)
        tot = fom.totals(U)
        adm = phys.is_admissible(U)
    blowup = None if nt == len(t_fom) else float(t[-1])
    results[name] = dict(t=t, A=A, U=U, err=tonp(err), S=tonp(S),
                         drift=tonp((tot - tot[0]).abs().max(-1).values), adm=tonp(adm),
                         wall=wall, blowup=blowup)
    print(f'{name:20s} err(T_train)={tonp(err)[min(k_train, nt - 1)]:.3e}  '
          f'err(end)={tonp(err)[-1]:.3e}  t_end={t[-1]:.3f}  {1e3 * wall:.3f} ms/step')

# ---------------------------------------------------------------- structure checks
checks = {}
with torch.no_grad():
    if 'structured' in nets:
        net, A = nets['structured'], results['NN structured']['A']
        vf = net.v_N(A)[:, net.free].unsqueeze(-1)
        J, R = net.operators(A)
        vJv = (vf.mT @ J @ vf).squeeze()
        vRv = (vf.mT @ R @ vf).squeeze()
        scale = vf.squeeze(-1).pow(2).sum(-1) * J.flatten(1).norm(dim=-1)
        checks['structured'] = dict(
            max_rel_vJv=(vJv.abs() / scale).max().item(),
            min_vRv=vRv.min().item(),
            max_conserved_rate=net(A)[:, prob.conserved].abs().max().item())
        print(f"structured: max |v^T J v|/(|v|^2 |J|) = {checks['structured']['max_rel_vJv']:.1e}, "
              f"min v^T R v = {checks['structured']['min_vRv']:.1e}")
    nus = {}
    for mt, net in nets.items():
        if not isinstance(net, ClosureNet):
            continue
        A = results[f'NN {mt}']['A']
        _, gu = net.precompute(A)
        hist = (history(A, torch.arange(len(A), device=A.device), net.n_delay, net.delay_every)
                if net.n_delay else None)
        nu = net.viscosity(A, hist=hist)
        nus[mt] = (results[f'NN {mt}']['t'], 1.0 + tonp(nu) / EPS)       # total (eps + nu) / eps
        vN = prob.teacher.v_N(A)
        rate_c = prob.dx * (vN * net.closure(nu, gu)).sum(-1)               # backscatter: may be > 0
        rate = prob.dx * (vN * batched_rate(f'NN {mt}', A)).sum(-1)        # whole model: <= 0
        checks[mt] = dict(max_model_dSdt=rate.max().item(), max_closure_dSdt=rate_c.max().item(),
                          min_total_visc_over_eps=1.0 + (nu.min() / EPS).item(),
                          max_nu_over_eps=(nu.max() / EPS).item(),
                          mean_nu_over_eps=(nu.mean() / EPS).item(),
                          frac_backscatter=(nu < 0).double().mean().item())
        print(f"{mt}: max model dS/dt = {checks[mt]['max_model_dSdt']:.1e} (<= 0), "
              f"max closure dS/dt = {checks[mt]['max_closure_dSdt']:.1e}, "
              f"min (eps+nu)/eps = {checks[mt]['min_total_visc_over_eps']:.3f}, "
              f"nu/eps mean {checks[mt]['mean_nu_over_eps']:.2f}, max {checks[mt]['max_nu_over_eps']:.2f}, "
              f"nu < 0 on {100 * checks[mt]['frac_backscatter']:.1f}% of edges")
    a = prob.A[::10]
    checks['teacher_max_entropy_residual'] = prob.teacher.entropy_residual(a).abs().max().item()

# ---------------------------------------------------------------- metrics
perr = prob.projection_error()
metrics = dict(problem=PROBLEM, K=K, eps=EPS, r=prob.r, n=prob.n, T_train=T_TRAIN, T_final=T_FINAL,
               projection_error=dict(T_train=float(perr[k_train]), T_final=float(perr[-1])),
               checks=checks, roms={})
for name, res in results.items():
    dS = np.diff(res['S'])
    metrics['roms'][name] = dict(
        err_T_train=float(res['err'][min(k_train, len(res['err']) - 1)]),
        err_end=float(res['err'][-1]), err_max=float(np.max(res['err'])),
        t_end=float(res['t'][-1]), blowup_time=res['blowup'],
        max_entropy_increase=float(max(dS.max(initial=0.0), 0.0)),
        max_totals_drift=float(res['drift'].max()),
        all_admissible=bool(res['adm'].all()), ms_per_step=1e3 * res['wall'])
with open(post_folder + 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------- figures
styles = {'ES-ROM (teacher)': 'k--', 'POD-Galerkin': 'C7:', 'NN structured': 'C0-', 'NN unconstrained': 'C3-',
          'NN closure': 'C1-', 'NN memory': 'C4-', 'NN backscatter': 'C0-',
          'NN closure_causal': 'C5-', 'NN memory_causal': 'C3-', 'NN backscatter_causal': 'C6-'}

fig, ax = plt.subplots(figsize=(6, 4))
for name, res in results.items():
    ax.semilogy(res['t'], res['err'], styles[name], label=name)
ax.semilogy(t_fom, perr, 'C2-.', lw=0.8, label='projection')
ax.axvline(T_TRAIN, color='k', lw=0.6)
ax.set_xlabel('t'); ax.set_ylabel(r'$\|Va - u\| / \|u\|$'); ax.legend(fontsize=8)
ax.set_title(f'{PROBLEM}: relative error (training window left of the line)')
fig.tight_layout(); fig.savefig(post_folder + 'error_vs_time.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_fom, tonp(prob.S_fom - prob.S_fom[0]), 'C2-', lw=2, alpha=0.5, label='FOM')
for name, res in results.items():
    ax.plot(res['t'], res['S'] - res['S'][0], styles[name], label=name)
ax.axvline(T_TRAIN, color='k', lw=0.6)
ax.set_xlabel('t'); ax.set_ylabel(r'$\mathcal{S}(t) - \mathcal{S}(0)$'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(post_folder + 'entropy_vs_time.png', dpi=150); plt.close(fig)

# entropy dissipation rate -dS/dt along each trajectory
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_fom, tonp(prob.D_fom), 'C2-', lw=2, alpha=0.5, label=r'FOM $\epsilon v^TKu$')
with torch.no_grad():
    for name, res in results.items():
        if name == 'POD-Galerkin':
            continue
        A = res['A']
        D = -prob.dx * (prob.teacher.v_N(A) * batched_rate(name, A)).sum(-1)
        ax.plot(res['t'], tonp(D), styles[name], label=name)
ax.axvline(T_TRAIN, color='k', lw=0.6)
ax.set_xlabel('t'); ax.set_ylabel(r'$-d\mathcal{S}/dt$'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(post_folder + 'dissipation_vs_time.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for name, res in results.items():
    ax.semilogy(res['t'], np.maximum(res['drift'], 1e-17), styles[name], label=name)
ax.set_xlabel('t'); ax.set_ylabel('max |totals(t) - totals(0)|'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(post_folder + 'conservation.png', dpi=150); plt.close(fig)

x = tonp(fom.x)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
k_mid = k_train if k_train < len(t_fom) - 1 else (len(t_fom) - 1) // 2   # reproductive: show mid and end
for ax, k in zip(axs, (k_mid, len(t_fom) - 1)):
    ax.plot(x, tonp(prob.U[k, 0]), 'C2-', lw=3, alpha=0.4, label='FOM')
    for name, res in results.items():
        if k < len(res['t']):
            ax.plot(x, tonp(res['U'][k, 0]), styles[name], lw=1, label=name)
    ax.set_title(f'{phys.names[0]} at t = {t_fom[k]:.3f}'); ax.set_xlabel('x')
axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(post_folder + 'profiles.png', dpi=150); plt.close(fig)

# total viscosity (eps + nu)/eps of the closure models (1 = ES-ROM): in space at a few times,
# and min/mean/max along the rollout
xe = tonp(fom.x) + prob.dx / 2
for mt, (t_nu, nu) in nus.items():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for tt in sorted({0.1, 0.3, 0.5 * T_FINAL, T_TRAIN, 0.5 * (T_TRAIN + T_FINAL), T_FINAL}):
        k = min(int(round(tt / (DT * SAVE_EVERY))), len(t_nu) - 1)
        axs[0].plot(xe, nu[k], lw=1, label=f't={t_nu[k]:.2f}')
    axs[0].axhline(1.0, color='k', lw=0.6, ls='--')
    axs[0].set_yscale('log'); axs[0].set_xlabel('x'); axs[0].set_ylabel(r'$(\epsilon+\nu_\theta)/\epsilon$')
    axs[0].legend(fontsize=7); axs[0].set_title(f'total viscosity ({mt}), r={prob.r}; 1 = ES-ROM')
    axs[1].plot(t_nu, nu.max(1), label='max over x'); axs[1].plot(t_nu, nu.mean(1), label='mean over x')
    axs[1].plot(t_nu, nu.min(1), label='min over x'); axs[1].axhline(1.0, color='k', lw=0.6, ls='--')
    axs[1].axvline(T_TRAIN, color='k', lw=0.6); axs[1].set_yscale('log'); axs[1].set_xlabel('t')
    axs[1].legend(); axs[1].set_title(r'$(\epsilon+\nu_\theta)/\epsilon$ along the rollout')
    fig.tight_layout(); fig.savefig(post_folder + f'viscosity_{mt}.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.semilogy(prob.sigma / prob.sigma[0], '.')
ax.axvline(prob.r - 1, color='k', lw=0.6)
ax.set_xlabel('index'); ax.set_ylabel(r'$\sigma_j/\sigma_1$')
ax.set_title('snapshot singular values (training window)')
fig.tight_layout(); fig.savefig(post_folder + 'singular_values.png', dpi=150); plt.close(fig)

print(f'figures and metrics.json written to {post_folder}')
