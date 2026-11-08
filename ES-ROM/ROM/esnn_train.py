"""Train the entropy stable neural ROM (or the unconstrained baseline).

Run from ES-ROM/ROM/:
    python esnn_train.py                                   # Euler, structured network
    ESNN_MODEL=unconstrained python esnn_train.py          # neural ODE baseline
    ESNN_PROBLEM=burgers ESNN_QUICK=1 python esnn_train.py # fast plumbing check
    ESNN_MODEL=memory ESNN_TRAIN=causal python -u esnn_train.py  # causal rollout training

The first run of a problem integrates the FOM and caches snapshots and the
POD basis in outputs/<problem>/fom.npz; later runs (and esnn_post.py) reuse them.
"""
from esnn_model import *


if __name__ == "__main__":

    print('----------------------------------------')
    print(f'ES neural ROM  problem: {PROBLEM}  model: {MODEL_TYPE}  training: {TRAIN_MODE}  '
          f'K: {K}  eps: {EPS}  device: {device}')
    print('----------------------------------------')

    if TRAIN_MODE == "causal":
        torch.set_num_threads(CAUSAL["threads"])
    os.makedirs(model_folder, exist_ok=True)
    prob = load_problem()

    # Basis and teacher sanity checks before spending time on training
    perr = prob.projection_error()
    k_train = int(prob.train_mask.sum()) - 1
    print(f'Basis: r={prob.r} per variable, n={prob.n}, '
          f'sigma_r/sigma_1={prob.sigma[prob.r - 1] / prob.sigma[0]:.2e}, '
          f'||V^T V - I||={(prob.V.T @ prob.V - torch.eye(prob.r, device=device)).abs().max():.1e}')
    print(f'Projection error: t=T_train {perr[k_train]:.2e}, t=T_final {perr[-1]:.2e}')
    with torch.no_grad():
        a = prob.A[prob.train_mask][::10]
        res = prob.teacher.entropy_residual(a).abs().max().item()
        rate = prob.teacher.entropy_rate(a).max().item()
    print(f'Teacher: max |convective entropy residual| {res:.1e}, max dS/dt {rate:.1e} (should be <= 0)')

    model = ESNNROM(prob, MODEL_TYPE, savept=model_folder + 'weights')
    start = time.time()
    model.train()
    total_time = time.time() - start
    print(f'Total time:  {(total_time/60):2.3f} mins \n')

    model.report(total_time, model_folder + 'report.txt')
    model.plot_loss(model_folder + 'loss.png')
