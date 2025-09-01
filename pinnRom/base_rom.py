from pinnRom_config import *

import numpy.linalg as LA
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Parameters
x = np.linspace(xmin, xmax, n_x)
dx = x[1] - x[0]
t_eval = np.linspace(tmin, tmax, n_t)

def exact_solution_field(t):
    return (x / (1 + t)) / (1 + np.sqrt((1 + t) / np.exp(Re / 8)) * np.exp(Re * (x**2) / (4 * (1 + t))))


def collect_snapshots_field():
    snapshot_matrix_total = np.zeros(shape=(np.shape(x)[0],np.shape(t_eval)[0]))

    trange = np.arange(np.shape(t_eval)[0])
    for t in trange:
        snapshot_matrix_total[:,t] = exact_solution_field(t_eval[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix_total,axis=1)
    snapshot_matrix = (snapshot_matrix_total.transpose()-snapshot_matrix_mean).transpose()

    return snapshot_matrix, snapshot_matrix_mean, snapshot_matrix_total


Y, Y_mean, Y_total = collect_snapshots_field()

U_fom = Y_total

# --- POD ---
V, S, WT = np.linalg.svd(Y, full_matrices=False)
energy = np.cumsum(S**2) / np.sum(S**2)
# r = np.searchsorted(energy, 0.97) + 1
# print(f"Using r = {r} modes to capture 97% of the energy.")
# r = 1
V_r = V[:, :r]

# print(S.shape)
# plt.figure()
# plt.semilogy(S, 'o-')
# plt.xlabel("Mode index")
# plt.ylabel("Singular value")
# plt.title("Singular Value Decay")
# plt.show()


 # --- Project Initial Condition ---
u0 = exact_solution_field(0)
a0 = V_r.T @ (u0 - Y_mean)   # project mean-subtracted initial condition


# --- Galerkin-Projected Burgers Equation ---
def burgers_rom_rhs(t, a):
    # Reconstruct u in physical space (add back mean)
    u = V_r @ a + Y_mean
    dudx = np.zeros_like(u)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    # Zero Dirichlet BCs for u and dudx
    dudx[0] = 0.0
    dudx[-1] = 0.0
    u[0] = 0.0
    u[-1] = 0.0
    nonlinear = -u * dudx
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    lap[0] = 0.0
    lap[-1] = 0.0
    rhs_phys = nonlinear + nu * lap
    return V_r.T @ rhs_phys

# --- Integrate ROM System ---
sol_rom = solve_ivp(burgers_rom_rhs, [tmin, tmax], a0, t_eval=t_eval, method='RK45')
A_sol = sol_rom.y  # shape: (r, timesteps)
U_rom = V_r @ A_sol + Y_mean[:, None]  # reconstruct physical field


show_plot = 0
if show_plot:
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    line_full, = ax1.plot(x, Y_total[:, 0], 'k-', label='Full')
    line_rom, = ax1.plot(x, U_rom[:, 0], 'r--', label='ROM')
    ax1.set_ylim(np.min(Y_total), np.max(Y_total))
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.legend()
    ax1.set_title("Solution and ROM")

    line_err, = ax2.plot(x, Y_total[:, 0] - U_rom[:, 0], 'b-')
    ax2.set_ylim(-np.max(np.abs(Y_total - U_rom)), np.max(np.abs(Y_total - U_rom)))
    ax2.set_xlabel("x")
    ax2.set_ylabel("Pointwise Error")
    ax2.set_title("Pointwise Error (Full - ROM)")

    suptitle = fig.suptitle(f"Time = {t_eval[0]:.3f}", fontsize=16)

    def update(frame):
        line_full.set_ydata(Y_total[:, frame])
        line_rom.set_ydata(U_rom[:, frame])
        line_err.set_ydata(Y_total[:, frame] - U_rom[:, frame])
        suptitle.set_text(f"Time = {t_eval[frame]:.3f}")
        return line_full, line_rom, line_err, suptitle

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=40, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# error = np.abs(Y_total - U_rom)  # same shape

# # Exact solution
# c0 = axes[0].contourf(x, t_eval, Y_total, levels=100, cmap='jet')
# axes[0].set_title('Exact Solution')
# axes[0].set_xlabel('t')
# axes[0].set_ylabel('x')
# fig.colorbar(c0, ax=axes[0])

# # Predicted solution
# c1 = axes[1].contourf(x, t_eval, U_rom, levels=100, cmap='jet')
# axes[1].set_title('Predicted Solution')
# axes[1].set_xlabel('t')
# axes[1].set_ylabel('x')
# fig.colorbar(c1, ax=axes[1])

# # Absolute error
# c2 = axes[2].contourf(x, t_eval, error, levels=100, cmap='jet')
# axes[2].set_title('Absolute Error')
# axes[2].set_xlabel('t')
# axes[2].set_ylabel('x')
# fig.colorbar(c2, ax=axes[2])

# plt.tight_layout()
# plt.show()
