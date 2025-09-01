from pinnRom_config import *
from pinnRom_model import DNN
from base_rom import U_rom, U_fom, x, t_eval, r

from scipy.io import savemat, loadmat
from matplotlib.animation import FuncAnimation

PI = np.pi

file_name = save_folder + '/weights_best.pt'

model = DNN()
mdata = torch.load(file_name)
# print(mdata)
# sys.exit()
model.load_state_dict(mdata)
model.eval()


X, T = np.meshgrid(x, t_eval)
X = X.flatten()[:,None]
T = T.flatten()[:,None]

u_model = model.calc(torch.Tensor(X), torch.Tensor(T)).detach().numpy()
u_pred = u_model[:,0:1]

u_pred_grid = u_pred.reshape((n_x, n_t), order='F')


plt.imshow(u_pred_grid, extent=[t_eval[0], t_eval[-1], x[0], x[-1]],
           aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label='u')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Predicted u(x, t)')
plt.show()
# sys.exit()

u_pred_grid = u_pred_grid + U_rom
show_plot = 1
if show_plot:
    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    line_full, = ax1.plot(x, U_fom[:, 0], 'k-', label='Full')
    line_rom, = ax1.plot(x, U_rom[:, 0], 'r--', label='ROM')
    ax1.set_ylim(np.min(U_fom), np.max(U_fom))
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.legend()
    ax1.set_title("Solution and ROM")

    line_full2, = ax2.plot(x, U_fom[:, 0], 'k-', label='Full')
    line_closure, = ax2.plot(x, u_pred_grid[:, 0], 'b--', label='PINN closure')
    line_rom2, = ax2.plot(x, U_rom[:, 0], 'r--', label='ROM')
    ax2.set_ylim(np.min(U_fom), np.max(U_fom))
    ax2.set_xlabel("x")
    ax2.set_ylabel("u")
    ax2.legend()
    ax2.set_title("Solution and ROM Closure")


    # line_err, = ax2.plot(x, U_fom[:, 0] - U_rom[:, 0], 'b-')
    # ax2.set_ylim(-np.max(np.abs(U_fom - U_rom)), np.max(np.abs(U_fom - U_rom)))
    # ax2.set_xlabel("x")
    # ax2.set_ylabel("Pointwise Error")
    # ax2.set_title("Pointwise Error (Full - ROM)")

    suptitle = fig.suptitle(f"Burgers equation with Re = {Re} and reduced order {r}\nTime = {t_eval[0]:.3f}", fontsize=16)

    def update(frame):
        line_full.set_ydata(U_fom[:, frame])
        line_rom.set_ydata(U_rom[:, frame])
        line_full2.set_ydata(U_fom[:, frame])
        line_rom2.set_ydata(U_rom[:, frame])
        line_closure.set_ydata(u_pred_grid[:, frame])
        # line_err.set_ydata(U_fom[:, frame] - U_rom[:, frame])
        suptitle.set_text(f"Burgers equation with Re = {Re} and reduced order {r}\nTime = {t_eval[frame]:.3f}")
        # return line_full, line_rom, line_err, suptitle
        return line_full, line_rom, line_full2, line_closure, suptitle

    ani = FuncAnimation(fig, update, frames=len(t_eval), interval=40, blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # ani.save('burgers.gif', writer='imagemagick', fps=25)
    plt.show()
