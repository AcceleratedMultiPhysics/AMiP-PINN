from burgers_config import *
from burgers_model import PINN, Sampler

def dirichlet(z, value):
	x = z[:, 0:1]
	y = z[:, 1:2]
	return value * np.ones(x.shape)

def ic(z):
	x = z[:, 0:1]
	# t = z[:, 1:2]
	u = x / (1 + np.sqrt(1 / np.exp(Re / 8)) * np.exp(Re * (x**2) / (4)))
	return u


if __name__ == "__main__":

	# Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(mesh_file)

	print('----------------------------------------')
	print(f"Burgers solution with Re: {Re}")
	print('----------------------------------------')

	# WALL left (x fixed)
	WALL_1_coords = np.array([[xmin, tmin], [xmin, tmax]])
	WALL_1_sampler = Sampler(WALL_1_coords, f_u=lambda X: dirichlet(X, value=0.0), name="WALL_1")

	# WALL right (x fixed)
	WALL_2_coords = np.array([[xmax, tmin], [xmax, tmax]])
	WALL_2_sampler = Sampler(WALL_2_coords, f_u=lambda X: dirichlet(X, value=0.0), name="WALL_2")

	# initial condition (t fixed)
	IC_coords = np.array([[xmin, tmin], [xmax, tmin]])
	IC_sampler = Sampler(IC_coords, f_u=lambda X: ic(X), name="IC")

	# interior residual points (2D grid)
	INTERIOR_coords = np.array([[xmin, tmin], [xmax, tmax]])
	res_sampler = Sampler(INTERIOR_coords, name="RESIDUAL")

	bcs_sampler = [WALL_1_sampler, WALL_2_sampler, IC_sampler]


	plot_points = 1

	if plot_points:
		X_wall1, u_wall1 = WALL_1_sampler.sample(n_t)
		X_wall2, u_wall2 = WALL_2_sampler.sample(n_t)
		X_ic, u_ic = IC_sampler.sample(n_x)
		X_res, _ = res_sampler.sample((n_x, n_t))  

		plt.scatter(X_res[:, 0:1], X_res[:, 1:2], marker='o', alpha=0.1, color='red')
		plt.scatter(X_wall1[:, 0:1], X_wall1[:, 1:2], marker='o', alpha=0.1, c=u_wall1, cmap='jet')
		plt.scatter(X_wall2[:, 0:1], X_wall2[:, 1:2], marker='o', alpha=0.1, c=u_wall2, cmap='jet')
		plt.scatter(X_ic[:, 0:1], X_ic[:, 1:2], marker='o', alpha=0.1, c=u_ic, cmap='jet')
		plt.show()
		sys.exit()


	# if FF:
	# 	model = PINN_FF(res_sampler, bcs_sampler, savept=save_folder + 'weights')
	# else:
	model = PINN(res_sampler, bcs_sampler, savept=save_folder + 'weights')
	start = time.time()
	model.train()
	total_time = time.time() - start
	print(f'Total time:  {(total_time/60):2.3f} mins \n')

	model.report(total_time, save_folder + 'report.txt')
	model.plot_loss(save_folder + 'loss.png')


	# Predict and plot
	def analytical_solution(x, t, Re):
	    return (x / (1 + t)) / (1 + np.sqrt((1 + t) / np.exp(Re / 8)) * np.exp(Re * (x**2) / (4 * (1 + t))))
		
	n_x_plot = 100
	n_t_plot = 100
	x_vals = np.linspace(0.0, 1.0, n_x_plot)
	t_vals = np.linspace(0.0, 1.0, n_t_plot)

	TT, XX = np.meshgrid(t_vals, x_vals, indexing='ij')  # shapes (n_t_plot, n_x_plot)
	x_flat = XX.reshape(-1, 1)  # (N,1)
	t_flat = TT.reshape(-1, 1)  # (N,1)

	u_pred_flat = model.predict(x_flat, t_flat)  # shape (N,1) as np.array
	u_pred = u_pred_flat.reshape(n_t_plot, n_x_plot)

	u_exact = analytical_solution(XX, TT, Re)  # shape (n_t_plot, n_x_plot)

	error = np.abs(u_exact - u_pred)  # same shape

	fig, axes = plt.subplots(1, 3, figsize=(18, 5))


	# Exact solution
	c0 = axes[0].contourf(t_vals, x_vals, u_exact.T, levels=100, cmap='jet')
	axes[0].set_title('Exact Solution')
	axes[0].set_xlabel('t')
	axes[0].set_ylabel('x')
	fig.colorbar(c0, ax=axes[0])

	# Predicted solution
	c1 = axes[1].contourf(t_vals, x_vals, u_pred.T, levels=100, cmap='jet')
	axes[1].set_title('Predicted Solution')
	axes[1].set_xlabel('t')
	axes[1].set_ylabel('x')
	fig.colorbar(c1, ax=axes[1])

	# Absolute error
	c2 = axes[2].contourf(t_vals, x_vals, error.T, levels=100, cmap='jet')
	axes[2].set_title('Absolute Error')
	axes[2].set_xlabel('t')
	axes[2].set_ylabel('x')
	fig.colorbar(c2, ax=axes[2])

	plt.tight_layout()

	# Save figure
	save_path = os.path.join(save_folder, "result.png")
	plt.savefig(save_path, dpi=300)
	print(f"Saved figure to {save_path}")

	plt.show()
	# # Exact solution
	# im0 = axes[0].pcolormesh(TT, XX, u_exact, shading='auto', cmap='jet')
	# axes[0].set_title('Exact Solution')
	# axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
	# fig.colorbar(im0, ax=axes[0])

	# # Predicted solution
	# im1 = axes[1].pcolormesh(TT, XX, u_pred, shading='auto', cmap='jet')
	# axes[1].set_title('Predicted Solution')
	# axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
	# fig.colorbar(im1, ax=axes[1])

	# # Absolute error
	# im2 = axes[2].pcolormesh(TT, XX, error, shading='auto', cmap='jet')
	# axes[2].set_title('Absolute Error')
	# axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
	# fig.colorbar(im2, ax=axes[2])

	# plt.tight_layout()

	# save_path = os.path.join(save_folder, "result.png")
	# plt.savefig(save_path, dpi=300)  # high resolution
	# print(f"Figure saved to: {save_path}")
	# plt.show()

