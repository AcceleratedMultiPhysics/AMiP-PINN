from kovasznay_config import *
from kovasznay_model import PINN, PINN_FF, Sampler

def dirichlet(z, value):
	x = z[:, 0:1]
	y = z[:, 1:2]
	return value * np.ones(x.shape)

def exact_u(z):
	x = z[:, 0:1]
	y = z[:, 1:2]
	u = u0 * (1 - np.exp(lambda_*x)*np.cos(2*PI*y))
	return u

def exact_v(z):
	x = z[:, 0:1]
	y = z[:, 1:2]
	v = u0 * (lambda_/(2*PI) * np.exp(lambda_*x)*np.sin(2*PI*y))
	return v


if __name__ == "__main__":

	Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(mesh_file)

	print('Ma:  ', Ma)
	print('tau: ', tau)
	print('Re:  ', Re)
	# sys.exit()

	# WALL Left (1) //
	WALL_1_coords = np.array([[xmin, ymin], [xmin, ymax]])
	WALL_1_sampler = Sampler(2, WALL_1_coords, f_rho =lambda x: dirichlet(x,value=1.0),
	                                           f_u   =lambda x: exact_u(x),
	                                           f_v   =lambda x: exact_v(x), name='WALL_1')

	# WALL Right (2) //
	WALL_2_coords = np.array([[xmax, ymin], [xmax, ymax]])
	WALL_2_sampler = Sampler(2, WALL_2_coords, f_rho =lambda x: dirichlet(x,value=1.0),
	                                           f_u   =lambda x: exact_u(x),
	                                           f_v   =lambda x: exact_v(x), name='WALL_2')

	# WALL Lower (3) //
	WALL_3_coords = np.array([[xmin, ymin], [xmax, ymin]])
	WALL_3_sampler = Sampler(2, WALL_3_coords, f_rho =lambda x: dirichlet(x,value=1.0),
	                                           f_u   =lambda x: exact_u(x),
	                                           f_v   =lambda x: exact_v(x), name='WALL_3')

	# WALL Upper (4) //
	WALL_4_coords = np.array([[xmin, ymax], [xmax, ymax]])
	WALL_4_sampler = Sampler(2, WALL_4_coords, f_rho =lambda x: dirichlet(x,value=1.0),
	                                           f_u   =lambda x: exact_u(x),
	                                           f_v   =lambda x: exact_v(x), name='WALL_4')

	bcs_sampler = [WALL_1_sampler, WALL_2_sampler, WALL_3_sampler, WALL_4_sampler]

	# Collocation point for equation residual
	dom_coords = np.array([[xmin, ymin], [xmax, ymax]])
	res_sampler = Sampler(2, dom_coords, f_rho =lambda x: dirichlet(x, 0),
	                                     f_u   =lambda x: dirichlet(x, 0),
	                                     f_v   =lambda x: dirichlet(x, 0), name='residual')

	# Visualize the collocation points
	# x, _, _, _ = res_sampler.sample(2000)
	# x1, q1_1, q2_1, q3_1 = WALL_1_sampler.sample(50)
	# x2, q1_2, q2_2, q3_2 = WALL_2_sampler.sample(50)
	# x3, q1_3, q2_3, q3_3 = WALL_3_sampler.sample(50)
	# x4, q1_4, q2_4, q3_4 = WALL_4_sampler.sample(50)

	# x_stationary  = np.vstack((x1, x2, x3, x4))
	# q1_stationary = np.vstack((q1_1, q1_2, q1_2, q1_4))
	# q2_stationary = np.vstack((q2_1, q2_2, q2_2, q2_4))
	# q3_stationary = np.vstack((q3_1, q3_2, q3_2, q3_4))

	# # print(x_stationary.shape)
	# # sys.exit(1)

	# plt.scatter(x[:, 0:1], x[:, 1:2], marker='o', alpha=0.1, color='red')
	# plt.scatter(x_stationary[:, 0:1], x_stationary[:, 1:2], marker='o', c=q2_stationary, cmap='jet')

	# # plt.scatter(x1[:, 0:1], x1[:, 1:2], marker='o', alpha=0.1, c=q2_1, cmap='jet')
	# # plt.scatter(x2[:, 0:1], x2[:, 1:2], marker='o', alpha=0.1, c=q2_2, cmap='jet')
	# # plt.scatter(x3[:, 0:1], x3[:, 1:2], marker='o', alpha=0.1, c=q2_3, cmap='jet')
	# # plt.scatter(x4[:, 0:1], x4[:, 1:2], marker='o', alpha=0.1, c=q2_4, cmap='jet')
	# plt.show()
	# sys.exit()

	# sigma = 10.0
	# print(torch.normal(mean=torch.Tensor([0.0]), std=torch.tensor([1.0])))
	# sys.exit()

	if FF:
		model = PINN_FF(res_sampler, bcs_sampler, savept=save_folder + 'weights')
	else:
		model = PINN(res_sampler, bcs_sampler, savept=save_folder + 'weights')
	start = time.time()
	model.train()
	total_time = time.time() - start
	print(f'Total time:  {(total_time/60):2.3f} mins \n')

	model.report(total_time, save_folder + 'report.txt')
	model.plot_loss(save_folder + 'loss.png')
	rho_PINN, u_PINN, v_PINN = model.predict(Vx, Vy)


	field_rho = np.empty((Ntriangles, 3))
	field_u   = np.empty((Ntriangles, 3))
	field_v   = np.empty((Ntriangles, 3))

	for i in range(Ntriangles):
		field_rho[i, 0] = rho_PINN[EtoV[i, 0]]
		field_rho[i, 1] = rho_PINN[EtoV[i, 1]]
		field_rho[i, 2] = rho_PINN[EtoV[i, 2]]

		field_u[i, 0] = u_PINN[EtoV[i, 0]]
		field_u[i, 1] = u_PINN[EtoV[i, 1]]
		field_u[i, 2] = u_PINN[EtoV[i, 2]]

		field_v[i, 0] = v_PINN[EtoV[i, 0]]
		field_v[i, 1] = v_PINN[EtoV[i, 1]]
		field_v[i, 2] = v_PINN[EtoV[i, 2]]

	fileName = save_folder + 'kovasznay.vtu'
	bns_plot_fields(fileName, mesh_file,
	                field_rho, field_u, field_v)

	xx = np.linspace(xmin, xmax,100)
	yy = np.linspace(ymin, ymax,100)
	X, Y = np.meshgrid(xx, yy)
	X = X.flatten()[:,None]
	Y = Y.flatten()[:,None]

	p, u, v = model.predict(X, Y)
	# plt.figure()
	plt.scatter(X, Y, c=u, cmap='jet')
	plt.colorbar()
	plt.show()


