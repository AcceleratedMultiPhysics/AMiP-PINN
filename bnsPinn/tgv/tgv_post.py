from tgv_config import *
from tgv_model import DNN_FF

from scipy.io import savemat, loadmat

PI = np.pi

def exact(x, y, t):

	u = -torch.cos(x)*torch.sin(y)*torch.exp(-2*t*nu)
	v = torch.sin(x)*torch.cos(y)*torch.exp(-2*t*nu)

	return u, v

file_name = save_folder + '/weights.pt'

if FF:
	model = DNN_FF()
	width = model.network_eq[0].in_features
	W = torch.normal(mean=0.0, std=1.0, size=(3, width // 2)).float()
else:
	model = DNN()
mdata = torch.load(file_name)
# print(mdata)
# sys.exit()
model.load_state_dict(mdata)
model.eval()


xx = np.linspace(xmin, xmax,50)
yy = np.linspace(ymin, ymax,50)
X, Y = np.meshgrid(xx, yy)
X = X.flatten()[:,None]
Y = Y.flatten()[:,None]

X, Y, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(mesh_file)
T = 0 * X


if FF:
	ruv = model.calc(torch.Tensor(X), torch.Tensor(Y), torch.Tensor(T), W).detach().numpy()
else:
	ruv = model.calc(torch.Tensor(X), torch.Tensor(Y)).detach().numpy()

r = ruv[:,0:1]
u = ruv[:,1:2]
v = ruv[:,2:3]

u_exact, v_exact = exact(torch.Tensor(X), torch.Tensor(Y), torch.Tensor(T))
r_exact = np.ones_like(u_exact)

r_error = np.abs(r_exact - r)
u_error = np.abs(u_exact - u)
v_error = np.abs(v_exact - v)


field_pred = [X, Y, r, u, v]
field_exact = [X, Y, r_exact, u_exact, v_exact]
field_error = [X, Y, r_error, u_error, v_error]

# Save the variables in a .mat file
# mdic = {"field_pred": field_pred, "label": "prediction"}
# scipy.io.savemat("kovasznay_pred.mat",mdict=mdic)

# mdic = {"field_exact": field_exact, "label": "exact"}
# scipy.io.savemat("kovasznay_exact.mat",mdict=mdic)

# mdic = {"field_error": field_error, "label": "error"}
# scipy.io.savemat("kovasznay_error.mat",mdict=mdic)

field_rho = np.empty((Ntriangles, 3))
field_u   = np.empty((Ntriangles, 3))
field_v   = np.empty((Ntriangles, 3))

for i in range(Ntriangles):
	field_rho[i, 0] = r_exact[EtoV[i, 0]]
	field_rho[i, 1] = r_exact[EtoV[i, 1]]
	field_rho[i, 2] = r_exact[EtoV[i, 2]]

	field_u[i, 0] = u_exact[EtoV[i, 0]]
	field_u[i, 1] = u_exact[EtoV[i, 1]]
	field_u[i, 2] = u_exact[EtoV[i, 2]]

	field_v[i, 0] = v_exact[EtoV[i, 0]]
	field_v[i, 1] = v_exact[EtoV[i, 1]]
	field_v[i, 2] = v_exact[EtoV[i, 2]]

fileName = save_folder + 'tgv_exact.vtu'
bns_plot_fields(fileName, mesh_file,
                field_rho, field_u, field_v)


for i in range(Ntriangles):
	field_rho[i, 0] = r[EtoV[i, 0]]
	field_rho[i, 1] = r[EtoV[i, 1]]
	field_rho[i, 2] = r[EtoV[i, 2]]

	field_u[i, 0] = u[EtoV[i, 0]]
	field_u[i, 1] = u[EtoV[i, 1]]
	field_u[i, 2] = u[EtoV[i, 2]]

	field_v[i, 0] = v[EtoV[i, 0]]
	field_v[i, 1] = v[EtoV[i, 1]]
	field_v[i, 2] = v[EtoV[i, 2]]

fileName = save_folder + 'tgv_pred.vtu'
bns_plot_fields(fileName, mesh_file,
                field_rho, field_u, field_v)


for i in range(Ntriangles):
	field_rho[i, 0] = r_error[EtoV[i, 0]]
	field_rho[i, 1] = r_error[EtoV[i, 1]]
	field_rho[i, 2] = r_error[EtoV[i, 2]]

	field_u[i, 0] = u_error[EtoV[i, 0]]
	field_u[i, 1] = u_error[EtoV[i, 1]]
	field_u[i, 2] = u_error[EtoV[i, 2]]

	field_v[i, 0] = v_error[EtoV[i, 0]]
	field_v[i, 1] = v_error[EtoV[i, 1]]
	field_v[i, 2] = v_error[EtoV[i, 2]]

fileName = save_folder + 'tgv_error.vtu'
bns_plot_fields(fileName, mesh_file,
                field_rho, field_u, field_v)


# plt.figure()
# plt.scatter(X, Y, c=u_exact-u, cmap='jet')
# plt.colorbar()
# plt.show()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# fig, axs = plt.subplots(2, 3,figsize=(16,9))
# u1 = axs[0, 0].scatter(X, Y, c=u, cmap='jet')
# axs[0, 0].set_title('$u_{pred}$')
# u2 = axs[0, 1].scatter(X, Y, c=u_exact, cmap='jet')
# axs[0, 1].set_title('$u_{exact}$')
# u3 = axs[0, 2].scatter(X, Y, c=np.abs(u_exact-u), cmap='jet')
# axs[0, 2].set_title('$u_{error}$')
# fig.colorbar(u1)
# fig.colorbar(u2)
# fig.colorbar(u3)
# v1 = axs[1, 0].scatter(X, Y, c=v, cmap='jet')
# axs[1, 0].set_title('$v_{pred}$')
# v2 = axs[1, 1].scatter(X, Y, c=v_exact, cmap='jet')
# axs[1, 1].set_title('$v_{exact}$')
# v3 = axs[1, 2].scatter(X, Y, c=np.abs(v_exact-v), cmap='jet')
# axs[1, 2].set_title('$v_{error}$')
# fig.colorbar(v1)
# fig.colorbar(v2)
# fig.colorbar(v3)
# fig.tight_layout()

# plt.savefig('kovasnay.pdf', dpi='figure')

# plt.show()

print("Relative u norm:" , np.linalg.norm(np.abs(u_exact - u)) / np.linalg.norm(np.abs(u_exact)))
print("Relative v norm:" , np.linalg.norm(np.abs(v_exact - v)) / np.linalg.norm(np.abs(u_exact)))


# plt.figure(1)
# plt.title('Abs Error u')
# plt.xlabel('x')
# plt.ylabel('y') 
# plt.imshow(np.abs(u_exact-u), aspect='auto', cmap='rainbow')
# plt.show()