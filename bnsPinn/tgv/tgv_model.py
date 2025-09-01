from tgv_config import *
import torch.nn as nn

# SQRT2 = np.sqrt(2)
# lambda_ = Re/2 - np.sqrt(Re*Re/4 + 4*np.pi**2)
# PI = np.pi
# SCALE = 1e4

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, f_rho, f_u, f_v, name=None):
        self.dim = dim
        self.coords = coords
        self.f_rho = f_rho
        self.f_u = f_u
        self.f_v = f_v
        self.name = name

        self.t0 = tstart
        self.t1 = tfinal


    def sample(self, N):
        # x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.uniform(0, 1, size=(N, self.dim))
        x   = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * lhs(2, N)
    
        if self.name == "IC":
            t = 0*x[:,0:1]
        else:
            t = self.t0 + (self.t1 - self.t0) * lhs(1, N)

        x = np.hstack((x,t))

        rho = self.f_rho(x)
        u   = self.f_u(x)
        v   = self.f_v(x)
        return x, rho, u, v


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class DNN_FF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network_eq = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 3)
          )

        self.network_Neq = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 3)
            )

        torch.nn.init.xavier_normal_(self.network_eq[0].weight)
        torch.nn.init.zeros_(self.network_eq[0].bias)
        torch.nn.init.xavier_normal_(self.network_Neq[0].weight)
        torch.nn.init.zeros_(self.network_Neq[0].bias)

    def forward(self, x, y, t, W):

        network_in = torch.hstack((x,y,t))
        network_in = torch.cat([torch.sin(torch.matmul(network_in, W)), 
                                torch.cos(torch.matmul(network_in, W))], 1)

        out_eq  = self.network_eq(network_in)
        out_Neq = self.network_Neq(network_in)
        rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
        n1, n2, n3 = out_Neq[:,0:1], out_Neq[:,1:2], out_Neq[:,2:3]

        return rho, u, v, n1, n2, n3

    def calc(self, x, y, t, W):
        network_in = torch.hstack((x, y, t))
        network_in = torch.cat([torch.sin(torch.matmul(network_in, W)), 
                                torch.cos(torch.matmul(network_in, W))], 1)
        out_eq  = self.network_eq(network_in)
        rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
        return torch.hstack((rho, u, v))

class PINN_FF():
    """ PINN Class """
    
    def __init__(self, res_sampler, bcs_sampler, savept=None):
        
        # Initialization
        print('Fourier feature embedding')
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.savept = savept
        self.Nepochs = Nepochs
        self.it = []; self.l2 = []; self.ll = []
        self.loss, self.losses = None, []
        self.best_loss = np.inf

        self.res_sampler = res_sampler
        self.WALL_1_sampler = bcs_sampler[0]
        self.WALL_2_sampler = bcs_sampler[1]
        self.WALL_3_sampler = bcs_sampler[2]
        self.WALL_4_sampler = bcs_sampler[3]
        self.IC_sampler = bcs_sampler[4]

        X_r, _, _, _ = self.res_sampler.sample(10000)
        X_w1, rho_w1, u_w1, v_w1 = self.WALL_1_sampler.sample(500)
        X_w2, rho_w2, u_w2, v_w2 = self.WALL_2_sampler.sample(500)
        X_w3, rho_w3, u_w3, v_w3 = self.WALL_3_sampler.sample(500)
        X_w4, rho_w4, u_w4, v_w4 = self.WALL_4_sampler.sample(500)
        X_ic, rho_ic, u_ic, v_ic = self.IC_sampler.sample(500)

        X_wall_stack = np.vstack((X_w1, X_w2, X_w3, X_w4, X_ic))
        rho_wall_stack = np.vstack((rho_w1, rho_w2, rho_w3, rho_w4, rho_ic))
        u_wall_stack = np.vstack((u_w1, u_w2, u_w3, u_w4, u_ic))
        v_wall_stack = np.vstack((v_w1, v_w2, v_w3, v_w4, v_ic))

        self.x_boundary   = torch.tensor(X_wall_stack[:, 0:1], requires_grad=True).float().to(device)
        self.y_boundary   = torch.tensor(X_wall_stack[:, 1:2], requires_grad=True).float().to(device)
        self.t_boundary   = torch.tensor(X_wall_stack[:, 2:3], requires_grad=True).float().to(device)
        self.rho_boundary = torch.tensor(rho_wall_stack).float().to(device)
        self.u_boundary   = torch.tensor(u_wall_stack).float().to(device)
        self.v_boundary   = torch.tensor(v_wall_stack).float().to(device)

        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 2:3], requires_grad=True).float().to(device)

        self.dnn = DNN_FF().to(device)
        width = self.dnn.network_eq[0].in_features
        # print(width)
        # print(self.dnn)

        self.W = torch.normal(mean=0.0, std=1.0, size=(3, width // 2)).float().to(device)
        # print(self.W)
        # sys.exit()
        
        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, verbose=True)
        self.step_size = 1000

    def exact(self, x, y, t):

        u = -torch.cos(x)*torch.sin(y)*torch.exp(-2*t*nu)
        v = torch.sin(x)*torch.cos(y)*torch.exp(-2*t*nu)

        q4_t = 4*nu/RT * torch.cos(x)*torch.sin(x)*torch.cos(y)*torch.sin(y)*torch.exp(-4*nu*t)
        q5_t = -4*nu/(SQRT2*RT) * torch.cos(x)**2 * torch.sin(y)**2 * torch.exp(-4*nu*t)
        q6_t = -4*nu/(SQRT2*RT) * torch.sin(x)**2 * torch.cos(y)**2 * torch.exp(-4*nu*t)

        q2_x =  1/sqrtRT * torch.sin(x)*torch.sin(y)*torch.exp(-2*nu*t)
        q2_y = -1/sqrtRT * torch.cos(x)*torch.cos(y)*torch.exp(-2*nu*t)

        q3_x =  1/sqrtRT * torch.cos(x)*torch.cos(y)*torch.exp(-2*nu*t)
        q3_y = -1/sqrtRT * torch.sin(x)*torch.sin(y)*torch.exp(-2*nu*t)

        neq_4 = -tau * (q4_t + sqrtRT * (q3_x + q2_y))
        neq_5 = -tau * (q5_t + sqrtRT * (SQRT2*q2_x))
        neq_6 = -tau * (q6_t + sqrtRT * (SQRT2*q3_y))


        return u, v, neq_4, neq_5, neq_6


    def net_u(self, x, y, t):
        """ Get the output of the networks """
        rho, u, v, n1, n2, n3 = self.dnn(x, y, t, self.W)
        
        return rho, u, v, n1, n2, n3

    def get_eq(self, rho, u, v):

        q1_eq = rho
        q2_eq = rho * u / sqrtRT
        q3_eq = rho * v / sqrtRT
        q4_eq = rho * u * v / RT
        q5_eq = rho * u * u / (SQRT2*RT)
        q6_eq = rho * v * v / (SQRT2*RT)

        return q1_eq, q2_eq, q3_eq, q4_eq, q5_eq, q6_eq

    def pde_residual(self, x, y, t):
        """ PDE Residual """

        rho, u, v, n1, n2, n3 = self.net_u(x, y, t)
        q1_eq, q2_eq, q3_eq, q4_eq, q5_eq, q6_eq = self.get_eq(rho, u, v)

        q1 = q1_eq
        q2 = q2_eq
        q3 = q3_eq
        q4 = q4_eq + n1/SCALE
        q5 = q5_eq + n2/SCALE
        q6 = q6_eq + n3/SCALE

        q1_t = grad(q1, t)
        q1_x = grad(q1, x)
        q1_y = grad(q1, y)

        q2_t = grad(q2, t)
        q2_x = grad(q2, x)
        q2_y = grad(q2, y)

        q3_t = grad(q3, t)
        q3_x = grad(q3, x)
        q3_y = grad(q3, y)

        q4_t = grad(q4, t)
        q4_x = grad(q4, x)
        q4_y = grad(q4, y)

        q5_t = grad(q5, t)
        q5_x = grad(q5, x)
        # q5_y = grad(q5, y)

        q6_t = grad(q6, t)
        # q6_x = grad(q6, x)
        q6_y = grad(q6, y)

        r1 = q1_t + sqrtRT * (q2_x + q3_y)
        r2 = q2_t + sqrtRT * (q1_x + q4_y + SQRT2*q5_x)
        r3 = q3_t + sqrtRT * (q4_x + q1_y + SQRT2*q6_y)

        r4 = q4_t + sqrtRT * (q3_x + q2_y) + 1./tau * (q4 - q4_eq)
        r5 = q5_t + sqrtRT * (SQRT2*q2_x)  + 1./tau * (q5 - q5_eq)
        r6 = q6_t + sqrtRT * (SQRT2*q3_y)  + 1./tau * (q6 - q6_eq)

        return r1, r2, r3, r4, r5, r6

    def dirichlet(self, x, y, t):
        rho, u, v, n1, n2, n3 = self.net_u(x, y, t)

        u_exact , v_exact, neq_4, neq_5, neq_6 = self.exact(x, y, t)

        f4_neq = SCALE * neq_4
        f5_neq = SCALE * neq_5
        f6_neq = SCALE * neq_6

        r4_neq = n1 - f4_neq
        r5_neq = n2 - f5_neq
        r6_neq = n3 - f6_neq

        return rho, u, v, r4_neq, r5_neq, r6_neq

    def loss_func(self):
        """ Loss function """
        
        self.optimizer.zero_grad()

        # Predictions
        self.rho_boundary_pred, self.u_boundary_pred, self.v_boundary_pred, self.r4_neq_boundary_pred, self.r5_neq_boundary_pred, self.r6_neq_boundary_pred = self.dirichlet(self.x_boundary, self.y_boundary, self.t_boundary)

        self.r1_boundary_pred, self.r2_boundary_pred, self.r3_boundary_pred, self.r4_boundary_pred, self.r5_boundary_pred, self.r6_boundary_pred = self.pde_residual(self.x_boundary, self.y_boundary, self.t_boundary)

        self.r1_pred, self.r2_pred, self.r3_pred, self.r4_pred, self.r5_pred, self.r6_pred = self.pde_residual(self.x_r, self.y_r, self.t_r)

        self.loss_r = torch.mean((self.r1_pred)**2) + torch.mean((self.r2_pred)**2) \
                    + torch.mean((self.r3_pred)**2) + torch.mean((self.r4_pred)**2) \
                    + torch.mean((self.r5_pred)**2) + torch.mean((self.r6_pred)**2) \


        self.loss_boundary = torch.mean((self.rho_boundary_pred - self.rho_boundary) ** 2) \
                      + torch.mean((self.u_boundary_pred - self.u_boundary) ** 2) \
                      + torch.mean((self.v_boundary_pred - self.v_boundary) ** 2) \
                      + torch.mean((self.r4_neq_boundary_pred) ** 2) \
                      + torch.mean((self.r5_neq_boundary_pred) ** 2) \
                      + torch.mean((self.r6_neq_boundary_pred) ** 2) \
                      + torch.mean((self.r1_boundary_pred)**2) \
                      + torch.mean((self.r2_boundary_pred)**2) \
                      + torch.mean((self.r3_boundary_pred)**2) \
                      + torch.mean((self.r4_boundary_pred)**2) \
                      + torch.mean((self.r5_boundary_pred)**2) \
                      + torch.mean((self.r6_boundary_pred)**2) \


        # Loss calculation
        w_bc = 1
        self.loss = self.loss_r + w_bc*self.loss_boundary
        
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            
            with torch.no_grad():
                print('Iter %d, Loss: %.3e, Residual Loss: %.3e, Boundary Loss: %.3e, t/iter: %.1e' % 
                     (self.iter, self.loss.item(), self.loss_r.item(), self.loss_boundary.item(), self.exec_time))
                print()
                
                self.it.append(self.iter)
                self.ll.append((self.loss.item()))

        # Optimizer step
        self.optimizer.step()
        self.losses.append(self.loss.item())

        if self.loss.item() < self.best_loss:
            self.best_loss = self.loss.item()
            torch.save(self.dnn.state_dict(), str(self.savept)+"_best.pt")
                
    def train(self):
        """ Train model """
        
        self.dnn.train()
        for epoch in range(self.Nepochs):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time = end_time - start_time
            if (epoch+1) % self.step_size == 0:
                self.scheduler.step()

        # Write data
        a = np.array(self.it)
        c = np.array(self.ll)
        # Stack them into a 2D array.
        d = np.column_stack((a, c))
        # Write to a txt file
        np.savetxt(save_folder + 'losses.txt', d, fmt='%2.5f, %2.5f')

        if self.savept != None:
            torch.save(self.dnn.state_dict(), str(self.savept)+".pt")
    
    def predict(self, x, y):
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        self.dnn.eval()
        rho, u, v, _, _, _ = self.net_u(x, y)
        rho = tonp(rho)
        u   = tonp(u)
        v   = tonp(v)
        return rho, u, v

    def plot_loss(self, savePath):
        plt.semilogy(np.arange(0, Nepochs, 1), self.losses, label='Loss')
        plt.legend()

        plt.savefig(savePath)
        plt.clf()

    def report(self, time, fileName):
        with open(fileName, "w+") as f:
          f.write(f'Total training time: {(time / 60):4.3f} mins.\n')
          f.write(f'Total epochs       : {self.Nepochs:6.1f}.\n')
          f.write(f'Best Loss          : {self.best_loss:1.6f}\n')
          f.write(f'NN Architecture    : {self.dnn}\n')
