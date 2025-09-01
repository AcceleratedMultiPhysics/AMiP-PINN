from kovasznay_config import *
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

    def sample(self, N):
        # x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.uniform(0, 1, size=(N, self.dim))
        x   = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * lhs(2, N)
        rho = self.f_rho(x)
        u   = self.f_u(x)
        v   = self.f_v(x)
        return x, rho, u, v

class DNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = tau
        self.c   = c
        self.network_eq = nn.Sequential(
                nn.Linear(2, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 3)
          )

        self.network_Neq = nn.Sequential(
                nn.Linear(2, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 40),
                nn.Tanh(),
                nn.Linear(40, 3)
            )

        torch.nn.init.xavier_normal_(self.network_eq[0].weight)
        torch.nn.init.zeros_(self.network_eq[0].bias)
        torch.nn.init.xavier_normal_(self.network_Neq[0].weight)
        torch.nn.init.zeros_(self.network_Neq[0].bias)

    def forward(self, x, y):

        network_in = torch.hstack((x, y))
        out_eq  = self.network_eq(network_in)
        out_Neq = self.network_Neq(network_in)
        rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
        n1, n2, n3 = out_Neq[:,0:1], out_Neq[:,1:2], out_Neq[:,2:3]

        return rho, u, v, n1, n2, n3

    def calc(self, x, y):
        network_in = torch.hstack((x, y))
        out_eq  = self.network_eq(network_in)
        rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
        return torch.hstack((rho, u, v))

class PINN():
    """ PINN Class """
    
    def __init__(self, res_sampler, bcs_sampler, savept=None):
        
        # Initialization
        self.rba = rba  # RBA weights
        self.sa = sa   # SA weights
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

        X_r, _, _, _ = self.res_sampler.sample(10000)
        X_w1, rho_w1, u_w1, v_w1 = self.WALL_1_sampler.sample(500)
        X_w2, rho_w2, u_w2, v_w2 = self.WALL_2_sampler.sample(500)
        X_w3, rho_w3, u_w3, v_w3 = self.WALL_3_sampler.sample(500)
        X_w4, rho_w4, u_w4, v_w4 = self.WALL_4_sampler.sample(500)

        X_wall_stack = np.vstack((X_w1, X_w2, X_w3, X_w4))
        rho_wall_stack = np.vstack((rho_w1, rho_w2, rho_w3, rho_w4))
        u_wall_stack = np.vstack((u_w1, u_w2, u_w3, u_w4))
        v_wall_stack = np.vstack((v_w1, v_w2, v_w3, v_w4))

        self.x_boundary   = torch.tensor(X_wall_stack[:, 0:1], requires_grad=True).float().to(device)
        self.y_boundary   = torch.tensor(X_wall_stack[:, 1:2], requires_grad=True).float().to(device)
        self.rho_boundary = torch.tensor(rho_wall_stack).float().to(device)
        self.u_boundary   = torch.tensor(u_wall_stack).float().to(device)
        self.v_boundary   = torch.tensor(v_wall_stack).float().to(device)

        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)

        self.dnn = DNN().to(device)
        width = self.dnn.network_eq[2].in_features
        print(width)
        sys.exit()
        
        # RBA initialization
        if self.rba:
            print('Residual based weighting is active.')
            self.rsum  = 0
            # self.r1sum = 0
            # self.r2sum = 0
            # self.r3sum = 0
            # self.r4sum = 0
            # self.r5sum = 0
            # self.r6sum = 0
            self.eta   = 0.001
            self.gamma = 0.999

        # SA weights initialization
        if self.sa:
            print('Self adaptive weighting is active.')
            self.N_r = tonp(self.x_r).size
            self.N_bc = tonp(self.x_boundary).size

            self.lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr1 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr2 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr3 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr4 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr5 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr6 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            self.lambc = torch.rand(self.N_bc, 1, requires_grad=True).float().to(device)*1

            self.lamr = torch.nn.Parameter(self.lamr)
            # self.lamr1 = torch.nn.Parameter(self.lamr1)
            # self.lamr2 = torch.nn.Parameter(self.lamr2)
            # self.lamr3 = torch.nn.Parameter(self.lamr3)
            # self.lamr4 = torch.nn.Parameter(self.lamr4)
            # self.lamr5 = torch.nn.Parameter(self.lamr5)
            # self.lamr6 = torch.nn.Parameter(self.lamr6)
            self.lambc = torch.nn.Parameter(self.lambc)
            # Optimizer2 (SA weights)
            self.optimizer2 = torch.optim.Adam([self.lamr] + [self.lambc] , lr=0.005, maximize=True)

        if not self.sa and not self.rba:
            print('Vanilla PINN')
            
        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, verbose=True)
        self.step_size = 1000

    def exact(self, x, y):

        u = u0 * (1 - torch.exp(lambda_*x)*torch.cos(2*PI*y))
        v = u0 * (lambda_/(2*PI) * torch.exp(lambda_*x)*torch.sin(2*PI*y))

        q2_x = u0/sqrtRT * (-lambda_ * torch.exp(lambda_*x) * torch.cos(2*PI*y))
        q2_y = u0/sqrtRT * (2*PI * torch.exp(lambda_ * x) * torch.sin(2*PI*y))

        q3_x = u0/sqrtRT * (lambda_*lambda_/(2*PI) * torch.exp(lambda_*x) * torch.sin(2*PI*y))
        q3_y = u0/sqrtRT * (lambda_ * torch.exp(lambda_*x) * torch.cos(2*PI*y))

        return u, v, q2_x, q2_y, q3_x, q3_y


    def net_u(self, x, y):
        """ Get the output of the networks """
        rho, u, v, n1, n2, n3 = self.dnn(x, y)
        
        return rho, u, v, n1, n2, n3

    def pde_residual(self, x, y):
        """ PDE Residual """

        rho, u, v, n1, n2, n3 = self.net_u(x, y)

        q1_eq = rho
        q2_eq = rho * u / sqrtRT
        q3_eq = rho * v / sqrtRT
        q4_eq = rho * u * v / RT
        q5_eq = rho * u * u / (SQRT2*RT)
        q6_eq = rho * v * v / (SQRT2*RT)

        q1 = q1_eq
        q2 = q2_eq
        q3 = q3_eq
        q4 = q4_eq + n1/SCALE
        q5 = q5_eq + n2/SCALE
        q6 = q6_eq + n3/SCALE

        q1_x = grad(q1, x)
        q1_y = grad(q1, y)

        q2_x = grad(q2, x)
        q2_y = grad(q2, y)

        q3_x = grad(q3, x)
        q3_y = grad(q3, y)

        q4_x = grad(q4, x)
        q4_y = grad(q4, y)

        q5_x = grad(q5, x)
        # q5_y = grad(q5, y)

        # q6_x = grad(q6, x)
        q6_y = grad(q6, y)

        r1 = sqrtRT * (q2_x + q3_y)
        r2 = sqrtRT * (q1_x + q4_y + SQRT2*q5_x)
        r3 = sqrtRT * (q4_x + q1_y + SQRT2*q6_y)

        r4 = sqrtRT * (q3_x + q2_y) + 1./tau * (q4 - q4_eq)
        r5 = sqrtRT * (SQRT2*q2_x)  + 1./tau * (q5 - q5_eq)
        r6 = sqrtRT * (SQRT2*q3_y)  + 1./tau * (q6 - q6_eq)

        return r1, r2, r3, r4, r5, r6

    def dirichlet(self, x, y):
        rho, u, v, n1, n2, n3 = self.net_u(x, y)

        u_exact , v_exact, q2_x, q2_y, q3_x, q3_y = self.exact(x, y)

        f4_neq = -tau * sqrtRT * (q3_x + q2_y)
        f5_neq = -tau * sqrtRT * (SQRT2*q2_x)
        f6_neq = -tau * sqrtRT * (SQRT2*q3_y)
        
        f4_neq = SCALE * f4_neq
        f5_neq = SCALE * f5_neq
        f6_neq = SCALE * f6_neq

        r4_neq = n1 - f4_neq
        r5_neq = n2 - f5_neq
        r6_neq = n3 - f6_neq

        return rho, u, v, r4_neq, r5_neq, r6_neq

    def loss_func(self):
        """ Loss function """
        
        self.optimizer.zero_grad()
        if self.sa == 1:
            self.optimizer2.zero_grad()

        # Predictions
        self.rho_boundary_pred, self.u_boundary_pred, self.v_boundary_pred, self.r4_neq_boundary_pred, self.r5_neq_boundary_pred, self.r6_neq_boundary_pred = self.dirichlet(self.x_boundary, self.y_boundary)

        self.r1_boundary_pred, self.r2_boundary_pred, self.r3_boundary_pred, self.r4_boundary_pred, self.r5_boundary_pred, self.r6_boundary_pred = self.pde_residual(self.x_boundary, self.y_boundary)

        self.r1_pred, self.r2_pred, self.r3_pred, self.r4_pred, self.r5_pred, self.r6_pred = self.pde_residual(self.x_r, self.y_r)

        if self.rba:
            r1_norm = self.eta*torch.abs(self.r1_pred)/torch.max(torch.abs(self.r1_pred))
            r2_norm = self.eta*torch.abs(self.r2_pred)/torch.max(torch.abs(self.r2_pred))
            r3_norm = self.eta*torch.abs(self.r3_pred)/torch.max(torch.abs(self.r3_pred))
            r4_norm = self.eta*torch.abs(self.r4_pred)/torch.max(torch.abs(self.r4_pred))
            r5_norm = self.eta*torch.abs(self.r5_pred)/torch.max(torch.abs(self.r5_pred))
            r6_norm = self.eta*torch.abs(self.r6_pred)/torch.max(torch.abs(self.r6_pred))

            mean_r_norm = (r1_norm + r2_norm + r3_norm + r4_norm + r5_norm + r6_norm) / 6

            self.rsum = (self.rsum*self.gamma + mean_r_norm).detach()
            loss_r1 = torch.mean((self.rsum*self.r1_pred)**2)
            loss_r2 = torch.mean((self.rsum*self.r2_pred)**2)
            loss_r3 = torch.mean((self.rsum*self.r3_pred)**2)
            loss_r4 = torch.mean((self.rsum*self.r4_pred)**2)
            loss_r5 = torch.mean((self.rsum*self.r5_pred)**2)
            loss_r6 = torch.mean((self.rsum*self.r6_pred)**2)

            # r2_norm = self.eta*torch.abs(self.r2_pred)/torch.max(torch.abs(self.r2_pred))
            # self.r2sum = (self.r2sum*self.gamma + r2_norm).detach()
            # loss_r2 = torch.mean((self.r2sum*self.r2_pred)**2)

            # r3_norm = self.eta*torch.abs(self.r3_pred)/torch.max(torch.abs(self.r3_pred))
            # self.r3sum = (self.r3sum*self.gamma + r3_norm).detach()
            # loss_r3 = torch.mean((self.r3sum*self.r3_pred)**2)

            # r4_norm = self.eta*torch.abs(self.r4_pred)/torch.max(torch.abs(self.r4_pred))
            # self.r4sum = (self.r4sum*self.gamma + r4_norm).detach()
            # loss_r4 = torch.mean((self.r4sum*self.r4_pred)**2)

            # r5_norm = self.eta*torch.abs(self.r5_pred)/torch.max(torch.abs(self.r5_pred))
            # self.r5sum = (self.r5sum*self.gamma + r5_norm).detach()
            # loss_r5 = torch.mean((self.r5sum*self.r5_pred)**2)

            # r6_norm = self.eta*torch.abs(self.r6_pred)/torch.max(torch.abs(self.r6_pred))
            # self.r6sum = (self.r6sum*self.gamma + r6_norm).detach()
            # loss_r6 = torch.mean((self.r6sum*self.r6_pred)**2)

            self.loss_r = loss_r1 + loss_r2 + loss_r3 + loss_r4 + loss_r5 + loss_r6
                    
        elif self.sa:
            self.loss_r = torch.mean((self.lamr*self.r1_pred)**2) + torch.mean((self.lamr*self.r2_pred)**2) \
                        + torch.mean((self.lamr*self.r3_pred)**2) + torch.mean((self.lamr*self.r4_pred)**2) \
                        + torch.mean((self.lamr*self.r5_pred)**2) + torch.mean((self.lamr*self.r6_pred)**2) \

            self.loss_boundary = torch.mean((self.lambc*(self.rho_boundary_pred - self.rho_boundary)) ** 2) \
                          + torch.mean((self.lambc*(self.u_boundary_pred - self.u_boundary)) ** 2) \
                          + torch.mean((self.lambc*(self.v_boundary_pred - self.v_boundary)) ** 2) \
                          + torch.mean((self.lambc*self.r4_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r5_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r6_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r1_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r2_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r3_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r4_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r5_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r6_boundary_pred)**2) \

        else:
            self.loss_r = torch.mean((self.r1_pred)**2) + torch.mean((self.r2_pred)**2) \
                        + torch.mean((self.r3_pred)**2) + torch.mean((self.r4_pred)**2) \
                        + torch.mean((self.r5_pred)**2) + torch.mean((self.r6_pred)**2) \

        if not self.sa or self.rba:

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
                # if self.rba == 1:
                #     print('loss_r1: %.3e, loss_r2: %.3e, loss_r3: %.3e, loss_r4: %.3e, loss_r5: %.3e loss_r6: %.3e' % 
                #          (loss_r1, loss_r2, loss_r3, loss_r4, loss_r5, loss_r6))
                print()
                
                self.it.append(self.iter)
                # self.l2.append(l2_rel)
                self.ll.append((self.loss.item()))

        # Optimizer step
        self.optimizer.step()
        self.losses.append(self.loss.item())
        if self.sa == True:
            self.optimizer2.step()

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class DNN_FF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = tau
        self.c   = c
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

    def forward(self, x, y, W):

        network_in = torch.hstack((x, y))

        network_in = torch.cat([torch.sin(torch.matmul(network_in, W)), 
                                torch.cos(torch.matmul(network_in, W))], 1)
        out_eq  = self.network_eq(network_in)
        out_Neq = self.network_Neq(network_in)
        rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
        n1, n2, n3 = out_Neq[:,0:1], out_Neq[:,1:2], out_Neq[:,2:3]

        return rho, u, v, n1, n2, n3

    def calc(self, x, y, W):
        network_in = torch.hstack((x, y))
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
        self.rba = rba  # RBA weights
        self.sa = sa   # SA weights
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

        X_r, _, _, _ = self.res_sampler.sample(10000)
        X_w1, rho_w1, u_w1, v_w1 = self.WALL_1_sampler.sample(500)
        X_w2, rho_w2, u_w2, v_w2 = self.WALL_2_sampler.sample(500)
        X_w3, rho_w3, u_w3, v_w3 = self.WALL_3_sampler.sample(500)
        X_w4, rho_w4, u_w4, v_w4 = self.WALL_4_sampler.sample(500)

        X_wall_stack = np.vstack((X_w1, X_w2, X_w3, X_w4))
        rho_wall_stack = np.vstack((rho_w1, rho_w2, rho_w3, rho_w4))
        u_wall_stack = np.vstack((u_w1, u_w2, u_w3, u_w4))
        v_wall_stack = np.vstack((v_w1, v_w2, v_w3, v_w4))

        self.x_boundary   = torch.tensor(X_wall_stack[:, 0:1], requires_grad=True).float().to(device)
        self.y_boundary   = torch.tensor(X_wall_stack[:, 1:2], requires_grad=True).float().to(device)
        self.rho_boundary = torch.tensor(rho_wall_stack).float().to(device)
        self.u_boundary   = torch.tensor(u_wall_stack).float().to(device)
        self.v_boundary   = torch.tensor(v_wall_stack).float().to(device)

        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.y_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)

        self.dnn = DNN_FF().to(device)
        self.nn_init(self.dnn)

        width = self.dnn.network_eq[0].in_features
        # print(width)
        # print(self.dnn)

        self.W = torch.normal(mean=0.0, std=1.0, size=(2, width // 2)).float().to(device)
        # print(self.W)
        # sys.exit()
        
        # RBA initialization
        if self.rba:
            print('Residual based weighting is active.')
            self.rsum  = 0
            # self.r1sum = 0
            # self.r2sum = 0
            # self.r3sum = 0
            # self.r4sum = 0
            # self.r5sum = 0
            # self.r6sum = 0
            self.eta   = 0.001
            self.gamma = 0.999

        # SA weights initialization
        if self.sa:
            print('Self adaptive weighting is active.')
            self.N_r = tonp(self.x_r).size
            self.N_bc = tonp(self.x_boundary).size

            self.lamr = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr1 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr2 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr3 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr4 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr5 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            # self.lamr6 = torch.rand(self.N_r, 1, requires_grad=True).float().to(device)*1
            self.lambc = torch.rand(self.N_bc, 1, requires_grad=True).float().to(device)*1

            self.lamr = torch.nn.Parameter(self.lamr)
            # self.lamr1 = torch.nn.Parameter(self.lamr1)
            # self.lamr2 = torch.nn.Parameter(self.lamr2)
            # self.lamr3 = torch.nn.Parameter(self.lamr3)
            # self.lamr4 = torch.nn.Parameter(self.lamr4)
            # self.lamr5 = torch.nn.Parameter(self.lamr5)
            # self.lamr6 = torch.nn.Parameter(self.lamr6)
            self.lambc = torch.nn.Parameter(self.lambc)
            # Optimizer2 (SA weights)
            self.optimizer2 = torch.optim.Adam([self.lamr] + [self.lambc] , lr=0.005, maximize=True)

        if not self.sa and not self.rba:
            print('Vanilla PINN')
            
        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=5e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, verbose=True)
        self.step_size = 1000

    def nn_init(self, dnn):
        for i in [0, 2, 4, 6]:
            torch.nn.init.xavier_normal_(dnn.network_eq[i].weight)
            torch.nn.init.zeros_(dnn.network_eq[i].bias)
            torch.nn.init.xavier_normal_(dnn.network_Neq[i].weight)
            torch.nn.init.zeros_(dnn.network_Neq[i].bias)

    def exact(self, x, y):

        u = u0 * (1 - torch.exp(lambda_*x)*torch.cos(2*PI*y))
        v = u0 * (lambda_/(2*PI) * torch.exp(lambda_*x)*torch.sin(2*PI*y))

        q2_x = u0/sqrtRT * (-lambda_ * torch.exp(lambda_*x) * torch.cos(2*PI*y))
        q2_y = u0/sqrtRT * (2*PI * torch.exp(lambda_ * x) * torch.sin(2*PI*y))

        q3_x = u0/sqrtRT * (lambda_*lambda_/(2*PI) * torch.exp(lambda_*x) * torch.sin(2*PI*y))
        q3_y = u0/sqrtRT * (lambda_ * torch.exp(lambda_*x) * torch.cos(2*PI*y))

        return u, v, q2_x, q2_y, q3_x, q3_y


    def net_u(self, x, y):
        """ Get the output of the networks """
        rho, u, v, n1, n2, n3 = self.dnn(x, y, self.W)
        
        return rho, u, v, n1, n2, n3

    def pde_residual(self, x, y):
        """ PDE Residual """

        rho, u, v, n1, n2, n3 = self.net_u(x, y)

        q1_eq = rho
        q2_eq = rho * u / sqrtRT
        q3_eq = rho * v / sqrtRT
        q4_eq = rho * u * v / RT
        q5_eq = rho * u * u / (SQRT2*RT)
        q6_eq = rho * v * v / (SQRT2*RT)

        q1 = q1_eq
        q2 = q2_eq
        q3 = q3_eq
        q4 = q4_eq + n1/SCALE
        q5 = q5_eq + n2/SCALE
        q6 = q6_eq + n3/SCALE

        q1_x = grad(q1, x)
        q1_y = grad(q1, y)

        q2_x = grad(q2, x)
        q2_y = grad(q2, y)

        q3_x = grad(q3, x)
        q3_y = grad(q3, y)

        q4_x = grad(q4, x)
        q4_y = grad(q4, y)

        q5_x = grad(q5, x)
        # q5_y = grad(q5, y)

        # q6_x = grad(q6, x)
        q6_y = grad(q6, y)

        r1 = sqrtRT * (q2_x + q3_y)
        r2 = sqrtRT * (q1_x + q4_y + SQRT2*q5_x)
        r3 = sqrtRT * (q4_x + q1_y + SQRT2*q6_y)

        r4 = sqrtRT * (q3_x + q2_y) + 1./tau * (q4 - q4_eq)
        r5 = sqrtRT * (SQRT2*q2_x)  + 1./tau * (q5 - q5_eq)
        r6 = sqrtRT * (SQRT2*q3_y)  + 1./tau * (q6 - q6_eq)

        return r1, r2, r3, r4, r5, r6

    def dirichlet(self, x, y):
        rho, u, v, n1, n2, n3 = self.net_u(x, y)

        u_exact , v_exact, q2_x, q2_y, q3_x, q3_y = self.exact(x, y)

        f4_neq = -tau * sqrtRT * (q3_x + q2_y)
        f5_neq = -tau * sqrtRT * (SQRT2*q2_x)
        f6_neq = -tau * sqrtRT * (SQRT2*q3_y)
        
        f4_neq = SCALE * f4_neq
        f5_neq = SCALE * f5_neq
        f6_neq = SCALE * f6_neq

        r4_neq = n1 - f4_neq
        r5_neq = n2 - f5_neq
        r6_neq = n3 - f6_neq

        return rho, u, v, r4_neq, r5_neq, r6_neq

    def loss_func(self):
        """ Loss function """
        
        self.optimizer.zero_grad()
        if self.sa == 1:
            self.optimizer2.zero_grad()

        # Predictions
        self.rho_boundary_pred, self.u_boundary_pred, self.v_boundary_pred, self.r4_neq_boundary_pred, self.r5_neq_boundary_pred, self.r6_neq_boundary_pred = self.dirichlet(self.x_boundary, self.y_boundary)

        self.r1_boundary_pred, self.r2_boundary_pred, self.r3_boundary_pred, self.r4_boundary_pred, self.r5_boundary_pred, self.r6_boundary_pred = self.pde_residual(self.x_boundary, self.y_boundary)

        self.r1_pred, self.r2_pred, self.r3_pred, self.r4_pred, self.r5_pred, self.r6_pred = self.pde_residual(self.x_r, self.y_r)

        if self.rba:
            r1_norm = self.eta*torch.abs(self.r1_pred)/torch.max(torch.abs(self.r1_pred))
            r2_norm = self.eta*torch.abs(self.r2_pred)/torch.max(torch.abs(self.r2_pred))
            r3_norm = self.eta*torch.abs(self.r3_pred)/torch.max(torch.abs(self.r3_pred))
            r4_norm = self.eta*torch.abs(self.r4_pred)/torch.max(torch.abs(self.r4_pred))
            r5_norm = self.eta*torch.abs(self.r5_pred)/torch.max(torch.abs(self.r5_pred))
            r6_norm = self.eta*torch.abs(self.r6_pred)/torch.max(torch.abs(self.r6_pred))

            mean_r_norm = (r1_norm + r2_norm + r3_norm + r4_norm + r5_norm + r6_norm) / 6

            self.rsum = (self.rsum*self.gamma + mean_r_norm).detach()
            loss_r1 = torch.mean((self.rsum*self.r1_pred)**2)
            loss_r2 = torch.mean((self.rsum*self.r2_pred)**2)
            loss_r3 = torch.mean((self.rsum*self.r3_pred)**2)
            loss_r4 = torch.mean((self.rsum*self.r4_pred)**2)
            loss_r5 = torch.mean((self.rsum*self.r5_pred)**2)
            loss_r6 = torch.mean((self.rsum*self.r6_pred)**2)

            # r2_norm = self.eta*torch.abs(self.r2_pred)/torch.max(torch.abs(self.r2_pred))
            # self.r2sum = (self.r2sum*self.gamma + r2_norm).detach()
            # loss_r2 = torch.mean((self.r2sum*self.r2_pred)**2)

            # r3_norm = self.eta*torch.abs(self.r3_pred)/torch.max(torch.abs(self.r3_pred))
            # self.r3sum = (self.r3sum*self.gamma + r3_norm).detach()
            # loss_r3 = torch.mean((self.r3sum*self.r3_pred)**2)

            # r4_norm = self.eta*torch.abs(self.r4_pred)/torch.max(torch.abs(self.r4_pred))
            # self.r4sum = (self.r4sum*self.gamma + r4_norm).detach()
            # loss_r4 = torch.mean((self.r4sum*self.r4_pred)**2)

            # r5_norm = self.eta*torch.abs(self.r5_pred)/torch.max(torch.abs(self.r5_pred))
            # self.r5sum = (self.r5sum*self.gamma + r5_norm).detach()
            # loss_r5 = torch.mean((self.r5sum*self.r5_pred)**2)

            # r6_norm = self.eta*torch.abs(self.r6_pred)/torch.max(torch.abs(self.r6_pred))
            # self.r6sum = (self.r6sum*self.gamma + r6_norm).detach()
            # loss_r6 = torch.mean((self.r6sum*self.r6_pred)**2)

            self.loss_r = loss_r1 + loss_r2 + loss_r3 + loss_r4 + loss_r5 + loss_r6
                    
        elif self.sa:
            self.loss_r = torch.mean((self.lamr*self.r1_pred)**2) + torch.mean((self.lamr*self.r2_pred)**2) \
                        + torch.mean((self.lamr*self.r3_pred)**2) + torch.mean((self.lamr*self.r4_pred)**2) \
                        + torch.mean((self.lamr*self.r5_pred)**2) + torch.mean((self.lamr*self.r6_pred)**2) \

            self.loss_boundary = torch.mean((self.lambc*(self.rho_boundary_pred - self.rho_boundary)) ** 2) \
                          + torch.mean((self.lambc*(self.u_boundary_pred - self.u_boundary)) ** 2) \
                          + torch.mean((self.lambc*(self.v_boundary_pred - self.v_boundary)) ** 2) \
                          + torch.mean((self.lambc*self.r4_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r5_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r6_neq_boundary_pred) ** 2) \
                          + torch.mean((self.lambc*self.r1_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r2_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r3_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r4_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r5_boundary_pred)**2) \
                          + torch.mean((self.lambc*self.r6_boundary_pred)**2) \

        else:
            self.loss_r = torch.mean((self.r1_pred)**2) + torch.mean((self.r2_pred)**2) \
                        + torch.mean((self.r3_pred)**2) + torch.mean((self.r4_pred)**2) \
                        + torch.mean((self.r5_pred)**2) + torch.mean((self.r6_pred)**2) \

        if not self.sa or self.rba:

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
        self.it.append(self.iter)
        # self.l2.append(l2_rel)
        self.ll.append((self.loss.item()))

        if self.iter % self.print_step == 0:
            
            with torch.no_grad():
                print('Iter %d, Loss: %.3e, Residual Loss: %.3e, Boundary Loss: %.3e, t/iter: %.1e' % 
                     (self.iter, self.loss.item(), self.loss_r.item(), self.loss_boundary.item(), self.exec_time))
                # if self.rba == 1:
                #     print('loss_r1: %.3e, loss_r2: %.3e, loss_r3: %.3e, loss_r4: %.3e, loss_r5: %.3e loss_r6: %.3e' % 
                #          (loss_r1, loss_r2, loss_r3, loss_r4, loss_r5, loss_r6))
                print()
                

        # Optimizer step
        self.optimizer.step()
        self.losses.append(self.loss.item())
        if self.sa == True:
            self.optimizer2.step()

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
