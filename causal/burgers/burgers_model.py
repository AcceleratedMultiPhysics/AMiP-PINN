from burgers_config import *
import torch.nn as nn


class Sampler:
    def __init__(self, coords, f_u=None, name=None):
        """
        coords: array of shape (2, 2)
            First row: [x_min, t_min]
            Second row: [x_max, t_max]
        f_u: optional function to evaluate u(x,t) at sampled points
        name: optional name for logging
        """
        self.coords = coords
        self.f_u = f_u
        self.name = name

    def sample(self, n_points):
        """
        n_points:
            - For walls or IC: an integer (number of points along the line)
            - For interior: a tuple (n_x, n_t)
        returns:
            X: array of shape (N, 2) with columns [x, t]
            u: array of shape (N, ?) if f_u is provided, else None
        """
        xmin, tmin = self.coords[0]
        xmax, tmax = self.coords[1]

        # WALL: x is fixed, t varies
        if xmin == xmax:
            n_t = n_points if isinstance(n_points, int) else n_points[0]
            x = np.full((n_t, 1), xmin)
            t = np.linspace(tmin, tmax, n_t)[:, None]
            X = np.hstack([x, t])

        # INITIAL or TERMINAL condition: t is fixed, x varies
        elif tmin == tmax:
            n_x = n_points if isinstance(n_points, int) else n_points[0]
            x = np.linspace(xmin, xmax, n_x)[:, None]
            t = np.full((n_x, 1), tmin)
            X = np.hstack([x, t])

        # INTERIOR: both vary, build a grid
        else:
            if not (isinstance(n_points, (tuple, list)) and len(n_points) == 2):
                raise ValueError("For interior sampling, provide n_points as (n_x, n_t).")
            n_x, n_t = n_points
            x_vals = np.linspace(xmin, xmax, n_x)
            t_vals = np.linspace(tmin, tmax, n_t)
            TT, XX = np.meshgrid(t_vals, x_vals, indexing='ij')  # shape (n_t, n_x)
            x = XX.reshape(-1, 1)
            t = TT.reshape(-1, 1)
            X = np.hstack([x, t])

        # Evaluate target function if provided
        u = self.f_u(X) if self.f_u is not None else None
        return X, u




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class DNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
          )

        torch.nn.init.xavier_normal_(self.network[0].weight)
        torch.nn.init.zeros_(self.network[0].bias)

    # def forward(self, x, y, t, W):
    def forward(self, x, t):

        network_in = torch.hstack((x,t))
        # network_in = torch.cat([torch.sin(torch.matmul(network_in, W)), 
        #                         torch.cos(torch.matmul(network_in, W))], 1)

        out  = self.network(network_in)
        u = out[:,0:1]

        return u

    def calc(self, x, t):
        network_in = torch.hstack((x,t))
        # network_in = torch.cat([torch.sin(torch.matmul(network_in, W)), 
        #                         torch.cos(torch.matmul(network_in, W))], 1)
        out  = self.network(network_in)
        u = out[:,0:1]

        return u

class PINN():
    """ PINN Class """
    
    def __init__(self, res_sampler, bcs_sampler, savept=None):
        
        # Initialization
        self.iter = 0
        self.exec_time = 0
        self.print_step = 100
        self.savept = savept
        self.Nepochs = Nepochs
        self.it = []; self.l2 = []; self.ll = []
        self.loss, self.losses = None, []
        self.best_loss = np.inf

        self.Re = Re
        self.nu = nu

        self.res_sampler = res_sampler
        self.WALL_1_sampler = bcs_sampler[0]
        self.WALL_2_sampler = bcs_sampler[1]
        self.IC_sampler = bcs_sampler[2]

        # self.M = torch.triu(torch.ones((n_t, n_t)), diagonal=1).T
        self.M = torch.triu(torch.ones((n_t, n_t), dtype=torch.float32, device=device), diagonal=1).T
        self.tol = tol


        X_w1, u_w1 = self.WALL_1_sampler.sample(n_t)
        X_w2, u_w2 = self.WALL_2_sampler.sample(n_t)
        X_ic, u_ic = self.IC_sampler.sample(n_x)
        X_r, _     = self.res_sampler.sample((n_x, n_t))  

        X_wall_stack = np.vstack((X_w1, X_w2))
        u_wall_stack = np.vstack((u_w1, u_w2))

        self.x_boundary   = torch.tensor(X_wall_stack[:, 0:1], requires_grad=True).float().to(device)
        self.t_boundary   = torch.tensor(X_wall_stack[:, 1:2], requires_grad=True).float().to(device)
        self.u_boundary   = torch.tensor(u_wall_stack).float().to(device)

        self.x_ic = torch.tensor(X_ic[:, 0:1], requires_grad=True).float().to(device)
        self.t_ic = torch.tensor(X_ic[:, 1:2], requires_grad=True).float().to(device)
        self.u_ic = torch.tensor(u_ic).float().to(device)

        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)

        self.dnn = DNN().to(device)
        width = self.dnn.network[0].in_features
        # print(width)
        # print(self.dnn)

        # self.W = torch.normal(mean=0.0, std=1.0, size=(3, width // 2)).float().to(device)
        # print(self.W)
        # sys.exit()
        
        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        self.step_size = 5000

    def exact(self, x, t):

        u = (x / (1 + t)) / (1 + torch.sqrt((1 + t) / torch.exp(self.Re / 8)) * np.exp(Re * (x**2) / (4 * (1 + t))))
        return u

    def net_u(self, x, t):
        u = self.dnn(x, t)
        return u        

    def pde_residual(self, x, t):
        """ PDE Residual """

        u = self.net_u(x,t)

        u_t = grad(u, t)
        u_x = grad(u, x)

        u_xx = grad(u_x, x)

        return u_t + u * u_x - self.nu * u_xx

    def residuals_and_weights(self, x, t, M, tol):
        r_pred = self.pde_residual(x, t)
        r_pred_grid = r_pred.view(n_t, n_x)

        L_r = torch.mean((r_pred_grid)**2,dim=1)

        L_r_vec = L_r.view(-1, 1)

        weighted = torch.matmul(M, L_r_vec)
        weighted_detached = weighted.detach()

        W = torch.exp(-tol * weighted_detached)

        return L_r, W

    def ic_residual(self, x, t):
        """ Initial condition loss"""

        u_ic_pred = self.net_u(x, t)
        L_ic = torch.mean((u_ic_pred-self.u_ic)**2)
        return L_ic

    def bc_residual(self, x, t):
        u_bc_pred = self.net_u(x,t)
        L_bc = torch.mean((u_bc_pred-self.u_boundary)**2)
        return L_bc

    def loss_func(self):
        """ Loss function """
        
        self.optimizer.zero_grad()

        # Predictions
        self.loss_r, self.W = self.residuals_and_weights(self.x_r, self.t_r, self.M, self.tol)
        self.loss_ic = self.ic_residual(self.x_ic, self.t_ic)
        self.loss_bc = self.bc_residual(self.x_boundary, self.t_boundary)


        # Loss calculation
        self.loss = 100*self.loss_ic + torch.mean(self.W * self.loss_r) + self.loss_bc
        
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:
            
            with torch.no_grad():
                print('Iter %d, Loss: %.3e, Residual Loss: %.3e, IC Loss: %.3e, BC Loss: %.3e, t/iter: %.1e' % 
                     (self.iter, self.loss.item(), self.loss_r.sum().item(), self.loss_bc.item(), self.loss_ic.item(), self.exec_time))
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
    
    def predict(self, x, t):
        x = torch.tensor(x).float().to(device)
        t = torch.tensor(t).float().to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = tonp(u)
        return u

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
