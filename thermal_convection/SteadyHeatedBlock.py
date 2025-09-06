import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import math

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf
print(tf.__version__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU:-1; GPU0: 1; GPU1: 0;

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
#tf.random.set_seed(1234)

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, WALL_1, WALL_2, WALL_3, WALL_4, SOURCE, uvt_layers, lb, ub, ExistModel=0, uvtDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub
        
        # Ra = 1e3
        Ra = 1000
        Pr = 0.7
        Re = 37.8
        # Mat. properties
        self.rho   = 1.0
        self.mu    = 1.0/Re
        self.rcp   = 1.0
        self.cond  = 1.0/(Re*Pr)

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_WALL_1 = WALL_1[:, 0:1]
        self.y_WALL_1 = WALL_1[:, 1:2]
        self.u_WALL_1 = WALL_1[:, 2:3]
        self.v_WALL_1 = WALL_1[:, 3:4]
        self.p_WALL_1 = WALL_1[:, 4:5]
        self.t_WALL_1 = WALL_1[:, 5:6]

        self.x_WALL_2 = WALL_2[:, 0:1]
        self.y_WALL_2 = WALL_2[:, 1:2]
        self.u_WALL_2 = WALL_2[:, 2:3]
        self.v_WALL_2 = WALL_2[:, 3:4]
        self.p_WALL_2 = WALL_2[:, 4:5]
        self.t_WALL_2 = WALL_2[:, 5:6]

        self.x_WALL_3 = WALL_3[:, 0:1]
        self.y_WALL_3 = WALL_3[:, 1:2]
        self.u_WALL_3 = WALL_3[:, 2:3]
        self.v_WALL_3 = WALL_3[:, 3:4]
        self.p_WALL_3 = WALL_3[:, 4:5]
        self.t_WALL_3 = WALL_3[:, 5:6]

        self.x_WALL_4 = WALL_4[:, 0:1]
        self.y_WALL_4 = WALL_4[:, 1:2]
        self.u_WALL_4 = WALL_4[:, 2:3]
        self.v_WALL_4 = WALL_4[:, 3:4]
        self.p_WALL_4 = WALL_4[:, 4:5]
        self.t_WALL_4 = WALL_4[:, 5:6]

        self.x_SOURCE = SOURCE[:, 0:1]
        self.y_SOURCE = SOURCE[:, 1:2]
        self.u_SOURCE = SOURCE[:, 2:3]
        self.v_SOURCE = SOURCE[:, 3:4]
        self.p_SOURCE = SOURCE[:, 4:5]
        self.t_SOURCE = SOURCE[:, 5:6]


        # Define layers
        self.uvt_layers = uvt_layers

        self.loss_rec = []

        # Initialize NNs
        if ExistModel== 0 :
            self.uvt_weights, self.uvt_biases = self.initialize_NN(self.uvt_layers)
        else:
            print("Loading uvt NN ...")
            self.uvt_weights, self.uvt_biases = self.load_NN(uvtDir, self.uvt_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        self.x_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL_1.shape[1]])
        self.y_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL_1.shape[1]])
        self.u_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL_1.shape[1]])
        self.v_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL_1.shape[1]])
        self.t_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.t_WALL_1.shape[1]])
        self.p_WALL_1_tf = tf.placeholder(tf.float32, shape=[None, self.p_WALL_1.shape[1]])

        self.x_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL_2.shape[1]])
        self.y_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL_2.shape[1]])
        self.u_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL_2.shape[1]])
        self.v_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL_2.shape[1]])
        self.t_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.t_WALL_2.shape[1]])
        self.p_WALL_2_tf = tf.placeholder(tf.float32, shape=[None, self.p_WALL_2.shape[1]])

        self.x_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL_3.shape[1]])
        self.y_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL_3.shape[1]])
        self.u_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL_3.shape[1]])
        self.v_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL_3.shape[1]])
        self.t_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.t_WALL_3.shape[1]])
        self.p_WALL_3_tf = tf.placeholder(tf.float32, shape=[None, self.p_WALL_3.shape[1]])

        self.x_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL_4.shape[1]])
        self.y_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL_4.shape[1]])
        self.u_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.u_WALL_4.shape[1]])
        self.v_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.v_WALL_4.shape[1]])
        self.t_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.t_WALL_4.shape[1]])
        self.p_WALL_4_tf = tf.placeholder(tf.float32, shape=[None, self.p_WALL_4.shape[1]])

        self.x_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.x_SOURCE.shape[1]])
        self.y_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.y_SOURCE.shape[1]])
        self.u_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.u_SOURCE.shape[1]])
        self.v_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.v_SOURCE.shape[1]])
        self.t_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.t_SOURCE.shape[1]])
        self.p_SOURCE_tf = tf.placeholder(tf.float32, shape=[None, self.p_SOURCE.shape[1]])


        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, self.t_pred, \
        self.ux_pred, self.vx_pred, self.tx_pred, self.ty_pred     = self.net_uvt(self.x_tf, self.y_tf)
        
        self.f_pred_u, self.f_pred_v, self.f_pred_p, self.f_pred_t = self.net_f(self.x_c_tf, self.y_c_tf)

        self.u_WALL_1_pred, self.v_WALL_1_pred, self.p_WALL_1_pred, self.t_WALL_1_pred, \
        self.ux_WALL_1_pred, self.vx_WALL_1_pred, self.tx_WALL_1_pred, self.ty_WALL_1_pred = self.net_uvt(self.x_WALL_1_tf, self.y_WALL_1_tf)

        self.u_WALL_2_pred, self.v_WALL_2_pred, self.p_WALL_2_pred, self.t_WALL_2_pred, \
        self.ux_WALL_2_pred, self.vx_WALL_2_pred, self.tx_WALL_2_pred, self.ty_WALL_2_pred = self.net_uvt(self.x_WALL_2_tf, self.y_WALL_2_tf)

        self.u_WALL_3_pred, self.v_WALL_3_pred, self.p_WALL_3_pred, self.t_WALL_3_pred, \
        self.ux_WALL_3_pred, self.vx_WALL_3_pred, self.tx_WALL_3_pred, self.ty_WALL_3_pred = self.net_uvt(self.x_WALL_3_tf, self.y_WALL_3_tf)

        self.u_WALL_4_pred, self.v_WALL_4_pred, self.p_WALL_4_pred, self.t_WALL_4_pred, \
        self.ux_WALL_4_pred, self.vx_WALL_4_pred, self.tx_WALL_4_pred, self.ty_WALL_4_pred = self.net_uvt(self.x_WALL_4_tf, self.y_WALL_4_tf)

        self.u_SOURCE_pred, self.v_SOURCE_pred, self.p_SOURCE_pred, self.t_SOURCE_pred, \
        self.ux_SOURCE_pred, self.vx_SOURCE_pred, self.tx_SOURCE_pred, self.ty_SOURCE_pred = self.net_uvt(self.x_SOURCE_tf, self.y_SOURCE_tf)


        # self.ux_WALL_2_pred, self.vx_WALL_2_pred, self.p_WALL_2_pred, self.tx_WALL_2_pred = self.net_uvt(self.x_WALL_2_tf, self.y_WALL_2_tf)
        # self.u_WALL_3_pred, self.v_WALL_3_pred, self.p_WALL_3_pred, self.t_WALL_3_pred    = self.net_uvt(self.x_WALL_3_tf, self.y_WALL_3_tf)
        # self.u_WALL_4_pred, self.v_WALL_4_pred, self.p_WALL_4_pred, self.ty_WALL_4_pred   = self.net_uvt(self.x_WALL_4_tf, self.y_WALL_4_tf)

        # self.u_SOURCE_pred, self.v_SOURCE_pred, self.p_SOURCE_pred, self.t_SOURCE_pred    = self.net_uvt(self.x_SOURCE_tf, self.y_SOURCE_tf)


        self.loss_f   = tf.reduce_mean(tf.square(self.f_pred_u))\
                      + tf.reduce_mean(tf.square(self.f_pred_v))\
                      + tf.reduce_mean(tf.square(self.f_pred_t))\
                      + tf.reduce_mean(tf.square(self.f_pred_p))
        
        # Dirichlet Velocity and Dirichlet Temperature            
        self.loss_WALL_1 = tf.reduce_mean(tf.square(self.u_WALL_1_pred - self.u_WALL_1_tf)) \
                         + tf.reduce_mean(tf.square(self.v_WALL_1_pred - self.v_WALL_1_tf)) \
                         + tf.reduce_mean(tf.square(self.t_WALL_1_pred - self.t_WALL_1_tf)) \

        # Neumann Velocity and Neumann Temperature and Dirichlet Pressure
        self.loss_WALL_2 = tf.reduce_mean(tf.square(self.ux_WALL_2_pred)) \
                         + tf.reduce_mean(tf.square(self.vx_WALL_2_pred)) \
                         + tf.reduce_mean(tf.square(self.tx_WALL_2_pred)) \
                         + tf.reduce_mean(tf.square(self.p_WALL_2_pred - self.p_WALL_2_tf))

        # Dirichlet Velocity and Dirichlet Temperature            
        self.loss_WALL_3 = tf.reduce_mean(tf.square(self.u_WALL_3_pred - self.u_WALL_3_tf)) \
                         + tf.reduce_mean(tf.square(self.v_WALL_3_pred - self.v_WALL_3_tf)) \
                         + tf.reduce_mean(tf.square(self.t_WALL_3_pred - self.t_WALL_3_tf))

        # Dirichlet Velocity and Neumann Temperature    
        self.loss_WALL_4 = tf.reduce_mean(tf.square(self.u_WALL_4_pred - self.u_WALL_4_tf)) \
                         + tf.reduce_mean(tf.square(self.v_WALL_4_pred - self.v_WALL_4_tf)) \
                         + tf.reduce_mean(tf.square(self.ty_WALL_4_pred))


        # Dirichlet Velocity and Dirichlet Temperature BC
        self.loss_SOURCE = tf.reduce_mean(tf.square(self.u_SOURCE_pred - self.u_SOURCE_tf)) \
                                + tf.reduce_mean(tf.square(self.v_SOURCE_pred - self.v_SOURCE_tf)) \
                                + tf.reduce_mean(tf.square(self.t_SOURCE_pred - self.t_SOURCE_tf))

        beta = 4.0
        self.loss = self.loss_f + beta*(self.loss_WALL_1 + self.loss_WALL_2 + self.loss_WALL_3 + 4.0*self.loss_WALL_4 + self.loss_SOURCE)

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uvt_weights + self.uvt_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})


        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uvt_weights + self.uvt_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

       
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)


    def save_NN(self, fileDir):

        uvt_weights = self.sess.run(self.uvt_weights)
        uvt_biases = self.sess.run(self.uvt_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uvt_weights, uvt_biases], f)
            print("Save uvt NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uvt_weights, uvt_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uvt_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uvt_weights[num], dtype=tf.float32)
                b = tf.Variable(uvt_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uvt(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uvt_weights, self.uvt_biases)
        psi = psips[:,0:1]
        p   = psips[:,1:2]
        t   = psips[:,2:3]
        u   = tf.gradients(psi, y)[0]
        v   =-tf.gradients(psi, x)[0]
        ux  = tf.gradients(u,   x)[0]
        vx  = tf.gradients(v,   x)[0]
        tx  = tf.gradients(t,   x)[0]
        ty  = tf.gradients(t,   y)[0]     
        return u, v, p, t, ux, vx, tx, ty

    # def net_tx_neumann(self, x, y):
    #     psips = self.neural_net(tf.concat([x, y], 1), self.uvt_weights, self.uvt_biases)
    #     psi = psips[:,0:1]
    #     p   = psips[:,1:2]
    #     t   = psips[:,2:3]
    #     u   = tf.gradients(psi, y)[0]
    #     v   =-tf.gradients(psi, x)[0]
    #     tx  = tf.gradients(t,   x)[0]
    #     return u, v, p, tx

    # def net_ty_neumann(self, x, y):
    #     psips = self.neural_net(tf.concat([x, y], 1), self.uvt_weights, self.uvt_biases)
    #     psi = psips[:,0:1]
    #     p   = psips[:,1:2]
    #     t   = psips[:,2:3]
    #     u   = tf.gradients(psi, y)[0]
    #     v   =-tf.gradients(psi, x)[0]
    #     ty  = tf.gradients(t,   y)[0]
    #     return u, v, p, ty

    # def net_uvt_neumann(self, x, y):
    #     psips = self.neural_net(tf.concat([x, y], 1), self.uvt_weights, self.uvt_biases)
    #     psi = psips[:,0:1]
    #     p   = psips[:,1:2]
    #     t   = psips[:,2:3]
    #     u   = tf.gradients(psi, y)[0]
    #     v   =-tf.gradients(psi, x)[0]
    #     ux  = tf.gradients(u,x)[0]
    #     vx  = tf.gradients(v,x)[0]
    #     tx  = tf.gradients(t,x)[0]
    #     return ux, vx, p, tx

    def net_f(self, x, y):

        rho  =self.rho
        mu   =self.mu
        rcp  =self.rcp
        cond =self.cond

        u, v, p, t, ux, vx, tx, ty = self.net_uvt(x, y)

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        t_x  = tf.gradients(t, x)[0]
        t_y  = tf.gradients(t, y)[0]
        t_xx = tf.gradients(t_x, x)[0]
        t_yy = tf.gradients(t_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        # f_u:=Sxx_x+Sxy_y
        f_u = rho*(u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) - t
        f_v = rho*(u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy)
        f_t = rcp*(u*t_x + v*t_y) - cond*(t_xx + t_yy)

        f_p = u_x + v_y

        return f_u, f_v, f_p, f_t


    def callback(self, loss):
        self.count = self.count+1
        self.loss_rec.append(loss)
        print('{} th iterations, Loss: {}'.format(self.count, loss))


    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_1_tf: self.x_WALL_1, self.y_WALL_1_tf: self.y_WALL_1, self.u_WALL_1_tf: self.u_WALL_1, self.v_WALL_1_tf: self.v_WALL_1, self.p_WALL_1_tf: self.p_WALL_1, self.t_WALL_1_tf: self.t_WALL_1,
                   self.x_WALL_2_tf: self.x_WALL_2, self.y_WALL_2_tf: self.y_WALL_2, self.u_WALL_2_tf: self.u_WALL_2, self.v_WALL_2_tf: self.v_WALL_2, self.p_WALL_2_tf: self.p_WALL_2, self.t_WALL_2_tf: self.t_WALL_2,
                   self.x_WALL_3_tf: self.x_WALL_3, self.y_WALL_3_tf: self.y_WALL_3, self.u_WALL_3_tf: self.u_WALL_3, self.v_WALL_3_tf: self.v_WALL_3, self.p_WALL_3_tf: self.p_WALL_3, self.t_WALL_3_tf: self.t_WALL_3,
                   self.x_WALL_4_tf: self.x_WALL_4, self.y_WALL_4_tf: self.y_WALL_4, self.u_WALL_4_tf: self.u_WALL_4, self.v_WALL_4_tf: self.v_WALL_4, self.p_WALL_4_tf: self.p_WALL_4, self.t_WALL_4_tf: self.t_WALL_4,
                   self.x_SOURCE_tf: self.x_SOURCE, self.y_SOURCE_tf: self.y_SOURCE, self.u_SOURCE_tf: self.u_SOURCE, self.v_SOURCE_tf: self.v_SOURCE, self.p_SOURCE_tf: self.p_SOURCE, self.t_SOURCE_tf: self.t_SOURCE,
                   self.learning_rate: learning_rate}

        loss_WALL_1 = []
        loss_WALL_2 = []
        loss_WALL_3 = []
        loss_WALL_4 = []
        loss_SOURCE = []
        loss_f = []

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            loss_WALL_1.append(self.sess.run(self.loss_WALL_1, tf_dict))
            loss_WALL_2.append(self.sess.run(self.loss_WALL_2, tf_dict))
            loss_WALL_3.append(self.sess.run(self.loss_WALL_3, tf_dict))
            loss_WALL_4.append(self.sess.run(self.loss_WALL_4, tf_dict))
            loss_SOURCE.append(self.sess.run(self.loss_SOURCE, tf_dict))
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))

        return loss_WALL_1, loss_WALL_2, loss_WALL_3, loss_WALL_4, loss_SOURCE, loss_f, self.loss

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_1_tf: self.x_WALL_1, self.y_WALL_1_tf: self.y_WALL_1, self.u_WALL_1_tf: self.u_WALL_1, self.v_WALL_1_tf: self.v_WALL_1, self.p_WALL_1_tf: self.p_WALL_1, self.t_WALL_1_tf: self.t_WALL_1,
                   self.x_WALL_2_tf: self.x_WALL_2, self.y_WALL_2_tf: self.y_WALL_2, self.u_WALL_2_tf: self.u_WALL_2, self.v_WALL_2_tf: self.v_WALL_2, self.p_WALL_2_tf: self.p_WALL_2, self.t_WALL_2_tf: self.t_WALL_2,
                   self.x_WALL_3_tf: self.x_WALL_3, self.y_WALL_3_tf: self.y_WALL_3, self.u_WALL_3_tf: self.u_WALL_3, self.v_WALL_3_tf: self.v_WALL_3, self.p_WALL_3_tf: self.p_WALL_3, self.t_WALL_3_tf: self.t_WALL_3,
                   self.x_WALL_4_tf: self.x_WALL_4, self.y_WALL_4_tf: self.y_WALL_4, self.u_WALL_4_tf: self.u_WALL_4, self.v_WALL_4_tf: self.v_WALL_4, self.p_WALL_4_tf: self.p_WALL_4, self.t_WALL_4_tf: self.t_WALL_4,
                   self.x_SOURCE_tf: self.x_SOURCE, self.y_SOURCE_tf: self.y_SOURCE, self.u_SOURCE_tf: self.u_SOURCE, self.v_SOURCE_tf: self.v_SOURCE, self.p_SOURCE_tf: self.p_SOURCE, self.t_SOURCE_tf: self.t_SOURCE}

        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.callback)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        t_star = self.sess.run(self.t_pred, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star, t_star

    def getloss(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
                   self.x_WALL_1_tf: self.x_WALL_1, self.y_WALL_1_tf: self.y_WALL_1, self.u_WALL_1_tf: self.u_WALL_1, self.v_WALL_1_tf: self.v_WALL_1, self.p_WALL_1_tf: self.p_WALL_1, self.t_WALL_1_tf: self.t_WALL_1,
                   self.x_WALL_2_tf: self.x_WALL_2, self.y_WALL_2_tf: self.y_WALL_2, self.u_WALL_2_tf: self.u_WALL_2, self.v_WALL_2_tf: self.v_WALL_2, self.p_WALL_2_tf: self.p_WALL_2, self.t_WALL_2_tf: self.t_WALL_2,
                   self.x_WALL_3_tf: self.x_WALL_3, self.y_WALL_3_tf: self.y_WALL_3, self.u_WALL_3_tf: self.u_WALL_3, self.v_WALL_3_tf: self.v_WALL_3, self.p_WALL_3_tf: self.p_WALL_3, self.t_WALL_3_tf: self.t_WALL_3,
                   self.x_WALL_4_tf: self.x_WALL_4, self.y_WALL_4_tf: self.y_WALL_4, self.u_WALL_4_tf: self.u_WALL_4, self.v_WALL_4_tf: self.v_WALL_4, self.p_WALL_4_tf: self.p_WALL_4, self.t_WALL_4_tf: self.t_WALL_4,
                   self.x_SOURCE_tf: self.x_SOURCE, self.y_SOURCE_tf: self.y_SOURCE, self.u_SOURCE_tf: self.u_SOURCE, self.v_SOURCE_tf: self.v_SOURCE, self.p_SOURCE_tf: self.p_SOURCE, self.t_SOURCE_tf: self.t_SOURCE}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_WALL_1 = self.sess.run(self.loss_WALL_1, tf_dict)
        loss_WALL_2 = self.sess.run(self.loss_WALL_2, tf_dict)
        loss_WALL_3 = self.sess.run(self.loss_WALL_3, tf_dict)
        loss_WALL_4 = self.sess.run(self.loss_WALL_4, tf_dict)
        loss_SOURCE = self.sess.run(self.loss_SOURCE, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)

        return loss_WALL_1, loss_WALL_2, loss_WALL_3, loss_WALL_4, loss_SOURCE, loss

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]


def postProcess(xmin, xmax, ymin, ymax, field_MIXED, s=2, alpha=0.5, marker='o'):

    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED, t_MIXED] = field_MIXED

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Plot MIXED result
    cf = ax[0].scatter(x_MIXED, y_MIXED, c=u_MIXED, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    ax[0].set_title(r'$u$ (m/s)')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    # cf = ax[1, 0].scatter(x_MIXED, y_MIXED, c=p_MIXED, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    # ax[1, 0].axis('square')
    # for key, spine in ax[1, 0].spines.items():
    #     if key in ['right','top','left','bottom']:
    #         spine.set_visible(False)
    # ax[1, 0].set_xticks([])
    # ax[1, 0].set_yticks([])
    # ax[1, 0].set_xlim([xmin, xmax])
    # ax[1, 0].set_ylim([ymin, ymax])
    # ax[1, 0].set_title('Pressure (Pa)')
    # fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

    # cf = ax[0, 1].scatter(x_MIXED, y_MIXED, c=v_MIXED, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    # ax[0, 1].axis('square')
    # for key, spine in ax[0, 1].spines.items():
    #     if key in ['right','top','left','bottom']:
    #         spine.set_visible(False)
    # ax[0, 1].set_xticks([])
    # ax[0, 1].set_yticks([])
    # ax[0, 1].set_xlim([xmin, xmax])
    # ax[0, 1].set_ylim([ymin, ymax])
    # ax[0, 1].set_title(r'$v$ (m/s)')
    # fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_MIXED, y_MIXED, c=t_MIXED, alpha=alpha, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax[1].axis('square')
    for key, spine in ax[1].spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    ax[1].set_title('Temperature (C)')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    plt.savefig('uvpt_HeatedBlock_w4_4loss4.pdf', dpi=300)
    plt.close('all')

def preprocess(dir='FenicsSol.mat'):
    '''
    Load reference solution from Fenics or Fluent
    '''
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    P = data['p']
    T = data['t']
    vx = data['vx']
    vy = data['vy']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    p_star = P.flatten()[:, None]
    t_star = T.flatten()[:, None]
    vx_star = vx.flatten()[:, None]
    vy_star = vy.flatten()[:, None]

    return x_star, y_star, vx_star, vy_star, p_star, t_star

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

def DelSquarePT(XY_c, xc=0.0, yc=0.0, w=1, h=1):
    '''
    delete points within a rectangle with width = w, and height = h
    '''
    
    dst = np.array([max(abs(xy[0] - xc)-0.5*w, abs(xy[1] - yc)-0.5*h) for xy in XY_c])
    return XY_c[dst>=0,:]


if __name__ == "__main__":

    # Domain bounds
    L = 1
    b = L
    H = L/10
    L1 = L
    L2 = 5 * L

    lb = np.array([0, 0])
    ub = np.array([L1+L+L2, b])
    
    # Network configuration
    uvt_layers = [2] + 8*[40] + [3]

    # WALL Left (1) // Dirichlet Temperature, Dirichlet Velocity
    Nwall1 = 40
    U1     = 1.0
    V1     = 0.0
    T1     = 0.0
    WALL_1 = [0.0, 0.0] + [0.0, b] * lhs(2, Nwall1)

    # WALL Right (2) // Dirichlet Temperature, Dirichlet Velocity
    Nwall2 = 40
    U2     = 0.0
    V2     = 0.0
    T2     = 0.0
    WALL_2 = [L1+L+L2, 0.0] + [0.0, b] * lhs(2, Nwall2)

    # WALL Lower (3) // Dirichlet Temperature, Dirichlet Pressure
    Nwall3_Left = 40
    Nwall3_Right = 5 * Nwall3_Left
    U3     = 0.0
    V3     = 0.0
    T3     = 1.0
    WALL_3_Left = [0.0, 0.0] + [L1, 0.0] * lhs(2, Nwall3_Left)
    WALL_3_Right= [L1+L, 0.0] + [L2, 0.0] * lhs(2,Nwall3_Right)
    WALL_3 = np.concatenate((WALL_3_Left, WALL_3_Right),0)

    # WALL Upper (4) // Dirichlet Temperature, Dirichlet Pressure
    Nwall4 = 250 
    U4     = 0.0
    V4     = 0.0
    T4     = 1.0
    WALL_4 = [0.0, b] + [L1+L+L2, 0.0] * lhs(2, Nwall4)

    # # Ra = 1e3
    # Ra = 1000
    # Pr = 0.71
    # Re = 100
    
    x_WALL_1 = WALL_1[:,0:1]
    y_WALL_1 = WALL_1[:,1:2]
    u_WALL_1 = U1*np.ones(x_WALL_1.shape)
    v_WALL_1 = V1*(1.0 + x_WALL_1)
    p_WALL_1 = 0*(1.0 + x_WALL_1)
    t_WALL_1 = T1*(1.0 + x_WALL_1) 
    WALL_1 = np.concatenate((WALL_1, u_WALL_1, v_WALL_1, p_WALL_1, t_WALL_1), 1)

    x_WALL_2 = WALL_2[:,0:1]
    y_WALL_2 = WALL_2[:,1:2]
    u_WALL_2 = U2*(1.0 - np.square(y_WALL_2))
    v_WALL_2 = V2*(0.0 + x_WALL_2)
    p_WALL_2 = 0*(0.0 + x_WALL_2)
    t_WALL_2 = T2*(1.0 + x_WALL_1)
    WALL_2 = np.concatenate((WALL_2, u_WALL_2, v_WALL_2, p_WALL_2, t_WALL_2), 1)

    x_WALL_3 = WALL_3[:,0:1]
    # x_WALL_3 = np.array([xx for xx in x_WALL_3 if xx<=L or xx>=L+L1])
    y_WALL_3 = WALL_3[:,1:2]
    u_WALL_3 = U3*(0.0 + y_WALL_3)
    v_WALL_3 = V3*(1.0 + y_WALL_3)
    p_WALL_3 = 0*(1.0 + y_WALL_3)
    t_WALL_3 = T3 * np.ones(x_WALL_3.shape)
    WALL_3 = np.concatenate((WALL_3, u_WALL_3, v_WALL_3, p_WALL_3, t_WALL_3), 1)

    x_WALL_4 = WALL_4[:,0:1]
    y_WALL_4 = WALL_4[:,1:2]
    u_WALL_4 = U4*(0.0 + y_WALL_4)
    v_WALL_4 = V4*(0.0 + y_WALL_4)
    p_WALL_4 = 0*(1.0 + y_WALL_4)
    t_WALL_4 = T4 * np.ones(x_WALL_4.shape)
    WALL_4 = np.concatenate((WALL_4, u_WALL_4, v_WALL_4, p_WALL_4, t_WALL_4), 1)

    # Left wall of the source block
    Nsource1 = 10
    USource = 0
    VSource = 0
    TSource = 1.0
    Source_WALL_1 = [L1, 0.0] + [0.0, H] * lhs(2,Nsource1)

    # Top wall of the source block
    Nsource2 = 100
    Source_WALL_2 = [L1, H] + [L, 0.0] * lhs(2,Nsource2)

    # Right wall of the source block
    Nsource3 = 10
    Source_WALL_3 = [L1+L, 0.0] + [0.0, H] * lhs(2,Nsource3)

    SOURCE = np.concatenate((Source_WALL_1, Source_WALL_2, Source_WALL_3),0)


    x_SOURCE = SOURCE[:,0:1]
    y_SOURCE = SOURCE[:,1:2]
    u_SOURCE = USource*(1.0 + x_SOURCE)
    v_SOURCE = VSource*(1.0 + x_SOURCE)
    p_SOURCE = 0*(1.0 + x_SOURCE)
    t_SOURCE = TSource * np.ones(x_SOURCE.shape)
    SOURCE   = np.concatenate((SOURCE, u_SOURCE, v_SOURCE, p_SOURCE, t_SOURCE), 1)


    Ndomain = 1500;
    # Collocation point for equation residual
    XY_c = lb + (ub - lb) * lhs(2, Ndomain)
    XY_c = DelSquarePT(XY_c, xc=L1+L/2, yc=H/2, w=L, h=H)

    XY_c = np.concatenate((XY_c, WALL_1[:,0:2], WALL_2[:,0:2], WALL_3[:,0:2], WALL_4[:,0:2]), 0)


    print(XY_c.shape)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(WALL_1[:, 0:1], WALL_1[:, 1:2], marker='o', alpha=0.2 ,color='green')
    plt.scatter(WALL_2[:, 0:1], WALL_2[:, 1:2], marker='o', alpha=0.2 ,color='orange')
    plt.scatter(WALL_3[:, 0:1], WALL_3[:, 1:2], marker='o', alpha=0.2 ,color='red')
    plt.scatter(WALL_4[:, 0:1], WALL_4[:, 1:2], marker='o', alpha=0.2 ,color='purple')
    plt.scatter(SOURCE[:, 0:1], SOURCE[:, 1:2], marker='o', alpha=0.2 ,color='black')
    plt.show()

    with tf.device('/device:GPU:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        #Train from scratch
        model = PINN_laminar_flow(XY_c, WALL_1, WALL_2, WALL_3, WALL_4, SOURCE, uvt_layers, lb, ub)

        Niter=25000
        start_time = time.time()
        loss_WALL_1, loss_WALL_2, loss_WALL_3, loss_WALL_4, loss_SOURCE, loss_f, loss = model.train(iter=Niter, learning_rate=5e-3)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Save neural network
        model.save_NN('HeatedBlock_w4_4loss4.pickle')

        # Save loss history
        with open('loss_history_HeatedBlock_w4_4loss4.pickle', 'wb') as f:
            pickle.dump(model.loss_rec, f)


        # # Load trained neural network
        # model = PINN_laminar_flow(XY_c, WALL_1, WALL_2, WALL_3, WALL_4, SOURCE, uvt_layers, lb, ub, ExistModel = 1, uvtDir = 'HeatedBlock_w4_4loss4.pickle')

        
        # Get mixed-form PINN prediction
        x_1 = np.linspace(0, L1, 30)
        y_1 = np.linspace(0, b, 30)
        X1, Y1 = np.meshgrid(x_1, y_1)
        X1 = X1.flatten()[:, None]
        Y1 = Y1.flatten()[:, None]

        x_2 = np.linspace(L1, L1+L, 30)
        y_2 = np.linspace(H, b, 30)
        X2, Y2 = np.meshgrid(x_2, y_2)
        X2 = X2.flatten()[:, None]
        Y2 = Y2.flatten()[:, None]

        x_3 = np.linspace(L1+L, L1+L+L2, 150)
        y_3 = np.linspace(0, b, 30)
        X3, Y3 = np.meshgrid(x_3, y_3)
        X3 = X3.flatten()[:, None]
        Y3 = Y3.flatten()[:, None]

        x_PINN = np.concatenate((X1, X2, X3), 0)
        y_PINN = np.concatenate((Y1, Y2, Y3), 0)


        # x_PINN = np.linspace(0, L1+L+L2, 251)
        # y_PINN = np.linspace(0.0, b, 251)
        # x_PINN, y_PINN = np.meshgrid(x_PINN, y_PINN)
        # x_PINN = x_PINN.flatten()[:, None]
        # y_PINN = y_PINN.flatten()[:, None]
        # dst = max(abs(x_PINN - (L1+L/2))-0.5*L, abs(y_PINN - H/2)-0.5*H)
        # dst = x_PINN - (L1+L/2) - 0.5*L + y_PINN - H/2 - 0.5*H
        # x_PINN = x_PINN[dst >= 0]
        # y_PINN = y_PINN[dst >= 0]
        # x_PINN = x_PINN.flatten()[:, None]
        # y_PINN = y_PINN.flatten()[:, None]


        u_PINN, v_PINN, p_PINN, t_PINN = model.predict(x_PINN, y_PINN)
        field_MIXED = [x_PINN, y_PINN, u_PINN, v_PINN, p_PINN, t_PINN]


        # Save the variables in a .mat file
        mdic = {"field_MIXED": field_MIXED, "label": "HeatedBlock_w4_4loss4"}
        scipy.io.savemat("HeatedBlock_w4_4loss4.mat",mdict=mdic)

        # Plot the comparison of u, v, p, t
        postProcess(xmin=0, xmax=L1+L+L2, ymin=0, ymax=b, field_MIXED=field_MIXED, s=4, alpha=0.5)
