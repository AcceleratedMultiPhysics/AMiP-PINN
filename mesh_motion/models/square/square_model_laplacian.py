import tensorflow as tf
import numpy as np
import timeit
import math
import sys
import pickle


class mesh_motion_laplacian:
  # Initialize the class
  def __init__(self, WALL_id, Vx, Vy, Xg, Yg, distance, layers, ExistModel=0, nnDir=''):

    # Count for callback function
    self.count=0

    self.Vx       = Vx
    self.Vy       = Vy
    self.Xg       = Xg
    self.Yg       = Yg
    self.distance = distance
  
    WALL_1_id = WALL_id[:, 0]
    WALL_2_id = WALL_id[:, 1]
    WALL_3_id = WALL_id[:, 2]
    WALL_4_id = WALL_id[:, 3]

    self.x_WALL_1 = self.Vx[WALL_1_id, :]
    self.y_WALL_1 = self.Vy[WALL_1_id, :]

    self.x_WALL_2 = self.Vx[WALL_2_id, :]
    self.y_WALL_2 = self.Vy[WALL_2_id, :]

    self.x_WALL_3 = self.Vx[WALL_3_id, :]
    self.y_WALL_3 = self.Vy[WALL_3_id, :]

    self.x_WALL_4 = self.Vx[WALL_4_id, :]
    self.y_WALL_4 = self.Vy[WALL_4_id, :]

    res_start = int(max(WALL_3_id[:])+1)
  
    self.res_x = self.Vx[res_start:, :]
    self.res_y = self.Vy[res_start:, :]

    # Define layers
    self.layers = layers

    self.loss_rec = []
    self.loss_f_rec = []
    self.loss_bd_rec = []

    # Initialize NNs
    if ExistModel== 0 :
      self.weights, self.biases = self.initialize_NN(self.layers)
    else:
      print("Loading uvt NN ...")
      self.weights, self.biases = self.load_NN(nnDir, self.layers)

    # tf placeholders
    self.x_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_WALL_1_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_WALL_1_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_WALL_2_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_WALL_2_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_WALL_3_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_WALL_3_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_WALL_4_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_WALL_4_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_c_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_c_tf = tf.placeholder(tf.float32, shape=(None, 1))

    # tf graphs
    self.X_pred, self.Y_pred = self.net_mesh(self.x_tf, self.y_tf)

    self.X_WALL_1_pred, self.Y_WALL_1_pred = self.net_mesh(self.x_WALL_1_tf, self.y_WALL_1_tf)
    self.X_WALL_2_pred, self.Y_WALL_2_pred = self.net_mesh(self.x_WALL_2_tf, self.y_WALL_2_tf)
    self.X_WALL_3_pred, self.Y_WALL_3_pred = self.net_mesh(self.x_WALL_3_tf, self.y_WALL_3_tf)
    self.X_WALL_4_pred, self.Y_WALL_4_pred = self.net_mesh(self.x_WALL_4_tf, self.y_WALL_4_tf)

    self.f_X_pred, self.f_Y_pred = self.net_laplacian(self.x_c_tf, self.y_c_tf)

    self.loss_residual = tf.reduce_mean(tf.square(self.f_X_pred)) \
                       + tf.reduce_mean(tf.square(self.f_Y_pred)) \

    self.loss_WALL_1 = tf.reduce_mean(tf.square(self.X_WALL_1_pred - self.x_WALL_1_tf))
    self.loss_WALL_2 = tf.reduce_mean(tf.square(self.X_WALL_2_pred - self.x_WALL_2_tf))
    # self.loss_WALL_3 = tf.reduce_mean(tf.square(self.Y_WALL_3_pred - self.y_WALL_3_tf))
    self.loss_WALL_3 = tf.reduce_mean(tf.square(self.Y_WALL_3_pred - (-1 + 0.1*tf.sin(np.pi*self.x_WALL_3_tf))))
    self.loss_WALL_4 = tf.reduce_mean(tf.square(self.Y_WALL_4_pred - (0  + 0.1*tf.cos(2*np.pi*self.x_WALL_4_tf))))


    self.loss_boundary = self.loss_WALL_1 + self.loss_WALL_2 + self.loss_WALL_3 + self.loss_WALL_4
    self.loss = self.loss_residual + 10*self.loss_boundary

    self.global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-3
    self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                    1000, 0.9, staircase=False)
    self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                      var_list=self.weights + self.biases,
                                                      global_step=self.global_step)

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

    weights = self.sess.run(self.weights)
    biases = self.sess.run(self.biases)

    with open(fileDir, 'wb') as f:
      pickle.dump([weights, biases], f)
      print("Save uvt NN parameters successfully...")

  def load_NN(self, fileDir, layers):
    mesh_weights = []
    mesh_biases = []
    num_layers = len(layers)
    with open(fileDir, 'rb') as f:
      weights, biases = pickle.load(f)

      # Stored model must has the same # of layers
      assert num_layers == (len(weights)+1)

      for num in range(0, num_layers - 1):
        W = tf.Variable(weights[num], dtype=tf.float32)
        b = tf.Variable(biases[num], dtype=tf.float32)
        mesh_weights.append(W)
        mesh_biases.append(b)
        print(" - Load NN parameters successfully...")
    return mesh_weights, mesh_biases

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

  def net_mesh(self, x, y):
    XY = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
    X = XY[:,0:1]
    Y = XY[:,1:2]
    return X, Y

  def net_laplacian(self, x, y):

    X, Y = self.net_mesh(x, y)

    # Mesh motion using laplacian operator
    X_x = tf.gradients(X, x)[0]
    X_y = tf.gradients(X, y)[0]
    Y_x = tf.gradients(Y, x)[0]
    Y_y = tf.gradients(Y, y)[0]

    X_xx = tf.gradients(X_x, x)[0]
    X_yy = tf.gradients(X_y, y)[0]
    Y_xx = tf.gradients(Y_x, x)[0]
    Y_yy = tf.gradients(Y_y, y)[0]

    # f_u:=Sxx_x+Sxy_y
    f_X = X_xx + X_yy
    f_Y = Y_xx + Y_yy

    return f_X, f_Y

  def callback(self, loss):
    self.count = self.count+1
    self.loss_rec.append(loss)
    print('{} th iterations, Loss: {}'.format(self.count, loss))

  def fetch_minibatch(self, sampler, N):
    X, u, v, p = sampler.sample(N)
    return X, u, v, p

  def train(self, iter):
    start_time = timeit.default_timer()

    tf_dict = {self.x_c_tf: self.res_x, self.y_c_tf: self.res_y,
               self.x_WALL_1_tf: self.x_WALL_1, self.y_WALL_1_tf: self.y_WALL_1,
               self.x_WALL_2_tf: self.x_WALL_2, self.y_WALL_2_tf: self.y_WALL_2,
               self.x_WALL_3_tf: self.x_WALL_3, self.y_WALL_3_tf: self.y_WALL_3,
               self.x_WALL_4_tf: self.x_WALL_4, self.y_WALL_4_tf: self.y_WALL_4}

    for it in range(iter):
      self.sess.run(self.train_op_Adam, tf_dict)

      # Print
      if it % 10 == 0:
        elapsed = timeit.default_timer() - start_time
        loss_value = self.sess.run(self.loss, tf_dict)
        print('It: %d, Loss: %.3e, Time: %.2f' %
              (it, loss_value, elapsed))
        start_time = timeit.default_timer()

      self.loss_rec.append(self.sess.run(self.loss, tf_dict))
      self.loss_f_rec.append(self.sess.run(self.loss_residual, tf_dict))
      self.loss_bd_rec.append(self.sess.run(self.loss_boundary, tf_dict))
    return self.loss


  def predict(self, x_star, y_star):
    X_star = self.sess.run(self.X_pred, {self.x_tf: x_star, self.y_tf: y_star})
    Y_star = self.sess.run(self.Y_pred, {self.x_tf: x_star, self.y_tf: y_star})
    return X_star, Y_star

class mesh_motion_laplacian_hardBC:
  # Initialize the class
  # def __init__(self, WALL_id, Vx, Vy, layers, it, ExistModel=0, nnDir=''):
  def __init__(self, WALL_id, Vx, Vy, Xg, Yg, distance, layers, ExistModel=0, nnDir=''):

    # Count for callback function
    self.count=0

    self.Vx       = Vx
    self.Vy       = Vy
    self.Xg       = Xg
    self.Yg       = Yg
    self.distance = distance
  
    # Define layers
    self.layers = layers

    self.loss_rec = []

    # Initialize NNs
    if ExistModel== 0 :
      self.weights, self.biases = self.initialize_NN(self.layers)
    else:
      print("Loading uvt NN ...")
      self.weights, self.biases = self.load_NN(nnDir, self.layers)

    # tf placeholders
    self.x_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_tf = tf.placeholder(tf.float32, shape=(None, 1))

    self.x_c_tf = tf.placeholder(tf.float32, shape=(None, 1))
    self.y_c_tf = tf.placeholder(tf.float32, shape=(None, 1))

    # tf graphs
    self.X_pred, self.Y_pred = self.net_mesh(self.x_tf, self.y_tf)

    self.f_X_pred, self.f_Y_pred = self.net_laplacian(self.x_c_tf, self.y_c_tf)

    self.loss_residual = tf.reduce_mean(tf.square(self.f_X_pred)) \
                       + tf.reduce_mean(tf.square(self.f_Y_pred)) \

    self.loss = self.loss_residual

    self.global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1e-3
    self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                    1000, 0.9, staircase=False)
    self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
    self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                      var_list=self.weights + self.biases,
                                                      global_step=self.global_step)

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

    weights = self.sess.run(self.weights)
    biases = self.sess.run(self.biases)

    with open(fileDir, 'wb') as f:
      pickle.dump([weights, biases], f)
      print("Save uvt NN parameters successfully...")

  def load_NN(self, fileDir, layers):
    mesh_weights = []
    mesh_biases = []
    num_layers = len(layers)
    with open(fileDir, 'rb') as f:
      weights, biases = pickle.load(f)

      # Stored model must has the same # of layers
      assert num_layers == (len(weights)+1)

      for num in range(0, num_layers - 1):
        W = tf.Variable(weights[num], dtype=tf.float32)
        b = tf.Variable(biases[num], dtype=tf.float32)
        mesh_weights.append(W)
        mesh_biases.append(b)
        print(" - Load NN parameters successfully...")
    return mesh_weights, mesh_biases

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

  def net_mesh(self, x, y):
    XY = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
    X = XY[:,0:1]
    Y = XY[:,1:2]
    return X, Y

  def net_laplacian(self, x, y):

    X_nn, Y_nn = self.net_mesh(x, y)

    X = self.Xg + 2*self.distance*X_nn
    Y = self.Yg + 2*self.distance*Y_nn

    # Mesh motion using laplacian operator
    X_x = tf.gradients(X, x)[0]
    X_y = tf.gradients(X, y)[0]
    Y_x = tf.gradients(Y, x)[0]
    Y_y = tf.gradients(Y, y)[0]

    X_xx = tf.gradients(X_x, x)[0]
    X_yy = tf.gradients(X_y, y)[0]
    Y_xx = tf.gradients(Y_x, x)[0]
    Y_yy = tf.gradients(Y_y, y)[0]

    # f_u:=Sxx_x+Sxy_y
    f_X = X_xx + X_yy
    f_Y = Y_xx + Y_yy

    return f_X, f_Y

  def callback(self, loss):
    self.count = self.count+1
    self.loss_rec.append(loss)
    print('{} th iterations, Loss: {}'.format(self.count, loss))

  def fetch_minibatch(self, sampler, N):
    X, u, v, p = sampler.sample(N)
    return X, u, v, p

  def train(self, iter):
    start_time = timeit.default_timer()

    tf_dict = {self.x_c_tf: self.Vx, self.y_c_tf: self.Vy}

    for it in range(iter):
      self.sess.run(self.train_op_Adam, tf_dict)

      # Print
      if it % 10 == 0:
        elapsed = timeit.default_timer() - start_time
        loss_value = self.sess.run(self.loss, tf_dict)
        print('It: %d, Loss: %.3e, Time: %.2f' %
              (it, loss_value, elapsed))
        start_time = timeit.default_timer()

      self.loss_rec.append(self.sess.run(self.loss, tf_dict))
    return self.loss

  def predict(self, x_star, y_star):
    X_star = self.sess.run(self.X_pred, {self.x_tf: x_star, self.y_tf: y_star})
    Y_star = self.sess.run(self.Y_pred, {self.x_tf: x_star, self.y_tf: y_star})
    return X_star, Y_star
