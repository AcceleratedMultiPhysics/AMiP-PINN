import numpy as np
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import math
import sys
from meshModel_laplacian import mesh_motion_laplacian, mesh_motion_laplacian_hardBC
# from meshModel_elastic import mesh_motion_elastic, mesh_motion_elastic_hardBC
sys.path.append("../../utils")
from meshReadGmsh import readMesh
from newMesh import mesh_motion_new_mesh

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf
print(tf.__version__)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CPU:-1; GPU0: 1; GPU1: 0;

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
#tf.random.set_seed(1234)

def transformation(x, y, theta, name='deg'):
  if name == 'deg':
    theta = np.deg2rad(theta)
  xR = x*np.cos(theta) - y*np.sin(theta)
  yR = x*np.sin(theta) + y*np.cos(theta)
  return xR, yR

if __name__ == "__main__":

  # Domain bounds
  lb = np.array([-10, -10])
  ub = np.array([10, 10])

  lbc = np.array([-5, -0.5])
  ubc = np.array([5, 0.5])

  # lb = np.array([-1, -1])
  # ub = np.array([1, 1])

  # lbc = np.array([-0.5, -0.01])
  # ubc = np.array([0.5, 0.01])

  Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh('struct.msh')
  # fileName = 'original_mesh' + '.vtu'  
  # mesh_motion_new_mesh(fileName, Vx, Vy, Ex, Ey, Ntriangles, Nnodes)
  # sys.exit()

  WALL_1_id = np.where(Vx==lb[0])[0].flatten()[:, None]
  WALL_2_id = np.where(Vx==ub[0])[0].flatten()[:, None]
  WALL_3_id = np.where(Vy==lb[1])[0].flatten()[:, None]
  WALL_4_id = np.where(Vy==ub[1])[0].flatten()[:, None]
  WALL_id = np.concatenate((WALL_1_id, WALL_2_id, WALL_3_id, WALL_4_id), axis=1)

  rec_1_id = []
  rec_2_id = []
  rec_3_id = []
  rec_4_id = []

  for i in np.where(Vx==lbc[0])[0]:
    if Vy[i, :] <= ubc[1] and Vy[i, :] >= lbc[1]:
      rec_1_id.append(i)

  for i in np.where(Vx==ubc[0])[0]:
    if Vy[i, :] <= ubc[1] and Vy[i, :] >= lbc[1]:
      rec_2_id.append(i)

  for i in np.where(Vy==lbc[1])[0]:
    if Vx[i, :] <= ubc[0] and Vx[i, :] >= lbc[0]:
      rec_3_id.append(i)

  for i in np.where(Vy==ubc[1])[0]:
    if Vx[i, :] <= ubc[0] and Vx[i, :] >= lbc[0]:
      rec_4_id.append(i)

  res_start = int(max(rec_3_id)) + 1

  # plt.scatter(Vx[WALL_1_id, :], Vy[WALL_1_id,:])
  # plt.scatter(Vx[WALL_2_id, :], Vy[WALL_2_id,:])
  # plt.scatter(Vx[WALL_3_id, :], Vy[WALL_3_id,:])
  # plt.scatter(Vx[WALL_4_id, :], Vy[WALL_4_id,:])
  # plt.scatter(Vx[rec_1_id, :], Vy[rec_1_id,:])
  # plt.scatter(Vx[rec_2_id, :], Vy[rec_2_id,:])
  # plt.scatter(Vx[rec_3_id, :], Vy[rec_3_id,:])
  # plt.scatter(Vx[rec_4_id, :], Vy[rec_4_id,:])
  # plt.scatter(Vx[res_start:, :], Vy[res_start:,:])
  # plt.show()
  # sys.exit()

  def dist(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

  # create a distance vector
  distance = np.zeros_like(Vx, dtype=float)

  for i in np.arange(res_start, Nnodes):
    # This loop is on the residual points
    x1 = Vx[i, :]
    y1 = Vy[i, :]
    for j in range(res_start):
      # This for loop starts from zero and ends in the last boundary id
      curr_dist = dist(x1, Vx[j,:], y1, Vy[j,:])
      if j == 0:
        min_dist = curr_dist
      else:
        min_dist = min(min_dist, curr_dist)

    distance[i,:] = min_dist

  # plt.scatter(Vx, Vy, c=distance, cmap='jet')
  # plt.show()
  # sys.exit()

  xmin = -5
  xmax = 5
  L = xmax - xmin

  # Network configuration
  depth = 7
  width = 50
  layers = [2] + depth*[width] + [2]

  # Training
  with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    for it in range(8):
      if it==0:
        X_p = Vx
        Y_p = Vy

      # NN_name = 'square_laplacian_test_softBC'
      NN_name = 'wing_laplacian_'+ str(it+1) +'_softBC'
      print(NN_name)
      load_network = True
      if load_network:
        # Load trained neural network
        model = mesh_motion_laplacian(WALL_id, rec_1_id, rec_2_id, rec_3_id,rec_4_id, it,
                                    xmin, xmax, X_p, Y_p, layers, 
                                    ExistModel=1, nnDir='NN/' + NN_name + '.pickle')
      else:
        model = mesh_motion_laplacian(WALL_id, rec_1_id, rec_2_id, rec_3_id, rec_4_id, it,
                                    xmin, xmax, X_p, Y_p, layers)
        Niter = 40000
        start_time = time.time()
        loss = model.train(iter=Niter)
        # model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save neural network
        model.save_NN(NN_name + '.pickle')
        # Save loss history
        with open(NN_name + '_loss.pickle', 'wb') as f:
          pickle.dump([model.loss_hist, model.loss_f_hist, model.loss_rec_hist, model.loss_bd_hist], f)

      X_pinn, Y_pinn = model.predict(X_p, Y_p)

      Ex_pred = np.zeros_like(EtoV, dtype=float)
      Ey_pred = np.zeros_like(EtoV, dtype=float)
      for i in range(EtoV.shape[0]):
        Ex_pred[i, 0] = X_pinn[EtoV[i, 0], :]
        Ex_pred[i, 1] = X_pinn[EtoV[i, 1], :]
        Ex_pred[i, 2] = X_pinn[EtoV[i, 2], :]

        Ey_pred[i, 0] = Y_pinn[EtoV[i, 0], :]
        Ey_pred[i, 1] = Y_pinn[EtoV[i, 1], :]
        Ey_pred[i, 2] = Y_pinn[EtoV[i, 2], :]

      fileName = NN_name + '.vtu'  
      mesh_motion_new_mesh(fileName, X_pinn, Y_pinn, Ex_pred, Ey_pred, Ntriangles, Nnodes)

      # Correction with hard BC
      Xg = np.copy(X_pinn)
      Yg = np.copy(Y_pinn)
      for i in range(Nnodes):
        if any(i in ls for ls in [WALL_1_id, WALL_2_id, WALL_3_id, WALL_4_id]):
          Xg[i, :] = X_p[i, :]
          Yg[i, :] = Y_p[i, :]
        if any(i in ls for ls in [rec_1_id, rec_2_id, rec_3_id, rec_4_id]):
          Xg[i, :] = X_p[i, :]
          if it in [0, 1, 6, 7]:
            Yg[i, :] = Y_p[i, :] + 2.0*np.sin(np.pi/2 * (X_p[i,:] + L/2)/L)
          else:
            Yg[i, :] = Y_p[i, :] - 2.0*np.sin(np.pi/2 * (X_p[i,:] + L/2)/L)

      # Hard Boundary condition enforcement
      corrected_NN_name = 'wing_laplacian_'+ str(it+1) +'_hardBC'
      print(corrected_NN_name)
      load_network = False
      if load_network:
        # Load trained neural network
        model = mesh_motion_laplacian_hardBC(WALL_id, X_pinn, Y_pinn, Xg, Yg, distance, layers, 
                                    ExistModel=1, nnDir='NN/' + corrected_NN_name + '.pickle')
      else:
        model = mesh_motion_laplacian_hardBC(WALL_id, X_pinn, Y_pinn, Xg, Yg, distance, layers)
        Niter = 5000
        start_time = time.time()
        loss = model.train(iter=Niter)
        # model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save neural network
        model.save_NN(corrected_NN_name + '.pickle')
        # Save loss history
        with open(corrected_NN_name + '_loss.pickle', 'wb') as f:
          pickle.dump([model.loss_rec], f)

      X_c, Y_c = model.predict(X_pinn, Y_pinn)
      X = Xg + distance*X_c
      Y = Yg + distance*Y_c

      Ex_pred = np.zeros_like(EtoV, dtype=float)
      Ey_pred = np.zeros_like(EtoV, dtype=float)
      for i in range(EtoV.shape[0]):
        Ex_pred[i, 0] = X[EtoV[i, 0], :]
        Ex_pred[i, 1] = X[EtoV[i, 1], :]
        Ex_pred[i, 2] = X[EtoV[i, 2], :]

        Ey_pred[i, 0] = Y[EtoV[i, 0], :]
        Ey_pred[i, 1] = Y[EtoV[i, 1], :]
        Ey_pred[i, 2] = Y[EtoV[i, 2], :]

      fileName = corrected_NN_name + '.vtu'  
      mesh_motion_new_mesh(fileName, X, Y, Ex_pred, Ey_pred, Ntriangles, Nnodes)

      with open(corrected_NN_name + '_results.pickle', 'wb') as f:
        pickle.dump([X, Y, Ex_pred, Ey_pred], f)

      X_p = X
      Y_p = Y
