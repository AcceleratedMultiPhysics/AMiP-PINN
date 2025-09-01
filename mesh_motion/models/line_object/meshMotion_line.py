import numpy as np
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import math
import sys
# from line_model_laplacian import mesh_motion_laplacian, mesh_motion_laplacian_hardBC
# from line_model_biharmonic import mesh_motion_biharmonic, mesh_motion_biharmonic_hardBC
from line_model_elastic import mesh_motion_elastic, mesh_motion_elastic_hardBC
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

def rotation(x, y, theta, name='deg'):
  if name == 'deg':
    theta = np.deg2rad(theta)
  xR = x*np.cos(theta) - y*np.sin(theta)
  yR = x*np.sin(theta) + y*np.cos(theta)
  return xR, yR

def bending(x, y, theta):
  angle = np.linspace(-theta, theta, x.shape[0])
  xB = x / theta * (np.sin(theta))
  yB = abs(x) / theta * (np.cos(theta) - 1)

  return xB, yB

def circle(x, y, theta):
  L = 5
  theta = theta / 2
  r = L/theta
  angle_min = np.pi / 2 - theta
  angle_max = np.pi / 2 + theta
  angle = np.linspace(angle_min, angle_max, x.shape[0])
  xc = 0
  yc = -r

  xB = xc + r*np.cos(angle)
  yB = yc + r*np.sin(angle)

  return xB, yB


if __name__ == "__main__":

  # Domain bounds
  lb = np.array([-10, -10])
  ub = np.array([10, 10])

  Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh('line.msh')



  WALL_1_id = np.where(Vx==lb[0])[0].flatten()[:, None]
  WALL_2_id = np.where(Vx==ub[0])[0].flatten()[:, None]
  WALL_3_id = np.where(Vy==lb[1])[0].flatten()[:, None]
  WALL_4_id = np.where(Vy==ub[1])[0].flatten()[:, None]
  WALL_id = np.concatenate((WALL_1_id, WALL_2_id, WALL_3_id, WALL_4_id), axis=1)

  line_id = np.array([], dtype=int)
  for i in np.where(Vy==0)[0]:
    if Vx[i,:] <= 5 and Vx[i,:] >= -5:
      line_id = np.append(line_id, i)

  boundary_id = np.concatenate((line_id, WALL_id[:,0], WALL_id[:,1], WALL_id[:,2], WALL_id[:,3]))
  res_id = np.setdiff1d(np.arange(Nnodes), boundary_id)

  s = 10
  theta = 0.1*np.pi

  def dist(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

  # create a distance vector
  distance = np.zeros_like(Vx, dtype=float)

  for i in res_id:
    # This loop is on the residual points
    x1 = Vx[i, :]
    y1 = Vy[i, :]
    cnt = 0
    for j in boundary_id:
      # This for loop is on the boundary points
      curr_dist = dist(x1, Vx[j,:], y1, Vy[j,:])
      if cnt == 0:
        min_dist = curr_dist
      else:
        min_dist = min(min_dist, curr_dist)
      cnt += 1

    distance[i,:] = min_dist

  # Network configuration
  depth = 7
  width = 50
  layers = [2] + depth*[width] + [2]

  # Training
  with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    for it in range(5):
      if it==0:
        X_p = Vx
        Y_p = Vy

      # NN_name = 'square_laplacian_test_softBC'
      NN_name = 'line_5it_rot_'+ str(it+1) +'_softBC'
      print(NN_name)

      load_network = False
      if load_network:
        # Load trained neural network
        model = mesh_motion_elastic(WALL_id, line_id, res_id,
                                      X_p, Y_p, layers, 
                                      ExistModel=1, nnDir='NN/laplacian/' + NN_name + '.pickle')
      else:
        model = mesh_motion_elastic(WALL_id, line_id, res_id,
                                      X_p, Y_p, layers)
        Niter = 40000
        start_time = time.time()
        loss = model.train(iter=Niter)
        # model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save neural network
        model.save_NN(NN_name + '.pickle')
        # Save loss history
        with open(NN_name + '_loss.pickle', 'wb') as f:
          pickle.dump([model.loss_hist, model.loss_f_hist, model.loss_line_hist, model.loss_bd_hist], f)

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
        elif i in line_id:
          Xg[i, :] = X_p[i, :]
          Xg[i, :], Yg[i, :] = rotation(X_p[i, :], Y_p[i, :], theta=0.05*np.pi, name='rad')

      corrected_NN_name = 'line_5it_rot_'+ str(it+1) +'_hardBC'
      print(corrected_NN_name)
      load_network = False
      if load_network:
        # Load trained neural network
        model = mesh_motion_elastic_hardBC(WALL_id, X_pinn, Y_pinn, Xg, Yg, distance, layers, 
                                    ExistModel=1, nnDir='NN/laplacian/' + corrected_NN_name + '.pickle')
      else:
        model = mesh_motion_elastic_hardBC(WALL_id, X_pinn, Y_pinn, Xg, Yg, distance, layers)
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
        pickle.dump([Ex_pred, Ey_pred], f)

      X_p = X
      Y_p = Y
