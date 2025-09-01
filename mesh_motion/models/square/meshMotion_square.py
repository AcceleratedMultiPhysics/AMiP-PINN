import numpy as np
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
import math
import sys
# from square_model_laplacian import mesh_motion_laplacian, mesh_motion_laplacian_hardBC
# from square_model_biharmonic import mesh_motion_biharmonic
from square_model_elastic import mesh_motion_elastic, mesh_motion_elastic_hardBC
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

def Y_interp(y0, y1, y, x):
  fy1 = 0.1 * np.cos(2*np.pi*x)
  v = y0 + (fy1 - y0) * (y-y0)/(y1-y0)
  return v

if __name__ == "__main__":

  # Domain bounds
  lb = np.array([0.0, 0.0])
  ub = np.array([1.0, 1.0])

  Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh('square.msh')
  fileName = 'square_elastic_squeeze_0_hardBC.vtu'

  mesh_motion_new_mesh(fileName, Vx, Vy, Ex, Ey, Ntriangles, Nnodes)
  sys.exit()

  distance = np.zeros((Nnodes, 1))

  for i in range(Nnodes):
    x_dist = min(abs((Vx[i, :] - lb[0])), abs((Vx[i, :] - ub[0])))
    y_dist = min(abs((Vy[i, :] - lb[1])), abs((Vy[i, :] - ub[1])))
    distance[i,:]= min(x_dist, y_dist)

  WALL_1_id = np.where(Vx==lb[0])[0].flatten()[:, None]
  WALL_2_id = np.where(Vx==ub[0])[0].flatten()[:, None]
  WALL_3_id = np.where(Vy==lb[1])[0].flatten()[:, None]
  WALL_4_id = np.where(Vy==ub[1])[0].flatten()[:, None]
  WALL_id = np.concatenate((WALL_1_id, WALL_2_id, WALL_3_id, WALL_4_id), axis=1)


  # Network configuration
  depth = 7
  width = 50
  layers = [2] + depth*[width] + [2]

  with tf.device('/device:GPU:1'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    for it in range(5):
      X_p = Vx
      Y_p = Vy
    # Training

      # NN_name = 'square_laplacian_test_softBC'
      NN_name = 'square_elastic_squeeze_' + str(it+1) + '_softBC'
      # NN_name = 'square_elastic_squeeze_softBC'
      print(NN_name)
      load_network = False
      if load_network:
        # Load trained neural network
        model = mesh_motion_elastic(WALL_id, X_p, Y_p, distance, layers, it,
                                      ExistModel=1, nnDir='NN/squeeze/' + NN_name + '.pickle')
      else:
        model = mesh_motion_elastic(WALL_id, X_p, Y_p, distance, layers, it)
        Niter = 30000
        start_time = time.time()
        loss = model.train(iter=Niter)
        # model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save neural network
        model.save_NN(NN_name + '.pickle')
        # Save loss history
        with open(NN_name + '_loss.pickle', 'wb') as f:
          pickle.dump([model.loss_rec], f)

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

      with open(NN_name + '_results.pickle', 'wb') as f:
        pickle.dump([X_pinn, Y_pinn, Ex_pred, Ey_pred], f)

      # Correction with hard BC
      Xg = np.copy(X_pinn)
      Yg = np.copy(Y_pinn)
      for i in range(Nnodes):
        if i in WALL_4_id:
          Xg[i, :] = X_p[i, :]
          Yg[i, :] = Y_p[i, :] - (it+1)*0.05 * np.sin(np.pi * X_p[i, :])
        elif i in WALL_3_id:
          Xg[i, :] = X_p[i, :]
          Yg[i, :] = Y_p[i, :] + (it+1)*0.05 * np.sin(np.pi * X_p[i, :])
        elif any(i in ls for ls in [WALL_1_id, WALL_2_id]):
          Xg[i, :] = X_p[i, :]
          Yg[i, :] = Y_p[i, :]

      corrected_NN_name = 'square_elastic_squeeze_' + str(it+1) + '_hardBC'
      # corrected_NN_name = 'square_elastic_squeeze_hardBC'
      print(corrected_NN_name)
      load_network = False
      if load_network:
        # Load trained neural network
        model = mesh_motion_elastic_hardBC(WALL_id, X_p, Y_p, Xg, Yg, distance, layers, 
                                    ExistModel=1, nnDir='NN/squeeze/'+corrected_NN_name + '.pickle')
      else:
        model = mesh_motion_elastic_hardBC(WALL_id, X_p, Y_p, Xg, Yg, distance, layers)
        Niter = 2500
        start_time = time.time()
        loss = model.train(iter=Niter)
        # model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        # Save neural network
        model.save_NN(corrected_NN_name + '.pickle')
        # Save loss history
        with open(corrected_NN_name + '_loss.pickle', 'wb') as f:
          pickle.dump([model.loss_rec], f)

      X_c, Y_c = model.predict(X_p, Y_p)
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
