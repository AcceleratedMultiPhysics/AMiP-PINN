import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append("../../utils")
from meshReadGmsh import readMesh
from newMesh import mesh_metrics


def computeJacobian(Ex, Ey):
	# Compute the Jacobian of a triangle element
	# In this function Ex and Ey are the vertex coordinates of the element
	A = np.zeros((2,2), dtype=float)

	A[0,0] = Ex[1] - Ex[0]
	A[0,1] = Ex[2] - Ex[0]
	A[1,0] = Ey[1] - Ey[0]
	A[1,1] = Ey[2] - Ey[0]

	return LA.det(A)

def maxEdgeLength(Ex, Ey):
	l_e = 0
	for i in range(Ex.shape[0]):
		x_dist = Ex[(i+1)%3] - Ex[i]
		y_dist = Ey[(i+1)%3] - Ey[i]
		l_curr = np.sqrt(x_dist**2  + y_dist**2)

		if l_curr > l_e:
			l_e = l_curr
	return l_e


Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh('struct.msh')

NN_name = 'wing_elastic_8_hardBC'
fileDir = 'solution/elastic/' + NN_name + '_results.pickle'
# NN_name = 'wing_laplacian_8_hardBC'
# fileDir = 'solution/laplacian/' + NN_name + '_results.pickle'
with open(fileDir, 'rb') as f:
	X, Y, Ex_pred, Ey_pred = pickle.load(f)

det0 = [] # Store the determinants of the original mesh
det  = [] # Store the determinants of the new mesh

for it in range(Ntriangles):
	det0.append(computeJacobian(Ex[it], Ey[it]))
	det.append(computeJacobian(Ex_pred[it], Ey_pred[it]))

A0 = np.array(det0) / 2.0
A  = np.array(det) / 2.0 

f_A = np.zeros(shape=Ntriangles, dtype=float)
f_AR = np.zeros(shape=Ntriangles, dtype=float)

for it in range(Ntriangles):
	f_A[it]  = abs(np.log(det[it]/det0[it]) / np.log(2.0))
	
	AR0 = maxEdgeLength(Ex[it], Ey[it])**2 / A0[it]
	AR  = maxEdgeLength(Ex_pred[it], Ey_pred[it])**2 / A[it]

	f_AR[it] = abs(np.log(AR/AR0) / np.log(2.0))

print(np.max(f_A))
print(np.max(f_AR))

size_metric  = np.empty((Ntriangles, 3))
shape_metric = np.empty((Ntriangles, 3))

for i in range(Ntriangles):
	size_metric[i, 0] = f_A[i]
	size_metric[i, 1] = f_A[i]
	size_metric[i, 2] = f_A[i]

	shape_metric[i, 0] = f_AR[i]
	shape_metric[i, 1] = f_AR[i]
	shape_metric[i, 2] = f_AR[i]

# fileName = 'wing_lap_8.vtu'
fileName = 'wing_elastic_8.vtu'
mesh_metrics(fileName, Ex_pred, Ey_pred, Ntriangles, size_metric, shape_metric)