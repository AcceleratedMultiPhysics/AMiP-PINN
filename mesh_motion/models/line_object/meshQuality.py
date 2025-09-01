import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import sys
import os
import pickle
sys.path.append("../../utils")
from meshReadGmsh import readMesh
from newMesh import mesh_metrics

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "grid.color": "0.5",
    "grid.linestyle": "-",
    "lines.color": "g",
})


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

f_A_inf = [0]
f_AR_inf = [0]

Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh('line.msh')

for it in range(10):
	NN_name = 'line_elastic_trans_' + str(it+1) +'_hardBC'
	fileDir = 'solution/elastic/' + NN_name + '_results.pickle'

	with open(fileDir, 'rb') as f:
		Ex_pred, Ey_pred = pickle.load(f)

	det0 = [] # Store the determinants of the original mesh
	det  = [] # Store the determinants of the new mesh

	for j_it in range(Ntriangles):
		det0.append(computeJacobian(Ex[j_it], Ey[j_it]))
		det.append(computeJacobian(Ex_pred[j_it], Ey_pred[j_it]))

	A0 = np.array(det0) / 2.0
	A  = np.array(det) / 2.0 

	f_A = np.zeros(shape=Ntriangles, dtype=float)
	f_AR = np.zeros(shape=Ntriangles, dtype=float)

	for m_it in range(Ntriangles):
		f_A[m_it]  = abs(np.log(det[m_it]/det0[m_it]) / np.log(2.0))
		
		AR0 = maxEdgeLength(Ex[m_it], Ey[m_it])**2 / A0[m_it]
		AR  = maxEdgeLength(Ex_pred[m_it], Ey_pred[m_it])**2 / A[m_it]

		f_AR[m_it] = abs(np.log(AR/AR0) / np.log(2.0))

	f_A_inf.append(np.max(f_A))
	f_AR_inf.append(np.max(f_AR))

	size_metric  = np.empty((Ntriangles, 3))
	shape_metric = np.empty((Ntriangles, 3))

	for i in range(Ntriangles):
		size_metric[i, 0] = f_A[i]
		size_metric[i, 1] = f_A[i]
		size_metric[i, 2] = f_A[i]

		shape_metric[i, 0] = f_AR[i]
		shape_metric[i, 1] = f_AR[i]
		shape_metric[i, 2] = f_AR[i]

	fileName = 'Figures/' + 'line_elastic_trans.vtu'

f_A_5_inf = [0]
f_AR_5_inf = [0]


for it in range(5):
	NN_name = 'line_5it_rot_' + str(it+1) +'_hardBC'
	fileDir = 'solution/rotation/' + NN_name + '_results.pickle'

	with open(fileDir, 'rb') as f:
		Ex_pred, Ey_pred = pickle.load(f)

	det0 = [] # Store the determinants of the original mesh
	det  = [] # Store the determinants of the new mesh

	for j_it in range(Ntriangles):
		det0.append(computeJacobian(Ex[j_it], Ey[j_it]))
		det.append(computeJacobian(Ex_pred[j_it], Ey_pred[j_it]))

	A0 = np.array(det0) / 2.0
	A  = np.array(det) / 2.0 

	f_A_5 = np.zeros(shape=Ntriangles, dtype=float)
	f_AR_5 = np.zeros(shape=Ntriangles, dtype=float)

	for m_it in range(Ntriangles):
		f_A_5[m_it]  = abs(np.log(det[m_it]/det0[m_it]) / np.log(2.0))
		
		AR0 = maxEdgeLength(Ex[m_it], Ey[m_it])**2 / A0[m_it]
		AR  = maxEdgeLength(Ex_pred[m_it], Ey_pred[m_it])**2 / A[m_it]

		f_AR_5[m_it] = abs(np.log(AR/AR0) / np.log(2.0))

	f_A_5_inf.append(np.max(f_A_5))
	f_AR_5_inf.append(np.max(f_AR_5))

	size_metric  = np.empty((Ntriangles, 3))
	shape_metric = np.empty((Ntriangles, 3))

	for i in range(Ntriangles):
		size_metric[i, 0] = f_A_5[i]
		size_metric[i, 1] = f_A_5[i]
		size_metric[i, 2] = f_A_5[i]

		shape_metric[i, 0] = f_AR_5[i]
		shape_metric[i, 1] = f_AR_5[i]
		shape_metric[i, 2] = f_AR_5[i]

	fileName = 'Figures/' + 'line_elastic_trans.vtu'

f_AR_best = [0]
with open('rot_bestStiffened_shape.csv', 'r') as f:
# with open('trans_bestStiffened_shape.csv', 'r') as f:
  content = f.readlines()
  for ii in range(10):
  	x, y = content[ii].split(',')
  	f_AR_best.append(float(y.strip()))

f_AR_zero = [0]
with open('rot_noStiffening_shape.csv', 'r') as f:
# with open('trans_noStiffened_shape.csv', 'r') as f:
  content = f.readlines()
  for ii in range(10):
  	if ii > 5:
  		f_AR_zero.append(6)
  		continue
  	x, y = content[ii].split(',')
  	f_AR_zero.append(float(y.strip()))

f_A_best = [0]
with open('rot_bestStiffened_area.csv', 'r') as f:
# with open('trans_bestStiffened_area.csv', 'r') as f:
  content = f.readlines()
  for ii in range(10):
  	x, y = content[ii].split(',')
  	f_A_best.append(float(y.strip()))

f_A_zero = [0]
# with open('rot_noStiffening_area.csv', 'r') as f:
with open('trans_noStiffened_area.csv', 'r') as f:
  content = f.readlines()
  for ii in range(10):
  	if ii > 5:
  		f_A_zero.append(6)
  		continue
  	x, y = content[ii].split(',')
  	f_A_zero.append(float(y.strip()))

trans_points = np.linspace(0.00, 0.25, num=11)
trans_points_5 = np.linspace(0.0, 0.25, num=6)

plt.figure(figsize=(10,4))
plt.subplot(1,2,2)
plt.plot(trans_points, f_AR_inf, 'k-', label='PINN (10)')
plt.plot(trans_points_5, f_AR_5_inf, 'k--', label='PINN (5)')
plt.plot(trans_points, f_AR_best, 'k-.', label='Stiffened FEM')
plt.plot(trans_points, f_AR_zero, 'k:', label='Classical FEM')
plt.ylim([0, 5])
plt.xlim([0, max(trans_points)])
plt.legend()
plt.ylabel(r"$| f_{AR}^{\infty}|$")
plt.xlabel(r"$\Delta \theta$")
plt.grid()
plt.rcParams['font.family'] = 'serif'
plt.title('Shape Change')

plt.subplot(1,2,1)
plt.plot(trans_points, f_A_inf, 'k-', label='PINN (10)')
plt.plot(trans_points_5, f_A_5_inf,'k--', label='PINN (5)')
plt.plot(trans_points, f_A_best, 'k-.', label='Stiffened FEM')
plt.plot(trans_points, f_A_zero, 'k:', label='Classical FEM')
plt.ylim([0, 5])
plt.xlim([0, max(trans_points)])
plt.legend()
plt.ylabel(r"$| f_{A}^{\infty}|$")
plt.xlabel(r"$\Delta \theta$")
plt.grid()
plt.title('Area Change')

# plt.savefig('rotation_metric_bw.pdf')
# plt.savefig('translation_metric_bw.pdf')

plt.show()