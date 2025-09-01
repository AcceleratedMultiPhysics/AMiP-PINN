import os
import sys
import time
import torch
import random
import scipy.io
import numpy as np
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import OrderedDict

sys.path.append("../utils")
from meshReadGmsh import readMesh
from bnsPlotFields import bns_plot_fields, bns_plot_q_fields

import os
import sys


PI = np.pi

xmin = -PI
xmax = PI
ymin = -PI
ymax = PI

tstart = 0
tfinal = 10.0


# Flow Parameters
# Re = 10
# u0 = 0.1581
# p0 = 0.05
RT = 100
# c  = 17.3205
# L  = 1
# Ma = u0/c


SQRT2 = np.sqrt(2)
SCALE = 1e4

sqrtRT = 10

nu  = 0.01
tau = nu / RT

FF  = True      # Fourier feature embeddings

save_folder = './outputs/tgv_FF/'
print(save_folder)
if os.path.exists(save_folder) == False:
    os.mkdir(save_folder)

# Learning Parameters
Nepochs = 2 * 10000

mesh_file = 'taylorGreen.msh'

mpl.rcParams.update(mpl.rcParamsDefault)
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['figure.max_open_warning'] = 4

if torch.cuda.is_available():
    """ Cuda support """
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not available')
    device = torch.device('cpu')

def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(23)
torch.cuda.empty_cache()

def tonp(tensor):
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def grad(u, x):
    """ Get grad """
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient