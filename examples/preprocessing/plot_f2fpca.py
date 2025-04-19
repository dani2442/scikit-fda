"""
Functional to Functional Principal Component Analysis
====================================================

Explores the way to do functional to functional principal component analysis.
"""

# Author: Daniel López-Montero
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

import skfda
from skfda.misc.metrics import LpDistance
from skfda.representation.grid import FDataGrid
from skfda.preprocessing.dim_reduction import F2FPCA, FF2FPCA

N = 200
n = 100
p = 10
data_matrix = np.random.random((n, N, p))
grid_points = np.linspace(0,1, N)
fd = FDataGrid(data_matrix, grid_points)
f2fpca = F2FPCA(1)
ff2fpca = FF2FPCA(1)

f2fpca = f2fpca.fit(fd)
Z1 = f2fpca.transform(fd)
Y1 = f2fpca.inverse_transform(Z1)

ff2fpca = ff2fpca.fit(fd)
Z2 = ff2fpca.transform(fd)
Y2 = ff2fpca.inverse_transform(Z2)

d = LpDistance(2)
print("Reconstruction error of F2FPCA: ", np.mean(d(fd,Y1)))
print("Reconstruction error of FF2FPCA: ", np.mean(d(fd,Y2)))