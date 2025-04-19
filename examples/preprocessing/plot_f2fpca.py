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
from skfda.representation.grid import FDataGrid

#data_matrix = [[[1, 0.3], [2, 0.4]], [[2, 0.5], [3, 0.6]]]
data_matrix = np.zeros((200, 100, 10))
#grid_points = [2, 4]
grid_points = np.linspace(0, 100, 100)
fd = FDataGrid(data_matrix, grid_points)
print(fd.dim_domain, fd.dim_codomain)