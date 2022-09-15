import numpy as np
from pyzonotope import MatrixZonotope
from scipy.spatial import ConvexHull


dim_x = 2
dim_u = 1
C = np.array([
    [0.5, 0.2, 0],
    [-0.1, 0.6, 0.5]
])

G = np.zeros((3, dim_x, dim_x+dim_u))

G[0,:] = np.array([
    [0.042, 0, 0],
    [0.072, 0.03, 0]
])

G[1,:] = np.array([
    [0.015, 0.019, 0],
    [0.009, 0.035, 0]
])

G[2,:] = np.array([
    [0, 0, 0.0397],
    [0, 0, 0.059]
])

Theta0 = MatrixZonotope(C, G)

TrueAB = C + 0.8 * G[0] + 0.2 * G[1] -0.5 * G[2]
A, B = TrueAB[:, :dim_x], TrueAB[:, dim_x:]

import pdb
pdb.set_trace()
cvxHull =Theta0.zonotope.convex_hull
