import numpy as np
from pyzonotope import MatrixZonotope
from scipy.spatial import ConvexHull, HalfspaceIntersection
import cvxpy as cp
from typing import Tuple

dim_x = 2
dim_u = 1
C = np.array([
    [0.5, 0.2, 0],
    [-0.1, 0.6, 0.5]
]).flatten()

dim_n = dim_x * (dim_x + dim_u)

center = np.array([[-1], [1]])
center = np.vstack((center, center))
A = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1]
])
b = center -np.ones((4,1))

def build_hypercube(center: np.ndarray, half_side_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds an hypercube around a center point. Returns [A,b], the halfspaces
    definining the hypercube Ax+b<=0

    :param center: center of the hypercube
    :param half_side_length: half the length of one side
    :return A: A contains (2*n rows, n cols), where n is the dimensionality of the center
    :return b: b contains 2n rows
    """
    assert isinstance(center, np.ndarray) or isinstance(center, list), 'Center is not an array'
    if isinstance(center, list):
        center = np.array(center)
    if len(center.shape) == 1:
        center = center[:, None]
    dim_n = center.shape[0]
    b = -half_side_length * np.ones((2 * dim_n, 1))
    A = np.vstack([np.eye(dim_n), -np.eye(dim_n)])

    _center = np.vstack((-center, center))

    return A, b + _center


def feasible_point(A: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes a feasible point that belongs to the intersection of the halfspaces defined in (A,B)
    Ax+b <=0
    """
    dim = A.shape[1]
    y = cp.Variable()
    x = cp.Variable((dim,1))

    constraint = [A @ x + y * cp.norm(A[i,:]) <= -b for i in range(A.shape[0])]
    problem = cp.Problem(cp.Maximize(y), constraint)
    res = problem.solve()
    return res, x.value.flatten()


#A, b = np.vstack((A1, A2)), np.vstack((b1, b2))

A,b = build_hypercube([3,1], 1)
_, interior_point = feasible_point(A,b)
import pdb
pdb.set_trace()

h_intersection = HalfspaceIntersection(np.hstack((A, b)), interior_point.flatten())
print(interior_point)
print(h_intersection.intersections)
# G = np.zeros((3, dim_x, dim_x+dim_u))

# G[0,:] = np.array([
#     [0.042, 0, 0],
#     [0.072, 0.03, 0]
# ])

# G[1,:] = np.array([
#     [0.015, 0.019, 0],
#     [0.009, 0.035, 0]
# ])

# G[2,:] = np.array([
#     [0, 0, 0.0397],
#     [0, 0, 0.059]
# ])

# Theta0 = MatrixZonotope(C, G)

# TrueAB = C + 0.8 * G[0] + 0.2 * G[1] -0.5 * G[2]
# A, B = TrueAB[:, :dim_x], TrueAB[:, dim_x:]

# import pdb
# pdb.set_trace()
# cvxHull =Theta0.zonotope.convex_hull
