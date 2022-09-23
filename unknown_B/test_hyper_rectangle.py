
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point, build_aligned_hypercube
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from conservative_intersection import ConservativeIntersection
from under_approximate_intersection import UnderApproximateIntersection
from over_approximate_intersection import OverApproximateIntersection
from classical_intersection import ClassicalIntersection
from hyper_rectangle import HyperRectangle
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
A = np.array([
    [0.6, 0.2],
    [-0.1, 0.4]
])

B = np.array([
    [1],
    [0.5]
])

# A = np.array([
#     [0.5]
# ])

# B = np.array([
#     [1]
# ])

dim_x, dim_u = B.shape
C = np.hstack((A, B)).flatten()

radius = 3e-1
std_u = 1e-1
std_w = 1e-1
delta = 1e-2
const_eps = 100
A_hs, b_hs = build_hypercube(C, radius)

half_space_intersection = HalfspaceIntersection(np.hstack((A_hs, b_hs)), C, incremental=True)

# Compute vertices
vertices_half_space = half_space_intersection.intersections


initial_set = HyperRectangle.build_hypercube(C, radius)
vertices_initial_set = initial_set.vertices

idxs = []

for i in range(vertices_initial_set.shape[0]):
    for j in range(vertices_half_space.shape[0]):
        if np.all(np.isclose(vertices_half_space[j], vertices_initial_set[i])):
            idxs.append((i,j))

print(len(idxs))