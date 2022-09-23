
import numpy as np
from scipy.spatial import HalfspaceIntersection
from utils import build_hypercube, compute_stabilizing_K

dim_x = 2
dim_u = 1
center = np.array([
    [0.9, 0.2, .1],
    [-0.1, 0.6, 0.5]
])

sys_A = center[:, :dim_x]
sys_B = center[:, dim_x:]

C = center.flatten()
radius = 1e-1


A, b = build_hypercube(C, radius)

half_space_intersection = HalfspaceIntersection(np.hstack((A, b)), C)

# Compute vertices
vertices_half_space = half_space_intersection.intersections

# Compute stabilizing K
K = compute_stabilizing_K(vertices_half_space, dim_x, dim_u)

# Print stabilizing controller, none if it does not exist
print(f"Stabilizing K: {K}")