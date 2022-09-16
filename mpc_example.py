
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point


A = np.array([
    [0.9, 0.2],
    [-0.1, 0.6]
])

B = np.array([
    [.1],
    [0.5]
])

dim_x, dim_u = B.shape
C = np.hstack((A, B)).flatten()

radius = 1e-1
std_u = 1e-1
std_w = 1e-1
A_hs, b_hs = build_hypercube(C, radius)

half_space_intersection = HalfspaceIntersection(np.hstack((A_hs, b_hs)), C, incremental=True)

# Compute vertices
vertices_half_space = half_space_intersection.intersections

# Compute stabilizing K
K = compute_stabilizing_K(vertices_half_space, dim_x, dim_u)


N = 100
X = np.zeros((dim_x, N+1))
U = np.zeros((dim_u, N))
Vol = np.zeros((N))
X[:,0] = np.random.normal(size=(dim_x))

A_hs_prev, b_hs_prev = A_hs, b_hs

for t in range(N):
    U[:, t] = std_u * np.random.normal(size=(dim_u))
    X[:, t+1] = A @ X[:, t] +  B @ U[:, t] + std_w * np.random.normal(size=(dim_x))

    # LS Estimate
    if t > 2:
        _X = X[:, :t+2]
        _U = U[:, :t+1]
        data = np.vstack((_X[:, :-1], _U))
        theta_t = _X[:, 1:] @ np.linalg.pinv(data)


        eps_t = 22 * np.log(t + 2) / (t + 1)

        A_hs_t, b_hs_t = build_hypercube(theta_t.flatten(), 2 * eps_t)
        A_intersection_t, b_intersection_t = np.vstack((A_hs_prev, A_hs_t)), np.vstack((b_hs_prev, b_hs_t))

        _, interior_point_t = feasible_point(A_intersection_t, b_intersection_t)


        half_space_intersection = HalfspaceIntersection(np.hstack((A_intersection_t, b_intersection_t)), interior_point_t)
        cvx_hull = ConvexHull(half_space_intersection.intersections)

        
        Vol[t] = cvx_hull.volume

        A_hs_prev, b_hs_prev = A_intersection_t, b_intersection_t

        is_inside = np.all(A_intersection_t @ C + b_intersection_t <= 0)

        print(f'[Iteration {t}] Center: {interior_point_t} - number of vertices {half_space_intersection.intersections.shape[0]} - Volume: {cvx_hull.volume} - Point inside: {is_inside}')

plt.plot(Vol[1:])
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('Volume')
plt.title('Volume of parameter set')
plt.yscale('log')
plt.show()