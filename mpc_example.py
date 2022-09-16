
import numpy as np
import polytope
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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

parameter_set_prev = np.hstack((A_hs, b_hs))
prev_vertices = half_space_intersection.intersections

for t in range(N):
    U[:, t] = std_u * np.random.normal(size=(dim_u))
    X[:, t+1] = A @ X[:, t] +  B @ U[:, t] + std_w * np.random.normal(size=(dim_x))

    # LS Estimate
    if t > 10:
        _X = X[:, :t+2]
        _U = U[:, :t+1]
        data = np.vstack((_X[:, :-1], _U))
        theta_t = _X[:, 1:] @ np.linalg.pinv(data)


        eps_t = 1 * np.log(t + 2) / (t + 1)

        delta_t = np.hstack(build_hypercube(theta_t.flatten(), 2 * eps_t))
        parameter_set_t = np.vstack((parameter_set_prev, delta_t))

        _, interior_point_t = feasible_point(parameter_set_t[:, :-1], parameter_set_t[:, -1:])
        

        half_space_intersection = HalfspaceIntersection(parameter_set_t, interior_point_t)

        _, interior_point_delta_t = feasible_point(delta_t[:, :-1], delta_t[:, -1:])
        hypercube_delta = HalfspaceIntersection(delta_t, interior_point_delta_t)

        #print(hypercube_delta.intersections)

        if not np.all(np.isclose(prev_vertices - half_space_intersection.intersections ,0)):
            # The delta_t is not bigger than the previous set
            parameter_set_prev = parameter_set_t

        cvx_hull = ConvexHull(half_space_intersection.intersections)

        Vol[t] = cvx_hull.volume

        

        is_inside = np.all(parameter_set_prev[:, :-1] @ C + parameter_set_prev[:, -1] <= 0)

        print(f'[Iteration {t}] Center: {interior_point_t} - number of vertices {half_space_intersection.intersections.shape[0]} - Volume: {cvx_hull.volume} - Point inside: {is_inside}')

plt.plot(Vol[1:])
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('Volume')
plt.title('Volume of parameter set')
plt.yscale('log')
plt.show()