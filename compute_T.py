
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_spectral_radius
import cvxpy as cp
from hyper_rectangle import HyperRectangle
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
A = np.array([
    [0.6, 0.2],
    [-0.1, 0.4]
])

B = np.array([
    [1],
    [0.5]
])


dim_x, dim_u = B.shape
C = np.hstack((A, B)).flatten()

radius = 3e-1

parameter_set = HyperRectangle.build_hypercube(C, radius)
vertices_parameter_set = parameter_set.vertices


# Compute stabilizing K
K = compute_stabilizing_K(vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')


def compute_T(F: np.ndarray, G: np.ndarray, K: np.ndarray, vertices: np.ndarray):
    dim_u, dim_x = K.shape
    num_vertices = vertices.shape[0]

    vertices = vertices.reshape(num_vertices, dim_x, dim_x+dim_u)

    P = cp.Variable((dim_x, dim_x), symmetric=True)

    constraints= [P >> 0]
    for i in range(num_vertices):
        A, B = vertices[i][:, :dim_x], vertices[i][:, dim_x:]
        phi = A+B@K
        constraints.append(P - phi.T @ P @ phi >> 0)

    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve(solver=cp.MOSEK)

    if P.value is None:
        raise Exception('Could not compute P!')

    P = P.value
    print(np.linalg.det(P))
    nc = F.shape[0]
    H = cp.Variable((nc, nc), symmetric=True)

    constraints = [cp.bmat([[H, F+G@K], [(F+G@K).T, P]]) >> 0]
    constraints.extend([H[i,i]<=1 for i in range(nc)])
    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve(solver=cp.MOSEK)

    if H.value is None:
        raise Exception("could not compute H!")

    return H.value


print(compute_spectral_radius(vertices_parameter_set, dim_x, dim_u, K))
F = 1e-1*np.eye(dim_x)
G = 1e-1*np.ones((dim_x, dim_u))
print(compute_T(F, G, K, vertices_parameter_set))
