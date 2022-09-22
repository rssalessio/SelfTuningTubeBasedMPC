
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_joint_spectral_radius, compute_contractive_polytope
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


rho, lmbd = compute_joint_spectral_radius(vertices_parameter_set, dim_x, dim_u, K)
F = 1e-1*np.eye(dim_x)
G = 1e-1*np.ones((dim_x, dim_u))
n = 1
T = compute_contractive_polytope(n, lmbd, F, G, K, vertices_parameter_set)

# def compute_H_Hc(vertices: np.ndarray, T: np.ndarray, K: np.ndarray, F: np.ndarray, G: np.ndarray) -> cp.Problem:
#     """
#     Note that the number of vertices is always fixed if we use hypercubes. This makes the problem more computationally efficient
#     so the number of vertices is not time varying
#     """
#     num_vertices = vertices.shape[0]
#     dalpha, dx = T.shape
#     dc = F.shape[0]
#     dim_u, dim_x = K.shape

#     vertices = vertices.reshape(num_vertices, dim_x, dim_x+dim_u)

#     matrices_H = [cp.Variable((dalpha, dalpha), nonneg=True) for j in range(num_vertices)]
#     Hc = cp.Variable((dc, dalpha), nonneg=True)
#     VParams = [cp.Parameter((vertices.shape[1:]), name=f'vertex_{j}') for j in range(num_vertices)]


#     objective = cp.sum(Hc)
#     constraints = [Hc @ T == F +  G @ K]

#     # # Constraints H
#     # for j in range(num_vertices):
#     #     Av, Bv = VParams[j][:, :dim_x], VParams[j][:, dim_x:]
#     #     constraints.append(matrices_H[j] @ T == T @ (Av + Bv @ K))
#     #     objective += cp.sum(matrices_H[j])
    
#     problem = cp.Problem(cp.Minimize(objective), constraints)
#     return problem

def compute_Hc(T: np.ndarray, K: np.ndarray, F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Compute matrix Hc
    """
    dalpha, dx = T.shape
    dc = F.shape[0]

    Hc = [cp.Variable((dalpha), nonneg=True) for i in range(dc)]
    FGK = F + G @ K
    for i in range(dc):
        problem = cp.Problem(cp.Minimize(cp.sum(Hc[i])), [Hc[i] @ T == FGK[i]])
        problem.solve()
    

    return np.stack([Hc[i].value for i in range(dc)])

def get_H_problem(vertices: np.ndarray, T: np.ndarray, K: np.ndarray, F: np.ndarray, G: np.ndarray) -> cp.Problem:
    """
    Note that the number of vertices is always fixed if we use hypercubes. This makes the problem more computationally efficient
    so the number of vertices is not time varying
    """
    num_vertices = vertices.shape[0]
    dalpha, dx = T.shape
    dc = F.shape[0]
    dim_u, dim_x = K.shape

    vertices = vertices.reshape(num_vertices, dim_x, dim_x+dim_u)

    h = [[cp.Variable((dalpha), nonneg=True) for i in range(dalpha)] for j in range(num_vertices)]
    VParams = [cp.Parameter((vertices.shape[1:]), name=f'vertex_{j}') for j in range(num_vertices)]

    problems: List[List[cp.Problem]] = []


    # Constraints H
    for j in range(num_vertices):
        Av, Bv = VParams[j][:, :dim_x], VParams[j][:, dim_x:]
        phi = Av + Bv @ K
        problems.append([])
        for i in range(dalpha):
            problem = cp.Problem(cp.Minimize(cp.sum(h[j][i])), [h[j][i] @ T == T[i] @ phi])
            problems[-1].append(problem)
    

    def solve_problem(vertices: np.ndarray) -> np.ndarray:
        vertices = vertices.reshape(num_vertices, dim_x, dim_x+dim_u)
        for i in range(num_vertices):
            VParams[i].value = vertices[j]

        res = [[problems[j][i].solve(verbose=False, warm_start=True) for i in range(dalpha)] for j in range(num_vertices)]
        matrices = [np.hstack(h[j]) for j in range(num_vertices)]
        return matrices, res

    return solve_problem

def set_vertices_value(problem: cp.Problem, vertices: np.ndarray, dim_x: int, dim_u: int) -> cp.Problem:
    vertices = vertices.reshape(vertices.shape[0], dim_x, dim_x+dim_u)
    for j in range(vertices.shape[0]):
        problem.param_dict[f'vertex_{j}'].value = vertices[j]
    return problem

Hc = compute_Hc(T, K, F, G)

problem_H = get_H_problem(vertices_parameter_set, T, K, F, G)

print(problem_H(vertices_parameter_set))
print(problem_H(vertices_parameter_set))
# problem = compute_H_Hc(vertices_parameter_set, T, K, F, G)
# problem = set_vertices_value(problem, vertices_parameter_set, dim_x, dim_u)
# problem.solve(verbose=True, solver=cp.MOSEK, warm_start=False)