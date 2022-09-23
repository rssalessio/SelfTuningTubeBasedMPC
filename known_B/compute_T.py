
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
C = A.flatten()

radius = 3e-1

parameter_set = HyperRectangle.build_hypercube(C, radius)
vertices_parameter_set = parameter_set.vertices


# Compute stabilizing K
K = compute_stabilizing_K(B, vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')


rho, lmbd = compute_joint_spectral_radius(B, vertices_parameter_set, dim_x, dim_u, K)
F = 1e-1*np.eye(dim_x)
G = 1e-1*np.ones((dim_x, dim_u))
n = 1
T = compute_contractive_polytope(n, lmbd, B, F, G, K, vertices_parameter_set)

print(f'Joint spectral radius: {rho} - lambda: {lmbd}')
# # def compute_H_Hc(vertices: np.ndarray, T: np.ndarray, K: np.ndarray, F: np.ndarray, G: np.ndarray) -> cp.Problem:
# #     """
# #     Note that the number of vertices is always fixed if we use hypercubes. This makes the problem more computationally efficient
# #     so the number of vertices is not time varying
# #     """
# #     num_vertices = vertices.shape[0]
# #     dalpha, dx = T.shape
# #     dc = F.shape[0]
# #     dim_u, dim_x = K.shape

# #     vertices = vertices.reshape(num_vertices, dim_x, dim_x+dim_u)

# #     matrices_H = [cp.Variable((dalpha, dalpha), nonneg=True) for j in range(num_vertices)]
# #     Hc = cp.Variable((dc, dalpha), nonneg=True)
# #     VParams = [cp.Parameter((vertices.shape[1:]), name=f'vertex_{j}') for j in range(num_vertices)]


# #     objective = cp.sum(Hc)
# #     constraints = [Hc @ T == F +  G @ K]

# #     # # Constraints H
# #     # for j in range(num_vertices):
# #     #     Av, Bv = VParams[j][:, :dim_x], VParams[j][:, dim_x:]
# #     #     constraints.append(matrices_H[j] @ T == T @ (Av + Bv @ K))
# #     #     objective += cp.sum(matrices_H[j])
    
# #     problem = cp.Problem(cp.Minimize(objective), constraints)
# #     return problem

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

def get_H_problem(B: np.ndarray, vertices: np.ndarray, T: np.ndarray, K: np.ndarray, F: np.ndarray, G: np.ndarray) -> cp.Problem:
    """
    Note that the number of vertices is always fixed if we use hypercubes. This makes the problem more computationally efficient
    so the number of vertices is not time varying
    """
    num_vertices = vertices.shape[0]
    dalpha, dx = T.shape
    dc = F.shape[0]
    dim_u, dim_x = K.shape

    vertices = vertices.reshape(num_vertices, dim_x, dim_x)

    h = [[cp.Variable((dalpha), nonneg=True) for i in range(dalpha)] for j in range(num_vertices)]
    VParams = [cp.Parameter((vertices.shape[1:]), name=f'vertex_{j}') for j in range(num_vertices)]

    objective = 0
    constraints= []


    # Constraints H
    for j in range(num_vertices):
        Av = VParams[j]
        phi = Av + B @ K
        for i in range(dalpha):
            objective += cp.sum(h[j][i])
            constraints.append(h[j][i] @ T == T[i] @ phi)
        
    problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve_problem(vertices: np.ndarray) -> np.ndarray:
        vertices = vertices.reshape(vertices.shape[0], dim_x, dim_x)
        for j in range(vertices.shape[0]):
            problem.param_dict[f'vertex_{j}'].value = vertices[j]
        #problem = set_vertices_value(problem, vertices, dim_x, dim_u)
        # vertices = vertices.reshape(num_vertices, dim_x, dim_x)
        # for i in range(num_vertices):
        #     VParams[i].value = vertices[j]

        # res = [[problems[j][i].solve(verbose=False, warm_start=True) for i in range(dalpha)] for j in range(num_vertices)]
        res = problem.solve(warm_start=True)
        matrices = [np.hstack(h[j]) for j in range(num_vertices)]
        return matrices, res

    return solve_problem

# def set_vertices_value(problem: cp.Problem, vertices: np.ndarray, dim_x: int, dim_u: int) -> cp.Problem:
#     vertices = vertices.reshape(vertices.shape[0], dim_x, dim_x)
#     for j in range(vertices.shape[0]):
#         problem.param_dict[f'vertex_{j}'].value = vertices[j]
#     return problem

Hc = compute_Hc(T, K, F, G)

problem_H = get_H_problem(B, vertices_parameter_set, T, K, F, G)
#problem_H = set_vertices_value(problem_H, vertices_parameter_set, dim_x, dim_u)
import time
start = time.time()
problem_H(vertices_parameter_set)

# problem_H(vertices_parameter_set)
print(time.time() - start)
start = time.time()
problem_H(vertices_parameter_set)
print(time.time() - start)
# problem_H(vertices_parameter_set)
problem_H(vertices_parameter_set)
print(time.time() - start)


def MPC_problem(K: np.ndarray, Hc: np.ndarray, T: np.ndarray, G: np.ndarray, Q: np.ndarray, R: np.ndarray, dim_x: int, dim_u: int, horizon: int):
    v = cp.Variable((horizon, dim_u))
    alpha = cp.Variable((horizon, T.shape[0]))
    x = cp.Variable(horizon+1, dim_x)

    A = cp.Parameter((dim_x, dim_x), name='A')
    B = cp.Parameter((dim_x, dim_u), name='B')
    P = cp.Parameter((dim_x, dim_x), name='P', symmetric=True)
    x0 = cp.Parameter((dim_x), name='x0')

    constraints = [
        x[0] == x0,
        x[1:] == (A + B @ K) @ x[:-1] + B @ v,
        Hc @ alpha + G @ v <= 1,
        Hc @ alpha[-1] <= 1,
        T @ x0 <= alpha[0]
    ]

    loss = cp.quad_form(x[-1], P) + cp.sum([cp.quad_form(x[i], Q) + cp.quad_form(v[i], R) for i in range(horizon)])




# start = time.time()
# problem_H(vertices_parameter_set)
# print(time.time() - start)
# start = time.time()
# problem_H(vertices_parameter_set)
# print(time.time() - start)
# # problem = compute_H_Hc(vertices_parameter_set, T, K, F, G)
# # problem = set_vertices_value(problem, vertices_parameter_set, dim_x, dim_u)
# # problem.solve(verbose=True, solver=cp.MOSEK, warm_start=False)