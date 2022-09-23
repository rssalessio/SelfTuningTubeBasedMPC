
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_joint_spectral_radius, compute_contractive_polytope, build_hypercube
import cvxpy as cp
from hyper_rectangle import HyperRectangle
from scipy import linalg
import time
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
F = 5e-1*np.eye(dim_x)
G = 5e-1*np.ones((dim_x, dim_u))
n = 1
T = compute_contractive_polytope(n, lmbd, B, F, G, K, vertices_parameter_set)


std_w = 1e-2
W = np.hstack(build_hypercube(np.zeros(dim_x), 3 *  std_w))

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
        matrices = [np.stack([h[j][i].value for i in range(dalpha)]) for j in range(num_vertices)]
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
H, _ = problem_H(vertices_parameter_set)

# problem_H(vertices_parameter_set)
# print(time.time() - start)
# start = time.time()
# problem_H(vertices_parameter_set)
# print(time.time() - start)
# # problem_H(vertices_parameter_set)
# problem_H(vertices_parameter_set)
# print(time.time() - start)


def MPC_problem(
        K: np.ndarray,
        Hc: np.ndarray,
        B: np.ndarray,
        T: np.ndarray,
        G: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        num_vertices: int) -> Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]],
            Tuple[np.ndarray, np.ndarray, float, float]
            ]:


    dim_x = Q.shape[0]
    dim_u = R.shape[0]
    v = cp.Variable((horizon, dim_u))
    alpha = cp.Variable((horizon+1, T.shape[0]))
    x = cp.Variable((horizon+1, dim_x))

    A = cp.Parameter((dim_x, dim_x), name='A')
    B = cp.Parameter((dim_x, dim_u), name='B')
    P_sqrt = cp.Parameter((dim_x, dim_x), name='P_sqrt', PSD=True)
    x0 = cp.Parameter((dim_x), name='x0')
    wbar = cp.Parameter((T.shape[0]), name='wbar')

    matrices_H = [cp.Parameter((T.shape[0], T.shape[0]), name=f'H_{i}') for i in range(num_vertices)]

    constraints = [
        x[0] == x0,
        x[1:].T == (A + B @ K) @ x[:-1].T + B @ v.T,
        Hc @ alpha[:-1].T + G @ v.T <= 1,
        Hc @ alpha[-1] <= 1,
        T @ x0 <= alpha[0]
    ]
    for i in range(num_vertices):
        constraints.append(
            matrices_H[i] @ alpha[-1] + wbar <= alpha[-1],
        )
        for k in range(horizon):
            constraints.append(matrices_H[i] @ alpha[k] + T @ B @ v[k] + wbar <= alpha[k+1])

    loss = cp.sum_squares(P_sqrt @ x[-1])
    for i in range(horizon):
        loss += cp.quad_form(x[i], Q) + cp.quad_form(v[i], R)

    problem = cp.Problem(cp.Minimize(loss), constraints)


    def solve_problem(
            A: np.ndarray,
            B: np.ndarray,
            P: np.ndarray,
            x0: np.ndarray,
            wbar: np.ndarray,
            H_matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        problem.param_dict['A'].value = A
        problem.param_dict['B'].value = B
        problem.param_dict['P_sqrt'].value = linalg.sqrtm(P)
        problem.param_dict['x0'].value = x0
        problem.param_dict['wbar'].value = wbar

        for i in range(num_vertices):
            problem.param_dict[f'H_{i}'].value = H_matrices[i]
        
        start = time.time()
        res = problem.solve(solver=cp.MOSEK, warm_start=True, enforce_dpp=True, verbose=True)


        elapsed_time = time.time() - start
        
        return v.value, alpha.value, res, elapsed_time

    return solve_problem

def solve_lyapunov(A: np.ndarray, B: np.ndarray, K: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    return linalg.solve_discrete_lyapunov(A + B @ K, Q + K.T @ R @ K)

def compute_wbar(W_hs: np.ndarray, T: np.ndarray):
    wbar = np.zeros((T.shape[0]))
    w = cp.Variable(T.shape[1])
    Tval = cp.Parameter((T.shape[1]))

    problem = cp.Problem(cp.Maximize(Tval @ w), [W_hs[:,:-1] @ w + W_hs[:,-1] <= 0])

    for i in range(T.shape[0]):
        Tval.value = T[i]
        wbar[i] = problem.solve(warm_start=True)

    return wbar



solve_mpc_problem = MPC_problem(K, Hc, B, T, G, np.eye(dim_x), np.eye(dim_u), 10, vertices_parameter_set.shape[0])


wbar = compute_wbar(W, T)


P = solve_lyapunov(A, B, K, np.eye(dim_x), np.eye(dim_u))
x0 = np.random.standard_normal(size=(dim_x))

v, alpha, res, time_elapsed = solve_mpc_problem(A, B, P, x0, wbar, H)
print(v)
print(alpha)
print(res)
print(time_elapsed)
# start = time.time()
# problem_H(vertices_parameter_set)
# print(time.time() - start)
# start = time.time()
# problem_H(vertices_parameter_set)
# print(time.time() - start)
# # problem = compute_H_Hc(vertices_parameter_set, T, K, F, G)
# # problem = set_vertices_value(problem, vertices_parameter_set, dim_x, dim_u)
# # problem.solve(verbose=True, solver=cp.MOSEK, warm_start=False)