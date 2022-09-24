
from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_joint_spectral_radius, compute_contractive_polytope, \
    build_hypercube, compute_Hc, get_H_problem, MPC_problem, compute_wbar, solve_lyapunov
import cvxpy as cp
from hyper_rectangle import HyperRectangle
from scipy import linalg
import matplotlib.pyplot as plt
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

radius = 7e-2

parameter_set = HyperRectangle.build_hypercube(C + 0.01*np.random.uniform(low=-1, high=1, size=((dim_x*2))), radius)

assert parameter_set.contains(A.flatten()), 'True matrix not contained in the parameter set'
vertices_parameter_set = parameter_set.vertices


# Compute stabilizing K
K = compute_stabilizing_K(B, vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')


rho, lmbd = compute_joint_spectral_radius(B, vertices_parameter_set, dim_x, dim_u, K)
F = np.array([ [0, -1/1.1], [0, 0]]) #[-1/(0.5), 0],
G = 2*np.array([[0], [1]]) #[0],
n = 1
T = compute_contractive_polytope(n, lmbd, B, F, G, K, vertices_parameter_set)


std_w = 1e-2
horizon = 10
W = np.hstack(build_hypercube(np.zeros(dim_x), 3 *  std_w))

print(f'Joint spectral radius: {rho} - lambda: {lmbd}')




Hc = compute_Hc(T, K, F, G)
problem_H = get_H_problem(B, vertices_parameter_set, T, K, F, G)
H, _ = problem_H(vertices_parameter_set)
solve_mpc_problem = MPC_problem(K, Hc, B, T, G, np.eye(dim_x), np.eye(dim_u), horizon, vertices_parameter_set.shape[0])

wbar = compute_wbar(W, T)




TOTAL_HORIZON = 500
x = np.zeros((TOTAL_HORIZON+1, dim_x))
u = np.zeros((TOTAL_HORIZON, dim_u))
epsilon = np.zeros(TOTAL_HORIZON)
volume = np.zeros(TOTAL_HORIZON)
const_eps = 40
delta=1e-3
const_eps2 = 1
N_LS = 2

x[0] = np.array([6,3])
A_t = parameter_set.center.reshape((dim_x, dim_x))

for t in range(TOTAL_HORIZON):
    volume[t] = parameter_set.volume
    print(f'[({t})] x_t: {x[t]} - A_t {A_t} - Contains A: {parameter_set.contains(A.flatten())} - Volume: {volume[t]} - {np.linalg.norm(A_t - A, ord="fro")}')
    

    P_t = solve_lyapunov(A_t, B, K, np.eye(dim_x), np.eye(dim_u))

    v, alpha, res, time_elapsed = solve_mpc_problem(A_t, B, P_t, x[t], wbar, H)
    if v is None:
        raise Exception('Problem unfeasible')


    u[t] = K@x[t] + v[0] + std_w * np.sqrt(dim_x) * np.random.standard_normal()
    x[t+1] = A  @ x[t] + B @ u[t] + std_w * np.random.standard_normal(dim_x)

    epsilon[t] = const_eps * (std_w ** 2) * (np.log(np.e/delta) + (dim_x+dim_u) * np.log(const_eps2 * std_w * dim_x * dim_u + (t + 1))) / (t + 1) ** 0.6


    if t > N_LS:
        #import pdb
        #pdb.set_trace()
        # LS Estimate
        _X = x[:t+2, :].T
        _U = u[:t+1, :].T
        #ata = np.vstack((_X[:, :-1], _U))
        #theta_t = (_X[:, 1:] @ np.linalg.pinv(data)).flatten()

        theta_t = ((_X[:, 1:] - B @ _U) @ np.linalg.pinv(_X[:,:-1])).flatten()


        delta_t = HyperRectangle.build_hypercube(theta_t, 2 * epsilon[t])
        parameter_set = parameter_set.intersect(delta_t)
        A_t = theta_t.reshape((dim_x, dim_x))

        vertices_parameter_set = parameter_set.vertices
        H, _ = problem_H(vertices_parameter_set)

plt.plot(volume)
plt.yscale('log')
plt.grid()
plt.show()

plt.plot(x[:,0], x[:, 1])
plt.grid()
plt.show()