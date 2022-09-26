
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_joint_spectral_radius, compute_contractive_polytope, \
    build_hypercube, compute_Hc, get_H_problem, MPC_problem, compute_wbar, solve_lyapunov
from hyper_rectangle import HyperRectangle
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

### CONSTANTS ###
A = np.array([
    [0.6, 0.2],
    [-0.1, 0.4]
])

B = np.array([
    [1],
    [0.5]
])

RADIUS_INITIAL_SET = 7e-2
NOISE_CENTER_INITIAL_SET = 1e-2
dim_x, dim_u = B.shape

STD_W = 1e-2
MPC_HORIZON = 10
TOTAL_HORIZON = 500


CONST_C = 40
DELTA = 1e-3
N_LS = 2

INITIAL_X0 = np.array([6,3])


## Constraints ##
# -1.1 <= x_2
# u<= 1/2
F = np.array([ [0, -1/1.1], [0, 0]]) #[-1/(0.5), 0],
G = 2*np.array([[0], [1]])
ORDER_CONTRACTIVE_POLYTOPE = 1


### Definition of Theta_0 ###
C = A.flatten()
noisy_center_initial_set = C + NOISE_CENTER_INITIAL_SET * np.random.uniform(low=-1, high=1, size=((dim_x*2)))
parameter_set = HyperRectangle.build_hypercube(noisy_center_initial_set, RADIUS_INITIAL_SET)

assert parameter_set.contains(A.flatten()), 'True matrix not contained in the parameter set'
vertices_parameter_set = parameter_set.vertices


### Compute stabilizing K ###
K = compute_stabilizing_K(B, vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')


### Compute T ###
rho, lmbd = compute_joint_spectral_radius(B, vertices_parameter_set, dim_x, dim_u, K)
T = compute_contractive_polytope(ORDER_CONTRACTIVE_POLYTOPE, lmbd, B, F, G, K, vertices_parameter_set)
print(f'Joint spectral radius: {rho} - lambda: {lmbd}')

### Compute cal{W}
W = np.hstack(build_hypercube(np.zeros(dim_x), 3 *  STD_W))


### Compute matrices Hc, H and define MPC problem
Hc = compute_Hc(T, K, F, G)
problem_H = get_H_problem(B, vertices_parameter_set, T, K, F, G)
H, _ = problem_H(vertices_parameter_set)
solve_mpc_problem = MPC_problem(K, Hc, B, T, G, np.eye(dim_x), np.eye(dim_u), MPC_HORIZON, vertices_parameter_set.shape[0])


# Compute wbar
wbar = compute_wbar(W, T)



### Solve MPC
x = np.zeros((TOTAL_HORIZON+1, dim_x))
u = np.zeros((TOTAL_HORIZON, dim_u))
epsilon = np.zeros(TOTAL_HORIZON)
volume = np.zeros(TOTAL_HORIZON)


x[0] = INITIAL_X0
A_t = parameter_set.center.reshape((dim_x, dim_x))

for t in range(TOTAL_HORIZON):
    volume[t] = parameter_set.volume
    print(f'[({t})] x_t: {x[t]} - A_t {A_t} - Contains A: {parameter_set.contains(A.flatten())} - Volume: {volume[t]} - {np.linalg.norm(A_t - A, ord="fro")}')
    
    P_t = solve_lyapunov(A_t, B, K, np.eye(dim_x), np.eye(dim_u))

    v, alpha, res, time_elapsed = solve_mpc_problem(A_t, B, P_t, x[t], wbar, H)
    if v is None:
        raise Exception('Problem unfeasible')


    u[t] = K@x[t] + v[0] + STD_W * np.sqrt(dim_x) * np.random.standard_normal()
    x[t+1] = A  @ x[t] + B @ u[t] + STD_W * np.random.standard_normal(dim_x)

    epsilon[t] = CONST_C * (STD_W ** 2) * (np.log(np.e / DELTA) + (dim_x+dim_u) * np.log(STD_W * dim_x * dim_u + (t + 1))) / (t + 1) ** 0.6


    if t > N_LS:
        # LS Estimate
        _X = x[:t+2, :].T
        _U = u[:t+1, :].T
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