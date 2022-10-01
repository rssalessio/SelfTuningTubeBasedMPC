
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_stabilizing_K, compute_joint_spectral_radius, compute_contractive_polytope, \
    build_hypercube, compute_Hc, get_H_problem, MPC_problem, compute_wbar, solve_lyapunov, feasible_point
from hyper_rectangle import HyperRectangle
import plot_constants
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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
MPC_HORIZON = 6
TOTAL_HORIZON = 6


CONST_C = 40
DELTA = 1e-2
N_LS = 2

INITIAL_X0 = np.array([6,3])


## Constraints ##
# X2_MIN <= x_2, X1_MIN <= x1
# u<= U_MAX
X2_MIN = -1.1
X1_MIN = -0.15
U_MAX = 1/2
F = np.array([ [0, 1/X2_MIN], [1/X1_MIN, 0], [0, 0]])
G = np.array([[0], [0], [1/U_MAX]])
ORDER_CONTRACTIVE_POLYTOPE = 1


### Definition of Theta_0 ###
C = A.flatten()
noisy_center_initial_set = np.array([0.57, 0.17, -0.12, 0.42])
parameter_set = HyperRectangle.build_hypercube(noisy_center_initial_set, RADIUS_INITIAL_SET)

assert parameter_set.contains(A.flatten()), 'True matrix not contained in the parameter set'
vertices_parameter_set = parameter_set.vertices


### Compute stabilizing K ###
K = compute_stabilizing_K(B, vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')

print(f'Stabilizing K: {K}')

### Compute T ###
rho, lmbd = compute_joint_spectral_radius(B, vertices_parameter_set, dim_x, dim_u, K)
T = compute_contractive_polytope(ORDER_CONTRACTIVE_POLYTOPE, lmbd, B, F, G, K, vertices_parameter_set)
print(f'Joint spectral radius: {rho} - lambda: {lmbd}')

lmbd = 0.999
### Compute cal{W}
W = np.hstack(build_hypercube(np.zeros(dim_x), 3 *  (1+ np.linalg.norm(B)) * STD_W))


### Compute matrices Hc, H and define MPC problem
Hc = compute_Hc(T, K, F, G)
problem_H = get_H_problem(B, vertices_parameter_set, T, K, F, G)
H, _ = problem_H(vertices_parameter_set)
solve_mpc_problem = MPC_problem(K, Hc, B, T, G, np.eye(dim_x), np.eye(dim_u), MPC_HORIZON, vertices_parameter_set.shape[0])


# Compute wbar
wbar = compute_wbar(W, T)



### Solve MPC
x = np.zeros((TOTAL_HORIZON+1, dim_x))
x_noconstraints = np.zeros((TOTAL_HORIZON+1, dim_x))
u = np.zeros((TOTAL_HORIZON, dim_u))
epsilon = np.zeros(TOTAL_HORIZON)
volume = np.zeros(TOTAL_HORIZON)


x[0] = INITIAL_X0
x_noconstraints[0] = x[0]
A_t = parameter_set.center.reshape((dim_x, dim_x))

polygons = []

for t in range(TOTAL_HORIZON):
    volume[t] = parameter_set.volume
    print(f'[({t})] x_t: {x[t]} - A_t {A_t} - Contains A: {parameter_set.contains(A.flatten())} - Volume: {volume[t]} - {np.linalg.norm(A_t - A, ord="fro")}')
    
    P_t = solve_lyapunov(A_t, B, K, np.eye(dim_x), np.eye(dim_u))

    v, alpha, est_x, res, time_elapsed = solve_mpc_problem(A_t, B, P_t, x[t], wbar, H)
    if v is None:
        raise Exception('Problem unfeasible')


    u[t] = K@x[t] + v[0] + STD_W * np.sqrt(dim_x) * np.random.standard_normal()
    w_t = STD_W * np.random.standard_normal(dim_x)
    x[t+1] = A  @ x[t] + B @ u[t] + w_t
    x_noconstraints[t+1] =  (A + B @ K) @ x[t] + w_t


    # Compute tube
    alpha_1 = alpha[1]

    xfeas = est_x[1]

    hs_int = HalfspaceIntersection(np.hstack((T, -alpha_1[:, None] -1e-6)), xfeas)
    hs_int_vertices = hs_int.intersections

    keys = np.arctan2(hs_int_vertices[:, 1] - xfeas[1], hs_int_vertices[:,0] - xfeas[0])
    hs_int_vertices = hs_int_vertices[np.argsort(keys)]
    hs_int_vertices = np.vstack((hs_int_vertices, hs_int_vertices[0]))

    polygons.append(Polygon(hs_int_vertices))


    # LS
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




d = np.linspace(-3,10,5000)
xd,yd = np.meshgrid(d,d)
fig, ax = plt.subplots()
ax.imshow( ((yd<X2_MIN) | (xd < X1_MIN)) , 
                extent=(xd.min(),xd.max(),yd.min(),yd.max()),origin="lower", cmap="Greys", alpha = 0.4)
#ax.plot(x_noconstraints[:,0], x_noconstraints[:, 1], linestyle='dashed', marker='o', color='red', linewidth=0.7,label='No MPC')
ax.plot(x[:,0], x[:, 1],  linestyle='dotted', marker='x', color='black', linewidth=0.7,label='STT-MPC')

for polygon in polygons:
    collection = PatchCollection([polygon],  facecolor='lightsalmon', edgecolor='black', lw=1, alpha=0.5)
    
    ax.add_collection(collection)


ax.set_xlabel('$x_1$', horizontalalignment='right', x=.95)
ax.set_ylabel('$x_2$', horizontalalignment='right', y=.95)
ax.legend(fancybox=True, facecolor="whitesmoke")
ax.grid()

ax.set_xlim(-0.35, 6.2)
ax.set_ylim(-1.3, 3.2)

plt.savefig('control.pdf',bbox_inches='tight')