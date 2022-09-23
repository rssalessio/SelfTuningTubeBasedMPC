
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point, build_aligned_hypercube
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from conservative_intersection import ConservativeIntersection
from under_approximate_intersection import UnderApproximateIntersection
from over_approximate_intersection import OverApproximateIntersection
from classical_intersection import ClassicalIntersection
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

# A = np.array([
#     [0.5]
# ])

# B = np.array([
#     [1]
# ])

dim_x, dim_u = B.shape
C = np.hstack((A, B)).flatten()

radius = 3e-1
std_u = 1e-1
std_w = std_u
delta = 1e-2
const_eps = 300
const_eps2 = 100

parameter_set = HyperRectangle.build_hypercube(C, radius)
vertices_parameter_set = parameter_set.vertices


# Compute stabilizing K
K = compute_stabilizing_K(vertices_parameter_set, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')

N = 2000
N_LS = 10
X = np.zeros((dim_x, N+1))
U = np.zeros((dim_u, N))



volume_intersection = np.zeros((N+1))
volume_theta_hypercube = np.zeros((N+1))
error = np.zeros((N))
epsilon = np.zeros((N))

X[:,0] = np.random.normal(size=(dim_x))


volume_intersection[0] = parameter_set.volume
volume_theta_hypercube[0] = 0

for t in range(N):
    U[:, t] =  std_u * np.random.normal(size=(dim_u)) * np.sqrt(np.sqrt(dim_x / (t + 1)))
    X[:, t+1] = A @ X[:, t] +  B @ U[:, t] + std_w * np.random.normal(size=(dim_x))

    epsilon[t] = const_eps * (std_w ** 2) * (np.log(np.e/delta) + (dim_x+dim_u) * np.log(const_eps2 * std_w * dim_x * dim_u + (t + 1))) / (t + 1)
    #epsilon[t] = min(epsilon[t], radius * 2)
    
    volume_intersection[t+1] = volume_intersection[t]
    volume_theta_hypercube[t+1] = volume_theta_hypercube[t]

    if t > N_LS:
        # LS Estimate
        _X = X[:, :t+2]
        _U = U[:, :t+1]
        data = np.vstack((_X[:, :-1], _U))
        theta_t = (_X[:, 1:] @ np.linalg.pinv(data)).flatten()

        error[t] = np.linalg.norm(theta_t - C)


        delta_t = HyperRectangle.build_hypercube(theta_t, 2 * epsilon[t])
        parameter_set = parameter_set.intersect(delta_t)

        volume_intersection[t+1] = parameter_set.volume
        volume_theta_hypercube[t+1] = delta_t.volume

        theta_reshaped = theta_t.reshape((dim_x, dim_x+dim_u))
        #error[t] = max(np.linalg.norm(A - theta_reshaped[:, :dim_x], ord=2), np.linalg.norm(B - theta_reshaped[:, dim_x:], ord=2))

        if t == N_LS + 1:
            for i in range(t+1):
                volume_theta_hypercube[i] = volume_theta_hypercube[t+1]

        print(f'[{t}] {volume_intersection[t+1]} - {parameter_set.contains(C)} - {theta_reshaped}')

plt.show()
fig, ax = plt.subplots(1,2)
ax[0].plot(epsilon[N_LS:], label=r'\epsilon_t')
ax[0].plot(error[N_LS:], label=r'\|\theta-\theta_t\|')
ax[0].grid()
ax[0].legend()
#ax[0].set_yscale('log')

ax[1].plot(volume_intersection[N_LS:] / volume_intersection[0], label='Volume  intersection')
ax[1].plot(volume_theta_hypercube[N_LS:] / volume_theta_hypercube[0], label='Volume hypercube around theta')
ax[1].grid()
#ax[1].set_yscale('log')
ax[1].legend()
ax[1].set_title('Decrease with respect to initial value')
plt.title(f"C = {const_eps} - delta = {delta}")
plt.show()
# plt.plot(Vol[1:])
# plt.grid()
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('Volume')
# plt.title('Volume of parameter set')
# plt.yscale('log')
# plt.show()