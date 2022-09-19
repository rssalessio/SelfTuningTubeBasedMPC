
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from conservative_intersection import ConservativeIntersection
from under_approximate_intersection import UnderApproximateIntersection
from over_approximate_intersection import OverApproximateIntersection
from classical_intersection import ClassicalIntersection
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
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
std_w = 1e-1
delta = 1e-2
const_eps = 100
A_hs, b_hs = build_hypercube(C, radius)

half_space_intersection = HalfspaceIntersection(np.hstack((A_hs, b_hs)), C, incremental=True)

# Compute vertices
vertices_half_space = half_space_intersection.intersections

# Compute stabilizing K
K = compute_stabilizing_K(vertices_half_space, dim_x, dim_u)

if K is None:
    raise Exception('K is not stabilizing')

N = 100
N_LS = 10
X = np.zeros((dim_x, N+1))
U = np.zeros((dim_u, N))
volume_conservative = np.zeros((N+1))
volume_underapproximate = np.zeros((N+1))
volume_overapproximate = np.zeros((N+1))
volume_classical = np.zeros((N+1))
volume_theta_hypercube = np.zeros((N))
error = np.zeros((N))
epsilon = np.zeros((N))

X[:,0] = np.random.normal(size=(dim_x))

parameter_set_prev = np.hstack((A_hs, b_hs))
prev_vertices = half_space_intersection.intersections
volume_conservative[0] = ConvexHull(half_space_intersection.intersections).volume
volume_underapproximate[0] = volume_conservative[0]
volume_overapproximate[0] = volume_conservative[0]
volume_classical[0] = volume_conservative[0]

conservative_intersection = ConservativeIntersection(A_hs, b_hs)
underapproximate_intersection = UnderApproximateIntersection(A_hs, b_hs)
overapproximate_intersection = OverApproximateIntersection(A_hs, b_hs)
classical_intersection = ClassicalIntersection(A_hs, b_hs)

if dim_x+dim_u==2:
    fig, ax = plt.subplots()
for t in range(N):
    U[:, t] = std_u * np.random.normal(size=(dim_u))
    X[:, t+1] = A @ X[:, t] +  B @ U[:, t] + std_w * np.random.normal(size=(dim_x))

    epsilon[t] = const_eps * (std_w ** 2) * (np.log(np.e/delta) + (dim_x+dim_u) * np.log(std_w * dim_x * dim_u + (t + 1))) / (t + 1)
    volume_underapproximate[t+1] = volume_underapproximate[t]
    volume_conservative[t+1] = volume_conservative[t]
    volume_overapproximate[t+1] = volume_overapproximate[t]
    volume_classical[t+1] = volume_classical[t]

    if t > N_LS:
        # LS Estimate
        _X = X[:, :t+2]
        _U = U[:, :t+1]
        data = np.vstack((_X[:, :-1], _U))
        theta_t = (_X[:, 1:] @ np.linalg.pinv(data)).flatten()

        error[t] = np.linalg.norm(theta_t - C)
 
        delta_t = np.hstack(build_hypercube(theta_t, 2 * epsilon[t]))



        _, interior_point_delta_t = feasible_point(delta_t[:, :-1], delta_t[:, -1:])
        hypercube_delta = HalfspaceIntersection(delta_t, interior_point_delta_t)
        cvx_hull_delta = ConvexHull(hypercube_delta.intersections)
        volume_theta_hypercube[t] = cvx_hull_delta.volume

        if dim_x+dim_u==2:
            if t%5 ==0:
                circle1 = plt.Circle(theta_t, 2* epsilon[t], color='r', fill=False)

                points = hypercube_delta.intersections
                ax.plot(points[cvx_hull_delta.vertices,0] , points[cvx_hull_delta.vertices,1], 'k')
                ax.add_patch(circle1)
            
            

        # import pdb
        # pdb.set_trace()
        conservative_intersection.intersect(delta_t[:,:-1], delta_t[:, -1:])
        underapproximate_intersection.intersect(delta_t[:,:-1], delta_t[:, -1:])
        overapproximate_intersection.intersect(delta_t[:,:-1], delta_t[:, -1:])
        #classical_intersection.intersect(delta_t[:,:-1], delta_t[:, -1:])
        volume_conservative[t+1] = conservative_intersection.current_volume
        volume_underapproximate[t+1] = underapproximate_intersection.current_volume
        volume_overapproximate[t+1] = overapproximate_intersection.current_volume
        volume_classical[t+1] = classical_intersection.current_volume


        # parameter_set_t = np.vstack((parameter_set_prev, delta_t))
        # _, interior_point_t = feasible_point(parameter_set_t[:, :-1], parameter_set_t[:, -1:])
        
        # half_space_intersection = HalfspaceIntersection(parameter_set_t, interior_point_t)
        # vertices = half_space_intersection.intersections

        # cvx_hull_intersection= ConvexHull(half_space_intersection.intersections)
        # volume_t = cvx_hull_intersection.volume

        # check_inclusion = False
        # for i in range(len(prev_vertices)):
        #     if np.any(parameter_set_t[:, :-1] @ prev_vertices[i] + parameter_set_t[:, -1] >  1e-15):
        #         check_inclusion = True
        #         break

        # if volume_t < volume[t] and check_inclusion:
        #     volume[t+1] = volume_t
            #parameter_set_prev = cvx_hull_intersection.equations
            # poly = polytope.qhull(vertices)
        
            # parameter_set_prev = np.hstack((poly.A, poly.b[:, None]))
            #point = feasible_point(parameter_set_prev[:, :-1], parameter_set_prev[:, -1:])
            #hs = HalfspaceIntersection(parameter_set_prev, point)
        


        # is_delta_included =  np.all(
        #         (parameter_set_prev[:, :-1] @ vertices.T) + np.tile(parameter_set_prev[:, -1], vertices.shape[0]).reshape(vertices.shape[0], parameter_set_prev[:, -1].shape[0]).T <= 1e-9
        # )

        current_volumes = f"{conservative_intersection.current_volume}- {underapproximate_intersection.current_volume}- {overapproximate_intersection.current_volume} - {classical_intersection.current_volume}"
        current_shapes = f"{conservative_intersection.current_H.shape} - {underapproximate_intersection.current_H.shape} - {overapproximate_intersection.current_H.shape} - {classical_intersection.current_H.shape}"
        print(f"{t} - {current_volumes} - {current_shapes} " )

  
        

        # #print(hypercube_delta.intersections)

        # if not np.all(np.isclose(prev_vertices - half_space_intersection.intersections ,0)):
        #     # The delta_t is not bigger than the previous set
        #     parameter_set_prev = parameter_set_t
        #     print('NOT BIG!')

        # cvx_hull = ConvexHull(half_space_intersection.intersections)

        # Vol[t] = cvx_hull.volume

        

        # is_inside = np.all(parameter_set_prev[:, :-1] @ C + parameter_set_prev[:, -1] <= 0)

#         print(f'[Iteration {t}] Error: {err[t]} -  Center: {interior_point_t} - number of vertices {half_space_intersection.intersections.shape[0]} - Volume: {cvx_hull.volume} - Point inside: {is_inside}')
plt.show()
fig, ax = plt.subplots(1,2)
ax[0].plot(epsilon[N_LS:], label=r'\epsilon_t')
ax[0].plot(error[N_LS:], label=r'\|\theta-\theta_t\|')
ax[0].grid()
ax[0].legend()
ax[0].set_yscale('log')

ax[1].plot(volume_conservative[N_LS:], label='Volume  conservative intersection')
ax[1].plot(volume_underapproximate[N_LS:], label='Volume  underapproximate intersection')
ax[1].plot(volume_overapproximate[N_LS:], label='Volume  overapproximate intersection')
ax[1].plot(volume_theta_hypercube[N_LS:], label='Volume hypercube around theta')
ax[1].grid()
ax[1].set_yscale('log')
ax[1].legend()
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