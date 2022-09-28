import numpy as np
import cvxpy as cp
import time
from typing import Tuple, Union, Optional, List, Callable
import scipy.linalg as linalg


def build_aligned_hypercube(A: np.ndarray, center: np.ndarray, half_side_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds an aligned hypercube around a center point. It is aligned with respect to another
    hyperectangle defined by A.
    Returns [A,b], the halfspaces definining the hypercube Ax+b<=0

    :param A: halfspaces definining an hyperrectangle
    :param center: center of the hypercube
    :param half_side_length: half the length of one side
    :return A: contains (2*n rows, n cols), where n is the dimensionality of the center
    :return b: b contains 2n rows
    """
    assert isinstance(center, np.ndarray) or isinstance(center, list), 'Center is not an array'
    if isinstance(center, list):
        center = np.array(center)
    if len(center.shape) == 1:
        center = center[:, None]    

    b = cp.Variable(A.shape[0])
    
    
    constraints = [A @ center + half_side_length * np.linalg.norm(A[i,:], ord=2) <= -b[:, None] for i in range(A.shape[0])]

    problem = cp.Problem(cp.Minimize(cp.norm(b, p=2)), constraints)
    res = problem.solve(solver=cp.MOSEK)

    if res is None or np.isclose(0, res):
        raise Exception('Could not find an aligned hypercube')

    return A, b.value[:, None]

def build_hypercube(center: np.ndarray, half_side_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds an hypercube around a center point. Returns [A,b], the halfspaces
    definining the hypercube Ax+b<=0

    :param center: center of the hypercube
    :param half_side_length: half the length of one side
    :return A: A contains (2*n rows, n cols), where n is the dimensionality of the center
    :return b: b contains 2n rows
    """
    assert isinstance(center, np.ndarray) or isinstance(center, list), 'Center is not an array'
    if isinstance(center, list):
        center = np.array(center)
    if len(center.shape) == 1:
        center = center[:, None]
    dim_n = center.shape[0]
    b = -half_side_length * np.ones((2 * dim_n, 1))
    A = np.vstack([np.eye(dim_n), -np.eye(dim_n)])

    _center = np.vstack((-center, center))

    return A, b + _center

def feasible_point(A: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes a feasible point that belongs to the intersection of the halfspaces defined in (A,B)
    Ax+b <=0
    """
    dim = A.shape[1]
    y = cp.Variable()
    x = cp.Variable((dim,1))

    constraint = [A @ x + y * cp.norm(A[i,:]) <= -b for i in range(A.shape[0])]
    problem = cp.Problem(cp.Maximize(y), constraint)
    res = problem.solve()
    return res, x.value.flatten()

def compute_stabilizing_K(B: np.ndarray, vertices: np.ndarray, dim_x: int, dim_u: int) -> Union[np.ndarray, type[None]]:
    """
    Computes a stabilizing feedback gain K for all (A) in a polytope.
    To do so, it requires the the vertices of the polytope.
    Remark: this function does not check that the vertices define a polytope

    :param vertices: a matrix with N rows, and (dim_x * (dim_x)) columns.
                     N is the number of vertices, and (dim_x * (dim_x)) is
                     the dimensionality of the problem
    :return K: stabilizing matrix K if it exists, None otherwise
    """
    assert isinstance(vertices, np.ndarray), "vertices should be an array of vertices"
    assert len(vertices.shape) == 2, "Vertices should be a matrix with N rows, and (dim_x * (dim_x )) columns"
    assert vertices.shape[1] == dim_x * (dim_x), "Each vertex should have dim_x elements"

    num_vertices = vertices.shape[0]
    X = cp.Variable((dim_x, dim_x), symmetric=True)
    Z = cp.Variable((dim_u, dim_x))

    constraints= [X >> 0]
    for i in range(num_vertices):
        A = np.reshape(vertices[i], (dim_x, dim_x))

        Fn = A @ X + B @ Z
        M = cp.bmat([[X, Fn], [Fn.T, X]])
        constraints.append(M >> 0)

    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve(solver=cp.MOSEK)

    if res is None or np.isclose(0, np.linalg.det(X.value)):
        return None

    return Z.value @ np.linalg.inv(X.value)
    

def compute_joint_spectral_radius(B: np.ndarray, vertices: np.ndarray, dim_x: int, dim_u: int, K: np.ndarray) -> Tuple[float, float]:
    """
    Computes the joint spectral radius of a closed loop system over a polytopic uncertain set for the model
    parameters

    See also eq. (5.72), Chapter 5 in Model Predictive Control, Kouvaritakis et al.

    :param vertices: a matrix where each row is a vertex of the polytope
    :param dim_x: dimensionality of the state
    :param dim_u: dimensionality of the control signal
    :param K: feedback gain (a dim_u x dim_x matrix)
    :return rho: spectral radius of the system
    :return feasible_lambda: a feasible value of lambda to definea lambda-contractive set
    """
    values = np.zeros(vertices.shape[0])
    vertices = vertices.reshape(vertices.shape[0], dim_x, dim_x)
    for i in range(vertices.shape[0]):
        A = vertices[i]
        values[i] = np.abs(np.linalg.eigvals(A +  B @ K)).max()
    
    rho = np.max(values)
    assert rho < 1, "Spectral radius is not lower than 1"
    feasible_lambda = (1 + rho) /2
    return rho, feasible_lambda


def compute_contractive_polytope(n: int, lmbd: float, B: np.ndarray, F: np.ndarray, G: np.ndarray, K: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Compute a lmbd-contractive polytope

    :param n: order
    :param lmbd: lambda value
    :param B: B matrix
    :param F: F matrix
    :param G: G matrix
    :param K: feedback gain
    :param vertices: vertices of the uncertain polytope
    :return: T matrix
    """
    dim_u, dim_x = K.shape
    num_vertices = vertices.shape[0]
    vertices = vertices.reshape(num_vertices, dim_x, dim_x)

    T = [F + G @ K]
    for ji in range(num_vertices):
        A = vertices[ji]
        phi = A + B @ K
        T.extend([T[0] @ phi / (lmbd ** (i + 1)) for i in range(n)])


    return np.vstack(T)

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
        # Initial constarints
        x[0] == x0,
        T @ x0 <= alpha[0],
        # System constraints
        x[1:].T == (A + B @ K) @ x[:-1].T + B @ v.T,
        # First tube constraint
        Hc @ alpha[:-1].T + G @ v.T <= 1,
        # First terminal constraint
        Hc @ alpha[-1] <= 1
    ]

    for i in range(num_vertices):
        # Second terminal constraint
        constraints.append(
            matrices_H[i] @ alpha[-1] + wbar <= alpha[-1],
        )
        for k in range(horizon):
            # Second tube constraint
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
        res = problem.solve(solver=cp.MOSEK, warm_start=True, enforce_dpp=True, verbose=False)


        elapsed_time = time.time() - start
        
        return v.value, alpha.value, x.value, res, elapsed_time

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