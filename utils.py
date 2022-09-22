import numpy as np
import cvxpy as cp
from typing import Tuple, Union, Optional


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

def compute_stabilizing_K(vertices: np.ndarray, dim_x: int, dim_u: int) -> Union[np.ndarray, type[None]]:
    """
    Computes a stabilizing feedback gain K for all pairs (A,B) in a polytope.
    To do so, it requires the the vertices of the polytope.
    Remark: this function does not check that the vertices define a polytope

    :param vertices: a matrix with N rows, and (dim_x * (dim_x + dim_u)) columns.
                     N is the number of vertices, and (dim_x * (dim_x + dim_u)) is
                     the dimensionality of the problem
    :return K: stabilizing matrix K if it exists, None otherwise
    """
    assert isinstance(vertices, np.ndarray), "vertices should be an array of vertices"
    assert len(vertices.shape) == 2, "Vertices should be a matrix with N rows, and (dim_x * (dim_x + dim_u)) columns"
    assert vertices.shape[1] == dim_x * (dim_x+dim_u), "Each vertex should have dim_x+dim_u elements"

    num_vertices = vertices.shape[0]
    X = cp.Variable((dim_x, dim_x), symmetric=True)
    Z = cp.Variable((dim_u, dim_x))

    constraints= [X >> 0]
    for i in range(num_vertices):
        vertex = np.reshape(vertices[i], (dim_x, dim_x + dim_u))
        A, B = vertex[:, :dim_x], vertex[:, dim_x:]

        Fn = A @ X + B @ Z
        M = cp.bmat([[X, Fn], [Fn.T, X]])
        constraints.append(M >> 0)

    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve(solver=cp.MOSEK)

    if res is None or np.isclose(0, np.linalg.det(X.value)):
        return None

    return Z.value @ np.linalg.inv(X.value)
    

def compute_spectral_radius(vertices: np.ndarray, dim_x: int, dim_u: int, K: np.ndarray) -> Tuple[float, float]:
    """
    Computes the spectral radius of a closed loop system over a polytopic uncertain set for the model
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
    vertices = vertices.reshape(vertices.shape[0], dim_x, dim_x+dim_u)
    for i in range(vertices.shape[0]):
        A, B = vertices[i, :, :dim_x], vertices[i, :, dim_x:]
        values[i] = np.abs(np.linalg.eigvals(A +  B @ K)).max()
    
    rho = np.max(values)
    assert rho < 1, "Spectral radius is not lower than 1"
    feasible_lambda = (1 + rho) /2
    return rho, feasible_lambda



