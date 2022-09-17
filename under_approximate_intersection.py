import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from utils import build_hypercube, compute_stabilizing_K, feasible_point

class UnderApproximateIntersection(object):
    initial_H: np.ndarray
    current_H: np.ndarray
    current_volume: float
    current_vertices: np.ndarray
    current_radius_inner_circle: float
    current_interior_point: np.ndarray

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.initial_H = np.hstack((A, b))
        self.current_H = self.initial_H.copy()

        # Compute interior point
        radius_inner_circle, interior = feasible_point(A, b)
        self.current_radius_inner_circle = radius_inner_circle
        self.current_interior_point = interior


        self.current_intersection = HalfspaceIntersection(self.current_H, self.current_interior_point)
        self.current_vertices = self.current_intersection.intersections

        self.current_volume = ConvexHull(self.current_vertices).volume
        

    def intersect(self, A: np.ndarray, b: np.ndarray):
        new_H = np.hstack((A, b))
        intersection_H = np.vstack((self.current_H, new_H))

        radius_inner_circle, interior_point = feasible_point(intersection_H[:, :-1], intersection_H[:, -1:])

        self.current_H = np.hstack(build_hypercube(interior_point, radius_inner_circle))
        
        half_space_intersection = HalfspaceIntersection(self.current_H, interior_point)
        vertices_intersections = half_space_intersection.intersections

        self.current_volume = ConvexHull(vertices_intersections).volume

   

        self.current_interior_point = interior_point
        self.current_vertices = vertices_intersections
        self.current_radius_inner_circle = radius_inner_circle
        self.current_interior_point = interior_point