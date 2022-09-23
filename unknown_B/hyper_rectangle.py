from __future__ import annotations
from typing import Union
from itertools import product
import numpy as np


class HyperRectangle(object):
    _dim: int
    _intervals: np.ndarray

    def __init__(self, intervals: np.ndarray):
        """
        Constructs a d-dimensional hyper-rectangle.

        :param intervals: a (d x 2) numpy array definining the intervals
            of the rectangle, s.t. d[i,0]< d[i,1]. Es: [(a1,a2), (b1,b2), ...]
        """
        assert len(intervals.shape) == 2 and intervals.shape[1] == 2, "intervals needs to be a (d x 2) numpy array"
        assert np.all(intervals[:,0] <= intervals[:,1]), "Intervals not well defined. The first column should be lower than the second column"
        self._dim = intervals.shape[0]
        self._intervals = intervals

    @property
    def dim(self) -> int:
        """ Dimensionality of the hyper rectangle """
        return self._dim

    @property
    def left_intervals(self) -> np.ndarray:
        """ Returns the left intervals """
        return self._intervals[:, 0]

    @property
    def right_intervals(self) -> np.ndarray:
        """ Returns the right intervals """
        return self._intervals[:, 1]

    @property
    def center(self) -> np.ndarray:
        """ Returns the center """
        return (self.left_intervals + self.right_intervals) / 2

    @property
    def vertices(self) -> np.ndarray:
        """
        Returns the list of vertices as a Nxd matrix, where N=2^d.
        """
        iterables = self._intervals.tolist()
        vertices = [list(v) for v in product(*iterables)]
        return np.array(vertices)

    @property
    def volume(self) -> float:
        """
        Returns the volume of the hyper rectangle
        """
        return np.prod(np.diff(self._intervals))

    def contains(self, point: np.ndarray) -> bool:
        """
        Checks if a given point is contained inside the hyper rectangle
        :param point: a d-dimensional array
        :return: true if the point is contained, false otherwise
        """
        _point = point.flatten()
        assert len(_point) == self.dim, "Wrong dimensionality"

        return np.all(self._intervals[:,0] <= point) and np.all(point <= self._intervals[:, 1])

    def intersect(self, operand: HyperRectangle) -> HyperRectangle:
        """
        Intersects two hyper rectangles.
        :param operand: hyper rectangle
        :return: the intersection between the two hyper rectangles
        """
        assert operand.dim == self.dim, "Wrong dimensionality"
        left_intervals = np.maximum(self.left_intervals, operand.left_intervals).flatten()
        right_intervals = np.minimum(self.right_intervals, operand.right_intervals).flatten()
        return HyperRectangle(np.vstack((left_intervals, right_intervals)).T)

    def scale(self, scaling_factor: Union[np.ndarray,float]) -> HyperRectangle:
        """
        Scales the hyper rectangle by a factor
        """
        if isinstance(scaling_factor, float):
            scaling_factor = np.array([scaling_factor] * self.dim)
        assert len(scaling_factor) == self.dim, "Wrong dimensionality for the scaling factor"
        assert np.all(scaling_factor > 0), "scaling_factor needs to be positive"
        return HyperRectangle(np.multiply(self._intervals.copy(), scaling_factor))

    def normalize(self) -> HyperRectangle:
        """
        Normalizes the hyper rectangle so that each interval has length 1
        """
        delta = np.diff(self._intervals).flatten()
        return self.scale(1/delta)

    def translate(self, new_center: np.ndarray) -> HyperRectangle:
        """
        Moves the center of the hyper rectangle
        """
        _point = new_center.flatten()
        assert len(_point) == self.dim, "Wrong dimensionality"
        delta = np.diff(self._intervals).flatten()
        left_intervals = _point - delta
        right_intervals = _point + delta
        return HyperRectangle(np.vstack(left_intervals, right_intervals).T)

    def copy(self) -> HyperRectangle:
        """ Returns a copy of the hyper rectangle """
        return HyperRectangle(self._intervals.copy())

    @staticmethod
    def build_hypercube(center: np.ndarray, half_side_length: float) -> HyperRectangle:
        """
        Builds an hypercube around a center point

        :param center: center of the hypercube
        :param half_side_length: half the length of one side
        :return: an hypercube
        """
        assert half_side_length > 0, "Length is not positive"
        point = center.flatten()
        dim = len(point)
        
        delta = np.ones((dim)) * half_side_length

        intervals = np.vstack((point - delta, point + delta))

        return HyperRectangle(intervals.T)
