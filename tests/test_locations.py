import unittest
import numpy as np
from yawisi.locations import Locations
from yawisi.display import display_points
import matplotlib.pyplot as plt


class TestLocations(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_points(self):
        pts = [(0, 0), (0, 1), (1, 0)]
        points = Locations.create("points")
        points.add_points(pts)
        self.assertEqual(3, len(points))

        distance_matrix = np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.41421356], [1.0, 1.41421356, 0.0]]
        )

        self.assertTrue(np.allclose(distance_matrix, points.get_distance_matrix()))

    def test_grid(self):
        grid = Locations.create(
            "grid",
            ny=10,
            nz=10,
            ymin=-50,
            ymax=50,
            zmin=-50,
            zmax=50,
        )
        display_points(grid)

        distance_map = grid.get_distance_matrix()
        self.assertEqual(distance_map.shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
