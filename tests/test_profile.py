import unittest
import os
from yawisi.locations import Locations
from yawisi.profile import PowerProfile
from yawisi.parameters import SimulationParameters
import matplotlib.pyplot as plt


class TestProfile(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_power_profile(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        points = Locations.create("points")
        points.add_points([(0, z) for z in range(20, 61)])

        params = SimulationParameters(filename)
        profile = PowerProfile(params)
        meanU = profile(points)
        plt.plot(points.z_array(), meanU)
        plt.show()


if __name__ == "__main__":
    unittest.main()
