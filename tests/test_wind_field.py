import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.wind_field import WindField
from yawisi.locations import Locations
from yawisi.display import display_coherence_function, display_field
from yawisi.io import to_bts, from_bts
import matplotlib.pyplot as plt


class TestWindField(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_points_wind_field(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)

        wind_field.locations = Locations.create("points")
        wind_field.locations.add_points([(0, z) for z in range(58, 61)])

        wind_field.compute()
        display_field(wind_field=wind_field)

    def test_grid_field(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)
        wind_field.compute()
        print(wind_field)
        # display_field(wind_field=wind_field)

        ts = wind_field.get_uvwt()
        print(ts.shape)
        for i in range(3):
            plt.plot(ts[i, :, 0, 0])
        plt.show()

        uhub = wind_field.get_umean()
        print(uhub)

    def test_to_bts(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        btsfile = os.path.join(os.path.dirname(__file__), "./data", "test_to_bts.bts")

        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)
        wind_field.compute()
        to_bts(wind_field=wind_field, path=btsfile)

    def test_from_bts(self):
        btsfile = os.path.join(os.path.dirname(__file__), "./data", "test_from_bts.bts")
        wind_field = from_bts(btsfile)
        print(wind_field)
        # ts = wind_field.get_uvwt()
        # print(ts.shape)
        # for i in range(3):
        #     plt.plot(ts[i, :, 0, 0])
        # plt.show()

        btsfile = os.path.join(os.path.dirname(__file__), "./data", "test_from_1y.bts")
        wind_field = from_bts(btsfile)
        print(wind_field)


if __name__ == "__main__":
    unittest.main()
