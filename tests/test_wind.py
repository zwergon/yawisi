import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.wind import Wind
from yawisi.display import display_wind

import numpy as np


class TestWind(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_wind(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        params = SimulationParameters(filename)
        params.n_samples = 4096
        params.sample_time = 0.05
        print(params)

        wind = Wind(params)
        wind.compute()

        u = wind.wind_values[:, 0]
        mean_u = np.mean(u)

        std_u = np.std(u - mean_u)
        print(
            mean_u,
            std_u,
            std_u * np.sqrt(params.sample_time),
            std_u / params.sigma_1,
        )

        display_wind(wind)


if __name__ == "__main__":
    unittest.main()
