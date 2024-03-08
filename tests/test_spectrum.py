import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum
from yawisi.display import display_spectrum
from yawisi.locations import SinglePoint
import numpy as np
import matplotlib.pyplot as plt


class TestSpectrum(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_spectrum_single(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        params = SimulationParameters(filename)
        params.n_samples = 10000
        params.sample_time = 0.05

        spectrum = Spectrum(params)
        Sf = spectrum.compute_at_hub()

        # test \sigma_k^2 = \int_{f=0}^{\infty} S_k(f)
        # df = 1 / (N * dt)
        print(np.sqrt(spectrum.df * np.sum(Sf, axis=0)))

        display_spectrum(spectrum.params.kind, spectrum.freq, Sf)


if __name__ == "__main__":
    unittest.main()
