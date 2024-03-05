import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum
from yawisi.display import display_spectrum

import numpy as np
import matplotlib.pyplot as plt


class TestSpectrum(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        params = SimulationParameters(filename)
        params.n_samples = 10000
        params.sample_time = 0.05
        spectrum = Spectrum(params)

        _, array = spectrum.freq, spectrum.array

        print(array.shape)
        print(np.sqrt(spectrum.df * np.sum(array, axis=0)))

        display_spectrum(spectrum)

    def test_symetrized(self):
        params = SimulationParameters(None)
        params.n_samples = 64
        params.sample_time = 0.05

        spectrum = Spectrum(params)
        symetrized = spectrum.symetrized(0)
        # plt.plot(symetrized.real)
        # plt.plot(symetrized.imag)
        # plt.show()

        signal = np.fft.ifft(symetrized)
        print(f"real: {np.std(signal.real)}, imaginary : {np.mean(signal.imag)}")


if __name__ == "__main__":
    unittest.main()
