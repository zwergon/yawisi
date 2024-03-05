import numpy as np

from yawisi.parameters import SimulationParameters
from yawisi.kernels import Kaimal, Karman


class Spectrum:
    def __init__(self, params: SimulationParameters):
        self.params = params
        try:
            kind = self.params.kind.lower()
            self.kernel = {"kaimal": Kaimal, "karman": Karman}[kind](params)
        except KeyError as er:
            raise KeyError(f"spectrum {kind} unknown (only kaimal, karman)")

        self.freq, self.array = self._compute(params.n_samples, params.sample_time)

    @property
    def df(self):
        return self.freq[1] - self.freq[0]

    @property
    def N_freq(self):
        return len(self.freq)

    def symetrized(self, i):
        one_d = self.array[:, i].astype(np.complex64)
        return np.hstack(
            [[0], one_d[:-1], np.real(one_d[-1]), np.conjugate(one_d[-2::-1])]
        )

    def _sampling_params(self, N, dt):
        fs = 1 / dt
        tmax = N * dt
        f0 = 1 / tmax
        fc = fs / 2  # Nyquist freq
        return np.arange(f0, fc + f0, f0)

    def _compute(self, N, dt):
        freq = self._sampling_params(N, dt)
        array = np.zeros(shape=(len(freq), 3))
        for i in range(3):
            array[:, i] = self.kernel(freq, i)

        # ensure \sigma_k^2 = \int_{f=0}^{\infty} S_k(f)
        # df = 1 / (N * dt)
        # array *= self.kernel.var_k / (df * np.sum(array, axis=0))

        return freq, array
