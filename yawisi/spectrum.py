import math
import numpy as np

from yawisi.parameters import SimulationParameters
from yawisi.locations import Locations, SinglePoint


class Kernel:
    def __init__(self, params: SimulationParameters):
        self.Lk = [params.scale_1, params.scale_2, params.scale_3]
        self.var_k = [params.sigma_1**2, params.sigma_2**2, params.sigma_3**2]
        self.Vhub = params.wind_mean

    def __call__(self, freq, k):
        pass


class Kaimal(Kernel):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def __call__(self, freq, k):
        numerator = 4 * self.var_k[k] * self.Lk[k] / self.Vhub
        denominator = (1 + 6 * freq * self.Lk[k] / self.Vhub) ** (5 / 3)
        return numerator / denominator


class Karman(Kernel):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def __call__(self, freq, k):
        fr = self.Lk[k] * freq / self.Vhub
        if k == 0:
            S = 4 * fr / (1 + 71 * fr**2) ** (5 / 6)
        else:
            S = 4 * fr * (1 + 755 * fr**2) / (1 + 283 * fr**2) ** (11 / 6)
        return S * self.var_k[k] / freq


class Spectrum:
    """
    This class defines the temporal correlation of a wind 1D Signal.
    """

    def __init__(self, params: SimulationParameters):
        self.params = params
        try:
            kind = self.params.kind.lower()
            self.kernel = {"kaimal": Kaimal, "karman": Karman}[kind](params)
        except KeyError as er:
            raise KeyError(f"spectrum {kind} unknown (only kaimal, karman)")

        self.freq = self._sampling_params(params.n_samples, params.sample_time)

    @property
    def df(self):
        return self.freq[1] - self.freq[0]

    @property
    def N_freq(self):
        return len(self.freq)

    def _sampling_params(self, N, dt):
        fs = 1 / dt
        tmax = N * dt
        f0 = 1 / tmax
        fc = fs / 2  # Nyquist freq
        return np.arange(f0, fc + f0, f0)

    def compute_at_hub(self):
        hub = SinglePoint(z=self.params.reference_height)
        Sf = self.compute(hub)
        return Sf.squeeze(0)

    def compute(self, locations: Locations):
        array = np.zeros(shape=(len(locations), self.N_freq, 3))
        # TODO so far, kernels don't use location but it should !
        for i_pt in range(len(locations)):
            for i in range(3):
                array[i_pt, :, i] = self.kernel(self.freq, i)
        return array
