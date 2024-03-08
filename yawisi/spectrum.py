import numpy as np

from yawisi.parameters import SimulationParameters
from yawisi.kernels import Kaimal, Karman
from yawisi.locations import Locations, SinglePoint


class Spectrum:
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
