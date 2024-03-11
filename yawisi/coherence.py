import numpy as np

from yawisi.parameters import SimulationParameters
from yawisi.locations import Locations
from yawisi.spectrum import Spectrum
from yawisi.profile import Profile


class CoherenceKernel:
    def compute(self, locations: Locations, spectrum: Spectrum, profile: Profile):
        pass


class FrostCoherenceKernel(CoherenceKernel):
    def compute(self, locations: Locations, spectrum: Spectrum, profile: Profile):
        mean_u = profile(locations)
        mean_u_jk = 0.5 * np.add.outer(mean_u.ravel(), mean_u.ravel())

        delta_r_jk = locations.get_distance_matrix()

        C = 7.5

        cru_jk = C * delta_r_jk / mean_u_jk

        Coh_jk = np.exp(-cru_jk[:, :, np.newaxis] * spectrum.freq)

        return Coh_jk


class Coherence:
    """
    This class defines the spatial coherence of the generated wind.
    """

    def __init__(self, params: SimulationParameters) -> None:
        self.profile = Profile(params)
        self.kernel = FrostCoherenceKernel()

    def compute(self, locations: Locations, spectrum: Spectrum):
        return self.kernel.compute(locations, spectrum, self.profile)
