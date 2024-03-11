from yawisi.parameters import SimulationParameters
from yawisi.locations import Locations


class Law:
    def __init__(self, params: SimulationParameters):
        self.params = params

    def __call__(self, locations: Locations):
        pass


class PowerLaw(Law):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)
        self.a = 0.15  # Power law coefficient

    def __call__(self, locations: Locations):
        return (
            self.params.wind_mean
            * (locations.z_array() / self.params.reference_height) ** self.a
        )


class Profile:
    """
    This class defines the profile in z of the wind
    """

    def __init__(self, params: SimulationParameters):
        self.law = PowerLaw(params)

    def __call__(self, locations: Locations):
        return self.law(locations=locations)
