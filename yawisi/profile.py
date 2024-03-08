from yawisi.parameters import SimulationParameters
from yawisi.locations import Locations


class Profile:
    def __init__(self, params: SimulationParameters):
        self.params = params

    def __call__(self, locations: Locations):
        pass


class PowerProfile(Profile):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)
        self.a = 0.15  # Power law coefficient

    def __call__(self, locations: Locations):
        return (
            self.params.wind_mean
            * (locations.z_array() / self.params.reference_height) ** self.a
        )
