import configparser


class SimulationParameters:
    def __init__(self, filename):
        # Initialisation
        self.Filename = filename  # Fichier d'initialisation

        self.n_samples = 1000
        self.sample_time = 0.1
        self.wind_mean = 10.0  # Lecture de la vitesse moyenne
        self.kind = "karman"
        self.scale_1 = 170  # turbulence length scale for longitudinal component
        self.scale_2 = 100  # turbulence length scale for transversal component
        self.scale_3 = 35  # turbulence length scale for vertical component
        self.sigma_1 = 2.6  # std of wind fluctuation of the longitudinal component
        self.sigma_2 = 2.2  # std of wind fluctuation of the transversal Component
        self.sigma_3 = 1.56  # std of wind fluctuation of the vertical component
        self.vertical_shear = 0.3  # Lecture du parametre de gradient
        self.reference_height = (
            80  # Lecture de la hauteur de reference du champ de vent
        )
        self.ny = 10
        self.nz = 10
        self.zmin = 0
        self.zmax = 60
        self.ymin = 0
        self.ymax = 100
        # self.grid_width = 100
        # self.grid_height = 100
        # self.grid_length = 11

        if filename is not None:
            self.__parse_ini(filename)

    def __parse_ini(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        simulation = config["Simulation"]
        self.n_samples = int(simulation.get("n_samples", self.n_samples))
        self.sample_time = float(simulation.get("sample_time", self.sample_time))

        spectrum = config["Spectrum"]
        self.kind = spectrum.get("kind", self.kind)
        self.wind_mean = int(
            spectrum.get("wind_mean", self.wind_mean)
        )  # Lecture de la vitesse moyenne
        self.scale_1 = float(spectrum.get("scale_1", self.scale_1))
        self.scale_2 = float(spectrum.get("scale_2", self.scale_2))
        self.scale_3 = float(spectrum.get("scale_3", self.scale_3))
        self.sigma_1 = float(spectrum.get("sigma_1", self.sigma_1))
        self.sigma_2 = float(spectrum.get("sigma_2", self.sigma_2))
        self.sigma_3 = float(spectrum.get("sigma_3", self.sigma_3))

        field = config["Field"]
        self.vertical_shear = float(field.get("vertical_shear", self.vertical_shear))
        self.reference_height = float(field.get("hub_height", self.reference_height))
        self.ny = int(field.get("Ny", self.ny))
        self.nz = int(field.get("Nz", self.nz))
        self.ymin = float(field.get("Ymin", self.ymin))
        self.ymax = float(field.get("Ymax", self.ymax))
        self.zmin = float(field.get("Zmin", self.zmin))
        self.zmax = float(field.get("Zmax", self.zmax))
        # self.grid_width = float(field.get("grid_width", self.grid_width))
        # self.grid_height = float(field.get("grid_heigth", self.grid_height))
        # self.grid_length = int(field.get("grid_length", self.grid_length))

    @property
    def total_time(self):
        return self.n_samples * self.sample_time

    @property
    def freq_max(self):
        return 1.0 / self.sample_time

    def __str__(self) -> str:
        msg = f"Number of Samples Initialized @ {self.n_samples}\n"
        msg += f"Sample Time Initialized @ {self.sample_time}\n"
        msg += f"Wind Mean Speed Initialized @ {self.wind_mean}\n"
        msg += f"turbulence length scale for longitudinal component @ {self.scale_1}\n"
        msg += f"turbulence length scale for transversal component @ {self.scale_2}\n"
        msg += f"turbulence length scale for vertical component @ {self.scale_3}\n"
        msg += (
            f"Std of wind fluctuation of the longitudinal component @ {self.sigma_1}\n"
        )
        msg += (
            f"std of wind fluctuation of the transversal component @  {self.sigma_2}\n"
        )
        msg += f"std of wind fluctuation of the vertical component @ {self.sigma_3}\n"
        msg += f"PL Exp Law initialized  @ {self.vertical_shear}\n"
        msg += f"Reference Height @ {self.reference_height}\n"
        msg += f"Ymin @ {self.ymin}\n"
        msg += f"Ymax @ {self.ymax}\n"
        msg += f"Zmin @ {self.zmin}\n"
        msg += f"Zmax @ {self.zmax}\n"        
        # msg += f"Grid Width @ {self.grid_width}\n"
        # msg += f"Grid Height @ {self.grid_width}\n"
        # msg += f"Grid Length @ {self.grid_length}\n"
        return msg
