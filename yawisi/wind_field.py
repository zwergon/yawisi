import numpy as np
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum
from yawisi.locations import Locations, Grid
from yawisi.profile import Profile
from yawisi.wind import Wind
from yawisi.coherence import Coherence

from tqdm import tqdm


class WindField:
    """
    cette classe permet de definir un champ de vent contenant un certain nombre de points,
    et permet de generer le vecteur de vent
    """

    def __init__(self, params: SimulationParameters):
        self.info = None
        self.params: SimulationParameters = (
            params  # Def des parametres de simulation pour le Wind Field
        )
        self.locations: Locations = Locations.create(
            "grid",
            ny=self.params.ny,
            nz=self.params.nz,
            ymin=self.params.ymin,
            ymax=self.params.ymax,
            zmin=self.params.zmin,
            zmax=self.params.zmax,
        )

        self.winds = []  # Objets vent contenus dans le champ

    def get_umean(self):
        assert isinstance(self.locations, Grid), " can only generate for a grid"
        n_points = len(self.locations)
        # center of the grid is the location in the middle of the list
        pt = self.locations.points[n_points // 2, :]
        wind = self.winds[n_points // 2]
        return wind.wind_values[:, 0].mean()

    def get_uvwt(self):
        """format (3 x nt x ny x nz )"""
        assert isinstance(self.locations, Grid), " can only generate for a grid"

        grid: Grid = self.locations

        ny, nz = grid.dims[0], grid.dims[1]
        nt = self.params.n_samples
        ts = np.empty((3, nt, ny, nz))
        for idx, w in enumerate(self.winds):
            i, j = grid.coords(idx)
            ts[:, :, i, j] = np.transpose(w.wind_values)

        return ts

    @property
    def is_initialized(self) -> bool:
        return len(self.winds) > 0

    def compute(self):
        N = self.params.n_samples
        dt = self.params.sample_time
        n_points = len(self.locations)

        spectrum = Spectrum(self.params)  # Spectre du signal de vent
        Sf = spectrum.compute(self.locations)

        coherence = Coherence(self.params)
        Coh_jk = coherence.compute(self.locations, spectrum)

        Sf_jk = (
            np.sqrt(Sf[:, np.newaxis, :, :] * Sf[np.newaxis, :, :, :])
            * Coh_jk[:, :, :, np.newaxis]
        )

        transposed = np.transpose(Sf_jk, axes=(3, 2, 1, 0))
        L = np.linalg.cholesky(transposed)

        one = np.ones(shape=(n_points, 1))
        hx1 = np.zeros(shape=(spectrum.N_freq, n_points, 3), dtype=np.complex128)
        for j in range(3):
            for i_f in range(spectrum.N_freq):
                vec = np.exp(1.0j * np.random.uniform(0, 2 * np.pi, size=n_points))
                X = np.diag(vec)
                hx1[i_f, :, j] = np.dot(np.dot(L[j, i_f, :, :], X), one).squeeze(1)

        full = np.vstack(
            [
                np.zeros(shape=(1, n_points, 3)),
                hx1[:-1, :, :],
                np.real(hx1[-1:, :, :]),
                np.conjugate(hx1[-2::-1, :, :]),
            ]
        )

        u = np.real(np.fft.ifft(full, axis=0)) * np.sqrt(spectrum.N_freq / dt)

        profile = Profile(self.params)
        for i_pt in range(n_points):
            wind = Wind(self.params)
            mean_u = profile(self.locations)
            wind_mean = np.array(
                [
                    mean_u[i_pt],
                    0,
                    0,
                ]
            )

            wind.wind_values = u[:, i_pt, :] + wind_mean[np.newaxis, :]
            self.winds.append(wind)

    def __repr__(self):
        s = "<{} object> with keys:\n".format(type(self).__name__)

        # calculate intermediate parameters
        y = np.sort(np.unique(self.locations.y_array()))
        z = np.sort(np.unique(self.locations.z_array()))
        ny = y.size  # no. of y points in grid
        nz = z.size  # no. of z points in grif
        nt = self.params.n_samples  # no. of time steps
        if y.size == 1:
            dy = 0
        else:
            dy = np.mean(y[1:] - y[:-1])  # hopefully will reduce possible errors
        if z.size == 1:
            dz = 0
        else:
            dz = np.mean(z[1:] - z[:-1])  # hopefully will reduce possible errors
        dt = self.params.sample_time  # time step
        zhub = z[z.size // 2]  # halfway up
        uhub = self.get_umean()  # mean of center of grid

        s += " - info: {}\n".format(self.info)
        s += " - z: [{} ... {}],  dz: {}, n: {} \n".format(z[0], z[-1], dz, nz)
        s += " - y: [{} ... {}],  dy: {}, n: {} \n".format(y[0], y[-1], dy, ny)
        s += " - t: [{} ... {}],  dt: {}, n: {} \n".format(0, dt * nt, dt, nt)
        s += " - uhub: ({}) zhub ({})\n".format(uhub, zhub)

        if isinstance(self.locations, Grid):
            uvwt = self.get_uvwt()
            s += " - u: ({} x {} x {} x {}) \n".format(*(uvwt.shape))
            ux, uy, uz = uvwt[0, :, :, :], uvwt[1, :, :, :], uvwt[2, :, :, :]
            s += "    ux: min: {}, max: {}, mean: {} \n".format(
                np.min(ux), np.max(ux), np.mean(ux)
            )
            s += "    uy: min: {}, max: {}, mean: {} \n".format(
                np.min(uy), np.max(uy), np.mean(uy)
            )
            s += "    uz: min: {}, max: {}, mean: {} \n".format(
                np.min(uz), np.max(uz), np.mean(uz)
            )

        return s
