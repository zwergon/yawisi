import numpy as np
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum
from yawisi.locations import Locations, Grid
from yawisi.kernels import CoherenceKernel
from yawisi.wind import Wind
from tqdm import tqdm

class WindField:
    """
    cette classe permet de definir un champ de vent contenant un certain nombre de points,
    et permet de generer le vecteur de vent
    """
    def __init__(self, params: SimulationParameters):
        self.info = None
        self.params: SimulationParameters = params #Def des parametres de simulation pour le Wind Field
        self.coherence_kernel = CoherenceKernel()
        self.locations: Locations = Locations.create("grid", 
                                                  ny=self.params.ny, 
                                                  nz=self.params.nz, 
                                                  ymin=self.params.ymin,
                                                  ymax=self.params.ymax,
                                                  zmin=self.params.zmin, 
                                                  zmax=self.params.zmax
                                                  ) 
        self.wind=[]   #Objets vent contenus dans le champ

    def get_umean(self):
        assert isinstance(self.locations, Grid), " can only generate for a grid"
        n_points = len(self.locations)
        # center of the grid is the location in the middle of the list
        pt = self.locations.points[n_points // 2, :]
        wind = self.wind[n_points//2]
        return wind.wind_values[:, 0].mean()

    def get_uvwt(self):
        """format (3 x nt x ny x nz )"""
        assert isinstance(self.locations, Grid), " can only generate for a grid"

        grid: Grid = self.locations 

        ny, nz = grid.dims[0], grid.dims[1]
        nt = self.params.n_samples
        ts = np.empty((3, nt, ny, nz))
        for idx, w in enumerate(self.wind):
            i, j = grid.coords(idx)
            ts[:, :, i, j] = np.transpose(w.wind_values)

        return ts
     
    @property
    def is_initialized(self) -> bool:
        return len(self.wind) > 0
    
    def get_coherence_function(self):
        N=self.params.n_samples
        Ts=self.params.sample_time

        freq = np.arange(0, 1/Ts, 1/(Ts*N))
        coherence_function = np.zeros(shape=(N, ))
        
        coherence_function[:N//2] = self.coherence_kernel(freq[:N//2])
        coherence_function = np.pad(coherence_function[:N//2], [0, N//2], mode='reflect')
        return freq, coherence_function
    
    def _get_coherence_matrix(self, factor, distance_matrix):
        return np.exp(-factor*distance_matrix)
    
    def compute(self):

        N = self.params.n_samples
        n_points = len(self.locations)

        spectrum = Spectrum(self.params) #Spectre du signal de vent
      
        #Definition des transformation de Fourier des seeds du vent en chaque point
        fft_seed = np.zeros(shape =(n_points, N, 3), dtype=np.complex64)
        for i_pt in range(n_points):
            fft_seed[i_pt, :, :] = Wind.get_initial_fftseed(N)
            
            #Multiplication par la matrice de coherence
        
        distance_matrix = self.locations.get_distance_matrix() # store a distance matrix 
        _, coherence_function = self.get_coherence_function()
        for i in tqdm(range(N)):
            coherence_matrix = self._get_coherence_matrix(coherence_function[i], distance_matrix)
            L = np.linalg.cholesky(coherence_matrix)
            fft_seed[:, i, :] = np.dot(L, fft_seed[:, i, :])  

      
        for i_pt in range(n_points):
            pt = self.locations.points[i_pt]
            wind = Wind(self.params)
            wind.wind_mean =  np.array(
              [
                 self.params.wind_mean*((self.params.reference_height+pt[1])/self.params.reference_height)**(self.params.vertical_shear),
                 0,
                 0   
              ])
            wind.compute(fft_seed=fft_seed[i_pt, :, :], spectrum=spectrum)
            self.wind.append(wind)

    def __repr__(self):
        s='<{} object> with keys:\n'.format(type(self).__name__)

         # calculate intermediate parameters
        y = np.sort(np.unique(self.locations.y_array()))
        z = np.sort(np.unique(self.locations.z_array()))
        ny = y.size  # no. of y points in grid
        nz = z.size  # no. of z points in grif
        nt = self.params.n_samples # no. of time steps
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

        s+=' - info: {}\n'.format(self.info)
        s+=' - z: [{} ... {}],  dz: {}, n: {} \n'.format(z[0], z[-1], dz, nz)
        s+=' - y: [{} ... {}],  dy: {}, n: {} \n'.format(y[0], y[-1], dy, ny)
        s+=' - t: [{} ... {}],  dt: {}, n: {} \n'.format(0, dt*nt, dt, nt)
        s+=' - uhub: ({}) zhub ({})\n'.format(uhub, zhub)
        
        if isinstance(self.locations, Grid):
            uvwt = self.get_uvwt()
            s+=' - u: ({} x {} x {} x {}) \n'.format(*(uvwt.shape))
            ux, uy, uz = uvwt[0, :, :, :], uvwt[1, :, :, :], uvwt[2, :, :, :]
            s+='    ux: min: {}, max: {}, mean: {} \n'.format(np.min(ux), np.max(ux), np.mean(ux))
            s+='    uy: min: {}, max: {}, mean: {} \n'.format(np.min(uy), np.max(uy), np.mean(uy))
            s+='    uz: min: {}, max: {}, mean: {} \n'.format(np.min(uz), np.max(uz), np.mean(uz))

        return s
