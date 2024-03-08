import numpy as np
import matplotlib.pyplot as plt
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum


class Wind:
    """
    Cette classe permet de definir un objet Vent, contenant une seed (pour chaque composante)
    Elle permet egalement de contenir les valeurs du vent, qui peuvent etre initialisee
    a partir de la classe spectre, ou par la fonction du champ de vent.
    """

    def __init__(self, params: SimulationParameters):
        # initialisation des seeds a  0 et du vent a 0
        self.params = params
        self.wind_mean = np.array([self.params.wind_mean, 0, 0])
        self.wind_values = np.zeros(shape=(params.n_samples, 3))

    def AddGust(self, Gust, GustTime):
        # cette fonction permet d'ajouter une gust sur la composante longitudinale
        # du signal de vent
        # self.WindValues[0,:]=self.WindValues[0,:]+Gust.GetGustSignal(self.params,GustTime)
        # Affichage (si decommente)
        # time=[0.0]*self.SimulationParameters.NSamples # def du vecteur de temps pour affichage
        # for i in range(len(time)):
        #    time[i]=float(i)*self.SimulationParameters.SampleTime
        # fig=plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(time,self.WindValues[0,:])
        # plt.show()
        pass

    def compute(self):
        # Création d'un spectre si aucun n'est donné.
        spectrum = Spectrum(self.params)

        # initialisation du bruit blanc - phase random entre [0, 2pi].
        fft_seed = np.exp(
            1.0j * np.random.uniform(0, 2 * np.pi, size=(spectrum.N_freq, 3))
        )

        # Multiplication du spectre de la seed, par le spectre (discret) du vent
        filtered = np.multiply(fft_seed, np.sqrt(spectrum.array.astype(np.complex128)))

        # On a besoin de reconstruire un tableau représentant une fft réelle.
        full = np.vstack(
            [
                np.zeros(shape=(1, 3)),
                filtered[:-1, :],
                np.real(filtered[-1, :]),
                np.conjugate(filtered[-2::-1, :]),
            ]
        )
        self.wind_values = (
            np.real(np.fft.ifft(full, axis=0))
            * np.sqrt(spectrum.N_freq / self.params.sample_time)
            + self.wind_mean
        )
