from abc import ABC, abstractmethod
import numpy as np

from particle import Particle


class Initializer(ABC):
    """
    Interfaz para estrategias de inicialización del enjambre en PSO.
    """

    @abstractmethod
    def init_swarm(self, swarm_size: int, dim: int, bounds):
        """
        swarm_size : número de partículas del enjambre.
        dim        : dimensiones del espacio de búsqueda.
        bounds     : tupla (lower_bounds, upper_bounds), cada una np.ndarray.
        """


class RandomUniformInitializer(Initializer):
    """
    Inicializa las partículas de manera aleatoria uniforme dentro de los límites
    del espacio de búsqueda. Las velocidades se generan aleatoriamente dentro
    del rango proporcional al dominio.
    """

    def __init__(self, vmax_fraction: float = 0.2, rng: np.random.Generator | None = None):
        """
        vmax_fraction : fracción del tamaño del dominio usada como límite
                        máximo para la velocidad inicial.
        rng           : generador aleatorio (para reproducibilidad).
        """
        self.vmax_fraction = float(vmax_fraction)
        self.rng = rng if rng is not None else np.random.default_rng()

    def init_swarm(self, swarm_size: int, dim: int, bounds):
        lower, upper = bounds
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        domain = upper - lower
        vmax = self.vmax_fraction * domain

        particles = []
        for _ in range(swarm_size):
            pos = self.rng.uniform(lower, upper)
            vel = self.rng.uniform(-vmax, vmax)
            fit = None  # el PSO hará la evaluación después
            particles.append(Particle(pos, vel, fit))

        return particles

