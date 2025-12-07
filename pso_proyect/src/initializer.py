from abc import ABC, abstractmethod
import numpy as np

from particle import Particle


class Initializer(ABC):
    """
    Interfaz para estrategias de inicialización del enjambre en PSO.
    """

    @abstractmethod
    def init_swarm(self, swarm_size: int, dim: int, bounds) -> list[Particle]:
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

if __name__ == "__main__":
    print("Probando RandomUniformInitializer...")

    initializer = RandomUniformInitializer(vmax_fraction=0.2)

    swarm_size = 5
    dim = 3
    lower = np.array([-5.0, -2.0, 0.0])
    upper = np.array([5.0,  3.0, 10.0])

    particles = initializer.init_swarm(
        swarm_size=swarm_size,
        dim=dim,
        bounds=(lower, upper)
    )

    # 1. Comprobar número de partículas
    assert len(particles) == swarm_size
    print(" - Número de partículas OK")

    # 2. Comprobar que todas las posiciones están dentro de los límites
    for p in particles:
        assert (p.position >= lower).all() and (p.position <= upper).all()
    print(" - Posiciones dentro de los límites OK")

    # 3. Comprobar velocidades dentro del límite calculado
    domain = upper - lower
    vmax = 0.2 * domain

    for p in particles:
        assert (p.velocity >= -vmax).all() and (p.velocity <= vmax).all()
    print(" - Velocidades dentro de los límites OK")

    # 4. Comprobar que fitness es None
    for p in particles:
        assert p.fitness is None
    print(" - Fitness inicial es None OK")

    print("Todas las pruebas de RandomUniformInitializer han pasado correctamente.")
