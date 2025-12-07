import numpy as np

class Particle:
    """
    Representa una partícula de PSO con lo mínimo imprescindible:
    - posición actual
    - velocidad actual
    - fitness actual
    - mejor posición personal (pbest) y su fitness
    """

    def __init__(self, position, velocity, fitness: float):
        # Convertimos a np.ndarray por robustez
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.fitness = float(fitness)

        # Mejor personal inicial = estado inicial
        self.pbest_pos = self.position.copy()
        self.pbest_fit = self.fitness

    def update_personal_best(self) -> None:
        """
        Actualiza el mejor personal si el fitness actual es mejor.
        Debes llamar a esto después de evaluar la función objetivo.
        """
        if self.fitness < self.pbest_fit:
            self.pbest_fit = self.fitness
            self.pbest_pos = self.position.copy()
