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
        self.fitness  = None if fitness is None else float(fitness)

        # Mejor personal inicial = estado inicial
        self.pbest_pos = self.position.copy()
        self.pbest_fit = float("inf") if self.fitness is None else self.fitness

    def update_personal_best(self) -> None:
        """
        Actualiza el mejor personal si el fitness actual es mejor.
        Debes llamar a esto después de evaluar la función objetivo.
        """
        if self.fitness is None:
            return
        if self.fitness < self.pbest_fit:
            self.pbest_fit = self.fitness
            self.pbest_pos = self.position.copy()



if __name__ == "__main__":
    print("Probando clase Particle...")

    # 1. Conversión a ndarray
    p = Particle(position=[1, 2], velocity=[0.5, -0.3], fitness=None)
    assert isinstance(p.position, np.ndarray)
    assert isinstance(p.velocity, np.ndarray)
    print(" - Conversión a ndarray OK")

    # 2. pbest_fit = +inf si fitness inicial es None
    assert p.pbest_fit == float("inf")
    print(" - pbest_fit inicial OK")

    # 3. Primera evaluación: fitness real mejora +inf
    p.fitness = 10.0
    p.update_personal_best()
    assert p.pbest_fit == 10.0
    assert np.allclose(p.pbest_pos, p.position)
    print(" - Primera mejora del pbest OK")

    # 4. Empeorar fitness → pbest no cambia
    old_pbest_pos = p.pbest_pos.copy()
    old_pbest_fit = p.pbest_fit

    p.fitness = 15.0  # peor
    p.position += 1.0  # mover posición para ver si cambia indebidamente
    p.update_personal_best()

    assert p.pbest_fit == old_pbest_fit
    assert np.allclose(p.pbest_pos, old_pbest_pos)
    print(" - No actualizar pbest si fitness empeora OK")

    # 5. Mejorar fitness → pbest cambia
    p.fitness = 5.0
    p.update_personal_best()

    assert p.pbest_fit == 5.0
    assert np.allclose(p.pbest_pos, p.position)
    print(" - Actualizar pbest si fitness mejora OK")

    # 6. pbest_pos debe ser copia, no referencia
    pos_before = p.pbest_pos.copy()
    p.position += 10.0  # cambiar position
    assert np.allclose(p.pbest_pos, pos_before)  # debe seguir igual
    print(" - pbest_pos es copia independiente OK")

    print("Todas las pruebas de Particle han pasado correctamente.")