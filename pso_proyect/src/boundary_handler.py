from abc import ABC, abstractmethod
import numpy as np


class BoundaryHandler(ABC):
    @abstractmethod
    def __call__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica la estrategia de tratamiento de bordes y devuelve
        (new_position, new_velocity).
        """
        pass


class ClipBoundaryHandler(BoundaryHandler):
    """
    Estrategia: clip + poner velocidad a 0 en dimensiones que tocan borde.
    """

    def __call__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Funcion clip
        new_position = np.minimum(np.maximum(position, lower_bounds), upper_bounds)

        # Máscara bolleana donde debemos poner velocidad a 0
        touched_lower = new_position == lower_bounds
        touched_upper = new_position == upper_bounds
        mask = touched_lower | touched_upper

        new_velocity = velocity.copy()
        new_velocity[mask] = 0.0

        return new_position, new_velocity

if __name__ == "__main__":
    # Pruebas básicas del ClipBoundaryHandler

    handler = ClipBoundaryHandler()

    lower = np.array([-5.0, -5.0])
    upper = np.array([5.0, 5.0])

    # Caso 1: posición dentro de los límites (no debería cambiar nada)
    pos = np.array([1.0, -2.0])
    vel = np.array([0.5, -0.5])
    new_pos, new_vel = handler(pos, vel, lower, upper)
    assert np.allclose(new_pos, pos)
    assert np.allclose(new_vel, vel)
    print("Caso 1 OK: dentro de límites.")

    # Caso 2: posición por debajo del límite inferior
    pos = np.array([-10.0, 0.0])
    vel = np.array([1.0, 2.0])
    new_pos, new_vel = handler(pos, vel, lower, upper)
    # Primera componente se clipa a -5.0 y su velocidad pasa a 0
    assert np.allclose(new_pos, np.array([-5.0, 0.0]))
    assert np.allclose(new_vel, np.array([0.0, 2.0]))
    print("Caso 2 OK: clip en límite inferior y velocidad a 0 en esa dimensión.")

    # Caso 3: posición por encima del límite superior
    pos = np.array([3.0, 10.0])
    vel = np.array([-1.0, -2.0])
    new_pos, new_vel = handler(pos, vel, lower, upper)
    # Segunda componente se clipa a 5.0 y su velocidad pasa a 0
    assert np.allclose(new_pos, np.array([3.0, 5.0]))
    assert np.allclose(new_vel, np.array([-1.0, 0.0]))
    print("Caso 3 OK: clip en límite superior y velocidad a 0 en esa dimensión.")

    print("Todas las pruebas de ClipBoundaryHandler han pasado correctamente.")