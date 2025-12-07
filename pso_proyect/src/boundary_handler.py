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
