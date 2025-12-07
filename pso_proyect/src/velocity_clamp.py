from abc import ABC, abstractmethod
import numpy as np


class VelocityClamp(ABC):
    """
    Interfaz para estrategias de control de velocidad en PSO.
    """

    @abstractmethod
    def apply(self, velocity: np.ndarray) -> np.ndarray:
        """
        velocity : vector de velocidad actual de la partícula.
        """
        ...


class DimensionalClamp(VelocityClamp):
    """
    Limita cada componente de la velocidad de forma independiente
    al rango [-vmax, vmax].
    """

    def __init__(self, vmax: float | np.ndarray):
        """
        vmax : valor máximo absoluto permitido para cada componente de la velocidad.
               Si es escalar, se aplica el mismo límite a todas las dimensiones.
        """
        self.vmax = vmax

    def apply(self, velocity: np.ndarray) -> np.ndarray:
        if np.isscalar(self.vmax):
            vmax_vec = np.full_like(velocity, fill_value=float(self.vmax))
        else:
            vmax_vec = np.asarray(self.vmax, dtype=float)
        return np.clip(velocity, -vmax_vec, vmax_vec)


class NormClamp(VelocityClamp):
    """
    Limita la norma de la velocidad: si ||v|| > vmax, se reescala v
    manteniendo la dirección pero con norma igual a vmax.
    """

    def __init__(self, vmax: float):
        """
        vmax : norma máxima permitida para el vector velocidad.
        """
        self.vmax = float(vmax)

    def apply(self, velocity: np.ndarray) -> np.ndarray:
        v = np.asarray(velocity, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0.0 or norm <= self.vmax:
            return v
        return v * (self.vmax / norm)


class DampeningClamp(VelocityClamp):
    """
    Aplica amortiguación a la velocidad cuando supera un umbral:
    si ||v|| > vmax, se multiplica por factor (< 1).
    """

    def __init__(self, vmax: float, factor: float):
        """
        vmax   : norma a partir de la cual se aplica la amortiguación.
        factor : factor de reducción de la velocidad (0 < factor <= 1).
        """
        self.vmax = float(vmax)
        self.factor = float(factor)

    def apply(self, velocity: np.ndarray) -> np.ndarray:
        v = np.asarray(velocity, dtype=float)
        norm = np.linalg.norm(v)
        if norm > self.vmax:
            return v * self.factor
        return v
