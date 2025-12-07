from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    """
    Interfaz base para todas las funciones objetivo del framework.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        Evalúa la función objetivo en la posición x.
        """
        pass

    @property
    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (lower_bounds, upper_bounds) como ndarrays.
        """
        pass

    @property
    def name(self) -> str:
        """
        Nombre de la función (útil para logs y experimentos).
        """
        return self.__class__.__name__
    



class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim: int):
        self.dim = dim
        self._lower = np.full(dim, -5.12)
        self._upper = np.full(dim,  5.12)

    def __call__(self, x: np.ndarray) -> float:
        """
        Función de Rastringin facilitada por el ejercicio para evaluar una solución
        """
        x = np.asarray(x, dtype=float)
        #assert all(abs(xi)<=5.12 for xi in x)
        A = 10.0
        return A * len(x) + sum([ (xi**2 - A * np.cos(2*np.pi*xi)) for xi in x ])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper
