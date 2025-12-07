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

if __name__ == "__main__":
    print("Probando RastriginFunction...")

    dim = 3
    f = RastriginFunction(dim=dim)

    # 1. Comprobar bounds
    lower, upper = f.bounds
    assert lower.shape == (dim,)
    assert upper.shape == (dim,)
    assert np.allclose(lower, -5.12)
    assert np.allclose(upper,  5.12)
    print(" - Bounds OK")

    # 2. Óptimo en x = 0 -> valor 0
    x0 = np.zeros(dim)
    val0 = f(x0)
    assert abs(val0 - 0.0) < 1e-12
    print(" - Óptimo en x=0 OK")

    # 3. Valor conocido para [1, 2]
    # Rastrigin([1, 2]) = 10*2 + (1^2 - 10*cos(2π)) + (2^2 - 10*cos(4π))
    #                    = 20 + (1 - 10*1) + (4 - 10*1) = 5
    x_test = np.array([1.0, 2.0])
    f_test = RastriginFunction(dim=2)
    val_test = f_test(x_test)
    assert abs(val_test - 5.0) < 1e-12
    print(" - Evaluación en [1, 2] OK")

    # 4. Comprobar name
    assert f.name == "RastriginFunction"
    print(" - Name OK")

    print("Todas las pruebas de RastriginFunction han pasado correctamente.")