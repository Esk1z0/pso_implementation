from abc import ABC, abstractmethod


class ParameterSchedule(ABC):
    """
    Interfaz para schedules de parámetros del PSO.
    Debe devolver (w, c1, c2) para cada iteración.
    """

    @abstractmethod
    def get(self, iter_idx: int, max_iter: int) -> tuple[float, float, float]:
        """
        Devuelve los parámetros (w, c1, c2) en la iteración actual.
        """
        pass

    @property
    def name(self) -> str:
        """Nombre del schedule (útil en logs/experimentos)."""
        return self.__class__.__name__
    
class FixedParameterSchedule(ParameterSchedule):
    """
    Devuelve siempre los mismos parámetros (w, c1, c2).
    """

    def __init__(self, w: float, c1: float, c2: float):
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def get(self, iter_idx: int, max_iter: int) -> tuple[float, float, float]:
        return self.w, self.c1, self.c2