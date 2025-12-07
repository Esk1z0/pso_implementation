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
    

if __name__ == "__main__":
    print("Probando FixedParameterSchedule...")

    w, c1, c2 = 0.7, 1.4, 1.6
    schedule = FixedParameterSchedule(w, c1, c2)

    # 1. Comprobar resultados en varias iteraciones
    for it in [0, 5, 10, 999]:
        ww, cc1, cc2 = schedule.get(it, 1000)
        assert ww == w
        assert cc1 == c1
        assert cc2 == c2
    print(" - Devuelve siempre los mismos parámetros OK")

    # 2. Comprobar el nombre
    assert schedule.name == "FixedParameterSchedule"
    print(" - Nombre OK")

    # 3. Valores correctos y tipo de retorno
    params = schedule.get(5, 100)
    assert isinstance(params, tuple)
    assert len(params) == 3
    print(" - Tipo de retorno OK")

    print("Todas las pruebas de FixedParameterSchedule han pasado correctamente.")