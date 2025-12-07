import time
from abc import ABC, abstractmethod


class StoppingCriterion(ABC):
    """
    Interfaz base para criterios de parada en PSO.
    """

    @abstractmethod
    def should_stop(self, iter_idx: int, max_iter: int, current_best: float) -> bool:
        """
        Devuelve True si se debe parar en esta iteración.

        Parámetros
        ----------
        iter_idx : int
            Iteración actual (0-based).
        max_iter : int
            Máximo de iteraciones previsto.
        current_best : float
            Mejor valor global actual de la función objetivo.
        """
        ...



class MaxIterationsCriterion(StoppingCriterion):
    """
    Criterio de parada por número máximo de iteraciones.
    """

    def should_stop(self, iter_idx: int, max_iter: int, current_best: float) -> bool:
        return iter_idx >= max_iter



class TimeLimitCriterion(StoppingCriterion):
    """
    Criterio de parada por tiempo máximo (en segundos).
    """

    def __init__(self, max_seconds: float):
        self.max_seconds = float(max_seconds)
        self.start_time = time.time()

    def should_stop(self, iter_idx: int, max_iter: int, current_best: float) -> bool:
        elapsed = time.time() - self.start_time
        return elapsed >= self.max_seconds



class StagnationCriterion(StoppingCriterion):
    """
    Criterio de parada por estancamiento:
    si el mejor global no cambia durante patience iteraciones consecutivas,
    se detiene la optimización.
    """

    def __init__(self, patience: int, eps: float = 0.0):
        """
        patience : número de iteraciones consecutivas sin cambio del mejor global permitidas.
        eps      : tolerancia mínima para considerar que el mejor global ha cambiado.
        """
        self.patience = int(patience)
        self.eps = float(eps)

        self.best_so_far: float | None = None
        self.last_change_iter: int | None = None

    def should_stop(self, iter_idx: int, max_iter: int, current_best: float) -> bool:
        # Primera llamada: inicializamos estado
        if self.best_so_far is None:
            self.best_so_far = current_best
            self.last_change_iter = iter_idx
            return False

        # ¿Ha cambiado el mejor global?
        if abs(current_best - self.best_so_far) > self.eps:
            self.best_so_far = current_best
            self.last_change_iter = iter_idx
            return False

        # No ha cambiado: miramos cuántas iteraciones llevamos estancados
        if iter_idx - self.last_change_iter >= self.patience:
            return True

        return False


if __name__ == "__main__":
    print("Probando criterios de parada...")

    # ============================================================
    # 1. MaxIterationsCriterion
    # ============================================================
    print(" - Probando MaxIterationsCriterion...")
    max_iter_crit = MaxIterationsCriterion()

    assert not max_iter_crit.should_stop(iter_idx=0,  max_iter=10, current_best=0.0)
    assert not max_iter_crit.should_stop(iter_idx=5,  max_iter=10, current_best=0.0)
    assert max_iter_crit.should_stop(iter_idx=10, max_iter=10, current_best=0.0)
    assert max_iter_crit.should_stop(iter_idx=11, max_iter=10, current_best=0.0)

    print("   MaxIterationsCriterion OK")

    # ============================================================
    # 2. TimeLimitCriterion
    # ============================================================
    print(" - Probando TimeLimitCriterion...")

    time_crit = TimeLimitCriterion(max_seconds=0.1)
    assert not time_crit.should_stop(0, 100, 0.0)  # Just created → should not stop

    time.sleep(0.15)
    assert time_crit.should_stop(1, 100, 0.0)

    print("   TimeLimitCriterion OK")

    # ============================================================
    # 3. StagnationCriterion
    # ============================================================
    print(" - Probando StagnationCriterion...")

    stag_crit = StagnationCriterion(patience=3, eps=0.0)

    # Primera llamada → inicializa estado
    assert not stag_crit.should_stop(iter_idx=0, max_iter=999, current_best=10.0)

    # Cambia el valor → debe resetear last_change_iter
    assert not stag_crit.should_stop(iter_idx=1, max_iter=999, current_best=8.0)

    # Repite el mismo valor durante iteraciones < patience
    assert not stag_crit.should_stop(iter_idx=2, max_iter=999, current_best=8.0)
    assert not stag_crit.should_stop(iter_idx=3, max_iter=999, current_best=8.0)

    # Ahora superamos patience (=3)
    # last_change_iter quedó en iter=1
    # iter 4 → 4 - 1 = 3 → debería parar
    assert stag_crit.should_stop(iter_idx=4, max_iter=999, current_best=8.0)

    print("   StagnationCriterion OK")

    print("Todas las pruebas de criterios de parada han pasado correctamente.")
