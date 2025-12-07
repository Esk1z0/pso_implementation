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

if __name__ == "__main__":
    print("Probando VelocityClamp...")

    # ============================================================
    # 1. DimensionalClamp
    # ============================================================
    print(" - Probando DimensionalClamp...")

    clamp_dim = DimensionalClamp(vmax=1.0)

    v = np.array([2.0, -3.0, 0.5])
    v_clamped = clamp_dim.apply(v)

    assert np.allclose(v_clamped, np.array([1.0, -1.0, 0.5]))
    print("   * Clip por dimensión OK")

    # Prueba: dentro del rango → no cambiar
    v2 = np.array([0.3, -0.7, 0.9])
    v2_clamped = clamp_dim.apply(v2)
    assert np.allclose(v2_clamped, v2)
    print("   * No modifica velocidades dentro del rango OK")


    # ============================================================
    # 2. NormClamp
    # ============================================================
    print(" - Probando NormClamp...")

    clamp_norm = NormClamp(vmax=1.0)

    v = np.array([3.0, 0.0, 0.0])  # norma 3 → se debe escalar a 1
    v_clamped = clamp_norm.apply(v)

    assert np.allclose(v_clamped, np.array([1.0, 0.0, 0.0]))
    print("   * Reducir norma correctamente OK")

    # Caso dentro del límite → no modificar
    v2 = np.array([0.3, 0.4, 0.1])
    assert np.linalg.norm(v2) < 1.0
    v2_clamped = clamp_norm.apply(v2)
    assert np.allclose(v2_clamped, v2)
    print("   * No modifica velocidad si ya está dentro del límite OK")

    # Caso norma = 0 → no debe dar error ni modificar nada
    v_zero = np.zeros(3)
    assert np.allclose(clamp_norm.apply(v_zero), v_zero)
    print("   * Caso norma cero OK")


    # ============================================================
    # 3. DampeningClamp
    # ============================================================
    print(" - Probando DampeningClamp...")

    clamp_damp = DampeningClamp(vmax=1.0, factor=0.5)

    v = np.array([2.0, 0.0, 0.0])  # norma 2.0 > 1.0 → aplicar dampening
    v_damped = clamp_damp.apply(v)

    # Debe ser v * factor = [1.0, 0, 0]
    assert np.allclose(v_damped, np.array([1.0, 0.0, 0.0]))
    print("   * Amortiguación OK")

    # Rango nominal → no modificar
    v2 = np.array([0.6, 0.2, 0.1])  # norma < 1.0
    v2_damped = clamp_damp.apply(v2)
    assert np.allclose(v2_damped, v2)
    print("   * No amortigua si velocidad dentro de límite OK")

    print("Todas las pruebas de VelocityClamp han pasado correctamente.")
