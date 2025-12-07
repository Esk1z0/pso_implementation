import numpy as np

from objective_functions import ObjectiveFunction
from particle import Particle
from initializer import Initializer
from boundary_handler import BoundaryHandler
from velocity_clamp import VelocityClamp
from parameter_schedule import ParameterSchedule
from stopping_criterion import StoppingCriterion


def run_pso(
    objective: ObjectiveFunction,
    initializer: Initializer,
    boundary_handler: BoundaryHandler,
    velocity_clamp: VelocityClamp,
    parameter_schedule: ParameterSchedule,
    stopping_criterion: StoppingCriterion,
    swarm_size: int,
    max_iter: int,
    rng: np.random.Generator | None = None,
):
    """
    Ejecuta un PSO gbest básico utilizando las interfaces definidas.

    objective           : función objetivo que implementa ObjectiveFunction.
    initializer         : estrategia de inicialización del enjambre.
    boundary_handler    : estrategia de tratamiento de bordes.
    velocity_clamp      : estrategia de control de velocidad.
    parameter_schedule  : estrategia de obtención de (w, c1, c2) por iteración.
    stopping_criterion  : criterio de parada.
    swarm_size          : número de partículas.
    max_iter            : máximo de iteraciones (límite duro).
    """
# ---------------------------- Inicializacion de variables ------------------------------
    # --- Generador de Números Aleatorios ---
    if rng is None:
        rng = np.random.default_rng()

    # --- Dimensiones y límites ---
    lower_bounds, upper_bounds = objective.bounds
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    dim = lower_bounds.size

    # --- Inicialización del enjambre ---
    particles = initializer.init_swarm(
                                        swarm_size=swarm_size,
                                        dim=dim,
                                        bounds=(lower_bounds, upper_bounds),
                                    )

    # --- Evaluación inicial del enjambre ---
    for p in particles:
        p.fitness = objective(p.position)
        p.pbest_pos = p.position.copy()
        p.pbest_fit = p.fitness

    # --- Inicialización del Mejor Global e Historial ---
    best_particle = min(particles, key=lambda p: p.pbest_fit)
    gbest_pos = best_particle.pbest_pos.copy()
    gbest_fit = best_particle.pbest_fit

    best_history = [gbest_fit]

# ---------------------------------- Bucle principal ------------------------------------
    for it in range(max_iter):
        w, c1, c2 = parameter_schedule.get(it, max_iter)

        # -------------------------- Bucle Actualización --------------------------------
        for p in particles:
            r1 = rng.random(dim)
            r2 = rng.random(dim)

            # --- Actualización de velocidad (gbest) ---
            cognitive = c1 * r1 * (p.pbest_pos - p.position)
            social    = c2 * r2 * (gbest_pos    - p.position)
            new_velocity = w * p.velocity + cognitive + social

            # Clamp de velocidad
            new_velocity = velocity_clamp.apply(new_velocity)

            # Nueva posición
            new_position = p.position + new_velocity

            # Tratamiento de bordes
            new_position, new_velocity = boundary_handler(
                new_position, new_velocity, lower_bounds, upper_bounds
            )

            # Actualizamos estado
            p.position = new_position
            p.velocity = new_velocity
            p.fitness = objective(p.position)
            p.update_personal_best()

        # -------------------------- Actualizar mejor global ----------------------------
        best_particle = min(particles, key=lambda p: p.pbest_fit)
        gbest_pos = best_particle.pbest_pos.copy()
        gbest_fit = best_particle.pbest_fit
        best_history.append(gbest_fit)

        # ---------------------------- Criterio de parada -------------------------------
        if stopping_criterion.should_stop(it, max_iter, gbest_fit):
            break

# resultados
    return gbest_pos, gbest_fit, best_history


if __name__ == "__main__":
    print("Probando run_pso (PSO completo)...")

    # Imports locales aquí para evitar dependencias circulares
    from objective_functions import RastriginFunction
    from initializer import RandomUniformInitializer
    from boundary_handler import ClipBoundaryHandler
    from velocity_clamp import DimensionalClamp
    from parameter_schedule import FixedParameterSchedule
    from stopping_criterion import MaxIterationsCriterion

    # ----------------------- Configuración del test -----------------------
    dim = 2
    objective = RastriginFunction(dim)

    initializer = RandomUniformInitializer(vmax_fraction=0.2)
    boundary_handler = ClipBoundaryHandler()
    velocity_clamp = DimensionalClamp(vmax=0.5)
    schedule = FixedParameterSchedule(w=0.7, c1=1.4, c2=1.4)
    stopping = MaxIterationsCriterion()

    swarm_size = 20
    max_iter = 200

    print(" - Ejecutando PSO gbest básico...")
    gbest_pos, gbest_fit, history = run_pso(
        objective=objective,
        initializer=initializer,
        boundary_handler=boundary_handler,
        velocity_clamp=velocity_clamp,
        parameter_schedule=schedule,
        stopping_criterion=stopping,
        swarm_size=swarm_size,
        max_iter=max_iter,
    )

    # ----------------------- Validaciones básicas ------------------------
    # 1. La historia debe tener al menos 1 elemento
    assert len(history) >= 1
    print("   * Historial no vacío OK")

    # 2. El mejor fitness debe ser un número real
    assert isinstance(gbest_fit, float)
    print("   * gbest_fit es float OK")

    # 3. La posición debe tener la dimensión correcta
    assert gbest_pos.shape == (dim,)
    print("   * Dimensión de gbest_pos OK")

    # 4. Comprobamos que el mejor fitness ha mejorado respecto al inicial
    initial_best = history[0]
    assert gbest_fit <= initial_best
    print("   * PSO ha mejorado o igualado el mejor valor inicial OK")

    # 5. Comprobación opcional: Rastrigin debe dar valor cercano a 0 si converge
    print(f"   * Valor final aproximado: {gbest_fit:.6f}")

    # ---------------- Mostrar historial completo ----------------
    print("\nHistorial de mejor global (best_history):")
    for i, val in enumerate(history):
        print(f" iter {i:03d}: {val:.6f}")
    

    print("PSO ejecutado correctamente.")
