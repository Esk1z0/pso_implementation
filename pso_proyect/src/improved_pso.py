import numpy as np

from objective_functions import ObjectiveFunction
from initializer import Initializer
from boundary_handler import BoundaryHandler
from velocity_clamp import VelocityClamp
from parameter_schedule import ParameterSchedule
from stopping_criterion import StoppingCriterion


def run_pso_improved(
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
    # ---------------------------- Inicialización ----------------------------
    if rng is None:
        rng = np.random.default_rng()

    lower_bounds, upper_bounds = objective.bounds
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    dim = lower_bounds.size

    particles = initializer.init_swarm(
        swarm_size=swarm_size,
        dim=dim,
        bounds=(lower_bounds, upper_bounds),
    )

    # Evaluación inicial
    for p in particles:
        p.fitness = objective(p.position)
        p.pbest_pos = p.position.copy()
        p.pbest_fit = p.fitness

    best_particle = min(particles, key=lambda p: p.pbest_fit)
    gbest_pos = best_particle.pbest_pos.copy()
    gbest_fit = best_particle.pbest_fit

    # ---------------------------- Métricas iniciales --------------------------
    best_history = [gbest_fit]
    gbest_positions = [gbest_pos.copy()]
    avg_velocity_history = []
    diversity_history = []
    improvements_history = []
    w_history, c1_history, c2_history = [], [], []

    # NUEVO: histórico de posiciones del enjambre
    # Cada elemento: array de shape (swarm_size, dim)
    swarm_positions_history = [
        np.stack([p.position.copy() for p in particles])
    ]

    # ---------------------------- Bucle principal -----------------------------
    for it in range(max_iter):
        w, c1, c2 = parameter_schedule.get(it, max_iter)

        # Guardar pbest ANTES de actualizar
        previous_pbests = [p.pbest_fit for p in particles]

        # ----------- 1. ACTUALIZACIÓN DE POSICIONES Y FITNESS (sin pbest) ----------
        for p in particles:
            r1 = rng.random(dim)
            r2 = rng.random(dim)

            cognitive = c1 * r1 * (p.pbest_pos - p.position)
            social    = c2 * r2 * (gbest_pos - p.position)
            new_velocity = w * p.velocity + cognitive + social
            new_velocity = velocity_clamp.apply(new_velocity)
            new_position = p.position + new_velocity

            new_position, new_velocity = boundary_handler(
                new_position, new_velocity, lower_bounds, upper_bounds
            )

            p.position = new_position
            p.velocity = new_velocity
            p.fitness = objective(p.position)

        # ----------- 2. CALCULAR MEJORAS (antes de actualizar pbest) ----------
        num_improving = sum(
            p.fitness < previous_pbests[i]
            for i, p in enumerate(particles)
        )
        improvements_history.append(num_improving)

        # ----------- 3. AHORA SÍ: ACTUALIZAR PBESTS ----------
        for p in particles:
            p.update_personal_best()

        # ----------- 4. ACTUALIZAR GBEST ----------
        best_particle = min(particles, key=lambda p: p.pbest_fit)
        gbest_pos = best_particle.pbest_pos.copy()
        gbest_fit = best_particle.pbest_fit
        best_history.append(gbest_fit)

        # ----------- 5. MÉTRICAS ----------
        w_history.append(w)
        c1_history.append(c1)
        c2_history.append(c2)

        gbest_positions.append(gbest_pos.copy())

        avg_vel = np.mean([np.linalg.norm(p.velocity) for p in particles])
        avg_velocity_history.append(avg_vel)

        diversity = np.mean([
            np.linalg.norm(p.position - gbest_pos)
            for p in particles
        ])
        diversity_history.append(diversity)

        # NUEVO: guardar snapshot de posiciones del enjambre en esta iteración
        swarm_positions_history.append(
            np.stack([p.position.copy() for p in particles])
        )

        # ----------- 6. PARADA ----------
        if stopping_criterion.should_stop(it, max_iter, gbest_fit):
            break

    # ----------------------------- Resultados ------------------------------
    return {
        "gbest_pos": gbest_pos,
        "gbest_fit": gbest_fit,
        "best_history": best_history,
        "gbest_positions": gbest_positions,
        "avg_velocity_history": avg_velocity_history,
        "diversity_history": diversity_history,
        "num_improving_history": improvements_history,
        "w_history": w_history,
        "c1_history": c1_history,
        "c2_history": c2_history,
        # NUEVO: histórico completo de posiciones del enjambre
        "swarm_positions_history": swarm_positions_history,
    }


if __name__=="__main__":
    print("\n===================== PSO IMPROVED TEST =====================\n")

    # --- Importaciones locales para evitar dependencias circulares ---
    from objective_functions import RastriginFunction
    from initializer import RandomUniformInitializer
    from boundary_handler import ClipBoundaryHandler
    from velocity_clamp import DimensionalClamp
    from parameter_schedule import FixedParameterSchedule
    from stopping_criterion import MaxIterationsCriterion

    # ---------------- CONFIGURACIÓN DEL EXPERIMENTO ----------------
    dim = 2
    swarm_size = 30
    max_iter = 20

    objective = RastriginFunction(dim)
    initializer = RandomUniformInitializer(vmax_fraction=0.2)
    boundary_handler = ClipBoundaryHandler()
    velocity_clamp = DimensionalClamp(vmax=0.5)
    schedule = FixedParameterSchedule(w=0.7, c1=1.4, c2=1.4)
    stopping = MaxIterationsCriterion()

    rng = np.random.default_rng(42)

    print("Ejecutando PSO mejorado sobre Rastrigin...\n")

    # ---------------- EJECUCIÓN DEL PSO ----------------
    results = run_pso_improved(
        objective=objective,
        initializer=initializer,
        boundary_handler=boundary_handler,
        velocity_clamp=velocity_clamp,
        parameter_schedule=schedule,
        stopping_criterion=stopping,
        swarm_size=swarm_size,
        max_iter=max_iter,
        rng=rng
    )

    # ---------------- RESUMEN DE RESULTADOS ----------------
    print("=============== RESULTADOS DEL EXPERIMENTO ===============")
    print(f"Mejor valor encontrado (gbest_fit): {results['gbest_fit']:.6f}")
    print(f"Mejor posición encontrada (gbest_pos):\n {results['gbest_pos']}\n")

    print("---- Estadísticas del historial ----")
    print(f"Valor inicial:   {results['best_history'][0]:.6f}")
    print(f"Valor final:     {results['best_history'][-1]:.6f}")
    print(f"Iteraciones ejecutadas: {len(results['best_history']) - 1}")

    if len(results['avg_velocity_history']):
        print(f"Velocidad promedio final: {results['avg_velocity_history'][-1]:.6f}")
    if len(results['diversity_history']):
        print(f"Diversidad final: {results['diversity_history'][-1]:.6f}")
    if len(results['num_improving_history']):
        print(f"Mejoras promedio por iteración: "
              f"{np.mean(results['num_improving_history']):.2f}")

    print("\n---- Historial de mejor fitness ----")
    for i, val in enumerate(results["best_history"]):
        print(f" iter {i:03d} -> {val:.6f}")

    print("\n=================== FIN DEL EXPERIMENTO ===================\n")
