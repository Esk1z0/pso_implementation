import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import csv

from objective_functions import RastriginFunction
from initializer import RandomUniformInitializer, NormalAroundPointInitializer, Initializer
from boundary_handler import ClipBoundaryHandler
from velocity_clamp import DimensionalClamp
from parameter_schedule import FixedParameterSchedule
from stopping_criterion import MaxIterationsCriterion
from improved_pso import run_pso_improved  # ajusta el nombre del módulo si es otro


class ExperimentRunner:
    """
    Clase sencilla para lanzar experimentos de PSO con la configuración fija
    que estamos usando (Rastrigin + PSO gbest clásico) y guardar resultados.
    """

    def __init__(self, base_output_dir: str = "results_pso"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Ejecutar un experimento
    # ------------------------------------------------------------------
    def run_experiment(
        self,
        config: Dict[str, Any],
        initializer: Initializer | None = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta un único experimento de PSO con la config indicada.

        Config esperada (claves principales):
          - experiment_name : nombre de la carpeta del experimento
          - dim             : dimensión de Rastrigin
          - swarm_size      : número de partículas
          - max_iter        : iteraciones máximas
          - w, c1, c2       : parámetros de PSO
          - vmax            : límite de velocidad para DimensionalClamp
          - initializer_vmax_fraction : fracción para la velocidad inicial
          - seed            : semilla opcional para reproducibilidad

        Si 'initializer' es None, se usa RandomUniformInitializer por defecto.
        """
        experiment_name = config.get("experiment_name", "experiment")
        exp_dir = self.base_output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- Configuración fija + overrides ----------------
        dim = int(config.get("dim", 5))
        swarm_size = int(config.get("swarm_size", 30))
        max_iter = int(config.get("max_iter", 200))

        w = float(config.get("w", 0.7))
        c1 = float(config.get("c1", 1.4))
        c2 = float(config.get("c2", 1.4))

        vmax = float(config.get("vmax", 0.5))
        initializer_vmax_fraction = float(config.get("initializer_vmax_fraction", 0.2))

        seed = config.get("seed", None)
        rng = np.random.default_rng(seed) if seed is not None else None

        # ---------------- LOG DE INICIO ----------------
        print("\n" + "=" * 70)
        print(f"[PSO] Iniciando experimento: {experiment_name}")
        print(f"[PSO] Carpeta de salida: {exp_dir}")
        print(f"[PSO] Parámetros principales:")
        print(f"      dim={dim}, swarm_size={swarm_size}, max_iter={max_iter}")
        print(f"      w={w}, c1={c1}, c2={c2}, vmax={vmax}")
        print(f"      initializer_vmax_fraction={initializer_vmax_fraction}")
        print(f"      seed={seed}")
        print("=" * 70)

        # ---------------- Construcción de componentes ----------------
        objective = RastriginFunction(dim)

        # ---------------- Selección del Initializer ----------------
        if initializer is None:
            init_type = config.get("initializer_type", "uniform")

            if init_type == "uniform":
                initializer = RandomUniformInitializer(
                    vmax_fraction=initializer_vmax_fraction,
                    rng=rng,
                )
                init_name = "RandomUniformInitializer"

            elif init_type == "normal":
                center = np.asarray(
                    config.get("initializer_center", [0.0] * dim),
                    dtype=float
                )
                sigma = float(config.get("initializer_std", 0.5))  # <- nombre correcto

                initializer = NormalAroundPointInitializer(
                    center=center,
                    sigma=sigma,
                    vmax_fraction=initializer_vmax_fraction,
                    rng=rng,
                )
                init_name = f"NormalAroundPointInitializer(center={center.tolist()}, sigma={sigma})"

            else:
                raise ValueError(f"initializer_type desconocido: {init_type}")

        else:
            init_name = initializer.__class__.__name__



        boundary_handler = ClipBoundaryHandler()
        velocity_clamp = DimensionalClamp(vmax=vmax)
        parameter_schedule = FixedParameterSchedule(w=w, c1=c1, c2=c2)
        stopping_criterion = MaxIterationsCriterion()

        print(f"[PSO] Initializer: {init_name}")
        print(f"[PSO] BoundaryHandler: {boundary_handler.__class__.__name__}")
        print(f"[PSO] VelocityClamp: {velocity_clamp.__class__.__name__}")
        print(f"[PSO] ParameterSchedule: {parameter_schedule.__class__.__name__}")
        print(f"[PSO] StoppingCriterion: {stopping_criterion.__class__.__name__}")
        print("-" * 70)

        # ---------------- Ejecución del PSO ----------------
        t0 = time.perf_counter()
        pso_results = run_pso_improved(
            objective=objective,
            initializer=initializer,
            boundary_handler=boundary_handler,
            velocity_clamp=velocity_clamp,
            parameter_schedule=parameter_schedule,
            stopping_criterion=stopping_criterion,
            swarm_size=swarm_size,
            max_iter=max_iter,
            rng=rng,
        )
        elapsed = time.perf_counter() - t0

        # Metadatos básicos
        pso_results["config"] = config
        pso_results["runtime_seconds"] = elapsed
        pso_results["iterations_executed"] = len(pso_results["best_history"]) - 1

        # Guardar configuración, resumen y métricas en CSV
        config_path = exp_dir / "config.json"
        results_txt_path = exp_dir / "results.txt"
        metrics_csv_path = exp_dir / "metrics.csv"
        trajectories_csv_path = exp_dir / "trajectories.csv"

        self._save_config(config, config_path)
        self.save_results(pso_results, results_txt_path)
        self._save_metrics_csv(pso_results, metrics_csv_path)
        self._save_trajectories_csv(pso_results, trajectories_csv_path)

        # LOG (si estás usando el logging que añadimos antes)
        print(f"[PSO] Ficheros generados:")
        print(f"      - {config_path}")
        print(f"      - {results_txt_path}")
        print(f"      - {metrics_csv_path}")
        print(f"      - {trajectories_csv_path}")
        print("=" * 70 + "\n")

        return pso_results


    # ------------------------------------------------------------------
    # 2) Ejecutar varios trials del mismo experimento
    # ------------------------------------------------------------------
    def run_multiple_trials(
        self,
        n_trials: int,
        config: Dict[str, Any],
        initializer: Initializer | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el mismo experimento n_trials veces, cambiando la semilla.
        Crea una carpeta por trial dentro del experimento.

        Si se pasa un 'initializer', se reutiliza el mismo objeto en todos
        los trials (útil para NormalAroundPointInitializer, por ejemplo).
        """
        experiment_name = config.get("experiment_name", "experiment")
        base_seed = config.get("seed", 1234)

        print("\n" + "#" * 70)
        print(f"[PSO] Lanzando {n_trials} trials para el experimento base '{experiment_name}'")
        print(f"[PSO] Carpeta base de resultados: {self.base_output_dir}")
        print("#" * 70 + "\n")

        all_results: List[Dict[str, Any]] = []

        for i in range(n_trials):
            trial_seed = base_seed + i if base_seed is not None else None
            cfg = dict(config)
            cfg["seed"] = trial_seed
            cfg["experiment_name"] = f"{experiment_name}_trial_{i:03d}"

            print(f"→ Ejecutando trial {i+1}/{n_trials} (seed={trial_seed})...")
            res = self.run_experiment(cfg, initializer=initializer)
            res["trial_index"] = i
            all_results.append(res)

        print("\n" + "#" * 70)
        print(f"[PSO] Todos los trials de '{experiment_name}' han finalizado.")
        print(f"[PSO] Directorio raíz de resultados: {self.base_output_dir}")
        print("#" * 70 + "\n")

        return all_results


    # ------------------------------------------------------------------
    # 3) Comparar parámetros (lista de configs)
    # ------------------------------------------------------------------
    def compare_parameters(
        self,
        param_grid: List[Dict[str, Any]],
        n_trials: int = 3,
        summary_filename: str = "summary_compare.txt",
    ) -> Dict[str, Any]:
        """
        Recibe una lista de configs (por ejemplo variando w, c1, c2)
        y ejecuta varios trials por cada una.
        Genera un resumen estadístico simple en un txt.
        """
        summary: Dict[str, Any] = {}
        lines: List[str] = []

        for cfg in param_grid:
            name = cfg.get("experiment_name", "config")
            print(f"\n===== EXPERIMENTO: {name} =====")
            results = self.run_multiple_trials(n_trials=n_trials, config=cfg)

            final_fits = [r["gbest_fit"] for r in results]
            runtimes = [r["runtime_seconds"] for r in results]

            stats = {
                "mean_final_fit": float(np.mean(final_fits)),
                "std_final_fit": float(np.std(final_fits)),
                "min_final_fit": float(np.min(final_fits)),
                "max_final_fit": float(np.max(final_fits)),
                "mean_runtime": float(np.mean(runtimes)),
                "std_runtime": float(np.std(runtimes)),
            }
            summary[name] = stats

            line = (
                f"{name}: "
                f"mean_fit={stats['mean_final_fit']:.6f}, "
                f"std_fit={stats['std_final_fit']:.6f}, "
                f"min_fit={stats['min_final_fit']:.6f}, "
                f"max_fit={stats['max_final_fit']:.6f}, "
                f"mean_time={stats['mean_runtime']:.4f}s"
            )
            print(line)
            lines.append(line)

        # Guardar resumen global
        summary_path = self.base_output_dir / summary_filename
        with open(summary_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        return summary

    # ------------------------------------------------------------------
    # 4) Guardar resultados en TXT (resumen organizado)
    # ------------------------------------------------------------------
    def save_results(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Guarda en un TXT los datos principales del entrenamiento
        de forma legible y mínimamente organizada.
        """
        bh = results["best_history"]
        avg_vel = results["avg_velocity_history"]
        div = results["diversity_history"]
        num_imp = results["num_improving_history"]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== RESUMEN EXPERIMENTO PSO ===\n\n")
            f.write(f"Mejor fitness final: {results['gbest_fit']:.6f}\n")
            f.write(f"Mejor posición final: {results['gbest_pos']}\n\n")

            f.write(f"Iteraciones ejecutadas: {results['iterations_executed']}\n")
            f.write(f"Tiempo total (s): {results['runtime_seconds']:.6f}\n\n")

            f.write("---- Historial de mejor fitness ----\n")
            f.write(f"  Valor inicial: {bh[0]:.6f}\n")
            f.write(f"  Valor final:   {bh[-1]:.6f}\n")
            f.write(f"  Mínimo global alcanzado en el historial: {min(bh):.6f}\n\n")

            if avg_vel:
                f.write("---- Velocidad media ----\n")
                f.write(f"  Velocidad media inicial: {avg_vel[0]:.6f}\n")
                f.write(f"  Velocidad media final:   {avg_vel[-1]:.6f}\n\n")

            if div:
                f.write("---- Diversidad ----\n")
                f.write(f"  Diversidad inicial: {div[0]:.6f}\n")
                f.write(f"  Diversidad final:   {div[-1]:.6f}\n\n")

            if num_imp:
                f.write("---- Mejoras por iteración ----\n")
                f.write(f"  Media de partículas que mejoran: {np.mean(num_imp):.2f}\n")
                f.write(f"  Máximo de partículas que mejoran: {np.max(num_imp)}\n")
                f.write(
                    f"  Veces que nadie mejora (0 mejoras): "
                    f"{sum(1 for x in num_imp if x == 0)}\n\n"
                )

            f.write("---- Parámetros usados ----\n")
            cfg = results.get("config", {})
            for k, v in cfg.items():
                f.write(f"  {k}: {v}\n")

    # ------------------------------------------------------------------
    # Utilidades internas: guardar config y métricas
    # ------------------------------------------------------------------
    def _save_config(self, config: Dict[str, Any], filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _save_metrics_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Guarda en un CSV las series temporales principales del experimento:
        - iter
        - best_fit (mejor global tras esa iteración)
        - avg_velocity
        - diversity
        - num_improving
        - w, c1, c2
        """
        best_history = results["best_history"]          # len = iters + 1 (incluye inicial)
        avg_vel_hist = results["avg_velocity_history"]  # len = iters
        div_hist = results["diversity_history"]         # len = iters
        imp_hist = results["num_improving_history"]     # len = iters
        w_hist = results["w_history"]
        c1_hist = results["c1_history"]
        c2_hist = results["c2_history"]

        n_iters = len(best_history) - 1  # número real de iteraciones ejecutadas

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Cabecera
            writer.writerow([
                "iter",
                "best_fit",
                "avg_velocity",
                "diversity",
                "num_improving",
                "w",
                "c1",
                "c2",
            ])

            for it in range(n_iters):
                best_fit = best_history[it + 1]  # después de aplicar la iteración it

                avg_v = avg_vel_hist[it] if it < len(avg_vel_hist) else ""
                div = div_hist[it] if it < len(div_hist) else ""
                num_imp = imp_hist[it] if it < len(imp_hist) else ""

                w = w_hist[it] if it < len(w_hist) else ""
                c1 = c1_hist[it] if it < len(c1_hist) else ""
                c2 = c2_hist[it] if it < len(c2_hist) else ""

                writer.writerow([
                    it,
                    best_fit,
                    avg_v,
                    div,
                    num_imp,
                    w,
                    c1,
                    c2,
                ])

    def _save_trajectories_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        """
        Guarda en un CSV las trayectorias completas de todas las partículas.

        Formato:
        - Columna 'iter' : índice de iteración (0 = estado inicial)
        - Columnas restantes: p{i}_x{d} para cada partícula i y dimensión d.
        """
        positions_history = results.get("swarm_positions_history", None)
        if positions_history is None or len(positions_history) == 0:
            return  # nada que guardar

        # positions_history[k] tiene shape (swarm_size, dim)
        last_snapshot = positions_history[-1]
        swarm_size, dim = last_snapshot.shape

        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Cabecera
            header = ["iter"]
            for i in range(swarm_size):
                for d in range(dim):
                    header.append(f"p{i}_x{d}")
            writer.writerow(header)

            # Filas: una por iteración
            for it, snapshot in enumerate(positions_history):
                # snapshot: (swarm_size, dim)
                row = [it]
                # aplanamos en el orden p0_x0, p0_x1, ..., p1_x0, p1_x1, ...
                for i in range(swarm_size):
                    for d in range(dim):
                        row.append(snapshot[i, d])
                writer.writerow(row)