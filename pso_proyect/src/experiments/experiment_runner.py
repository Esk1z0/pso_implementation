import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from objective_functions import RastriginFunction
from initializer import RandomUniformInitializer
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
    def run_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
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

        # ---------------- Construcción de componentes ----------------
        objective = RastriginFunction(dim)
        initializer = RandomUniformInitializer(
            vmax_fraction=initializer_vmax_fraction,
            rng=rng,
        )
        boundary_handler = ClipBoundaryHandler()
        velocity_clamp = DimensionalClamp(vmax=vmax)
        parameter_schedule = FixedParameterSchedule(w=w, c1=c1, c2=c2)
        stopping_criterion = MaxIterationsCriterion()

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

        # Guardar configuración y resultados
        self._save_config(config, exp_dir / "config.json")
        self.save_results(pso_results, exp_dir / "results.txt")
        self._save_plots(pso_results, exp_dir)

        return pso_results

    # ------------------------------------------------------------------
    # 2) Ejecutar varios trials del mismo experimento
    # ------------------------------------------------------------------
    def run_multiple_trials(
        self,
        n_trials: int,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el mismo experimento n_trials veces, cambiando la semilla.
        Crea una carpeta por trial dentro del experimento.
        """
        experiment_name = config.get("experiment_name", "experiment")
        base_seed = config.get("seed", 1234)

        all_results: List[Dict[str, Any]] = []

        for i in range(n_trials):
            trial_seed = base_seed + i if base_seed is not None else None
            cfg = dict(config)
            cfg["seed"] = trial_seed
            cfg["experiment_name"] = f"{experiment_name}_trial_{i:03d}"

            print(f"→ Ejecutando trial {i+1}/{n_trials} (seed={trial_seed})...")
            res = self.run_experiment(cfg)
            res["trial_index"] = i
            all_results.append(res)

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
                f.write(f"  Veces que nadie mejora (0 mejoras): "
                        f"{sum(1 for x in num_imp if x == 0)}\n\n")

            f.write("---- Parámetros usados ----\n")
            cfg = results.get("config", {})
            for k, v in cfg.items():
                f.write(f"  {k}: {v}\n")

    # ------------------------------------------------------------------
    # Utilidades internas: guardar config y plots
    # ------------------------------------------------------------------
    def _save_config(self, config: Dict[str, Any], filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _save_plots(self, results: Dict[str, Any], out_dir: Path) -> None:
        """
        Genera gráficas básicas:
          - best_history
          - avg_velocity_history
          - diversity_history
          - num_improving_history
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # Best history
        plt.figure()
        plt.plot(results["best_history"])
        plt.xlabel("Iteración")
        plt.ylabel("Mejor fitness global")
        plt.title("Evolución del mejor fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "best_history.png")
        plt.close()

        # Velocidad media
        if results["avg_velocity_history"]:
            plt.figure()
            plt.plot(results["avg_velocity_history"])
            plt.xlabel("Iteración")
            plt.ylabel("Velocidad media")
            plt.title("Evolución de la velocidad media")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "avg_velocity.png")
            plt.close()

        # Diversidad
        if results["diversity_history"]:
            plt.figure()
            plt.plot(results["diversity_history"])
            plt.xlabel("Iteración")
            plt.ylabel("Diversidad (distancia media al gbest)")
            plt.title("Evolución de la diversidad")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "diversity.png")
            plt.close()

        # Mejoras por iteración
        if results["num_improving_history"]:
            plt.figure()
            plt.plot(results["num_improving_history"])
            plt.xlabel("Iteración")
            plt.ylabel("Nº de partículas que mejoran")
            plt.title("Mejoras por iteración")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / "improvements.png")
            plt.close()
