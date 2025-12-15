from pathlib import Path
from experiments.experiment_runner import ExperimentRunner
from initializer import NormalAroundPointInitializer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experimentos_C" / "experimento_C3"

if __name__ == "__main__":

    runner = ExperimentRunner(
        base_output_dir=DEFAULT_RESULTS_DIR
    )

    # ===============================================================
    # CONFIGURACIÓN BASE DEL EXPERIMENTO
    # ===============================================================
    A1 = {
        "experiment_name": "A1",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.2,
        "c1": 2.5,
        "c2": 0.3,
        "vmax": 0.2,
        "initializer_vmax_fraction": 0.5,
    }

    A2 = {
        "experiment_name": "A2",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    A3 = {
        "experiment_name": "A3",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 1.2,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 1.5,
        "initializer_vmax_fraction": 1.0,
    }


    B1 = {
        "experiment_name": "B1",
        "dim": 2,
        "swarm_size": 10,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    B2 = {
        "experiment_name": "B2",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    B3 = {
        "experiment_name": "B3",
        "dim": 2,
        "swarm_size": 40,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    C1 = {
        "experiment_name": "C1",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    C2 = {
        "experiment_name": "C2",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_type": "normal",
        "initializer_center": [0.0, 0.0],
        "initializer_std": 0.5,
    }

    C3 = {
        "experiment_name": "C3",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_type": "normal",
        "initializer_center": [3.0, 3.0],
        "initializer_std": 0.5,
    }

    D1 = {
        "experiment_name": "D1",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 2.5,
        "c2": 0.5,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    D2 = {
        "experiment_name": "D2",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }

    D3 = {
        "experiment_name": "D3",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 0.5,
        "c2": 2.5,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }
    D4 = {
        "experiment_name": "D4",
        "dim": 2,
        "swarm_size": 20,
        "max_iter": 200,
        "w": 0.7,
        "c1": 2.0,
        "c2": 0.0,
        "vmax": 0.5,
        "initializer_vmax_fraction": 1.0,
    }



    # ===============================================================
    #                  Ejecucion Experimento
    # ===============================================================
    results = runner.run_multiple_trials(30, C3)