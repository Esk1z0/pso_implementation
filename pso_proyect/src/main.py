from experiments.experiment_runner import ExperimentRunner  # donde lo guardes

if __name__ == "__main__":
    

    runner = ExperimentRunner(base_output_dir="/home/juanes/Escritorio/CLASE/METAHEURISTICA/PRACTICA3/results")

    base_config = {
        "experiment_name": "rastrigin_basico",
        "dim": 2,
        "swarm_size": 30,
        "max_iter": 200,
        "w": 0.7,
        "c1": 1.4,
        "c2": 1.4,
        "vmax": 0.5,
        "initializer_vmax_fraction": 0.2,
        "seed": 1234,
    }

    # Un único experimento
    results = runner.run_experiment(base_config)

    # Si quieres varios trials:
    # runner.run_multiple_trials(5, base_config)
