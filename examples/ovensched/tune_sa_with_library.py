#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Hyperparameter tuning for Simulated Annealing using discrete-optimization's built-in Optuna integration.

This demonstrates the library's generic_optuna_experiment_monoproblem() function which handles:
- Study creation with JournalStorage
- Hyperparameter suggestion via solver.suggest_hyperparameters_with_optuna()
- Automatic pruning based on intermediate results
- Trial management and error handling

LIVE MONITORING:
    View results in real-time with optuna-dashboard:

    Terminal 1:
        uv run python examples/ovensched/tune_sa_with_library.py

    Terminal 2:
        optuna-dashboard optuna-journal-sa.log

    Browser:
        http://localhost:8080
"""

import logging

import numpy as np

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    PortfolioMutation,
)
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
)
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.solution_vector import (
    VectorOvenSchedulingSolution,
)
from discrete_optimization.ovensched.solvers.mutations import (
    OvenPermutationMixedMutation,
    ScheduleAwareMixedMutation,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TunableSimulatedAnnealing(SimulatedAnnealing):
    """
    Simulated Annealing with additional hyperparameters for Optuna tuning.

    Extends SimulatedAnnealing to define tunable hyperparameters:
    - temperature: Initial temperature
    - cooling_exp: Cooling rate (coefficient = 1 - 10^(-cooling_exp))
    - nb_iteration_no_improvement: Restart threshold
    - weight_schedule_aware: Weight for schedule-aware mutations
    """

    hyperparameters = SimulatedAnnealing.hyperparameters + [
        FloatHyperparameter(
            name="temperature",
            low=0.1,
            high=100.0,
            default=10.0,
            depends_on=(),
        ),
        FloatHyperparameter(
            name="cooling_exp",
            low=1.0,
            high=10.0,
            default=5.0,
            depends_on=(),
        ),
        IntegerHyperparameter(
            name="nb_iteration_no_improvement",
            low=50,
            high=500,
            default=100,
            depends_on=(),
        ),
        FloatHyperparameter(
            name="weight_schedule_aware",
            low=0.0,
            high=1.0,
            default=0.5,
            depends_on=(),
        ),
    ]

    def __init__(self, problem, **kwargs):
        """Initialize with custom hyperparameters."""
        # Extract custom hyperparameters
        temperature = kwargs.pop("temperature", 10.0)
        cooling_exp = kwargs.pop("cooling_exp", 5.0)
        nb_iteration_no_improvement = kwargs.pop("nb_iteration_no_improvement", 100)
        weight_schedule_aware = kwargs.pop("weight_schedule_aware", 0.5)

        # Compute cooling coefficient
        coefficient = 1.0 - 10 ** (-cooling_exp)

        # Create mutation portfolio
        weight_permutation = 1.0 - weight_schedule_aware
        mutation = PortfolioMutation(
            problem=problem,
            list_mutations=[
                ScheduleAwareMixedMutation(problem),
                OvenPermutationMixedMutation(problem),
            ],
            weight_mutations=[weight_schedule_aware, weight_permutation],
        )

        # Create restart and temperature handlers
        restart_handler = RestartHandlerLimit(
            nb_iteration_no_improvement=nb_iteration_no_improvement
        )
        temperature_handler = TemperatureSchedulingFactor(
            temperature=temperature,
            restart_handler=restart_handler,
            coefficient=coefficient,
        )

        # Initialize parent with configured components
        super().__init__(
            problem=problem,
            mutator=mutation,
            restart_handler=restart_handler,
            temperature_handler=temperature_handler,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=False,
            **kwargs,
        )


def run_tuning(
    instance_name: str,
    n_trials: int = 30,
    time_limit_per_trial: int = 40,
    storage_path: str = "optuna-journal-sa.log",
):
    """
    Run hyperparameter tuning using the library's generic function.

    Args:
        instance_name: Problem instance name
        n_trials: Number of Optuna trials
        time_limit_per_trial: Time limit per trial in seconds
        storage_path: Path to JournalStorage file
    """
    # Load problem
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING WITH DISCRETE-OPTIMIZATION LIBRARY")
    logger.info("=" * 80)
    logger.info(f"Loading instance: {instance_name}")
    files = get_data_available()
    file_path = [f for f in files if instance_name in f][0]
    problem = parse_dat_file(file_path)
    logger.info(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    logger.info(f"Additional data: {problem.additional_data}")

    # Create deterministic initial solution
    initial_perm = np.arange(problem.n_jobs, dtype=np.int_)
    initial_solution = VectorOvenSchedulingSolution(
        problem=problem, permutation=initial_perm
    )
    logger.info(
        f"Initial solution: Sequential permutation [0, 1, ..., {problem.n_jobs - 1}]"
    )

    # Configuration for fixed parameters
    kwargs_fixed = {
        "time_limit": time_limit_per_trial,
        "callbacks": [TimerStopper(total_seconds=time_limit_per_trial)],
        "nb_iteration_max": 100_000_000,
    }

    logger.info("\n" + "=" * 80)
    logger.info("LIVE MONITORING")
    logger.info("=" * 80)
    logger.info(f"Storage: {storage_path}")
    logger.info("To view results live, run in another terminal:")
    logger.info(f"  optuna-dashboard {storage_path}")
    logger.info("Then open: http://localhost:8080")
    logger.info("=" * 80)

    logger.info(f"\nStarting tuning: {n_trials} trials × {time_limit_per_trial}s")
    logger.info(f"Estimated time: {n_trials * time_limit_per_trial / 60:.1f} minutes\n")

    # Run generic optuna experiment
    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=[TunableSimulatedAnnealing],
        kwargs_fixed_by_solver={TunableSimulatedAnnealing: kwargs_fixed},
        n_trials=n_trials,
        check_satisfy=True,
        computation_time_in_study=False,  # Report by iteration, not time
        study_basename="sa_tuning_library",
        create_another_study=False,  # Use same study name to allow resume
        overwrite_study=False,  # Don't overwrite, allow continuation
        storage_path=storage_path,
        seed=42,  # Reproducible results
        min_time_per_solver=10,  # Wait 10s before pruning
    )

    # Report results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.0f}")
    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        # Remove solver prefix from parameter names
        clean_key = key.replace("TunableSimulatedAnnealing.", "")
        if clean_key == "cooling_exp":
            coef = 1.0 - 10 ** (-value)
            logger.info(f"  {clean_key}: {value:.4f} (coefficient: {coef:.10f})")
        else:
            logger.info(f"  {clean_key}: {value}")

    # Parameter importance
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER IMPORTANCE")
    logger.info("=" * 80)
    try:
        import optuna

        importance = optuna.importance.get_param_importances(study)
        for param, imp in importance.items():
            clean_param = param.replace("TunableSimulatedAnnealing.", "")
            logger.info(f"  {clean_param}: {imp:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute parameter importance: {e}")

    return study


def validate_best_params(study, instance_name: str, n_runs: int = 5):
    """
    Validate best parameters from study.

    Args:
        study: Completed Optuna study
        instance_name: Problem instance name
        n_runs: Number of validation runs
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION")
    logger.info("=" * 80)

    # Load problem
    files = get_data_available()
    file_path = [f for f in files if instance_name in f][0]
    problem = parse_dat_file(file_path)

    # Extract best parameters (remove solver prefix)
    best_params = {
        k.replace("TunableSimulatedAnnealing.", ""): v
        for k, v in study.best_params.items()
        if "." in k  # Only solver-specific params
    }

    logger.info(f"Running {n_runs} validation runs with best parameters...")

    costs = []
    for run in range(n_runs):
        logger.info(f"\nRun {run + 1}/{n_runs}")

        # Create deterministic initial solution
        initial_perm = np.arange(problem.n_jobs, dtype=np.int_)
        initial_solution = VectorOvenSchedulingSolution(
            problem=problem, permutation=initial_perm
        )

        # Create solver with best params
        solver = TunableSimulatedAnnealing(problem=problem, **best_params)
        solver.init_model()
        solver.set_warm_start(initial_solution)

        # Solve
        result = solver.solve(
            nb_iteration_max=100_000_000,
            callbacks=[TimerStopper(total_seconds=60)],
        )

        best_sol = result.get_best_solution()
        cost = solver.aggreg_from_sol(best_sol)
        costs.append(cost)
        logger.info(f"  Cost: {cost:.0f}")

    avg_cost = sum(costs) / len(costs)
    std_cost = (sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5

    logger.info(f"\nValidation results:")
    logger.info(f"  Average: {avg_cost:.0f}")
    logger.info(f"  Std dev: {std_cost:.0f}")
    logger.info(f"  Best: {min(costs):.0f}")
    logger.info(f"  Worst: {max(costs):.0f}")


def main():
    """Main function."""
    instance_name = "87RandomOvenSchedulingInstance-n250-k2-a5--2212-22.47.15.dat"

    # Run tuning
    study = run_tuning(
        instance_name=instance_name,
        n_trials=30,
        time_limit_per_trial=40,
        storage_path="optuna-journal-sa.log",
    )

    # Validate
    validate_best_params(study, instance_name, n_runs=5)


if __name__ == "__main__":
    main()
