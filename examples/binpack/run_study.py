#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import pandas as pd

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.problem import BinPackProblem
from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.binpack.solvers.greedy import (
    BinSelectionStrategy,
    GreedyBinPackSolver,
    SortingStrategy,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ProblemEvaluateLogger,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import SolverConfig
from discrete_optimization.generic_tools.study.database import is_empty_metrics
from discrete_optimization.generic_tools.study.study import Study

logging.basicConfig(level=logging.INFO)
study_name = "bppc-study-big-instance"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
max_retry = 0  # retry failed xps (and how many times) ?
time_limit = 60
n_instances = 20  # number of largest instances to solve


def get_largest_instances():
    # Get the largest instances
    all_files = get_data_available_bppc()
    problems_instance: list[tuple[BinPackProblem, str]] = [
        (parse_bin_packing_constraint_file(f), f) for f in all_files
    ]
    sorted_instances = sorted(problems_instance, key=lambda x: x[0].nb_items)[
        :n_instances
    ]
    return [f for _, f in sorted_instances]


def problem_factory(instance: str) -> BinPackProblem:
    """Factory to create BinPackProblem from instance filename."""
    problem = parse_bin_packing_constraint_file(instance)
    return problem


def solver_configs():
    # Configure solvers
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    solver_configs = {}

    for proc in [16]:
        p = ParametersCp.default_cpsat()
        p.nb_process = proc
        for modeling in [ModelingBinPack.SCHEDULING]:  # , ModelingBinPack.BINARY]:
            solver_configs[f"cpsat-proc-{proc}-{modeling.name}"] = SolverConfig(
                cls=CpSatBinPackSolver,
                kwargs=dict(
                    time_limit=time_limit,
                    modeling=modeling,
                    ortools_cpsat_solver_kwargs={"log_search_progress": True},
                    parameters_cp=p,
                ),
            )

    solver_configs["asp"] = SolverConfig(
        cls=AspBinPackingSolver, kwargs=dict(time_limit=time_limit)
    )

    # Greedy solver with different configurations
    solver_configs["greedy-first-fit"] = SolverConfig(
        cls=GreedyBinPackSolver,
        kwargs=dict(
            sorting_strategy=SortingStrategy.NONE,
            bin_selection_strategy=BinSelectionStrategy.FIRST_FIT,
        ),
    )
    solver_configs["greedy-weight-best-fit"] = SolverConfig(
        cls=GreedyBinPackSolver,
        kwargs=dict(
            sorting_strategy=SortingStrategy.WEIGHT_DESC_CONFLICT_ASC,
            bin_selection_strategy=BinSelectionStrategy.BEST_FIT_MIN_WEIGHT,
        ),
    )
    solver_configs["greedy-bfd"] = SolverConfig(
        cls=GreedyBinPackSolver,
        kwargs=dict(
            sorting_strategy=SortingStrategy.CONFLICT_DESC_WEIGHT_DESC,
            bin_selection_strategy=BinSelectionStrategy.BEST_FIT_MIN_REMAINING,
        ),
    )
    return solver_configs


def run_study():
    study = Study(
        name=study_name,
        instances=get_largest_instances(),
        solver_configs=solver_configs(),
        overwrite=overwrite,
        max_retry=max_retry,
        problem_factory=problem_factory,
    )

    for problem, solver, solver_kwargs in study:
        try:
            # Compute upper bound using greedy solver
            greedy = GreedyBinPackSolver(problem=problem)
            res = greedy.solve()
            sol = res[-1][0]
            nb_bins = problem.evaluate(sol)["nb_bins"]
            # Configure callbacks based on solver type
            if solver.__class__ == CpSatBinPackSolver:
                stats_cb = StatsWithBoundsCallback()
                callbacks = [
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ProblemEvaluateLogger(logging.INFO, logging.INFO),
                    ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
                ]
                # Initialize model with upper bound for CP-SAT solver
                solver.init_model(**solver_kwargs, upper_bound=nb_bins)
            else:
                stats_cb = BasicStatsCallback()
                callbacks = [
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ProblemEvaluateLogger(logging.INFO, logging.INFO),
                ]
            # Solve
            result_store = solver.solve(callbacks=callbacks, **solver_kwargs)
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            logging.error(e)
            reason = f"{type(e)}: {str(e)}"
            success = False
        else:
            # get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            success = not is_empty_metrics(metrics)
            if success:
                logging.info("experiment successful")
            else:
                logging.info("experiment unsuccessful (no metrics found)")
            reason = ""
            logging.info(
                f"Instance {study.get_current_instance()}, "
                f"Solver config {study.get_current_config_name()}, "
                # f"value={solver.get_current_best_internal_objective_value()}, "
                # f"bound={solver.get_current_best_internal_objective_bound()}"
            )
            if len(result_store) > 0:
                logging.info(f"Fit = {result_store[-1][1]}")

        # store corresponding experiment
        study.store_current_xp(
            metrics=metrics, status=status, reason=reason, success=success
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_study()
