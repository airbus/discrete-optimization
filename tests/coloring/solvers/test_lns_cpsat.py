#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import plot_coloring_solution, plt
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringSolution,
    transform_coloring_problem,
)
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.lns_cp import (
    FixColorsCpSatConstraintHandler,
    InitialColoring,
    InitialColoringMethod,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat

logging.basicConfig(level=logging.INFO)


def test_lns_cpsat_coloring(caplog):
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    solver = CpSatColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        modeling=ModelingCpSat.INTEGER,
        do_warmstart=False,
        value_sequence_chain=False,
        used_variable=True,
        symmetry_on_used=True,
        nb_colors=60,
    )
    solver_lns = LnsOrtoolsCpSat(
        problem=color_problem,
        subsolver=solver,
        initial_solution_provider=None,
        constraint_handler=FixColorsCpSatConstraintHandler(
            problem=color_problem, fraction_to_fix=0.3
        ),
        post_process_solution=None,
    )
    p = ParametersCp.default_cpsat()
    logging.info("Starting solve")
    with caplog.at_level(logging.WARNING):
        result_store = solver_lns.solve(
            skip_initial_solution_provider=True,
            nb_iteration_lns=5,
            callbacks=[
                NbIterationTracker(step_verbosity_level=logging.INFO),
                ObjectiveLogger(
                    step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
                ),
                TimerStopper(total_seconds=30),
            ],
            parameters_cp=p,
            time_limit_subsolver=20,
            stop_first_iteration_if_optimal=False,
        )
    assert "`time_limit` arg will be overriden by" not in caplog.text
    print("Status solver : ", solver.status_solver)
    solution, fit = result_store.get_best_solution_fit()
    # plot_coloring_solution(solution)
    # plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    assert color_problem.satisfy(solution)


def test_lns_cpsat_coloring_with_constraints():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    color_problem = transform_coloring_problem(
        color_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ColoringConstraints(color_constraint={0: 0, 1: 1, 2: 2}),
    )
    solver = CpSatColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        modeling=ModelingCpSat.INTEGER,
        do_warmstart=False,
        value_sequence_chain=False,
        used_variable=True,
        symmetry_on_used=True,
        nb_colors=20,
    )
    solver_lns = LnsOrtoolsCpSat(
        problem=color_problem,
        subsolver=solver,
        initial_solution_provider=None,
        constraint_handler=FixColorsCpSatConstraintHandler(
            problem=color_problem, fraction_to_fix=0.3
        ),
        post_process_solution=None,
    )
    p = ParametersCp.default_cpsat()
    logging.info("Starting solve")
    result_store = solver_lns.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=100,
        time_limit_subsolver=20,
        callbacks=[
            NbIterationTracker(step_verbosity_level=logging.INFO),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            TimerStopper(total_seconds=30),
        ],
        parameters_cp=p,
    )
    print("Status solver : ", solver.status_solver)
    solution, fit = result_store.get_best_solution_fit()
    # plot_coloring_solution(solution)
    # plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    assert color_problem.satisfy(solution)


def test_lns_cpsat_coloring_warm_start():
    """Test `set_warm_start()` api."""
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)

    def solver_lns_factory():
        solver = CpSatColoringSolver(color_problem, params_objective_function=None)
        solver.init_model(
            modeling=ModelingCpSat.INTEGER,
            do_warmstart=False,
            value_sequence_chain=False,
            used_variable=True,
            symmetry_on_used=True,
            nb_colors=60,
        )

        return LnsOrtoolsCpSat(
            problem=color_problem,
            subsolver=solver,
            initial_solution_provider=InitialColoring(
                problem=color_problem, initial_method=InitialColoringMethod.DUMMY
            ),
            constraint_handler=FixColorsCpSatConstraintHandler(
                problem=color_problem, fraction_to_fix=0.3
            ),
            post_process_solution=None,
        )

    p = ParametersCp.default_cpsat()

    # w/o warm start
    solver_lns = solver_lns_factory()
    result_store = solver_lns.solve(
        nb_iteration_lns=3, parameters_cp=p, time_limit_subsolver=10
    )

    # start solution
    start_solution: ColoringSolution = (
        GreedyColoringSolver(
            problem=color_problem,
        )
        .solve(strategy=NxGreedyColoringMethod.largest_first)
        .get_best_solution()
    )

    # with warm start
    solver_lns = solver_lns_factory()
    solver_lns.set_warm_start(start_solution)
    result_store2 = solver_lns.solve(
        nb_iteration_lns=3,
        parameters_cp=p,
    )

    assert result_store[0][0].colors != start_solution.colors
    assert result_store2[0][0].colors == start_solution.colors
