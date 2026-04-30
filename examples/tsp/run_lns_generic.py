#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorPortfolio,
    SchedulingConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    TrivialInitialSolution,
)
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver

logging.basicConfig(level=logging.INFO)


def run_lns():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    problem = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=problem)
    from discrete_optimization.tsp.solvers.ortools_routing import ORtoolsTspSolver

    solver = ORtoolsTspSolver(problem=problem)
    res = solver.solve(time_limit=10)
    initial_solution_provider = TrivialInitialSolution(res)
    subsolver = CpSatTspSolver(
        problem, params_objective_function=params_objective_function
    )
    subsolver.init_model()
    n = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderSubPart(problem=problem, nb_cut_part=4),
            NeighborRandom(problem=problem, fraction_subproblem=0.6),
        ],
        weight_neighbor=[1 / 2] * 2,
        verbose=True,
    )
    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(
            plus_delta_primary=100,
            minus_delta_primary=100,
            plus_delta_secondary=10,
            minus_delta_secondary=10,
        ),
        # DummyConstraintExtractor(),
    ]
    constraints_extractor = ConstraintExtractorPortfolio(
        extractors=extractors,
        weights=[1 / len(extractors)] * len(extractors),
    )
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=n,
        constraints_extractor=constraints_extractor,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        initial_solution_provider=initial_solution_provider,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 15
    res = solver.solve(
        callbacks=[WarmStartCallback()],
        nb_iteration_lns=100,
        time_limit_subsolver=20,
        time_limit_subsolver_iter0=40,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
        ortools_cpsat_solver_kwargs={
            "log_search_progress": False,
            "fix_variables_to_their_hinted_value": False,
        },
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=problem, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    run_lns()
