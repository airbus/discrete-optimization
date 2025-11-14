#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    ParamsConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborBuilderTimeWindow,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver
from examples.fjsp.run_lns_cpsat import ReinitModelCallback

logging.basicConfig(level=logging.INFO)


def run_lns_cpsat_jsp():
    file_path = get_data_available()[4]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    solver = CpSatJspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        # neighbor_builder=NeighborRandom(problem,
        #                                0.5, 10,
        #                                0.1),
        neighbor_builder=NeighborBuilderMix(
            [NeighborBuilderSubPart(problem, 5), NeighborRandom(problem, 0.5, 10, 0.1)],
            [0.5, 0.5],
        ),
        params_constraint_extractor=ParamsConstraintExtractor(
            minus_delta_primary=10000,
            plus_delta_primary=10000,
            minus_delta_secondary=300,
            plus_delta_secondary=300,
            constraint_to_current_solution_makespan=True,
            margin_rel_to_current_solution_makespan=0.05,
            fix_primary_tasks_modes=False,
            fix_secondary_tasks_modes=True,
        ),
        objective_subproblem=ObjectiveSubproblem.GLOBAL_MAKESPAN,
        # constraints_extractor=ConstraintExtractorList([])
    )

    lns_solver = LnsOrtoolsCpSat(
        problem=problem, subsolver=solver, constraint_handler=constraint_handler
    )
    res = lns_solver.solve(
        callbacks=[ReinitModelCallback(), WarmStartCallback()],
        skip_initial_solution_provider=True,
        nb_iteration_lns=1000,
        parameters_cp=p,
        time_limit_subsolver_iter0=2,
        time_limit_subsolver=5,
        # ortools_cpsat_solver_kwargs={"log_search_progress": True}
    )
    sol, fit = res.get_best_solution_fit()
    print("Evaluate", problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_lns_cpsat_jsp()
