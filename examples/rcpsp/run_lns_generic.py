#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    ParamsConstraintExtractor,
    SchedulingConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import (
    ReinitModelCallback,
    TrivialInitialSolution,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver

logging.basicConfig(level=logging.INFO)


def run_lns_generic(problem: RcpspProblem):
    subsolver = CpSatRcpspSolver(problem=problem)

    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)

    constraints_extractor = ConstraintExtractorList(
        extractors=[
            SchedulingConstraintExtractor(
                minus_delta_primary=200,
                plus_delta_primary=200,
                minus_delta_secondary=10,
                plus_delta_secondary=10,
            ),
            MultimodeConstraintExtractor(
                fix_primary_tasks_modes=False,
                fix_secondary_tasks_modes=True,
            ),
        ]
    )
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        neighbor_builder=NeighborRandom(problem=problem, fraction_subproblem=0.1),
        params_constraint_extractor=ParamsConstraintExtractor(
            constraint_to_current_solution_makespan=False,
            margin_rel_to_current_solution_makespan=0.05,
            fix_primary_tasks_modes=False,
            fix_secondary_tasks_modes=True,
        ),
        constraints_extractor=constraints_extractor,
    )
    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = solver.solve(
        callbacks=[
            ReinitModelCallback(),
            WarmStartCallback(
                warm_start_best_solution=False, warm_start_last_solution=True
            ),
            # ReinitModelCallback()
        ],
        nb_iteration_lns=1000,
        time_limit_subsolver_iter0=20,
        time_limit_subsolver=10,
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    print("Satisfy : ", problem.satisfy(sol))
    print("Evaluate : ", problem.evaluate(sol))


def run_on_mm():
    model = "j1010_1.mm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    run_lns_generic(problem)


def run_on_sm():
    model = "j1201_3.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    run_lns_generic(problem)


if __name__ == "__main__":
    run_on_sm()
