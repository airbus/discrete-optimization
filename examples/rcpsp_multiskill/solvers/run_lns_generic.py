#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import numpy as np

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)

logging.basicConfig(level=logging.INFO)


def run_lns_generic():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parse_file(file, max_horizon=1000)
    for emp in problem.employees:
        problem.employees[emp].calendar_employee = np.array(
            problem.employees[emp].calendar_employee
        )
        problem.employees[emp].calendar_employee[5:10] = 0
    problem.update_functions()
    subsolver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    parameters_cp = ParametersCp.default()

    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(
            minus_delta_primary=100,
            plus_delta_primary=100,
            minus_delta_secondary=10,
            plus_delta_secondary=10,
        ),
        MultimodeConstraintExtractor(),
        SubresourcesAllocationConstraintExtractor(),
    ]
    constraints_extractor = ConstraintExtractorList(extractors=extractors)

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        constraints_extractor=constraints_extractor,
    )

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
        callbacks=[],
        nb_iteration_lns=100,
        time_limit_subsolver_iter0=2,
        time_limit_subsolver=5,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_lns_generic()
