#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import numpy as np

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.transformations.generic_scheduling_impl import (
    GenericSchedulingToMultiskillRcpspTransformation,
    MultiskillRcpspToGenericSchedulingTransformation,
)


def test_transfo_to_from_generic_scheduling():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parse_file(file, max_horizon=1000)
    for emp in problem.employees:
        problem.employees[emp].calendar_employee = np.array(
            problem.employees[emp].calendar_employee
        )
        problem.employees[emp].calendar_employee[5:10] = 0
    problem.update_problem()
    transfo = MultiskillRcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(source_problem=problem)
    solver = GenericSchedulingAutoCpSatImplSolver(problem=generic_problem)
    parameters_cp = ParametersCp.default()
    generic_solution: GenericSchedulingImplSolution = solver.solve(
        parameters_cp=parameters_cp,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    solution = transfo.back_transform_solution(generic_solution, source_problem=problem)
    problem.satisfy(solution)

    back_transfo = GenericSchedulingToMultiskillRcpspTransformation()
    generic_problem_2 = transfo.transform_problem(
        source_problem=back_transfo.transform_problem(source_problem=generic_problem)
    )
    solver_2 = GenericSchedulingAutoCpSatImplSolver(problem=generic_problem_2)
    generic_solution_2: GenericSchedulingImplSolution = solver_2.solve(
        parameters_cp=parameters_cp,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()

    assert generic_solution_2.raw_sol == generic_solution.raw_sol
