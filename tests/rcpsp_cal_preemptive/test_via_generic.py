#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    RcpspToGenericSchedulingTransformation,
)
from discrete_optimization.rcpsp_cal_preemptive.utils import (
    load_calendar_preemptive_rcpsp_problem,
)


def test_solve_via_generic(
    generic_solver_cls, generic_solver_kwargs, generic_solver_check_solution
):
    problem = parse_file([f for f in get_data_available() if "j301_1.sm" in f][0])
    problem = load_calendar_preemptive_rcpsp_problem(problem)
    transfo = RcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(source_problem=problem)
    parameters_cp = ParametersCp.default()
    solver = generic_solver_cls(generic_problem)
    solver.init_model()
    generic_solution: GenericSchedulingImplSolution = solver.solve(
        parameters_cp=parameters_cp,
        time_limit=10,
        callbacks=[NbIterationStopper(1)],
        **generic_solver_kwargs,
    ).get_best_solution()
    if generic_solver_check_solution:
        solution = transfo.back_transform_solution(
            generic_solution, source_problem=problem
        )
        assert problem.satisfy(solution)
