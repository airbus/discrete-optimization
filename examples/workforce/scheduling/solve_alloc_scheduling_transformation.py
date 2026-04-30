#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)
from discrete_optimization.workforce.scheduling.transformations.to_fjsp import (
    WorkforceSchedulingToFjspTransformation,
)
from discrete_optimization.workforce.scheduling.transformations.to_multiskill import (
    WorkforceSchedulingToMultiskillTransformation,
)
from discrete_optimization.workforce.scheduling.utils import (
    plot_schedule_comparison,
    plt,
)


def run_solver_via_multiskill():
    files = get_data_available()
    problem: AllocSchedulingProblem = parse_json_to_problem(files[0])
    print(problem.tasks_list, " tasks ")
    solver = TransformationSolver(
        transformation=WorkforceSchedulingToMultiskillTransformation(),
        problem=problem,
        solver_brick=SubBrick(
            CpSatMultiskillRcpspSolver,
            kwargs=dict(
                parameters_cp=ParametersCp.default_cpsat(),
                time_limit=20,
                ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            ),
        ),
    )
    result = solver.solve()
    sol: AllocSchedulingSolution = result[-1][0]
    plot_schedule_comparison(sol, sol, problem)
    print(sol.schedule)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    plt.show()


def run_solver_via_fjobshop():
    files = get_data_available()
    problem: AllocSchedulingProblem = parse_json_to_problem(files[0])
    print(problem.tasks_list, " tasks ")
    solver = TransformationSolver(
        transformation=WorkforceSchedulingToFjspTransformation(),
        problem=problem,
        solver_brick=SubBrick(
            CpSatFjspSolver,
            kwargs=dict(
                parameters_cp=ParametersCp.default_cpsat(),
                time_limit=20,
                ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            ),
        ),
    )
    result = solver.solve()
    sol: AllocSchedulingSolution = result[-1][0]
    plot_schedule_comparison(sol, sol, problem)
    print(sol.schedule)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    plt.show()


if __name__ == "__main__":
    run_solver_via_fjobshop()
