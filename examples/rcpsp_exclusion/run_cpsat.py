#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.plot_utils import (
    plot_ressource_view,
    plot_task_gantt,
    plt,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    RcpspToGenericSchedulingTransformation,
)
from discrete_optimization.rcpsp_exclusion.utils import (
    create_exclusion_rcpsp_problem,
    get_data_available,
    parse_file,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    base_problem = parse_file(file)
    problem = create_exclusion_rcpsp_problem(
        problem=base_problem,
        nb_exclusion_resource=1,
        proportion_blocked_tasks=0.1,
        proportion_blocking_tasks=0.1,
    )
    solver = CpSatRcpspSolver(problem=problem)
    solver.init_model()
    params_cp = ParametersCp.default_cpsat()
    res = solver.solve(
        parameters_cp=params_cp,
        time_limit=20,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    for r in problem.exclusion_resources_list:
        print("---Exclusion resource:", r, "---")
        print("Block : ", problem.get_tasks_possibly_exclude_others(r))
        print("->Excluded", problem.get_tasks_possibly_excluded(r))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


def run_via_generic():
    file = [f for f in get_data_available() if "j1201_1.sm" in f][0]
    base_problem = parse_file(file)
    problem = create_exclusion_rcpsp_problem(
        problem=base_problem,
        nb_exclusion_resource=20,
        proportion_blocked_tasks=0.1,
        proportion_blocking_tasks=0.2,
        range_capacities=(1, 1),
    )
    p = ParametersCp.default_cpsat()
    solver = TransformationSolver(
        transformation=RcpspToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            kwargs=dict(
                parameters_cp=p,
                time_limit=15,
                use_cpm_for_task_bounds=True,
                use_energy_constraints=True,
                ortools_cpsat_solver_kwargs={"log_search_progress": True},
            ),
        ),
        source_problem=problem,
    )
    callback = ProblemEvaluateLogger(
        step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
    )
    res = solver.solve(
        callbacks=[callback],
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    for r in problem.exclusion_resources_list:
        print("---Exclusion resource:", r, "---")
        print("Block : ", problem.get_tasks_possibly_exclude_others(r))
        print("->Excluded", problem.get_tasks_possibly_excluded(r))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


if __name__ == "__main__":
    run_via_generic()
