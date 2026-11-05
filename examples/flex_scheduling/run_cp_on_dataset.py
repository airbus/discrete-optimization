#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from copy import deepcopy
from enum import Enum

from discrete_optimization.flex_scheduling.parser import (
    get_data_available,
    load_problem_from_json,
)
from discrete_optimization.flex_scheduling.problem import (
    FlexProblem,
    ObjectiveParamEarliness,
    ObjectivesEnum,
)
from discrete_optimization.flex_scheduling.solvers.cpsat import (
    ConstraintIncluding,
    CpSatFlexSolver,
    DurationEncodingEnum,
)
from discrete_optimization.flex_scheduling.solvers.optal import (
    OptalFlexProblemSolver,
)
from discrete_optimization.flex_scheduling.solvers.sequential_solver import (
    SequentialFlexSolver,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallbackLastRun,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver


class CpSolverBackend(Enum):
    CPSAT = "cpsat"
    CPSAT_GENERIC = "cpsat_generic"
    OPTAL = "optal"


logging.basicConfig(level=logging.INFO)


def run_cp(
    problem: FlexProblem,
    use_sequential: bool = True,
    nb_batches: int = 5,
    backend_solver: CpSolverBackend = CpSolverBackend.CPSAT,
    time_limit_sequential: int = 60,
    time_limit_main_solve: int = 120,
):
    # Solve with sequential solver using small batches and short time limit
    warm_start_solution = None
    if use_sequential:
        solver = SequentialFlexSolver(problem=problem)
        result_storage = solver.solve(
            nb_batches=nb_batches,
            time_limit_per_batch=int(time_limit_sequential / nb_batches),
        )
        warm_start_solution = result_storage[-1][0]
    solver = None
    if backend_solver == CpSolverBackend.CPSAT:
        solver = CpSatFlexSolver(problem)
        solver.init_model(
            constraint_including=ConstraintIncluding(
                include_constraints_on_groups=True,
                include_non_released_resource=True,
                include_generalized_time_constraints=True,
                include_group_variables=True,
                include_constraint_precedence_on_groups=True,
                include_variable_resource=True,
            ),
            duration_encoding=DurationEncodingEnum.INDICATOR,
        )
        if warm_start_solution:
            solver.set_warm_start_from_sol(warm_start_solution)
    if backend_solver == CpSolverBackend.CPSAT_GENERIC:
        from discrete_optimization.flex_scheduling.solvers.cpsat_auto import (
            CpSatAutoFlexSchedulingSolver,
        )

        solver = CpSatAutoFlexSchedulingSolver(problem)
        solver.init_model(logging_objectives=True)
        if warm_start_solution:
            solver.set_warm_start(warm_start_solution)
    if backend_solver == CpSolverBackend.OPTAL:
        solver = OptalFlexProblemSolver(problem)
        solver.init_model()
        if warm_start_solution:
            solver.set_warm_start(solution=warm_start_solution)
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 16
    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        time_limit=time_limit_main_solve,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    # Get solution
    solution = result_storage.get_best_solution()
    kpis = problem.evaluate(solution)
    satisfy = problem.satisfy(solution)
    print("Satisfy ? : ", satisfy)
    print("Evaluation : ", kpis)
    return solution


def run_lexico(
    problem: FlexProblem,
    use_sequential: bool = True,
    nb_batches: int = 5,
    backend_solver: CpSolverBackend = CpSolverBackend.CPSAT,
    time_limit_sequential: int = 60,
    time_limit_main_solve: int = 120,
):
    warm_start_solution = None
    if use_sequential:
        solver = SequentialFlexSolver(problem=problem)
        result_storage = solver.solve(
            nb_batches=nb_batches,
            time_limit_per_batch=int(time_limit_sequential / nb_batches),
        )
        warm_start_solution = result_storage[-1][0]
    from discrete_optimization.flex_scheduling.solvers.cpsat_auto import (
        CpSatAutoFlexSchedulingSolver,
    )

    solver = CpSatAutoFlexSchedulingSolver(problem)
    solver.init_model(logging_objectives=True, use_cpm_for_task_bounds=False)
    if warm_start_solution:
        solver.set_warm_start(warm_start_solution)
    lexico = LexicoSolver(subsolver=solver, problem=problem)
    callbacks = [WarmStartCallbackLastRun()]
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = lexico.solve(
        callbacks=callbacks,
        objectives=[
            Objective.EARLINESS_ONLY,
            Objective.TARDINESS_ONLY,
            Objective.MAKESPAN,
        ],
        time_limit=1000,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        parameters_cp=p,
    )
    solution = res[-1][0]
    kpis = problem.evaluate(solution)
    satisfy = problem.satisfy(solution)
    print("Satisfy ? : ", satisfy)
    print("Evaluation : ", kpis)
    return solution


def main():
    problems = get_data_available()
    problem_name = "H3"
    problem_file = [p for p in problems if problem_name in p][0]
    problem = load_problem_from_json(problem_file)
    # problem.validate_resource_usage_with_regard_to_span_constraints()
    # problem.fix_resource_usage_with_regard_to_span_constraints()
    # problem.objective_params.params_obj.pop(ObjectivesEnum.TARDINESS)
    # problem.objective_params.params_obj.pop(ObjectivesEnum.RESOURCE_COST)
    problem.objective_params.params_obj[ObjectivesEnum.EARLINESS] = (
        ObjectiveParamEarliness(
            weight_per_task=deepcopy(
                problem.objective_params.params_obj[
                    ObjectivesEnum.WORK_IN_PROGRESS
                ].weight_per_task
            ),
            weight_per_groups={},
        )
    )
    if True:
        run_lexico(
            problem,
            use_sequential=True,
            nb_batches=5,
            time_limit_sequential=200,
            time_limit_main_solve=100,
        )
    if False:
        run_cp(
            problem,
            use_sequential=True,
            backend_solver=CpSolverBackend.CPSAT,
            nb_batches=10,
            time_limit_sequential=200,
            time_limit_main_solve=100,
        )


if __name__ == "__main__":
    main()
