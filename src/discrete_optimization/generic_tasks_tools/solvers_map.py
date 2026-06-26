#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

BaseProblemType = GenericSchedulingImplProblem


solvers: dict[str, list[tuple[type[SolverDO], dict[str, Any]]]] = {
    "cp": [
        (GenericSchedulingAutoCpSatImplSolver, {}),
    ],
    "lns-scheduling": [
        (
            LnsOrtoolsCpSat,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
                "subsolver_subbrick": SubBrick(
                    cls=GenericSchedulingAutoCpSatImplSolver, kwargs={}
                ),
                "constraint_handler_subbrick": SubBrick(
                    cls=TasksConstraintHandler, kwargs={}
                ),
                "skip_initial_solution_provider": True,
            },
        ),
    ],
}

solvers_map: dict[type[SolverDO], tuple[str, dict[str, Any]]] = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[SolverDO], list[type[Problem]]] = {
    solver: [BaseProblemType] for solver in solvers_map
}


def look_for_solver(
    domain: Problem,
) -> list[type[SolverDO]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[Problem],
) -> list[type[SolverDO]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: type[SolverDO],
    problem: Problem,
    **kwargs: Any,
) -> ResultStorage:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs)


def solve_return_solver(
    method: type[SolverDO],
    problem: Problem,
    **kwargs: Any,
) -> tuple[ResultStorage, SolverDO]:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs), solver


def return_solver(
    method: type[SolverDO],
    problem: Problem,
    **kwargs: Any,
) -> SolverDO:
    solver: SolverDO
    solver = method(problem=problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver


def get_solver_default_arguments(
    method: type[SolverDO],
) -> dict[str, Any]:
    try:
        return solvers_map[method][1]
    except KeyError:
        raise KeyError(
            f"{method} is not in the list of available solvers for {BaseProblemType.__name__}."
        )
