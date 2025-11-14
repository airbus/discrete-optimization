#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from pytest_cases import fixture, param_fixtures

from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ALLOCATION_OBJECTIVES,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver

TIME_LIMIT_SUBSOLVER = 5


@fixture
def subsolver(problem):
    solver = CpSatKnapsackSolver(problem)
    solver.init_model()
    return solver


@pytest.mark.parametrize(
    "nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_mode",
    [
        (False, False, True, False, True),
        (False, True, True, False, False),
        (True, False, False, False, False),
        (False, False, False, True, False),
    ],
)
def test_lns_cpsat(
    problem,
    subsolver,
    nb_changes,
    nb_usages,
    subtasks,
    subresources,
    fix_secondary_tasks_mode,
):
    parameters_cp = ParametersCp.default()

    extractors: list[BaseConstraintExtractor] = []
    if nb_changes:
        extractors.append(NbChangesAllocationConstraintExtractor())
    if nb_usages:
        extractors.append(NbUsagesAllocationConstraintExtractor())
    if subresources:
        extractors.append(SubresourcesAllocationConstraintExtractor())
    if subtasks:
        extractors.append(
            SubtasksAllocationConstraintExtractor(
                fix_secondary_tasks_allocation=fix_secondary_tasks_mode
            )
        )
    constraints_extractor = ConstraintExtractorList(extractors=extractors)

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        constraints_extractor=constraints_extractor,
    )

    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize(
    "objective_subproblem",
    ALLOCATION_OBJECTIVES,
)
def test_lns_cpsat_subobjective(problem, objective_subproblem, subsolver):
    parameters_cp = ParametersCp.default()

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
    )

    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)
