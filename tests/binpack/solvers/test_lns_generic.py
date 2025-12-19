#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ALLOCATION_OBJECTIVES,
    SCHEDULING_OBJECTIVES,
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution


@pytest.mark.parametrize(
    "objective_subproblem",
    (ObjectiveSubproblem.INITIAL_OBJECTIVE,) + ALLOCATION_OBJECTIVES,
)
def test_lns_binary_subobjectives(objective_subproblem, problem):
    subsolver = CpSatBinPackSolver(problem=problem)
    subsolver.init_model(upper_bound=20, modeling=ModelingBinPack.BINARY)

    initial_res = subsolver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    neighbor_builder = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderSubPart(
                problem=problem,
            ),
            NeighborRandom(problem=problem),
        ],
        weight_neighbor=[0.5, 0.5],
    )
    constraints_extractor = NbChangesAllocationConstraintExtractor()
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
        neighbor_builder=neighbor_builder,
        constraints_extractor=constraints_extractor,
    )
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=5,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize(
    "nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_mode",
    [
        (False, False, True, False, True),
        (False, True, True, False, False),
        (True, False, False, False, False),
        (False, False, False, True, False),
    ],
)
def test_lns_binary_params_allocation(
    problem, nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_mode
):
    subsolver = CpSatBinPackSolver(problem=problem)
    subsolver.init_model(upper_bound=20, modeling=ModelingBinPack.BINARY)

    initial_res = subsolver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    neighbor_builder = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderSubPart(
                problem=problem,
            ),
            NeighborRandom(problem=problem),
        ],
        weight_neighbor=[0.5, 0.5],
    )
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
        neighbor_builder=neighbor_builder,
        constraints_extractor=constraints_extractor,
    )
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=5,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize(
    "objective_subproblem",
    (ObjectiveSubproblem.INITIAL_OBJECTIVE,) + SCHEDULING_OBJECTIVES,
)
def test_lns_scheduling_subobjectives(problem, objective_subproblem):
    subsolver = CpSatBinPackSolver(problem=problem)
    subsolver.init_model(upper_bound=20, modeling=ModelingBinPack.SCHEDULING)

    initial_res = subsolver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    neighbor_builder = NeighborBuilderMix(
        list_neighbor=[
            NeighborBuilderSubPart(
                problem=problem,
            ),
            NeighborRandom(problem=problem),
        ],
        weight_neighbor=[0.5, 0.5],
    )
    constraints_extractor = SchedulingConstraintExtractor()
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
        neighbor_builder=neighbor_builder,
        constraints_extractor=constraints_extractor,
    )
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=5,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)
