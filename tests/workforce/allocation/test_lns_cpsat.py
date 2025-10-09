import pytest

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
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.workforce.allocation.parser import (
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.scheduling.parser import get_data_available


@pytest.fixture()
def problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    return parse_to_allocation_problem(instance, multiobjective=True)


TIME_LIMIT_SUBSOLVER = 5


@pytest.mark.parametrize(
    "nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_mode",
    [
        (False, False, True, False, True),
        (False, True, True, False, False),
        (True, False, False, False, False),
        (False, False, False, True, False),
    ],
)
@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_lns_cpsat(
    problem,
    modelisation_allocation,
    nb_changes,
    nb_usages,
    subtasks,
    subresources,
    fix_secondary_tasks_mode,
):
    subsolver = CpsatTeamAllocationSolver(
        problem=problem,
    )
    subsolver.init_model(modelisation_allocation=modelisation_allocation)
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

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
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
@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_lns_cpsat_subobjective(problem, objective_subproblem, modelisation_allocation):
    subsolver = CpsatTeamAllocationSolver(
        problem=problem,
    )
    subsolver.init_model(modelisation_allocation=modelisation_allocation)
    parameters_cp = ParametersCp.default()

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
    )

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
    )
    res = solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=TIME_LIMIT_SUBSOLVER,
        parameters_cp=parameters_cp,
        skip_initial_solution_provider=True,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize(
    "objective_subproblem",
    SCHEDULING_OBJECTIVES,
)
def test_subobjective_nok(problem, objective_subproblem):
    with pytest.raises(ValueError):
        TasksConstraintHandler(
            problem=problem, objective_subproblem=objective_subproblem
        )
