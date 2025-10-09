import pytest

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ChainingConstraintExtractor,
    ConstraintExtractorList,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
)


@pytest.fixture()
def problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    return parse_json_to_problem(instance)


TIME_LIMIT_SUBSOLVER = 5


@pytest.mark.parametrize(
    "chaining, nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_mode",
    [
        (False, False, False, True, False, True),
        (True, False, True, True, False, False),
        (True, True, False, False, False, False),
        (False, False, False, False, True, False),
    ],
)
def test_lns_cpsat(
    problem,
    chaining,
    nb_changes,
    nb_usages,
    subtasks,
    subresources,
    fix_secondary_tasks_mode,
):
    subsolver = CPSatAllocSchedulingSolver(
        problem=problem,
    )
    parameters_cp = ParametersCp.default()

    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(),
    ]
    if chaining:
        extractors.append(ChainingConstraintExtractor())
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
    [
        ObjectiveSubproblem.SUM_END_SUBTASKS,
        ObjectiveSubproblem.NB_UNARY_RESOURCES_USED,
        ObjectiveSubproblem.GLOBAL_MAKESPAN,
        ObjectiveSubproblem.NB_TASKS_DONE,
    ],
)
def test_lns_cpsat_subobjective(problem, objective_subproblem):
    subsolver = CPSatAllocSchedulingSolver(
        problem=problem,
    )
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
