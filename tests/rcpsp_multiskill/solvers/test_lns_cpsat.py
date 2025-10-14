import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ChainingConstraintExtractor,
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    NbChangesAllocationConstraintExtractor,
    NbUsagesAllocationConstraintExtractor,
    ParamsConstraintExtractor,
    SchedulingConstraintExtractor,
    SubresourcesAllocationConstraintExtractor,
    SubtasksAllocationConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborRandom,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


@pytest.fixture()
def problem():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parse_file(file, max_horizon=1000)
    for emp in problem.employees:
        problem.employees[emp].calendar_employee = np.array(
            problem.employees[emp].calendar_employee
        )
        problem.employees[emp].calendar_employee[5:10] = 0
    problem.update_functions()
    return problem


@pytest.fixture()
def problem_multimode(problem):
    task = 2
    old_mode = 1
    new_mode = 2
    new_details = dict(problem.mode_details[task][old_mode])
    new_details["duration"] *= 10
    new_details["Q8"] = 2
    problem.mode_details[task][new_mode] = new_details
    problem.update_functions()
    return problem


TIME_LIMIT_SUBSOLVER = 5


@pytest.mark.parametrize(
    "chaining, nb_changes, nb_usages, subtasks, subresources, fix_secondary_tasks_allocation",
    [
        (False, False, False, True, False, True),
        (True, False, True, True, False, False),
        (True, True, False, False, False, False),
        (False, False, False, False, True, False),
    ],
)
def test_lns_cpsat(
    problem_multimode,
    chaining,
    nb_changes,
    nb_usages,
    subtasks,
    subresources,
    fix_secondary_tasks_allocation,
):
    problem = problem_multimode
    subsolver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    parameters_cp = ParametersCp.default()

    extractors: list[BaseConstraintExtractor] = [
        SchedulingConstraintExtractor(),
        MultimodeConstraintExtractor(),
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
                fix_secondary_tasks_allocation=fix_secondary_tasks_allocation
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
    "subojective",
    [ObjectiveSubproblem.SUM_END_SUBTASKS, ObjectiveSubproblem.NB_UNARY_RESOURCES_USED],
)
def test_lns_cpsat_subobjective(problem_multimode, subojective):
    problem = problem_multimode
    subsolver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    parameters_cp = ParametersCp.default()

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=subojective,
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
    "params",
    [
        ParamsConstraintExtractor(),
        ParamsConstraintExtractor(
            chaining=True,
            nb_usages=True,
            nb_changes=True,
            allocation_subtasks=False,
            allocation_subresources=True,
        ),
    ],
)
def test_default_constraint_handler(problem, params):
    constraint_handler = TasksConstraintHandler(
        problem=problem, params_constraint_extractor=params
    )
