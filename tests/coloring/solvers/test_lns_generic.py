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

modeling, value_sequence_chain, used_variable, symmetry_on_used = param_fixtures(
    "modeling, value_sequence_chain, used_variable, symmetry_on_used",
    [
        (ModelingCpSat.BINARY, False, False, False),
        (ModelingCpSat.INTEGER, False, False, False),
        (ModelingCpSat.INTEGER, True, False, False),
        (ModelingCpSat.INTEGER, False, True, False),
        (ModelingCpSat.INTEGER, False, True, True),
        (ModelingCpSat.INTEGER, True, True, True),
    ],
)


@fixture
def solver(
    problem,
    with_coloring_constraint,
    modeling,
    value_sequence_chain,
    used_variable,
    symmetry_on_used,
):
    if with_coloring_constraint and value_sequence_chain:
        pytest.skip(
            "chosen coloring constraint not compatible with value_sequence_chain"
        )
    init_model_kwargs = dict(
        nb_colors=20,
        modeling=modeling,
        value_sequence_chain=value_sequence_chain,
        used_variable=used_variable,
        symmetry_on_used=symmetry_on_used,
    )
    solver = CpSatColoringSolver(problem)
    solver.init_model(**init_model_kwargs)
    return solver


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
def test_lns_cpsat(
    problem,
    solver,
    nb_changes,
    nb_usages,
    subtasks,
    subresources,
    fix_secondary_tasks_mode,
):
    subsolver = solver
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
def test_lns_cpsat_subobjective(problem, objective_subproblem, solver):
    subsolver = solver
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
