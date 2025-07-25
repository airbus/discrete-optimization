import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpmpy import (
    CPMpyTeamAllocationSolver,
    CPMpyTeamAllocationSolverStoreConstraintInfo,
    ModelisationAllocationCP,
)
from discrete_optimization.workforce.allocation.utils import plot_allocation_solution
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_allocation_disruption,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def test_cpmpy_multiobj():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=True)
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model()
    # check solve
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    sol = res.get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)
    # check plot
    plot_allocation_solution(
        problem=allocation_problem,
        sol=sol,
        display=False,
    )


def test_cpmpy_monoobj():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=False)
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model()
    sol = solver.solve(
        time_limit=5,
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)
    plot_allocation_solution(
        problem=allocation_problem,
        use_color_map=True,
        sol=sol,
        display=False,
    )


def test_cpmpy_cnf_compatible_nok():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=True)
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    with pytest.raises(NotImplementedError):
        solver.init_model(
            modelisation_allocation=ModelisationAllocationCP.CNF_COMPATIBLE
        )


@pytest.mark.parametrize(
    "include_pair_overlap, overlapping_advanced, symmbreak_on_used, add_lower_bound_nb_teams",
    [
        (False, True, True, False),
        (True, True, False, False),
        (False, True, True, True),
    ],
)
def test_cpmpy_integer_params(
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)

    kwargs = dict(
        modelisation_allocation=ModelisationAllocationCP.INTEGER,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
    )

    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize(
    "include_pair_overlap, overlapping_advanced, symmbreak_on_used, add_lower_bound_nb_teams, include_all_binary_vars",
    [
        (True, False, False, False, False),
        (False, True, False, False, False),
        (True, False, True, False, False),
        (True, True, True, True, False),
        (True, False, True, False, True),
    ],
)
def test_cpmpy_binary_params(
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
    include_all_binary_vars,
):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)

    kwargs = dict(
        modelisation_allocation=ModelisationAllocationCP.BINARY,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
        include_all_binary_vars=include_all_binary_vars,
    )

    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_dispersion", list(ModelisationDispersion))
def test_cpmpy_dispersion(modelisation_dispersion):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=True)
    kwargs = dict(
        modelisation_allocation=ModelisationAllocationCP.BINARY,
        modelisation_dispersion=modelisation_dispersion,
    )
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationCP))
@pytest.mark.parametrize("multiobjective", [True, False])
def test_cpmpy_additional_constraints(modelisation_allocation, multiobjective):
    if modelisation_allocation == ModelisationAllocationCP.CNF_COMPATIBLE:
        pytest.skip(
            "cnf_compatible modelisation not available for CPMpyTeamAllocationSolver."
        )
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(
        instance, multiobjective=multiobjective
    )
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    disruption = generate_allocation_disruption(
        original_allocation_problem=allocation_problem,
        original_solution=sol,
        params_randomness=ParamsRandomness(
            lower_nb_disruption=1,
            upper_nb_disruption=1,
            lower_nb_teams=1,
            upper_nb_teams=2,
            lower_time=0,
            upper_time=600,
            duration_discrete_distribution=(
                [15, 30, 60, 120],
                [0.25, 0.25, 0.25, 0.25],
            ),
        ),
    )

    disrupted_problem = disruption["new_allocation_problem"]
    assert disrupted_problem.allocation_additional_constraint is not None
    solver = CPMpyTeamAllocationSolver(disrupted_problem)
    solver.init_model(modelisation_allocation=modelisation_allocation)
    sol = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    disrupted_problem.evaluate(sol)
    assert disrupted_problem.satisfy(sol)


def test_cpmpy_lexico():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance)
    solver = CPMpyTeamAllocationSolver(allocation_problem)
    solver.init_model()
    objectives = solver.get_lexico_objectives_available()
    for o in objectives:
        assert isinstance(o, str)
    assert len(objectives) == 2

    lexico = LexicoSolver(subsolver=solver, problem=allocation_problem)
    parameters_cp = ParametersCp.default()  # 1 process for exact iteration stop
    res = lexico.solve(
        objectives=objectives,
        subsolver_callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
    )
    # assert len(res) == len(objectives)  # not always a new solution found for each objective


@pytest.mark.parametrize(
    "modelisation_allocation, include_pair_overlap, overlapping_advanced, symmbreak_on_used, add_lower_bound_nb_teams, include_all_binary_vars",
    [
        (ModelisationAllocationCP.BINARY, True, True, True, True, True),
        (ModelisationAllocationCP.CNF_COMPATIBLE, True, True, True, True, True),
    ],
)
def test_cpmpy_storeconstraintinfo(
    modelisation_allocation,
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
    include_all_binary_vars,
):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    allocation_problem = parse_to_allocation_problem(instance, multiobjective=True)
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(allocation_problem)
    kwargs = dict(
        modelisation_allocation=modelisation_allocation,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
        include_all_binary_vars=include_all_binary_vars,
    )
    solver.init_model(**kwargs)
    # check solve
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    sol = res.get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)
