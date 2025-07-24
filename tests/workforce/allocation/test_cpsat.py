from stringprep import in_table_c3

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.sequential_solvers_callback import (
    RetrieveSubRes,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsCpsatCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
    ModelisationDispersion,
)
from discrete_optimization.workforce.allocation.utils import plot_allocation_solution
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_allocation_disruption,
)


def test_cpsat_multiobj():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1], multiobjective=True)
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model()
    parameters_cp = ParametersCp.default()  # 1 process for exact iteration stop
    # check solve + callback (1 iteration)
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)
    assert len(res) == 1
    # check plot
    plot_allocation_solution(
        problem=allocation_problem,
        sol=sol,
        display=False,
    )


def test_cpsat_monoobj():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1], multiobjective=False)
    solver = CpsatTeamAllocationSolver(allocation_problem)
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


@pytest.mark.parametrize(
    "include_pair_overlap, overlapping_advanced, symmbreak_on_used, add_lower_bound_nb_teams",
    [
        (True, False, False, False),
        (False, True, False, False),
        (True, False, True, False),
        (True, True, True, True),
    ],
)
def test_cpsat_integer_params(
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])

    kwargs = dict(
        modelisation_allocation=ModelisationAllocationOrtools.INTEGER,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
    )

    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize(
    "modelisation_allocation, include_pair_overlap, overlapping_advanced, symmbreak_on_used, add_lower_bound_nb_teams, include_all_binary_vars",
    [
        (ModelisationAllocationOrtools.BINARY, True, False, False, False, False),
        (ModelisationAllocationOrtools.BINARY, False, True, False, False, False),
        (ModelisationAllocationOrtools.BINARY, True, False, True, False, False),
        (ModelisationAllocationOrtools.BINARY, True, True, True, True, False),
        (ModelisationAllocationOrtools.BINARY, True, False, True, False, True),
        (
            ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES,
            True,
            True,
            True,
            True,
            False,
        ),
    ],
)
def test_cpsat_binary_params(
    modelisation_allocation,
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
    include_all_binary_vars,
):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])

    kwargs = dict(
        modelisation_allocation=modelisation_allocation,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
        include_all_binary_vars=include_all_binary_vars,
    )

    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    if (
        modelisation_allocation
        != ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_dispersion", list(ModelisationDispersion))
def test_cpsat_dispersion(modelisation_dispersion):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1], multiobjective=True)
    kwargs = dict(
        modelisation_allocation=ModelisationAllocationOrtools.BINARY,
        modelisation_dispersion=modelisation_dispersion,
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_cpsat_warm_start(modelisation_allocation):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1], multiobjective=True)
    kwargs = dict(
        modelisation_allocation=modelisation_allocation,
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=2)]
    ).get_best_solution()

    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(**kwargs)
    solver.set_warm_start(sol)
    sol1 = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    # assert sol1.allocation == sol.allocation # not always equality


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
@pytest.mark.parametrize("multiobjective", [True, False])
def test_cpsat_additional_constraints(modelisation_allocation, multiobjective):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(
        instances[1], multiobjective=multiobjective
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
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
    solver = CpsatTeamAllocationSolver(disrupted_problem)
    sol = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        modelisation_allocation=modelisation_allocation,
    ).get_best_solution()
    disrupted_problem.evaluate(sol)
    if (
        modelisation_allocation
        != ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        assert disrupted_problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_cpsat_delta(modelisation_allocation):
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    base_solution = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    solver.init_model(
        modelisation_allocation=modelisation_allocation, base_solution=base_solution
    )
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    allocation_problem.evaluate(sol)
    if (
        modelisation_allocation
        != ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        assert allocation_problem.satisfy(sol)


def test_cpsat_agg_obj():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model()
    solver.set_model_obj_aggregated([("nb_teams", 10), ("duration", 5)])

    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    allocation_problem.evaluate(sol)
    assert allocation_problem.satisfy(sol)


def test_cpsat_lexico():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    objectives = solver.get_lexico_objectives_available()
    for o in objectives:
        assert isinstance(o, str)
    assert len(objectives) == 2

    base_solution = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    solver.init_model(base_solution=base_solution)
    objectives = solver.get_lexico_objectives_available()
    for o in objectives:
        assert isinstance(o, str)
    assert len(objectives) == 3

    lexico = LexicoSolver(subsolver=solver, problem=allocation_problem)
    parameters_cp = ParametersCp.default()  # 1 process for exact iteration stop
    res = lexico.solve(
        objectives=objectives,
        subsolver_callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
    )
    # assert len(res) == len(objectives)  # not always a new solution found for each objective


def test_cpsat_solve_n_best_solution():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    res = solver.solve_n_best_solution(
        time_limit=5,
        n_best_solution=3,
    )
    assert len(res) > 1


def test_cpsat_solve_n_best_solution_with_priority():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    res = solver.solve_n_best_solution(
        time_limit=5, n_best_solution=3, priority={0: 5, 2: 3}
    )
    assert len(res) > 1


def test_compute_task_relaxation_alternatives():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    res_final, res_optim = solver.compute_task_relaxation_alternatives(
        time_limit=5,
        time_limit_per_iteration=1,
    )
    # assert len(res_final) > len(res_optim)
    # assert len(res_optim)>1


def test_compute_sufficient_assumptions():
    instances = [p for p in get_data_available()]
    allocation_problem = parse_to_allocation_problem(instances[1])
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.compute_sufficient_assumptions(
        time_limit=5,
    )
