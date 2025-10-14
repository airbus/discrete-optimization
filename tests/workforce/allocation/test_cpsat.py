import pytest
from qiskit.quantum_info.analysis.make_observable import make_dict_observable

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.problem import TeamAllocationSolution
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


@pytest.fixture()
def problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    return parse_to_allocation_problem(instance, multiobjective=True)


def test_cpsat_multiobj(problem):
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model()
    parameters_cp = ParametersCp.default()  # 1 process for exact iteration stop
    # check solve + callback (1 iteration)
    res = solver.solve(
        time_limit=5,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    problem.evaluate(sol)
    assert problem.satisfy(sol)
    assert len(res) == 1
    # check plot
    plot_allocation_solution(
        problem=problem,
        sol=sol,
        display=False,
    )


def test_cpsat_monoobj(problem):
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model()
    sol = solver.solve(
        time_limit=5,
    ).get_best_solution()
    problem.evaluate(sol)
    assert problem.satisfy(sol)
    plot_allocation_solution(
        problem=problem,
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
    problem,
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
):
    kwargs = dict(
        modelisation_allocation=ModelisationAllocationOrtools.INTEGER,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
    )

    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    problem.evaluate(sol)
    assert problem.satisfy(sol)


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
    problem,
    modelisation_allocation,
    include_pair_overlap,
    overlapping_advanced,
    symmbreak_on_used,
    add_lower_bound_nb_teams,
    include_all_binary_vars,
):
    kwargs = dict(
        modelisation_allocation=modelisation_allocation,
        include_pair_overlap=include_pair_overlap,
        overlapping_advanced=overlapping_advanced,
        add_lower_bound_nb_teams=add_lower_bound_nb_teams,
        symmbreak_on_used=symmbreak_on_used,
        include_all_binary_vars=include_all_binary_vars,
    )

    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    problem.evaluate(sol)
    if (
        modelisation_allocation
        != ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        assert problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_dispersion", list(ModelisationDispersion))
def test_cpsat_dispersion(problem, modelisation_dispersion):
    kwargs = dict(
        modelisation_allocation=ModelisationAllocationOrtools.BINARY,
        modelisation_dispersion=modelisation_dispersion,
    )
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    problem.evaluate(sol)
    assert problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_cpsat_warm_start(problem, modelisation_allocation):
    kwargs = dict(
        modelisation_allocation=modelisation_allocation,
    )
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(**kwargs)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=2)]
    ).get_best_solution()

    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(**kwargs)
    solver.set_warm_start(sol)
    sol1 = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    # assert sol1.allocation == sol.allocation # not always equality


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
@pytest.mark.parametrize("multiobjective", [True, False])
def test_cpsat_additional_constraints(problem, modelisation_allocation, multiobjective):
    solver = CpsatTeamAllocationSolver(problem)
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    disruption = generate_allocation_disruption(
        original_allocation_problem=problem,
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
def test_cpsat_delta(problem, modelisation_allocation):
    solver = CpsatTeamAllocationSolver(problem)
    base_solution = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    solver.init_model(
        modelisation_allocation=modelisation_allocation, base_solution=base_solution
    )
    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    problem.evaluate(sol)
    if (
        modelisation_allocation
        != ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        assert problem.satisfy(sol)


def test_cpsat_agg_obj(problem):
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model()
    solver.set_model_obj_aggregated([("nb_teams", 10), ("duration", 5)])

    sol = solver.solve(
        time_limit=5, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    problem.evaluate(sol)
    assert problem.satisfy(sol)


def test_cpsat_lexico(problem):
    solver = CpsatTeamAllocationSolver(problem)
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

    lexico = LexicoSolver(subsolver=solver, problem=problem)
    parameters_cp = ParametersCp.default()  # 1 process for exact iteration stop
    res = lexico.solve(
        objectives=objectives,
        subsolver_callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
    )
    # assert len(res) == len(objectives)  # not always a new solution found for each objective


def test_cpsat_solve_n_best_solution(problem):
    solver = CpsatTeamAllocationSolver(problem)
    res = solver.solve_n_best_solution(
        time_limit=5,
        n_best_solution=3,
    )
    assert len(res) > 1


def test_cpsat_solve_n_best_solution_with_priority(problem):
    solver = CpsatTeamAllocationSolver(problem)
    res = solver.solve_n_best_solution(
        time_limit=5, n_best_solution=3, priority={0: 5, 2: 3}
    )
    assert len(res) > 1


def test_compute_task_relaxation_alternatives(problem):
    solver = CpsatTeamAllocationSolver(problem)
    res_final, res_optim = solver.compute_task_relaxation_alternatives(
        time_limit=5,
        time_limit_per_iteration=1,
    )
    # assert len(res_final) > len(res_optim)
    # assert len(res_optim)>1


def test_compute_sufficient_assumptions(problem):
    solver = CpsatTeamAllocationSolver(problem)
    solver.compute_sufficient_assumptions(
        time_limit=5,
    )


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_constraint_nb_usages(problem, modelisation_allocation):
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(modelisation_allocation=modelisation_allocation)
    sol: TeamAllocationSolution
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    nb_usages_total = sol.compute_nb_unary_resource_usages()
    unary_resource = "adba7"
    nb_usages = sol.compute_nb_unary_resource_usages(unary_resources=(unary_resource,))

    if modelisation_allocation in [
        ModelisationAllocationOrtools.BINARY,
        ModelisationAllocationOrtools.INTEGER,
    ]:
        # constraint on total nb usages: infeasible (by hypothesis exactly 1 team by task)
        constraints = solver.add_constraint_on_total_nb_usages(
            sign=SignEnum.UP, target=nb_usages_total
        )
        sol = solver.solve(
            callbacks=[NbIterationStopper(nb_iteration_max=1)], time_limit=5
        ).get_best_solution()
        assert sol is None
    else:
        target = int(nb_usages_total / 2)
        constraints = solver.add_constraint_on_total_nb_usages(
            sign=SignEnum.LESS, target=target
        )
        sol = solver.solve(
            callbacks=[NbIterationStopper(nb_iteration_max=1)], time_limit=5
        ).get_best_solution()
        assert sol.compute_nb_unary_resources_used() < target

    # constraint on a unary resource usage: increase team usage
    solver.remove_constraints(constraints)
    solver.add_constraint_on_unary_resource_nb_usages(
        sign=SignEnum.UP, target=nb_usages, unary_resource=unary_resource
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_unary_resource_usages(unary_resources=(unary_resource,))
        > nb_usages
    )


def prepare_solver_for_binary_optional(solver):
    if (
        solver.modelisation_allocation
        == ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        # init with dummy empty solution
        dummy_sol = TeamAllocationSolution(
            allocation=[None] * len(solver.problem.tasks_list), problem=solver.problem
        )
        solver.set_warm_start(dummy_sol)


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_constraint_nb_allocation_changes(problem, modelisation_allocation):
    solver = CpsatTeamAllocationSolver(problem)
    solver.init_model(modelisation_allocation=modelisation_allocation)
    sol: TeamAllocationSolution
    ref: TeamAllocationSolution

    # get a ref anti-optimal (we maximize nb_teams)
    obj = solver.get_nb_unary_resources_used_variable()
    solver.cp_model.maximize(obj)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=3)])
    ref, _ = res[-1]

    # get back initial model
    solver.init_model()

    if (
        modelisation_allocation
        == ModelisationAllocationOrtools.BINARY_OPTIONAL_ACTIVITIES
    ):
        nb_changes_max = 117
    else:
        nb_changes_max = 10
    # w/o constraint
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    print([sol.compute_nb_allocation_changes(ref) for sol, _ in res])
    sol, _ = res[-1]
    assert sol.compute_nb_allocation_changes(ref) > nb_changes_max
    # with constraint
    solver.add_constraint_on_nb_allocation_changes(ref=ref, nb_changes=nb_changes_max)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    sol, _ = res[-1]
    assert sol.compute_nb_allocation_changes(ref) <= nb_changes_max


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_objective_nb_tasks_done(problem, modelisation_allocation):
    solver = CpsatTeamAllocationSolver(problem)
    sol: TeamAllocationSolution
    solver.init_model(modelisation_allocation=modelisation_allocation)
    objective = -solver.get_nb_tasks_done_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == -sum(
        max(
            sol.is_allocated(task, unary_resource)
            for unary_resource in problem.unary_resources_list
        )
        for task in problem.tasks_list
    )


@pytest.mark.parametrize("modelisation_allocation", list(ModelisationAllocationOrtools))
def test_objective_nb_unary_resources_used(problem, modelisation_allocation):
    solver = CpsatTeamAllocationSolver(problem)
    sol: TeamAllocationSolution
    solver.init_model(modelisation_allocation=modelisation_allocation)
    objective = solver.get_nb_unary_resources_used_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        max(sol.is_allocated(task, unary_resource) for task in problem.tasks_list)
        for unary_resource in problem.unary_resources_list
    )
