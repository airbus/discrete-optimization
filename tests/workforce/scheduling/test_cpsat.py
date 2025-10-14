#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.workforce.commons.fairness_modeling import (
    ModelisationDispersion,
)
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingSolution
from discrete_optimization.workforce.scheduling.solvers.alloc_scheduling_lb import (
    ApproximateBoundAllocScheduling,
    BoundResourceViaRelaxedProblem,
    LBoundAllocScheduling,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.utils import (
    compute_changes_between_solution,
    plotly_schedule_comparison,
)


@pytest.fixture()
def problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    return parse_json_to_problem(instance)


def test_cpsat(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly
    # test with callback
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
        time_limit=5,
    )
    assert len(res) == 1
    sol = res[-1][0]
    assert problem.satisfy(sol)
    problem.evaluate(sol)

    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=2)],
        parameters_cp=parameters_cp,
        time_limit=10,
    )
    # assert len(res) == 2  # not always 2 solutions found
    sol = res[-1][0]
    assert problem.satisfy(sol)
    problem.evaluate(sol)

    # test warm-start
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    solver.set_warm_start(solution=sol)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
        time_limit=10,
    )
    sol2 = res[0][0]
    assert (sol.schedule == sol2.schedule).all()
    assert (sol.allocation == sol2.allocation).all()

    # generate disruption
    d = generate_scheduling_disruption(
        original_scheduling_problem=problem,
        original_solution=sol,
        list_drop_resource=None,
        params_randomness=ParamsRandomness(
            lower_nb_disruption=1,
            upper_nb_disruption=2,
            lower_nb_teams=1,
            upper_nb_teams=1,
        ),
    )
    solver_relaxed = CPSatAllocSchedulingSolver(d["scheduling_problem"])
    solver_relaxed.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
        ],
        additional_constraints=d["additional_constraint_scheduling"],
        optional_activities=True,
        base_solution=sol,
    )
    res = solver_relaxed.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
        time_limit=5,
    )

    # compare solutions
    changes = compute_changes_between_solution(solution_a=sol, solution_b=res[-1][0])
    for key in ["nb_reallocated", "nb_shift", "mean_shift", "sum_shift", "max_shift"]:
        assert key in changes

    # plots
    plotly_schedule_comparison(
        base_solution=sol,
        updated_solution=res[-1][0],
        problem=d["scheduling_problem"],
        use_color_map_per_task=False,
        color_map_per_task={},
        plot_team_breaks=True,
    )


@pytest.mark.parametrize(
    "symmbreak_on_used, optional_activities, adding_redundant_cumulative, add_lower_bound, lower_bound_method",
    [
        (False, False, False, False, None),
        (False, True, False, False, None),
        (True, False, False, False, None),
        (False, False, True, False, None),
        (
            False,
            False,
            False,
            True,
            SubBrick(
                BoundResourceViaRelaxedProblem,
                kwargs=dict(adding_precedence_constraint=False, time_limit=2),
            ),
        ),
        (
            False,
            False,
            False,
            True,
            SubBrick(
                BoundResourceViaRelaxedProblem,
                kwargs=dict(adding_precedence_constraint=True, time_limit=2),
            ),
        ),
        (
            False,
            False,
            False,
            True,
            SubBrick(
                LBoundAllocScheduling,
                kwargs=dict(),
            ),
        ),
        (
            False,
            False,
            False,
            True,
            SubBrick(
                ApproximateBoundAllocScheduling,
                kwargs=dict(),
            ),
        ),
    ],
)
def test_cpsat_params(
    problem,
    symmbreak_on_used,
    optional_activities,
    adding_redundant_cumulative,
    add_lower_bound,
    lower_bound_method,
):
    kwargs = dict(
        symmbreak_on_used=symmbreak_on_used,
        optional_activities=optional_activities,
        adding_redundant_cumulative=adding_redundant_cumulative,
        add_lower_bound=add_lower_bound,
        lower_bound_method=lower_bound_method,
    )
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly

    solver = CPSatAllocSchedulingSolver(problem, **kwargs)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.NB_DONE_AC], **kwargs
    )
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
        **kwargs,
    )
    assert len(res) == 1
    sol = res[-1][0]
    problem.evaluate(sol)
    if not optional_activities:
        assert problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_dispersion", list(ModelisationDispersion))
def test_cpsat_modelisation_dispersion(problem, modelisation_dispersion):
    kwargs = dict(
        modelisation_dispersion=modelisation_dispersion,
    )
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly

    solver = CPSatAllocSchedulingSolver(problem, **kwargs)
    solver.init_model(
        objectives=[ObjectivesEnum.DISPERSION],
        adding_redundant_cumulative=True,
        **kwargs,
    )
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
        **kwargs,
    )
    assert len(res) == 1
    sol = res[-1][0]
    assert problem.satisfy(sol)
    problem.evaluate(sol)


def test_cpsat_set_model_obj_aggregated(problem):
    solver = CPSatAllocSchedulingSolver(problem)

    # solution to compare with for DELTA_TO_EXISTING_SOLUTION
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
    )
    base_solution = res.get_best_solution()

    objectives = [
        ObjectivesEnum.NB_TEAMS,
        ObjectivesEnum.MAKESPAN,
        ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
    ]
    objs_weights = [
        (ObjectivesEnum.NB_TEAMS, 0.1),
        ("MAKESPAN", 0.5),
        ("max_delta_schedule", 1.5),
    ]
    solver.init_model(objectives=objectives, base_solution=base_solution)
    solver.set_model_obj_aggregated(objs_weights=objs_weights)
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly
    # test with callback
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
        time_limit=5,
    )
    assert len(res) == 1
    sol = res[-1][0]
    assert problem.satisfy(sol)
    problem.evaluate(sol)


def test_cpsat_lexico(problem):
    solver = CPSatAllocSchedulingSolver(problem)

    # solution to compare with for DELTA_TO_EXISTING_SOLUTION
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
    )
    base_solution = res.get_best_solution()

    # objectives to consider
    objectives = list(ObjectivesEnum)

    solver.init_model(
        objectives=objectives, base_solution=base_solution, optional_activities=True
    )
    objectives_available = solver.get_lexico_objectives_available()
    for o in objectives_available:
        assert isinstance(o, str)
    assert (
        len(solver.get_lexico_objectives_available()) == len(objectives) + 4
    )  # DELTA_TO_EXISTING_SOLUTION => add 4 sous-obj

    # generic lexico solver
    lexico_solver = LexicoSolver(problem=problem, subsolver=solver)
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly
    objectives = ["MAKESPAN", "NB_TEAMS", "sum_delta_schedule"]
    res = lexico_solver.solve(
        subsolver_callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=10,
        parameters_cp=parameters_cp,
        objectives=objectives,
    )
    # assert len(res) == len(objectives)  # not always a new solution found for each objective


def test_task_constraint(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    sol: AllocSchedulingSolution = res.get_best_solution()
    print(problem.tasks_list)
    print(problem.unary_resources_list)
    i_task = 0
    print(sol.schedule[i_task, :])
    print(sol.allocation[i_task])
    print(problem.index_to_task[i_task])

    task = "80719"
    start_or_end = StartOrEnd.START
    sign = SignEnum.UEQ
    time = 15

    # before adding the constraint, not already satisfied
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # add constraint: should be now satisfied
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # check constraints can be effectively removed
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_chaining_constraint(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    task1 = "80719"
    task2 = "21963"

    # before adding the constraint, not already satisfied
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # add constraint
    cstrs = solver.add_constraint_chaining_tasks(task1=task1, task2=task2)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # remove constraint
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)


def test_objective_global_makespan(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()

    objective = solver.get_global_makespan_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sol.get_max_end_time()


def test_objective_subtasks_makespan(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()
    subtasks = ["80719", "21963"]

    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == max(
        sol.get_end_time(task) for task in subtasks
    )


def test_objective_subtasks_sum_starts(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()
    subtasks = ["80719", "21963"]
    objective = solver.get_subtasks_sum_start_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        sol.get_start_time(task) for task in subtasks
    )


def test_objective_subtasks_sum_ends(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()
    subtasks = ["80719", "21963"]
    objective = solver.get_subtasks_sum_end_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        sol.get_end_time(task) for task in subtasks
    )


def test_objective_nb_tasks_done(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()
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


def test_objective_nb_unary_resources_used(problem):
    solver = CPSatAllocSchedulingSolver(problem)
    sol: AllocSchedulingSolution
    solver.init_model()
    objective = solver.get_nb_unary_resources_used_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        max(sol.is_allocated(task, unary_resource) for task in problem.tasks_list)
        for unary_resource in problem.unary_resources_list
    )


def test_constraint_nb_usages(problem):
    solver = CPSatAllocSchedulingSolver(
        problem=problem,
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    nb_usages_total = sol.compute_nb_unary_resource_usages()
    unary_resource = "adba7"
    nb_usages = sol.compute_nb_unary_resource_usages(unary_resources=(unary_resource,))

    # constraint on total nb usages: infeasible (by hypothesis exactly 1 team by task)
    constraints = solver.add_constraint_on_total_nb_usages(
        sign=SignEnum.UP, target=nb_usages_total
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol is None

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


def test_constraint_nb_allocation_changes(problem):
    solver = CPSatAllocSchedulingSolver(
        problem=problem,
    )
    solver.init_model()
    ref: AllocSchedulingSolution
    sol: AllocSchedulingSolution

    # get a ref anti-optimal (we maximze nb_teams instead of minimizing it)
    solver.cp_model.maximize(solver.get_nb_unary_resources_used_variable())
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=3)])
    ref, _ = res[-1]

    # get back initial model
    solver.init_model()

    nb_changes_max = 2
    # w/o constraint
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=3)])
    sol, _ = res[-1]
    assert sol.compute_nb_allocation_changes(ref) > nb_changes_max
    # with constraint
    solver.add_constraint_on_nb_allocation_changes(ref=ref, nb_changes=nb_changes_max)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=3)])
    sol, _ = res[-1]
    assert sol.compute_nb_allocation_changes(ref) <= nb_changes_max
