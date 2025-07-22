#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
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


def test_cpsat():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
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

    # do not succeed always in finding 2 solutions: try 5 times
    for _ in range(5):
        res = solver.solve(
            callbacks=[NbIterationStopper(nb_iteration_max=2)],
            parameters_cp=parameters_cp,
            time_limit=10,
        )
        if len(res) == 2:
            break
    assert len(res) == 2
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
    symmbreak_on_used,
    optional_activities,
    adding_redundant_cumulative,
    add_lower_bound,
    lower_bound_method,
):
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])

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
        **kwargs
    )
    assert len(res) == 1
    sol = res[-1][0]
    problem.evaluate(sol)
    if not optional_activities:
        assert problem.satisfy(sol)


@pytest.mark.parametrize("modelisation_dispersion", list(ModelisationDispersion))
def test_cpsat_modelisation_dispersion(modelisation_dispersion):
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])

    kwargs = dict(
        modelisation_dispersion=modelisation_dispersion,
    )
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly

    solver = CPSatAllocSchedulingSolver(problem, **kwargs)
    solver.init_model(
        objectives=[ObjectivesEnum.DISPERSION],
        adding_redundant_cumulative=True,
        **kwargs
    )
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=5,
        parameters_cp=parameters_cp,
        **kwargs
    )
    assert len(res) == 1
    sol = res[-1][0]
    assert problem.satisfy(sol)
    problem.evaluate(sol)


def test_cpsat_set_model_obj_aggregated():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
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


def test_cpsat_lexico():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
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
    assert len(res) == len(objectives)
