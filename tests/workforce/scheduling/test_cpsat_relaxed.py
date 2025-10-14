#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
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
from discrete_optimization.workforce.scheduling.solvers.cpsat_relaxed import (
    CPSatAllocSchedulingSolverCumulative,
)


def test_cpsat_relaxed():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)

    solver = CPSatAllocSchedulingSolverCumulative(problem)
    solver.init_model()

    # solve + cb
    parameters_cp = ParametersCp.default()
    cb = NbIterationTracker()
    cb_scheduling = NbIterationTracker()
    cb_allocation = NbIterationTracker()
    res = solver.solve(
        callbacks=[cb],
        kwargs_scheduling=dict(
            parameters_cp=parameters_cp,
            time_limit=10,
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_scheduling],
        ),
        kwargs_allocation=dict(
            parameters_cp=parameters_cp,
            time_limit=10,
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_allocation],
        ),
    )
    assert len(res) == 1  # only last solution is kept for this solver
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    problem.evaluate(sol)

    assert cb.nb_iteration == 2
    assert cb_scheduling.nb_iteration == 1
    assert cb_allocation.nb_iteration == 1


@pytest.mark.parametrize(
    "optional_activities, adding_redundant_cumulative, add_lower_bound, lower_bound_method",
    [
        (False, True, False, None),
        (True, False, False, None),
        (
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
            True,
            SubBrick(
                BoundResourceViaRelaxedProblem,
                kwargs=dict(adding_precedence_constraint=True, time_limit=2),
            ),
        ),
        (
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
            True,
            SubBrick(
                ApproximateBoundAllocScheduling,
                kwargs=dict(),
            ),
        ),
    ],
)
def test_cpsat_relaxed_params(
    optional_activities,
    adding_redundant_cumulative,
    add_lower_bound,
    lower_bound_method,
):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)

    kwargs = dict(
        optional_activities=optional_activities,
        adding_redundant_cumulative=adding_redundant_cumulative,
        add_lower_bound=add_lower_bound,
        lower_bound_method=lower_bound_method,
    )

    solver = CPSatAllocSchedulingSolverCumulative(problem)
    solver.init_model(
        objectives=[
            ObjectivesEnum.NB_TEAMS,
            ObjectivesEnum.MAKESPAN,
            ObjectivesEnum.NB_DONE_AC,
        ],
        **kwargs,
    )
    # solve + cb
    parameters_cp = ParametersCp.default()
    cb = NbIterationTracker()
    cb_scheduling = NbIterationTracker()
    cb_allocation = NbIterationTracker()
    res = solver.solve(
        callbacks=[cb],
        kwargs_scheduling=dict(
            parameters_cp=parameters_cp,
            time_limit=10,
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_scheduling],
        ),
        kwargs_allocation=dict(
            parameters_cp=parameters_cp,
            time_limit=10,
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_allocation],
        ),
    )
    assert len(res) == 1  # only last solution is kept for this solver
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    problem.evaluate(sol)

    assert cb.nb_iteration == 2
    assert cb_scheduling.nb_iteration == 1
    assert cb_allocation.nb_iteration == 1


def test_cpsat_relaxed_set_model_obj_aggregated():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)

    solver = CPSatAllocSchedulingSolverCumulative(problem)
    kwargs = dict(optional_activities=True)
    objectives = [ObjectivesEnum.NB_TEAMS, ObjectivesEnum.MAKESPAN]
    objs_weights = [
        (ObjectivesEnum.NB_TEAMS, 0.1),
        ("MAKESPAN", 0.5),
    ]
    solver.init_model(objectives=objectives, **kwargs)
    solver.set_model_obj_aggregated(objs_weights=objs_weights)
    # solve + cb
    parameters_cp = ParametersCp.default()
    cb = NbIterationTracker()
    cb_scheduling = NbIterationTracker()
    cb_allocation = NbIterationTracker()
    res = solver.solve(
        callbacks=[cb],
        kwargs_scheduling=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_scheduling]
        ),
        kwargs_allocation=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_allocation]
        ),
        parameters_cp=parameters_cp,  # no multiprocess to have iteration stopper working exactly
        time_limit=10,
    )
    assert len(res) == 1  # only last solution is kept for this solver
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    problem.evaluate(sol)

    assert cb.nb_iteration == 2
    assert cb_scheduling.nb_iteration == 1
    assert cb_allocation.nb_iteration == 1

    # other solution for warmstart?
    res = solver.solve(
        kwargs_scheduling=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=2), cb_scheduling]
        ),
        kwargs_allocation=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=2), cb_allocation]
        ),
        parameters_cp=parameters_cp,  # no multiprocess to have iteration stopper working exactly
        time_limit=10,
    )
    sol1 = res.get_best_solution()

    # warm start
    solver.set_warm_start(sol1)
    res = solver.solve(
        callbacks=[cb],
        kwargs_scheduling=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_scheduling]
        ),
        kwargs_allocation=dict(
            callbacks=[NbIterationStopper(nb_iteration_max=1), cb_allocation]
        ),
        parameters_cp=parameters_cp,  # no multiprocess to have iteration stopper working exactly
        time_limit=10,
    )
    sol2 = res.get_best_solution()
    # assert problem.evaluate(sol2) == problem.evaluate(sol1)  # not always same sol


@pytest.mark.parametrize(
    "obj", CPSatAllocSchedulingSolverCumulative.not_implemented_objectives
)
def test_cpsat_relaxed_not_impl_objs(obj):
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = CPSatAllocSchedulingSolverCumulative(problem)
    with pytest.raises(NotImplementedError):
        solver.init_model(objectives=[obj])


def test_cpsat_relaxed_lexico():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = CPSatAllocSchedulingSolverCumulative(problem)

    # objectives to consider
    objectives = [
        o
        for o in ObjectivesEnum
        if o not in CPSatAllocSchedulingSolverCumulative.not_implemented_objectives
    ]

    solver.init_model(objectives=objectives, optional_activities=True)
    objectives_available = solver.get_lexico_objectives_available()
    for o in objectives_available:
        assert isinstance(o, str)

    assert len(solver.get_lexico_objectives_available()) == len(objectives)

    # generic lexico solver
    lexico_solver = LexicoSolver(problem=problem, subsolver=solver)
    parameters_cp = ParametersCp.default()  # keep 1 process to check stopper properly
    res = lexico_solver.solve(
        kwargs_scheduling=dict(callbacks=[NbIterationStopper(nb_iteration_max=1)]),
        kwargs_allocation=dict(callbacks=[NbIterationStopper(nb_iteration_max=1)]),
        parameters_cp=parameters_cp,
        time_limit=10,
    )
    # assert len(res) == len(objectives)  # not always a new solution found for each objective
