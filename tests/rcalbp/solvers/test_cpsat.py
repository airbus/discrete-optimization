#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.alb.rcalbp.solvers.cpsat import (
    CpSatRcAlbpSolver,
    ModelingShared,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp


def test_cpsat_folded_basic(simple_problem):
    """Test CP-SAT solver with FOLDED modeling."""
    solver = CpSatRcAlbpSolver(simple_problem)
    params_cp = ParametersCp.default_cpsat()

    result_storage = solver.solve(
        modeling=ModelingShared.FOLDED,
        parameters_cp=params_cp,
        time_limit=10,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs={"log_search_progress": False},
    )

    assert len(result_storage) > 0
    solution = result_storage.get_best_solution()
    assert simple_problem.satisfy(solution), "Solution should be feasible"
    assert len(solution.task_assignment) == simple_problem.nb_tasks


def test_cpsat_calendar_basic(simple_problem):
    """Test CP-SAT solver with CALENDAR modeling."""
    solver = CpSatRcAlbpSolver(simple_problem)
    params_cp = ParametersCp.default_cpsat()

    result_storage = solver.solve(
        modeling=ModelingShared.CALENDAR,
        parameters_cp=params_cp,
        time_limit=10,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs={"log_search_progress": False},
    )

    assert len(result_storage) > 0
    solution = result_storage.get_best_solution()
    assert simple_problem.satisfy(solution), "Solution should be feasible"


def test_cpsat_shared_resources_folded(shared_resource_problem):
    """Test FOLDED modeling with shared resources."""
    solver = CpSatRcAlbpSolver(shared_resource_problem)

    result_storage = solver.solve(
        modeling=ModelingShared.FOLDED,
        time_limit=15,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs={"log_search_progress": False},
    )

    if len(result_storage) > 0:
        solution = result_storage.get_best_solution()
        eval_result = shared_resource_problem.evaluate(solution)
        # Shared resource constraint should be satisfied
        assert eval_result.get("penalty_resource_shared", 0) == 0


def test_cpsat_shared_resources_calendar(shared_resource_problem):
    """Test CALENDAR modeling with shared resources."""
    solver = CpSatRcAlbpSolver(shared_resource_problem)

    result_storage = solver.solve(
        modeling=ModelingShared.CALENDAR,
        time_limit=15,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs={"log_search_progress": False},
    )

    if len(result_storage) > 0:
        solution = result_storage.get_best_solution()
        eval_result = shared_resource_problem.evaluate(solution)
        # Shared resource constraint should be satisfied
        assert eval_result.get("penalty_resource_shared", 0) == 0
