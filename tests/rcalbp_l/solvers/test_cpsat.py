#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcalbp_l.problem import RCALBPLSolution
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver


@fixture
def solver(problem):
    """Create and initialize a CP-SAT solver."""
    solver = CpSatRCALBPLSolver(problem)
    solver.init_model()
    return solver


def test_cpsat_solver_basic(problem, solver):
    """Test basic CP-SAT solving with a short time limit."""
    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=p,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    # Check that we got at least one solution
    assert len(result_storage) > 0
    assert len(result_storage) <= 1

    sol: RCALBPLSolution = result_storage[-1][0]

    # Check that solution is feasible
    assert problem.satisfy(sol), "Solution should be feasible"

    # Check solution structure
    assert sol.wks is not None
    assert sol.raw is not None
    assert sol.start is not None
    assert sol.cyc is not None

    # Check that all tasks are assigned
    assert len(sol.wks) == problem.nb_tasks

    # Check that cycle times are within bounds
    for p_idx in problem.periods:
        assert sol.cyc[p_idx] >= problem.c_target
        assert sol.cyc[p_idx] <= problem.c_max


def test_cpsat_solver_with_heuristic(problem):
    """Test CP-SAT solver with heuristic constraints."""
    solver = CpSatRCALBPLSolver(problem)
    solver.init_model(add_heuristic_constraint=True)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=p,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol)


def test_cpsat_solver_minimize_cycle_time(problem):
    """Test CP-SAT solver with minimize_used_cycle_time option."""
    solver = CpSatRCALBPLSolver(problem)
    solver.init_model(minimize_used_cycle_time=True, add_heuristic_constraint=False)

    p = ParametersCp.default_cpsat()
    p.nb_process = 4

    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=p,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )

    assert len(result_storage) > 0
    sol: RCALBPLSolution = result_storage[-1][0]
    assert problem.satisfy(sol)


def test_cpsat_early_stopping(problem, solver):
    """Test that early stopping callback works."""
    result_storage = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=2)],
        time_limit=30,
    )

    # Should stop after at most 3 iterations
    assert len(result_storage) <= 3

    if len(result_storage) > 0:
        sol: RCALBPLSolution = result_storage[-1][0]
        assert problem.satisfy(sol)
