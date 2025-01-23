#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.toulbar import (
    ColoringConstraintHandlerToulbar,
    ToulbarColoringSolverForLns,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    InitialSolutionFromSolver,
)

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_lns_toulbar():
    file = [f for f in get_data_available() if "gc_50_9" in f][0]
    color_problem = parse_file(file)
    initial = InitialSolutionFromSolver(
        solver=GreedyColoringSolver(problem=color_problem),
        strategy=NxGreedyColoringMethod.best,
    )
    solver = ToulbarColoringSolverForLns(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=None,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=1,
    )
    lns = BaseLns(
        problem=color_problem,
        subsolver=solver,
        initial_solution_provider=initial,
        constraint_handler=ColoringConstraintHandlerToulbar(fraction_node=0.25),
    )
    res = lns.solve(
        nb_iteration_lns=100,
        time_limit_subsolver=5,
        callbacks=[TimerStopper(total_seconds=15)],
    )
    sol = res[-1][0]
    assert color_problem.satisfy(sol)
