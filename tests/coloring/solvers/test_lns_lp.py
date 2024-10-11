#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.lns_lp import (
    FixColorsGurobiConstraintHandler,
    FixColorsMathOptConstraintHandler,
)
from discrete_optimization.coloring.solvers.lp import (
    GurobiColoringSolver,
    MathOptColoringSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.lns_mip import LnsMilp
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "solver_cls, constraint_handler_cls",
    [
        (MathOptColoringSolver, FixColorsMathOptConstraintHandler),
        (GurobiColoringSolver, FixColorsGurobiConstraintHandler),
    ],
)
def test_lns_lp_coloring(solver_cls, constraint_handler_cls):
    if issubclass(solver_cls, GurobiMilpSolver) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    solver = solver_cls(color_problem)
    solver.init_model()
    solver_lns = LnsMilp(
        problem=color_problem,
        subsolver=solver,
        initial_solution_provider=None,
        constraint_handler=constraint_handler_cls(
            problem=color_problem, fraction_to_fix=0.3
        ),
        post_process_solution=None,
    )
    logging.info("Starting solve")
    nb_iteration_lns = 5
    result_store = solver_lns.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=nb_iteration_lns,
        callbacks=[
            NbIterationTracker(step_verbosity_level=logging.INFO),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            TimerStopper(total_seconds=30),
        ],
        time_limit_subsolver=20,
        stop_first_iteration_if_optimal=False,
    )
    print("Status solver : ", solver.status_solver)
    solution, fit = result_store.get_best_solution_fit()
    # plot_coloring_solution(solution)
    # plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    assert color_problem.satisfy(solution)
    assert len(result_store) == nb_iteration_lns
