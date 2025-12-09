#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    ObjectiveSubproblem,
    TasksConstraintHandler,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver


@pytest.mark.parametrize(
    "objective_subproblem",
    (ObjectiveSubproblem.INITIAL_OBJECTIVE, ObjectiveSubproblem.MAKESPAN_SUBTASKS),
)
def test_lns(objective_subproblem, problem):
    subsolver = CpSatTspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)
    constraint_handler = TasksConstraintHandler(
        problem=problem,
        objective_subproblem=objective_subproblem,
    )
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = lns_solver.solve(
        nb_iteration_lns=3,
        time_limit_subsolver=5,
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)
