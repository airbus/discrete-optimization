#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback
from tutorial_new_problem import (
    MyKnapsackProblem,
    MyKnapsackSolution,
)

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class MyKnapsackCpSatSolver(OrtoolsCpSatSolver):
    """CP-SAT solver for the knapsack problem."""

    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def init_model(self, **kwargs: Any) -> None:
        """Init the CP model."""
        super().init_model(**kwargs)  # initialize self.cp_model
        # create the boolean variables for each item
        self.variables = [
            self.cp_model.new_bool_var(name=f"x_{i}")
            for i in range(len(self.problem.items))
        ]
        # add weight constraint
        total_weight = sum(
            self.variables[i] * weight
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.cp_model.add(total_weight <= self.problem.max_capacity)
        # maximize value
        total_value = sum(
            self.variables[i] * value
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.cp_model.maximize(total_value)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        """Translate a cpsat solution into a d-o solution.

        Args:
            cpsolvercb:  the ortools callback called when the cpsat solver finds a new solution.

        Returns:

        """
        return MyKnapsackSolution(
            problem=self.problem,
            list_taken=[bool(cpsolvercb.Value(var)) for var in self.variables],
        )


if __name__ == "__main__":
    import logging

    from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger

    logging.basicConfig(level=logging.DEBUG)

    # instantiate a knapsack problem
    problem = MyKnapsackProblem(
        max_capacity=10,
        items=[
            (2, 5),  # item 0: value=2, weight=5
            (3, 1),  # item 1: value=3, weight=1
            (2, 4),  # item 2: value=2, weight=4
            (5, 9),  # item 3: value=5, weight=9
        ],
    )

    # instantiate the greedy solver
    solver = MyKnapsackCpSatSolver(problem=problem)

    # solve with a logging callback
    result_storage = solver.solve(callbacks=[ObjectiveLogger()])

    # display best solution
    sol, fit = result_storage.get_best_solution_fit()
    items_taken_indices = [i for i, taken in enumerate(sol.list_taken) if taken]
    print(f"Best fitness: {fit}")
    print(f"Taking items n°: {items_taken_indices}")

    # check solution satisfies the problem
    assert problem.satisfy(sol)
