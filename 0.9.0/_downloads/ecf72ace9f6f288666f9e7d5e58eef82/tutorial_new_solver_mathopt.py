#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Callable

from ortools.math_opt.python import mathopt
from tutorial_new_problem import MyKnapsackProblem, MyKnapsackSolution

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.lp_tools import OrtoolsMathOptMilpSolver


class MyKnapsackMathOptSolver(OrtoolsMathOptMilpSolver):
    problem: MyKnapsackProblem  # will be set by SolverDO.__init__(), useful to help the IDE typing correctly

    def init_model(self, **kwargs: Any) -> None:
        """Create mathopt `model` to encode the knapsack problem."""
        self.model = mathopt.Model()
        self.variables = [
            self.model.add_binary_variable(name=f"x_{i}")
            for i in range(len(self.problem.items))
        ]
        # add weight constraint
        total_weight = mathopt.LinearSum(
            self.variables[i] * weight
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.model.add_linear_constraint(total_weight <= self.problem.max_capacity)
        # maximize value
        total_value = mathopt.LinearSum(
            self.variables[i] * value
            for i, (value, weight) in enumerate(self.problem.items)
        )
        self.model.maximize(total_value)

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """Translate the mathopt solution into a d-o solution

        Args:
            get_var_value_for_current_solution: mapping a mathopt variable to its value in the solution
            get_obj_value_for_current_solution: returning the mathopt objective value

        Returns:

        """
        return MyKnapsackSolution(
            problem=self.problem,
            list_taken=[
                get_var_value_for_current_solution(var)
                >= 0.5  # represented by a float between 0. and 1.
                for var in self.variables
            ],
        )

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        assert isinstance(solution, MyKnapsackSolution)
        return {
            var: float(taken) for var, taken in zip(self.variables, solution.list_taken)
        }


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
    solver = MyKnapsackMathOptSolver(problem=problem)
    solver.init_model()

    # warm start (NB: not working with all mathopt subsolvers)
    solver.set_warm_start(
        MyKnapsackSolution(problem=problem, list_taken=[False, False, False, True])
    )

    # solve with a logging callback and GSCIP subsolver
    result_storage = solver.solve(
        callbacks=[ObjectiveLogger()], mathopt_solver_type=mathopt.SolverType.GSCIP
    )

    # display each solution: first solution the one fixed by warm start
    for i_sol, (sol, fit) in enumerate(result_storage):
        items_taken_indices = [i for i, taken in enumerate(sol.list_taken) if taken]
        print(f"\nSolution #{i_sol}")
        print(f"Fitness: {fit}")
        print(f"Taking items n°: {items_taken_indices}")

        # check solution satisfies the problem
        assert problem.satisfy(sol)
