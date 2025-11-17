#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)

Item = tuple[int, int]  # value, weight


class MyKnapsackProblem(Problem):
    def __init__(self, items: list[Item], max_capacity: int):
        self.items = items
        self.max_capacity = max_capacity

    def get_solution_type(self) -> type[Solution]:
        """Specify associated solution type."""
        return MyKnapsackSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Specify the different objectives and if we need to aggregate them."""
        return ObjectiveRegister(
            dict_objective_to_doc=dict(
                # total value of taken items: main objective
                value=ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
                # weight violation (how much we exceed the max capactity): penalty to be removed with a big coefficient
                weight_violation=ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-1000.0
                ),
            ),
            objective_handling=ObjectiveHandling.AGGREGATE,  # aggregate both objective
            objective_sense=ModeOptim.MAXIMIZATION,  # maximize resulting objective
        )

    def get_attribute_register(self) -> EncodingRegister:
        """Describe attributes of a solution.

        To be used by evolutionary solvers to choose the appropriate mutations
        without implementing a dedicated one.

        """
        return EncodingRegister(
            dict_attribute_to_type={
                "list_taken": {
                    "name": "list_taken",
                    "type": [TypeAttribute.LIST_BOOLEAN],
                    "n": len(self.items),
                }
            }
        )

    def evaluate(self, variable: Solution) -> dict[str, float]:
        """Evaluate the objectives corresponding to the solution.

        The objectives must match the ones defined in `get_objective_register`.

        """
        if not isinstance(variable, MyKnapsackSolution):
            raise ValueError("variable must be a `MyKnapsackSolution`")
        value = self.compute_total_value(variable)
        weight = self.compute_total_weight(variable)
        return dict(value=value, weight_violation=max(0, weight - self.max_capacity))

    def satisfy(self, variable: Solution) -> bool:
        """Check that the solution satisfies the problem.

        Check the weight violation.

        """
        if not isinstance(variable, MyKnapsackSolution):
            return False
        return self.compute_total_weight(variable) <= self.max_capacity

    def compute_total_weight(self, variable: MyKnapsackSolution) -> int:
        """Compute the total weight of taken items."""
        return sum(
            taken * weight
            for taken, (value, weight) in zip(variable.list_taken, self.items)
        )

    def compute_total_value(self, variable: MyKnapsackSolution) -> int:
        """Compute the total value of taken items."""
        return sum(
            taken * value
            for taken, (value, weight) in zip(variable.list_taken, self.items)
        )


class MyKnapsackSolution(Solution):
    """Solution class for MyKnapsackProblem.

    Args:
        problem: problem instance for which this is a solution
        list_taken: list of booleans specifying if corresponding item has been taken.
            Must be of same length as problem.items

    """

    problem: MyKnapsackProblem  # help the IDE to type correctly

    def __init__(self, problem: MyKnapsackProblem, list_taken: list[bool]):
        super().__init__(problem=problem)
        self.list_taken = list_taken

    def copy(self) -> Solution:
        """Deep copy the solution."""
        return MyKnapsackSolution(
            problem=self.problem, list_taken=list(self.list_taken)
        )

    def lazy_copy(self) -> Solution:
        """Shallow copy the solution.

        Not mandatory to implement but can increase the speed of evolutionary algorithms.

        """
        return MyKnapsackSolution(problem=self.problem, list_taken=self.list_taken)


if __name__ == "__main__":
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

    # sub-optimal solution (total_weight = 2 <= 10, total_value = 5)
    solution_0 = MyKnapsackSolution(
        problem=problem, list_taken=[True, False, False, False]
    )
    assert problem.satisfy(solution_0)
    print(problem.evaluate(solution_0))

    # violating solution (total_weight > 10)
    solution_1 = MyKnapsackSolution(
        problem=problem, list_taken=[True, False, False, True]
    )
    assert not problem.satisfy(solution_1)
    print(problem.evaluate(solution_1))

    # optimal solution (total_value = 8)
    solution_2 = MyKnapsackSolution(
        problem=problem, list_taken=[False, True, False, True]
    )
    assert problem.satisfy(solution_2)
    print(problem.evaluate(solution_2))
