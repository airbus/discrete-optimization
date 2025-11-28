#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

from discrete_optimization.generic_tools.do_problem import (
    AttributeType,
    Problem,
    Solution,
)


class LocalMove:
    @abstractmethod
    def apply_local_move(self, solution: Solution) -> Solution: ...

    @abstractmethod
    def backtrack_local_move(self, solution: Solution) -> Solution: ...


class LocalMoveDefault(LocalMove):
    """
    Not clever local move
    If you're lazy or don't have the choice,
    don't do in place modification of the previous solution, so that you can retrieve it directly.
    So the backward operator is then obvious.
    """

    def __init__(self, prev_solution: Solution, new_solution: Solution):
        self.prev_solution = prev_solution
        self.new_solution = new_solution

    def apply_local_move(self, solution: Solution) -> Solution:
        return self.new_solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        return self.prev_solution


class Mutation(ABC):
    @classmethod
    def build(cls, problem: Problem, **kwargs: Any) -> "Mutation":
        return cls(problem=problem, **kwargs)

    def __init__(self, problem: Problem, **kwargs: Any):
        self.problem = problem

    @abstractmethod
    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]: ...

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> tuple[Solution, LocalMove, dict[str, float]]:
        sol, move = self.mutate(solution=solution)
        res = self.problem.evaluate(sol)
        return sol, move, res


class SingleAttributeMutation(Mutation):
    attribute_type_cls: type[AttributeType]

    def __init__(self, problem, attribute: str | None = None, **kwargs: Any):
        super().__init__(problem, **kwargs)
        register = self.problem.get_attribute_register()
        if attribute is None:
            self.attribute = register.get_first_attribute_of_type(
                self.attribute_type_cls
            )
        else:
            self.attribute = attribute
        self.attribute_type = register[self.attribute]

    def __repr__(self):
        return f"{type(self).__name__}(attribute='{self.attribute}')"
