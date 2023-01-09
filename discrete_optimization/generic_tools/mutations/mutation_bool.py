#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tools.do_mutation import LocalMove, Mutation
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    TypeAttribute,
)
from discrete_optimization.generic_tools.mutations.mutation_util import (
    get_attribute_for_type,
)


class BitFlipMove(LocalMove):
    def __init__(self, attribute: str, list_index_flip: Iterable[int]):
        self.attribute = attribute
        self.list_index_flip = list_index_flip

    def apply_local_move(self, solution: Solution) -> Solution:
        l = getattr(solution, self.attribute)
        for index in self.list_index_flip:
            l[index] = 1 - l[index]
        return solution

    def backtrack_local_move(self, solution: Solution) -> Solution:
        return self.apply_local_move(solution)


class MutationBitFlip(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs: Any) -> "MutationBitFlip":
        return MutationBitFlip(problem, **kwargs)

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        probability_flip: float = 0.1,
    ):
        self.problem = problem
        self.probability_flip = probability_flip
        register = problem.get_attribute_register()
        if attribute is None:
            self.attribute = get_attribute_for_type(
                register, TypeAttribute.LIST_BOOLEAN
            )
        else:
            self.attribute = attribute
        self.length = register.dict_attribute_to_type[self.attribute]["n"]

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        indexes = np.where(np.random.random(self.length) <= self.probability_flip)
        move = BitFlipMove(self.attribute, indexes[0])
        return move.apply_local_move(solution), move

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, move = self.mutate(solution)
        f = self.problem.evaluate(s)
        return s, move, f
