#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import (
    ListBoolean,
    Problem,
    Solution,
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


class BitFlipMutation(SingleAttributeMutation):
    attribute_type_cls = ListBoolean
    attribute_type: ListBoolean

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        probability_flip: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.probability_flip = probability_flip
        self.length = self.attribute_type.length

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        indexes = np.where(np.random.random(self.length) <= self.probability_flip)
        move = BitFlipMove(self.attribute, indexes[0])
        sol = solution.lazy_copy()
        return move.apply_local_move(sol), move
