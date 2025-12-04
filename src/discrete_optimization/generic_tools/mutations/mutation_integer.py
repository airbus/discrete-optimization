#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Optional

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.encoding_register import ListInteger


class IntegerMutation(SingleAttributeMutation):
    attribute_type_cls = ListInteger
    attribute_type: ListInteger

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        probability_flip: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, attribute=attribute, **kwargs)
        self.probability_flip = probability_flip
        self.lows = self.attribute_type.lows
        self.ups = self.attribute_type.ups
        self.range_arities = [
            list(range(l, up + 1)) for l, up in zip(self.lows, self.ups)
        ]
        self.size = len(self.range_arities)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = getattr(s2, self.attribute)
        for k in range(self.size):
            if random.random() <= self.probability_flip:
                new_arity = random.choice(self.range_arities[k])
                vector[k] = new_arity
        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)
