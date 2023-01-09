#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, List, Optional, Tuple

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    Mutation,
)
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    TypeAttribute,
    lower_bound_vector_encoding_from_dict,
    upper_bound_vector_encoding_from_dict,
)
from discrete_optimization.generic_tools.mutations.mutation_util import (
    get_attribute_for_type,
)


class MutationIntegerSpecificArity(Mutation):
    @staticmethod
    def build(
        problem: Problem, solution: Solution, **kwargs: Any
    ) -> "MutationIntegerSpecificArity":
        return MutationIntegerSpecificArity(
            problem,
            attribute=kwargs.get("attribute", None),
            arities=kwargs.get("arities", None),
            probability_flip=kwargs.get("probability_flip", 0.1),
            min_value=kwargs.get("min_value", 1),
        )

    def __init__(
        self,
        problem: Problem,
        attribute: Optional[str] = None,
        arities: Optional[List[int]] = None,
        probability_flip: float = 0.1,
        min_value: int = 1,
    ):
        self.problem = problem
        self.probability_flip = probability_flip
        lows = None
        ups = None
        register = problem.get_attribute_register()
        if attribute is None:
            attribute_key = get_attribute_for_type(
                register, TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY
            )
            self.attribute = register.dict_attribute_to_type[attribute_key]["name"]
            arities = register.dict_attribute_to_type[attribute_key]["arities"]
            lows = lower_bound_vector_encoding_from_dict(
                register.dict_attribute_to_type[attribute_key]
            )
            ups = upper_bound_vector_encoding_from_dict(
                register.dict_attribute_to_type[attribute_key]
            )
        else:
            self.attribute = attribute
        if arities is None:
            attribute_key = get_attribute_for_type(
                register, TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY
            )
            self.arities = register.dict_attribute_to_type[attribute_key]["arities"]
        else:
            self.arities = arities
        if lows is None:
            self.lows = [min_value for i in range(len(self.arities))]
        else:
            self.lows = lows
        if ups is None:
            self.ups = [
                min_value + self.arities[i] - 1 for i in range(len(self.arities))
            ]
        else:
            self.ups = ups
        self.range_arities = [
            list(range(l, up + 1)) for l, up in zip(self.lows, self.ups)
        ]
        self.size = len(self.range_arities)

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = getattr(s2, self.attribute)
        for k in range(self.size):
            if random.random() <= self.probability_flip:
                new_arity = random.choice(self.range_arities[k])
                vector[k] = new_arity
        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, m = self.mutate(solution)
        obj = self.problem.evaluate(s)
        return s, m, obj
