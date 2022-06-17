import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    Mutation,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    Problem,
    Solution,
    TypeAttribute,
    lower_bound_vector_encoding_from_dict,
    upper_bound_vector_encoding_from_dict,
)
from discrete_optimization.generic_tools.mutations.mutation_util import (
    get_attribute_for_type,
)


class MutationIntegerSpecificArrity(Mutation):
    @staticmethod
    def build(problem: Problem, solution: Solution, **kwargs):
        return MutationIntegerSpecificArrity(
            problem,
            attribute=kwargs.get("attribute", None),
            arrities=kwargs.get("arrities", None),
            probability_flip=kwargs.get("probability_flip", 0.1),
            min_value=kwargs.get("min_value", 1),
        )

    def __init__(
        self,
        problem: Problem,
        attribute: str = None,
        arrities: List[int] = None,
        probability_flip: float = 0.1,
        min_value: int = 1,
    ):
        self.problem = problem
        self.attribute = attribute
        self.arrities = arrities
        self.probability_flip = probability_flip
        self.lows = None
        self.ups = None
        if self.attribute is None:
            register = problem.get_attribute_register()
            attributes = [
                (k, register.dict_attribute_to_type[k]["name"])
                for k in register.dict_attribute_to_type
                for t in register.dict_attribute_to_type[k]["type"]
                if t == TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY
            ]
            self.attribute = attributes[0][1]
            self.arrities = register.dict_attribute_to_type[attributes[0][0]][
                "arrities"
            ]
            self.lows = lower_bound_vector_encoding_from_dict(
                register.dict_attribute_to_type[attributes[0][0]]
            )
            self.ups = upper_bound_vector_encoding_from_dict(
                register.dict_attribute_to_type[attributes[0][0]]
            )
        if self.lows is None:
            self.lows = [min_value for i in range(len(self.arrities))]
        if self.ups is None:
            self.ups = [
                min_value + self.arrities[i] - 1 for i in range(len(self.arrities))
            ]
        self.range_arrities = [
            list(range(l, up + 1)) for l, up in zip(self.lows, self.ups)
        ]
        self.size = len(self.range_arrities)

    def mutate(self, solution: Solution) -> Tuple[Solution, LocalMove]:
        s2 = solution.copy()
        vector = getattr(s2, self.attribute)
        for k in range(self.size):
            if random.random() <= self.probability_flip:
                new_arrity = random.choice(self.range_arrities[k])
                vector[k] = new_arrity
        setattr(s2, self.attribute, vector)
        return s2, LocalMoveDefault(solution, s2)

    def mutate_and_compute_obj(
        self, solution: Solution
    ) -> Tuple[Solution, LocalMove, Dict[str, float]]:
        s, m = self.mutate(solution)
        obj = self.problem.evaluate(s)
        return s, m, obj
