#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from discrete_optimization.generic_rcpsp_tools.attribute_type import (
    ListIntegerRcpsp,
    PermutationRcpsp,
)
from discrete_optimization.generic_rcpsp_tools.mutation import (
    DeadlineRcpspMutation,
    RcpspMutation,
)
from discrete_optimization.generic_tools.do_mutation import (
    SingleAttributeMutation,
)
from discrete_optimization.generic_tools.do_problem import (
    Problem,
)
from discrete_optimization.generic_tools.encoding_register import (
    AttributeType,
    ListBoolean,
    ListInteger,
    Permutation,
)
from discrete_optimization.generic_tools.mutations.mutation_bool import BitFlipMutation
from discrete_optimization.generic_tools.mutations.mutation_integer import (
    IntegerMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_permutation import (
    PartialShuffleMutation,
    ShuffleMutation,
    SwapMutation,
    TwoOptMutation,
)
from discrete_optimization.knapsack.mutation import (
    BitFlipKnapsackMutation,
    SingleBitFlipKnapsackMutation,
)
from discrete_optimization.knapsack.problem import ListBooleanKnapsack
from discrete_optimization.tsp.mutation import (
    SwapTspMutation,
    TwoOptIntersectionTspMutation,
    TwoOptTspMutation,
)
from discrete_optimization.tsp.problem import PermutationTsp

logger = logging.getLogger(__name__)


generic_permutation_mutations: dict[
    str, tuple[type[SingleAttributeMutation], dict[str, Any]]
] = {
    "total_shuffle": (ShuffleMutation, {}),
    "partial_shuffle": (PartialShuffleMutation, {}),
    "swap": (SwapMutation, {}),
    "2opt_gen": (TwoOptMutation, {}),
}

dictionnary_mutation: dict[
    type[AttributeType], dict[str, tuple[type[SingleAttributeMutation], dict[str, Any]]]
] = {
    Permutation: generic_permutation_mutations,
    PermutationTsp: {
        "2opt": (TwoOptTspMutation, {"nb_test": 200}),
        "2opt_interection": (
            TwoOptIntersectionTspMutation,
            {"nb_test": 200},
        ),
        "swap_tsp": (SwapTspMutation, {}),
        **generic_permutation_mutations,
    },
    PermutationRcpsp: {
        "total_shuffle_rcpsp": (
            RcpspMutation,
            {"other_mutation_cls": ShuffleMutation},
        ),
        "deadline": (
            RcpspMutation,
            {"other_mutation_cls": DeadlineRcpspMutation},
        ),
        "partial_shuffle_rcpsp": (
            RcpspMutation,
            {"other_mutation_cls": PartialShuffleMutation},
        ),
        "swap_rcpsp": (
            RcpspMutation,
            {"nb_swap": 3, "other_mutation_cls": SwapMutation},
        ),
        "2opt_gen_rcpsp": (
            RcpspMutation,
            {"other_mutation_cls": TwoOptMutation},
        ),
    },
    ListBoolean: {"bitflip": (BitFlipMutation, {})},
    ListBooleanKnapsack: {
        "bitflip-kp": (BitFlipKnapsackMutation, {}),
        "singlebitflip-kp": (SingleBitFlipKnapsackMutation, {}),
    },
    ListInteger: {
        "random_flip": (IntegerMutation, {}),
    },
    ListIntegerRcpsp: {
        "random_flip_modes_rcpsp": (
            RcpspMutation,
            {"other_mutation_cls": IntegerMutation},
        ),
    },
}


def get_available_mutations(
    problem: Problem,
) -> list[tuple[type[SingleAttributeMutation], dict[str, Any], str]]:
    list_mutations = [
        (mutation_cls, mutation_kwargs, attribute_name)
        for attribute_name, attribute_type in problem.get_attribute_register().items()
        if type(attribute_type) in dictionnary_mutation
        for mutation_name, (mutation_cls, mutation_kwargs) in dictionnary_mutation[
            type(attribute_type)
        ].items()
    ]
    logger.debug(f"{len(list_mutations)} mutation available for your problem")
    return list_mutations
