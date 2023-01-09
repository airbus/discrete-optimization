#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    TypeAttribute,
)
from discrete_optimization.generic_tools.mutations.mutation_bool import MutationBitFlip
from discrete_optimization.generic_tools.mutations.mutation_integer import (
    MutationIntegerSpecificArity,
)
from discrete_optimization.generic_tools.mutations.permutation_mutations import (
    PermutationPartialShuffleMutation,
    PermutationShuffleMutation,
    PermutationSwap,
    TwoOptMutation,
)
from discrete_optimization.knapsack.mutation.mutation_knapsack import (
    KnapsackMutationSingleBitFlip,
    MutationKnapsack,
)
from discrete_optimization.rcpsp.mutations.mutation_rcpsp import (
    DeadlineMutationRCPSP,
    PermutationMutationRCPSP,
)
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    Mutation2OptIntersection,
    MutationSwapTSP,
)

logger = logging.getLogger(__name__)

dictionnary_mutation: Dict[
    TypeAttribute, Dict[str, Tuple[Type[Mutation], Dict[str, Any]]]
] = {
    TypeAttribute.PERMUTATION: {
        "total_shuffle": (PermutationShuffleMutation, {}),
        "partial_shuffle": (PermutationPartialShuffleMutation, {"proportion": 0.2}),
        "swap": (PermutationSwap, {"nb_swap": 1}),
        "2opt_gen": (TwoOptMutation, {}),
    },
    TypeAttribute.PERMUTATION_TSP: {
        "2opt": (Mutation2Opt, {"test_all": False, "nb_test": 200}),
        "2opt_interection": (
            Mutation2OptIntersection,
            {"test_all": False, "nb_test": 200},
        ),
        "swap_tsp": (MutationSwapTSP, {}),
    },
    TypeAttribute.PERMUTATION_RCPSP: {
        "total_shuffle_rcpsp": (
            PermutationMutationRCPSP,
            {"other_mutation": PermutationShuffleMutation},
        ),
        "deadline": (
            PermutationMutationRCPSP,
            {"other_mutation": DeadlineMutationRCPSP},
        ),
        "partial_shuffle_rcpsp": (
            PermutationMutationRCPSP,
            {"proportion": 0.2, "other_mutation": PermutationPartialShuffleMutation},
        ),
        "swap_rcpsp": (
            PermutationMutationRCPSP,
            {"nb_swap": 3, "other_mutation": PermutationSwap},
        ),
        "2opt_gen_rcpsp": (
            PermutationMutationRCPSP,
            {"other_mutation": TwoOptMutation},
        ),
    },
    TypeAttribute.LIST_BOOLEAN: {
        "bitflip": (MutationBitFlip, {"probability_flip": 0.1})
    },
    TypeAttribute.LIST_BOOLEAN_KNAP: {
        "bitflip-kp": (MutationKnapsack, {}),
        "singlebitflip-kp": (KnapsackMutationSingleBitFlip, {}),
    },
    TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY: {
        "random_flip": (MutationIntegerSpecificArity, {"probability_flip": 0.1}),
        "random_flip_modes_rcpsp": (
            PermutationMutationRCPSP,
            {"other_mutation": MutationIntegerSpecificArity, "probability_flip": 0.1},
        ),
    },
}


def get_available_mutations(
    problem: Problem, solution: Optional[Solution] = None
) -> Tuple[
    Dict[TypeAttribute, Dict[str, Tuple[Type[Mutation], Dict[str, Any]]]],
    List[Tuple[Type[Mutation], Dict[str, Any]]],
]:
    register = problem.get_attribute_register()
    present_types = set(register.get_types())
    mutations: Dict[
        TypeAttribute, Dict[str, Tuple[Type[Mutation], Dict[str, Any]]]
    ] = {}
    mutations_list: List[Tuple[Type[Mutation], Dict[str, Any]]] = []
    nb_mutations = 0
    for pr_type in present_types:
        if pr_type in dictionnary_mutation:
            mutations[pr_type] = dictionnary_mutation[pr_type]
            mutations_list += list(dictionnary_mutation[pr_type].values())
            nb_mutations += len(dictionnary_mutation[pr_type])
    logger.debug(f"{nb_mutations} mutation available for your problem")
    logger.debug(mutations)
    return mutations, mutations_list
