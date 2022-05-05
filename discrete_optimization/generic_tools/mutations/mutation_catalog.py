import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from discrete_optimization.generic_tools.do_mutation import LocalMove, Mutation
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    Problem,
    Solution,
    TypeAttribute,
)
from discrete_optimization.generic_tools.mutations.mutation_bool import MutationBitFlip
from discrete_optimization.generic_tools.mutations.mutation_integer import (
    MutationIntegerSpecificArrity,
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
from discrete_optimization.vrp.mutation.mutation_vrp import (
    MutationRelocate,
    MutationSwap,
    MutationTwoOptVRP,
)

dictionnary_mutation = {
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
    TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY: {
        "random_flip": (MutationIntegerSpecificArrity, {"probability_flip": 0.1}),
        "random_flip_modes_rcpsp": (
            PermutationMutationRCPSP,
            {"other_mutation": MutationIntegerSpecificArrity, "probability_flip": 0.1},
        ),
    },
    TypeAttribute.VRP_PATHS: {
        "relocate": (MutationRelocate, {}),
        "swap": (MutationSwap, {}),
        "2opt": (MutationTwoOptVRP, {"test_all": False, "nb_test": 200}),
    },
}


def get_available_mutations(problem: Problem, solution: Solution = None, verbose=True):
    register = problem.get_attribute_register()
    present_types = set(register.get_types())
    mutations = {}
    mutations_list = []
    nb_mutations = 0
    for pr_type in present_types:
        if pr_type in dictionnary_mutation:
            mutations[pr_type] = dictionnary_mutation[pr_type]
            mutations_list += list(dictionnary_mutation[pr_type].values())
            nb_mutations += len(dictionnary_mutation[pr_type])
    if verbose:
        print(nb_mutations, " mutation available for your problem")
        print(mutations)
    return mutations, mutations_list
