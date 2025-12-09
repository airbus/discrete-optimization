#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import Problem


def generic_mutate_wrapper(
    individual: list[int],
    problem: Problem,
    attribute_name: str,
    custom_mutation: Mutation,
) -> tuple[list[int]]:
    custom_sol = problem.build_solution_from_encoding(
        int_vector=individual, encoding_name=attribute_name
    )
    new_custom_sol = custom_mutation.mutate(custom_sol)[0]
    new_individual = individual
    tmp_vector: Sequence = getattr(new_custom_sol, attribute_name)
    for i in range(len(tmp_vector)):
        new_individual[i] = tmp_vector[i]
    return (new_individual,)
