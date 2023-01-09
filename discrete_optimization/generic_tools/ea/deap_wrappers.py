#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, MutableSequence, Sequence, Tuple, Type, TypeVar

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import Problem, Solution

T = TypeVar("T")


def generic_mutate_wrapper(
    individual: MutableSequence[T],
    problem: Problem,
    encoding_name: str,
    indpb: Any,
    solution_fn: Type[Solution],
    custom_mutation: Mutation,
) -> Tuple[MutableSequence[T]]:
    kwargs: Dict[str, Any] = {encoding_name: individual, "problem": problem}
    custom_sol: Solution = solution_fn(**kwargs)  # type: ignore
    new_custom_sol = custom_mutation.mutate(custom_sol)[0]
    new_individual = individual
    tmp_vector: Sequence = getattr(new_custom_sol, encoding_name)
    for i in range(len(tmp_vector)):
        new_individual[i] = tmp_vector[i]
    return (new_individual,)
