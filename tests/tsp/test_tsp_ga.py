#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    Mutation2OptIntersection,
)
from discrete_optimization.tsp.tsp_parser import get_data_available
from discrete_optimization.tsp.tsp_parser import parse_file as tsp_parse_file


def testing_tsp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    tsp_model = tsp_parse_file(files[0])
    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    ga_solver = Ga(
        tsp_model,
        encoding="permutation_from0",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["length"],
        objective_weights=[-1],
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert tsp_model.satisfy(sol)


def testing_tsp_with_specific_mutation():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    tsp_model = tsp_parse_file(files[0])
    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    crossover = DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED
    selection = DeapSelection.SEL_TOURNAMENT
    mutation = Mutation2Opt(tsp_model=tsp_model, nb_test=200)
    ga_solver = Ga(
        tsp_model,
        encoding="permutation_from0",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["length"],
        objective_weights=[-1],
        mutation=mutation,
        crossover=crossover,
        selection=selection,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert tsp_model.satisfy(sol)


if __name__ == "__main__":
    testing_tsp()
    # testing_tsp_with_specific_mutation()
