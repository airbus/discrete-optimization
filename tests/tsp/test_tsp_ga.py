import os
import sys

from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.mutations.permutation_mutations import (
    PermutationPartialShuffleMutation,
)
from discrete_optimization.generic_tools.path_tools import abspath_from_file
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    Mutation2OptIntersection,
    MutationSwapTSP,
    SwapTSPMove,
)
from discrete_optimization.tsp.tsp_model import SolutionTSP
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
    print(tsp_model.satisfy(sol))
    assert tsp_model.satisfy(sol)


def testing_tsp_with_specific_mutation():
    # print(__file__)
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    tsp_model = tsp_parse_file(files[0])
    # tsp_model = tsp_parse_file(abspath_from_file(__file__, "data/tsp_51_1"))
    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    # mutation = Mutation2OptIntersection(tsp_model=tsp_model, nb_test=10)
    # mutation = Mutation2Opt(tsp_model=tsp_model, nb_test=10)
    # mutation = MutationSwapTSP(tsp_model=tsp_model)
    crossover = DeapCrossover.CX_UNIFORM_PARTIALY_MATCHED
    selection = DeapSelection.SEL_TOURNAMENT
    mutation = Mutation2OptIntersection(tsp_model=tsp_model, nb_test=10)  # Work OK
    mutation = Mutation2Opt(tsp_model=tsp_model, nb_test=200)  # Work OK
    # mutation = MutationSwapTSP(tsp_model)
    # doesn't work as expected here because the attribute permutation_from0 is never updated in the mutation,
    # or maybe would work with Ga(encoding="permutation") but another problem occur

    # mutation = PermutationPartialShuffleMutation(tsp_model,
    #                                              tsp_model.get_dummy_solution(),
    #                                              attribute="permutation_from0",
    # important here otherwise you're doomed.
    #                                              proportion=0.1)
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
