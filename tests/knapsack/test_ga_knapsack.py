import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
import time

from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    TypeAttribute,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.path_tools import abspath_from_file
from discrete_optimization.knapsack.knapsack_model import KnapsackSolution
from discrete_optimization.knapsack.knapsack_parser import files_available, parse_file
from discrete_optimization.knapsack.knapsack_solvers import (
    CPKnapsackMZN2,
    GreedyBest,
    GreedyDummy,
    KnapsackORTools,
    LPKnapsackCBC,
    LPKnapsackGurobi,
    ParametersMilp,
    solve,
    solvers,
)
from discrete_optimization.knapsack.mutation.mutation_knapsack import (
    KnapsackMutationSingleBitFlip,
    MutationKnapsack,
)


def testing_ga_init_1():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(knapsack_model)
    ga_solver.solve()


def testing_ga_init_2():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(knapsack_model, encoding="list_taken")


def testing_ga_init_3():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(
        knapsack_model,
        encoding={
            "name": "list_taken",
            "type": [TypeAttribute.LIST_BOOLEAN],
            "n": knapsack_model.nb_items,
        },
    )


def testing_ga_solve():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(knapsack_model)
    kp_sol = ga_solver.solve()
    print("kp_sol: ", kp_sol.list_taken)
    print("kp_evaluate: ", knapsack_model.evaluate(kp_sol))
    print("kp_satisfy: ", knapsack_model.satisfy(kp_sol))


def testing_ga_solve_2():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(
        knapsack_model,
        objective_handling=ObjectiveHandling.SINGLE,
        objectives="value",
        mutation=DeapMutation.MUT_FLIP_BIT,
    )
    kp_sol = ga_solver.solve()
    print("kp_sol: ", kp_sol.list_taken)
    print("kp_evaluate: ", knapsack_model.evaluate(kp_sol))
    print("kp_satisfy: ", knapsack_model.satisfy(kp_sol))


def testing_ga_solve_4():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(
        knapsack_model,
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["value", "weight_violation"],
        objective_weights=[1, -1000000],
        mutation=DeapMutation.MUT_FLIP_BIT,
        max_evals=3000,
    )
    kp_sol = ga_solver.solve()
    print("kp_sol: ", kp_sol.list_taken)
    print("kp_evaluate: ", knapsack_model.evaluate(kp_sol))
    print("kp_satisfy: ", knapsack_model.satisfy(kp_sol))


def testing_ga_solve_default_objective_settings():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    ga_solver = Ga(knapsack_model, mutation=DeapMutation.MUT_FLIP_BIT, max_evals=3000)
    kp_sol = ga_solver.solve()
    print("kp_sol: ", kp_sol.list_taken)
    print("kp_evaluate: ", knapsack_model.evaluate(kp_sol))
    print("kp_satisfy: ", knapsack_model.satisfy(kp_sol))


def testing_own_bitflip_kp_mutation():
    files = [f for f in files_available if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    mutation_1 = KnapsackMutationSingleBitFlip(knapsack_model)
    mutation_2 = MutationKnapsack(knapsack_model)
    mutation = BasicPortfolioMutation(
        [mutation_1, mutation_2], weight_mutation=[0.001, 0.5]
    )
    objective_handling, objectives, weights = get_default_objective_setup(
        knapsack_model
    )
    ga_solver = Ga(
        knapsack_model,
        objective_handling=objective_handling,
        objectives=objectives,
        objective_weights=weights,
        mutation=mutation,
        max_evals=3000,
    )
    sol = ga_solver.solve()
    print(sol)
    knapsack_model.evaluate(sol)
    print("Solution GA", sol)


if __name__ == "__main__":
    testing_ga_init_1()
    testing_ga_init_2()
    testing_ga_init_3()
    testing_ga_solve()
    testing_ga_solve_2()
    testing_ga_solve_4()
    testing_ga_solve_default_objective_settings()
    testing_own_bitflip_kp_mutation()
