import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
from discrete_optimization.generic_tools.mutations.mutation_bool import MutationBitFlip
from discrete_optimization.knapsack.knapsack_model import KnapsackModel
from discrete_optimization.knapsack.knapsack_parser import files_available, parse_file
from discrete_optimization.knapsack.mutation.mutation_knapsack import MutationKnapsack

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))


# from discrete_optimization.generic_tools.simulated_annealing import SimulatedAnnealing, RestartHandler, \
#     TemperatureSchedulingFactor, ModeMutation, ModeOptim


def testing_on_knapsack():
    model_file = [f for f in files_available if "ks_60_0" in f][0]
    model: KnapsackModel = parse_file(model_file, force_recompute_values=True)
    solution = model.get_dummy_solution()
    mutation = MutationBitFlip(model)
    mutation_2 = MutationKnapsack(model)
    for i in range(1000):
        sol, move, f = mutation_2.mutate_and_compute_obj(solution)
        print(sol, f)
    sol_back = move.backtrack_local_move(sol)
    f = model.evaluate(sol_back)
    print(sol_back, f)


if __name__ == "__main__":
    testing_on_knapsack()
