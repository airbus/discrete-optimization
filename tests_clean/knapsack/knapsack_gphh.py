from discrete_optimization.knapsack.knapsack_model import (
    ItemMultidimensional,
    KnapsackModel,
    KnapsackSolution,
    KnapsackSolutionMultidimensional,
    MultidimensionalKnapsack,
)
from discrete_optimization.knapsack.knapsack_parser import files_available, parse_file
from discrete_optimization.knapsack.knapsack_solvers import GreedyBest, GreedyDummy
from discrete_optimization.knapsack.solvers.gphh_knapsack import (
    GPHH,
    FeatureEnum,
    ParametersGPHH,
)


def from_kp_to_multi(knapsack_model: KnapsackModel):
    return MultidimensionalKnapsack(
        list_items=[
            ItemMultidimensional(index=x.index, value=x.value, weights=[x.weight])
            for x in knapsack_model.list_items
        ],
        max_capacities=[knapsack_model.max_capacity],
    )


def run_one_example():
    one_file = files_available[10]
    knapsack_model: KnapsackModel = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_model)
    trainings = [from_kp_to_multi(parse_file(files_available[i])) for i in range(10)]
    params_gphh = ParametersGPHH.default()
    params_gphh.pop_size = 40
    params_gphh.crossover_rate = 0.7
    params_gphh.mutation_rate = 0.1
    params_gphh.n_gen = 50
    params_gphh.min_tree_depth = 1
    params_gphh.max_tree_depth = 5
    gphh_solver = GPHH(
        training_domains=trainings,
        domain_model=multidimensional_knapsack,
        params_gphh=params_gphh,
    )
    gphh_solver.init_model()
    rs = gphh_solver.solve()
    sol, fit = rs.get_best_solution_fit()
    for k in range(10, len(files_available)):
        kp = parse_file(files_available[k])
        mdkp = from_kp_to_multi(kp)
        rs = gphh_solver.build_result_storage_for_domain(mdkp)
        print("Greedy :", GreedyBest(kp).solve().get_best_solution_fit()[1])
        print("Rule : ", rs.get_best_solution_fit()[1])
    gphh_solver.plot_solution()
    print(fit)


if __name__ == "__main__":
    run_one_example()
