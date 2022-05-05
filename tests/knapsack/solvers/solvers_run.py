import os
import sys

from discrete_optimization.generic_tools.lp_tools import MilpSolverName

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
    LPKnapsack,
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


def main_run():
    file = [f for f in files_available if "ks_60_0" in f][0]
    knapsack_model = parse_file(file)
    do_gurobi = False
    methods = solvers.keys()
    methods = ["cp"]
    for method in methods:
        print("method : ", method)
        for submethod in solvers[method]:
            if submethod[0] == LPKnapsackGurobi and not do_gurobi:
                continue
            print(submethod[0])
            t = time.time()
            solution = solve(submethod[0], knapsack_model, **submethod[1])
            print(time.time() - t, " seconds to solve")
            print("Solution : ", solution[0])


def run_lns():
    file = [f for f in files_available if "ks_100_0" in f][0]
    knapsack_model = parse_file(file)
    gurobi_model = LPKnapsack(knapsack_model)
    gurobi_model.init_model()
    greedy = GreedyDummy(knapsack_model)
    init_solution = greedy.solve().get_best_solution()
    solutions = gurobi_model.solve(ParametersMilp.default())
    print("sol ", solutions.get_best_solution().value)
    gurobi_model.solve_lns(
        ParametersMilp.default(),
        init_solution=init_solution,
        fraction_decision_fixed=0.1,
        nb_iteration_max=10000,
    )


def run_lns_cp():
    file = [f for f in files_available if "ks_100_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPKnapsackMZN2(knapsack_model)
    cp_model.init_model()
    greedy = GreedyDummy(knapsack_model)
    init_solution = greedy.solve().get_best_solution()
    # gurobi_model = LPKnapsackGurobi(knapsack_model)
    # gurobi_model.init_model()
    # solutions = gurobi_model.solve(ParametersGurobi.default())
    # print("sol ", solutions[0].value)
    sol, results = cp_model.solve_lns(
        init_solution=init_solution,
        fraction_decision_fixed=0.92,
        nb_iteration_max=500,
        max_time_per_iteration_s=0.2,
        save_results=True,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)
    ax[0].plot(results["objective"][6:])
    ax[0].set_title("Evolution of the objective")
    ax[0].set_xlabel("iteration LNS")
    ax[1].plot(results["weight"][6:])
    ax[1].set_title("Evolution of the KP weight")
    ax[1].axhline(y=knapsack_model.max_capacity, color="b")
    ax[1].set_xlabel("iteration LNS")
    plt.show()


def run_lp():
    file = [f for f in files_available if "ks_10000_0" in f][0]
    knapsack_model = parse_file(file)
    # gurobi_solver = LPKnapsackGurobi(knapsack_model)
    pymip_solver = LPKnapsack(knapsack_model, milp_solver_name=MilpSolverName.GRB)
    from mip import CBC, GRB

    pymip_solver.init_model(solver_name=GRB)  # (or CBC)..
    parameters_milp = ParametersMilp.default()
    solutions = pymip_solver.solve(parameters_milp=parameters_milp)
    print(solutions.get_best_solution())


if __name__ == "__main__":
    run_lp()