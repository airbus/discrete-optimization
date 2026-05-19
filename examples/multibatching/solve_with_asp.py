import logging

from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.multibatching.parser import get_data_available, parse_file
from discrete_optimization.multibatching.solvers.asp import ClingconMultibatchingSolver
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    CpsatPackingSubproblem,
)

logging.basicConfig(level=logging.INFO)

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.multibatching.solvers.two_steps import (
    GreedyPackingForMultibatching,
    PackingViaBinPacking,
)


def script():
    # 1. Generate the problem instance
    print("=" * 80)
    print("Multibatching Two-Step Solver Example")
    print("=" * 80)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        datasets = get_data_available()
        if not datasets:
            print("No datasets found. Please run:")
            print(
                ">>> from discrete_optimization.datasets import fetch_data_from_multibatching"
            )
            print(">>> fetch_data_from_multibatching()")
            return
        dataset_path = datasets[0]
        print(f"      Using dataset: {dataset_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Parse the problem
    print("\n[2/5] Parsing problem...")
    problem = parse_file(
        dataset_path,
        scale_capacity=1.0 / 10**4,  # Scale down capacities
        scale_size=1.0 / 10**4,  # Scale down product sizes
        scale_co2=1.0 / 10**6,
    )
    solver = ClingconMultibatchingSolver(problem)
    use_shortest_path_heuristic = True
    sp_tolerance = 0.2
    solver.init_model(
        restrict_to_shortest_paths=use_shortest_path_heuristic,
        shortest_path_tolerance=sp_tolerance,
    )
    time_limit = 100
    print(f"Starting solve with {time_limit}s timeout...")
    result_storage = solver.solve(
        callbacks=[],
        time_limit=time_limit,
    )
    # 4. Retrieve and display results
    solution, fitness = result_storage.get_best_solution_fit()
    if solution == None:
        print("UNSAT")
    else:
        solution = result_storage[-1][0]
        best_sol_for_postpro = None
        best_sol_ = None
        best_val = float("inf")
        for i in range(len(result_storage)):
            pack = GreedyPackingForMultibatching(problem)
            pack.init_from_solution(result_storage[i][0])
            res = pack.solve()
            sol_ = res.get_best_solution()
            value = sum(problem.evaluate(sol_).values())
            if value < best_val:
                best_val = value
                best_sol_ = sol_
                best_sol_for_postpro = result_storage[i][0]
        print("total costs Greedy : ", best_val, f"({best_val:.2e})")

        time_limit_per_link = 5

        pack = PackingViaBinPacking(problem)
        pack.init_from_solution(best_sol_for_postpro)
        p = ParametersCp.default_cpsat()
        p.nb_process = 8
        res = pack.solve(
            bin_packing_solver=SubBrick(AspBinPackingSolver, {}), time_limit_per_link=time_limit_per_link
        )
        if len(res) == 0:
            logger.info("Bin packing found no solution")
        sol_ = res.get_best_solution()
        value = sum(problem.evaluate(sol_).values())
        print("total costs ASP : ", value, f"({value:.2e})")
        print(problem.satisfy(sol_))

        pack = PackingViaBinPacking(problem)
        pack.init_from_solution(best_sol_for_postpro)
        p = ParametersCp.default_cpsat()
        p.nb_process = 8
        res = pack.solve(
            bin_packing_solver=SubBrick(
                CpSatBinPackSolver,
                {"parameters_cp": p, "modeling": ModelingBinPack.SCHEDULING},
            ),
            time_limit_per_link=time_limit_per_link,
        )
        if len(res) == 0:
            logger.info("Bin packing found no solution")
        sol_ = res.get_best_solution()
        value = sum(problem.evaluate(sol_).values())
        print("total costs CP RbR : ", value, f"({value:.2e})")
        print(problem.satisfy(sol_))

        pack = CpsatPackingSubproblem(problem)
        pack.init_from_solution(best_sol_for_postpro)
        res = pack.solve(parameters_cp=p, time_limit=100)
        sol_ = res.get_best_solution()
        value = sum(problem.evaluate(sol_).values())
        print("total costs GLOBAL CP : ", value, f"({value:.2e})")
        print(problem.satisfy(sol_))

if __name__ == "__main__":
    script()
