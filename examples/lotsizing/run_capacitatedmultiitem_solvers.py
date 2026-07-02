#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test all capacitated multi-item lot sizing solvers on pigment20c benchmark."""

import logging

from discrete_optimization.lotsizing.capacitatedmultiitem.parser import parse_file
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers import (
    ChangeoverModel,
    CpSatLotSizingSolver,
    CpSatSchedulingCapacitatedLotSizing,
    GreedyLotSizingSolver,
    GreedyStrategy,
    GurobiCapacitatedLotSizingSolver,
    MathOptCapacitatedLotSizingSolver,
)
from discrete_optimization.lotsizing.parser import get_data_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_solvers():
    """Run all solvers on pigment20c benchmark."""
    # Load pigment20c benchmark
    instances = get_data_available()
    pigment = [inst for inst in instances if "pigment20c.psp" in inst][0]
    problem = parse_file(pigment)
    print(problem.infos["known_bound"])
    print("=" * 80)
    print("Capacitated Multi-Item Lot Sizing: pigment20c Benchmark")
    print("=" * 80)
    print(f"Problem details:")
    print(f"  Items: {problem.nb_items}")
    print(f"  Horizon: {problem.horizon}")
    print(f"  Capacity per period: {problem.get_available_production_time(0)}")
    print(
        f"  Total demand: {sum(problem.get_total_demand(i) for i in problem.items_list)}"
    )
    print(f"  Allow backlog: {problem.is_backlog_allowed()}")
    print(
        f"  Backlog penalty: {problem.get_backlog_cost_per_unit(0, 0):.0f} per unit-period"
    )
    print()

    results = {}

    # Test greedy solvers
    print("=" * 80)
    print("Greedy Solvers")
    print("=" * 80)
    for strategy in GreedyStrategy:
        solver = GreedyLotSizingSolver(problem)
        result = solver.solve(strategy=strategy)
        sol, fit = result[0]
        objectives = problem.evaluate(sol)

        print(f"\n{strategy.name}:")
        print(f"  Cost: {fit:.2f}")
        print(f"  Feasible: {problem.satisfy(sol)}")
        print(f"  Inventory: {objectives['inventory_cost']:.2f}")
        print(f"  Changeover: {objectives['changeover_cost']:.2f}")
        print(f"  Backlog: {objectives['backlog_cost']:.2f}")

        results[f"Greedy_{strategy.name}"] = (sol, fit)

    # Get best greedy solution for warmstart
    best_greedy_sol = min(results.values(), key=lambda x: x[1])[0]
    print(f"\nBest greedy cost: {min(results.values(), key=lambda x: x[1])[1]:.2f}")

    # Test CP-SAT solvers (with different changeover encodings)
    print("\n" + "=" * 80)
    print("CP-SAT Solvers (30s time limit)")
    print("=" * 80)

    for model_type in ChangeoverModel:
        print(f"\nCP-SAT {model_type.name}:")
        solver = CpSatLotSizingSolver(problem)
        result = solver.solve(
            changeover_model=model_type,
            time_limit=30.0,
        )
        if len(result) > 0:
            sol, fit = result.get_best_solution_fit()
            objectives = problem.evaluate(sol)
            print(f"  Cost: {fit:.2f}")
            print(f"  Feasible: {problem.satisfy(sol)}")
            print(f"  Inventory: {objectives['inventory_cost']:.2f}")
            print(f"  Changeover: {objectives['changeover_cost']:.2f}")
            print(f"  Backlog: {objectives['backlog_cost']:.2f}")
            results[f"CPSAT_{model_type.name}"] = (sol, fit)
        else:
            print(f"  No solution found")

    # Test CP-SAT with warmstart
    print("\n" + "=" * 80)
    print("CP-SAT with Warmstart (30s time limit)")
    print("=" * 80)
    solver = CpSatLotSizingSolver(problem)
    solver.init_model(changeover_model=ChangeoverModel.STATE_BASED)
    solver.set_warm_start(best_greedy_sol)
    result = solver.solve(time_limit=30.0)
    if len(result) > 0:
        sol, fit = result.get_best_solution_fit()
        objectives = problem.evaluate(sol)
        print(f"  Cost: {fit:.2f}")
        print(f"  Feasible: {problem.satisfy(sol)}")
        print(f"  Inventory: {objectives['inventory_cost']:.2f}")
        print(f"  Changeover: {objectives['changeover_cost']:.2f}")
        print(f"  Backlog: {objectives['backlog_cost']:.2f}")
        results["CPSAT_WARMSTART"] = (sol, fit)

    # Test CP-SAT Scheduling solver
    print("\n" + "=" * 80)
    print("CP-SAT Scheduling Solver (30s time limit)")
    print("=" * 80)
    solver = CpSatSchedulingCapacitatedLotSizing(problem)
    result = solver.solve(time_limit=30.0)
    if len(result) > 0:
        sol, fit = result.get_best_solution_fit()
        objectives = problem.evaluate(sol)
        print(f"  Cost: {fit:.2f}")
        print(f"  Feasible: {problem.satisfy(sol)}")
        print(f"  Inventory: {objectives['inventory_cost']:.2f}")
        print(f"  Changeover: {objectives['changeover_cost']:.2f}")
        print(f"  Backlog: {objectives['backlog_cost']:.2f}")
        results["CPSAT_SCHEDULING"] = (sol, fit)

    # Test MILP solvers
    print("\n" + "=" * 80)
    print("MILP Solvers (30s time limit)")
    print("=" * 80)

    # MathOpt (CP-SAT backend)
    try:
        print("\nMathOpt (CP-SAT):")
        solver = MathOptCapacitatedLotSizingSolver(problem)
        result = solver.solve(time_limit=30.0)
        if len(result) > 0:
            sol, fit = result.get_best_solution_fit()
            objectives = problem.evaluate(sol)
            print(f"  Cost: {fit:.2f}")
            print(f"  Feasible: {problem.satisfy(sol)}")
            print(f"  Inventory: {objectives['inventory_cost']:.2f}")
            print(f"  Changeover: {objectives['changeover_cost']:.2f}")
            print(f"  Backlog: {objectives['backlog_cost']:.2f}")
            results["MATHOPT"] = (sol, fit)
    except Exception as e:
        print(f"  Error: {e}")

    # Gurobi (if available)
    try:
        print("\nGurobi:")
        solver = GurobiCapacitatedLotSizingSolver(problem)
        result = solver.solve(time_limit=30.0)
        if len(result) > 0:
            sol, fit = result.get_best_solution_fit()
            objectives = problem.evaluate(sol)
            print(f"  Cost: {fit:.2f}")
            print(f"  Feasible: {problem.satisfy(sol)}")
            print(f"  Inventory: {objectives['inventory_cost']:.2f}")
            print(f"  Changeover: {objectives['changeover_cost']:.2f}")
            print(f"  Backlog: {objectives['backlog_cost']:.2f}")
            results["GUROBI"] = (sol, fit)
    except Exception as e:
        print(f"  Gurobi not available: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, (sol, fit) in sorted(results.items(), key=lambda x: x[1][1]):
        print(f"{name:30s}: {fit:10.2f}")

    print(
        "\nBest solution cost: {:.2f}".format(
            min(results.values(), key=lambda x: x[1])[1]
        )
    )


if __name__ == "__main__":
    test_all_solvers()
