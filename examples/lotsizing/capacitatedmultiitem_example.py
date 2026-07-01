#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example usage of capacitated multi-item lot sizing solver on pigment20c benchmark."""

import logging

from discrete_optimization.lotsizing.capacitatedmultiitem.parser import parse_file
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)
from discrete_optimization.lotsizing.parser import get_data_available

logging.basicConfig(level=logging.INFO)


def run_greedy_on_pigment20c():
    """Run greedy solver on pigment20c benchmark."""
    # Load pigment20c benchmark
    instances = get_data_available()
    pigment = [inst for inst in instances if "pigment20c.psp" in inst][0]
    problem = parse_file(pigment)

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
    print(f"  Allow backlog (hard constraint): {problem.is_backlog_allowed()}")
    print(
        f"  Backlog penalty: {problem.get_backlog_cost_per_unit(0, 0):.0f} per unit-period"
    )

    print("\n" + "=" * 80)
    print("Testing Greedy Strategies")
    print("=" * 80)

    best_strategy = None
    best_fit = float("inf")
    best_sol = None

    for strategy in GreedyStrategy:
        solver = GreedyLotSizingSolver(problem)
        result = solver.solve(strategy=strategy)
        sol, fit = result[0]
        objectives = problem.evaluate(sol)

        print(f"\n{strategy.name}:")
        print(f"  Fit: {fit:.2f}")
        print(f"  Feasible: {problem.satisfy(sol)}")
        print(f"  Inventory cost: {objectives['inventory_cost']:.2f}")
        print(f"  Changeover cost: {objectives['changeover_cost']:.2f}")
        print(f"  Backlog cost: {objectives['backlog_cost']:.2f}")

        if fit < best_fit:
            best_fit = fit
            best_strategy = strategy
            best_sol = sol

    print("\n" + "=" * 80)
    print(f"Best Strategy: {best_strategy.name}")
    print("=" * 80)

    objectives = problem.evaluate(best_sol)
    print(f"Fit: {best_fit:.2f}")
    print(f"Feasible: {problem.satisfy(best_sol)}")
    print(f"\nCost breakdown:")
    print(f"  Inventory cost: {objectives['inventory_cost']:.2f}")
    print(f"  Changeover cost: {objectives['changeover_cost']:.2f}")
    print(f"  Backlog cost: {objectives['backlog_cost']:.2f}")
    print(f"  Total: {best_sol.compute_total_cost():.2f}")

    # Analyze backlog
    total_backlog_periods = sum(
        best_sol.get_backlog_quantity(item, t)
        for item in problem.items_list
        for t in range(problem.horizon)
    )
    print(f"\nBacklog analysis:")
    print(f"  Total unit-periods of backlog: {total_backlog_periods}")
    print(
        f"  Demand satisfied on time: {best_sol.check_demand_satisfaction(allow_delays=False)}"
    )
    print(
        f"  All demand eventually satisfied: {best_sol.check_demand_satisfaction(allow_delays=True)}"
    )

    return best_sol


if __name__ == "__main__":
    run_greedy_on_pigment20c()
