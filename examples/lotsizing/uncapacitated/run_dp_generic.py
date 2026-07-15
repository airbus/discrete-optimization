import logging

from matplotlib import pyplot as plt

from discrete_optimization.lotsizing.generic_solver.dp.generic_dp_solver import (
    GenericLotSizingDp,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem import (
    generate_random_instance,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
)

logging.basicConfig(level=logging.INFO)


def run_dp_generic():
    problem = generate_random_instance(
        horizon=30,
        avg_demand=5,
        setup_cost=1,
        production_cost=5,
        inventory_cost=3,
        seed=42,
    )
    print(problem.has_capacity_limits())
    solver = GenericLotSizingDp(problem)
    solver.init_model()
    res = solver.solve(
        solver="LNBS",
        time_limit=30,
    )
    sol = res[-1][0]
    plot_production_schedule(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_solution_summary(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))
    plt.show()


if __name__ == "__main__":
    run_dp_generic()
