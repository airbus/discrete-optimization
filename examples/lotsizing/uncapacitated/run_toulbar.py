import logging

from matplotlib import pyplot as plt

from discrete_optimization.lotsizing.uncapacitatedsingleitem import (
    generate_random_instance,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.solvers.toulbar import (
    ToulbarUncapacitatedSingleItemSolver,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
)

logging.basicConfig(level=logging.INFO)


def run_toulbar():
    problem = generate_random_instance(
        horizon=100,
        avg_demand=10,
        setup_cost=1,
        production_cost=5,
        inventory_cost=3,
        seed=42,
    )
    solver = ToulbarUncapacitatedSingleItemSolver(problem)
    solver.init_model()
    res = solver.solve(time_limit=100)
    sol = res[-1][0]
    plot_production_schedule(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_solution_summary(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))
    plt.show()


if __name__ == "__main__":
    run_toulbar()
