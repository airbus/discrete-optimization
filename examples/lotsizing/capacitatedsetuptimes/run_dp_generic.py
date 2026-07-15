import logging

from discrete_optimization.lotsizing.capacitatedsetuptimes.parser import (
    create_simple_instance,
)
from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesSolution,
)
from discrete_optimization.lotsizing.generic_solver.dp.generic_dp_solver import (
    GenericLotSizingDp,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
    plt,
)

logging.basicConfig(level=logging.INFO)


def run_dp():
    problem = create_simple_instance(capacity=15, setup_time=2)
    solver = GenericLotSizingDp(problem)
    solver.allow_backorder = False
    solver.penalty_advance_time = 0
    solver.init_model()
    res = solver.solve(
        solver="LNBS",
        time_limit=30,
    )
    sol: CapacitatedSetupTimesSolution = res[-1][0]
    plot_solution_summary(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_production_schedule(problem, sol)
    print(sol.productions)
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))
    plt.show()


if __name__ == "__main__":
    run_dp()
