import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.capacitatedsetuptimes.parser import (
    create_simple_instance,
)
from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesSolution,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.generic_lotsizing_cpsat_scheduling import (
    GenericLotSizingCpsatScheduling,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
    plt,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    problem = create_simple_instance(capacity=15, setup_time=2)
    solver = GenericLotSizingCpsatScheduling(problem)
    solver.init_model()
    res = solver.solve(
        time_limit=30,
        parameters_cp=ParametersCp.default_cpsat(),
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
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
    run_cpsat()
