import logging

from discrete_optimization.generic_tools.lp_tools import ParametersMilp, mathopt
from discrete_optimization.lotsizing.capacitatedsetuptimes.parser import (
    create_simple_instance,
)
from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesSolution,
)
from discrete_optimization.lotsizing.generic_solver.milp.generic_lotsizing_milp import (
    MathOptGenericLotSizingMilp,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
    plt,
)

logging.basicConfig(level=logging.INFO)


def run_milp():
    problem = create_simple_instance(capacity=15, setup_time=2)
    solver = MathOptGenericLotSizingMilp(problem)
    solver.init_model()
    p = ParametersMilp.default()
    res = solver.solve(
        parameters_milp=p, time_limit=30, mathopt_solver_type=mathopt.SolverType.CP_SAT
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
    run_milp()
