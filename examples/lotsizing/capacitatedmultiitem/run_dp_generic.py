import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.generic_solver.dp.generic_dp_solver import (
    GenericLotSizingDp,
)
from discrete_optimization.lotsizing.utils import (
    plot_inventory_and_costs,
    plot_production_schedule,
    plot_solution_summary,
)

logging.basicConfig(level=logging.INFO)


def main():
    file = [f for f in get_data_available() if "PSP_100_1.psp" in f][0]
    problem = parse_file(file)
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    solver = GenericLotSizingDp(problem)
    solver.max_backorder = 10
    solver.allow_backorder = False
    solver.lookahead_demand = 10
    solver.force_unmet_zero = False
    solver.add_transition_dominance = True
    solver.penalty_advance_time = 10000
    solver.init_model()
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        solver="LNBS",
        threads=4,
        time_limit=200,
    )
    sol = res[-1][0]
    print(solver.aggreg_from_sol(sol))
    print(problem.infos["known_bound"])
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_summary(problem, sol)
    plot_inventory_and_costs(problem, sol)
    plot_production_schedule(problem, sol)
    plt.show()


if __name__ == "__main__":
    main()
