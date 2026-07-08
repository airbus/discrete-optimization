import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers import (
    SimulatedAnnealingLotSizingSolverFast,
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
    solver = SimulatedAnnealingLotSizingSolverFast(
        problem,
        T0=37.0,
        alpha=0.999,
        beta=0.7,
        n_a=12049,
        n_s=60240,
        max_iterations=10e6,
    )
    solver.init_model()
    res = solver.solve(
        parameters_cp=p,
        time_limit=200,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
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
