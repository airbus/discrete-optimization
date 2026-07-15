import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.toulbar import (
    ToulbarCapacitatedLotSizingSolver,
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
    solver = ToulbarCapacitatedLotSizingSolver(problem)
    solver.init_model()
    res = solver.solve(
        time_limit=100,
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
