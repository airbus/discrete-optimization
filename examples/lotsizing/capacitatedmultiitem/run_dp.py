import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.dp import (
    DpSchedCapacitatedLotSizingSolver,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)

logging.basicConfig(level=logging.INFO)


def main():
    file = [f for f in get_data_available() if "pigment20c.psp" in f][0]
    problem = parse_file(file)
    solver = GreedyLotSizingSolver(problem)
    res = solver.solve(strategy=GreedyStrategy.EARLIEST_DEMAND_FIRST)
    sol = res[-1][0]
    print(solver.aggreg_from_sol(sol))
    print(problem.infos["known_bound"])
    print(problem.evaluate(sol), problem.satisfy(sol))
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    solver = DpSchedCapacitatedLotSizingSolver(problem)
    solver.init_model()
    res = solver.solve(
        solver="CABS",
        nb_threads=10,
        time_limit=30,
    )
    sol = res[-1][0]
    print(solver.aggreg_from_sol(sol))
    print(problem.infos["known_bound"])
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    main()
