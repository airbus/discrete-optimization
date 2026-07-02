import logging

from discrete_optimization.generic_tools.lp_tools import ParametersMilp, mathopt
from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.lp import (
    GurobiCapacitatedLotSizingSolver,
    MathOptCapacitatedLotSizingSolver,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.lp_milp import (
    MilpCapacitatedLotSizingSolver,
)

logging.basicConfig(level=logging.INFO)


def main_mathopt():
    file = [f for f in get_data_available() if "pigment20c.psp" in f][0]
    problem = parse_file(file)
    p = ParametersMilp.default()
    solver = MathOptCapacitatedLotSizingSolver(problem)
    solver.init_model()
    res = solver.solve(
        mathopt_solver_type=mathopt.SolverType.HIGHS,
        mathopt_enable_output=True,
        parameters_milp=p,
        time_limit=30,
    )
    sol = res[-1][0]
    print(solver.aggreg_from_sol(sol))
    print(problem.infos["known_bound"])
    print(problem.evaluate(sol), problem.satisfy(sol))


def main_gurobi():
    file = [f for f in get_data_available() if "pigment20c.psp" in f][0]
    problem = parse_file(file)
    p = ParametersMilp.default()
    solver = GurobiCapacitatedLotSizingSolver(problem)
    solver = MilpCapacitatedLotSizingSolver(problem)
    solver.init_model()
    res = solver.solve(
        gurobi_solver_kwargs={"NoRelHeurTime": 3, "Heuristics": 0.2, "Threads": 10},
        parameters_milp=p,
        time_limit=30,
    )
    sol = res[-1][0]
    print(solver.aggreg_from_sol(sol))
    print(problem.infos["known_bound"])
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    main_gurobi()
