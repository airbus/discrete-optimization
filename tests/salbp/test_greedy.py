from discrete_optimization.alb.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.alb.salbp.solvers.greedy import GreedySalbpSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp


def test_greedy():
    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = GreedySalbpSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)
