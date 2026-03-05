from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.greedy import GreedySalbpSolver


def run_greedy():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_337.alb" in f][0]
    problem = parse_alb_file(file)
    solver = GreedySalbpSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_greedy()
