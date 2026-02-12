from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.dp import DpSalbpSolver, dp


def run_dp():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_296" in f][0]
    problem = parse_alb_file(file)
    solver = DpSalbpSolver(problem)
    solver.init_model()
    res = solver.solve(retrieve_intermediate_solutions=True, solver=dp.CABS, threads=8)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_dp()
