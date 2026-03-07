from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.dp import DpSalbpSolver, dp


def test_dp():
    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = DpSalbpSolver(problem)
    solver.init_model()
    # greedy = GreedySalbpSolver(problem)
    # sol = greedy.solve()[-1][0]
    # solver.set_warm_start(sol)
    res = solver.solve(
        retrieve_intermediate_solutions=True, solver=dp.CABS, time_limit=10, threads=8
    )
    sol = res[-1][0]
    assert problem.satisfy(sol)
