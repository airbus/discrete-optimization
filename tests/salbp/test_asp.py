import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.asp import AspSalbpSolver
from discrete_optimization.salbp.solvers.greedy import GreedySalbpSolver


def test_clingo():
    files = get_data_available()
    file = [f for f in files if "instance_n=20_10" in f][0]
    problem = parse_alb_file(file)
    solver = AspSalbpSolver(problem)
    greedy = GreedySalbpSolver(problem)
    res = greedy.solve()
    sol = res[-1][0]
    nb_stations = problem.evaluate(sol)["nb_stations"]
    print(nb_stations)
    solver.init_model(upper_bound=nb_stations)
    solver.set_warm_start(res[-1][0])
    res = solver.solve(
        time_limit=10,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol = res[-1][0]
    assert problem.satisfy(sol)
