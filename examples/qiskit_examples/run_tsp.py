from discrete_optimization.tsp.problem import Point2D, Point2DTspProblem
from discrete_optimization.tsp.solvers.quantum import VqeTspSolver


def tsp_example():

    """
    We are using a quantum simulator here, these simulator can assume a very low number of variable,
    so we can use it only on very little example
    """

    # a TSP problem with five points, we can change start_index and end_index as we want,
    # they can be None too, in this case start_index = end_index = 0

    p1 = Point2D(0, 0)
    p2 = Point2D(-1, 1)
    p3 = Point2D(1, -1)
    p4 = Point2D(1, 1)
    p5 = Point2D(1, -2)

    tspProblem = Point2DTspProblem([p1, p2, p3, p4, p5], 5, start_index=0, end_index=4)
    tspSolver = VqeTspSolver(tspProblem)
    kwargs = {"maxiter": 500}
    tspSolver.init_model()
    res = tspSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol)
    print(tspProblem.satisfy(sol))
