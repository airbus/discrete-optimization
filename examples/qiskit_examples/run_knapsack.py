from discrete_optimization.knapsack.problem import Item, KnapsackProblem
from discrete_optimization.knapsack.solvers.quantum import VqeKnapsackSolver


def knapsack_example():

    """
    We are using a quantum simulator here, these simulator can assume a very low number of variable,
    so we can use it only on very little example
    """

    # define a knapsack problem with 6 items and a capacity of 10
    max_capacity = 10

    i1 = Item(0, 4, 2)
    i2 = Item(1, 5, 2)
    i3 = Item(2, 4, 3)
    i4 = Item(3, 2, 1)
    i5 = Item(4, 5, 3)
    i6 = Item(5, 2, 1)

    # solving the knapsack problem using the VQE algorithm
    knapsackProblem = KnapsackProblem([i1, i2, i3, i4, i5, i6], max_capacity)
    knapsackSolver = VqeKnapsackSolver(knapsackProblem)
    knapsackSolver.init_model()
    kwargs = {"maxiter": 200}
    res = knapsackSolver.solve(**kwargs)
    sol, _ = res.get_best_solution_fit()
    print(sol)
