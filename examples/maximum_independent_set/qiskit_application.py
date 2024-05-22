import networkx as nx

from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.solvers.mis_quantum import (
    QAOAMisSolver,
)


def qiskit_example():

    """
    We are using a quantum simulator here, these simulator can assume a very low number of variable,
    so we can use it only on very little example
    """

    # we construct a little graph with 6 nodes and 8 edges
    # here the mis is {1,5,6}

    graph = nx.Graph()

    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)

    # we create an instance of MisProblem
    misProblem = MisProblem(graph)
    # we create a solver using the QAOA algorithm
    misSolver = QAOAMisSolver(misProblem)
    # then we just have to initialize the model
    # here it's correspond to pass the problem into this quadratic formulation
    # then we can solve the mis problem
    misSolver.init_model()
    res = misSolver.solve()

    sol, fit = res.get_best_solution_fit()
    print(sol.chosen)
