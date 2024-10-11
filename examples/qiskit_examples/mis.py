import networkx as nx
from qiskit_aer import AerSimulator

from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.quantum import QaoaMisSolver


def quantum_mis():
    """
    in this example we solve a small mis problem using a quantum hybrid algorithm : QAOA
    this algorithm is an approximate algorithm and it's not deterministic
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
    # we create an instance of a QaoaMisSolver
    misSolver = QaoaMisSolver(misProblem)
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()
    # we solve the mis problem
    """
    by default we use a quantum simulator to solve the problem, a AerSimulator() but it's possible to use
    any backend (the same simulator with defined parameters, an other simulator or any real quantum device we can
    use as a qiskit backend)
    for this you have just to define your own backend and then pass it at the creation of the solver or
    when you use the solve function of the solver
    """
    backend = AerSimulator()
    res = misSolver.solve(backend=backend)

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print("This solution respect all constraints : ", misProblem.satisfy(sol))
