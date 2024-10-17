from qiskit_aer import AerSimulator

from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.coloring.solvers.quantum import (
    FeasibleNbColorQaoaColoringSolver,
)
from discrete_optimization.generic_tools.graph_api import Graph


def quantum_coloring():

    """
    in this example we solve a small coloring problem using a quantum hybrid algorithm : QAOA
    this algorithm is an approximate algorithm and it's not deterministic
    """

    # We construct a graph with 4 nodes and three edges, two colors are sufficiant
    nodes = [(1, {}), (2, {}), (3, {}), (4, {})]
    edges = [(1, 2, {}), (1, 3, {}), (2, 4, {})]

    # can make the problem unsat + the number of variable depend on this parameter
    nb_color = 2

    # we create an instance of ColoringProblem
    coloringProblem = ColoringProblem(Graph(nodes=nodes, edges=edges))
    # we create an instance of a QaoaMisSolver
    coloringSolver = FeasibleNbColorQaoaColoringSolver(
        coloringProblem, nb_color=nb_color
    )
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    coloringSolver.init_model()
    # we solve the mis problem
    """
    by default we use a quantum simulator to solve the problem, a AerSimulator() but it's possible to use
    any backend (the same simulator with defined parameters, an other simulator or any real quantum device we can
    use as a qiskit backend)
    for this you have just to define your own backend and then pass it at the creation of the solver or
    when you use the solve function of the solver
    """
    backend = AerSimulator()
    kwargs = {"reps": 4, "optimization_level": 1, "maxiter": 300}
    res = coloringSolver.solve(backend=backend, **kwargs)

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print(
        "Two nodes connected by an edge have never the same color and all nodes have a color: ",
        coloringProblem.satisfy(sol),
    )
