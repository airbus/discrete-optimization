import networkx as nx

from discrete_optimization.generic_tools.qiskit_tools import (
    GeneralQaoaSolver,
    GeneralVqeSolver,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.gurobi import GurobiMisSolver


def quantum_generalQAOA():

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
    # we create a Milp Solver to create the MILP model
    milpSolver = GurobiMisSolver(problem=misProblem)

    # we create the retrieve function solution, for misProblem no reconstruction of the solution is needed
    def fun(x):
        return MisSolution(problem=misProblem, chosen=x)

    # we create an instance of a GeneralQaoaSolver
    misSolver = GeneralQaoaSolver(
        problem=misProblem, model=milpSolver, retrieve_solution=fun
    )
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()
    # we solve the mis problem
    res = misSolver.solve()

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print("This solution respect all constraints : ", misProblem.satisfy(sol))


def quantum_generalVQE():

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
    # we create a Milp Solver to create the MILP model
    milpSolver = GurobiMisSolver(problem=misProblem)

    # we create the retrieve function solution, for misProblem no reconstruction of the solution is needed
    def fun(x):
        return MisSolution(problem=misProblem, chosen=x)

    # we create an instance of a GeneralQaoaSolver
    misSolver = GeneralVqeSolver(
        problem=misProblem, model=milpSolver, retrieve_solution=fun
    )
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()
    # we solve the mis problem
    res = misSolver.solve()

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print("This solution respect all constraints : ", misProblem.satisfy(sol))
