import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.quantum import (
    QaoaMisSolver,
    VqeMisSolver,
)


def quantum_personnalized_QAOA():
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
    for a more advanced usage of QAOA algorith, it's possible to choose how we initialize qubits
    and what "mixer operator" we want to use in our ansatz.
    The parameter "initial_state must be a QuantumCircuit Object
    The parameter "mixer_operator must be BaseOperator or QuantumCircuit Object
    """
    backend = AerSimulator()
    kwargs = {
        "initial_state": QuantumCircuit(misSolver.quadratic_programm.get_num_vars())
    }
    num_qubits = misSolver.quadratic_programm.get_num_vars()

    # The Basic Mixer is just a sum of single gate X's on each qubit. For this example
    # we simply define a sum of single gate Z's on each qubit.
    mixer_terms = [
        ("I" * left + "Z" + "I" * (num_qubits - left - 1), 1)
        for left in range(num_qubits)
    ]
    mixer = SparsePauliOp.from_list(mixer_terms)
    kwargs["mixer_operator"] = mixer
    res = misSolver.solve(backend=backend, **kwargs)

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print("This solution respect all constraints : ", misProblem.satisfy(sol))


def quantum_personnalized_VQE():
    """
    in this example we solve a small mis problem using a quantum hybrid algorithm : VQE
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
    # we create an instance of a VqeMisSolver
    misSolver = VqeMisSolver(misProblem)
    # we initialize the solver, in fact this step transform the problem in a QUBO formulation
    misSolver.init_model()
    # we solve the mis problem
    """
    for a more advanced usage of VQE algorithm, it's possible to choose yourself
    the parametrized ansatz you want to use. He must have the same number of qubits that the problem.
    Here we use the EfficientSU2 fonction of qiskit, to see all possible parameters you can look here :
    https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.EfficientSU2
    The parameter "personnalized_ansatz" must be a QuantumCircuit Object
    """
    backend = AerSimulator()
    kwargs = {
        "personnalized_ansatz": EfficientSU2(
            misSolver.quadratic_programm.get_num_vars()
        )
    }
    res = misSolver.solve(backend=backend, **kwargs)

    sol, fit = res.get_best_solution_fit()
    print(sol)
    print("This solution respect all constraints : ", misProblem.satisfy(sol))
