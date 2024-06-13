import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.utils import validate_bounds, validate_initial_point
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import SamplerV2, Session
    from qiskit_optimization.converters import QuadraticProgramToQubo
except ImportError:
    qiskit_available = False
    msg = (
        "QiskitQAOASolver and QiskitVQESolver need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed. "
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)

    class QiskitQAOASolver(SolverDO):
        def __init__(
            self,
            problem: Problem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            backend: Optional = None,
            **kwargs: Any,
        ):
            raise RuntimeError(msg)

    class QiskitVQESolver(SolverDO):
        def __init__(
            self,
            problem: Problem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            backend: Optional = None,
            **kwargs: Any,
        ):
            raise RuntimeError(msg)

else:
    qiskit_available = True


def get_result_from_dict_result(dict_result: Dict[(str, int)]) -> np.ndarray:
    """
    @param dict_result: dictionnary where keys are qubit's value and values are the number of time where this qubit's value have been chosen
    @return: the qubit's value the must often chose
    """
    m = 0
    k = None
    for key, value in dict_result.items():
        if value > m:
            m = value
            k = key
    res = list(k)
    res.reverse()
    res = [int(x) for x in res]
    return np.array(res)


def cost_func(params, ansatz, hamiltonian, estimator, callback_dict):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        callback_dict: dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    cost = result[0].data.evs[0]

    callback_dict["iters"] += 1
    callback_dict["prev_vector"] = params
    callback_dict["cost_history"].append(cost)
    print(f"Iters. done: {callback_dict['iters']} [Current cost: {cost}]")

    return cost


def execute_ansatz_with_Hamiltonian(
    backend, ansatz, hamiltonian, **kwargs
) -> np.ndarray:
    """
    @param backend: the backend use to run the circuit (simulator or real device)
    @param ansatz: the quantum circuit
    @param hamiltonian: the hamiltonian corresponding to the problem
    @param kwargs: a list of parameters who can be specified
    @return: the qubit's value the must often chose
    """

    if backend is None:
        backend = AerSimulator()
    with_session = kwargs.get("with_session", False)
    optimization_level = kwargs.get("optimization_level", 3)
    nb_shots = kwargs.get("nb_shot", 10000)
    maxiter = kwargs.get("maxiter", None)
    disp = kwargs.get("disp", False)

    # transpile and optimize the quantum circuit depending on the device who are going to use
    # there are four level_optimization, to 0 to 3, 3 is the better but also the longest
    target = backend.target
    pm = generate_preset_pass_manager(
        target=target, optimization_level=optimization_level
    )
    ansatz = pm.run(ansatz)
    hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    # open a session
    if with_session:
        session = Session(backend=backend, max_time="2h")
    else:
        session = None

    # Configure estimator
    estimator = Estimator(backend=backend, session=session)
    estimator.options.default_shots = nb_shots

    # Configure sampler
    sampler = SamplerV2(backend=backend, session=session)
    sampler.options.default_shots = nb_shots

    callback_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }
    # step of minimization
    res = minimize(
        cost_func,
        validate_initial_point(point=None, circuit=ansatz),
        args=(ansatz, hamiltonian, estimator, callback_dict),
        method="COBYLA",
        bounds=validate_bounds(ansatz),
        options={"maxiter": maxiter, "disp": disp},
    )

    # Assign solution parameters to our ansatz
    qc = ansatz.assign_parameters(res.x)
    # Add measurements to our circuit
    qc.measure_all()
    # transpile our circuit
    qc = pm.run(qc)
    # run our circuit with optimal parameters find at the minimization step
    results = sampler.run([qc]).result()
    # extract a dictionnary of results, key is binary values of variable and value is number of time of these values has been found
    result = get_result_from_dict_result(results[0].data.meas.get_counts())

    # Close the session since we are now done with it
    if with_session:
        session.close()

    return result


class QiskitQAOASolver(SolverDO):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function)
        self.quadratic_programm = None
        self.backend = backend

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        backend: Optional = None,
        **kwargs: Any,
    ) -> ResultStorage:

        reps = kwargs.get("reps", 2)

        if backend is not None:
            self.backend = backend

        if self.quadratic_programm is None:
            self.init_model()
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )

        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.quadratic_programm)
        hamiltonian, offset = qubo.to_ising()
        ansatz = QAOAAnsatz(hamiltonian, reps=reps)

        result = execute_ansatz_with_Hamiltonian(
            self.backend, ansatz, hamiltonian, **kwargs
        )
        result = conv.interpret(result)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            [(sol, fit)], mode_optim=self.params_objective_function.sense_function
        )

    @abstractmethod
    def init_model(self) -> None:
        ...

    @abstractmethod
    def retrieve_current_solution(self, result) -> Solution:
        """Retrieve current solution from qiskit result.

        Args:
            result: list of value for each binary variable of the problem

        Returns:
            the converted solution at d-o format

        """
        ...


class QiskitVQESolver(SolverDO):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function)
        self.quadratic_programm = None
        self.nb_variable = 0
        self.backend = backend

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        backend: Optional = None,
        **kwargs: Any,
    ) -> ResultStorage:

        if backend is not None:
            self.backend = backend

        if self.quadratic_programm is None or self.nb_variable == 0:
            self.init_model()
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )
            if self.nb_variable == 0:
                raise RuntimeError(
                    "self.variable must not be 0 after self.init_model()."
                )

        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.quadratic_programm)
        hamiltonian, offset = qubo.to_ising()
        ansatz = EfficientSU2(hamiltonian.num_qubits)

        result = execute_ansatz_with_Hamiltonian(
            self.backend, ansatz, hamiltonian, **kwargs
        )
        result = conv.interpret(result)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            [(sol, fit)], mode_optim=self.params_objective_function.sense_function
        )

    @abstractmethod
    def init_model(self) -> None:
        ...

    @abstractmethod
    def retrieve_current_solution(self, result) -> Solution:
        """Retrieve current solution from qiskit result.

        Args:
            result: list of value for each binary variable of the problem

        Returns:
            the converted solution at d-o format

        """
        ...
