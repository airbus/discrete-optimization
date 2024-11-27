import logging
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np

try:
    import gurobipy
except ImportError:
    gurobi_available = False
    Model = object
else:
    gurobi_available = True
    from gurobipy import Model

from scipy.optimize import minimize

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
    from qiskit.quantum_info.operators.base_operator import BaseOperator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_algorithms.utils import validate_bounds, validate_initial_point
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import SamplerV2, Session
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import from_gurobipy
except ImportError:
    qiskit_available = False
    msg = (
        "QiskitQaoaSolver and QiskitVqeSolver need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed. "
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)

    class QiskitQaoaSolver(SolverDO):
        def __init__(
            self,
            problem: Problem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            backend: Optional = None,
            **kwargs: Any,
        ):
            raise RuntimeError(msg)

    class QiskitVqeSolver(SolverDO):
        def __init__(
            self,
            problem: Problem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            backend: Optional = None,
            **kwargs: Any,
        ):
            raise RuntimeError(msg)

    QuadraticProgram = object

else:
    qiskit_available = True


def get_result_from_dict_result(dict_result: dict[(str, int)]) -> np.ndarray:
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
    logger.info(f"Iters. done: {callback_dict['iters']} [Current cost: {cost}]")

    return cost


def execute_ansatz_with_Hamiltonian(
    solver, backend, ansatz, hamiltonian, use_session: Optional[bool] = False, **kwargs
) -> np.ndarray:
    """
    @param solver: the solver who solve the problem, must be a QiskitSolver
    @param backend: the backend use to run the circuit (simulator or real device)
    @param ansatz: the quantum circuit
    @param hamiltonian: the hamiltonian corresponding to the problem
    @param use_session: boolean to set to True for use a session
    @param kwargs: a list of hyperparameters who can be specified
    @return: the qubit's value the must often chose
    """

    # If no backend is given we use a quantum simulator
    if backend is None:
        backend = AerSimulator()

    optimization_level = kwargs["optimization_level"]
    nb_shots = kwargs["nb_shots"]

    # transpile and optimize the quantum circuit depending on the device who are going to use
    # there are four level_optimization, to 0 to 3, 3 is the better but also the longest
    target = backend.target
    pm = generate_preset_pass_manager(
        target=target, optimization_level=optimization_level
    )
    new_ansatz = pm.run(ansatz)
    solver.set_executed_ansatz(new_ansatz)
    hamiltonian = hamiltonian.apply_layout(new_ansatz.layout)

    # open a session if desired, only useful for real device
    if use_session:
        max_time = kwargs.get("session_time", "2h")
        session = Session(backend=backend, max_time=max_time)
        mode = session
    else:
        mode = backend

    # Configure estimator
    estimator = Estimator(mode=mode)
    estimator.options.default_shots = nb_shots

    # Configure sampler
    sampler = SamplerV2(mode=mode)
    sampler.options.default_shots = nb_shots

    callback_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }
    # step of minimization
    if kwargs.get("optimizer"):

        def fun(x):
            pub = (new_ansatz, [hamiltonian], [x])
            result = estimator.run(pubs=[pub]).result()
            cost = result[0].data.evs[0]
            callback_dict["iters"] += 1
            callback_dict["prev_vector"] = x
            callback_dict["cost_history"].append(cost)
            logger.info(f"Iters. done: {callback_dict['iters']} [Current cost: {cost}]")
            return cost

        optimizer = kwargs["optimizer"]
        res = optimizer.minimize(
            fun,
            validate_initial_point(point=None, circuit=ansatz),
            bounds=validate_bounds(ansatz),
        )

    else:

        method = kwargs["method"]
        if kwargs.get("options"):
            options = kwargs["options"]
        else:
            if method == "COBYLA":
                options = {
                    "maxiter": kwargs["maxiter"],
                    "rhobeg": kwargs["rhobeg"],
                    "tol": kwargs["tol"],
                }
            else:
                options = {}

        res = minimize(
            cost_func,
            validate_initial_point(point=None, circuit=ansatz),
            args=(new_ansatz, hamiltonian, estimator, callback_dict),
            method=method,
            bounds=validate_bounds(ansatz),
            options=options,
        )

    # Assign solution parameters to our ansatz
    qc = new_ansatz.assign_parameters(res.x)
    solver.set_final_ansatz((qc.copy()))
    # Add measurements to our circuit
    qc.measure_all()
    # transpile our circuit
    qc = pm.run(qc)
    # run our circuit with optimal parameters find at the minimization step
    results = sampler.run([qc]).result()
    # extract a dictionnary of results, key is binary values of variable and value is number of time of these values has been found
    best_result = get_result_from_dict_result(results[0].data.meas.get_counts())

    # Close the session since we are now done with it
    if use_session:
        session.close()

    return best_result


class QiskitSolver(SolverDO):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.executed_ansatz = None
        self.final_ansatz = None

    def set_executed_ansatz(self, ansatz):
        self.executed_ansatz = ansatz

    def set_final_ansatz(self, ansatz):
        self.final_ansatz = ansatz


class QiskitQaoaSolver(QiskitSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="reps", low=1, high=6, default=2),
        IntegerHyperparameter(name="optimization_level", low=0, high=3, default=1),
        CategoricalHyperparameter(name="method", choices=["COBYLA"], default="COBYLA"),
        IntegerHyperparameter(
            name="nb_shots", low=10000, high=100000, step=10000, default=10000
        ),
        IntegerHyperparameter(name="maxiter", low=100, high=1000, step=50, default=300),
        FloatHyperparameter(name="rhobeg", low=0.5, high=1.5, default=1.0),
        CategoricalHyperparameter(name="tol", choices=[1e-1, 1e-2, 1e-3], default=1e-2),
    ]

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.quadratic_programm = None
        self.backend = backend
        self.ansatz = None

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        backend: Optional = None,
        use_session: Optional[bool] = False,
        **kwargs: Any,
    ) -> ResultStorage:

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        reps = kwargs["reps"]

        if backend is not None:
            self.backend = backend

        if self.quadratic_programm is None:
            self.init_model()
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )

        # if needed, in the most case the self.quadratic_programm is already in a QUBO form
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.quadratic_programm)

        hamiltonian, offset = qubo.to_ising()
        mixer_operator = None
        if kwargs.get("mixer_operator"):
            mixer_operator = kwargs["mixer_operator"]
            if not (
                isinstance(mixer_operator, QuantumCircuit)
                or isinstance(mixer_operator, BaseOperator)
            ):
                raise RuntimeError(
                    "your personnalized mixer_operator must be a QuantumCircuit or a BaseOperator Object"
                )
        initial_state = None
        if kwargs.get("initial_state"):
            initial_state = kwargs["initial_state"]
            if not isinstance(initial_state, QuantumCircuit):
                raise RuntimeError("the initial_state must be a QuantumCircuit Object")
        self.ansatz = QAOAAnsatz(
            hamiltonian,
            reps=reps,
            mixer_operator=mixer_operator,
            initial_state=initial_state,
        )

        if kwargs.get("parameter_bounds"):
            if len(kwargs["parameter_bounds"]) != 2 * reps:
                raise RuntimeError(
                    "For custom parameters of the quantum circuit you need to pass 2*reps bounds"
                )
            self.ansatz.parameter_bounds = kwargs["parameter_bounds"]

        else:
            """
            by default only hyperparameters of the mixer operator are initialized
            but for some optimizer we need to initialize also hyperparameters of the cost operator
            """
            bounds = []
            for hp in self.ansatz.parameter_bounds:
                if hp == (None, None):
                    hp = (0, np.pi)
                bounds.append(hp)
            self.ansatz.parameter_bounds = bounds

        result = execute_ansatz_with_Hamiltonian(
            self, self.backend, self.ansatz, hamiltonian, use_session, **kwargs
        )
        result = conv.interpret(result)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )

    @abstractmethod
    def init_model(self, **kwargs: any) -> None:
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


class QiskitVqeSolver(QiskitSolver):
    hyperparameters = [
        IntegerHyperparameter(name="reps", low=1, high=6, default=3),
        IntegerHyperparameter(name="optimization_level", low=0, high=3, default=1),
        CategoricalHyperparameter(name="method", choices=["COBYLA"], default="COBYLA"),
        IntegerHyperparameter(
            name="nb_shots", low=10000, high=100000, step=10000, default=10000
        ),
        IntegerHyperparameter(name="maxiter", low=100, high=2000, step=50, default=300),
        FloatHyperparameter(name="rhobeg", low=0.5, high=1.5, default=1.0),
        CategoricalHyperparameter(name="tol", choices=[1e-1, 1e-2, 1e-3], default=1e-2),
    ]

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.quadratic_programm = None
        self.backend = backend
        self.ansatz = None

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        backend: Optional = None,
        use_session: Optional[bool] = False,
        **kwargs: Any,
    ) -> ResultStorage:

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if backend is not None:
            self.backend = backend

        if self.quadratic_programm is None:
            self.init_model()
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )

        # if needed, in the most case the self.quadratic_programm is already in a QUBO form
        conv = QuadraticProgramToQubo()
        qubo = conv.convert(self.quadratic_programm)

        hamiltonian, offset = qubo.to_ising()
        if kwargs.get("personnalized_ansatz"):
            self.ansatz = kwargs["personnalized_ansatz"]
            if self.ansatz.num_qubits != hamiltonian.num_qubits:
                raise RuntimeError(
                    "your personnalized ansatz must have the same number of qubits that the number of binary variable of the problem"
                )
            if not isinstance(self.ansatz, QuantumCircuit):
                raise RuntimeError(
                    "your personnalized ansatz must be a QuantumCircuit Object"
                )
        else:
            self.ansatz = EfficientSU2(hamiltonian.num_qubits, reps=kwargs["reps"])

        result = execute_ansatz_with_Hamiltonian(
            self, self.backend, self.ansatz, hamiltonian, use_session, **kwargs
        )
        result = conv.interpret(result)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
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


def gurobi_to_qubo(model):
    return from_gurobipy(model)


class GeneralQaoaSolver(QiskitQaoaSolver):
    def __init__(
        self,
        problem: Problem,
        model: Union[Model, GurobiMilpSolver],
        retrieve_solution: Callable,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, backend=backend)
        self.quadratic_programm = None
        self.model = model
        self.retrieve_solution = retrieve_solution

    def retrieve_current_solution(self, result: np.ndarray) -> Solution:
        return self.retrieve_solution(result)

    def init_model(self, **kwargs: Any) -> None:
        if isinstance(self.model, Model):
            self.quadratic_programm = gurobi_to_qubo(self.model)
        else:
            self.model.init_model(**kwargs)
            self.quadratic_programm = gurobi_to_qubo(self.model.model)


class GeneralVqeSolver(QiskitVqeSolver):
    def __init__(
        self,
        problem: Problem,
        model: Union[Model, GurobiMilpSolver],
        retrieve_solution: Callable,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        backend: Optional = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, backend=backend, **kwargs)
        self.quadratic_programm = None
        self.model = model
        self.retrieve_solution = retrieve_solution

    def retrieve_current_solution(self, result) -> Solution:
        return self.retrieve_solution(result)

    def init_model(self, **kwargs: Any) -> None:
        if isinstance(self.model, Model):
            self.quadratic_programm = gurobi_to_qubo(self.model)
        else:
            self.model.init_model(**kwargs)
            self.quadratic_programm = gurobi_to_qubo(self.model.model)


def matrix(quad: QuadraticProgram):
    """
    @param quad: a quadratic programm, must be in QUBO form
    @return: the QUBO matrix
    """
    num_var = quad.get_num_vars()
    m = np.zeros((num_var, num_var))
    obj = quad.objective.quadratic.to_dict()
    for key, val in obj.items():
        m[key[0], key[1]] = val
    return m


def compute_energy(matrix, x):
    """
    @param matrix: a matrix of a QUBO formulation
    @param x: a binary vector
    @return: the value of the matrix for the giving vector
    """
    energy = 0
    for i in range(0, len(x)):
        for j in range(i, len(x)):
            if x[i] == 1 and x[j] == 1:
                energy += matrix[i][j]
    return energy
