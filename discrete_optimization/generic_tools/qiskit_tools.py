from abc import abstractmethod
from typing import Any, List, Optional

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

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


class QiskitQAOASolver(SolverDO):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any
    ):
        super().__init__(problem, params_objective_function)
        self.quadratic_programm = None

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:

        optimizer = kwargs.get("optimizer", SPSA(maxiter=250))
        reps = kwargs.get("reps", 5)

        if self.quadratic_programm is None:
            self.init_model(**kwargs)
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )

        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
        algorithm = MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(self.quadratic_programm)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            [(sol, fit)], mode_optim=self.params_objective_function.sense_function
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


class QiskitVQESolver(SolverDO):
    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any
    ):
        super().__init__(problem, params_objective_function)
        self.quadratic_programm = None
        self.nb_variable = 0

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:

        optimizer = kwargs.get("optimizer", SPSA(maxiter=300))
        reps = kwargs.get("reps", 5)

        if self.quadratic_programm is None or self.nb_variable == 0:
            self.init_model(**kwargs)
            if self.quadratic_programm is None:
                raise RuntimeError(
                    "self.quadratic_programm must not be None after self.init_model()."
                )
            if self.nb_variable == 0:
                raise RuntimeError(
                    "self.variable must not be 0 after self.init_model()."
                )

        ry = TwoLocal(self.nb_variable, "ry", "cz", reps=reps, entanglement="linear")
        vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)
        algorithm = MinimumEigenOptimizer(vqe)
        result = algorithm.solve(self.quadratic_programm)

        sol = self.retrieve_current_solution(result)
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            [(sol, fit)], mode_optim=self.params_objective_function.sense_function
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
