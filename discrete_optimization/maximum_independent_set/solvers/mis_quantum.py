from typing import Optional, Union

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.qiskit_tools import (
    QiskitQAOASolver,
    QiskitVQESolver,
)
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class MisQiskit(OptimizationApplication):
    def __init__(self, problem: MisProblem) -> None:
        self.problem = problem
        self.nb_variable = self.problem.number_nodes

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> MisSolution:
        return MisSolution(problem=self.problem, chosen=self._result_to_x(result))

    def to_quadratic_program(self) -> QuadraticProgram:

        quadratic_program = QuadraticProgram()
        var_names = {}
        for x in range(0, self.problem.number_nodes):
            x_new = quadratic_program.binary_var("v" + str(x))
            var_names[x] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        for i in range(0, self.problem.number_nodes):
            linear[var_names[i]] = 1

        # we maximize the number of node in the independent set
        quadratic_program.maximize(constant, linear, quadratic)

        # for each edge, his two nodes can't be in the same independent set
        for edge in self.problem.graph_nx.edges():
            constraint = {
                self.problem.nodes_to_index[edge[0]]: 1,
                self.problem.nodes_to_index[edge[1]]: 1,
            }
            quadratic_program.linear_constraint(constraint, "<=", 1)

        return quadratic_program


class QAOAMisSolver(MisSolver, QiskitQAOASolver):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        super().__init__(problem, params_objective_function)
        self.mis_qiskit = MisQiskit(problem)

    def init_model(self):
        self.quadratic_programm = self.mis_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result):
        return self.mis_qiskit.interpret(result)


class VQEMisSolver(MisSolver, QiskitVQESolver):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        super().__init__(problem, params_objective_function)
        self.mis_qiskit = MisQiskit(problem)

    def init_model(self):
        self.quadratic_programm = self.mis_qiskit.to_quadratic_program()
        self.nb_variable = self.mis_qiskit.nb_variable

    def retrieve_current_solution(self, result) -> Solution:
        return self.mis_qiskit.interpret(result)
