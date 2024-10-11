#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Optional, Union

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.qiskit_tools import (
    QiskitQaoaSolver,
    QiskitVqeSolver,
    qiskit_available,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

logger = logging.getLogger(__name__)


if qiskit_available:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import OptimizationResult
    from qiskit_optimization.applications import OptimizationApplication
else:
    msg = (
        "MisQiskit, QaoaMisSolver, and VqeMisSolver need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed."
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)
    OptimizationApplication = object
    OptimizationResult = object
    QuadraticProgram = object


class MisQiskit(OptimizationApplication):
    def __init__(self, problem: MisProblem) -> None:
        self.problem = problem

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> MisSolution:
        return MisSolution(problem=self.problem, chosen=self._result_to_x(result))

    def to_quadratic_program(self) -> QuadraticProgram:

        adj = nx.to_numpy_array(self.problem.graph_nx)
        J = np.identity(self.problem.number_nodes)
        A = J - adj

        quadratic_program = QuadraticProgram()
        var_names = {}
        for x in range(0, self.problem.number_nodes):
            x_new = quadratic_program.binary_var("v" + str(x))
            var_names[x] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        for i in range(0, self.problem.number_nodes):
            for j in range(i, self.problem.number_nodes):
                quadratic[(var_names[i], var_names[j])] = A[i][j]

        # we maximize the number of node in the independent set
        quadratic_program.maximize(constant, linear, quadratic)

        return quadratic_program


class QaoaMisSolver(MisSolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.mis_qiskit = MisQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.mis_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result):
        return self.mis_qiskit.interpret(result)


class VqeMisSolver(MisSolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.mis_qiskit = MisQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.mis_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.mis_qiskit.interpret(result)
