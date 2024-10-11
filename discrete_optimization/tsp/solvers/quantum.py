#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

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
from discrete_optimization.tsp.problem import Point2DTspProblem, TspSolution, length
from discrete_optimization.tsp.solvers import TspSolver

logger = logging.getLogger(__name__)

if qiskit_available:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import OptimizationResult
    from qiskit_optimization.applications import OptimizationApplication
else:
    msg = (
        "Tsp2dQiskit, QaoaTspSolver, VqeTspSolver, "
        "need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed."
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)
    OptimizationApplication = object
    OptimizationResult = object
    QuadraticProgram = object


class Tsp2dQiskit(OptimizationApplication):
    def __init__(self, problem: Point2DTspProblem) -> None:
        """
        Args:
            problem : the TSP problem instance
        """
        self.problem = problem

    def to_quadratic_program(self) -> QuadraticProgram:
        quadratic_program = QuadraticProgram()

        # X_i,j == 1 if the point i is take in j_ième position

        var_names = {}
        for i in range(0, self.problem.length_permutation):
            for j in range(0, self.problem.length_permutation):
                x_new = quadratic_program.binary_var("x" + str(i) + str(j))
                var_names[(i, j)] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        for i in range(0, self.problem.length_permutation):

            var = var_names[(i, 0)]
            coeff = length(
                self.problem.list_points[
                    self.problem.original_indices_to_permutation_indices[i]
                ],
                self.problem.list_points[self.problem.start_index],
            )
            quadratic[var, var] = coeff

            var = var_names[(i, self.problem.length_permutation - 1)]
            coeff = length(
                self.problem.list_points[
                    self.problem.original_indices_to_permutation_indices[i]
                ],
                self.problem.list_points[self.problem.end_index],
            )
            quadratic[var, var] = coeff

            for k in range(0, self.problem.length_permutation):
                for j in range(0, self.problem.length_permutation - 1):
                    if k != i:
                        var1 = var_names[(i, j)]
                        var2 = var_names[(k, j + 1)]
                        coeff = length(
                            self.problem.list_points[
                                self.problem.original_indices_to_permutation_indices[i]
                            ],
                            self.problem.list_points[
                                self.problem.original_indices_to_permutation_indices[k]
                            ],
                        )
                        quadratic[var1, var2] = coeff

        # each point must be taken exactly one times ( indice i )
        # each position must be chosen exactly one times ( indice j )

        p = self.problem.evaluate(self.problem.get_dummy_solution())["length"]

        for i in range(0, self.problem.length_permutation):
            for j in range(0, self.problem.length_permutation):
                if j != 0 and j != self.problem.length_permutation - 1:
                    quadratic[var_names[(i, j)], var_names[(i, j)]] = -p
                else:
                    quadratic[var_names[(i, j)], var_names[(i, j)]] += -p
                for k in range(j + 1, self.problem.length_permutation):
                    quadratic[var_names[(i, j)], var_names[(i, k)]] = 2 * p

        for j in range(0, self.problem.length_permutation):
            for i in range(0, self.problem.length_permutation):
                quadratic[var_names[(i, j)], var_names[(i, j)]] += -p
                for k in range(i + 1, self.problem.length_permutation):
                    quadratic[var_names[(i, j)], var_names[(k, j)]] = 2 * p

        quadratic_program.minimize(constant, linear, quadratic)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        x = self._result_to_x(result)

        start_index = self.problem.start_index
        end_index = self.problem.end_index
        permutation = [None] * self.problem.length_permutation

        index_curr = 0
        for i in range(0, self.problem.length_permutation):
            for j in range(0, self.problem.length_permutation):
                if x[index_curr] >= 0.5:
                    permutation[j] = i + 1
                index_curr += 1

        sol = TspSolution(
            problem=self.problem,
            start_index=start_index,
            end_index=end_index,
            permutation=permutation,
        )

        return sol


class QaoaTspSolver(TspSolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: Point2DTspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.tsp_qiskit = Tsp2dQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.tsp_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.tsp_qiskit.interpret(result)


class VqeTspSolver(TspSolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: Point2DTspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.tsp_qiskit = Tsp2dQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.tsp_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.tsp_qiskit.interpret(result)
