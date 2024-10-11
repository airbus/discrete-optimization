#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

import numpy as np

from discrete_optimization.facility.problem import (
    FacilityProblem,
    FacilitySolution,
    length,
)
from discrete_optimization.facility.solvers import FacilitySolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.qiskit_tools import (
    QiskitQaoaSolver,
    QiskitVqeSolver,
    qiskit_available,
)

logger = logging.getLogger(__name__)

if qiskit_available:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import OptimizationResult
    from qiskit_optimization.applications import OptimizationApplication
    from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary
else:
    msg = (
        "FacilityQiskit, QaoaFacilitySolver, VqeFacilitySolver, "
        "need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed."
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)
    OptimizationApplication = object
    OptimizationResult = object
    QuadraticProgram = object


class FacilityQiskit(OptimizationApplication):
    def __init__(self, problem: FacilityProblem) -> None:
        """
        Args:
            problem : the TSP problem instance
        """
        self.problem = problem

    def to_quadratic_program(self) -> QuadraticProgram:
        quadratic_program = QuadraticProgram()

        # X_i == 1 if facility i is used by at least one client
        # X_i,j == 1 facility i is given to customers j

        var_names = {}
        for i in range(0, self.problem.facility_count):
            for j in range(0, self.problem.customer_count):
                x_new = quadratic_program.binary_var("x" + str(i) + str(j))
                var_names[(i, j)] = x_new.name
            x_new = quadratic_program.binary_var("x" + str(i))
            var_names[i] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        for i in range(0, self.problem.facility_count):
            quadratic[var_names[i], var_names[i]] = self.problem.facilities[
                i
            ].setup_cost
            for j in range(0, self.problem.customer_count):
                quadratic[var_names[(i, j)], var_names[(i, j)]] = length(
                    self.problem.facilities[i].location,
                    self.problem.customers[j].location,
                )

        p = 0
        for i in range(0, self.problem.facility_count):
            p += self.problem.facilities[i].setup_cost
            for j in range(0, self.problem.customer_count):
                p += length(
                    self.problem.facilities[i].location,
                    self.problem.customers[j].location,
                )

        # a facility is used if unless one customer used it
        # X_i >= X_i_j pour tout j
        for i in range(0, self.problem.facility_count):
            for j in range(0, self.problem.customer_count):
                quadratic[var_names[i], var_names[(i, j)]] = -p
                quadratic[var_names[(i, j)], var_names[(i, j)]] += p

        # the sum of customer's demand who used a facility can't excess the facility's capacity
        # sum j X_i_j*j.demand <= i.capacity

        for i in range(0, self.problem.facility_count):
            c2 = {}
            for j in range(0, self.problem.customer_count):
                c2[var_names[(i, j)]] = self.problem.customers[j].demand
            quadratic_program.linear_constraint(
                c2, "<=", self.problem.facilities[i].capacity
            )

        # transform the inequality constraint into an equality constraint adding integer variable
        conv = InequalityToEquality()
        quadratic_program = conv.convert(quadratic_program)

        # TODO optimize bounds of integer variable

        # transform the integer variable into binary variables
        conv = IntegerToBinary()
        quadratic_program = conv.convert(quadratic_program)

        for j in range(0, self.problem.facility_count):

            row = quadratic_program.get_linear_constraint(0).linear.to_dict()

            for i in row.keys():

                weight = 2 * row[i] * self.problem.facilities[j].capacity - row[i] ** 2
                if i < self.problem.customer_count:
                    quadratic[
                        quadratic_program.get_variable(i).name,
                        quadratic_program.get_variable(i).name,
                    ] += (
                        -p * weight
                    )
                else:
                    quadratic[
                        quadratic_program.get_variable(i).name,
                        quadratic_program.get_variable(i).name,
                    ] = (
                        -p * weight
                    )

                for k in row.keys():
                    if k > i:
                        weight = row[i] * row[k]
                        quadratic[
                            quadratic_program.get_variable(i).name,
                            quadratic_program.get_variable(k).name,
                        ] = (
                            2 * p * weight
                        )

            quadratic_program.remove_linear_constraint(0)

        # only one facility by customer
        # sum i X_i_j == 1

        for i in range(0, self.problem.customer_count):
            for j in range(0, self.problem.facility_count):
                quadratic[var_names[(i, j)], var_names[(i, j)]] += -p
                for k in range(j + 1, self.problem.facility_count):
                    quadratic[var_names[(i, j)], var_names[(i, k)]] += 2 * p
            constant += p

        quadratic_program.minimize(constant, linear, quadratic)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        x = self._result_to_x(result)

        facility_for_customers = [-1] * self.problem.customer_count

        for i in range(0, self.problem.facility_count):
            for j in range(0, self.problem.customer_count):
                if x[self.problem.facility_count * j + i + j] == 1:
                    facility_for_customers[j] = self.problem.facilities[i].index

        sol = FacilitySolution(
            self.problem, facility_for_customers=facility_for_customers
        )

        return sol


class QaoaFacilitySolver(FacilitySolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.facility_qiskit = FacilityQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.facility_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.facility_qiskit.interpret(result)


class VqeFacilitySolver(FacilitySolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.facility_qiskit = FacilityQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.facility_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.facility_qiskit.interpret(result)
