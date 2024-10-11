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
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers import KnapsackSolver

logger = logging.getLogger(__name__)

if qiskit_available:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import OptimizationResult
    from qiskit_optimization.applications import OptimizationApplication
    from qiskit_optimization.converters import (
        InequalityToEquality,
        IntegerToBinary,
        MaximizeToMinimize,
        QuadraticProgramToQubo,
    )
else:
    msg = (
        "KnapsackQiskit, QaoaKnapsackSolver, VqeKnapsackSolver, "
        "need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed."
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)
    OptimizationApplication = object
    OptimizationResult = object
    QuadraticProgram = object


class KnapsackQiskit(OptimizationApplication):
    def __init__(self, problem: KnapsackProblem) -> None:
        """
        Args:
            problem : the knapsack problem instance
        """
        self.problem = problem

    def to_quadratic_program(self) -> QuadraticProgram:

        quadratic_program = QuadraticProgram()
        var_names = {}
        for x in range(0, self.problem.nb_items):
            x_new = quadratic_program.binary_var("item" + str(x))
            var_names[x] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        # add to the objective function the value of each item
        for item in self.problem.list_items:
            i = item.index
            quadratic[var_names[i], var_names[i]] = item.value

        # create the constraint to respect the capacity constraint
        constraint = {}
        for item in self.problem.list_items:
            i = item.index
            constraint[var_names[i]] = item.weight
        quadratic_program.linear_constraint(constraint, "<=", self.problem.max_capacity)

        # transform the inequality constraint into an equality constraint adding integer variable
        conv = InequalityToEquality()
        quadratic_program = conv.convert(quadratic_program)

        # TODO optimize bounds of integer variable

        # transform the integer variable into binary variables
        conv = IntegerToBinary()
        quadratic_program = conv.convert(quadratic_program)

        # add the constraint as a penalities function in the objective
        p = sum([item.value for item in self.problem.list_items])

        row = quadratic_program.get_linear_constraint(0).linear.to_dict()

        for i in range(0, quadratic_program.get_num_vars()):

            weight = 2 * row[i] * self.problem.max_capacity - row[i] ** 2
            if i < self.problem.nb_items:
                quadratic[
                    quadratic_program.get_variable(i).name,
                    quadratic_program.get_variable(i).name,
                ] += (
                    p * weight
                )
            else:
                quadratic[
                    quadratic_program.get_variable(i).name,
                    quadratic_program.get_variable(i).name,
                ] = (
                    p * weight
                )

            for j in range(i + 1, (quadratic_program.get_num_vars())):
                weight = row[i] * row[j]
                quadratic[
                    quadratic_program.get_variable(i).name,
                    quadratic_program.get_variable(j).name,
                ] = (
                    -2 * p * weight
                )

        quadratic_program.remove_linear_constraint(0)

        quadratic_program.maximize(constant, linear, quadratic)
        conv = MaximizeToMinimize()
        quadratic_program = conv.convert(quadratic_program)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        list_taken = list(self._result_to_x(result))[: self.problem.nb_items]

        objective = 0
        weight = 0
        for item in self.problem.list_items:
            i = item.index
            if list_taken[i] == 1:
                objective += item.value
                weight += item.weight

        sol = KnapsackSolution(
            problem=self.problem,
            value=objective,
            weight=weight,
            list_taken=list_taken,
        )

        return sol


class QaoaKnapsackSolver(KnapsackSolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.knapsack_qiskit = KnapsackQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.knapsack_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.knapsack_qiskit.interpret(result)


class VqeKnapsackSolver(KnapsackSolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.knapsack_qiskit = KnapsackQiskit(problem)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.knapsack_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.knapsack_qiskit.interpret(result)
