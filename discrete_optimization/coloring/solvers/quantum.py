#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union

import numpy as np

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
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
else:
    msg = (
        "ColoringQiskit_MinimizeNbColor, QAOAColoringSolver_MinimizeNbColor, VQEColoringSolver_MinimizeNbColor, "
        "ColoringQiskit_FeasibleNbColor, QAOAColoringSolver_FeasibleNbColor and VQEColoringSolver_FeasibleNbColor, "
        "need qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime, "
        "and qiskit_optimization to be installed."
        "You can use the command `pip install discrete-optimization[quantum]` to install them."
    )
    logger.warning(msg)
    OptimizationApplication = object
    OptimizationResult = object
    QuadraticProgram = object


class MinimizeNbColorColoringQiskit(OptimizationApplication):
    def __init__(self, problem: ColoringProblem, nb_max_color=None) -> None:
        """
        Args:
            problem : the coloring problem instance
        """
        self.problem = problem
        if nb_max_color is None:
            nb_max_color = self.problem.number_of_nodes
        self.nb_max_color = nb_max_color

    def to_quadratic_program(self) -> QuadraticProgram:
        quadratic_program = QuadraticProgram()

        # X_i,j == 1 if node i take color j
        # C_j == 1 if color j is choosen at least one time

        p = self.nb_max_color

        var_names = {}
        for i in range(0, self.nb_max_color):
            for j in range(0, self.problem.number_of_nodes):
                x_new = quadratic_program.binary_var("x" + str(j) + str(i))
                var_names[(j, i)] = x_new.name
            color_new = quadratic_program.binary_var("color" + str(i))
            var_names[i] = color_new.name

        # We are looking to minimize the number of color used

        constant = 0
        linear = {}
        quadratic = {}

        for i in range(0, self.nb_max_color):
            quadratic[var_names[i], var_names[i]] = 1

        """
        On va ici intégrer sous forme de pénalité les différentes contraintes afin d'avoir directement une formulation QUBO
        x <= y devient P(x-xy)
        x1 + ... + xi = 1 devient P(-x1 + ... + -xi + 2x1x2 + ... + 2x1xi + 2x2x3 + .... + 2x2xi + ... + 2x(i-1)xi)
        x + y <= 1 devient P(xy)
        où P est un scalaire qui doit idéalement être ni trop petit, ni trop grand (ici on prend le nombre de couleur max autorisé)
        """

        # if color j is given to a node, C_j must be 1
        for i in range(0, self.problem.number_of_nodes):
            for j in range(0, self.nb_max_color):
                quadratic[var_names[(i, j)], var_names[(i, j)]] = p
                quadratic[var_names[(i, j)], var_names[j]] = -p

        # each node have only one color
        for i in range(0, self.problem.number_of_nodes):
            for j in range(0, self.nb_max_color):
                quadratic[var_names[(i, j)], var_names[(i, j)]] += -p
                for k in range(j + 1, self.nb_max_color):
                    quadratic[var_names[(i, j)], var_names[(i, k)]] = 2 * p
            constant += p

        # two adjacent nodes can't have the same color
        for edge in self.problem.graph.graph_nx.edges():
            for j in range(0, self.nb_max_color):
                quadratic[
                    var_names[(self.problem.index_nodes_name[edge[0]], j)],
                    var_names[(self.problem.index_nodes_name[edge[1]], j)],
                ] = p

        quadratic_program.minimize(constant, linear, quadratic)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        x = self._result_to_x(result)

        colors = [None] * self.problem.number_of_nodes
        nb_color = 0

        for node in range(0, self.problem.number_of_nodes):
            color_find = False
            color = 0
            while not color_find and color < self.nb_max_color:
                if x[self.problem.number_of_nodes * color + node + color] == 1:
                    colors[node] = color
                    color_find = True
                color += 1

        for color in range(0, self.nb_max_color):
            if (
                x[
                    self.problem.number_of_nodes * color
                    + self.problem.number_of_nodes
                    + color
                ]
                == 1
            ):
                nb_color += 1

        sol = ColoringSolution(self.problem, colors=colors, nb_color=nb_color)

        return sol


class MinimizeNbColorQaoaColoringSolver(ColoringSolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: ColoringProblem,
        nb_max_color=None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.coloring_qiskit = MinimizeNbColorColoringQiskit(
            problem, nb_max_color=nb_max_color
        )

    def init_model(self, **kwargs):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)


class MinimizeNbColorVqeColoringSolver(ColoringSolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: ColoringProblem,
        nb_max_color=None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.coloring_qiskit = MinimizeNbColorColoringQiskit(
            problem, nb_max_color=nb_max_color
        )

    def init_model(self, **kwargs):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)


class FeasibleNbColorColoringQiskit(OptimizationApplication):
    def __init__(self, problem: ColoringProblem, nb_color=None) -> None:
        """
        Args:
            problem : the coloring problem instance
        """
        self.problem = problem
        if nb_color is None:
            nb_color = self.problem.number_of_nodes
        self.nb_color = nb_color

    def to_quadratic_program(self) -> QuadraticProgram:
        quadratic_program = QuadraticProgram()

        # X_i,j == 1 if node i take color j

        var_names = {}
        for i in range(0, self.nb_color):
            for j in range(0, self.problem.number_of_nodes):
                x_new = quadratic_program.binary_var("x" + str(j) + str(i))
                var_names[(j, i)] = x_new.name

        constant = 0
        linear = {}
        quadratic = {}

        p = self.nb_color

        # each node has a unique color
        for i in range(0, self.problem.number_of_nodes):
            for j in range(0, self.nb_color):
                quadratic[var_names[(i, j)], var_names[(i, j)]] = -p
                for k in range(j + 1, self.nb_color):
                    quadratic[var_names[(i, j)], var_names[(i, k)]] = 2 * p

        # two nodes of an edge can't have the same color
        for edge in self.problem.graph.graph_nx.edges():
            for j in range(0, self.nb_color):
                quadratic[
                    var_names[(self.problem.index_nodes_name[edge[0]], j)],
                    var_names[(self.problem.index_nodes_name[edge[1]], j)],
                ] = p

        quadratic_program.minimize(constant, linear, quadratic)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        x = self._result_to_x(result)

        colors = [None] * self.problem.number_of_nodes

        color_used = set()

        for node in range(0, self.problem.number_of_nodes):
            color_find = False
            color = 0
            while not color_find and color < self.nb_color:
                if x[self.problem.number_of_nodes * color + node] == 1:
                    colors[node] = color
                    color_find = True
                    color_used.add(color)
                color += 1

        sol = ColoringSolution(self.problem, colors=colors, nb_color=len(color_used))

        return sol


class FeasibleNbColorQaoaColoringSolver(ColoringSolver, QiskitQaoaSolver):
    def __init__(
        self,
        problem: ColoringProblem,
        nb_color=None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.coloring_qiskit = FeasibleNbColorColoringQiskit(problem, nb_color=nb_color)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)


class FeasibleNbColorVqeColoringSolver(ColoringSolver, QiskitVqeSolver):
    def __init__(
        self,
        problem: ColoringProblem,
        nb_color=None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.coloring_qiskit = FeasibleNbColorColoringQiskit(problem, nb_color=nb_color)

    def init_model(self, **kwargs):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)
