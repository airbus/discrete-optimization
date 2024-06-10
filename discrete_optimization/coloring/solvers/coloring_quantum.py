from typing import Optional, Union

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication

from discrete_optimization.coloring.coloring_model import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.generic_tools.qiskit_tools import QiskitQAOASolver, QiskitVQESolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, Solution


class ColoringQiskit(OptimizationApplication):

    def __init__(self, problem: ColoringProblem, nb_max_color=None) -> None:
        """
        Args:
            problem : the coloring problem instance
        """
        super().__init__(problem)
        self.problem = problem
        if nb_max_color is None:
            nb_max_color = self.problem.number_of_nodes
        self.nb_max_color = nb_max_color
        self.nb_variable = self.problem.number_of_nodes * self.nb_max_color + self.nb_max_color

    def to_quadratic_program(self) -> QuadraticProgram:
        quadratic_program = QuadraticProgram()

        # TODO supprimer les X_i_j et se ramener à un problème "peut on colorer nos noeuds avec n nolors?" ??
        # TODO faire les deux ??
        # serait plus adapté pour qaoa ??

        # X_i,j == 1 si le noeud i prend la couleur j
        # C_j == 1 si la couleur j est choisit au moins une fois

        var_names = {}
        for i in range(0, self.nb_max_color):
            for j in range(0, self.problem.number_of_nodes):
                x_new = quadratic_program.binary_var("x" + str(j) + str(i))
                var_names[(j, i)] = x_new.name
            color_new = quadratic_program.binary_var("color" + str(i))
            var_names[i] = color_new.name

        # on cherche à minimiser le nombre de couleurs utilisées

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

        p = self.nb_max_color

        # si une couleur j est attribué à un noeud, la contrainte C_j doit valoir 1
        for i in range(0, self.problem.number_of_nodes):
            for j in range(0, self.nb_max_color):
                # quadratic[var_names[(i, j)], var_names[(i, j)]] = p
                quadratic[var_names[(i, j)], var_names[j]] = -p

        # chaque noeud doit avoir une unique couleur
        for i in range(0, self.problem.number_of_nodes):
            for j in range(0, self.nb_max_color):
                # quadratic[var_names[(i, j)], var_names[(i, j)]] = -p
                for k in range(j + 1, self.nb_max_color):
                    quadratic[var_names[(i, j)], var_names[(i, k)]] = p

        # deux noeuds adjacents ne peuvent avoir la même couleur
        for edge in self.problem.graph.graph_nx.edges():
            for j in range(0, self.nb_max_color):
                quadratic[var_names[(self.problem.index_nodes_name[edge[0]], j)], var_names[
                    (self.problem.index_nodes_name[edge[1]], j)]] = p

        quadratic_program.minimize(constant, linear, quadratic)

        return quadratic_program

    def interpret(self, result: Union[OptimizationResult, np.ndarray]):

        x = self._result_to_x(result)

        colors = [0] * self.problem.number_of_nodes
        nb_color = 0

        for node in range(0, self.problem.number_of_nodes):
            color_find = False
            color = 0
            while not color_find and color < self.nb_max_color:
                if x[self.problem.number_of_nodes * color + node + color] == 1:
                    colors[node] = color
                    color_find = True
                color += 1

            # TODO think about what we want to do when a node has no color

        for color in range(0, self.nb_max_color):
            if x[self.problem.number_of_nodes * color + self.problem.number_of_nodes + color] == 1:
                nb_color += 1

        sol = ColoringSolution(self.problem, colors=colors, nb_color=nb_color)

        return sol


class QAOAColoringSolver(SolverColoring, QiskitQAOASolver):

    def __init__(self, problem: ColoringProblem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 nb_max_color=None):
        super().__init__(problem, params_objective_function)
        self.coloring_qiskit = ColoringQiskit(problem, nb_max_color=nb_max_color)

    def init_model(self):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)


class VQEColoringSolver(SolverColoring, QiskitVQESolver):

    def __init__(self, problem: ColoringProblem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 nb_max_color=None):
        super().__init__(problem, params_objective_function)
        self.coloring_qiskit = ColoringQiskit(problem, nb_max_color=nb_max_color)

    def init_model(self):
        self.quadratic_programm = self.coloring_qiskit.to_quadratic_program()
        self.nb_variable = self.coloring_qiskit.nb_variable

    def retrieve_current_solution(self, result) -> Solution:
        return self.coloring_qiskit.interpret(result)
