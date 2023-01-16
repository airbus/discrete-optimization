"""Implementation of the famous graph coloring problem (https://en.wikipedia.org/wiki/Graph_coloring).

Graph coloring problem consists at assigning different colors to vertexes in a graph.
The only constraint is that adjacent vertices should be colored by different colors
"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Type, Union

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph


class ColoringSolution(Solution):
    """Solution class for graph coloring problem.

    The object contains a pointer to the problem instance,
    a list or numpy array giving colors values to each vertices of the graph.
    number of different colors or violation can also be stored to avoid some unnecessary computation
    (think about local search algorithms).

    Attributes:
        problem (ColoringProblem): instance of the graph coloring problem
        colors (Union[List[int], np.array]): list/array of the same size of problem.number_of_nodes number.
                                             It should contain integer values representing discrete colors.
        nb_color (Optional[int]): number of different colors present in colors attributes. can be used directly
                                  by the problem.evaluate function if this attribute is provided
                                  (be careful to keep coherence between colors and nb_color !)

    """

    def __init__(
        self,
        problem: Problem,
        colors: Optional[Union[List[int], np.ndarray]] = None,
        nb_color: Optional[int] = None,
        nb_violations: Optional[int] = None,
    ):
        """Init of ColoringSolution."""
        self.problem = problem
        self.colors = colors
        self.nb_color = nb_color
        self.nb_violations = nb_violations

    def copy(self) -> "ColoringSolution":
        """Efficient way of copying a coloring solution without deepcopying unnecessary attribute (problem).

        Algorithms can therefore copy this object, modify mutable attributes of interest (i.e colors)
        without modifying the initial object.

        Returns: A copy that can be considered as a deep copy of the current object.

        """
        colors: Optional[Union[List[int], np.ndarray]]
        if self.colors is None:
            colors = None
        elif isinstance(self.colors, np.ndarray):
            colors = np.array(self.colors)
        else:
            colors = list(self.colors)
        return ColoringSolution(
            problem=self.problem,
            colors=colors,
            nb_color=self.nb_color,
            nb_violations=self.nb_violations,
        )

    def lazy_copy(self) -> "ColoringSolution":
        """Shallow copy of the coloring solution object.

        Examples:
            x = y.lozy_copy()
            id(x) == id(y)  # will be false
            x.colors==y.colors # will be true
        Returns: a new ColoringSolution object where the mutable attributes will point to the same object
                 than the original.

        """
        return ColoringSolution(
            problem=self.problem,
            colors=self.colors,
            nb_color=self.nb_color,
            nb_violations=self.nb_violations,
        )

    def to_reformated_solution(self) -> "ColoringSolution":
        """Computes a new solution where the colors array has a value precede chain (see more details https://www.minizinc.org/doc-2.5.3/en/lib-globals.html#index-62) property.

         Examples : [1, 4, 4, 3, 2, 4] doesnt respect the value_precede_chain because 4 appears before 2 and 3.
         A transformed vector would be : [1, 2, 2, 3, 4, 2].
         For the coloring problem it is the same solution but this time we respect value_precede.

        The resulting solution is equivalent for the optimization problem to solve, but can reduce a lot the search space
        Returns: New ColoringSolution object with value precede chain property.

        """
        colors: List[int]
        if self.colors is None:
            raise ValueError(
                "self.colors should not be None when calling to_reformated_solution()"
            )
        elif isinstance(self.colors, np.ndarray):
            colors = self.colors.tolist()
        else:
            colors = self.colors
        sol = ColoringSolution(
            problem=self.problem,
            colors=transform_color_values_to_value_precede(colors),
            nb_color=self.nb_color,
            nb_violations=self.nb_violations,
        )
        return sol

    def __str__(self) -> str:
        return (
            "nb_color = "
            + str(self.nb_color)
            + "\n"
            + "nb_violations="
            + str(self.nb_violations)
            + "\n"
            + "colors="
            + str(self.colors)
        )

    def change_problem(self, new_problem: Problem) -> None:
        """Change the reference to the problem instance of the solution.

        If two coloring problems have the same number of nodes, we can build a solution of the problem
        from the solution of another one.

        Args:
            new_problem: One another ColoringProblem.

        Returns: None, but change in place the object

        """
        if not isinstance(new_problem, ColoringProblem):
            raise ValueError(
                "new_problem must a ColoringProblem for a ColoringSolution."
            )
        colors: Optional[Union[List[int], np.ndarray]]
        if self.colors is None:
            self.colors = None
        elif isinstance(self.colors, np.ndarray):
            self.colors = np.array(self.colors)
        else:
            self.colors = list(self.colors)
        self.problem = new_problem


def transform_color_values_to_value_precede(color_vector: List[int]) -> List[int]:
    """See method ColoringSolution.to_reformated_solution().

    Args:
        color_vector (List[int]): vector representing colors of vertices

    Returns: A vector with value precede chain property

    """
    index_value_color = {}
    new_value = 0
    new_colors_vector = []
    for k in range(len(color_vector)):
        if color_vector[k] not in index_value_color:
            index_value_color[color_vector[k]] = new_value
            new_value += 1
        new_colors_vector += [index_value_color[color_vector[k]]]
    return new_colors_vector


class ColoringProblem(Problem):
    """Coloring problem class implementation.

    Attributes:
        graph (Graph): a graph object representing vertices and edges.
        number_of_nodes (int): number of nodes in the graph
        nodes_name (List[Hashable]): list of id of the graph vertices.
        index_nodes_name (Dict[Hashable, int]): dictionnary node_name->index
        index_to_nodes_name (Dict[int, Hashable]): index->node_name

    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.number_of_nodes = len(self.graph.nodes_infos_dict)
        self.nodes_name = self.graph.nodes_name
        self.index_nodes_name = {
            self.nodes_name[i]: i for i in range(self.number_of_nodes)
        }
        self.index_to_nodes_name = {
            i: self.nodes_name[i] for i in range(self.number_of_nodes)
        }

    def evaluate(self, variable: ColoringSolution) -> Dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        """Evaluation implementation for ColoringProblem.

        Compute number of colors and violation of the current solution.
        """
        if variable.nb_color is None:
            if variable.colors is None:
                raise ValueError(
                    "variable.colors must not be None if variable.nb_color is None."
                )
            else:
                variable.nb_color = len(set(variable.colors))
        if variable.nb_violations is None:
            if variable.colors is None:
                raise ValueError(
                    "variable.colors must not be None if variable.nb_color is None."
                )
            else:
                variable.nb_violations = self.count_violations(variable)
        return {"nb_colors": variable.nb_color, "nb_violations": variable.nb_violations}

    def satisfy(self, variable: ColoringSolution) -> bool:  # type: ignore  # avoid isinstance checks for efficiency
        """Check the color constraint of the solution.

        Check for each edges in the graph if the allocated color of the vertices are different.
        When one counterexample is found, the function directly returns False.

        Args:
            variable (ColoringSolution): the solution object we want to check the feasibility

        Returns: boolean indicating if the solution fulfills the constraint.
        """
        if len(self.graph.edges) > 0:
            if variable.colors is None:
                raise ValueError("variable.colors must not be None")
            for e in self.graph.edges:
                if (
                    variable.colors[self.index_nodes_name[e[0]]]
                    == variable.colors[self.index_nodes_name[e[1]]]
                ):
                    return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        """Attribute documenation for ColoringSolution object.

        Returns: an EncodingRegister specifying the colors attribute.

        """
        dict_register = {
            "colors": {
                "name": "colors",
                "type": [TypeAttribute.LIST_INTEGER],
                "n": self.number_of_nodes,
                "arity": self.number_of_nodes,
                "low": 1,  # integer
                "up": self.number_of_nodes,  # integer
            }
        }
        return EncodingRegister(dict_register)

    def get_solution_type(self) -> Type[Solution]:
        """Returns the class of a solution instance for ColoringProblem."""
        return ColoringSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Specifies the default objective settings to be used with the evaluate function output."""
        dict_objective = {
            "nb_colors": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=-1.0
            ),
            "nb_violations": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def get_dummy_solution(self) -> ColoringSolution:
        """Returns a dummy solution.

        A dummy feasible solution consists in giving one different color per vertices.
        Returns: A feasible and dummiest ColoringSolution

        """
        colors = list(range(self.number_of_nodes))
        solution = ColoringSolution(self, colors)
        self.evaluate(solution)
        return solution

    def count_violations(self, variable: ColoringSolution) -> int:
        """Count number of violation in graph coloring solution.

        Args:
            variable: ColoringSolution to evaluate the number of violation to the color constraint

        Returns: an integer representing the number of edges (link between 2 vertices) where the colors are equal (thus being a violation)

        """
        val = 0
        if len(self.graph.edges) > 0:
            if variable.colors is None:
                raise ValueError("variable.colors must not be None.")
            for e in self.graph.edges:
                if (
                    variable.colors[self.index_nodes_name[e[0]]]
                    == variable.colors[self.index_nodes_name[e[1]]]
                ):
                    val += 1
        return val

    def evaluate_from_encoding(
        self, int_vector: List[int], encoding_name: str
    ) -> Dict[str, float]:
        """Can be used in GA algorithm to build an object solution and evaluate from a int_vector representation.

        Args:
            int_vector: representing the colors vector of our problem
            encoding_name: name of the attribute in ColoringSolution corresponding to the int_vector given.
             In our case, will only work for encoding_name="colors"
        Returns: the evaluation of the (int_vector, encoding) object on the coloring problem.

        """
        coloring_sol: ColoringSolution
        if encoding_name == "colors":
            coloring_sol = ColoringSolution(problem=self, colors=int_vector)
        elif encoding_name == "custom":
            kwargs = {encoding_name: int_vector}
            coloring_sol = ColoringSolution(problem=self, **kwargs)  # type: ignore
        else:
            raise ValueError("encoding_name can only be 'colors' or 'custom'.")
        objectives = self.evaluate(coloring_sol)
        return objectives
