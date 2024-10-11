#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
import random
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numba import njit

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


class TspSolution(Solution):
    permutation_from0: list[int]
    start_index: int
    end_index: int
    permutation: list[int]
    lengths: Optional[
        list[float]
    ]  # to store the details of length of the tsp if you want.
    length: Optional[
        float
    ]  # to store the length of the tsp, in case your mutation computes it :)

    def __init__(
        self,
        problem: "TspProblem",
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        permutation: Optional[list[int]] = None,
        lengths: Optional[
            list[float]
        ] = None,  # to store the details of length of the tsp if you want.
        length: Optional[
            float
        ] = None,  # to store the length of the tsp, in case your mutation computes it :)
        permutation_from0: Optional[list[int]] = None,
    ):
        if permutation is None and permutation_from0 is None:
            raise ValueError("permutation and permutation_from0 cannot be both None.")
        self.lengths = lengths
        self.length = length
        self.problem = problem
        if start_index is None:
            self.start_index = problem.start_index
        else:
            self.start_index = start_index
        if end_index is None:
            self.end_index = problem.end_index
        else:
            self.end_index = end_index
        # convert perm
        if permutation is None:
            if permutation_from0 is None:
                raise ValueError(
                    "permutation and permutation_from0 cannot be both None."
                )
            self.permutation = self.problem.convert_perm_from0_to_original_perm(
                permutation_from0
            )
            self.permutation_from0 = permutation_from0
        elif permutation_from0 is None:
            self.permutation_from0 = self.problem.convert_original_perm_to_perm_from0(
                permutation
            )
            self.permutation = permutation
        else:
            self.permutation = permutation
            self.permutation_from0 = permutation_from0
        if self.length is None:
            self.problem.evaluate(self)

    def copy(self) -> "TspSolution":
        if self.lengths is None:
            lengths = None
        else:
            lengths = list(self.lengths)
        return TspSolution(
            problem=self.problem,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=list(self.permutation),
            lengths=lengths,
            length=self.length,
            permutation_from0=list(self.permutation_from0),
        )

    def lazy_copy(self) -> "TspSolution":
        return TspSolution(
            problem=self.problem,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=self.permutation,
            lengths=self.lengths,
            length=self.length,
            permutation_from0=self.permutation_from0,
        )

    def __str__(self) -> str:
        return "perm :" + str(self.permutation) + "\nobj=" + str(self.length)

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, TspProblem):
            raise ValueError("new_problem must a TspProblem for a TspSolution.")
        self.problem = new_problem
        self.permutation = list(self.permutation)
        if self.lengths is not None:
            self.lengths = list(self.lengths)
        self.permutation_from0 = list(self.permutation_from0)


class Point:
    ...


class TspProblem(Problem):
    list_points: Sequence[Point]
    np_points: np.ndarray
    node_count: int

    def __init__(
        self,
        list_points: Sequence[Point],
        node_count: int,
        start_index: int = 0,
        end_index: int = 0,
    ):
        self.list_points = list_points
        self.node_count = node_count
        self.start_index = start_index
        self.end_index = end_index
        self.ind_in_permutation = [
            i
            for i in range(self.node_count)
            if i != self.start_index and i != self.end_index
        ]
        self.length_permutation = len(self.ind_in_permutation)
        self.original_indices_to_permutation_indices = [
            i
            for i in range(self.node_count)
            if i != self.start_index and i != self.end_index
        ]
        self.original_indices_to_permutation_indices_dict = {}
        counter = 0
        for i in range(self.node_count):
            if i != self.start_index and i != self.end_index:
                self.original_indices_to_permutation_indices_dict[i] = counter
                counter += 1

    # for a given tsp kind of problem, you should provide a custom evaluate function, for now still abstract.
    @abstractmethod
    def evaluate_function(self, var_tsp: TspSolution) -> tuple[Iterable[float], float]:
        ...

    @abstractmethod
    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        ...

    def evaluate_from_encoding(
        self, int_vector: Iterable[int], encoding_name: str
    ) -> dict[str, float]:
        if encoding_name == "permutation_from0":
            tsp_sol = TspSolution(
                problem=self,
                start_index=self.start_index,
                end_index=self.end_index,
                permutation=self.convert_perm_from0_to_original_perm(int_vector),
            )
        elif encoding_name == "permutation":
            tsp_sol = TspSolution(
                problem=self,
                start_index=self.start_index,
                end_index=self.end_index,
                permutation=list(int_vector),
            )
        else:
            kwargs = {
                encoding_name: int_vector,
                "start_index": self.start_index,
                "end_index": self.end_index,
            }
            tsp_sol = TspSolution(problem=self, **kwargs)  # type: ignore
        objectives = self.evaluate(tsp_sol)
        return objectives

    def evaluate(self, var_tsp: TspSolution) -> dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        if None in var_tsp.permutation:
            return {"length": -1}
        lengths, obj = self.evaluate_function(var_tsp)
        var_tsp.length = obj
        var_tsp.lengths = list(lengths)
        return {"length": obj}

    def satisfy(self, var_tsp: TspSolution) -> bool:  # type: ignore # avoid isinstance checks for efficiency
        if None in var_tsp.permutation:
            return False
        b = (
            var_tsp.start_index == self.start_index
            and var_tsp.end_index == self.end_index
        )
        if not b:
            return False
        if len(var_tsp.permutation) != self.length_permutation:
            return False
        if not all(x in var_tsp.permutation for x in self.ind_in_permutation):
            return False
        return True

    def get_dummy_solution(self) -> TspSolution:
        var = TspSolution(
            problem=self,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=list(self.ind_in_permutation),
            permutation_from0=None,
            lengths=None,
            length=None,
        )
        self.evaluate(var)
        return var

    def get_random_dummy_solution(self) -> TspSolution:
        a = list(self.ind_in_permutation)
        random.shuffle(a)
        var = TspSolution(
            problem=self,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=a,
            permutation_from0=None,
            lengths=None,
            length=None,
        )
        self.evaluate(var)
        return var

    def __str__(self) -> str:
        return "TSP problem with number of nodes :  : " + str(self.node_count)

    def convert_perm_from0_to_original_perm(
        self, perm_from0: Iterable[int]
    ) -> list[int]:
        perm = []
        for i in perm_from0:
            if i is not None:
                perm.append(self.original_indices_to_permutation_indices[i])
            else:
                perm.append(None)
        return perm

    def convert_original_perm_to_perm_from0(self, perm: Iterable[int]) -> list[int]:
        perm_from0 = []
        for i in perm:
            if i is not None:
                perm_from0.append(self.original_indices_to_permutation_indices_dict[i])
            else:
                perm_from0.append(None)
        return perm_from0

    def get_solution_type(self) -> type[Solution]:
        return TspSolution

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
            "permutation_from0": {
                "name": "permutation_from0",
                "type": [TypeAttribute.PERMUTATION],
                "range": range(len(self.original_indices_to_permutation_indices)),
                "n": len(self.original_indices_to_permutation_indices),
            },
            "permutation": {
                "name": "permutation",
                "type": [TypeAttribute.PERMUTATION, TypeAttribute.PERMUTATION_TSP],
                "range": self.ind_in_permutation,
                "n": self.length_permutation,
            },
        }
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "length": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0)
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc=dict_objective,
        )


@dataclass(frozen=True)
class Point2D(Point):
    x: float
    y: float


class Point2DTspProblem(TspProblem):
    list_points: Sequence[Point2D]

    def __init__(
        self,
        list_points: Sequence[Point2D],
        node_count: int,
        start_index: int = 0,
        end_index: int = 0,
        use_numba: bool = True,
    ):
        TspProblem.__init__(
            self, list_points, node_count, start_index=start_index, end_index=end_index
        )
        self.np_points = np.zeros((node_count, 2))
        for i in range(self.node_count):
            self.np_points[i, 0] = self.list_points[i].x
            self.np_points[i, 1] = self.list_points[i].y
        self.evaluate_function_2d: Callable[[list[int]], tuple[Iterable[float], float]]
        if use_numba:
            self.evaluate_function_2d = build_evaluate_function_np(self)
        else:
            self.evaluate_function_2d = build_evaluate_function(self)

    def evaluate_function(self, var_tsp: TspSolution) -> tuple[Iterable[float], float]:
        return self.evaluate_function_2d(var_tsp.permutation)

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return length(self.list_points[index_1], self.list_points[index_2])


class DistanceMatrixTspProblem(TspProblem):
    def __init__(
        self,
        list_points: Sequence[Point],
        distance_matrix: np.ndarray,
        node_count: int,
        start_index: int = 0,
        end_index: int = 0,
        use_numba: bool = True,
    ):
        TspProblem.__init__(
            self,
            list_points=list_points,
            node_count=node_count,
            start_index=start_index,
            end_index=end_index,
        )
        self.distance_matrix = distance_matrix
        self.evaluate_function_2d = build_evaluate_function_matrix(self)

    def evaluate_function(self, var_tsp: TspSolution) -> tuple[list[int], int]:
        return self.evaluate_function_2d(var_tsp.permutation)

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return int(self.distance_matrix[index_1, index_2])


def length(point1: Point2D, point2: Point2D) -> float:
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def compute_length(
    solution: list[int],
    start_index: int,
    end_index: int,
    list_points: Sequence[Point2D],
    node_count: int,
    length_permutation: int,
) -> tuple[list[float], float]:
    obj = length(list_points[start_index], list_points[solution[0]])
    lengths = [obj]
    for index in range(0, length_permutation - 1):
        ll = length(list_points[solution[index]], list_points[solution[index + 1]])
        obj += ll
        lengths += [ll]
    lengths += [length(list_points[end_index], list_points[solution[-1]])]
    obj += lengths[-1]
    return lengths, obj


# More efficient implementation
@njit
def compute_length_np(
    solution: list[int],
    start_index: int,
    end_index: int,
    np_points: np.ndarray,
    node_count: int,
    length_permutation: int,
) -> tuple[npt.NDArray[np.float64], float]:
    obj = np.sqrt(
        (np_points[start_index, 0] - np_points[solution[0], 0]) ** 2
        + (np_points[start_index, 1] - np_points[solution[0], 1]) ** 2
    )
    lengths = np.zeros(node_count)
    lengths[0] = obj
    for index in range(0, length_permutation - 1):
        ll = math.sqrt(
            (np_points[solution[index], 0] - np_points[solution[index + 1], 0]) ** 2
            + (np_points[solution[index], 1] - np_points[solution[index + 1], 1]) ** 2
        )
        obj += ll
        lengths[index + 1] = ll
    lengths[node_count - 1] = np.sqrt(
        (np_points[end_index, 0] - np_points[solution[-1], 0]) ** 2
        + (np_points[end_index, 1] - np_points[solution[-1], 1]) ** 2
    )
    obj += lengths[node_count - 1]
    return lengths, obj


@njit
def compute_length_matrix(
    solution: Union[list[int], np.ndarray],
    start_index: int,
    end_index: int,
    distance_matrix: np.ndarray,
    node_count: int,
    length_permutation: int,
) -> tuple[list[int], int]:
    obj = int(distance_matrix[start_index, solution[0]])
    lengths = np.zeros(node_count, dtype=np.int_)
    lengths[0] = obj
    for index in range(0, length_permutation - 1):
        ll = int(distance_matrix[solution[index], solution[index + 1]])
        obj += ll
        lengths[index + 1] = ll
    lengths[node_count - 1] = int(distance_matrix[solution[-1], end_index])
    obj += lengths[node_count - 1]
    return list(lengths), obj


def build_evaluate_function(
    tsp_model: TspProblem,
) -> Callable[[list[int]], tuple[list[float], float]]:
    return partial(
        compute_length,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        length_permutation=tsp_model.length_permutation,
        list_points=tsp_model.list_points,
        node_count=tsp_model.node_count,
    )


def build_evaluate_function_np(
    tsp_model: TspProblem,
) -> Callable[[list[int]], tuple[npt.NDArray[np.float64], float]]:
    return partial(
        compute_length_np,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        length_permutation=tsp_model.length_permutation,
        np_points=tsp_model.np_points,
        node_count=tsp_model.node_count,
    )


def build_evaluate_function_matrix(
    tsp_model: DistanceMatrixTspProblem,
) -> Callable[[list[int]], tuple[list[int], int]]:
    return partial(
        compute_length_matrix,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        distance_matrix=tsp_model.distance_matrix,
        node_count=tsp_model.node_count,
        length_permutation=tsp_model.length_permutation,
    )
