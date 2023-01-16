#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
import random
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

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


class SolutionTSP(Solution):
    permutation_from0: List[int]
    start_index: int
    end_index: int
    permutation: List[int]
    lengths: Optional[
        List[float]
    ]  # to store the details of length of the tsp if you want.
    length: Optional[
        float
    ]  # to store the length of the tsp, in case your mutation computes it :)

    def __init__(
        self,
        problem: "TSPModel",
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        permutation: Optional[List[int]] = None,
        lengths: Optional[
            List[float]
        ] = None,  # to store the details of length of the tsp if you want.
        length: Optional[
            float
        ] = None,  # to store the length of the tsp, in case your mutation computes it :)
        permutation_from0: Optional[List[int]] = None,
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

    def copy(self) -> "SolutionTSP":
        if self.lengths is None:
            lengths = None
        else:
            lengths = list(self.lengths)
        return SolutionTSP(
            problem=self.problem,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=list(self.permutation),
            lengths=lengths,
            length=self.length,
            permutation_from0=list(self.permutation_from0),
        )

    def lazy_copy(self) -> "SolutionTSP":
        return SolutionTSP(
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
        if not isinstance(new_problem, TSPModel):
            raise ValueError("new_problem must a TSPModel for a TSPSolution.")
        self.problem = new_problem
        self.permutation = list(self.permutation)
        if self.lengths is not None:
            self.lengths = list(self.lengths)
        self.permutation_from0 = list(self.permutation_from0)


class Point:
    ...


class TSPModel(Problem):
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
    def evaluate_function(self, var_tsp: SolutionTSP) -> Tuple[Iterable[float], float]:
        ...

    @abstractmethod
    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        ...

    def evaluate_from_encoding(
        self, int_vector: Iterable[int], encoding_name: str
    ) -> Dict[str, float]:
        if encoding_name == "permutation_from0":
            tsp_sol = SolutionTSP(
                problem=self,
                start_index=self.start_index,
                end_index=self.end_index,
                permutation=self.convert_perm_from0_to_original_perm(int_vector),
            )
        elif encoding_name == "permutation":
            tsp_sol = SolutionTSP(
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
            tsp_sol = SolutionTSP(problem=self, **kwargs)  # type: ignore
        objectives = self.evaluate(tsp_sol)
        return objectives

    def evaluate(self, var_tsp: SolutionTSP) -> Dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        lengths, obj = self.evaluate_function(var_tsp)
        var_tsp.length = obj
        var_tsp.lengths = list(lengths)
        return {"length": obj}

    def satisfy(self, var_tsp: SolutionTSP) -> bool:  # type: ignore # avoid isinstance checks for efficiency
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

    def get_dummy_solution(self) -> SolutionTSP:
        var = SolutionTSP(
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

    def get_random_dummy_solution(self) -> SolutionTSP:
        a = list(self.ind_in_permutation)
        random.shuffle(a)
        var = SolutionTSP(
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
    ) -> List[int]:
        perm = [self.original_indices_to_permutation_indices[x] for x in perm_from0]
        return perm

    def convert_original_perm_to_perm_from0(self, perm: Iterable[int]) -> List[int]:
        perm_from0 = [
            self.original_indices_to_permutation_indices_dict[i] for i in perm
        ]
        return perm_from0

    def get_solution_type(self) -> Type[Solution]:
        return SolutionTSP

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


class TSPModel2D(TSPModel):
    list_points: Sequence[Point2D]

    def __init__(
        self,
        list_points: Sequence[Point2D],
        node_count: int,
        start_index: int = 0,
        end_index: int = 0,
        use_numba: bool = True,
    ):
        TSPModel.__init__(
            self, list_points, node_count, start_index=start_index, end_index=end_index
        )
        self.np_points = np.zeros((node_count, 2))
        for i in range(self.node_count):
            self.np_points[i, 0] = self.list_points[i].x
            self.np_points[i, 1] = self.list_points[i].y
        self.evaluate_function_2d: Callable[[List[int]], Tuple[Iterable[float], float]]
        if use_numba:
            self.evaluate_function_2d = build_evaluate_function_np(self)
        else:
            self.evaluate_function_2d = build_evaluate_function(self)

    def evaluate_function(self, var_tsp: SolutionTSP) -> Tuple[Iterable[float], float]:
        return self.evaluate_function_2d(var_tsp.permutation)

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return length(self.list_points[index_1], self.list_points[index_2])


class TSPModelDistanceMatrix(TSPModel):
    def __init__(
        self,
        list_points: Sequence[Point],
        distance_matrix: np.ndarray,
        node_count: int,
        start_index: int = 0,
        end_index: int = 0,
        use_numba: bool = True,
    ):
        TSPModel.__init__(
            self,
            list_points=list_points,
            node_count=node_count,
            start_index=start_index,
            end_index=end_index,
        )
        self.distance_matrix = distance_matrix
        self.evaluate_function_2d = build_evaluate_function_matrix(self)

    def evaluate_function(self, var_tsp: SolutionTSP) -> Tuple[List[int], int]:
        return self.evaluate_function_2d(var_tsp.permutation)

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return int(self.distance_matrix[index_1, index_2])


def length(point1: Point2D, point2: Point2D) -> float:
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def compute_length(
    solution: List[int],
    start_index: int,
    end_index: int,
    list_points: Sequence[Point2D],
    node_count: int,
    length_permutation: int,
) -> Tuple[List[float], float]:
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
    solution: List[int],
    start_index: int,
    end_index: int,
    np_points: np.ndarray,
    node_count: int,
    length_permutation: int,
) -> Tuple[npt.NDArray[np.float_], float]:
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
    solution: Union[List[int], np.ndarray],
    start_index: int,
    end_index: int,
    distance_matrix: np.ndarray,
    node_count: int,
    length_permutation: int,
) -> Tuple[List[int], int]:
    obj = int(distance_matrix[start_index, solution[0]])
    lengths = np.zeros(node_count, dtype=int)
    lengths[0] = obj
    for index in range(0, length_permutation - 1):
        ll = int(distance_matrix[solution[index], solution[index + 1]])
        obj += ll
        lengths[index + 1] = ll
    lengths[node_count - 1] = int(distance_matrix[solution[-1], end_index])
    obj += lengths[node_count - 1]
    return list(lengths), obj


def build_evaluate_function(
    tsp_model: TSPModel,
) -> Callable[[List[int]], Tuple[List[float], float]]:
    return partial(
        compute_length,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        length_permutation=tsp_model.length_permutation,
        list_points=tsp_model.list_points,
        node_count=tsp_model.node_count,
    )


def build_evaluate_function_np(
    tsp_model: TSPModel,
) -> Callable[[List[int]], Tuple[npt.NDArray[np.float_], float]]:
    return partial(
        compute_length_np,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        length_permutation=tsp_model.length_permutation,
        np_points=tsp_model.np_points,
        node_count=tsp_model.node_count,
    )


def build_evaluate_function_matrix(
    tsp_model: TSPModelDistanceMatrix,
) -> Callable[[List[int]], Tuple[List[int], int]]:
    return partial(
        compute_length_matrix,
        start_index=tsp_model.start_index,
        end_index=tsp_model.end_index,
        distance_matrix=tsp_model.distance_matrix,
        node_count=tsp_model.node_count,
        length_permutation=tsp_model.length_permutation,
    )
