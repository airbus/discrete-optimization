#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
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


class VrpSolution(Solution):
    def copy(self) -> "VrpSolution":
        return VrpSolution(
            problem=self.problem,
            list_start_index=self.list_start_index,
            list_end_index=self.list_end_index,
            list_paths=deepcopy(self.list_paths),
            lengths=deepcopy(self.lengths),
            length=self.length,
            capacities=deepcopy(self.capacities),
        )

    def lazy_copy(self) -> "VrpSolution":
        return VrpSolution(
            problem=self.problem,
            list_start_index=self.list_start_index,
            list_end_index=self.list_end_index,
            list_paths=self.list_paths,
            lengths=self.lengths,
            length=self.length,
            capacities=self.capacities,
        )

    def __str__(self) -> str:
        return "\n".join([str(self.list_paths[i]) for i in range(len(self.list_paths))])

    def __init__(
        self,
        problem: "VrpProblem",
        list_start_index: List[int],
        list_end_index: List[int],
        list_paths: List[List[int]],
        capacities: Optional[List[float]] = None,
        length: Optional[float] = None,
        lengths: Optional[List[List[float]]] = None,
    ):
        self.problem = problem
        self.list_start_index = list_start_index
        self.list_end_index = list_end_index
        self.list_paths = list_paths
        self.length = length
        self.lengths = lengths
        self.capacities = capacities

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, VrpProblem):
            raise ValueError("new_problem must a VrpProblem for a VrpSolution.")
        self.problem = new_problem
        self.list_paths = deepcopy(self.list_paths)
        self.lengths = deepcopy(self.lengths)
        self.capacities = deepcopy(self.capacities)


class BasicCustomer:
    def __init__(self, name: Union[str, int], demand: float):
        self.name = name
        self.demand = demand


class VrpProblem(Problem):
    customers: Sequence[BasicCustomer]

    def __init__(
        self,
        vehicle_count: int,
        vehicle_capacities: List[float],
        customer_count: int,
        customers: Sequence[BasicCustomer],
        start_indexes: List[int],
        end_indexes: List[int],
    ):
        self.vehicle_count = vehicle_count
        self.vehicle_capacities = vehicle_capacities
        self.customer_count = customer_count
        self.customers = customers
        self.start_indexes = (
            start_indexes  # for vehicle i : indicate what is the start index
        )
        self.end_indexes = end_indexes  # for vehicle i : indicate what is the end index

    # for a given tsp kind of problem, you should provide a custom evaluate function, for now still abstract.
    @abstractmethod
    def evaluate_function(
        self, var_tsp: VrpSolution
    ) -> Tuple[List[List[float]], List[float], float, List[float]]:
        ...

    @abstractmethod
    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        ...

    def evaluate(self, variable: VrpSolution) -> Dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        if (
            variable.lengths is None
            or variable.length is None
            or variable.capacities is None
        ):
            lengths, obj_list, obj, capacity_list = self.evaluate_function(variable)
            variable.length = obj
            variable.lengths = lengths
            variable.capacities = capacity_list
        violation = 0.0
        for i in range(self.vehicle_count):
            violation += max(variable.capacities[i] - self.vehicle_capacities[i], 0)
        return {"length": variable.length, "capacity_violation": violation}

    def satisfy(self, variable: VrpSolution) -> bool:  # type: ignore # avoid isinstance checks for efficiency
        d = self.evaluate(variable)
        return d["capacity_violation"] == 0

    def get_attribute_register(self) -> EncodingRegister:
        dict_encoding = {
            "list_paths": {"name": "list_paths", "type": [TypeAttribute.VRP_PATHS]}
        }
        return EncodingRegister(dict_encoding)

    def get_solution_type(self) -> Type[Solution]:
        return VrpSolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "length": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "capacity_violation": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def __str__(self) -> str:
        s = (
            "Vrp problem with \n"
            + str(self.customer_count)
            + " customers \nand "
            + str(self.vehicle_count)
            + " vehicles "
        )
        return s

    def get_dummy_solution(self) -> VrpSolution:
        s, fit = trivial_solution(self)
        return s

    def get_stupid_solution(self) -> VrpSolution:
        s, fit = stupid_solution(self)
        return s


class Customer2D(BasicCustomer):
    def __init__(self, name: Union[str, int], demand: float, x: float, y: float):
        super().__init__(name=name, demand=demand)
        self.x = x
        self.y = y


def length(point1: Customer2D, point2: Customer2D) -> float:
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class VrpProblem2D(VrpProblem):
    customers: Sequence[Customer2D]

    def __init__(
        self,
        vehicle_count: int,
        vehicle_capacities: List[float],
        customer_count: int,
        customers: Sequence[Customer2D],
        start_indexes: List[int],
        end_indexes: List[int],
    ):
        super().__init__(
            vehicle_count=vehicle_count,
            vehicle_capacities=vehicle_capacities,
            customer_count=customer_count,
            customers=customers,
            start_indexes=start_indexes,
            end_indexes=end_indexes,
        )
        self.evaluate_function_2d = build_evaluate_function(self)

    def evaluate_function(
        self, vrp_sol: VrpSolution
    ) -> Tuple[List[List[float]], List[float], float, List[float]]:
        return self.evaluate_function_2d(vrp_sol)

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return length(self.customers[index_1], self.customers[index_2])


def trivial_solution(vrp_model: VrpProblem) -> Tuple[VrpSolution, Dict[str, float]]:
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours: List[List[int]] = []
    customers = range(vrp_model.customer_count)
    nb_vehicles = vrp_model.vehicle_count
    nb_customers = vrp_model.customer_count
    remaining_capacity_vehicle = {
        v: vrp_model.vehicle_capacities[v] for v in range(nb_vehicles)
    }
    remaining_customers = set(customers)
    for v in range(nb_vehicles):
        start = vrp_model.start_indexes[v]
        end = vrp_model.end_indexes[v]
        remaining_capacity_vehicle[v] -= vrp_model.customers[start].demand
        if end != start:
            remaining_capacity_vehicle[v] -= vrp_model.customers[end].demand
        if start in remaining_customers:
            remaining_customers.remove(start)
        if end in remaining_customers:
            remaining_customers.remove(end)
    for v in range(nb_vehicles):
        vehicle_tours.append([])
        cur_node = vrp_model.start_indexes[v]
        while (
            sum(
                [
                    remaining_capacity_vehicle[v]
                    >= vrp_model.customers[customer].demand
                    for customer in remaining_customers
                ]
            )
            > 0
        ):
            used = set()
            order = sorted(
                remaining_customers,
                key=lambda x: -vrp_model.customers[x].demand * nb_customers + x,
            )
            order = sorted(
                remaining_customers,
                key=lambda x: vrp_model.evaluate_function_indexes(cur_node, x),
            )
            for customer in order:
                if (
                    remaining_capacity_vehicle[v]
                    >= vrp_model.customers[customer].demand
                ):
                    remaining_capacity_vehicle[v] -= vrp_model.customers[
                        customer
                    ].demand
                    vehicle_tours[v].append(customer)
                    cur_node = customer
                    used.add(customer)
            remaining_customers -= used
    solution = VrpSolution(
        problem=vrp_model,
        list_start_index=vrp_model.start_indexes,
        list_end_index=vrp_model.end_indexes,
        list_paths=vehicle_tours,
        length=None,
        lengths=None,
        capacities=None,
    )
    fit = vrp_model.evaluate(solution)
    return solution, fit


def stupid_solution(vrp_model: VrpProblem) -> Tuple[VrpSolution, Dict[str, float]]:
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours: List[List[int]] = []
    customers = range(vrp_model.customer_count)
    nb_vehicles = vrp_model.vehicle_count
    remaining_capacity_vehicle = {
        v: vrp_model.vehicle_capacities[v] for v in range(nb_vehicles)
    }
    remaining_customers = set(customers)
    for v in range(nb_vehicles):
        start = vrp_model.start_indexes[v]
        end = vrp_model.end_indexes[v]
        remaining_capacity_vehicle[v] -= vrp_model.customers[start].demand
        if end != start:
            remaining_capacity_vehicle[v] -= vrp_model.customers[end].demand
        if start in remaining_customers:
            remaining_customers.remove(start)
        if end in remaining_customers:
            remaining_customers.remove(end)
    for v in range(nb_vehicles):
        vehicle_tours.append([])
    vehicle_tours[0] = list(sorted(remaining_customers))
    solution = VrpSolution(
        problem=vrp_model,
        list_start_index=vrp_model.start_indexes,
        list_end_index=vrp_model.end_indexes,
        list_paths=vehicle_tours,
        length=None,
        lengths=None,
    )
    fit = vrp_model.evaluate(solution)
    return solution, fit


def compute_length(
    start_index: int,
    end_index: int,
    solution: List[int],
    list_customers: Sequence[BasicCustomer],
    method: Callable[[int, int], float],
) -> Tuple[List[float], float, float]:
    if len(solution) > 0:
        obj = method(start_index, solution[0])
        lengths = [obj]
        capacity = list_customers[start_index].demand
        capacity += list_customers[solution[0]].demand
        for index in range(0, len(solution) - 1):
            ll = method(solution[index], solution[index + 1])
            obj += ll
            lengths += [ll]
            capacity += list_customers[solution[index + 1]].demand
        lengths += [method(end_index, solution[-1])]
        if end_index != start_index:
            capacity += list_customers[end_index].demand
        obj += lengths[-1]
    else:
        obj = method(start_index, end_index)
        lengths = [obj]
        capacity = list_customers[start_index].demand
        if end_index != start_index:
            capacity += list_customers[end_index].demand
    return lengths, obj, capacity


# More efficient implementation
@njit
def compute_length_np(
    start_index: int,
    end_index: int,
    solution: Union[List[int], np.ndarray],
    np_points: np.ndarray,
) -> Tuple[Union[List[float], np.ndarray], float]:
    obj = np.sqrt(
        (np_points[start_index, 0] - np_points[solution[0], 0]) ** 2
        + (np_points[start_index, 1] - np_points[solution[0], 1]) ** 2
    )
    len_sol = len(solution)
    lengths = np.zeros((len_sol + 1))
    lengths[0] = obj
    for index in range(0, len_sol - 1):
        ll = math.sqrt(
            (np_points[solution[index], 0] - np_points[solution[index + 1], 0]) ** 2
            + (np_points[solution[index], 1] - np_points[solution[index + 1], 1]) ** 2
        )
        obj += ll
        lengths[index + 1] = ll
    lengths[len_sol] = np.sqrt(
        (np_points[end_index, 0] - np_points[solution[-1], 0]) ** 2
        + (np_points[end_index, 1] - np_points[solution[-1], 1]) ** 2
    )
    obj += lengths[len_sol]
    return lengths, obj


def sequential_computing(
    vrp_sol: VrpSolution, vrp_model: VrpProblem
) -> Tuple[List[List[float]], List[float], float, List[float]]:
    lengths_list: List[List[float]] = []
    obj_list: List[float] = []
    capacity_list: List[float] = []
    sum_obj = 0.0
    for i in range(len(vrp_sol.list_paths)):
        lengths, obj, capacity = compute_length(
            start_index=vrp_sol.list_start_index[i],
            end_index=vrp_sol.list_end_index[i],
            solution=vrp_sol.list_paths[i],
            list_customers=vrp_model.customers,
            method=vrp_model.evaluate_function_indexes,
        )
        lengths_list += [lengths]
        obj_list += [obj]
        capacity_list += [capacity]
        sum_obj += obj
    return lengths_list, obj_list, sum_obj, capacity_list


def build_evaluate_function(
    vrp_model: VrpProblem,
) -> Callable[[VrpSolution], Tuple[List[List[float]], List[float], float, List[float]]]:
    return partial(sequential_computing, vrp_model=vrp_model)
