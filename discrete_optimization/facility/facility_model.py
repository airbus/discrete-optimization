"""Implementation of the facility location problem with capacity constraint
(https://en.wikipedia.org/wiki/Facility_location_problem#Capacitated_facility_location)

Facility location problem consist in choosing where to locate facilities and allocate customers to those. Each customers
have a demand and facility have a capacity. The sum of demand of customers allocated to a given location can't exceed the
capacity of the facility.
"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math
from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

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


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Facility:
    index: int
    setup_cost: float
    capacity: float
    location: Point


@dataclass(frozen=True)
class Customer:
    index: int
    demand: float
    location: Point


class FacilitySolution(Solution):
    """Solution object for facility location

    Attributes:
        problem (FacilityProblem): facility problem instance
        facility_for_customers (List[int]): for each customers, specify the index of the facility
        dict_details (Dict): giving more metrics of the solution such as the capacities used, the setup cost etc.
        See problem.evaluate(sol) implementation for FacilityProblem

    """

    def __init__(
        self,
        problem: Problem,
        facility_for_customers: List[int],
        dict_details: Optional[Dict[str, Any]] = None,
    ):
        self.problem = problem
        self.facility_for_customers = facility_for_customers
        self.dict_details = dict_details

    def copy(self) -> "FacilitySolution":
        return FacilitySolution(
            self.problem,
            facility_for_customers=list(self.facility_for_customers),
            dict_details=deepcopy(self.dict_details),
        )

    def lazy_copy(self) -> "FacilitySolution":
        return FacilitySolution(
            self.problem,
            facility_for_customers=self.facility_for_customers,
            dict_details=self.dict_details,
        )

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, FacilityProblem):
            raise ValueError(
                "new_problem must a FacilityProblem for a FacilitySolution."
            )
        self.problem = new_problem
        self.facility_for_customers = list(self.facility_for_customers)
        self.dict_details = deepcopy(self.dict_details)


class FacilityProblem(Problem):
    """Base class for the facility problem.

    Attributes:
        facility_count (int): number of facilities
        customer_count (int): number of customers
        facilities (List[Facility]): list of facilities object, facility has a setup cost, capacity and location
        customers (List[Customer]): list of customers object, which has demand and location.


    """

    def __init__(
        self,
        facility_count: int,
        customer_count: int,
        facilities: List[Facility],
        customers: List[Customer],
    ):
        self.facility_count = facility_count
        self.customer_count = customer_count
        self.facilities = facilities
        self.customers = customers

    @abstractmethod
    def evaluate_customer_facility(
        self, facility: Facility, customer: Customer
    ) -> float:
        """Compute the cost of allocating a customer to a facility. This function is not implemented by default.

        Args:
            facility (Facility): facility
            customer (Customer): customer

        Returns (float): a cost as a float

        """
        ...

    def evaluate(self, variable: FacilitySolution) -> Dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        """Computes the KPI of a FacilitySolution.

        Args:
            variable (FacilitySolution): solution to evaluate

        Returns: dictionnary of kpi, see get_objective_register() for details of kpi.

        """
        if variable.dict_details is not None:
            return variable.dict_details
        d = self.evaluate_cost(variable)
        capacity_constraint_violation = 0
        for f in d["details"]:
            capacity_constraint_violation = max(
                d["details"][f]["capacity_used"] - self.facilities[f].capacity, 0
            )
        d["capacity_constraint_violation"] = capacity_constraint_violation
        return d

    def evaluate_from_encoding(
        self, int_vector: List[int], encoding_name: str
    ) -> Dict[str, float]:
        """Evaluate function based on direct integer vector (used in GA algorithms only)

        Args:
            int_vector (List[int]): vector encoding the solution
            encoding_name (str): name of encoding (see get_attribute_register) for available encoding

        Returns: kpi of the solution

        """
        kp_sol: FacilitySolution
        if encoding_name == "facility_for_customers":
            kp_sol = FacilitySolution(problem=self, facility_for_customers=int_vector)
        elif encoding_name == "custom":
            kwargs: Dict[str, Any] = {encoding_name: int_vector}
            kp_sol = FacilitySolution(problem=self, **kwargs)
        else:
            raise ValueError(
                f"encoding_name must be 'facility_for_customers' or 'custom'"
            )
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate_cost(self, variable: FacilitySolution) -> Dict[str, Any]:
        """Compute the allocation cost of the solution along with setup cost too.

        Args:
            variable (FacilitySolution): facility solution to evaluate.

        Returns: a dictionnary containing the cost of allocation ("cost"), setup cost ("setup_cost"),
        and details by facilities ("details")

        """
        facility_details = {}
        cost = 0.0
        setup_cost = 0.0
        for i in range(self.customer_count):
            f = variable.facility_for_customers[i]
            if f not in facility_details:
                facility_details[f] = {
                    "capacity_used": 0.0,
                    "customers": set(),
                    "cost": 0.0,
                    "setup_cost": self.facilities[f].setup_cost,
                }
                setup_cost += self.facilities[f].setup_cost
            facility_details[f]["capacity_used"] += self.customers[i].demand  # type: ignore
            facility_details[f]["customers"].add(i)  # type: ignore
            c = self.evaluate_customer_facility(
                facility=self.facilities[f], customer=self.customers[i]
            )
            facility_details[f]["cost"] += c  # type: ignore
            cost += c
        return {"cost": cost, "setup_cost": setup_cost, "details": facility_details}

    def satisfy(self, variable: FacilitySolution) -> bool:  # type: ignore # avoid isinstance checks for efficiency
        """Satisfaction function of a facility solution.

        We only check that the capacity constraint is fulfilled. We admit that the values of the vector are in the
        allowed range.

        Args:
            variable (FacilitySolution): facility solution to check satisfaction

        Returns (bool): true if solution satisfies constraint.

        """
        d = self.evaluate(variable)
        return d["capacity_constraint_violation"] == 0.0

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = dict()
        dict_register["facility_for_customers"] = {
            "name": "facility_for_customers",
            "type": [TypeAttribute.LIST_INTEGER],
            "n": self.customer_count,
            "arity": self.facility_count,
        }
        return EncodingRegister(dict_register)

    def get_dummy_solution(self) -> FacilitySolution:
        """Returns Dummy solution (that is not necessary fulfilling the constraints)

        All customers are allocated to the first facility (which usually will break the capacity constraint)
        Returns (FacilitySolution): a dummy solution

        """
        return FacilitySolution(self, [0] * self.customer_count)

    def get_solution_type(self) -> Type[Solution]:
        return FacilitySolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "setup_cost": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=-1.0
            ),
            "capacity_constraint_violation": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=-10000.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )


def length(point1: Point, point2: Point) -> float:
    """Classic euclidian distance between two points.

    Args:
        point1 (Point): origin point
        point2 (Point): target point

    Returns (float): euclidian distance between the 2 points

    """
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


class FacilityProblem2DPoints(FacilityProblem):
    def evaluate_customer_facility(
        self, facility: Facility, customer: Customer
    ) -> float:
        """Implementation of a distance based cost for allocation of facility to customers.

        It uses the euclidian distance as cost.

        Args:
            facility (Facility): facility
            customer (Customer): customer

        Returns: a float cost

        """
        return length(facility.location, customer.location)
