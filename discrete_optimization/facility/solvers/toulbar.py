#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import tqdm

try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = False
import logging

from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers.facility_solver import FacilitySolver
from discrete_optimization.facility.solvers.lp import prune_search_space
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.toulbar_tools import (
    ToulbarSolver,
    to_lns_toulbar,
)

logger = logging.getLogger(__name__)


class ModelingToulbarFacility(Enum):
    INTEGER = 0
    BINARY = 1


class ToulbarFacilitySolver(ToulbarSolver, FacilitySolver, WarmstartMixin):
    hyperparameters = ToulbarSolver.hyperparameters + [
        EnumHyperparameter(
            name="modeling",
            enum=ModelingToulbarFacility,
            default=ModelingToulbarFacility.INTEGER,
        )
    ]
    modeling: ModelingToulbarFacility
    key_to_index: Optional[dict]
    used_var_to_index: Optional[dict]
    client_var_to_index: Optional[dict]

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == ModelingToulbarFacility.INTEGER:
            self.init_model_integer_variable(**kwargs)
            self.modeling = kwargs["modeling"]
        if kwargs["modeling"] == ModelingToulbarFacility.BINARY:
            self.init_model_boolean_variable(**kwargs)
            self.modeling = kwargs["modeling"]

    def init_model_boolean_variable(self, **kwargs):
        """
        Init a model with boolean variable array, indicating if a factory f is assigned to customer c
        """
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        if "vns" in kwargs:
            model = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8), vns=kwargs["vns"])
        else:
            model = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8))
        x: Dict[Tuple[int, int], Union[int, Any]] = {}
        key_to_index = {}
        index = 0
        matrix_fc_indicator, matrix_length = prune_search_space(
            self.problem, n_cheapest=nb_facilities, n_shortest=nb_facilities
        )
        for f in range(nb_facilities):
            for c in range(nb_customers):
                x[f, c] = model.AddVariable(name=f"x_{(f,c)}", values=[0, 1])
                model.AddFunction([f"x_{(f,c)}"], [0, matrix_length[f, c]])
                key_to_index[(f, c)] = index
                index += 1
        for c in range(nb_customers):
            model.AddSumConstraint(
                [key_to_index[(f, c)] for f in range(nb_facilities)],
                operand="==",
                rightcoef=1,
            )
        used_var = {}
        index_var = index
        used_var_to_index = {}
        for i in range(nb_facilities):
            used_var[i] = model.AddVariable(f"used_{i}", [0, 1])
            used_var_to_index[i] = index_var
            index_var += 1
            model.AddFunction([f"used_{i}"], [0, self.problem.facilities[i].setup_cost])
            for c in range(nb_customers):
                # model.CFN.wcsp.postKnapsackConstraint(
                #     [index_var - 1, key_to_index[(i, c)]], "0 1 1 1 1 1 -1", False, True
                # )
                model.AddLinearConstraint(
                    [1, -1], [f"used_{i}", f"x_{i, c}"], operand=">=", rightcoef=0
                )
                # constraint triggering the used facility binary variable

        # Capacity constraint.
        for i in range(nb_facilities):
            param_c = ""
            for c in range(nb_customers):
                param_c += (
                    " "
                    + str(1)
                    + " "
                    + str(1)
                    + " "
                    + str(-int(self.problem.customers[c].demand))
                )
            model.CFN.wcsp.postKnapsackConstraint(
                [key_to_index[(i, c)] for c in range(nb_customers)],
                str(-self.problem.facilities[i].capacity) + param_c,
                False,
                True,
            )
            # Capacity/Knapsack constraint on facility.
        self.model = model
        self.key_to_index = key_to_index
        self.used_var_to_index = used_var_to_index

    def init_model_integer_variable(self, **kwargs):
        """
        Integer encoding, where main decision variable is an array of integer, giving for each customer what is
        the facility index.
        """
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        if "vns" in kwargs:
            model = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8), vns=kwargs["vns"])
        else:
            model = pytoulbar2.CFN(kwargs.get("upper_bound", 10e8))
        index = 0
        matrix_fc_indicator, matrix_length = prune_search_space(
            self.problem, n_cheapest=nb_facilities, n_shortest=nb_facilities
        )
        client_var_to_index = {}
        for c in tqdm.tqdm(range(nb_customers)):
            model.AddVariable(f"x_{c}", values=range(nb_facilities))
            model.AddFunction(
                [f"x_{c}"], [int(matrix_length[f, c]) for f in range(nb_facilities)]
            )
            client_var_to_index[c] = index
            index += 1
        index_var = index
        used_var_to_index = {}
        for i in tqdm.tqdm(range(nb_facilities)):
            model.AddVariable(f"used_{i}", [0, 1])
            used_var_to_index[i] = index_var
            model.AddFunction([f"used_{i}"], [0, self.problem.facilities[i].setup_cost])
            for c in range(nb_customers):
                model.AddFunction(
                    [f"used_{i}", f"x_{c}"],
                    [
                        10e8 if b == 0 and f == i else 0
                        for b in [0, 1]
                        for f in range(nb_facilities)
                    ],
                )
            index_var += 1
            # Force that when x_{c} == i, used_i = 1
        # capacity constraint on facility.
        for i in tqdm.tqdm(range(nb_facilities)):
            # More low level version of the capacity constraint.
            # params_constraints = ""
            # for c in range(nb_customers):
            #     params_constraints += " "+str(1)+" "+str(i)+" "+str(-int(self.problem.customers[c].demand))
            # Problem.CFN.wcsp.postKnapsackConstraint([client_var_to_index[c] for c in range(nb_customers)],
            #                                         str(int(-self.problem.facilities[i].capacity))+params_constraints,
            #                                         False, True)
            model.AddGeneralizedLinearConstraint(
                [
                    (f"x_{c}", i, int(self.problem.customers[c].demand))
                    for c in range(nb_customers)
                ],
                operand="<=",
                rightcoef=int(self.problem.facilities[i].capacity),
            )
        self.model = model
        self.used_var_to_index = used_var_to_index
        self.client_var_to_index = client_var_to_index

    def retrieve_solution(self, solution) -> FacilitySolution:
        if self.modeling == ModelingToulbarFacility.BINARY:
            return self.retrieve_solution_binary(solution)
        if self.modeling == ModelingToulbarFacility.INTEGER:
            return self.retrieve_solution_integer(solution)

    def retrieve_solution_binary(self, solution) -> FacilitySolution:
        sol = FacilitySolution(
            problem=self.problem,
            facility_for_customers=[None] * self.problem.customer_count,
        )
        for x in self.key_to_index:
            index = self.key_to_index[x]
            if solution[0][index] == 1:
                sol.facility_for_customers[x[1]] = x[0]
        return sol

    def retrieve_solution_integer(self, solution) -> FacilitySolution:
        return FacilitySolution(
            problem=self.problem,
            facility_for_customers=solution[0][: self.problem.customer_count],
        )

    def set_warm_start(self, solution: FacilitySolution) -> None:
        if self.modeling == ModelingToulbarFacility.INTEGER:
            self.set_warm_start_integer(solution)
        if self.modeling == ModelingToulbarFacility.BINARY:
            self.set_warm_start_binary(solution)

    def set_warm_start_binary(self, solution: FacilitySolution) -> None:
        for f, c in self.key_to_index:
            if solution.facility_for_customers[c] == f:
                self.model.CFN.wcsp.setBestValue(self.key_to_index[f, c], 1)
            else:
                self.model.CFN.wcsp.setBestValue(self.key_to_index[f, c], 0)
        facility_used = set(solution.facility_for_customers)
        for i in self.used_var_to_index:
            if i in facility_used:
                self.model.CFN.wcsp.setBestValue(self.used_var_to_index[i], 1)
            else:
                self.model.CFN.wcsp.setBestValue(self.used_var_to_index[i], 0)

    def set_warm_start_integer(self, solution: FacilitySolution) -> None:
        for c in self.client_var_to_index:
            self.model.CFN.wcsp.setBestValue(
                self.client_var_to_index[c], solution.facility_for_customers[c]
            )
        facility_used = set(solution.facility_for_customers)
        for i in self.used_var_to_index:
            if i in facility_used:
                self.model.CFN.wcsp.setBestValue(self.used_var_to_index[i], 1)
            else:
                self.model.CFN.wcsp.setBestValue(self.used_var_to_index[i], 0)


ToulbarFacilitySolverForLns = to_lns_toulbar(ToulbarFacilitySolver)


class FacilityConstraintHandlerToulbar(ConstraintHandler):
    """
    The allocated facility is frozen for a subpart of customers (fraction given in the constructor)
    """

    def __init__(
        self, problem: FacilityProblem, fraction_of_customers: Optional[float] = 0.4
    ):
        self.problem = problem
        self.fraction_of_customers = fraction_of_customers

    def adding_constraint_from_results_store(
        self,
        solver: ToulbarFacilitySolverForLns,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        sol: FacilitySolution = result_storage.get_best_solution_fit()[0]
        customers = random.sample(
            range(self.problem.customer_count),
            k=int(self.fraction_of_customers * self.problem.customer_count),
        )
        solver.model.CFN.timer(100)
        if solver.modeling == ModelingToulbarFacility.INTEGER:
            text = ",".join(
                f"{solver.client_var_to_index[c]}={sol.facility_for_customers[c]}"
                for c in sorted(customers)
            )
            text = "," + text
            solver.model.Parse(text)
        if solver.modeling == ModelingToulbarFacility.BINARY:
            text = ",".join(
                [
                    f"{solver.key_to_index[f,c]}={1 if sol.facility_for_customers[c]==f else 0}"
                    for c in customers
                    for f in range(self.problem.facility_count)
                ]
            )
            text = "," + text
            solver.model.Parse(text)
        solver.set_warm_start(solution=sol)

    def remove_constraints_from_previous_iteration(
        self,
        solver: ToulbarFacilitySolverForLns,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        pass


class FacilityConstraintHandlerDestroyFacilityToulbar(ConstraintHandler):
    """
    Select some factories, where all current allocated customers to those factories are totally free to be
    reallocated in the reduced problem.
    """

    def __init__(self, problem: FacilityProblem):
        self.problem = problem

    def adding_constraint_from_results_store(
        self,
        solver: ToulbarFacilitySolverForLns,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        sol: FacilitySolution = result_storage.get_best_solution_fit()[0]
        facilities_used = set(sol.facility_for_customers)
        nb_facilities = len(facilities_used)
        facilities_to_dst = random.sample(
            range(nb_facilities), k=max(1, int(0.2 * nb_facilities))
        )
        solver.model.CFN.timer(100)
        if solver.modeling == ModelingToulbarFacility.INTEGER:
            text = ",".join(
                f"{solver.client_var_to_index[c]}={sol.facility_for_customers[c]}"
                for c in range(self.problem.customer_count)
                if sol.facility_for_customers[c] not in facilities_to_dst
            )
            text = "," + text
            try:
                solver.model.Parse(text)
            except:
                solver.model.ClearPropagationQueues()
                pass
        if solver.modeling == ModelingToulbarFacility.BINARY:
            text = ",".join(
                [
                    f"{solver.key_to_index[f,c]}={1 if sol.facility_for_customers[c]==f else 0}"
                    for c in range(self.problem.customer_count)
                    for f in range(self.problem.facility_count)
                    if f not in facilities_to_dst
                ]
            )
            text = "," + text
            solver.model.Parse(text)
        solver.set_warm_start(solution=sol)

    def remove_constraints_from_previous_iteration(
        self,
        solver: ToulbarFacilitySolverForLns,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        pass
