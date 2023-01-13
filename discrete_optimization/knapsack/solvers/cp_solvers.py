#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import random
from datetime import timedelta
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from minizinc import Instance, Model, Result, Solver

from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    CPSolverName,
    MinizincCPSolver,
    ParametersCP,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    TupleFitness,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
    KnapsackSolutionMultidimensional,
    MultidimensionalKnapsack,
    MultiScenarioMultidimensionalKnapsack,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))


class KnapsackSol:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective: int, _output_item: Optional[str], **kwargs: Any):
        self.objective = objective
        self.dict = kwargs
        logger.debug(f"One solution {self.objective}")
        logger.debug(f"Output {_output_item}")

    def check(self) -> bool:
        return True


class CPKnapsackMZN(MinizincCPSolver, SolverKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        self.silent_solve_error = silent_solve_error
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["list_items"]
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        l_items = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                l_items.append(result[i, "list_items"])
                objectives.append(result[i, "objective"])
        else:
            l_items.append(result["list_items"])
            objectives.append(result["objective"])
        list_solutions_fit: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for items, objective in zip(l_items, objectives):
            taken = [0] * self.knapsack_model.nb_items
            weight = 0
            value = 0
            for i in range(len(items)):
                if items[i] != 0:
                    taken[self.knapsack_model.list_items[items[i] - 1].index] = 1
                    weight += self.knapsack_model.list_items[items[i] - 1].weight
                    value += self.knapsack_model.list_items[items[i] - 1].value
            sol = KnapsackSolution(
                problem=self.knapsack_model,
                value=value,
                weight=weight,
                list_taken=taken,
            )
            fit = self.aggreg_sol(sol)
            list_solutions_fit.append((sol, fit))
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        # Load n-Queens model from file
        model = Model(os.path.join(this_path, "../minizinc/knapsack_mzn.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["nb_items"] = self.knapsack_model.nb_items
        instance["values"] = [0] + [
            self.knapsack_model.list_items[i].value
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["weights"] = [0] + [
            self.knapsack_model.list_items[i].weight
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["max_capacity"] = self.knapsack_model.max_capacity
        self.instance = instance


class CPKnapsackMZN2(MinizincCPSolver, SolverKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        self.silent_solve_error = silent_solve_error
        self.cp_solver_name = cp_solver_name
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        # Load n-Queens model from file
        model = Model(os.path.join(this_path, "../minizinc/knapsack_global.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["nb_items"] = self.knapsack_model.nb_items
        instance["values"] = [
            self.knapsack_model.list_items[i].value
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["weights"] = [
            self.knapsack_model.list_items[i].weight
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["max_capacity"] = self.knapsack_model.max_capacity
        self.instance = instance

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        l_items_taken = []
        intermediate_solution = parameters_cp.intermediate_solution
        if intermediate_solution:
            for i in range(len(result)):
                l_items_taken.append(result[i, "taken"])
        else:
            l_items_taken.append(result["taken"])
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for items_taken in l_items_taken:
            taken = [0] * self.knapsack_model.nb_items
            weight = 0.0
            value = 0.0
            for i in range(len(items_taken)):
                if items_taken[i] != 0:
                    taken[self.knapsack_model.list_items[i].index] = 1
                    weight += self.knapsack_model.list_items[i].weight
                    value += self.knapsack_model.list_items[i].value
            sol = KnapsackSolution(
                problem=self.knapsack_model,
                value=value,
                weight=weight,
                list_taken=taken,
            )
            fit = self.aggreg_sol(sol)
            list_solution_fits.append((sol, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )

    def retrieve(self, items_taken: List[int]) -> List[KnapsackSolution]:
        taken = [0] * self.knapsack_model.nb_items
        weight = 0.0
        value = 0.0
        for i in range(len(items_taken)):
            if items_taken[i] != 0:
                taken[self.knapsack_model.list_items[i].index] = 1
                weight += self.knapsack_model.list_items[i].weight
                value += self.knapsack_model.list_items[i].value
        return [
            KnapsackSolution(
                problem=self.knapsack_model,
                value=value,
                weight=weight,
                list_taken=taken,
            )
        ]


class CPMultidimensionalSolver(MinizincCPSolver):
    def __init__(
        self,
        knapsack_model: MultidimensionalKnapsack,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        self.knapsack_model = knapsack_model
        self.silent_solve_error = silent_solve_error
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["list_items"]
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )
        self.custom_output_type = False

    def init_model(self, **kwargs: Any) -> None:
        model = Model(
            os.path.join(this_path, "../minizinc/multidimension_knapsack.mzn")
        )
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        custom_output_type = kwargs.get("output_type", False)
        if custom_output_type:
            model.output_type = KnapsackSol
            self.custom_output_type = True
        instance = Instance(solver, model)
        instance["nb_items"] = self.knapsack_model.nb_items
        instance["nb_dimension"] = len(self.knapsack_model.max_capacities)
        instance["values"] = [
            int(self.knapsack_model.list_items[i].value)
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["weights"] = [
            [
                self.knapsack_model.list_items[i].weights[j]
                for j in range(instance["nb_dimension"])
            ]
            for i in range(self.knapsack_model.nb_items)
        ]
        instance["max_capacity"] = self.knapsack_model.max_capacities
        self.instance = instance

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        l_taken = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                if self.custom_output_type:
                    l_taken.append(result[i].dict["taken"])
                    objectives.append(result[i].objective)
                else:
                    l_taken.append(result[i, "taken"])
                    objectives.append(result[i, "objective"])
        else:
            if self.custom_output_type:
                l_taken.append(result.dict["taken"])
                objectives.append(result.objective)
            else:
                l_taken.append(result["taken"])
                objectives.append(result["objective"])
        list_solutions_fit: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for taken, objective in zip(l_taken, objectives):
            sol = KnapsackSolutionMultidimensional(
                problem=self.knapsack_model, list_taken=taken
            )
            fit = self.aggreg_sol(sol)
            list_solutions_fit.append((sol, fit))
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )


class CPMultidimensionalMultiScenarioSolver(MinizincCPSolver):
    def __init__(
        self,
        knapsack_model: MultiScenarioMultidimensionalKnapsack,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        self.silent_solve_error = silent_solve_error
        self.knapsack_model = knapsack_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["list_items"]
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.knapsack_model,
            params_objective_function=params_objective_function,
        )
        self.custom_output_type = False

    def init_model(self, **kwargs: Any) -> None:
        model = Model(
            os.path.join(this_path, "../minizinc/multidim_multiscenario_knapsack.mzn")
        )
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        custom_output_type = kwargs.get("output_type", False)
        if custom_output_type:
            model.output_type = KnapsackSol
            self.custom_output_type = True
        instance = Instance(solver, model)
        list_problems: Sequence[
            MultidimensionalKnapsack
        ] = self.knapsack_model.list_problem
        instance["nb_items"] = list_problems[0].nb_items
        instance["nb_dimension"] = len(list_problems[0].max_capacities)
        instance["nb_scenario"] = len(list_problems)
        instance["values"] = [
            [
                int(list_problems[j].list_items[i].value)
                for j in range(instance["nb_scenario"])
            ]
            for i in range(instance["nb_items"])
        ]
        instance["weights"] = [
            [
                [
                    list_problems[k].list_items[i].weights[j]
                    for k in range(instance["nb_scenario"])
                ]
                for j in range(instance["nb_dimension"])
            ]
            for i in range(instance["nb_items"])
        ]
        instance["max_capacity"] = [
            [list_problems[s].max_capacities[k] for s in range(instance["nb_scenario"])]
            for k in range(instance["nb_dimension"])
        ]
        self.instance = instance

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        l_taken = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                if self.custom_output_type:
                    l_taken.append(result[i].dict["taken"])
                    objectives.append(result[i].objective)
                else:
                    l_taken.append(result[i, "taken"])
                    objectives.append(result[i, "objective"])
        else:
            if self.custom_output_type:
                l_taken.append(result.dict["taken"])
                objectives.append(result.objective)
            else:
                l_taken += [result["taken"]]
                objectives += [result["objective"]]
        list_solutions_fit: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for taken, objective in zip(l_taken, objectives):
            sol = KnapsackSolutionMultidimensional(
                problem=self.knapsack_model, list_taken=taken
            )
            fit = self.aggreg_sol(sol)
            list_solutions_fit.append((sol, fit))
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=None,
            mode_optim=self.params_objective_function.sense_function,
        )


class KnapConstraintHandler(ConstraintHandler):
    def __init__(self, fraction_fix: float = 0.95):
        self.fraction_fix = fraction_fix

    def adding_constraint_from_results_store(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if not isinstance(cp_solver, CPMultidimensionalMultiScenarioSolver):
            raise ValueError(
                "cp_solver must a CPMultidimensionalMultiScenarioSolver for this constraint."
            )
        if last_result_store is None:
            raise ValueError("This constraint need last_result_store to be not None.")
        strings = []
        nb_item = cp_solver.knapsack_model.list_problem[0].nb_items
        range_item = range(nb_item)
        subpart_item = set(random.sample(range_item, int(self.fraction_fix * nb_item)))
        current_best_solution = last_result_store.get_last_best_solution()[0]
        if not isinstance(
            current_best_solution, (KnapsackSolution, KnapsackSolutionMultidimensional)
        ):
            raise RuntimeError(
                "current_best_solution must be a KnapsackSolution "
                "or a KnapsackSolutionMultidimensional."
            )
        for i in range_item:
            if i in subpart_item:
                strings += [
                    "constraint taken["
                    + str(i + 1)
                    + "] == "
                    + str(1 if current_best_solution.list_taken[i] else 0)
                    + ";\n"
                ]
                child_instance.add_string(strings[-1])
        return strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        previous_constraints: Iterable[Any],
    ) -> None:
        if not isinstance(cp_solver, CPMultidimensionalMultiScenarioSolver):
            raise ValueError(
                "cp_solver must a CPMultidimensionalMultiScenarioSolver for this constraint."
            )
        pass
