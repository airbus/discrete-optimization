#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import random
from typing import Any, Iterable, Optional, Sequence

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    CPSolverName,
    MinizincCPSolver,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
    FloatHyperparameter,
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


class CPKnapsackMZN(MinizincCPSolver, SolverKnapsack):
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        )
    ]

    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.key_decision_variable = ["list_items"]

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> KnapsackSolution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        items = kwargs["list_items"]
        taken = [0] * self.problem.nb_items
        weight = 0
        value = 0
        for i in range(len(items)):
            if items[i] != 0:
                taken[items[i] - 1] = 1
                weight += self.problem.list_items[items[i] - 1].weight
                value += self.problem.list_items[items[i] - 1].value
        return KnapsackSolution(
            problem=self.problem,
            value=value,
            weight=weight,
            list_taken=taken,
        )

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cp_solver_name = kwargs["cp_solver_name"]
        model = Model(os.path.join(this_path, "../minizinc/knapsack_mzn.mzn"))
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, model)
        instance["nb_items"] = self.problem.nb_items
        instance["values"] = [0] + [
            self.problem.list_items[i].value for i in range(self.problem.nb_items)
        ]
        instance["weights"] = [0] + [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        instance["max_capacity"] = self.problem.max_capacity
        self.instance = instance


class CPKnapsackMZN2(MinizincCPSolver, SolverKnapsack):
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        )
    ]

    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cp_solver_name = kwargs["cp_solver_name"]
        model = Model(os.path.join(this_path, "../minizinc/knapsack_global.mzn"))
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, model)
        instance["nb_items"] = self.problem.nb_items
        instance["values"] = [
            self.problem.list_items[i].value for i in range(self.problem.nb_items)
        ]
        instance["weights"] = [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        instance["max_capacity"] = self.problem.max_capacity
        self.instance = instance

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> KnapsackSolution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        items_taken = kwargs["taken"]
        taken = [0] * self.problem.nb_items
        weight = 0.0
        value = 0.0
        for i in range(len(items_taken)):
            if items_taken[i] != 0:
                taken[
                    self.problem.index_to_index_list[self.problem.list_items[i].index]
                ] = 1
                weight += self.problem.list_items[i].weight
                value += self.problem.list_items[i].value
        return KnapsackSolution(
            problem=self.problem,
            value=value,
            weight=weight,
            list_taken=taken,
        )


class CPMultidimensionalSolver(MinizincCPSolver):
    problem: MultidimensionalKnapsack
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        )
    ]

    def __init__(
        self,
        problem: MultidimensionalKnapsack,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.key_decision_variable = ["list_items"]

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cp_solver_name = kwargs["cp_solver_name"]
        model = Model(
            os.path.join(this_path, "../minizinc/multidimension_knapsack.mzn")
        )
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, model)
        instance["nb_items"] = self.problem.nb_items
        instance["nb_dimension"] = len(self.problem.max_capacities)
        instance["values"] = [
            int(self.problem.list_items[i].value) for i in range(self.problem.nb_items)
        ]
        instance["weights"] = [
            [
                self.problem.list_items[i].weights[j]
                for j in range(instance["nb_dimension"])
            ]
            for i in range(self.problem.nb_items)
        ]
        instance["max_capacity"] = self.problem.max_capacities
        self.instance = instance

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> KnapsackSolutionMultidimensional:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        taken = kwargs["taken"]
        return KnapsackSolutionMultidimensional(problem=self.problem, list_taken=taken)


class CPMultidimensionalMultiScenarioSolver(MinizincCPSolver):
    problem: MultiScenarioMultidimensionalKnapsack
    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CPSolverName, default=CPSolverName.CHUFFED
        )
    ]

    def __init__(
        self,
        problem: MultiScenarioMultidimensionalKnapsack,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.key_decision_variable = ["list_items"]

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cp_solver_name = kwargs["cp_solver_name"]
        model = Model(
            os.path.join(this_path, "../minizinc/multidim_multiscenario_knapsack.mzn")
        )
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, model)
        list_problems: Sequence[MultidimensionalKnapsack] = self.problem.list_problem
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

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> KnapsackSolutionMultidimensional:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        taken = kwargs["taken"]
        return KnapsackSolutionMultidimensional(problem=self.problem, list_taken=taken)


class KnapConstraintHandler(ConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_fix", default=0.95, low=0.0, high=1.0),
    ]

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
        nb_item = cp_solver.problem.list_problem[0].nb_items
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
