#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random
from typing import Any, Iterable

from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.tsp.mutation import find_intersection
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.solvers.tsp_solver import TspProblem, TspSolver
from discrete_optimization.tsp.utils import build_matrice_distance

logger = logging.getLogger(__name__)
try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = False


class ToulbarTspSolver(ToulbarSolver, TspSolver, WarmstartMixin):
    hyperparameters = ToulbarSolver.hyperparameters + [
        CategoricalHyperparameter(
            name="encoding_all_diff",
            choices=["binary", "salldiff", "salldiffdp", "salldiffkp", "walldiff"],
            default="salldiffkp",
        )
    ]

    def set_warm_start(self, solution: TspSolution) -> None:
        indexes = self.problem.original_indices_to_permutation_indices_dict
        for i in range(len(solution.permutation)):
            self.model.CFN.wcsp.setBestValue(i, indexes[solution.permutation[i]])

    def __init__(self, problem: TspProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.problem.end_index, self.problem.start_index] = 0

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        model = pytoulbar2.CFN(
            vns=kwargs.get("vns", None), ubinit=kwargs.get("ub", None)
        )
        nb_nodes_variable = len(self.problem.original_indices_to_permutation_indices)
        indexes = self.problem.original_indices_to_permutation_indices_dict
        rev_indexes = {indexes[i]: i for i in indexes}
        for i in range(nb_nodes_variable):
            model.AddVariable(f"perm_{i}", values=range(nb_nodes_variable))
        logger.debug("Starting set all diff")
        model.AddAllDifferent(
            scope=range(nb_nodes_variable),
            encoding=kwargs["encoding_all_diff"],
            incremental=False,
        )
        logger.debug("set all diff done")
        model.AddFunction(
            [f"perm_0"],
            [
                self.distance_matrix[self.problem.start_index, rev_indexes[i]]
                for i in range(nb_nodes_variable)
            ],
        )

        model.AddFunction(
            [f"perm_{nb_nodes_variable-1}"],
            [
                self.distance_matrix[rev_indexes[i], self.problem.end_index]
                for i in range(nb_nodes_variable)
            ],
        )
        for i in range(nb_nodes_variable - 1):
            model.AddFunction(
                [f"perm_{i}", f"perm_{i+1}"],
                [
                    self.distance_matrix[rev_indexes[ii], rev_indexes[jj]]
                    for ii in range(nb_nodes_variable)
                    for jj in range(nb_nodes_variable)
                ],
            )
        self.model = model
        self.rev_indexes = rev_indexes
        logger.debug("Init done")

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> TspSolution:
        return TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=[self.rev_indexes[x] for x in solution_from_toulbar2[0]],
        )


class TspConstraintHandlerToulbar(ConstraintHandler):
    def __init__(self, fraction_nodes: float = 0.5):
        self.fraction_nodes = fraction_nodes

    def adding_constraint_from_results_store(
        self, solver: ToulbarTspSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        sol: TspSolution = result_storage.get_best_solution_fit()[0]

        list_ = find_intersection(
            variable=sol, points=solver.problem.list_points, nb_tests=1000
        )
        nb_nodes = len(sol.permutation)
        subset = random.sample(range(nb_nodes), k=int(self.fraction_nodes * nb_nodes))
        if len(list_) > 0:
            subset = [i for i in subset if i not in [list_[0][0], list_[0][1]]]
        solver.model.CFN.timer(100)
        indexes = solver.problem.original_indices_to_permutation_indices_dict
        text = ",".join(f"{i}={indexes[sol.permutation[i]]}" for i in subset)
        text = "," + text
        solver.model.Parse(text)
        solver.set_warm_start(sol)

    def remove_constraints_from_previous_iteration(
        self, solver: SolverDO, previous_constraints: Iterable[Any], **kwargs: Any
    ) -> None:
        pass
