#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
import os
import random
from typing import Any, Iterable, Optional

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.toulbar_tools import (
    ToulbarSolver,
    to_lns_toulbar,
)

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True
import tqdm

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

this_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class ToulbarMisSolver(ToulbarSolver, MisSolver, WarmstartMixin):
    hyperparameters = ToulbarSolver.hyperparameters

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        return MisSolution(problem=self.problem, chosen=solution_from_toulbar2[0][:])

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        model = pytoulbar2.CFN(kwargs.get("UB", 0), vns=kwargs["vns"])

        for i in range(self.problem.number_nodes):
            model.AddVariable(name=f"x_{i}", values=[0, 1])
            model.AddFunction([f"x_{i}"], [0, -int(self.problem.attr_list[i])])
        for e in tqdm.tqdm(self.problem.edges):
            i0 = self.problem.nodes_to_index[e[0]]
            i1 = self.problem.nodes_to_index[e[1]]
            # print("e", e)
            # model.AddLinearConstraint([1, 1], [f"x_{i0}", f"x_{i1}"], "<=", 1)
            model.AddSumConstraint([f"x_{i0}", f"x_{i1}"], operand="<=", rightcoef=1)
            # model.AddFunction([f"x_{i0}", f"x_{i1}"],
            #                    [1000 if x == y == 1 else 0
            #                    for x in [0, 1] for y in [0, 1]])
            # Problem.AddFunction([vars[i0], vars[i1]],
            #                      [10 ** 12 if x == y == 1 else 0
            #                       for x in [0, 1] for y in [0, 1]])
        self.model = model

    def set_warm_start(self, solution: MisSolution) -> None:
        for i in range(self.problem.number_nodes):
            self.model.CFN.wcsp.setBestValue(i, solution.chosen[i])


ToulbarMisSolverForLns = to_lns_toulbar(ToulbarMisSolver)


class MisConstraintHandlerToulbar(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self,
        solver: ToulbarMisSolverForLns,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        pass

    def __init__(self, fraction_node: float = 0.3):
        self.fraction_node = fraction_node

    def adding_constraint_from_results_store(
        self,
        solver: ToulbarMisSolverForLns,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        best_sol: MisSolution = result_storage.get_best_solution_fit()[0]
        problem: MisProblem = solver.problem
        random_indexes = random.sample(
            range(problem.number_nodes),
            k=int(self.fraction_node * problem.number_nodes),
        )
        text = ",".join(
            f"{index}={int(best_sol.chosen[index])}" for index in random_indexes
        )
        text = "," + text
        # circumvent some timeout issue when calling Parse(text). TODO : investigate.
        solver.model.CFN.timer(100)
        try:
            solver.model.Parse(text)
        except Exception as e:
            solver.model.ClearPropagationQueues()
            logger.warning(f"Error raised during parsing certificate : {e}")
        solver.set_warm_start(best_sol)
