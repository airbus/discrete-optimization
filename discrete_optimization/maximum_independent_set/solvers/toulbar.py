#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Iterable, Optional

from discrete_optimization.generic_tools.lns_tools import ConstraintHandler

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


class ToulbarMisSolver(MisSolver, WarmstartMixin):
    model: "pytoulbar2.CFN"

    def init_model(self, **kwargs):
        model = pytoulbar2.CFN(kwargs.get("UB", 0))

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

    def solve(self, **kwargs: Any) -> ResultStorage:
        time_limit = kwargs.get("time_limit", 20)
        self.model.CFN.timer(time_limit)
        solution = self.model.Solve(showSolutions=1)
        logger.info(f"=== Solution === \n {solution}")
        if solution is None:
            return ResultStorage(
                mode_optim=self.params_objective_function.sense_function,
                list_solution_fits=[],
            )
        sol = MisSolution(
            problem=self.problem,
            chosen=solution[0][:],
        )
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            mode_optim=self.params_objective_function.sense_function,
            list_solution_fits=[(sol, fit)],
        )

    def set_warm_start(self, solution: MisSolution) -> None:
        for i in range(self.problem.number_nodes):
            self.model.CFN.wcsp.setBestValue(i, solution.chosen[i])


class ToulbarMisSolverForLns(ToulbarMisSolver):
    depth: int = 0

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.model.SolveFirst()
        self.depth = self.model.Depth()
        self.model.Store()
        self.initub = self.model.GetUB()

    def solve(self, time_limit: Optional[int] = 20, **kwargs: Any) -> ResultStorage:
        try:
            solution = self.model.SolveNext(showSolutions=1, timeLimit=time_limit)
            logger.info(f"=== Solution === \n {solution}")
            logger.info(
                f"Best solution = {solution[1]}, Bound = {self.model.GetDDualBound()}"
            )
            # Reinit for next iteration.
            self.model.Restore(self.depth)
            self.model.Store()
            self.model.SetUB(self.initub)
            if solution is not None:
                sol = MisSolution(problem=self.problem, chosen=solution[0])
                fit = self.aggreg_from_sol(sol)
                return self.create_result_storage(
                    [(sol, fit)],
                )
            else:
                return self.create_result_storage()
        except:
            self.model.ClearPropagationQueues()
            self.model.Restore(self.depth)
            self.model.Store()
            self.model.SetUB(self.initub)
            logger.info("Solve failed in given time")
            return self.create_result_storage()


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
