#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import contextlib
import logging
import random
import sys
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from minizinc import Instance

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import MinizincCPSolver, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lns_tools import (
    BaseLNS,
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


logger = logging.getLogger(__name__)


class MznConstraintHandler(ConstraintHandler):
    @abstractmethod
    def adding_constraint_from_results_store(
        self,
        solver: MinizincCPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        ...

    def remove_constraints_from_previous_iteration(
        self,
        solver: SolverDO,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """Remove previous constraints.

        Nothing to do for minizinc solvers as the constraints were added only to the child instance.

        """
        pass


class LNS_CP(BaseLNS):
    """Large Neighborhood Search solver using a minzinc cp solver at each iteration."""

    subsolver: MinizincCPSolver
    """Sub-solver used by this lns solver at each iteration."""

    constraint_handler: MznConstraintHandler

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[MinizincCPSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[MznConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            subsolver=subsolver,
            initial_solution_provider=initial_solution_provider,
            constraint_handler=constraint_handler,
            post_process_solution=post_process_solution,
            params_objective_function=params_objective_function,
            **kwargs,
        )

    def init_model(self, **kwargs: Any) -> None:
        if self.subsolver.instance is None:
            self.subsolver.init_model(**kwargs)
            if self.subsolver.instance is None:  # for mypy
                raise RuntimeError(
                    "CP model instance must not be None after calling init_model()!"
                )

    def create_submodel(self) -> contextlib.AbstractContextManager:
        """Create a branch of the current instance, wrapped in a context manager."""
        return self.subsolver.instance.branch()

    def solve_with_subsolver(
        self,
        iteration: int,
        instance: Instance,
        parameters_cp: ParametersCP,
        **kwargs: Any,
    ) -> ResultStorage:
        if iteration == 0:
            parameters_cp0 = parameters_cp.copy()
            parameters_cp0.time_limit = parameters_cp.time_limit_iter0
            result_store = self.subsolver.solve(
                parameters_cp=parameters_cp0, instance=instance
            )
        else:
            result_store = self.subsolver.solve(
                parameters_cp=parameters_cp, instance=instance
            )
        return result_store

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return super().solve(
            parameters_cp=parameters_cp,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            skip_first_iteration=skip_first_iteration,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            callbacks=callbacks,
            **kwargs,
        )


class ConstraintStatus(TypedDict):
    nb_usage: int
    nb_improvement: int
    name: str


class ConstraintHandlerMix(MznConstraintHandler):
    def __init__(
        self,
        problem: Problem,
        list_constraints_handler: List[MznConstraintHandler],
        list_proba: List[float],
        update_proba: bool = True,
        tag_constraint_handler: Optional[List[str]] = None,
        sequential: bool = False,
    ):
        self.problem = problem
        self.list_constraints_handler = list_constraints_handler
        self.sequential = sequential
        if tag_constraint_handler is None:
            self.tag_constraint_handler = [
                str(i) for i in range(len(self.list_constraints_handler))
            ]
        else:
            self.tag_constraint_handler = tag_constraint_handler
        self.list_proba = np.array(list_proba)
        self.list_proba = self.list_proba / np.sum(self.list_proba)
        self.index_np = np.array(range(len(self.list_proba)), dtype=np.int_)
        self.current_iteration = 0
        self.status: Dict[int, ConstraintStatus] = {
            i: {
                "nb_usage": 0,
                "nb_improvement": 0,
                "name": self.tag_constraint_handler[i],
            }
            for i in range(len(self.list_constraints_handler))
        }
        self.last_index_param: Optional[int] = None
        self.last_fitness: Optional[fitness_class] = None
        self.update_proba = update_proba

    def adding_constraint_from_results_store(
        self,
        solver: MinizincCPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        new_fitness = result_storage.get_best_solution_fit()[1]
        if self.last_index_param is not None:
            if new_fitness != self.last_fitness:
                self.status[self.last_index_param]["nb_improvement"] += 1
                self.last_fitness = new_fitness
                if self.update_proba:
                    self.list_proba[self.last_index_param] *= 1.05
                    self.list_proba = self.list_proba / np.sum(self.list_proba)
            else:
                if self.update_proba:
                    self.list_proba[self.last_index_param] *= 0.95
                    self.list_proba = self.list_proba / np.sum(self.list_proba)
        else:
            self.last_fitness = new_fitness
        if self.sequential:
            if self.last_index_param is not None:
                choice = (self.last_index_param + 1) % len(
                    self.list_constraints_handler
                )
            else:
                choice = 0
        else:
            if random.random() <= 0.95:
                choice = np.random.choice(self.index_np, size=1, p=self.list_proba)[0]
            else:
                max_improvement = max(
                    [
                        self.status[x]["nb_improvement"]
                        / max(self.status[x]["nb_usage"], 1)
                        for x in self.status
                    ]
                )
                choice = random.choice(
                    [
                        x
                        for x in self.status
                        if self.status[x]["nb_improvement"]
                        / max(self.status[x]["nb_usage"], 1)
                        == max_improvement
                    ]
                )
        ch = self.list_constraints_handler[int(choice)]
        self.current_iteration += 1
        self.last_index_param = choice
        self.status[self.last_index_param]["nb_usage"] += 1
        logger.debug(f"Status {self.status}")
        return ch.adding_constraint_from_results_store(
            solver, child_instance, result_storage, last_result_store
        )
