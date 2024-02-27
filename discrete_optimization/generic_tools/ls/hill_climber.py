#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import pickle
import time
from typing import Any, List, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_evaluate_function_aggregated,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandler,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    ResultStorage,
)

logger = logging.getLogger(__name__)


class HillClimber(SolverDO):
    def __init__(
        self,
        problem: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.mutator = mutator
        self.restart_handler = restart_handler
        self.mode_mutation = mode_mutation
        self.params_objective_function = params_objective_function
        if params_objective_function is not None:
            self.mode_optim = params_objective_function.sense_function
        else:
            self.mode_optim = ModeOptim.MAXIMIZATION
        self.store_solution = store_solution

    def solve(
        self,
        initial_variable: Solution,
        nb_iteration_max: int,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        callbacks_list = CallbackList(callbacks=callbacks)

        objective = self.aggreg_from_dict(self.problem.evaluate(initial_variable))
        cur_variable = initial_variable.copy()
        if self.store_solution:
            store = ResultStorage(
                list_solution_fits=[(initial_variable, objective)],
                best_solution=initial_variable.copy(),
                limit_store=True,
                nb_best_store=1000,
            )
        else:
            store = ResultStorage(
                list_solution_fits=[(initial_variable, objective)],
                best_solution=initial_variable.copy(),
                limit_store=True,
                nb_best_store=1,
            )
        cur_objective = objective
        cur_best_objective = objective
        self.restart_handler.best_fitness = objective
        self.restart_handler.solution_best = initial_variable.copy()
        iteration = 0
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_sol(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict(objective_dict_values)
            if self.mode_optim == ModeOptim.MINIMIZATION and objective < cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective < cur_best_objective
            elif (
                self.mode_optim == ModeOptim.MAXIMIZATION and objective > cur_objective
            ):
                accept = True
                local_improvement = True
                global_improvement = objective > cur_best_objective
            if accept:
                cur_objective = objective
                cur_variable = nv
            else:
                cur_variable = move.backtrack_local_move(nv)
            if self.store_solution:
                store.add_solution(nv.copy(), objective)
            if global_improvement:
                logger.debug(f"iter {iteration}")
                logger.debug(f"new obj {objective} better than {cur_best_objective}")
                cur_best_objective = objective
                if not self.store_solution:
                    store.add_solution(cur_variable.copy(), objective)
            # Update the temperature
            self.restart_handler.update(
                nv, objective, global_improvement, local_improvement
            )
            # Update info in restart handler
            cur_variable, cur_objective = self.restart_handler.restart(
                cur_variable, cur_objective
            )
            # possibly restart somewhere
            iteration += 1

            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(
                step=iteration, res=store, solver=self
            )
            if stopping:
                break

        store.finalize()
        # end of solve callback
        callbacks_list.on_solve_end(res=store, solver=self)
        return store


class HillClimberPareto(HillClimber):
    def __init__(
        self,
        problem: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
    ):
        super().__init__(
            problem=problem,
            mutator=mutator,
            restart_handler=restart_handler,
            mode_mutation=mode_mutation,
            params_objective_function=params_objective_function,
            store_solution=store_solution,
        )

    def solve(
        self,
        initial_variable: Solution,
        nb_iteration_max: int,
        update_iteration_pareto: int = 1000,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ParetoFront:
        callbacks_list = CallbackList(callbacks=callbacks)

        objective = self.aggreg_from_dict(self.problem.evaluate(initial_variable))
        pareto_front = ParetoFront(
            list_solution_fits=[(initial_variable, objective)],
            best_solution=initial_variable.copy(),
            limit_store=True,
            nb_best_store=1000,
        )
        cur_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        self.restart_handler.best_fitness = objective
        self.restart_handler.solution_best = initial_variable.copy()
        iteration = 0
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if iteration % update_iteration_pareto == 0:
                pareto_front.finalize()
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_sol(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict(objective_dict_values)
            if self.mode_optim == ModeOptim.MINIMIZATION and objective < cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective < cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif (
                self.mode_optim == ModeOptim.MINIMIZATION and objective == cur_objective
            ):
                accept = True
                local_improvement = True
                global_improvement = objective == cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif (
                self.mode_optim == ModeOptim.MAXIMIZATION and objective > cur_objective
            ):
                accept = True
                local_improvement = True
                global_improvement = objective > cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif (
                self.mode_optim == ModeOptim.MAXIMIZATION and objective == cur_objective
            ):
                accept = True
                local_improvement = True
                global_improvement = objective == cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            if accept:
                logger.debug(f"Accept : {objective}")
                cur_objective = objective
                cur_variable = nv
            else:
                cur_variable = move.backtrack_local_move(nv)
            if global_improvement:
                logger.debug(f"iter {iteration}")
                logger.debug(f"new obj {objective} better than {cur_best_objective}")
                cur_best_objective = objective
            # Update the temperature
            self.restart_handler.update(
                nv, objective, global_improvement, local_improvement
            )
            logger.debug(f"Len pareto : {pareto_front.len_pareto_front()}")
            # Update info in restart handler
            cur_variable, cur_objective = self.restart_handler.restart(
                cur_variable, cur_objective
            )
            # possibly restart somewhere
            iteration += 1

            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(
                step=iteration, res=pareto_front, solver=self
            )
            if stopping:
                break

        pareto_front.finalize()
        # end of solve callback
        callbacks_list.on_solve_end(res=pareto_front, solver=self)
        return pareto_front
