#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import pickle
import time
from typing import Optional

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_evaluate_function_aggregated,
)
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandler,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    ResultStorage,
)

logger = logging.getLogger(__name__)


class HillClimber:
    def __init__(
        self,
        evaluator: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
        nb_solutions: int = 1000,
    ):
        self.evaluator = evaluator
        self.mutator = mutator
        self.restart_handler = restart_handler
        self.mode_mutation = mode_mutation
        self.params_objective_function = params_objective_function
        if params_objective_function is not None:
            self.mode_optim = params_objective_function.sense_function
        else:
            self.mode_optim = ModeOptim.MAXIMIZATION
        (
            self.aggreg_from_solution,
            self.aggreg_from_dict_values,
        ) = build_evaluate_function_aggregated(
            evaluator, params_objective_function=self.params_objective_function
        )
        self.store_solution = store_solution
        self.nb_solutions = nb_solutions

    def solve(
        self,
        initial_variable: Solution,
        nb_iteration_max: int,
        max_time_seconds: Optional[int] = None,
        pickle_result: bool = False,
        pickle_name: str = "debug",
    ) -> ResultStorage:
        objective = self.aggreg_from_dict_values(
            self.evaluator.evaluate(initial_variable)
        )
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
        cur_best_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        init_time = time.time()
        self.restart_handler.best_fitness = objective
        iteration = 0
        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_solution(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict_values(objective_dict_values)
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
                cur_best_variable = cur_variable.copy()
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
            if pickle_result and iteration % 20000 == 0:
                pickle.dump(cur_best_variable, open(pickle_name + ".pk", "wb"))
            if max_time_seconds is not None and iteration % 1000 == 0:
                if time.time() - init_time > max_time_seconds:
                    break
        store.finalize()
        return store


class HillClimberPareto(HillClimber):
    def __init__(
        self,
        evaluator: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
        nb_solutions: int = 1000,
    ):
        super().__init__(
            evaluator=evaluator,
            mutator=mutator,
            restart_handler=restart_handler,
            mode_mutation=mode_mutation,
            params_objective_function=params_objective_function,
            store_solution=store_solution,
            nb_solutions=nb_solutions,
        )

    def solve(
        self,
        initial_variable: Solution,
        nb_iteration_max: int,
        max_time_seconds: Optional[int] = None,
        pickle_result: bool = False,
        pickle_name: str = "tsp",
        update_iteration_pareto: int = 1000,
    ) -> ParetoFront:
        init_time = time.time()
        objective = self.aggreg_from_dict_values(
            self.evaluator.evaluate(initial_variable)
        )
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
        iteration = 0
        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if iteration % update_iteration_pareto == 0:
                pareto_front.finalize()
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_solution(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict_values(objective_dict_values)
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
            if max_time_seconds is not None and iteration % 1000 == 0:
                if time.time() - init_time > max_time_seconds:
                    break

        pareto_front.finalize()
        return pareto_front
