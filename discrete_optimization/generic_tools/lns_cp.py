#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
import sys
import time
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from minizinc import Instance

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    MinizincCPSolver,
    ParametersCP,
    StatusSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    PostProcessSolution,
    TrivialPostProcessSolution,
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


class ConstraintHandler(Hyperparametrizable):
    @abstractmethod
    def adding_constraint_from_results_store(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        ...

    @abstractmethod
    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        previous_constraints: Iterable[Any],
    ) -> None:
        ...


class LNS_CP(SolverDO):
    hyperparameters = [
        SubBrickHyperparameter("cp_solver_cls", choices=[], default=None),
        SubBrickKwargsHyperparameter(
            "cp_solver_kwargs", subbrick_hyperparameter="cp_solver_cls"
        ),
        SubBrickHyperparameter(
            "initial_solution_provider_cls", choices=[], default=None
        ),
        SubBrickKwargsHyperparameter(
            "initial_solution_provider_kwargs",
            subbrick_hyperparameter="initial_solution_provider_cls",
        ),
        SubBrickHyperparameter(
            "constraint_handler_cls",
            choices=[],
            default=None,
        ),
        SubBrickKwargsHyperparameter(
            "constraint_handler_kwargs",
            subbrick_hyperparameter="constraint_handler_cls",
        ),
        SubBrickHyperparameter(
            "post_process_solution_cls",
            choices=[],
            default=TrivialPostProcessSolution,
        ),
        SubBrickKwargsHyperparameter(
            "post_process_solution_kwargs",
            subbrick_hyperparameter="post_process_solution_cls",
        ),
        CategoricalHyperparameter(
            name="skip_first_iteration", choices=[True, False], default=False
        ),
    ]

    def __init__(
        self,
        problem: Problem,
        cp_solver: Optional[MinizincCPSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if cp_solver is None:
            if kwargs["cp_solver_kwargs"] is None:
                cp_solver_kwargs = kwargs
            else:
                cp_solver_kwargs = kwargs["cp_solver_kwargs"]
            if kwargs["cp_solver_cls"] is None:
                raise ValueError(
                    "`cp_solver_cls` cannot be None if `cp_solver` is not specified."
                )
            else:
                cp_solver_cls = kwargs["cp_solver_cls"]
                cp_solver = cp_solver_cls(problem=self.problem, **cp_solver_kwargs)
                cp_solver.init_model(**cp_solver_kwargs)
        self.cp_solver = cp_solver

        if constraint_handler is None:
            if kwargs["constraint_handler_kwargs"] is None:
                constraint_handler_kwargs = kwargs
            else:
                constraint_handler_kwargs = kwargs["constraint_handler_kwargs"]
            if kwargs["constraint_handler_cls"] is None:
                raise ValueError(
                    "`constraint_handler_cls` cannot be None if `constraint_handler` is not specified."
                )
            else:
                constraint_handler_cls = kwargs["constraint_handler_cls"]
                constraint_handler = constraint_handler_cls(
                    problem=self.problem, **constraint_handler_kwargs
                )
        self.constraint_handler = constraint_handler

        if post_process_solution is None:
            if kwargs["post_process_solution_kwargs"] is None:
                post_process_solution_kwargs = kwargs
            else:
                post_process_solution_kwargs = kwargs["post_process_solution_kwargs"]
            if kwargs["post_process_solution_cls"] is None:
                post_process_solution = None
            else:
                post_process_solution_cls = kwargs["post_process_solution_cls"]
                post_process_solution = post_process_solution_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **post_process_solution_kwargs,
                )
        self.post_process_solution = post_process_solution

        if initial_solution_provider is None:
            if kwargs["initial_solution_provider_kwargs"] is None:
                initial_solution_provider_kwargs = kwargs
            else:
                initial_solution_provider_kwargs = kwargs[
                    "initial_solution_provider_kwargs"
                ]
            if kwargs["initial_solution_provider_cls"] is None:
                initial_solution_provider = (
                    None  # ok if solve_lns with skip_first_iteration
                )
            else:
                initial_solution_provider_cls = kwargs["initial_solution_provider_cls"]
                initial_solution_provider = initial_solution_provider_cls(
                    problem=self.problem,
                    params_objective_function=self.params_objective_function,
                    **initial_solution_provider_kwargs,
                )
        self.initial_solution_provider = initial_solution_provider

    def solve_lns(
        self,
        parameters_cp: ParametersCP,
        nb_iteration_lns: int,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        # manage None post_process_solution (can happen in subclasses __init__)
        if self.post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()

        sense = self.params_objective_function.sense_function
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2 * nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        if self.cp_solver.instance is None:
            self.cp_solver.init_model()
            if self.cp_solver.instance is None:  # for mypy
                raise RuntimeError(
                    "CP model instance must not be None after calling init_model()!"
                )
        if not skip_first_iteration:
            if self.initial_solution_provider is None:
                raise ValueError(
                    "self.initial_solution_provider cannot be None if not skip_first_iteration."
                )
            store_lns = self.initial_solution_provider.get_starting_solution()
            store_lns = self.post_process_solution.build_other_solution(store_lns)
            init_solution, objective = store_lns.get_best_solution_fit()
            if init_solution is None:
                satisfy = False
            else:
                satisfy = self.problem.satisfy(init_solution)
            logger.debug(f"Satisfy Initial solution {satisfy}")
            try:
                logger.debug(
                    f"Nb task preempted = {init_solution.get_nb_task_preemption()}"  # type: ignore
                )
                logger.debug(f"Nb max preemption = {init_solution.get_max_preempted()}")  # type: ignore
            except:
                pass
            best_objective = objective
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=0, res=store_lns, solver=self)
        else:
            best_objective = (
                float("inf") if sense == ModeOptim.MINIMIZATION else -float("inf")
            )
            store_lns = None
            stopping = False

        result_store: ResultStorage
        if not stopping:
            for iteration in range(nb_iteration_lns):
                logger.info(
                    f"Starting iteration n° {iteration} current objective {best_objective}"
                )
                with self.cp_solver.instance.branch() as child:
                    if iteration == 0 and not skip_first_iteration or iteration >= 1:
                        constraint_iterable = self.constraint_handler.adding_constraint_from_results_store(
                            cp_solver=self.cp_solver,
                            child_instance=child,
                            result_storage=store_lns,
                            last_result_store=store_lns
                            if iteration == 0
                            else result_store,
                        )
                    try:
                        if iteration == 0:
                            parameters_cp0 = parameters_cp.copy()
                            parameters_cp0.time_limit = parameters_cp.time_limit_iter0
                            result_store = self.cp_solver.solve(
                                parameters_cp=parameters_cp0, instance=child
                            )

                        else:
                            result_store = self.cp_solver.solve(
                                parameters_cp=parameters_cp, instance=child
                            )
                        logger.info(f"iteration n° {iteration} Solved !!!")
                        logger.info(self.cp_solver.status_solver)
                        if len(result_store.list_solution_fits) > 0:
                            logger.debug("Solved !!!")
                            bsol, fit = result_store.get_best_solution_fit()
                            logger.debug(f"Fitness Before = {fit}")
                            if bsol is not None:
                                logger.debug(
                                    f"Satisfaction Before = {self.problem.satisfy(bsol)}"
                                )
                            else:
                                logger.debug(f"Satisfaction Before = {False}")
                            logger.debug("Post Process..")
                            result_store = (
                                self.post_process_solution.build_other_solution(
                                    result_store
                                )
                            )
                            bsol, fit = result_store.get_best_solution_fit()
                            if bsol is not None:
                                logger.debug(
                                    f"Satisfaction After = {self.problem.satisfy(bsol)}"
                                )
                            else:
                                logger.debug(f"Satisfaction After = {False}")
                            if (
                                sense == ModeOptim.MAXIMIZATION
                                and fit >= best_objective
                            ):
                                if fit > best_objective:
                                    current_nb_iteration_no_improvement = 0
                                else:
                                    current_nb_iteration_no_improvement += 1
                                best_objective = fit
                            elif sense == ModeOptim.MAXIMIZATION:
                                current_nb_iteration_no_improvement += 1
                            elif (
                                sense == ModeOptim.MINIMIZATION
                                and fit <= best_objective
                            ):
                                if fit < best_objective:
                                    current_nb_iteration_no_improvement = 0
                                else:
                                    current_nb_iteration_no_improvement += 1
                                best_objective = fit
                            elif sense == ModeOptim.MINIMIZATION:
                                current_nb_iteration_no_improvement += 1
                            if skip_first_iteration and iteration == 0:
                                store_lns = result_store
                            else:
                                for s, f in list(result_store.list_solution_fits):
                                    store_lns.add_solution(solution=s, fitness=f)
                        else:
                            current_nb_iteration_no_improvement += 1
                        if (
                            skip_first_iteration
                            and self.cp_solver.status_solver == StatusSolver.OPTIMAL
                            and iteration == 0
                            and self.problem.satisfy(bsol)
                            and stop_first_iteration_if_optimal
                        ):
                            logger.info("Finish LNS because found optimal solution")
                            break
                    except Exception as e:
                        current_nb_iteration_no_improvement += 1
                        logger.warning(f"Failed ! reason : {e}")
                    logger.debug(
                        f"{current_nb_iteration_no_improvement} / {nb_iteration_no_improvement}"
                    )
                    if (
                        current_nb_iteration_no_improvement
                        > nb_iteration_no_improvement
                    ):
                        logger.info("Finish LNS with maximum no improvement iteration ")
                        break
                # end of step callback: stopping?
                if skip_first_iteration:
                    step = iteration
                else:
                    step = iteration + 1
                stopping = callbacks_list.on_step_end(
                    step=step, res=store_lns, solver=self
                )
                if stopping:
                    break

        # end of solve callback
        callbacks_list.on_solve_end(res=store_lns, solver=self)
        return store_lns

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
        return self.solve_lns(
            parameters_cp=parameters_cp,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            skip_first_iteration=skip_first_iteration,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            callbacks=callbacks,
            **kwargs,
        )


class LNS_CPlex(SolverDO):
    def __init__(
        self,
        problem: Problem,
        cp_solver: CPSolver,
        initial_solution_provider: InitialSolution,
        constraint_handler: ConstraintHandler,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.cp_solver = cp_solver
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.post_process_solution: PostProcessSolution
        if post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()
        else:
            self.post_process_solution = post_process_solution

    def solve_lns(
        self,
        parameters_cp: ParametersCP,
        nb_iteration_lns: int,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        sense = self.params_objective_function.sense_function
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2 * nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            store_lns = self.post_process_solution.build_other_solution(store_lns)
            init_solution, objective = store_lns.get_best_solution_fit()
            satisfy = self.problem.satisfy(init_solution)
            logger.debug(f"Satisfy Initial solution {satisfy}")
            try:
                logger.debug(
                    f"Nb task preempted = {init_solution.get_nb_task_preemption()}"
                )
                logger.debug(f"Nb max preemption = {init_solution.get_max_preempted()}")
            except:
                pass
            best_objective = objective
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=0, res=store_lns, solver=self)
        else:
            best_objective = (
                float("inf") if sense == ModeOptim.MINIMIZATION else -float("inf")
            )
            store_lns = None
            stopping = False

        result_store: ResultStorage
        if not stopping:
            for iteration in range(nb_iteration_lns):
                logger.debug(
                    f"Starting iteration n° {iteration} current objective {best_objective}"
                )
                if iteration == 0 and not skip_first_iteration or iteration >= 1:
                    constraint_iterable = (
                        self.constraint_handler.adding_constraint_from_results_store(
                            cp_solver=self.cp_solver,
                            child_instance=None,
                            result_storage=store_lns,
                            last_result_store=store_lns
                            if iteration == 0
                            else result_store,
                        )
                    )

                try:
                    if iteration == 0:
                        p = parameters_cp.default()
                        p.time_limit = parameters_cp.time_limit_iter0
                        result_store = self.cp_solver.solve(parameters_cp=parameters_cp)
                    else:
                        result_store = self.cp_solver.solve(parameters_cp=parameters_cp)
                    logger.debug(f"iteration n° {iteration} Solved !!!")
                    if len(result_store.list_solution_fits) > 0:
                        logger.debug("Solved !!!")
                        bsol, fit = result_store.get_best_solution_fit()
                        logger.debug(f"Fitness Before = {fit}")
                        logger.debug(
                            f"Satisfaction Before = {self.problem.satisfy(bsol)}"
                        )
                        logger.debug("Post Process..")
                        result_store = self.post_process_solution.build_other_solution(
                            result_store
                        )
                        bsol, fit = result_store.get_best_solution_fit()
                        logger.debug(f"Satisfy after : {self.problem.satisfy(bsol)}")
                        if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                            if fit > best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_objective = fit
                        elif sense == ModeOptim.MAXIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        elif sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                            if fit < best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_objective = fit
                        elif sense == ModeOptim.MINIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        if skip_first_iteration and iteration == 0:
                            store_lns = result_store
                        else:
                            for s, f in list(result_store.list_solution_fits):
                                store_lns.add_solution(solution=s, fitness=f)
                    else:
                        current_nb_iteration_no_improvement += 1
                except Exception as e:
                    current_nb_iteration_no_improvement += 1
                    logger.warning(f"Failed ! reason : {e}")
                logger.debug(
                    f"{current_nb_iteration_no_improvement} / {nb_iteration_no_improvement}"
                )
                if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                    logger.info("Finish LNS with maximum no improvement iteration ")
                    break

                # end of step callback: stopping?
                if skip_first_iteration:
                    step = iteration
                else:
                    step = iteration + 1
                stopping = callbacks_list.on_step_end(
                    step=step, res=store_lns, solver=self
                )
                if stopping:
                    break

        # end of solve callback
        callbacks_list.on_solve_end(res=store_lns, solver=self)
        return store_lns

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return self.solve_lns(
            parameters_cp=parameters_cp,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            skip_first_iteration=skip_first_iteration,
            callbacks=callbacks,
            **kwargs,
        )


class ConstraintStatus(TypedDict):
    nb_usage: int
    nb_improvement: int
    name: str


class ConstraintHandlerMix(ConstraintHandler):
    def __init__(
        self,
        problem: Problem,
        list_constraints_handler: List[ConstraintHandler],
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
        cp_solver: CPSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
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
            cp_solver, child_instance, result_storage, last_result_store
        )

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CPSolver,
        child_instance: Instance,
        previous_constraints: Iterable[Any],
    ) -> None:
        pass
