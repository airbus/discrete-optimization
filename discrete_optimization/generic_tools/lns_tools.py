#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import contextlib
import logging
from abc import abstractmethod
from typing import Any, Iterable, List, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class ConstraintHandler(Hyperparametrizable):
    @abstractmethod
    def adding_constraint_from_results_store(
        self, solver: SolverDO, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        ...

    @abstractmethod
    def remove_constraints_from_previous_iteration(
        self,
        solver: SolverDO,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        ...


class InitialSolution(Hyperparametrizable):
    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        ...


class InitialSolutionFromSolver(InitialSolution):
    def __init__(self, solver: SolverDO, **kwargs: Any):
        self.solver = solver
        self.dict = kwargs

    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        return self.solver.solve(**self.dict)


class TrivialInitialSolution(InitialSolution):
    def __init__(self, solution: ResultStorage, **kwargs: Any):
        self.solution = solution
        self.dict = kwargs

    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        return self.solution


class PostProcessSolution(Hyperparametrizable):
    # From solution from MIP or CP you can build other solution.
    # Here you can have many different approaches:
    # if solution from mip/cp are not feasible you can code a repair function
    # you can also do mall changes (filling gap in a schedule) to try to improve the solution
    # you can also run algorithms from the new found solution.
    @abstractmethod
    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        ...


class TrivialPostProcessSolution(PostProcessSolution):
    def __init__(self, **kwargs):
        ...

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        return result_storage


class BaseLNS(SolverDO):
    """Base class for Large Neighborhood Search solvers."""

    subsolver: SolverDO
    """Sub-solver used by this lns solver at each iteration."""

    constraint_handler: ConstraintHandler
    initial_solution_provider: Optional[InitialSolution]
    post_process_solution: Optional[PostProcessSolution]

    hyperparameters = [
        SubBrickHyperparameter("subsolver_cls", choices=[], default=None),
        SubBrickKwargsHyperparameter(
            "subsolver_kwargs", subbrick_hyperparameter="subsolver_cls"
        ),
        SubBrickHyperparameter(
            "initial_solution_provider_cls",
            choices=[],
            default=None,
            depends_on=("skip_first_iteration", [False]),
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
        subsolver: Optional[SolverDO] = None,
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

        if subsolver is None:
            if kwargs["subsolver_kwargs"] is None:
                subsolver_kwargs = kwargs
            else:
                subsolver_kwargs = kwargs["subsolver_kwargs"]
            if kwargs["subsolver_cls"] is None:
                if "build_default_subsolver" in kwargs:
                    subsolver = kwargs["build_default_subsolver"](
                        self.problem, **subsolver_kwargs
                    )
                else:
                    raise ValueError(
                        "`subsolver_cls` cannot be None if `subsolver` is not specified."
                    )
            else:
                subsolver_cls = kwargs["subsolver_cls"]
                subsolver = subsolver_cls(problem=self.problem, **subsolver_kwargs)
                subsolver.init_model(**subsolver_kwargs)
        self.subsolver = subsolver

        if constraint_handler is None:
            if kwargs["constraint_handler_kwargs"] is None:
                constraint_handler_kwargs = kwargs
            else:
                constraint_handler_kwargs = kwargs["constraint_handler_kwargs"]
            if kwargs["constraint_handler_cls"] is None:
                if "build_default_contraint_handler" in kwargs:
                    constraint_handler = kwargs["build_default_contraint_handler"](
                        self.problem, **constraint_handler_kwargs
                    )
                else:
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
                if "build_default_post_process_solution" in kwargs:
                    post_process_solution = kwargs[
                        "build_default_post_process_solution"
                    ](
                        self.problem,
                        self.params_objective_function,
                        **post_process_solution_kwargs,
                    )
                else:
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
                if "build_default_initial_solution_provider" in kwargs:
                    initial_solution_provider = kwargs[
                        "build_default_initial_solution_provider"
                    ](
                        self.problem,
                        self.params_objective_function,
                        **initial_solution_provider_kwargs,
                    )
                else:
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

    def create_submodel(self) -> contextlib.AbstractContextManager:
        return _dummy_contextmanager()

    def solve_with_subsolver(
        self, iteration: int, instance: Any, **kwargs: Any
    ) -> ResultStorage:
        """Solve with the subsolver.

        Args:
            iteration: current iteration number
            instance: current instance of the model (only for minizinc)
            **kwargs: kwargs passed to this lns solver `solve()`

        Returns:

        """
        return self.subsolver.solve(**kwargs)

    def solve(
        self,
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

        self.init_model(**kwargs)

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
        lsn_contraints: Optional[Iterable[Any]] = None
        if not stopping:
            for iteration in range(nb_iteration_lns):
                logger.info(
                    f"Starting iteration n° {iteration} current objective {best_objective}"
                )
                with self.create_submodel() as child:
                    if iteration == 0 and not skip_first_iteration or iteration >= 1:
                        lsn_contraints = self.constraint_handler.adding_constraint_from_results_store(
                            solver=self.subsolver,
                            child_instance=child,
                            result_storage=store_lns,
                            last_result_store=store_lns
                            if iteration == 0
                            else result_store,
                        )
                    try:
                        result_store = self.solve_with_subsolver(
                            iteration=0, instance=child, **kwargs
                        )
                        logger.info(f"iteration n° {iteration} Solved !!!")
                        if hasattr(self.subsolver, "status_solver"):
                            logger.info(self.subsolver.status_solver)
                        if len(result_store) > 0:
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
                                for s, f in list(result_store):
                                    store_lns.append((s, f))
                        else:
                            current_nb_iteration_no_improvement += 1
                        if (
                            skip_first_iteration
                            and self.subsolver.is_optimal()
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
                if lsn_contraints is not None:
                    self.constraint_handler.remove_constraints_from_previous_iteration(
                        solver=self.subsolver, previous_constraints=lsn_contraints
                    )
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


@contextlib.contextmanager
def _dummy_contextmanager():
    yield None
