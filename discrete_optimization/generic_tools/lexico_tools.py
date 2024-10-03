#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tools for lexicographic optimization."""
import logging
from collections.abc import Iterable
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class LexicoSolver(SolverDO):

    subsolver: SolverDO

    def __init__(
        self,
        subsolver: Optional[SolverDO],
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        # ensure no aggregation performed
        if params_objective_function is None:
            params_objective_function = get_default_objective_setup(problem)
        params_objective_function.objective_handling = ObjectiveHandling.MULTI_OBJ

        # SolverDO init after updating params_objective_function
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )

        # get subsolver (directly or from its hyperparameters to allow optuna tuning)
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

        # check compatibility with lexico optimization
        if not subsolver.implements_lexico_api():
            logger.warning(
                "The chosen subsolver may not be implementing the api needed by LexicoSolver!"
            )

    def init_model(self, **kwargs: Any) -> None:
        self.subsolver.init_model(**kwargs)

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        objectives: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        if objectives is None:
            objectives = self.subsolver.get_lexico_objectives_available()

        self._objectives = objectives
        res = ResultStorage(
            mode_optim=self.params_objective_function.sense_function,
            list_solution_fits=[],
        )
        if "subsolver_callbacks" not in kwargs:
            kwargs["subsolver_callbacks"] = None
        for i_obj, obj in enumerate(objectives):

            # log
            logger.debug(f"Optimizing on {obj}")

            # optimize next objective
            self.subsolver.set_lexico_objective(obj)
            res.extend(
                self.subsolver.solve(callbacks=kwargs["subsolver_callbacks"], **kwargs)
            )
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=i_obj, res=res, solver=self)
            if stopping:
                break

            # add constraint on current objective for next one
            fit = self.subsolver.get_lexico_objective_value(obj, res)
            logger.debug(f"Found {fit} when optimizing {obj}")
            self.subsolver.add_lexico_constraint(obj, fit)

        # end of solve callback
        callbacks_list.on_solve_end(res=res, solver=self)

        return res
