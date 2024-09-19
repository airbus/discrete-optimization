import logging
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    ListHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class SequentialMetasolver(SolverDO):
    """Sequential metasolver.

    The problem will be solved sequentially, each subsolver being warm started by the previous one.
    Therefore each subsolver must inherit from WarmstartMixin, except the first one.

    """

    hyperparameters = [
        SubBrickHyperparameter(  # first subsolver: can be non-warmstartable
            name="subsolver_0",
            choices=[],  # to update in an optuna study via suggest_optuna_kwargs_by_name_by_solver
        ),
        ListHyperparameter(
            name="next_subsolvers",
            hyperparameter_template=SubBrickHyperparameter(
                "subsolver",
                choices=[],  # to update in an optuna study via suggest_optuna_kwargs_by_name_by_solver
            ),
            length_high=4,  # to update in optuna study via suggest_optuna_kwargs_by_name_by_solver
            length_low=0,
            numbering_start=1,  # the first subsolver of this list will be named "subsolver_1"
            # to avoid overriding "subsolver_0" defined by the first hyperparameter.
        ),
    ]

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        list_subbricks: Optional[list[SubBrick]] = None,
        **kwargs,
    ):
        """

        Args:
            list_subbricks: list of subsolvers class and kwargs to be used sequentially

        """
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        if list_subbricks is None:
            list_subbricks = []
        if "subsolver_0" in kwargs:  # optuna first subsolver choice
            list_subbricks = [kwargs["subsolver_0"]] + list_subbricks
        if "next_subsolvers" in kwargs:  # optuna last subsolvers choice
            list_subbricks = list_subbricks + kwargs["next_subsolvers"]
        self.list_subbricks = list_subbricks
        self.nb_solvers = len(list_subbricks)

        # checks
        if len(self.list_subbricks) == 0:
            raise ValueError("list_subbricks must contain at least one subbrick.")
        for i_subbrick, subbrick in enumerate(self.list_subbricks):
            if not issubclass(subbrick.cls, SolverDO):
                raise ValueError("Each subsolver must inherit SolverDO.")
            if i_subbrick > 0 and not issubclass(subbrick.cls, WarmstartMixin):
                raise ValueError(
                    "Each subsolver except the first one must inherit WarmstartMixin."
                )

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        # iterate over next solvers
        res_tot = self.create_result_storage()
        for i_subbrick, subbrick in enumerate(self.list_subbricks):
            # get previous best solution
            if i_subbrick > 0:
                start_solution = res.get_best_solution()
            else:
                start_solution = None
            # update subbrick kwargs with functions of previous solutions
            kwargs_updated = dict(subbrick.kwargs)
            if subbrick.kwargs_from_solution is not None:
                kwargs_updated.update(
                    {
                        k: fun(start_solution)
                        for k, fun in subbrick.kwargs_from_solution.items()
                    }
                )
            # init subsolver
            subsolver: SolverDO = subbrick.cls(problem=self.problem, **kwargs_updated)
            subsolver.init_model(**kwargs_updated)
            # warm start
            if start_solution is not None:
                subsolver.set_warm_start(start_solution)
            # solve
            res = subsolver.solve(**kwargs_updated)
            # store results
            res_tot.extend(res)

            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(
                step=i_subbrick, res=res_tot, solver=self
            )
            if len(res) == 0:
                # no solution => warning + stopping if first subsolver
                logger.warning(f"Subsolver #{i_subbrick} did not find any solution.")
                if i_subbrick == 0:
                    stopping = True
            if stopping:
                break

        # end of solve callback
        callbacks_list.on_solve_end(res=res_tot, solver=self)
        return res_tot
