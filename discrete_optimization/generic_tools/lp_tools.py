#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import copy
import datetime
import logging
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional, Union

import mip
from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    GRB = gurobipy.GRB

try:
    import docplex
    from docplex.mp.progress import SolutionListener
    from docplex.mp.solution import SolveSolution
except ImportError:
    cplex_available = False
else:
    cplex_available = True


logger = logging.getLogger(__name__)


class MilpSolverName(Enum):
    CBC = 0
    GRB = 1


map_solver = {MilpSolverName.GRB: mip.GRB, MilpSolverName.CBC: mip.CBC}


class ParametersMilp:
    def __init__(
        self,
        pool_solutions: int,
        mip_gap_abs: float,
        mip_gap: float,
        retrieve_all_solution: bool,
        pool_search_mode: int = 0,
    ):
        self.pool_solutions = pool_solutions
        self.mip_gap_abs = mip_gap_abs
        self.mip_gap = mip_gap
        self.retrieve_all_solution = retrieve_all_solution
        self.pool_search_mode = pool_search_mode

    @staticmethod
    def default() -> "ParametersMilp":
        return ParametersMilp(
            pool_solutions=10000,
            mip_gap_abs=0.0000001,
            mip_gap=0.000001,
            retrieve_all_solution=True,
        )


class MilpSolver(SolverDO):
    model: Optional[Any]

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        ...

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        """Retrieve solutions found by internal solver.

        Args:
            parameters_milp:

        Returns:

        """
        if parameters_milp.retrieve_all_solution:
            n_solutions = self.nb_solutions
        else:
            n_solutions = 1
        list_solution_fits: list[tuple[Solution, Union[float, TupleFitness]]] = []
        for i in range(n_solutions):
            solution = self.retrieve_ith_solution(i=i)
            fit = self.aggreg_from_sol(solution)
            list_solution_fits.append((solution, fit))
        return self.create_result_storage(
            list_solution_fits,
        )

    @abstractmethod
    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """Retrieve current solution from internal gurobi solution.

        This converts internal gurobi solution into a discrete-optimization Solution.
        This method can be called after the solve in `retrieve_solutions()`
        or during solve within a gurobi/pymilp/cplex callback. The difference will be the
        `get_var_value_for_current_solution` and `get_obj_value_for_current_solution` callables passed.

        Args:
            get_var_value_for_current_solution: function extracting the value of the given variable for the current solution
                will be different when inside a callback or after the solve is finished
            get_obj_value_for_current_solution: function extracting the value of the objective for the current solution.

        Returns:
            the converted solution at d-o format

        """
        ...

    def retrieve_ith_solution(self, i: int) -> Solution:
        """Retrieve i-th solution from internal milp model.

        Args:
            i:

        Returns:

        """
        get_var_value_for_current_solution = (
            lambda var: self.get_var_value_for_ith_solution(var=var, i=i)
        )
        get_obj_value_for_current_solution = (
            lambda: self.get_obj_value_for_ith_solution(i=i)
        )
        return self.retrieve_current_solution(
            get_var_value_for_current_solution=get_var_value_for_current_solution,
            get_obj_value_for_current_solution=get_obj_value_for_current_solution,
        )

    @abstractmethod
    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_milp: Optional[ParametersMilp] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        ...

    @abstractmethod
    def get_var_value_for_ith_solution(self, var: Any, i: int) -> float:
        """Get value for i-th solution of a given variable."""

    @abstractmethod
    def get_obj_value_for_ith_solution(self, i: int) -> float:
        """Get objective value for i-th solution."""

    @property
    @abstractmethod
    def nb_solutions(self) -> int:
        """Number of solutions found by the solver."""


class PymipMilpSolver(MilpSolver):
    """Milp solver wrapping a solver from pymip library."""

    model: Optional[mip.Model] = None
    milp_solver_name: MilpSolverName
    solver_name: str

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        milp_solver_name: MilpSolverName = MilpSolverName.CBC,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.set_milp_solver_name(milp_solver_name=milp_solver_name)

    def set_milp_solver_name(self, milp_solver_name: MilpSolverName):
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        self.optimize_model(
            parameters_milp=parameters_milp, time_limit=time_limit, **kwargs
        )
        return self.retrieve_solutions(parameters_milp=parameters_milp)

    def prepare_model(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> None:
        """Set Gurobi Model parameters according to parameters_milp"""
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.model.max_mip_gap = parameters_milp.mip_gap
        self.model.max_mip_gap_abs = parameters_milp.mip_gap_abs
        self.model.sol_pool_size = parameters_milp.pool_solutions
        if time_limit is not None:
            self.model.max_seconds = time_limit

    def optimize_model(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> None:
        """Optimize the mip Model.

        The solutions are yet to be retrieved via `self.retrieve_solutions()`.

        """
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.prepare_model(
            parameters_milp=parameters_milp, time_limit=time_limit, **kwargs
        )
        self.model.optimize()

        logger.info(f"Solver found {self.model.num_solutions} solutions")
        logger.info(f"Objective : {self.model.objective_value}")

    def get_var_value_for_ith_solution(self, var: mip.Var, i: int) -> float:  # type: ignore # avoid isinstance checks for efficiency
        """Get value for i-th solution of a given variable."""
        return var.xi(i)

    def get_obj_value_for_ith_solution(self, i: int) -> float:
        """Get objective value for i-th solution."""
        if self.model is None:  # for mypy
            raise RuntimeError(
                "self.model should not be None when calling this method."
            )
        return self.model.objective_values[i]

    @property
    def nb_solutions(self) -> int:
        """Number of solutions found by the solver."""
        if self.model is None:
            return 0
        else:
            return self.model.num_solutions


class OrtoolsMathOptMilpSolver(MilpSolver, WarmstartMixin):
    """Milp solver wrapping a solver from pymip library."""

    hyperparameters = [
        EnumHyperparameter(
            name="mathopt_solver_type",
            enum=mathopt.SolverType,
        )
    ]

    solution_hint: Optional[mathopt.SolutionHint] = None

    @abstractmethod
    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        See https://or-tools.github.io/docs/pdoc/ortools/math_opt/python/model_parameters.html#SolutionHint
        for more information.

        Override it in subclasses to have a proper warm start.

        """
        ...

    def convert_to_dual_values(
        self, solution: Solution
    ) -> dict[mathopt.LinearConstraint, float]:
        """Convert a solution to a mapping between model contraints and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.dual_values.
        See https://or-tools.github.io/docs/pdoc/ortools/math_opt/python/model_parameters.html#SolutionHint
        for more information.
        Generally MIP solvers do not need this part, but LP solvers do.

        """
        return dict()

    def set_warm_start(self, solution: Solution) -> None:
        self.solution_hint = mathopt.SolutionHint(
            variable_values=self.convert_to_variable_values(solution),
            dual_values=self.convert_to_dual_values(solution),
        )

    def get_var_value_for_ith_solution(self, var: Any, i: int) -> float:
        raise NotImplementedError()

    def get_obj_value_for_ith_solution(self, i: int) -> float:
        raise NotImplementedError()

    @property
    def nb_solutions(self) -> int:
        raise NotImplementedError()

    model: Optional[mathopt.Model] = None
    termination: mathopt.Termination
    early_stopping_exception: Optional[Exception] = None

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_milp: Optional[ParametersMilp] = None,
        mathopt_solver_type: mathopt.SolverType = mathopt.SolverType.CP_SAT,
        time_limit: Optional[float] = 30.0,
        mathopt_enable_output: bool = False,
        mathopt_model_parameters: Optional[mathopt.ModelSolveParameters] = None,
        mathopt_additional_solve_parameters: Optional[mathopt.SolveParameters] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve with OR-Tools MathOpt API

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_milp: parameters passed to the MILP solver
            mathopt_solver_type: underlying solver type to use.
                Passed as `solver_type` to `mathopt.solve()`
            time_limit: the solve process stops after this time limit (in seconds).
                If None, no time limit is applied.
            mathopt_enable_output: turn on mathopt logging
            mathopt_model_parameters: passed to `mathopt.solve()` as `model_params`
            mathopt_additional_solve_parameters: passed to `mathopt.solve()` as `params`,
                except that parameters defined by above `time_limit` and `parameters_milp`
                will be overriden by them.
            **kwargs: passed to init_model() if model not yet existing

        Returns:

        """
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        # wrap user callback in a mathopt callback
        mathopt_cb = MathOptCallback(do_solver=self, callback=callbacks_list)

        # optimize
        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            mathopt_solver_type=mathopt_solver_type,
            mathopt_cb=mathopt_cb,
            mathopt_enable_output=mathopt_enable_output,
            mathopt_model_parameters=mathopt_model_parameters,
            mathopt_additional_solve_parameters=mathopt_additional_solve_parameters,
            **kwargs,
        )

        # raise potential exception found during callback (useful for optuna pruning, and debugging)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception

        # get result storage
        res = mathopt_cb.res

        # callback: solve end
        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def optimize_model(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        mathopt_solver_type: mathopt.SolverType = mathopt.SolverType.CP_SAT,
        time_limit: Optional[float] = 30.0,
        mathopt_cb: Optional[mathopt.SolveCallback] = None,
        mathopt_enable_output: bool = False,
        mathopt_model_parameters: Optional[mathopt.ModelSolveParameters] = None,
        mathopt_additional_solve_parameters: Optional[mathopt.SolveParameters] = None,
        **kwargs: Any,
    ) -> mathopt.SolveResult:
        """

        Args:
            parameters_milp: parameters for the milp solver
            mathopt_solver_type: underlying solver type to use.
                Passed as `solver_type` to `mathopt.solve()`
            time_limit: the solve process stops after this time limit (in seconds).
                If None, no time limit is applied.
            mathopt_cb: a mathopt callback passed to `mathopt.solve()` called at each new solution found
            mathopt_enable_output: turn on mathopt logging
            mathopt_model_parameters: passed to `mathopt.solve()` as `model_params`
            mathopt_additional_solve_parameters: passed to `mathopt.solve()` as `params`,
                except that parameters defined by above `time_limit`, `parameters_milp`, and `mathopt_enable_output`
                will be overriden by them.
            **kwargs: passed to init_model() if model not yet existing

        Returns:

        """
        """Optimize the mip Model.

        The solutions are yet to be retrieved via `self.retrieve_solutions()`.

        Args:
            time_limit

        """
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )

        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()

        # solve parameters
        if mathopt_additional_solve_parameters is None:
            params = mathopt.SolveParameters()
        else:
            params = copy.deepcopy(mathopt_additional_solve_parameters)
        params.time_limit = datetime.timedelta(seconds=time_limit)
        params.absolute_gap_tolerance = parameters_milp.mip_gap_abs
        params.relative_gap_tolerance = parameters_milp.mip_gap
        params.enable_output = mathopt_enable_output
        if mathopt_solver_type != mathopt.SolverType.HIGHS:
            # solution_pool_size not supported for HIGHS solver
            params.solution_pool_size = parameters_milp.pool_solutions

        # model specific solve parameters
        if mathopt_model_parameters is None:
            model_params = mathopt.ModelSolveParameters()
        else:
            model_params = copy.deepcopy(mathopt_model_parameters)
        if self.solution_hint is not None:
            # warm start: add a solution hint corresponding to the warm start
            model_params.solution_hints = [
                self.solution_hint
            ] + model_params.solution_hints

        callback_reg = mathopt.CallbackRegistration(events={mathopt.Event.MIP_SOLUTION})
        mathopt_res = mathopt.solve(
            self.model,
            solver_type=mathopt_solver_type,
            params=params,
            model_params=model_params,
            callback_reg=callback_reg,
            cb=mathopt_cb,
        )
        self.termination = mathopt_res.termination
        logger.info(f"Solver found {len(mathopt_res.solutions)} solutions")
        if mathopt_res.termination.reason in [
            mathopt.TerminationReason.OPTIMAL,
            mathopt.TerminationReason.FEASIBLE,
        ]:
            logger.info(f"Objective : {mathopt_res.objective_value()}")
        return mathopt_res


def _mathopt_cb_get_obj_value_for_current_solution():
    raise RuntimeError("Cannot retrieve objective!")


class MathOptCallback:
    def __init__(self, do_solver: OrtoolsMathOptMilpSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def __call__(self, callback_data: mathopt.CallbackData) -> mathopt.CallbackResult:
        cb_sol = callback_data.solution
        try:
            # retrieve and store new solution
            sol = self.do_solver.retrieve_current_solution(
                get_var_value_for_current_solution=lambda var: cb_sol[var],
                get_obj_value_for_current_solution=_mathopt_cb_get_obj_value_for_current_solution,
            )
            fit = self.do_solver.aggreg_from_sol(sol)
            self.res.append((sol, fit))
            self.nb_solutions += 1
            # end of step callback: stopping?
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            # catch exceptions because gurobi ignore them and do not stop solving
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        return mathopt.CallbackResult(terminate=stopping)


class GurobiMilpSolver(MilpSolver):
    """Milp solver wrapping a solver from gurobi library."""

    model: Optional["gurobipy.Model"] = None
    early_stopping_exception: Optional[Exception] = None

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> ResultStorage:
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.prepare_model(
            parameters_milp=parameters_milp, time_limit=time_limit, **kwargs
        )

        # wrap user callback in a gurobi callback
        gurobi_callback = GurobiCallback(do_solver=self, callback=callbacks_list)
        self.model.optimize(gurobi_callback)
        # raise potential exception found during callback (useful for optuna pruning, and debugging)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        # get result storage
        res = gurobi_callback.res

        # callback: solve end
        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def prepare_model(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> None:
        """Set Gurobi Model parameters according to parameters_milp"""
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if time_limit is not None:
            self.model.setParam(gurobipy.GRB.Param.TimeLimit, time_limit)
        self.model.setParam(gurobipy.GRB.Param.MIPGapAbs, parameters_milp.mip_gap_abs)
        self.model.setParam(gurobipy.GRB.Param.MIPGap, parameters_milp.mip_gap)
        self.model.setParam(
            gurobipy.GRB.Param.PoolSolutions, parameters_milp.pool_solutions
        )
        self.model.setParam("PoolSearchMode", parameters_milp.pool_search_mode)

    def optimize_model(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> None:
        """Optimize the Gurobi Model.

        The solutions are yet to be retrieved via `self.retrieve_solutions()`.
        No callbacks are passed to the internal solver, and no result_storage is created

        """
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.prepare_model(
            parameters_milp=parameters_milp, time_limit=time_limit, **kwargs
        )
        self.model.optimize()

        logger.info(f"Problem has {self.model.NumObj} objectives")
        logger.info(f"Solver found {self.model.SolCount} solutions")
        logger.info(f"Objective : {self.model.getObjective().getValue()}")

    def get_var_value_for_ith_solution(self, var: "gurobipy.Var", i: int):  # type: ignore # avoid isinstance checks for efficiency
        """Get value for i-th solution of a given variable."""
        if self.model is None:  # for mypy
            raise RuntimeError(
                "self.model should not be None when calling this method."
            )
        self.model.params.SolutionNumber = i
        return var.getAttr("Xn")

    def get_obj_value_for_ith_solution(self, i: int) -> float:
        """Get objective value for i-th solution."""
        if self.model is None:  # for mypy
            raise RuntimeError(
                "self.model should not be None when calling this method."
            )
        self.model.params.SolutionNumber = i
        return self.model.getAttr("PoolObjVal")

    @property
    def nb_solutions(self) -> int:
        """Number of solutions found by the solver."""
        if self.model is None:
            return 0
        else:
            return self.model.SolCount


class GurobiCallback:
    def __init__(self, do_solver: GurobiMilpSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def __call__(self, model, where) -> None:
        if where == GRB.Callback.MIPSOL:
            try:
                # retrieve and store new solution
                sol = self.do_solver.retrieve_current_solution(
                    get_var_value_for_current_solution=model.cbGetSolution,
                    get_obj_value_for_current_solution=lambda: model.cbGet(
                        GRB.Callback.MIPSOL_OBJ
                    ),
                )
                fit = self.do_solver.aggreg_from_sol(sol)
                self.res.append((sol, fit))
                self.nb_solutions += 1
                # end of step callback: stopping?
                stopping = self.callback.on_step_end(
                    step=self.nb_solutions, res=self.res, solver=self.do_solver
                )
            except Exception as e:
                # catch exceptions because gurobi ignore them and do not stop solving
                self.do_solver.early_stopping_exception = e
                stopping = True
            else:
                if stopping:
                    self.do_solver.early_stopping_exception = SolveEarlyStop(
                        f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                    )
            if stopping:
                model.terminate()


class CplexMilpSolver(MilpSolver):
    model: Optional["docplex.mp.model.Model"]
    results_solve: Optional[list["SolveSolution"]]

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> ResultStorage:
        if not cplex_available:
            logger.debug(
                "One or several docplex didn't work, therefore your script might crash."
            )
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if time_limit is not None:
            self.model.time_limit = time_limit
        self.model.parameters.mip.tolerances.mipgap = parameters_milp.mip_gap
        listener = None
        if parameters_milp.retrieve_all_solution:

            class SolutionStorage(SolutionListener):
                def __init__(self):
                    super().__init__()
                    self.intermediate_solutions = []

                def notify_solution(self, sol):
                    self.intermediate_solutions += [sol]

            listener = SolutionStorage()
            self.model.add_progress_listener(listener)
        results: "SolveSolution" = self.model.solve(log_output=True)
        if listener is None:
            self.results_solve = [results]
        else:
            self.results_solve = listener.intermediate_solutions + [results]
        # logger.info(f"Solver found {results.get()} solutions")
        logger.info(f"Objective : {self.results_solve[-1].get_objective_value()}")
        return self.retrieve_solutions(parameters_milp=parameters_milp)

    def get_var_value_for_ith_solution(
        self, var: "docplex.mp.dvar.Var", i: int
    ) -> float:
        return self.results_solve[i].get_var_value(var)

    def get_obj_value_for_ith_solution(self, i: int) -> float:
        return self.results_solve[i]

    @property
    def nb_solutions(self) -> int:
        """Number of solutions found by the solver."""
        if self.results_solve is None:
            return 0
        else:
            return len(self.results_solve)
