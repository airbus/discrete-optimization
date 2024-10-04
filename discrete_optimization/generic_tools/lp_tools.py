#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import datetime
import logging
import math
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

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
from discrete_optimization.generic_tools.do_solver import (
    SolverDO,
    StatusSolver,
    WarmstartMixin,
)
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
    map_gurobi_status_to_do_status: dict[int, StatusSolver] = defaultdict(
        lambda: StatusSolver.UNKNOWN,
        {
            gurobipy.GRB.status.OPTIMAL: StatusSolver.OPTIMAL,
            gurobipy.GRB.status.INFEASIBLE: StatusSolver.UNSATISFIABLE,
            gurobipy.GRB.status.SUBOPTIMAL: StatusSolver.SATISFIED,
        },
    )

try:
    import docplex
    from docplex.mp.progress import SolutionListener
    from docplex.mp.solution import SolveSolution
except ImportError:
    cplex_available = False
else:
    cplex_available = True


logger = logging.getLogger(__name__)

# types aliases
if gurobi_available:
    VariableType = Union[gurobipy.Var, mathopt.Variable]
    ConstraintType = Union[
        gurobipy.Constr,
        gurobipy.MConstr,
        gurobipy.QConstr,
        gurobipy.GenConstr,
        mathopt.LinearConstraint,
    ]

else:
    VariableType = Union[mathopt.Variable]
    ConstraintType = Union[mathopt.LinearConstraint]


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

    @staticmethod
    @abstractmethod
    def create_empty_model(name: str = "") -> Any:
        """Generate an empty milp model.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        """
        ...

    @abstractmethod
    def add_linear_constraint(self, expr: Any, name: str = "") -> Any:
        """Add a linear constraint to the model.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        """
        ...

    @abstractmethod
    def add_binary_variable(self, name: str = "") -> Any:
        """Add a binary variable to the model.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        """
        ...

    @abstractmethod
    def add_integer_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> Any:
        """Add an integer variable to the model.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        Args:
            lb: lower bound
            ub: upper bound

        """
        ...

    @abstractmethod
    def add_continuous_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> Any:
        """Add a continuous variable to the model.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        Args:
            lb: lower bound
            ub: upper bound

        """
        ...

    @abstractmethod
    def set_model_objective(self, expr: Any, minimize: bool) -> None:
        """Define the model objective.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        Args:
            expr:
            minimize: if True, objective will be minimized, else maximized

        Returns:

        """
        ...

    @staticmethod
    @abstractmethod
    def construct_linear_sum(expr: Iterable) -> Any:
        """Generate a linear sum (with variables) ready for the internal model."""
        ...


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
        """Make the solver warm start from the given solution."""
        self.set_warm_start_from_values(
            variable_values=self.convert_to_variable_values(solution),
            dual_values=self.convert_to_dual_values(solution),
        )

    def set_warm_start_from_values(
        self,
        variable_values: dict[mathopt.Variable, float],
        dual_values: Optional[dict[mathopt.LinearConstraint, float]] = None,
    ) -> None:
        if dual_values is None:
            dual_values = {}
        self.solution_hint = mathopt.SolutionHint(
            variable_values=variable_values,
            dual_values=dual_values,
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
        self.status_solver = map_mathopt_status_to_do_status[self.termination.reason]

        logger.info(f"Solver found {len(mathopt_res.solutions)} solutions")
        if mathopt_res.termination.reason in [
            mathopt.TerminationReason.OPTIMAL,
            mathopt.TerminationReason.FEASIBLE,
        ]:
            logger.info(f"Objective : {mathopt_res.objective_value()}")
        return mathopt_res

    @staticmethod
    def create_empty_model(name: str = "") -> mathopt.Model:
        return mathopt.Model(name=name)

    def add_linear_constraint(
        self, expr: Any, name: str = ""
    ) -> mathopt.LinearConstraint:
        return self.model.add_linear_constraint(expr, name=name)

    def add_binary_variable(self, name: str = "") -> mathopt.Variable:
        return self.model.add_binary_variable(name=name)

    def add_integer_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> mathopt.Variable:
        return self.model.add_integer_variable(lb=lb, ub=ub, name=name)

    def add_continuous_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> mathopt.Variable:
        return self.model.add_variable(lb=lb, ub=ub, is_integer=False, name=name)

    def set_model_objective(self, expr: Any, minimize: bool) -> None:
        """Define the model objective.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        Args:
            expr:
            minimize: if True, objective will be minimized, else maximized

        Returns:

        """
        self.model.set_objective(expr, is_maximize=not minimize)

    @staticmethod
    def construct_linear_sum(expr: Iterable) -> Any:
        """Generate a linear sum (with variables) ready for the internal model."""
        return mathopt.LinearSum(expr)


map_mathopt_status_to_do_status: dict[mathopt.TerminationReason, StatusSolver] = {
    mathopt.TerminationReason.OPTIMAL: StatusSolver.OPTIMAL,
    mathopt.TerminationReason.INFEASIBLE: StatusSolver.UNSATISFIABLE,
    mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.UNBOUNDED: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.FEASIBLE: StatusSolver.SATISFIED,
    mathopt.TerminationReason.NO_SOLUTION_FOUND: StatusSolver.UNSATISFIABLE,
    mathopt.TerminationReason.IMPRECISE: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.NUMERICAL_ERROR: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.OTHER_ERROR: StatusSolver.UNKNOWN,
}


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


class GurobiMilpSolver(MilpSolver, WarmstartMixin):
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
        self.status_solver = map_gurobi_status_to_do_status[self.model.Status]

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

    @staticmethod
    def create_empty_model(name: str = "") -> gurobipy.Model:
        return gurobipy.Model(name=name)

    def add_linear_constraint(self, expr: Any, name: str = "") -> gurobipy.Constr:
        return self.model.addLConstr(expr, name=name)

    def add_binary_variable(self, name: str = "") -> gurobipy.Var:
        return self.model.addVar(vtype=gurobipy.GRB.BINARY, name=name)

    def add_integer_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> gurobipy.Var:
        return self.model.addVar(vtype=gurobipy.GRB.INTEGER, lb=lb, ub=ub, name=name)

    def add_continuous_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> gurobipy.Var:
        return self.model.addVar(vtype=gurobipy.GRB.CONTINUOUS, lb=lb, ub=ub, name=name)

    def set_model_objective(self, expr: Any, minimize: bool) -> None:
        """Define the model objective.

        Useful to write an `init_model()` common to gurobi and ortools/mathopt.

        Args:
            expr:
            minimize: if True, objective will be minimized, else maximized

        Returns:

        """
        if minimize:
            sense = gurobipy.GRB.MINIMIZE
        else:
            sense = gurobipy.GRB.MAXIMIZE
        self.model.setObjective(expr, sense=sense)

    @staticmethod
    def construct_linear_sum(expr: Iterable) -> Any:
        """Generate a linear sum (with variables) ready for the internal model."""
        return gurobipy.quicksum(expr)

    @abstractmethod
    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by `set_warm_start()`.

        Override it in subclasses to have a proper warm start. You can also override
        `set_warm_start()` if default behaviour is not sufficient.

        """
        return {}

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        By default, this is using `convert_to_variable_values()`. If not sufficient,
        you can override it. (And for instance make implementation of `convert_to_variable_values()`
        raise a `NotImplementedError`.)

        """
        self.set_warm_start_from_values(
            variable_values=self.convert_to_variable_values(solution),
        )

    def set_warm_start_from_values(
        self, variable_values: dict[gurobipy.Var, float]
    ) -> None:
        for var, val in variable_values.items():
            var.Start = val
            var.VarHintVal = val


class GurobiCallback:
    def __init__(self, do_solver: GurobiMilpSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def __call__(self, model, where) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            try:
                # retrieve and store new solution
                sol = self.do_solver.retrieve_current_solution(
                    get_var_value_for_current_solution=model.cbGetSolution,
                    get_obj_value_for_current_solution=lambda: model.cbGet(
                        gurobipy.GRB.Callback.MIPSOL_OBJ
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

    @staticmethod
    def create_empty_model(name: str = "") -> Any:
        raise NotImplementedError()

    def add_linear_constraint(self, expr: Any, name: str = "") -> Any:
        raise NotImplementedError()

    def add_binary_variable(self, name: str = "") -> Any:
        raise NotImplementedError()

    def add_integer_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> Any:
        raise NotImplementedError()

    def add_continuous_variable(
        self, lb: float = 0.0, ub: float = math.inf, name: str = ""
    ) -> Any:
        raise NotImplementedError()

    def set_model_objective(self, expr: Any, minimize: bool) -> None:
        raise NotImplementedError()

    @staticmethod
    def construct_linear_sum(expr: Iterable) -> Any:
        raise NotImplementedError()
