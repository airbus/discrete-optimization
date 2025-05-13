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
from enum import Enum
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
    BoundsProviderMixin,
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
from discrete_optimization.generic_tools.unsat_tools import MetaConstraint

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


class InequalitySense(Enum):
    """Sense of an inequality/equality."""

    LOWER_OR_EQUAL = "<="
    GREATER_OR_EQUAL = ">="
    EQUAL = "=="


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

    def add_linear_constraint_with_indicator(
        self,
        binvar: VariableType,
        binval: int,
        lhs: Any,
        sense: InequalitySense,
        rhs: Any,
        penalty_coeff=100000,
        name: str = "",
    ) -> list[ConstraintType]:
        """Add a linear constraint depending on the value of an indicator (boolean var)

        This mirrors `gurobipy.Model.addGenConstrIndicator().

        Args:
            binvar:
            binval:
            lhs:
            sense:
            rhs:
            penalty_coeff:
            name:

        Returns:

        """
        if binval == 1:
            penalty = penalty_coeff * (1 - binvar)
        elif binval == 0:
            penalty = penalty_coeff * (binvar - 0)
        else:
            raise ValueError("binval should only be 0 or 1.")

        constraints = []
        if sense in [InequalitySense.LOWER_OR_EQUAL, InequalitySense.EQUAL]:
            if name:
                name_lower = name + "_lower"
            else:
                name_lower = ""
            constraints.append(
                self.add_linear_constraint(lhs <= rhs + penalty, name=name_lower)
            )
        if sense in [InequalitySense.GREATER_OR_EQUAL, InequalitySense.EQUAL]:
            if name:
                name_upper = name + "_upper"
            else:
                name_upper = ""
            constraints.append(
                self.add_linear_constraint(lhs >= rhs - penalty, name=name_upper)
            )
        return constraints

    def set_warm_start_from_values(
        self, variable_values: dict[VariableType, float]
    ) -> None:
        """Make the model variables warm start from the given values."""
        raise NotImplementedError()


class OrtoolsMathOptMilpSolver(MilpSolver, WarmstartMixin, BoundsProviderMixin):
    """Milp solver wrapping a solver from pymip library."""

    hyperparameters = [
        EnumHyperparameter(
            name="mathopt_solver_type",
            enum=mathopt.SolverType,
        )
    ]

    solution_hint: Optional[mathopt.SolutionHint] = None
    model: Optional[mathopt.Model] = None
    termination: mathopt.Termination
    early_stopping_exception: Optional[Exception] = None
    has_quadratic_objective: bool = False
    """Flag telling that the objective is a quadratic expression.

    This is use to pass the proper function to `retrieve_current_solution` for the objective.
    Should be overriden to True in solvers with quadratic objectives.

    """
    random_seed: Optional[int] = None
    mathopt_res: Optional[mathopt.SolveResult] = None

    _current_internal_objective_best_value: Optional[float] = None
    _current_internal_objective_best_bound: Optional[float] = None

    def remove_constraints(self, constraints: Iterable[Any]) -> None:
        for cstr in constraints:
            self.model.delete_linear_constraint(cstr)

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

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_milp: Optional[ParametersMilp] = None,
        mathopt_solver_type: mathopt.SolverType = mathopt.SolverType.CP_SAT,
        time_limit: Optional[float] = 30.0,
        mathopt_enable_output: bool = False,
        mathopt_model_parameters: Optional[mathopt.ModelSolveParameters] = None,
        mathopt_additional_solve_parameters: Optional[mathopt.SolveParameters] = None,
        store_mathopt_res: bool = False,
        extract_solutions_from_mathopt_res: Optional[bool] = None,
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
            store_mathopt_res: whether to store the `mathopt.SolveResult` generated by `mathopt.solve()`
            extract_solutions_from_mathopt_res: whether to extract solutions from the `mathopt.SolveResult` generated by `mathopt.solve()`.
                If False, the solutions are rather extracted on the fly inside a callback at each mip solution improvement.
                By default, this is False except for HiGHS solver type, as its OR-Tools wrapper does not yet integrate callbacks.
            **kwargs: passed to init_model() if model not yet existing

        Returns:

        """
        # Default parameters
        if extract_solutions_from_mathopt_res is None:
            extract_solutions_from_mathopt_res = (
                mathopt_solver_type == mathopt.SolverType.HIGHS
            )

        # reset best bound and obj
        self._current_internal_objective_best_value = None
        self._current_internal_objective_best_bound = None

        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        # wrap user callback in a mathopt callback
        mathopt_cb = MathOptCallback(
            do_solver=self,
            callback=callbacks_list,
            mathopt_solver_type=mathopt_solver_type,
        )

        # optimize
        mathopt_res = self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            mathopt_solver_type=mathopt_solver_type,
            mathopt_cb=mathopt_cb,
            mathopt_enable_output=mathopt_enable_output,
            mathopt_model_parameters=mathopt_model_parameters,
            mathopt_additional_solve_parameters=mathopt_additional_solve_parameters,
            **kwargs,
        )
        if store_mathopt_res:
            self.mathopt_res = mathopt_res

        # update best bound and obj
        self._current_internal_objective_best_value = mathopt_res.primal_bound()
        self._current_internal_objective_best_bound = mathopt_res.dual_bound()

        # get result storage
        if extract_solutions_from_mathopt_res:
            res = self._extract_result_storage(mathopt_res)
        else:
            res = mathopt_cb.res

        # callback: solve end
        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def set_random_seed(self, random_seed: int) -> None:
        self.random_seed = random_seed

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
        if self.random_seed is not None:
            params.random_seed = self.random_seed

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

        # raise potential exception found during callback (useful for optuna pruning, and debugging)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception

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

    def _extract_result_storage(
        self, mathopt_res: mathopt.SolveResult
    ) -> ResultStorage:
        list_solution_fits = []
        for internal_sol in mathopt_res.solutions:
            get_var_value_for_current_solution = (
                lambda var: internal_sol.primal_solution.variable_values[var]
            )
            get_obj_value_for_current_solution = (
                lambda: internal_sol.primal_solution.objective_value
            )
            sol = self.retrieve_current_solution(
                get_var_value_for_current_solution=get_var_value_for_current_solution,
                get_obj_value_for_current_solution=get_obj_value_for_current_solution,
            )
            fit = self.aggreg_from_sol(sol)
            list_solution_fits.append((sol, fit))
        return self.create_result_storage(list(reversed(list_solution_fits)))

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self._current_internal_objective_best_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self._current_internal_objective_best_value


map_mathopt_status_to_do_status: dict[mathopt.TerminationReason, StatusSolver] = {
    mathopt.TerminationReason.OPTIMAL: StatusSolver.OPTIMAL,
    mathopt.TerminationReason.INFEASIBLE: StatusSolver.UNSATISFIABLE,
    mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.UNBOUNDED: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.FEASIBLE: StatusSolver.SATISFIED,
    mathopt.TerminationReason.NO_SOLUTION_FOUND: StatusSolver.UNSATISFIABLE,
    mathopt.TerminationReason.IMPRECISE: StatusSolver.UNKNOWN,
    mathopt.TerminationReason.NUMERICAL_ERROR: StatusSolver.ERROR,
    mathopt.TerminationReason.OTHER_ERROR: StatusSolver.ERROR,
}


class MathOptCallback:
    def __init__(
        self,
        do_solver: OrtoolsMathOptMilpSolver,
        callback: Callback,
        mathopt_solver_type: mathopt.SolverType,
    ):
        self.solver_type = mathopt_solver_type
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def __call__(self, callback_data: mathopt.CallbackData) -> mathopt.CallbackResult:
        cb_sol = callback_data.solution
        try:
            # retrieve and store new solution
            get_var_value_for_current_solution = lambda var: cb_sol[var]
            if self.do_solver.has_quadratic_objective:
                get_obj_value_for_current_solution = lambda: self.do_solver.model.objective.as_quadratic_expression().evaluate(
                    cb_sol
                )
            else:
                get_obj_value_for_current_solution = lambda: self.do_solver.model.objective.as_linear_expression().evaluate(
                    cb_sol
                )
            sol = self.do_solver.retrieve_current_solution(
                get_var_value_for_current_solution=get_var_value_for_current_solution,
                get_obj_value_for_current_solution=get_obj_value_for_current_solution,
            )
            fit = self.do_solver.aggreg_from_sol(sol)
            self.res.append((sol, fit))
            self.nb_solutions += 1
            # store mip stats
            if self.solver_type != mathopt.SolverType.CP_SAT:
                self.do_solver._current_internal_objective_best_value = (
                    callback_data.mip_stats.primal_bound
                )
                self.do_solver._current_internal_objective_best_bound = (
                    callback_data.mip_stats.dual_bound
                )
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


class GurobiMilpSolver(MilpSolver, WarmstartMixin, BoundsProviderMixin):
    """Milp solver wrapping a solver from gurobi library."""

    model: Optional["gurobipy.Model"] = None
    early_stopping_exception: Optional[Exception] = None

    _current_internal_objective_best_value: Optional[float] = None
    _current_internal_objective_best_bound: Optional[float] = None

    def remove_constraints(self, constraints: Iterable[Any]) -> None:
        self.model.remove(list(constraints))
        self.model.update()

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> ResultStorage:
        self.early_stopping_exception = None
        # reset best bound and obj
        self._current_internal_objective_best_value = None
        self._current_internal_objective_best_bound = None

        callbacks_list = CallbackList(callbacks=callbacks)

        # callback: solve start
        callbacks_list.on_solve_start(solver=self)

        # wrap user callback in a gurobi callback
        gurobi_callback = GurobiCallback(do_solver=self, callback=callbacks_list)

        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            gurobi_callback=gurobi_callback,
            **kwargs,
        )

        # update best bound and obj
        if hasattr(self.model, "ObjVal"):
            self._current_internal_objective_best_value = self.model.ObjVal
        if hasattr(self.model, "ObjBound"):
            self._current_internal_objective_best_bound = self.model.ObjBound

        # get result storage
        res = gurobi_callback.res

        # callback: solve end
        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def explain_unsat_fine(self) -> list[Any]:
        """Explain unsatisfiability of the problem via fine constraints.

        Returns:
            subset minimal list of constraints leading to unsatisfiability.

        Note:
            running several times may lead to a different (minimal) subset of constraints.

        """
        assert self.status_solver == StatusSolver.UNSATISFIABLE, (
            "self.solve() must have been run "
            "and self.status_solver must be SolverStatus.UNSATISFIABLE"
        )
        self.model.computeIIS()
        constraints = []
        # linear constraints
        for cstr in self.model.getConstrs():
            if cstr.IISConstr:
                constraints.append(cstr)
        # quadratic constraints
        for cstr in self.model.getQConstrs():
            if cstr.IISQConstr:
                constraints.append(cstr)
        # generic constraints
        for cstr in self.model.getGenConstrs():
            if cstr.IISGenConstr:
                constraints.append(cstr)
        # SOS constraints
        for cstr in self.model.getSOSs():
            if cstr.IISSOS:
                constraints.append(cstr)
        # lower and upper bounds
        for var in self.model.getVars():
            if var.IISLB:
                constraints.append(var >= var.LB)
            if var.IISUB:
                constraints.append(var <= var.UB)

        return constraints

    def explain_unsat_meta(
        self,
        meta_constraints: Optional[list[MetaConstraint]] = None,
    ) -> list[MetaConstraint]:
        """Explain unsatisfiability of the problem via meta-constraints.

        Meta-constraints are gathering several finer constraints in order to be more human readable.

        Args:
            meta_constraints: list of meta-constraints.
                Default to the ones returned by `get_default_meta_constraints()`.

        Returns:
            subset minimal list of meta-constraints leading to unsatisfiability.

        """
        if meta_constraints is None:
            meta_constraints = self.get_meta_constraints()
        mus_constraints = self.explain_unsat_fine()
        mus_meta_constraints = set()
        for c in mus_constraints:
            for meta in meta_constraints:
                if c in meta:
                    mus_meta_constraints.add(meta)
        return list(mus_meta_constraints)

    def get_meta_constraints(self) -> list[MetaConstraint]:
        """Get meta-constraints defining the internal model.

        To be used to explain unsatisfiability. See `explain_unsat_meta()`.

        Returns:
            default set of meta-constraints defining the problem

        """
        raise NotImplementedError("No meta constraints defined for this model.")

    def set_random_seed(self, random_seed: int) -> None:
        self.model.setParam(gurobipy.GRB.Param.Seed, random_seed)

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
        gurobi_callback: Optional[GurobiCallback] = None,
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
        self.model.optimize(gurobi_callback)
        self.status_solver = map_gurobi_status_to_do_status[self.model.Status]

        # raise potential exception found during callback (useful for optuna pruning, and debugging)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception

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

    def add_linear_constraint_with_indicator(
        self,
        binvar: VariableType,
        binval: int,
        lhs: Any,
        sense: InequalitySense,
        rhs: Any,
        penalty_coeff=100000,
        name: str = "",
    ) -> list[ConstraintType]:
        """Add a linear constraint dependending on the value of an indicator (boolean var)

        This wraps `gurobipy.Model.addGenConstrIndicator().

        Args:
            binvar:
            binval:
            lhs:
            sense:
            rhs:
            penalty_coeff:
            name:

        Returns:

        """
        if sense == InequalitySense.EQUAL:
            expr = lhs == rhs
        elif sense == InequalitySense.LOWER_OR_EQUAL:
            expr = lhs <= rhs
        else:
            expr = lhs >= rhs

        return [
            self.model.addGenConstrIndicator(
                binvar,
                binval,
                expr,
                name=name,
            )
        ]

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

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self._current_internal_objective_best_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self._current_internal_objective_best_value


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
                # store mip stats
                self.do_solver._current_internal_objective_best_value = model.cbGet(
                    gurobipy.GRB.Callback.MIPSOL_OBJ
                )
                self.do_solver._current_internal_objective_best_bound = model.cbGet(
                    gurobipy.GRB.Callback.MIPSOL_OBJBND
                )
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
