#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import mip

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
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

try:
    import docplex
    from docplex.mp.constr import LinearConstraint
    from docplex.mp.dvar import Var
    from docplex.mp.model import Model
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
        time_limit: int,
        pool_solutions: int,
        mip_gap_abs: float,
        mip_gap: float,
        retrieve_all_solution: bool,
        n_solutions_max: int,
        pool_search_mode: int = 0,
    ):
        self.time_limit = time_limit
        self.pool_solutions = pool_solutions
        self.mip_gap_abs = mip_gap_abs
        self.mip_gap = mip_gap
        self.retrieve_all_solution = retrieve_all_solution
        self.n_solutions_max = n_solutions_max
        self.pool_search_mode = pool_search_mode

    @staticmethod
    def default() -> "ParametersMilp":
        return ParametersMilp(
            time_limit=30,
            pool_solutions=10000,
            mip_gap_abs=0.0000001,
            mip_gap=0.000001,
            retrieve_all_solution=True,
            n_solutions_max=10000,
        )


class MilpSolver(SolverDO):
    model: Optional[Any]

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        ...

    def retrieve_solutions(
        self, parameters_milp: ParametersMilp, **kwargs
    ) -> ResultStorage:
        """Retrieve solutions found by internal solver.

        Args:
            parameters_milp:
            **kwargs: passed to ResultStorage.__init__()

        Returns:

        """
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for i in range(n_solutions):
            solution = self.retrieve_ith_solution(i=i)
            fit = self.aggreg_from_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
            best_solution=min(list_solution_fits, key=lambda x: x[1])[0],
            **kwargs,
        )

    @abstractmethod
    def retrieve_ith_solution(self, i: int) -> Solution:
        """Retrieve i-th solution from internal milp model."""
        ...

    @abstractmethod
    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> ResultStorage:
        ...

    @abstractmethod
    def get_var_value_for_ith_solution(self, var: Any, i: int) -> float:
        """Get value for i-th solution of a given variable."""
        pass

    @abstractmethod
    def get_obj_value_for_ith_solution(self, i: int) -> float:
        """Get objective value for i-th solution."""
        pass

    @property
    @abstractmethod
    def nb_solutions(self) -> int:
        """Number of solutions found by the solver."""
        pass


class PymipMilpSolver(MilpSolver):
    """Milp solver wrapping a solver from pymip library."""

    model: Optional[mip.Model] = None

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        self.optimize_model(parameters_milp=parameters_milp, **kwargs)
        return self.retrieve_solutions(parameters_milp=parameters_milp)

    def prepare_model(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
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
        self.model.max_seconds = parameters_milp.time_limit
        self.model.max_solutions = parameters_milp.n_solutions_max

    def optimize_model(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
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
        self.prepare_model(parameters_milp=parameters_milp, **kwargs)
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


class GurobiMilpSolver(MilpSolver):
    """Milp solver wrapping a solver from gurobi library."""

    model: Optional["gurobipy.Model"] = None

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        self.optimize_model(parameters_milp=parameters_milp, **kwargs)
        return self.retrieve_solutions(parameters_milp=parameters_milp)

    def prepare_model(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
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
        self.model.setParam(gurobipy.GRB.Param.TimeLimit, parameters_milp.time_limit)
        self.model.setParam(gurobipy.GRB.Param.MIPGapAbs, parameters_milp.mip_gap_abs)
        self.model.setParam(gurobipy.GRB.Param.MIPGap, parameters_milp.mip_gap)
        self.model.setParam(
            gurobipy.GRB.Param.PoolSolutions, parameters_milp.pool_solutions
        )
        self.model.setParam("PoolSearchMode", parameters_milp.pool_search_mode)

    def optimize_model(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> None:
        """Optimize the Gurobi Model.

        The solutions are yet to be retrieved via `self.retrieve_solutions()`.

        """
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.prepare_model(parameters_milp=parameters_milp, **kwargs)
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


def gurobi_callback(model, where):
    ...


class CplexMilpSolver(MilpSolver):
    model: Optional["docplex.mp.model.Model"]
    results_solve: Optional[List["SolveSolution"]]

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
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
        self.model.time_limit = parameters_milp.time_limit
        self.model.parameters.mip.tolerances.mipgap = parameters_milp.mip_gap
        listener = None
        if parameters_milp.retrieve_all_solution or parameters_milp.n_solutions_max > 1:

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
