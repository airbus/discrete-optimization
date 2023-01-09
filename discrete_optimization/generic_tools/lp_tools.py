#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

import mip

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


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

    @abstractmethod
    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
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
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        self.model.max_mip_gap = parameters_milp.mip_gap
        self.model.max_mip_gap_abs = parameters_milp.mip_gap_abs
        self.model.sol_pool_size = parameters_milp.pool_solutions

        self.model.optimize(
            max_seconds=parameters_milp.time_limit,
            max_solutions=parameters_milp.n_solutions_max,
        )

        logger.info(f"Solver found {self.model.num_solutions} solutions")
        logger.info(f"Objective : {self.model.objective_value}")

        return self.retrieve_solutions(parameters_milp=parameters_milp)

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

        self.model.optimize()

        logger.info(f"Problem has {self.model.NumObj} objectives")
        logger.info(f"Solver found {self.model.SolCount} solutions")
        logger.info(f"Objective : {self.model.getObjective().getValue()}")

        return self.retrieve_solutions(parameters_milp=parameters_milp)

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
        return self.model.getAttr("ObjVal")

    def get_pool_obj_value_for_ith_solution(self, i: int) -> float:
        """Get pool objective value for i-th solution."""
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
