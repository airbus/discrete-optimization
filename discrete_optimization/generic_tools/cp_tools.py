"""
Constraint programming common utilities and class that should be used by any solver using CP

"""
from abc import abstractmethod
from enum import Enum
from typing import Optional

from minizinc import Instance

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class CPSolverName(Enum):
    """
    Enum choice of underlying CP/LP solver used by Minizinc typically
    """

    CHUFFED = 0
    GECODE = 1
    CPLEX = 2
    CPOPT = 3
    GUROBI = 4
    ORTOOLS = 5


map_cp_solver_name = {
    CPSolverName.CHUFFED: "chuffed",
    CPSolverName.GECODE: "gecode",
    CPSolverName.CPLEX: "cplex",
    CPSolverName.CPOPT: "cpo",
    # need to install https://github.com/IBMDecisionOptimization/cpofzn
    CPSolverName.GUROBI: "gurobi",
    CPSolverName.ORTOOLS: "ortools",
}


class ParametersCP:
    """
    Parameters that can be used by any cp - solver
    """

    time_limit: int
    time_limit_iter0: int
    intermediate_solution: bool
    all_solutions: bool
    nr_solutions: int
    free_search: bool
    multiprocess: bool
    nb_process: int

    def __init__(
        self,
        time_limit,
        intermediate_solution: bool,
        all_solutions: bool,
        nr_solutions: int,
        time_limit_iter0=None,
        free_search: bool = False,
        multiprocess: bool = False,
        nb_process: int = 1,
    ):
        """

        :param time_limit: in seconds, the time limit of solving the cp model
        :param intermediate_solution: retrieve intermediate solutions
        :param all_solutions: returns all solutions found by the cp solver
        :param nr_solutions: the requested number of solutions
        """
        self.time_limit = time_limit
        if time_limit_iter0 is None:
            self.time_limit_iter0 = time_limit
        else:
            self.time_limit_iter0 = time_limit_iter0
        self.intermediate_solution = intermediate_solution
        self.all_solutions = all_solutions
        self.nr_solutions = nr_solutions
        self.free_search = free_search
        self.multiprocess = multiprocess
        self.nb_process = nb_process

    @staticmethod
    def default():
        return ParametersCP(
            time_limit=100,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=False,
        )

    @staticmethod
    def default_fast_lns():
        return ParametersCP(
            time_limit=10,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=False,
        )

    @staticmethod
    def default_free():
        return ParametersCP(
            time_limit=100,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=1000,
            free_search=True,
        )

    def copy(self):
        return ParametersCP(
            time_limit=self.time_limit,
            time_limit_iter0=self.time_limit_iter0,
            intermediate_solution=self.intermediate_solution,
            all_solutions=self.all_solutions,
            nr_solutions=self.nr_solutions,
            free_search=self.free_search,
            multiprocess=self.multiprocess,
            nb_process=self.nb_process,
        )


class SignEnum(Enum):
    EQUAL = "=="
    LEQ = "<="
    UEQ = ">="
    LESS = "<"
    UP = ">"


class CPSolver(SolverDO):
    """
    Additional function to be implemented by a CP Solver.
    """

    instance: Optional[Instance]

    @abstractmethod
    def init_model(self, **args):
        """
        Instantiate a CP model instance

        Afterwards, self.instance should not be None anymore.

        """
        ...

    @abstractmethod
    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        """
        Returns a storage solution coherent with the given parameters.
        :param result: Result storage returned by the cp solver
        :param parameters_cp: parameters of the CP solver.
        :return:
        """
        ...

    @abstractmethod
    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **args
    ) -> ResultStorage:
        ...
