#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Thanks to Leuven university for the cpmyp library.
from abc import abstractmethod
from typing import Any, Optional

import cpmpy.tools.explain as cpx
from cpmpy.expressions.core import Expression
from cpmpy.model import Model
from cpmpy.solvers.solver_interface import ExitStatus, SolverStatus

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

map_exitstatus2statussolver = {
    ExitStatus.NOT_RUN: StatusSolver.UNKNOWN,
    ExitStatus.OPTIMAL: StatusSolver.OPTIMAL,
    ExitStatus.FEASIBLE: StatusSolver.SATISFIED,
    ExitStatus.UNSATISFIABLE: StatusSolver.UNSATISFIABLE,
    ExitStatus.ERROR: StatusSolver.UNKNOWN,
    ExitStatus.UNKNOWN: StatusSolver.UNKNOWN,
}


class CpmpySolver(SolverDO):
    """Generic cpmpy solver."""

    model: Optional[Model] = None
    cpm_status: SolverStatus = SolverStatus("Model")

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any
    ) -> ResultStorage:
        """

        time_limit: the solve process stops after this time limit (in seconds).
                If None, no time limit is applied.
        Args:
            time_limit:
            **kwargs:

        Returns:

        """
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        solver = kwargs.get("solver", "ortools")
        self.model.solve(solver, time_limit=time_limit)
        self.cpm_status = self.model.cpm_status
        self.status_solver = map_exitstatus2statussolver[self.cpm_status.exitstatus]

        if self.cpm_status.exitstatus in [ExitStatus.UNSATISFIABLE, ExitStatus.ERROR]:
            res = self.create_result_storage([])
        else:
            sol = self.retrieve_current_solution()
            fit = self.aggreg_from_sol(sol)
            res = self.create_result_storage(
                [(sol, fit)],
            )

        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def explain_unsat(self) -> list[Expression]:
        """Explain unsatisfiability of the problem.

        Returns:
            subset minimal list of constraints leading to unsatisfiability.

        Note:
            running several times may lead to a different (minimal) subset of constraints.

        """
        assert self.status_solver == StatusSolver.UNSATISFIABLE, (
            "self.solve() must have been run "
            "and self.status_solver must be SolverStatus.UNSATISFIABLE"
        )
        return cpx.mus(self.model.constraints)

    @abstractmethod
    def retrieve_current_solution(self) -> Solution:
        """Construct a do solution from the cpmpy solver internal solution.

        It will be called after self.model.solve()

        Returns:
            the solution, at do format.

        """
        ...
