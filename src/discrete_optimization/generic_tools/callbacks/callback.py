#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # making annotations strings

from typing import TYPE_CHECKING, Optional

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

if TYPE_CHECKING:  # avoid cycling imports due solely to annotations
    from discrete_optimization.generic_tools.do_solver import SolverDO


class Callback:
    """Base class used to build new callbacks.

    Callbacks can be passed to solvers `solve()` in order to hook into the various stages of the solve.


    To create a custom callback, subclass `discrete_optimization.generic_tools.callbacks.Callback` and
    override the method associated with the stage of interest.

    """

    def set_params(self, params):
        self.params = params

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        """Called at the end of an optimization step.

        Args:
            step: index of step
            res: current result storage
            solver: solvers using the callback

        Returns:
            If `True`, the optimization process is stopped, else it goes on.

        """

    def on_solve_start(self, solver: SolverDO):
        """Called at the start of solve.

        Args:
            solver: solvers using the callback

        """

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        """Called at the end of solve.

        Args:
            res: current result storage
            solver: solvers using the callback

        """


class CallbackList(Callback):
    """Container abstracting a list of callbacks."""

    def __init__(
        self,
        callbacks=None,
        **params,
    ):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_step_end(...)`).

        Args:
            callbacks: List of `Callback` instances.
            **params: If provided, parameters will be passed to each `Callback`
                via `Callback.set_params`.
        """
        if callbacks:
            if isinstance(callbacks, Callback):
                self.callbacks = [callbacks]
            else:
                self.callbacks = callbacks
        else:
            self.callbacks = []

        if params:
            self.set_params(params)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        stopping = False
        for callback in self.callbacks:
            decision = callback.on_step_end(step=step, res=res, solver=solver)
            stopping = stopping or decision
        return stopping

    def on_solve_start(self, solver: SolverDO):
        for callback in self.callbacks:
            callback.on_solve_start(solver=solver)

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        for callback in self.callbacks:
            callback.on_solve_end(res=res, solver=solver)
