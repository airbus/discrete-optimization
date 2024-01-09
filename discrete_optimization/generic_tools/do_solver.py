"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, List, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class SolverDO:
    """Base class for a discrete-optimization solver."""

    @abstractmethod
    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Generic solving function.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: any argument specific to the solver

        Solvers deriving from SolverDo should use callbacks methods .on_step_end(), ...
        during solve(). But some solvers are not yet updated and are just ignoring it.

        Returns (ResultStorage): a result object containing potentially a pool of solutions
        to a discrete-optimization problem
        """
        ...
