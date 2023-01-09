"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class SolverDO:
    """Base class for a discrete-optimization solver."""

    @abstractmethod
    def solve(self, **kwargs: Any) -> ResultStorage:
        """Generic solving function.

        Args:
            **kwargs: any argument specific to the solver

        Returns (ResultStorage): a result object containing potentially a pool of solutions
        to a discrete-optimization problem
        """
        ...
