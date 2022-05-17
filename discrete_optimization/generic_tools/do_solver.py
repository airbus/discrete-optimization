"""Minimal API for a discrete-optimization solver."""

from abc import abstractmethod

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class SolverDO:
    """Base class for a discrete-optimization solver."""
    @abstractmethod
    def solve(self, **kwargs) -> ResultStorage:
        """Generic solving function.

        Args:
            **kwargs: any argument specific to the solver

        Returns (ResultStorage): a result object containing potentially a pool of solutions
        to a discrete-optimization problem
        """
        ...
