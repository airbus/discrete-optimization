#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Any, Generic

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    Item,
)


class LotSizingGenericSolver(SolverDO, Generic[Item]):
    problem: GenericLotSizingProblem[Item]

    @abstractmethod
    def get_production_var(self, item: Item, period: int) -> Any: ...

    @abstractmethod
    def get_inventory_var(self, item: Item, period: int) -> Any: ...

    @abstractmethod
    def get_backlog_var(self, item: Item, period: int) -> Any: ...
