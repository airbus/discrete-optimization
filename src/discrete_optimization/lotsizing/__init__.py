#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Lot sizing module for discrete optimization.

This module provides a flexible mixin-based architecture for lot sizing problems,
following the pattern from generic_tasks_tools.

Main classes:
- GenericLotSizingProblem: Composition of all feature mixins
- GenericLotSizingSolution: Solution class with all features
- ProductionBasedSolution: Base solution with inventory/backlog computation

Mixins:
- DemandsProblem/Solution: Demand requirements (core)
- SetupCostsProblem/Solution: Setup costs
- ProductionCostsProblem/Solution: Variable production costs
- InventoryCostsProblem/Solution: Inventory holding costs
- CapacityProblem/Solution: Production capacity constraints
- BacklogProblem/Solution: Backlogged demand
- SetupTimesProblem/Solution: Setup times consuming capacity
- ChangeoverCostsProblem/Solution: Sequence-dependent changeover costs

"Without" variants:
- WithoutCapacityProblem/Solution: For uncapacitated problems
- WithoutBacklogProblem/Solution: When backlog not allowed
- WithoutSetupTimesProblem: When setup times don't consume capacity
- WithoutChangeoverCostsProblem/Solution: Sequence-independent problems

Example:
    >>> from discrete_optimization.lotsizing import (
    ...     GenericLotSizingProblem,
    ...     DemandsArrayProblem,
    ...     CostsArrayProblem,
    ...     WithoutCapacityProblem,
    ... )
    >>> # Define uncapacitated single-item problem
    >>> class MyProblem(
    ...     CostsArrayProblem[int],
    ...     DemandsArrayProblem[int],
    ...     WithoutCapacityProblem[int],
    ...     GenericLotSizingProblem[int],
    ... ):
    ...     pass
"""

# Base classes
from discrete_optimization.lotsizing.backlog import (
    BacklogProblem,
    BacklogSolution,
    WithoutBacklogProblem,
    WithoutBacklogSolution,
)
from discrete_optimization.lotsizing.base import (
    Item,
    LotSizingProblem,
    LotSizingSolution,
)
from discrete_optimization.lotsizing.capacity import (
    CapacityProblem,
    CapacitySolution,
    WithoutCapacityProblem,
    WithoutCapacitySolution,
)
from discrete_optimization.lotsizing.changeover import (
    ChangeoverCostsProblem,
    ChangeoverCostsSolution,
    WithoutChangeoverCostsProblem,
    WithoutChangeoverCostsSolution,
)
from discrete_optimization.lotsizing.costs import (
    CostsArrayProblem,
    InventoryCostsProblem,
    InventoryCostsSolution,
    ProductionCostsProblem,
    ProductionCostsSolution,
    SetupCostsProblem,
    SetupCostsSolution,
    SingleItemCostsArrayProblem,
    WithoutInventoryCostsProblem,
    WithoutInventoryCostsSolution,
    WithoutProductionCostsProblem,
    WithoutProductionCostsSolution,
    WithoutSetupCostsProblem,
    WithoutSetupCostsSolution,
)

# Mixins - Problem classes
from discrete_optimization.lotsizing.demands import (
    DemandsArrayProblem,
    DemandsProblem,
    DemandsSolution,
    SingleItemDemandsArrayProblem,
)

# Generic composition
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    GenericLotSizingSolution,
)

# Production-based solution
from discrete_optimization.lotsizing.production_solution import (
    ProductionBasedSolution,
    ProductionDecision,
)
from discrete_optimization.lotsizing.setup_times import (
    SetupTimesProblem,
    SetupTimesSolution,
    WithoutSetupTimesProblem,
)

__all__ = [
    # Base
    "Item",
    "LotSizingProblem",
    "LotSizingSolution",
    # Production solution
    "ProductionBasedSolution",
    "ProductionDecision",
    # Demands
    "DemandsProblem",
    "DemandsSolution",
    "DemandsArrayProblem",
    "SingleItemDemandsArrayProblem",
    # Costs
    "SetupCostsProblem",
    "SetupCostsSolution",
    "ProductionCostsProblem",
    "ProductionCostsSolution",
    "InventoryCostsProblem",
    "InventoryCostsSolution",
    "CostsArrayProblem",
    "SingleItemCostsArrayProblem",
    "WithoutSetupCostsProblem",
    "WithoutSetupCostsSolution",
    "WithoutProductionCostsProblem",
    "WithoutProductionCostsSolution",
    "WithoutInventoryCostsProblem",
    "WithoutInventoryCostsSolution",
    # Capacity
    "CapacityProblem",
    "CapacitySolution",
    "WithoutCapacityProblem",
    "WithoutCapacitySolution",
    # Backlog
    "BacklogProblem",
    "BacklogSolution",
    "WithoutBacklogProblem",
    "WithoutBacklogSolution",
    # Setup times
    "SetupTimesProblem",
    "SetupTimesSolution",
    "WithoutSetupTimesProblem",
    # Changeover
    "ChangeoverCostsProblem",
    "ChangeoverCostsSolution",
    "WithoutChangeoverCostsProblem",
    "WithoutChangeoverCostsSolution",
    # Generic
    "GenericLotSizingProblem",
    "GenericLotSizingSolution",
]
