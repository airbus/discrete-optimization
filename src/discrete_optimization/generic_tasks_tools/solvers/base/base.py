#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task, TasksProblem
from discrete_optimization.generic_tools.cp_tools import CpSolver


class TasksSolver(CpSolver, Generic[Task]):
    """Base class for cp solver handling tasks problems."""

    problem: TasksProblem[Task]
