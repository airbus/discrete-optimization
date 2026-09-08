#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from ortools.sat.python.cp_model import LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task, TasksCpSolver
from discrete_optimization.generic_tasks_tools.utils import optional_override
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver

logger = logging.getLogger(__name__)


class TasksCpSatSolver(TasksCpSolver[Task], OrtoolsCpSatSolver):
    @optional_override
    def get_task_is_present_variable(self, task: Task) -> LinearExprT:
        """Get the boolean variable whether the (optional) task is present.

        Default implementation returns 1 for mandatory task.
        To be overriden when optional tasks to get an actual boolean variable.

        """
        if self.problem.is_optional(task):
            raise NotImplementedError()
        return 1
