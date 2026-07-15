#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Scheduling to RCPSP."""

import numpy as np

from discrete_optimization.generic_tasks_tools.from_sched_to_generic import (
    SchedulingToGenericTransformation,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplSolution,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)


class WorkforceSchedulingToGenericTransformation(SchedulingToGenericTransformation):
    def back_transform_solution(
        self,
        solution: GenericSchedulingImplSolution,
        source_problem: AllocSchedulingProblem,
    ) -> AllocSchedulingSolution:
        schedule = np.zeros((source_problem.number_tasks, 2), dtype=int)
        allocation = np.zeros(source_problem.number_tasks, dtype=int)
        for t in solution.problem.tasks_list:
            start = solution.get_start_time(t)
            end = solution.get_end_time(t)
            schedule[source_problem.get_index_from_task(t), 0] = start
            schedule[source_problem.get_index_from_task(t), 1] = end
            alloc = solution.get_task_allocation(t)
            for res in alloc:
                team = source_problem.get_index_from_unary_resource(res)
                allocation[source_problem.get_index_from_task(t)] = team
        return AllocSchedulingSolution(source_problem, schedule, allocation)
