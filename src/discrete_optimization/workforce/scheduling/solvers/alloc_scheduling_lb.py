#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from ortools.sat.python.cp_model import CpModel, CpSolver

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingProblem
from discrete_optimization.workforce.scheduling.utils import get_working_time_teams

logger = logging.getLogger(__name__)


class BaseAllocSchedulingLowerBoundProvider(Hyperparametrizable, ABC):
    status: Optional[str] = None

    def __init__(self, problem: AllocSchedulingProblem):
        self.problem = problem

    @abstractmethod
    def get_lb_nb_teams(self, **kwargs: Any) -> int: ...


class LBoundAllocScheduling(BaseAllocSchedulingLowerBoundProvider):
    def get_lb_nb_teams(self, **args) -> int:
        work_time = get_working_time_teams(problem=self.problem)
        sorted_working_time = np.array(sorted(list(work_time.values()), reverse=True))
        cumulated = np.cumsum(sorted_working_time)
        sum_duration_task = sum(
            [self.problem.tasks_data[t].duration_task for t in self.problem.tasks_data]
        )
        index = next(
            (i for i in range(cumulated.shape[0]) if cumulated[i] >= sum_duration_task)
        )
        return index + 1


class ApproximateBoundAllocScheduling(BaseAllocSchedulingLowerBoundProvider):
    def get_lb_nb_teams(self, **args) -> int:
        st_lb = [
            (
                int(self.problem.get_lb_start_window(t)),
                int(self.problem.get_ub_start_window(t)),
                int(self.problem.get_lb_end_window(t)),
                int(self.problem.get_ub_end_window(t)),
            )
            for t in self.problem.tasks_list
        ]
        usage = np.zeros(self.problem.horizon, dtype=float)
        i = 0
        for t in self.problem.tasks_list:
            description = self.problem.tasks_data[t]
            duration = description.duration_task
            tuple_ = st_lb[i]
            span_ = tuple_[-1] - tuple_[0]
            fraction = float(duration / span_)
            usage[tuple_[0] : tuple_[-1]] += fraction
            i += 1
        bound = np.max(usage)
        return int(math.ceil(bound))
        # return usage, int(math.ceil(bound))


class BoundResourceViaRelaxedProblem(BaseAllocSchedulingLowerBoundProvider):
    """
    See add_lower_bound_nb_teams function in cpmpy_alloc_scheduling_solver
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="adding_precedence_constraint", choices=[False, True], default=False
        ),
    ]

    def get_lb_nb_teams(self, **kwargs) -> int:
        kwargs = self.complete_with_default_hyperparameters(kwargs=kwargs)
        model = CpModel()
        st_lb = [
            (
                int(self.problem.get_lb_start_window(t)),
                int(self.problem.get_ub_start_window(t)),
                int(self.problem.get_lb_end_window(t)),
                int(self.problem.get_ub_end_window(t)),
            )
            for t in self.problem.tasks_list
        ]
        starts = [
            model.NewIntVar(lb=st_lb[i][0], ub=st_lb[i][1], name=f"start_{i}")
            for i in range(len(st_lb))
        ]
        ends = [
            model.NewIntVar(lb=st_lb[i][2], ub=st_lb[i][3], name=f"end_{i}")
            for i in range(len(st_lb))
        ]
        intervals = [
            model.NewIntervalVar(
                start=starts[i],
                end=ends[i],
                size=self.problem.tasks_data[self.problem.tasks_list[i]].duration_task,
                name=f"interval_{i}",
            )
            for i in range(len(starts))
        ]
        capacity = model.NewIntVar(lb=1, ub=self.problem.number_teams, name=f"capacity")
        model.AddCumulative(
            intervals, demands=[1] * self.problem.number_tasks, capacity=capacity
        )
        model.Minimize(capacity)
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = kwargs.get("time_limit", 5)
        solver.parameters.num_workers = kwargs.get("num_workers", 16)
        solver.parameters.log_search_progress = kwargs.get("log_search_progress", False)
        res = solver.solve(model=model)
        bound = solver.BestObjectiveBound()
        best_sol = solver.ObjectiveValue()
        logging.info(f"Bound is {bound}, objective is {best_sol}")
        logging.info(f"Status : {solver.status_name(res)}")
        self.status = solver.status_name(res)
        return int(bound)
