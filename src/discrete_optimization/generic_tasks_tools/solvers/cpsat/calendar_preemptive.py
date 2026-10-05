#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Generic

import numpy as np
from ortools.sat.python.cp_model import Domain, LinearExpr

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    CalendarPreemptiveProblem,
    CumulativeResource,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    MultimodeSchedulingCpSatSolver,
)


class ModelingCalendarPreemptive(Enum):
    INDICATOR = 0
    ELEMENT = 1


class CalendarPreemptiveCpSatSolver(
    MultimodeSchedulingCpSatSolver[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    problem: CalendarPreemptiveProblem[Task, CumulativeResource, OtherCalendarResource]
    _indicator_variables: dict
    _is_preempted: dict
    _nb_preempted_tasks: LinearExpr
    _nb_preempted_tasks_init = False
    _duration_per_mode: dict[tuple[Task, int], LinearExpr] = None

    def get_optional_duration_of_task(self, task: Task, mode: int) -> LinearExpr:
        # Needed for the modeling=ModelingPreemptive.ELEMENT option
        if self._duration_per_mode is None:
            self._duration_per_mode = {}
        if not self.problem.is_task_calendar_preempted(task):
            return self.problem.get_task_mode_duration(task, mode)
        if (task, mode) not in self._duration_per_mode:
            possible_durations = self.problem.get_possible_durations_for_task_mode(
                task, mode
            )
            self._duration_per_mode[(task, mode)] = (
                self.cp_model.new_int_var_from_domain(
                    Domain.from_values(possible_durations),
                    name=f"duration_{task}_{mode}",
                )
            )
            self.cp_model.add(
                self._duration_per_mode[(task, mode)]
                == self.get_duration_expression(task)
            ).only_enforce_if(self.get_task_mode_is_present_variable(task, mode))
        return self._duration_per_mode[(task, mode)]

    def create_preemptive_duration_constraints(
        self,
        modeling: ModelingCalendarPreemptive = ModelingCalendarPreemptive.INDICATOR,
    ):
        match modeling:
            case ModelingCalendarPreemptive.INDICATOR:
                self.constraint_duration_of_tasks_indicator()
            case ModelingCalendarPreemptive.ELEMENT:
                self.constraint_duration_of_tasks_element()
            case _:
                raise NotImplementedError(
                    f"{modeling} is not supported in calendar preemptive modeler"
                )

    def constraint_duration_of_tasks_indicator(self):
        """
        Tricky constraint : should take into account the partial preemption possibility,
        which makes duration variable based on calendars
        """

        self._indicator_variables = {}
        durs = self.problem.get_calendar_preemption_data().durations
        dictionary_indicators = {}
        for task, mode in durs:
            d = self.constraint_duration_of_task_indicator(
                task=task,
                mode=mode,
                duration_per_interval=durs[(task, mode)][1],
            )
            dictionary_indicators.update(d)
        self._indicator_variables = dictionary_indicators
        for task in self.problem.get_all_tasks_calendar_preempted():
            all_key = [x for x in self._indicator_variables if x[0][0] == task]
            self.cp_model.add_exactly_one(
                [self._indicator_variables[x] for x in all_key]
            )

    def constraint_duration_of_task_indicator(
        self,
        task: Task,
        mode: int,
        duration_per_interval: dict[int, list[tuple[int, int]]],
    ):
        dictionary_indicators = {}
        positive_durations = [d for d in duration_per_interval if d >= 0]
        is_present_mode = self.get_task_mode_is_present_variable(task=task, mode=mode)
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.add_linear_expression_in_domain(
                self.get_task_start_or_end_variable(
                    task, start_or_end=StartOrEnd.START
                ),
                interval,
            ).only_enforce_if(is_present_mode)
            self.cp_model.add(self.get_duration_variable(task) == dur).only_enforce_if(
                is_present_mode
            )
            dictionary_indicators[((task, mode), dur)] = is_present_mode
        else:
            for possible_duration in duration_per_interval:
                if possible_duration < 0:
                    continue
                interval = Domain.FromIntervals(
                    duration_per_interval[possible_duration]
                )
                dictionary_indicators[((task, mode), possible_duration)] = (
                    self.cp_model.new_bool_var(f"d_{(task, mode), possible_duration}")
                )
                self.cp_model.add_linear_expression_in_domain(
                    self.get_task_start_or_end_variable(task, StartOrEnd.START),
                    interval,
                ).OnlyEnforceIf(
                    dictionary_indicators[((task, mode), possible_duration)]
                )
                self.cp_model.add(
                    self.get_duration_variable(task) == int(possible_duration)
                ).OnlyEnforceIf(
                    dictionary_indicators[((task, mode), possible_duration)]
                )
            self.cp_model.Add(
                sum([dictionary_indicators[k] for k in dictionary_indicators])
                == is_present_mode
            )
        return dictionary_indicators

    def constraint_duration_of_tasks_element(self):
        durs = self.problem.compute_task_durations_with_calendar_preemption().durations
        for task, mode in durs:
            self.constraint_duration_of_task_element(
                task=task,
                mode=mode,
                duration_data=durs[(task, mode)],
            )

    def constraint_duration_of_task_element(
        self,
        task: int,
        mode: int,
        duration_data: tuple[np.ndarray, dict[int, list[tuple[int, int]]]],
    ):
        positive_durations = [int(d) for d in duration_data[0] if d >= 0]
        duration_per_interval = duration_data[1]
        duration_list = [int(d) for d in duration_data[0]]
        is_present_mode = self.get_task_mode_is_present_variable(task, mode)
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.add_linear_expression_in_domain(
                self.get_task_start_or_end_variable(task, StartOrEnd.START), interval
            ).only_enforce_if(is_present_mode)
            self.cp_model.add(
                self.get_duration_expression(task) == dur
            ).only_enforce_if(is_present_mode)
        else:
            if len(duration_list) == 0:
                if len(duration_per_interval) == 1 and 0 in duration_per_interval:
                    self.cp_model.add(
                        self.get_optional_duration_of_task(task, mode) == 0
                    )
                    return
            self.cp_model.add_element(
                self.get_task_start_or_end_variable(task, StartOrEnd.START),
                expressions=duration_list,
                target=self.get_optional_duration_of_task(task, mode),
            )

    def compute_nb_preempted_tasks(self):
        if not self._nb_preempted_tasks_init:
            is_preempted_task_mode = {}
            is_different_dur = {}
            for t in self.problem.get_all_tasks_calendar_preempted():
                for mode in self.problem.get_task_modes(t):
                    is_preempted_task_mode[t, mode] = self.cp_model.NewBoolVar(
                        name=f"{t, mode}_preempted"
                    )
                    original_duration = self.problem.get_task_mode_duration(t, mode)
                    actual_dur = self.get_duration_expression(t)
                    is_present_mode = self.get_task_mode_is_present_variable(t, mode)
                    is_different_dur[t, mode] = self.cp_model.NewBoolVar(
                        name=f"{t, mode}_is_diff_duration"
                    )
                    self.cp_model.add(actual_dur != original_duration).only_enforce_if(
                        is_different_dur[t, mode]
                    )
                    self.cp_model.add(actual_dur == original_duration).only_enforce_if(
                        ~is_different_dur[t, mode]
                    )
                    self.cp_model.add(
                        is_preempted_task_mode[t, mode] == 1
                    ).only_enforce_if(is_present_mode, is_different_dur[t, mode])
                    if isinstance(is_present_mode, int):
                        self.cp_model.add(
                            is_preempted_task_mode[t, mode] == is_different_dur[t, mode]
                        )
                    else:
                        self.cp_model.add(
                            is_preempted_task_mode[t, mode] == 0
                        ).only_enforce_if(
                            is_present_mode.Not(), is_different_dur[t, mode]
                        )
                        self.cp_model.add(
                            is_preempted_task_mode[t, mode] == 0
                        ).only_enforce_if(
                            is_present_mode, is_different_dur[t, mode].Not()
                        )
                        self.cp_model.add(
                            is_preempted_task_mode[t, mode] == 0
                        ).only_enforce_if(
                            is_present_mode.Not(), is_different_dur[t, mode].Not()
                        )

            self._is_preempted = is_preempted_task_mode
            self._nb_preempted_tasks = sum(
                [is_preempted_task_mode[k] for k in is_preempted_task_mode]
            )
        return self._nb_preempted_tasks
