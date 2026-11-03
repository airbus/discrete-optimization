#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from functools import reduce
from typing import Generic

from discrete_optimization.generic_tasks_tools.alternative_subproblems import (
    AlternativeSchedulingProblem,
    AlternativeSchedulingSubProblem,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.base.multimode_scheduling import (
    MultimodeSchedulingCpBaseSolver,
)


class AlternativeSubproblemCpBaseSolver(
    MultimodeSchedulingCpBaseSolver[Task], Generic[Task]
):
    problem: AlternativeSchedulingProblem[Task]

    def create_alternative_subproblems_constraints(self):
        subproblems = self.problem.get_alternative_scheduling_subproblem()
        for i in range(len(subproblems)):
            self.create_alternative_path_constraint(subproblems[i], str(i), True)

    def create_alternative_path_constraint(
        self,
        alternative_problem: AlternativeSchedulingSubProblem,
        tag_alternative_problem: str,
        strict_alternative_path: bool,
    ):
        ps = [
            [p for p in path if p in self.problem.optional_tasks_list]
            for path in alternative_problem.list_paths
        ]
        sum_len = sum([len(p) for p in ps])
        merged = reduce(lambda x, y: x.union(set(y)), ps, set())
        len_merged = len(merged)
        if len_merged == sum_len:
            # Disjoint paths, nominal case.
            for p in ps:
                for p0, p1 in zip(p[:-1], p[1:]):
                    self.add_implication(
                        self.get_task_is_present_variable(p0),
                        self.get_task_is_present_variable(p1),
                    )
            nb_to_do = alternative_problem.nb_path_to_do
            if len(ps) >= nb_to_do:
                if nb_to_do == 1:
                    self.add_exactly_one(
                        [self.get_task_is_present_variable(p[0]) for p in ps]
                    )
                else:
                    self.add_bool_constraint(
                        self.sum([self.get_task_is_present_variable(p[0]) for p in ps])
                        == nb_to_do
                    )
            if alternative_problem.is_path_successors:
                for p in ps:
                    for p0, p1 in zip(p[:-1], p[1:]):
                        expr = self.get_task_start_or_end_variable(
                            p1, StartOrEnd.START
                        ) >= self.get_task_start_or_end_variable(p0, StartOrEnd.END)
                        self.enforce_only_if_tasks_present(expr=expr, tasks=(p1, p0))
                for p in alternative_problem.list_paths:
                    for p0, p1 in zip(p[:-1], p[1:]):
                        expr = self.get_task_start_or_end_variable(
                            p1, StartOrEnd.START
                        ) >= self.get_task_start_or_end_variable(p0, StartOrEnd.END)
                        self.enforce_only_if_tasks_present(
                            expr=expr,
                            tasks=(p1, p0),
                        )

                    if p[0] != alternative_problem.source_task:
                        expr = self.get_task_start_or_end_variable(
                            p[0], StartOrEnd.START
                        ) >= self.get_task_start_or_end_variable(
                            alternative_problem.source_task, StartOrEnd.END
                        )
                        self.enforce_only_if_tasks_present(
                            expr=expr,
                            tasks=(p[0], alternative_problem.source_task),
                        )
                    if p[-1] != alternative_problem.sink_task:
                        expr = self.get_task_start_or_end_variable(
                            alternative_problem.sink_task, StartOrEnd.START
                        ) >= self.get_task_start_or_end_variable(p[-1], StartOrEnd.END)
                        self.enforce_only_if_tasks_present(
                            expr=expr,
                            tasks=(alternative_problem.sink_task, p[-1]),
                        )
        else:
            path_taken = [
                self.create_bool_var(name=f"{tag_alternative_problem}_{i}")
                for i in range(len(ps))
            ]
            for i in range(len(path_taken)):
                self.add_min_equality(
                    path_taken[i],
                    [self.get_task_is_present_variable(p) for p in ps[i]],
                )
            nb_to_do = alternative_problem.nb_path_to_do
            if nb_to_do == 1:
                self.add_exactly_one(path_taken)
            else:
                self.add_bool_constraint(self.sum(path_taken) == nb_to_do)
            if alternative_problem.is_path_successors:
                for i in range(len(alternative_problem.list_paths)):
                    path = alternative_problem.list_paths[i]
                    for p0, p1 in zip(path[:-1], path[1:]):
                        (
                            self.add_implication(
                                path_taken[i],
                                self.get_task_start_or_end_variable(
                                    p1, StartOrEnd.START
                                )
                                >= self.get_task_start_or_end_variable(
                                    p0, StartOrEnd.END
                                ),
                            )
                        )
                    if path[0] != alternative_problem.source_task:
                        (
                            self.add_implication(
                                path_taken[i],
                                self.get_task_start_or_end_variable(
                                    path[0], StartOrEnd.START
                                )
                                >= self.get_task_start_or_end_variable(
                                    alternative_problem.source_task, StartOrEnd.END
                                ),
                            )
                        )
                    if path[-1] != alternative_problem.sink_task:
                        (
                            self.add_implication(
                                path_taken[i],
                                self.get_task_start_or_end_variable(
                                    alternative_problem.sink_task, StartOrEnd.START
                                )
                                >= self.get_task_start_or_end_variable(
                                    path[-1], StartOrEnd.END
                                ),
                            )
                        )
        if strict_alternative_path:
            path_used = [
                self.create_bool_var(name=f"{tag_alternative_problem}_{i}")
                for i in range(len(ps))
            ]
            for i in range(len(path_used)):
                self.add_max_equality(
                    path_used[i],
                    [self.get_task_is_present_variable(p) for p in ps[i]],
                )
            self.add_bool_constraint(
                self.sum(path_used) <= alternative_problem.nb_path_to_do
            )
