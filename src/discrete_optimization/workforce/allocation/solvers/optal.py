#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.workforce.allocation.problem import (
    Task,
    TeamAllocationProblem,
    TeamAllocationProblemMultiobj,
    TeamAllocationSolution,
    UnaryResource,
)
from discrete_optimization.workforce.allocation.utils import compute_all_overlapping


class OptalTeamAllocationSolver(AllocationOptalSolver[Task, UnaryResource]):
    problem: TeamAllocationProblem

    def __init__(
        self,
        problem: TeamAllocationProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        intervals_per_team = {}
        for task in self.problem.tasks_list:
            intervals[task] = self.cp_model.interval_var(
                start=int(self.problem.schedule[task][0]),
                end=int(self.problem.schedule[task][1]),
                length=int(
                    self.problem.schedule[task][1] - self.problem.schedule[task][0]
                ),
                optional=False,
                name=f"interval_{task}",
            )
            for res in self.problem.unary_resources_list:
                if self.problem.is_compatible_task_unary_resource(task, res):
                    if res not in intervals_per_team:
                        intervals_per_team[res] = {}
                    intervals_per_team[res][task] = self.cp_model.interval_var(
                        start=int(self.problem.schedule[task][0]),
                        end=int(self.problem.schedule[task][1]),
                        length=int(
                            self.problem.schedule[task][1]
                            - self.problem.schedule[task][0]
                        ),
                        optional=True,
                        name=f"interval_{task}_{res}",
                    )
            self.cp_model.alternative(
                intervals[task],
                [
                    intervals_per_team[res][task]
                    for res in intervals_per_team
                    if task in intervals_per_team[res]
                ],
            )

        for res in intervals_per_team:
            self.cp_model.no_overlap(
                [intervals_per_team[res][task] for task in intervals_per_team[res]]
            )
        used_team = {
            res: self.cp_model.max(
                [
                    self.cp_model.presence(intervals_per_team[res][task])
                    for task in intervals_per_team[res]
                ]
            )
            for res in intervals_per_team
        }
        self.variables["used"] = used_team
        self.variables["intervals"] = intervals
        self.variables["intervals_per_team"] = intervals_per_team
        self.additional_constraint()
        objectives_expr = []
        objectives_weights = []
        self.variables["objs"] = {}
        self.variables["objs"]["nb_teams"] = self.cp_model.sum(used_team.values())
        nb_max_team = len(self.problem.unary_resources_list)
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(
                        self.cp_model.interval_var(
                            start=cp.IntervalMin, end=cp.IntervalMax
                        ),
                        nb_max_team - self.variables["objs"]["nb_teams"],
                    )
                ]
                + [self.cp_model.pulse(intervals[t], 1) for t in intervals]
            )
            <= nb_max_team
        )
        set_overlaps = compute_all_overlapping(team_allocation_problem=self.problem)
        max_ = max([len(o) for o in set_overlaps])
        self.cp_model.enforce(self.variables["objs"]["nb_teams"] >= max_)
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            print(obj, weight)
            if obj == "nb_teams":
                objectives_expr += [self.variables["objs"]["nb_teams"]]
                objectives_weights += [weight]
            if isinstance(self.problem, TeamAllocationProblemMultiobj):
                if obj in self.problem.attributes_cumul_activities:
                    objectives_expr += [
                        self.add_multiobj(
                            key_objective=obj,
                        )
                    ]
                    objectives_weights += [weight]
                    self.variables["objs"][obj] = objectives_expr[-1]
        mode_optim = self.params_objective_function.sense_function
        if mode_optim == ModeOptim.MAXIMIZATION:
            self.cp_model.minimize(
                self.cp_model.sum(
                    [-w * expr for w, expr in zip(objectives_weights, objectives_expr)]
                )
            )
        else:
            self.cp_model.minimize(
                self.cp_model.sum(
                    [w * expr for w, expr in zip(objectives_weights, objectives_expr)]
                )
            )

    def additional_constraint(self):
        add = self.problem.allocation_additional_constraint
        if add.same_allocation is not None:
            for set_task in add.same_allocation:
                tasks = list(set_task)
                teams = [
                    res
                    for res in self.variables["intervals_per_team"]
                    if all(
                        task in self.variables["intervals_per_team"][res]
                        for task in tasks
                    )
                ]
                for i in range(len(tasks) - 1):
                    for team in teams:
                        self.cp_model.enforce(
                            self.cp_model.presence(
                                self.variables["intervals_per_team"][team][tasks[i]]
                            )
                            == self.cp_model.presence(
                                self.variables["intervals_per_team"][team][tasks[i + 1]]
                            )
                        )
        if add.allowed_allocation is not None:
            for task in add.allowed_allocation:
                self.cp_model.alternative(
                    self.variables["intervals"][task],
                    [
                        self.variables["intervals_per_team"][t][task]
                        for t in add.allowed_allocation[task]
                        if task in self.variables["intervals_per_team"][t]
                    ],
                )
        if add.forced_allocation is not None:
            for task in add.forced_allocation:
                self.cp_model.enforce(
                    self.get_task_unary_resource_is_present_variable(
                        task, add.forced_allocation[task]
                    )
                )
        if add.all_diff_allocation is not None:
            for tasks in add.all_diff_allocation:
                for res in self.problem.unary_resources_list:
                    sum_ = self.cp_model.sum(
                        [
                            self.get_task_unary_resource_is_present_variable(task, res)
                            for task in tasks
                        ]
                    )
                    self.cp_model.enforce(sum_ <= 1)
        if add.forbidden_allocation is not None:
            for task in add.forbidden_allocation:
                for team in add.forbidden_allocation[task]:
                    self.cp_model.enforce(
                        ~self.get_task_unary_resource_is_present_variable(task, team)
                    )
        if add.disjunction is not None:
            for list_task_team in add.disjunction:
                self.cp_model.enforce(
                    self.cp_model.max(
                        [
                            self.get_task_unary_resource_is_present_variable(task, team)
                            for task, team in list_task_team
                        ]
                    )
                    == 1
                )
        if add.nb_max_teams is not None:
            self.cp_model.enforce(
                self.cp_model.sum(self.variables["used"].values()) <= add.nb_max_teams
            )

    def add_multiobj(
        self,
        key_objective: str,
        **kwargs,
    ):
        assert isinstance(self.problem, TeamAllocationProblemMultiobj)
        values = [
            int(
                self.problem.attributes_of_activities[key_objective][
                    self.problem.activities_name[i]
                ]
            )
            for i in range(self.problem.number_of_activity)
        ]
        max_value = sum(values)
        cumul_value_nz = {}
        cumul_value = {}
        for res in self.problem.unary_resources_list:
            cumul_value[res] = self.cp_model.sum(
                [
                    values[self.problem.tasks_list.index(task)]
                    * self.cp_model.presence(
                        self.variables["intervals_per_team"][res][task]
                    )
                    for task in self.variables["intervals_per_team"][res]
                ]
            )
            cumul_value_nz[res] = self.cp_model.int_var(
                min=0, max=max_value, name=f"cumul_nz_{res}"
            )
            self.cp_model.enforce(
                self.cp_model.implies(
                    self.variables["used"][res] == 1,
                    cumul_value_nz[res] == cumul_value[res],
                )
            )
            self.cp_model.enforce(
                self.cp_model.implies(
                    self.variables["used"][res] == 0, cumul_value_nz[res] == max_value
                )
            )
        min_value = self.cp_model.min([cumul_value_nz[res] for res in cumul_value_nz])
        max_value = self.cp_model.max([cumul_value[res] for res in cumul_value_nz])
        self.cp_model.enforce(max_value - min_value >= 0)
        return max_value - min_value

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        if task in self.variables["intervals_per_team"][unary_resource]:
            return self.cp_model.presence(
                self.variables["intervals_per_team"][unary_resource][task]
            )
        return 0

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        allocation = [None for _ in range(len(self.problem.tasks_list))]
        for task in self.problem.tasks_list:
            i_task = self.problem.tasks_list.index(task)
            for res in self.variables["intervals_per_team"]:
                if task in self.variables["intervals_per_team"][res]:
                    if result.solution.is_present(
                        self.variables["intervals_per_team"][res][task]
                    ):
                        allocation[i_task] = self.problem.get_index_from_unary_resource(
                            res
                        )
        return TeamAllocationSolution(problem=self.problem, allocation=allocation)
