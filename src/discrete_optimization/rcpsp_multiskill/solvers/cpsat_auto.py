#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
)

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    TemporarySolution,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    NB_EMPLOYEES_LB,
    CumulativeResource,
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    NonRenewableResource,
    Task,
    UnaryResource,
)

logger = logging.getLogger(__name__)


class CpSatAutoMultiskillRcpspSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ],
):
    hyperparameters = [
        CategoricalHyperparameter(
            name="redundant_skill_cumulative", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="redundant_worker_cumulative", choices=[True, False], default=True
        ),
    ]
    problem: MultiskillRcpspProblem

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> Solution:
        """Convert solution from autosolver format into do format.

        To be used in `self.retrieve_solution()`.

        Args:
            temp_sol:

        Returns:

        """
        modes_dict = {}
        schedule = {}
        employee_usage = {}
        for task, task_variable in temp_sol.task_variables.items():
            schedule[task] = {
                "start_time": task_variable.start,
                "end_time": task_variable.end,
            }
            modes_dict[task] = task_variable.mode
            employee_usage[task] = {
                worker: contrib
                for worker, contrib in task_variable.info["contrib"].items()
                if len(contrib) > 0
            }
        sol = MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
        sol._internal_obj = temp_sol.metadata
        return sol

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> TemporarySolution[Task, UnaryResource]:
        """Construct each task variable from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.value(VARIABLE_NAME)`.

        We override the method from generic auto solver to add the worker skills contribution
        and internal objective value.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the task variables for the intermediate solution

        """
        temp_sol = super().retrieve_tasks_variables(cpsolvercb)

        # worker skills contribution per task
        for task, task_variable in temp_sol.task_variables.items():
            task_variable.info["contrib"] = {}
            for worker in task_variable.allocated:
                try:
                    skills_used_vars = self.skills_used_per_worker[task][worker]
                except KeyError:
                    contrib = set()
                else:
                    contrib = {
                        s
                        for s, used_var in skills_used_vars.items()
                        if cpsolvercb.value(used_var)
                    }
                task_variable.info["contrib"][worker] = contrib

        # internal objective
        temp_sol.metadata["makespan"] = cpsolvercb.value(
            self.get_global_makespan_variable()
        )

        return temp_sol

    def include_constraint_on_cumulative_resource(
        self, resource: CumulativeResource
    ) -> bool:
        """Whether the cp model should take into account the constraint on the given cumulative resource.

        The constraints on skills as cumulative resources and on number of employees are to be added
        according to `redundant_skill_cumulative` and `redundant_worker_cumulative` hyperparameters.

        Args:
            resource:

        Returns:

        """
        if resource in self.problem.skills_set:
            return self._redundant_skill_cumulative
        elif resource == NB_EMPLOYEES_LB:
            return self._redundant_worker_cumulative
        else:
            return super().include_constraint_on_cumulative_resource(resource)

    def is_compatible_task_unary_resource(
        self, task: Task, unary_resource: UnaryResource
    ) -> bool:
        if len(self.problem.skills_of_task[task]) == 0:
            # never allocate an employee to a task not needing a skill
            return False
        elif self._one_worker_per_task:
            return any(
                all(
                    self.problem.employees[unary_resource].get_skill_level(s)
                    >= self.problem.mode_details[task][mode].get(s, 0)
                    for s in self.problem.skills_of_task[task]
                )
                for mode in self.problem.mode_details[task]
            )
        else:
            return super().is_compatible_task_unary_resource(task, unary_resource)

    def init_model(self, **kwargs: Any) -> None:
        if self.problem.is_preemptive():
            raise NotImplementedError()

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        # redundant cumulative resources to consider
        self._redundant_skill_cumulative = kwargs["redundant_skill_cumulative"]
        self._redundant_worker_cumulative = kwargs["redundant_worker_cumulative"]

        # additional constraints (subproblem)
        self._one_worker_per_task = kwargs.get("one_worker_per_task", False)
        self._one_skill_per_task = kwargs.get("one_skill_per_task", False)
        self.at_most_one_unary_resource_per_task = self._one_worker_per_task

        self._exact_skill = kwargs.get("exact_skill", False)
        self._slack_skill = kwargs.get("slack_skill", False)

        super().init_model(**kwargs)

        self.create_skills_used_per_worker()
        self.create_skills_per_task_variables()
        self.create_skills_constraint_worker()
        self.create_skills_constraints_v2()
        # self.create_workload_variables()  # not used for now

    def create_skills_used_per_worker(self):
        one_skill_per_task = self._one_skill_per_task
        skills_used_var = {}
        for task in self.problem.tasks_list:
            skills_of_task = self.problem.skills_of_task[task]
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            skills_used_var[task] = {}
            for worker in self.problem.employees:
                if self.is_compatible_task_unary_resource(
                    task=task, unary_resource=worker
                ):
                    skills_used_var[task][worker] = {}
                    skills_of_worker = self.problem.employees[
                        worker
                    ].get_non_zero_skills()
                    for s in skills_of_task:
                        if s in skills_of_worker:
                            if not one_skill_per_task or len(skills_of_worker) == 1:
                                skills_used_var[task][worker][s] = (
                                    self.allocation_is_present[task][worker]
                                )
                            else:
                                skills_used_var[task][worker][s] = (
                                    self.cp_model.new_bool_var(
                                        name=f"skill_{task}_{worker}_{s}"
                                    )
                                )
                    for s in skills_used_var[task][worker]:
                        self.cp_model.add(
                            skills_used_var[task][worker][s]
                            <= self.allocation_is_present[task][worker]
                        )
                    self.cp_model.add_bool_or(
                        [
                            skills_used_var[task][worker][s]
                            for s in skills_used_var[task][worker]
                        ]
                    ).only_enforce_if(self.allocation_is_present[task][worker])
                    if one_skill_per_task:
                        if len(skills_used_var[task][worker]) >= 1:
                            self.cp_model.add_at_most_one(
                                [
                                    skills_used_var[task][worker][s]
                                    for s in skills_used_var[task][worker]
                                ]
                            )

        self.skills_used_per_worker = skills_used_var

    def create_skills_per_task_variables(self):
        skills_var = {}
        for task in self.problem.tasks_list:
            skills_var[task] = {}
            for s in self.problem.skills_of_task[task]:
                possible_values = [
                    self.problem.mode_details[task][mode].get(s, 0)
                    for mode in self.problem.mode_details[task]
                ]
                skills_var[task][s] = self.cp_model.new_int_var_from_domain(
                    domain=Domain.from_values(possible_values),
                    name=f"skills_{task}_{s}",
                )
                for mode, is_present_mode in self.modes_is_present[task].items():
                    val = self.problem.mode_details[task][mode].get(s, 0)
                    self.cp_model.add(skills_var[task][s] == val).OnlyEnforceIf(
                        is_present_mode
                    )
        self.skills_per_task = skills_var

    def create_workload_variables(self):
        workload = {}
        for emp in self.problem.employees:
            workload[emp] = sum(
                [
                    # NB: we use duration of mode 1 instead of actual duration variable to avoid quadratic constraint
                    # (impossible in cpsat)
                    # the constraint is correct in single mode
                    self.problem.get_task_mode_duration(task=task, mode=1)
                    * is_present_task_worker[emp]
                    for task, is_present_task_worker in self.allocation_is_present.items()
                    if emp in is_present_task_worker
                ]
            )
        max_workload = self.cp_model.new_int_var(
            lb=0, ub=self.problem.horizon, name=f"max_workload"
        )
        min_workload = self.cp_model.new_int_var(
            lb=0, ub=self.problem.horizon, name=f"min_workload"
        )
        self.cp_model.add_max_equality(
            max_workload, [workload[emp] for emp in workload]
        )
        self.cp_model.add_min_equality(
            min_workload, [workload[emp] for emp in workload]
        )
        self.max_workload_var = max_workload
        self.min_workload_var = min_workload

    def create_skills_constraint_worker(self):
        exact_skill = self._exact_skill
        slack_skill = self._slack_skill
        if slack_skill:
            slack_skill_dict = {}
        for task in self.skills_per_task:
            if slack_skill:
                slack_skill_dict[task] = {}
            for s in self.skills_per_task[task]:
                if slack_skill:
                    slack_skill_dict[task][s] = self.cp_model.new_int_var(
                        lb=0, ub=5, name=f"slack_{task}_{s}"
                    )

                skill_value_put_on_task = sum(
                    skill_value * is_present_worker
                    for worker, is_present_worker in self.allocation_is_present[
                        task
                    ].items()
                    if (
                        skill_value := self.problem.employees[worker].get_skill_level(s)
                    )
                    > 0
                )

                if exact_skill:
                    if not slack_skill:
                        self.cp_model.add(
                            skill_value_put_on_task == self.skills_per_task[task][s]
                        )
                    else:
                        self.cp_model.add(
                            skill_value_put_on_task
                            == self.skills_per_task[task][s] + slack_skill_dict[task][s]
                        )

                else:
                    if not slack_skill:
                        self.cp_model.add(
                            skill_value_put_on_task >= self.skills_per_task[task][s]
                        )
                    else:
                        self.cp_model.add(
                            skill_value_put_on_task
                            >= self.skills_per_task[task][s] + slack_skill_dict[task][s]
                        )
        if slack_skill:
            self.slack_skill_v1_variables = slack_skill_dict

    def create_skills_constraints_v2(self):
        """
        using skills_used variable
        """
        exact_skill = self._exact_skill
        slack_skill = self._slack_skill
        if slack_skill:
            slack_skill_dict = {}
        for task in self.skills_per_task:
            if slack_skill:
                slack_skill_dict[task] = {}
            for s in self.skills_per_task[task]:
                if slack_skill:
                    slack_skill_dict[task][s] = self.cp_model.new_int_var(
                        lb=0, ub=5, name=f"slack_{task}_{s}"
                    )
                skill_value_put_on_task = sum(
                    self.problem.employees[worker].get_skill_level(s)
                    * skills_used_var[s]
                    for worker, skills_used_var in self.skills_used_per_worker[
                        task
                    ].items()
                    if s in skills_used_var
                )

                if exact_skill:
                    if not slack_skill:
                        self.cp_model.add(
                            skill_value_put_on_task == self.skills_per_task[task][s]
                        )
                    else:
                        self.cp_model.add(
                            skill_value_put_on_task
                            == self.skills_per_task[task][s] + slack_skill_dict[task][s]
                        )

                else:
                    if not slack_skill:
                        self.cp_model.add(
                            skill_value_put_on_task >= self.skills_per_task[task][s]
                        )
                    else:
                        self.cp_model.add(
                            skill_value_put_on_task
                            >= self.skills_per_task[task][s] + slack_skill_dict[task][s]
                        )
        if slack_skill:
            self.slack_skill_v2_variables = slack_skill_dict

    def create_total_cost_variable(self):
        self.total_cost_var = sum(
            int(10 * self.problem.employees[worker].salary)
            * is_allocated
            * self.duration_variables[task]
            for task, is_allocated_vars in self.allocation_is_present.items()
            for worker, is_allocated in is_allocated_vars.items()
        )
