#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
)

from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
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
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryResource,
)

logger = logging.getLogger(__name__)


class CpSatAutoMultiskillRcpspSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
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
    use_only_skill_to_allocate = True  # do not allocate worker without relevant skills

    problem: MultiskillRcpspProblem

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> Solution:
        """Convert solution from autosolver format into do format.

        To be used in `self.retrieve_solution()`.

        Args:
            raw_sol:

        Returns:

        """
        modes_dict = {}
        schedule = {}
        employee_usage = {}
        for task, task_variable in raw_sol.task_variables.items():
            schedule[task] = {
                "start_time": task_variable.start,
                "end_time": task_variable.end,
            }
            modes_dict[task] = task_variable.mode
            employee_usage[task] = {
                worker: skills
                for worker, skills in task_variable.allocated.items()
                if len(skills) > 0
            }
        sol = MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
        sol._internal_obj = raw_sol.metadata
        return sol

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> RawSolution[Task, UnaryResource, Skill]:
        """Construct each task variable from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.value(VARIABLE_NAME)`.

        We override the method from generic auto solver to add internal objective value.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the task variables for the intermediate solution

        """
        raw_sol = super().retrieve_tasks_variables(cpsolvercb)

        # internal objective
        raw_sol.metadata["makespan"] = cpsolvercb.value(
            self.get_global_makespan_variable()
        )

        return raw_sol

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
        if resource == NB_EMPLOYEES_LB:
            return self._redundant_worker_cumulative
        else:
            return super().include_constraint_on_cumulative_resource(resource)

    def init_model(self, **kwargs: Any) -> None:
        if self.problem.is_preemptive():
            raise NotImplementedError()

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        # redundant cumulative resources to consider
        self.add_redundant_skill_cumulative_constraints = kwargs[
            "redundant_skill_cumulative"
        ]
        self._redundant_worker_cumulative = kwargs["redundant_worker_cumulative"]

        # additional constraints (we solve a subproblem)
        # allocation
        self.at_most_one_unary_resource_per_task = kwargs.get(
            "one_worker_per_task", False
        )
        # skill
        self.use_exact_skill = kwargs.get("exact_skill", False)
        self.use_slack_for_skill = kwargs.get("slack_skill", False)

        super().init_model(**kwargs)

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

    def create_total_cost_variable(self):
        self.total_cost_var = sum(
            int(10 * self.problem.employees[worker].salary)
            * is_allocated
            * self.duration_variables[task]
            for task, is_allocated_vars in self.allocation_is_present.items()
            for worker, is_allocated in is_allocated_vars.items()
        )
