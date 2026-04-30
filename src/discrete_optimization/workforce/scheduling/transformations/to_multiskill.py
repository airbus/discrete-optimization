#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Scheduling to Multiskill RCPSP.

This provides an alternative to the standard RCPSP transformation, mapping
teams directly to employees (workers) in multiskill RCPSP.
"""

from typing import Optional

import numpy as np

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationMetadata,
    lossy_transformation,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee,
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    SkillDetail,
    SpecialConstraintsDescription,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)


class WorkforceSchedulingToMultiskillTransformation(
    ProblemTransformation[
        AllocSchedulingProblem,
        AllocSchedulingSolution,
        MultiskillRcpspProblem,
        MultiskillRcpspSolution,
    ]
):
    """Transform Workforce Scheduling to Multiskill RCPSP.

    Mapping:
    - Tasks → Tasks
    - Teams → Employees/Workers (dict indexed by team name)
    - Team availability → Employee calendars
    - Task duration → Task duration
    - Precedence → Task successors
    - Available teams for activity → Skills (ONE skill per unique eligibility pattern)

    Skill Mapping Strategy:
    Each unique set of eligible teams gets ONE skill. All teams in that set possess
    that skill. When a task requires that skill, ANY employee (team) with the skill
    can perform it. This correctly models "task needs one team from eligible set".

    Example:
    - Task A can be done by Team1 OR Team2 → Skill_A
    - Task B can be done by Team1 OR Team2 → Skill_A (same eligibility)
    - Task C can be done by Team3 → Skill_C
    - Team1 and Team2 have Skill_A; Team3 has Skill_C

    This transformation is LOSSY:
    - Same_allocation constraints are approximated
    - Cumulative resource consumption may be lost
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (WorkforceScheduling → MultiskillRCPSP).

        This direction is LOSSY but provides access to multiskill RCPSP solvers.
        """
        losses = [
            InformationLoss(
                name="same_allocation_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Tasks that must be assigned to the same team",
                reason="Multiskill RCPSP has no built-in same-employee constraint",
                impact=LossImpact.MODERATE,
                workaround="Can be approximated with additional constraints or post-processing",
            ),
            InformationLoss(
                name="cumulative_resources",
                loss_type=LossType.CONSTRAINT,
                description="Cumulative resource consumption beyond team assignment",
                reason="Direct mapping focuses on employee (team) assignment",
                impact=LossImpact.MINOR,
                workaround="Use standard RCPSP transformation for full resource modeling",
            ),
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "Teams map directly to employees/workers",
                "One skill per task (based on team eligibility)",
                "Precedence constraints preserved",
                "Time windows preserved",
            ],
            use_cases=[
                "Workforce problems with team-based allocation",
                "Access to multiskill RCPSP solvers",
                "Employee scheduling with skill requirements",
            ],
            warnings=[
                "Same_allocation constraints not enforced",
                "Verify solution feasibility in original problem",
            ],
        )

    def transform_problem(
        self, source_problem: AllocSchedulingProblem
    ) -> MultiskillRcpspProblem:
        """Transform Workforce Scheduling to Multiskill RCPSP.

        Args:
            source_problem: AllocSchedulingProblem instance

        Returns:
            Equivalent MultiskillRcpspProblem
        """
        # Create employees from teams
        employees = {}
        for i, team in enumerate(source_problem.team_names):
            # Get calendar for this team
            employee = Employee(
                dict_skill={},  # Will add skills based on tasks
                calendar_employee=source_problem.get_resource_calendar(
                    team, source_problem.horizon
                ),
            )
            employees[team] = employee  # Use team name as employee ID

        # Create tasks with skill requirements
        # Strategy: Create ONE skill per unique set of eligible teams
        # All employees in that set have that skill
        # Task requires only that ONE skill (meaning any employee with it can do the task)
        tasks = {}
        mode_details = {}
        skills_set = set()
        resources = source_problem.resources_list
        # Map from frozenset of team indices to skill name
        eligibility_to_skill = {}

        for task_idx, task in enumerate(source_problem.tasks_list):
            task_desc = source_problem.tasks_data[task]
            duration = task_desc.duration_task

            # Get eligible teams for this task
            eligible_teams = source_problem.available_team_for_activity.get(task, set())

            # If no eligible teams, make all teams eligible
            if not eligible_teams:
                eligible_teams = set(source_problem.team_names)

            # Create a skill for this eligibility pattern (use team names, not indices)
            eligibility_key = frozenset(eligible_teams)

            if eligibility_key not in eligibility_to_skill:
                # Create new skill for this eligibility pattern
                skill_name = f"skill_{len(eligibility_to_skill)}"
                eligibility_to_skill[eligibility_key] = skill_name
                skills_set.add(skill_name)

                # Add this skill to all employees (teams) in the eligibility set
                for team in eligibility_key:
                    if (
                        team in employees
                        and skill_name not in employees[team].dict_skill
                    ):
                        employees[team].dict_skill[skill_name] = SkillDetail(
                            skill_value=1, efficiency_ratio=1.0, experience=0.0
                        )

            # Get the skill for this task
            skill_name = eligibility_to_skill[eligibility_key]

            # Task requires only this ONE skill
            # Any employee with this skill can perform the task
            required_skills = {skill_name: 1}

            # Get successors from precedence constraints
            successors = source_problem.precedence_constraints.get(task, set())

            tasks[task] = {
                "duration": duration,
                "skills": required_skills,
                "successors": list(successors),
            }

            # Create mode (single mode per task)
            mode_details[task] = {1: {"duration": duration}}
            mode_details[task][1].update(required_skills)
            for r in task_desc.resource_consumption:
                if task_desc.resource_consumption[r] > 0:
                    mode_details[task][1][r] = task_desc.resource_consumption[r]

        # Build skills dictionary
        skills = {skill: None for skill in skills_set}

        # Create multiskill RCPSP problem
        normal_tasks = list(mode_details.keys())
        source_task = "source_ms"
        sink_task = "sink_ms"
        mode_details[source_task] = {1: {"duration": 0}}
        mode_details[sink_task] = {1: {"duration": 0}}
        tasks[source_task] = {"successors": normal_tasks}
        tasks[sink_task] = {"successors": []}
        for t in normal_tasks:
            tasks[t]["successors"].append(sink_task)
        start_times_window = {}
        end_times_window = {}
        for t in source_problem.start_window:
            start_times_window[t] = source_problem.start_window[t]
            end_times_window[t] = source_problem.end_window[t]

        special_constraints = SpecialConstraintsDescription(
            start_times_window=start_times_window, end_times_window=end_times_window
        )
        return MultiskillRcpspProblem(
            skills_set=set(skills.keys()),
            resources_set=set(
                source_problem.resources_list
            ),  # No non-renewable resources
            non_renewable_resources=set(),
            tasks_list=[source_task] + normal_tasks + [sink_task],
            resources_availability={
                r: source_problem.get_resource_calendar(r, source_problem.horizon)
                for r in source_problem.resources_list
            },
            special_constraints=special_constraints,
            employees=employees,
            mode_details=mode_details,
            successors={task: tasks[task]["successors"] for task in tasks},
            horizon=source_problem.horizon,
            sink_task=sink_task,
            source_task=source_task,
        )

    def back_transform_solution(
        self,
        solution: MultiskillRcpspSolution,
        source_problem: AllocSchedulingProblem,
    ) -> AllocSchedulingSolution:
        """Transform Multiskill RCPSP solution back to Workforce Scheduling.

        Args:
            solution: MultiskillRcpspSolution
            source_problem: Original AllocSchedulingProblem

        Returns:
            Equivalent AllocSchedulingSolution
        """
        n_tasks = len(source_problem.tasks_list)

        # Create schedule array
        schedule = np.zeros((n_tasks, 2), dtype=int)

        # Create allocation array
        allocation = np.zeros(n_tasks, dtype=int)

        for task_idx, task in enumerate(source_problem.tasks_list):
            # Get start time
            start = solution.schedule.get(task, {}).get("start_time", 0)
            end = solution.schedule.get(task, {}).get("end_time", 0)
            schedule[task_idx, 0] = start
            schedule[task_idx, 1] = end
            # Get employee assignment (employee ID is team name)
            employee_id_keys = solution.employee_usage.get(task, {})
            if len(employee_id_keys) > 0:
                emp_id = list(employee_id_keys)[0]
                allocation[task_idx] = source_problem.teams_to_index[emp_id]
            else:
                allocation[task_idx] = 0  # Default to first team

        return AllocSchedulingSolution(
            problem=source_problem,
            schedule=schedule,
            allocation=allocation,
        )

    def forward_transform_solution(
        self,
        solution: AllocSchedulingSolution,
        target_problem: MultiskillRcpspProblem,
    ) -> Optional[MultiskillRcpspSolution]:
        """Transform Workforce Scheduling solution to Multiskill RCPSP (for warmstart).

        Args:
            solution: AllocSchedulingSolution
            target_problem: Target MultiskillRcpspProblem

        Returns:
            Equivalent MultiskillRcpspSolution
        """
        schedule = {}
        employee_usage = {}

        for task_idx, task in enumerate(solution.problem.tasks_list):
            start = int(solution.schedule[task_idx, 0])
            end = int(solution.schedule[task_idx, 1])
            team_idx = int(solution.allocation[task_idx])

            # Convert team index to team name
            team_name = solution.problem.index_to_team.get(
                team_idx, solution.problem.team_names[0]
            )

            schedule[task] = {
                "start_time": start,
                "end_time": end,
            }

            # Employee usage (mode 1) - employee ID is team name
            employee_usage[task] = {1: team_name}

        return MultiskillRcpspSolution(
            problem=target_problem,
            schedule=schedule,
            employee_usage=employee_usage,
            modes={task: 1 for task in schedule.keys()},  # All mode 1
        )
