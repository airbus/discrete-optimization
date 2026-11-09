#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto import (
    GenericSchedulingAutoCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    LinearExpr,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)

logger = logging.getLogger(__name__)


class GenericSchedulingAutoCpImplSolver(
    GenericSchedulingAutoCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ]
):
    """Generic cp solver for scheduling problems (with or without allocation) applied to `GenericSchedulingImplProblem`."""

    problem: GenericSchedulingImplProblem

    def __init__(
        self,
        problem: GenericSchedulingImplProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        custom_objective_factory: Optional[
            Callable[[GenericSchedulingAutoCpImplSolver], LinearExpr]
        ] = None,
        **kwargs: Any,
    ):
        """

        Args:
            problem: GenericSchedulingProblem to optimize
            params_objective_function: Objective function settings to optimize
            custom_objective_factory: callable constructing the custom objective expression using this solver variables,
                corresponding to the custom objective computed by any `CustomComputer`
                in `problem.get_list_objective_computer()`.
            **kwargs:
        """
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.custom_objective_factory = custom_objective_factory

    def get_makespan_upper_bound(self) -> int:
        if self.new_horizon is None:
            return super().get_makespan_upper_bound()
        else:
            return min(self.new_horizon, super().get_makespan_upper_bound())

    def get_custom_objective_expr(self) -> LinearExpr:
        if self.custom_objective_factory is None:
            raise RuntimeError(
                "Not custom_objective_factory specified to translate the problem custom objective."
            )
        return self.custom_objective_factory(self)

    def init_model(
        self,
        new_horizon: Optional[int] = None,
        tasks_bounds: Optional[dict[Task, tuple[int, int, int, int]]] = None,
        use_cpm_for_task_bounds: Optional[bool] = None,
        avoid_interval_optional_for_unary_resources: Optional[bool] = None,
        avoid_interval_optional_for_cumulative_resources: Optional[bool] = None,
        duplicate_start_var_per_mode: Optional[bool] = None,
        use_energy_constraints: Optional[bool] = None,
        keep_only_most_nested_energy_constraints: Optional[bool] = None,
        add_redundant_skill_cumulative_constraints: Optional[bool] = None,
        exactly_one_unary_resource_per_task: Optional[bool] = None,
        at_most_one_unary_resource_per_task: Optional[bool] = None,
        use_exact_skill: Optional[bool] = None,
        use_slack_for_skill: Optional[bool] = None,
        max_slack_for_skill: Optional[int] = None,
        use_only_skill_to_allocate: Optional[bool] = None,
        use_no_overlap_for_capa_1: Optional[bool] = None,
        use_cumulative_for_capa_1: Optional[bool] = None,
        logging_objectives: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.new_horizon = new_horizon

        # override default parameters if given, for those not already managed by parent class
        if logging_objectives is not None:
            self.logging_objectives = logging_objectives
        if at_most_one_unary_resource_per_task is not None:
            self.at_most_one_unary_resource_per_task = (
                at_most_one_unary_resource_per_task
            )
        if exactly_one_unary_resource_per_task is not None:
            self.exactly_one_unary_resource_per_task = (
                exactly_one_unary_resource_per_task
            )
            if exactly_one_unary_resource_per_task:
                self.at_most_one_unary_resource_per_task = True
        if use_exact_skill is not None:
            self.use_exact_skill = use_exact_skill
        if use_slack_for_skill is not None:
            self.use_slack_for_skill = use_slack_for_skill
        if max_slack_for_skill is not None:
            self.max_slack_for_skill = max_slack_for_skill
        if use_only_skill_to_allocate is not None:
            self.use_only_skill_to_allocate = use_only_skill_to_allocate
        if use_no_overlap_for_capa_1 is not None:
            self.use_no_overlap_for_capa_1 = use_no_overlap_for_capa_1
        if use_cumulative_for_capa_1 is not None:
            self.use_cumulative_for_capa_1 = use_cumulative_for_capa_1

        super().init_model(
            tasks_bounds=tasks_bounds,
            use_cpm_for_task_bounds=use_cpm_for_task_bounds,
            avoid_interval_optional_for_unary_resources=avoid_interval_optional_for_unary_resources,
            avoid_interval_optional_for_cumulative_resources=avoid_interval_optional_for_cumulative_resources,
            duplicate_start_var_per_mode=duplicate_start_var_per_mode,
            use_energy_constraints=use_energy_constraints,
            keep_only_most_nested_energy_constraints=keep_only_most_nested_energy_constraints,
            add_redundant_skill_cumulative_constraints=add_redundant_skill_cumulative_constraints,
            **kwargs,
        )

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingImplSolution:
        return GenericSchedulingImplSolution(problem=self.problem, raw_sol=raw_sol)
