#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from ortools.sat.python.cp_model import LinearExprT

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
    Objective,
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)

logger = logging.getLogger(__name__)


class GenericSchedulingAutoCpSatImplSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]
):
    """Generic implementation of cpsat solver for scheduling problems (with or without allocation).

    It implements abstract class `GenericSchedulingAutoCpSatSolver`.

    """

    problem: GenericSchedulingImplProblem
    objective = Objective.CUSTOM  # do not set the objective during super().init_model()

    def __init__(
        self,
        problem: GenericSchedulingImplProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        custom_objective_factory: Optional[
            Callable[[GenericSchedulingAutoCpSatImplSolver], LinearExprT]
        ] = None,
        **kwargs: Any,
    ):
        """

        Args:
            problem:
            params_objective_function:
            custom_objective_factory: callable constructing the custom objective variable using this solver variables.
                It should correspond to `problem.custom_evaluate_fn`. It will be used as a way to compute
                the subobjective "custom" if appearing in `params_objective_function.objectives`.
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

    def init_model(
        self,
        new_horizon: Optional[int] = None,
        tasks_bounds: Optional[dict[Task, tuple[int, int, int, int]]] = None,
        use_cpm_for_task_bounds: Optional[bool] = None,
        avoid_interval_optional: Optional[bool] = None,
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
        **kwargs: Any,
    ) -> None:
        self.new_horizon = new_horizon

        # override default parameters if given, for those not already managed by parent class
        if exactly_one_unary_resource_per_task is not None:
            self.exactly_one_unary_resource_per_task = (
                exactly_one_unary_resource_per_task
            )
        if at_most_one_unary_resource_per_task is not None:
            self.at_most_one_unary_resource_per_task = (
                at_most_one_unary_resource_per_task
            )
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
            avoid_interval_optional=avoid_interval_optional,
            duplicate_start_var_per_mode=duplicate_start_var_per_mode,
            use_energy_constraints=use_energy_constraints,
            keep_only_most_nested_energy_constraints=keep_only_most_nested_energy_constraints,
            add_redundant_skill_cumulative_constraints=add_redundant_skill_cumulative_constraints,
            **kwargs,
        )

        # use the params_objective_function to define the objective
        match self.params_objective_function.objective_handling:
            case ObjectiveHandling.SINGLE:
                objective_var = self.params_objective_function.weights[
                    0
                ] * self.get_objective_variable(
                    Objective(self.params_objective_function.objectives[0])
                )
            case ObjectiveHandling.AGGREGATE:
                objective_var = sum(
                    weight * self.get_objective_variable(Objective(objective_name))
                    for objective_name, weight in zip(
                        self.params_objective_function.objectives,
                        self.params_objective_function.weights,
                    )
                )
            case _:
                raise NotImplementedError()
        if self.params_objective_function.sense_function == ModeOptim.MAXIMIZATION:
            self.cp_model.maximize(objective_var)
        else:
            self.cp_model.minimize(objective_var)

    def get_objective_variable(self, objective: Objective) -> LinearExprT:
        if objective == Objective.CUSTOM:
            if self.custom_objective_factory is None:
                raise RuntimeError(
                    "`custom_objective_factory` not defined, so `Objective.CUSTOM` cannot be translated as a cpsat variable."
                )
            return self.custom_objective_factory(self)
        else:
            return super().get_objective_variable(objective)

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericSchedulingImplSolution:
        return GenericSchedulingImplSolution(problem=self.problem, raw_sol=raw_sol)
