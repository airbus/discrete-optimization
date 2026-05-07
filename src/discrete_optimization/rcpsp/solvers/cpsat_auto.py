#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Native ortools-cpsat implementation for multimode rcpsp with resource calendar.
#  Note : Model could most likely be improved with
#  https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
import logging
from collections.abc import Iterable
from typing import Any

from ortools.sat.python.cp_model import (
    Constraint,
    CpSolverSolutionCallback,
    ObjLinearExprT,
)

from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    Objective,
    TemporarySolution,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import (
    RcpspProblem,
    RcpspSolution,
    Task,
)
from discrete_optimization.rcpsp.solution import (
    CumulativeResource,
    NonRenewableResource,
)
from discrete_optimization.rcpsp.solvers import RcpspSolver

logger = logging.getLogger(__name__)


class CpSatAutoRcpspSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, NoUnaryResource, CumulativeResource, NonRenewableResource
    ],
    RcpspSolver,
):
    problem: RcpspProblem
    variables: dict[str, Any]

    def create_mode_pair_constraint(self):
        pair_mode_constraint = self.problem.special_constraints.pair_mode_constraint
        if pair_mode_constraint is None:
            return

        if pair_mode_constraint.allowed_mode_assignment is not None:
            for task1, task2 in pair_mode_constraint.allowed_mode_assignment:
                pairs_allowed = pair_mode_constraint.allowed_mode_assignment[
                    (task1, task2)
                ]
                all_modes_task1 = set([x[0] for x in pairs_allowed])
                all_modes_task2 = set([x[1] for x in pairs_allowed])
                for mode in self.problem.get_task_modes(task1):
                    if mode not in all_modes_task1:
                        self.cp_model.add(
                            self.get_task_mode_is_present_variable(
                                task=task1, mode=mode
                            )
                            == 0
                        )
                for mode in self.problem.get_task_modes(task2):
                    if mode not in all_modes_task2:
                        self.cp_model.add(
                            self.get_task_mode_is_present_variable(
                                task=task2, mode=mode
                            )
                            == 0
                        )

                for mode1, mode2 in pairs_allowed:
                    self.cp_model.add_allowed_assignments(
                        [
                            self.get_task_mode_is_present_variable(
                                task=task1, mode=mode1
                            ),
                            self.get_task_mode_is_present_variable(
                                task=task2, mode=mode2
                            ),
                        ],
                        [(1, 1), (0, 0)],
                    )
            return
        else:
            score_task = {}
            task_mode_score = pair_mode_constraint.score_mode
            for task in set([x[0] for x in task_mode_score]):
                min_score = min(
                    task_mode_score[task, mode]
                    for mode in self.problem.get_task_modes(task)
                )
                max_score = max(
                    task_mode_score[task, mode]
                    for mode in self.problem.get_task_modes(task)
                )
                score_task[task] = self.cp_model.new_int_var(
                    lb=min_score, ub=max_score, name=f"score_{task}"
                )
                self.cp_model.add(
                    score_task[task]
                    == sum(
                        [
                            task_mode_score[task, mode]
                            * self.get_task_mode_is_present_variable(
                                task=task, mode=mode
                            )
                            for mode in self.problem.get_task_modes(task)
                        ]
                    )
                )
            for task1, task2 in pair_mode_constraint.same_score_mode:
                # way 1
                self.cp_model.add(score_task[task1] == score_task[task2])

    def add_special_temporal_constraints(
        self,
    ):
        """Add special temporal constraints to the CP model."""
        model = self.cp_model
        # start_to_start_min_time_lag: start(t1) + offset <= start(t2) where offset >= 0 (minimum time lag)
        for (
            t1,
            t2,
            offset,
        ) in self.problem.special_constraints.start_to_start_min_time_lag:
            model.add(
                self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.START
                )
                + offset
                <= self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
            )

        # start_to_start_max_time_lag: start(t2) <= start(t1) + offset where offset >= 0 (maximum time lag)
        for (
            t1,
            t2,
            offset,
        ) in self.problem.special_constraints.start_to_start_max_time_lag:
            model.add(
                self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
                <= self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.START
                )
                + offset
            )

        # start_together: start(t1) == start(t2)
        for t1, t2 in self.problem.special_constraints.start_together:
            model.add(
                self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.START
                )
                == self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
            )

        # start_at_end: end(t1) == start(t2)
        for t1, t2 in self.problem.special_constraints.start_at_end:
            model.add(
                self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.END
                )
                == self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
            )

        # start_at_end_plus_offset: end(t1) + offset <= start(t2)
        for t1, t2, offset in self.problem.special_constraints.start_at_end_plus_offset:
            model.add(
                self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.END
                )
                + offset
                <= self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
            )

        # disjunctive_tasks: tasks cannot overlap (pairwise)
        # Either end(t1) <= start(t2) OR end(t2) <= start(t1)
        for t1, t2 in self.problem.special_constraints.disjunctive_tasks:
            b = model.new_bool_var(f"disjunctive_{t1}_{t2}")
            model.add(
                self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.END
                )
                <= self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.START
                )
            ).only_enforce_if(b)
            model.add(
                self.get_task_start_or_end_variable(
                    task=t2, start_or_end=StartOrEnd.END
                )
                <= self.get_task_start_or_end_variable(
                    task=t1, start_or_end=StartOrEnd.START
                )
            ).only_enforce_if(~b)

    def init_model(self, **kwargs):
        """Init CP model."""
        include_special_constraints = kwargs.get(
            "include_special_constraints", self.problem.includes_special_constraint()
        )
        super().init_model(**kwargs)
        if include_special_constraints:
            self.create_mode_pair_constraint()
            self.add_special_temporal_constraints()

    def get_global_makespan_variable(self) -> Any:
        self.remove_constraints_on_objective()
        return self.get_task_start_or_end_variable(
            task=self.problem.sink_task, start_or_end=StartOrEnd.END
        )

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> RcpspSolution:
        schedule = {}
        modes_dict = {}
        for task, task_variable in temp_sol.task_variables.items():
            schedule[task] = {
                "start_time": task_variable.start,
                "end_time": task_variable.end,
            }
            modes_dict[task] = task_variable.mode
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )


class CpSatAutoResourceRcpspSolver(CpSatAutoRcpspSolver):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we don't sum up the resource at a given time, and it suits/makes sense mostly
    for disjunctive resource (machines)
    """

    objective = (
        Objective.CUSTOM
    )  # custom objective (linear combination of makespan and nb_used_resources)

    def init_model(self, **kwargs):
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)
        super().init_model(**kwargs)

        nb_used_resources_var = self.get_nb_resources_used_variable()
        makespan_var = self.get_global_makespan_variable()

        self.cp_model.minimize(
            weight_on_used_resource * nb_used_resources_var
            + weight_on_makespan * makespan_var
        )

    def _internal_objective(self, obj: str) -> ObjLinearExprT:
        if obj == "makespan":
            return self.get_global_makespan_variable()
        elif obj == "used_resource":
            return self.get_nb_resources_used_variable()
        else:
            raise ValueError(f"Unknown objective '{obj}'.")

    def set_lexico_objective(self, obj: str) -> None:
        """Update internal model objective.

        Args:
            obj: a string representing the desired objective.
                Should be one of "makespan" or "used_resource".

        Returns:

        """
        self.cp_model.minimize(self._internal_objective(obj))

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Constraint]:
        """

        Args:
            obj: a string representing the desired objective.
                Should be one of `self.problem.get_objective_names()`.
            value: the limiting value.
                If the optimization direction is maximizing, this is a lower bound,
                else this is an upper bound.

        Returns:
            the created constraints.

        """
        return [self.cp_model.add(self._internal_objective(obj) <= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> TemporarySolution[Task, UnaryResource]:
        """Construct each task variable from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.value(VARIABLE_NAME)`.

        We override the method from generic auto solver to add the internal objective value.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the task variables for the intermediate solution

        """
        temp_sol = super().retrieve_tasks_variables(cpsolvercb)

        temp_sol.metadata.update(
            {
                obj: cpsolvercb.value(self._internal_objective(obj))
                for obj in self.get_lexico_objectives_available()
            }
        )

        return temp_sol

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> RcpspSolution:
        """Convert temporary solution to rcpsp format.

        Add internal objectives.

        """
        sol = super().convert_task_variables_to_solution(temp_sol=temp_sol)
        sol._internal_objectives = temp_sol.metadata
        return sol

    def get_lexico_objectives_available(self) -> list[str]:
        return ["makespan", "used_resource"]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [sol._internal_objectives[obj] for sol, fit in res.list_solution_fits]
        return min(values)


class CpSatAutoCumulativeResourceRcpspSolver(CpSatAutoRcpspSolver):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we sum up the resource for each given time to do the resource optimisation.
    """

    objective = (
        Objective.CUSTOM
    )  # custom objective (linear combination of makespan and nb_used_resources)

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)

        super().init_model(**kwargs)

        resources_consumption_var = (
            self.get_aggregated_resources_consumptions_variable()
        )
        makespan_var = self.get_global_makespan_variable()

        self.cp_model.minimize(
            weight_on_used_resource * resources_consumption_var
            + weight_on_makespan * makespan_var
        )

    def _internal_objective(self, obj: str) -> ObjLinearExprT:
        if obj == "makespan":
            return self.get_global_makespan_variable()
        elif obj == "used_resource":
            return self.get_aggregated_resources_consumptions_variable()
        else:
            raise ValueError(f"Unknown objective '{obj}'.")

    def set_lexico_objective(self, obj: str) -> None:
        """Update internal model objective.

        Args:
            obj: a string representing the desired objective.
                Should be one of "makespan" or "used_resource".

        Returns:

        """
        self.cp_model.minimize(self._internal_objective(obj))

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Constraint]:
        """

        Args:
            obj: a string representing the desired objective.
                Should be one of "makespan" or "used_resource".
            value: the limiting value.
                If the optimization direction is maximizing, this is a lower bound,
                else this is an upper bound.

        Returns:
            the created constraints.

        """
        return [self.cp_model.add(self._internal_objective(obj) <= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def retrieve_tasks_variables(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> TemporarySolution[Task, UnaryResource]:
        """Construct each task variable from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.value(VARIABLE_NAME)`.

        We override the method from generic auto solver to add the internal objective value.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the task variables for the intermediate solution

        """
        temp_sol = super().retrieve_tasks_variables(cpsolvercb)

        temp_sol.metadata.update(
            {
                obj: cpsolvercb.value(self._internal_objective(obj))
                for obj in self.get_lexico_objectives_available()
            }
        )

        return temp_sol

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> RcpspSolution:
        """Convert temporary solution to rcpsp format.

        Add internal objectives.

        """
        sol = super().convert_task_variables_to_solution(temp_sol=temp_sol)
        sol._internal_objectives = temp_sol.metadata
        return sol

    def get_lexico_objectives_available(self) -> list[str]:
        return ["makespan", "used_resource"]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [sol._internal_objectives[obj] for sol, fit in res.list_solution_fits]
        return min(values)
