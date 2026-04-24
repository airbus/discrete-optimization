#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Native ortools-cpsat implementation for multimode rcpsp with resource calendar.
#  Note : Model could most likely be improved with
#  https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
import logging
from collections.abc import Hashable, Iterable
from typing import Any, Optional

from ortools.sat.python.cp_model import (
    Constraint,
    CpModel,
    CpSolverSolutionCallback,
    IntervalVar,
    IntVar,
    LinearExpr,
    LinearExprT,
    ObjLinearExprT,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    CumulativeResourceSchedulingCpSatSolver,
    NonRenewableCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import (
    RcpspProblem,
    RcpspSolution,
    Resource,
    Task,
)
from discrete_optimization.rcpsp.solution import NonRenewableResource
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.special_constraints import PairModeConstraint
from discrete_optimization.rcpsp.utils import (
    get_end_bounds_from_additional_constraint,
    get_start_bounds_from_additional_constraint,
)

logger = logging.getLogger(__name__)


class CpSatRcpspSolver(
    CumulativeResourceSchedulingCpSatSolver[Task, Resource],
    NonRenewableCpSatSolver[Task, NonRenewableResource],
    RcpspSolver,
    WarmstartMixin,
):
    problem: RcpspProblem
    variables: dict[str, Any]

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )

    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        return self.variables["interval_var"][(task, mode)]

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            key = "start"
        else:
            key = "end"

        return self.variables[key][task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        return self.variables["is_present"][task, mode]

    def init_temporal_variable(
        self, model: CpModel
    ) -> tuple[
        dict[Hashable, IntVar],
        dict[Hashable, IntVar],
        dict[tuple[Hashable, int], IntVar],
        dict[tuple[Hashable, int], IntervalVar],
        dict[Hashable, set[tuple[Hashable, int]]],
    ]:
        starts_var = {}
        ends_var = {}
        is_present_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            lbs, ubs = get_start_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            lbe, ube = get_end_bounds_from_additional_constraint(
                rcpsp_problem=self.problem, activity=task
            )
            starts_var[task] = model.NewIntVar(lb=lbs, ub=ubs, name=f"start_{task}")
            ends_var[task] = model.NewIntVar(lb=lbe, ub=ube, name=f"end_{task}")
        interval_per_tasks = {}
        for task in self.problem.mode_details:
            interval_per_tasks[task] = set()
            for mode in self.problem.mode_details[task]:
                is_present_var[(task, mode)] = model.NewBoolVar(
                    f"is_present_{task, mode}"
                )
                interval_var[(task, mode)] = model.NewOptionalIntervalVar(
                    start=starts_var[task],
                    size=self.problem.mode_details[task][mode]["duration"],
                    end=ends_var[task],
                    is_present=is_present_var[(task, mode)],
                    name=f"interval_{task, mode}",
                )
                interval_per_tasks[task].add((task, mode))
        return starts_var, ends_var, is_present_var, interval_var, interval_per_tasks

    def add_classical_precedence_constraints(
        self,
        model: CpModel,
        starts_var: dict[Hashable, IntVar],
        ends_var: dict[Hashable, IntVar],
    ):
        # Precedence constraints
        for task in self.problem.successors:
            for successor_task in self.problem.successors[task]:
                model.Add(starts_var[successor_task] >= ends_var[task])

    def add_one_mode_selected_per_task(
        self,
        model: CpModel,
        is_present_var: dict[tuple[Hashable, int], IntVar],
        interval_per_tasks: dict[Hashable, set[tuple[Hashable, int]]],
    ):
        # 1 mode selected
        for task in interval_per_tasks:
            model.AddExactlyOne([is_present_var[k] for k in interval_per_tasks[task]])

    def create_cumulative_constraint(
        self,
        resource: str,
    ):
        if resource in self.problem.non_renewable_resources:
            self.create_non_renewable_resources_constraint(resource=resource)
        else:
            self.create_renewable_resources_constraint(resource=resource)

    def create_mode_pair_constraint(
        self,
        model: CpModel,
        interval_per_tasks: dict[Hashable, set[tuple[Hashable, int]]],
        is_present_var: dict[tuple[Hashable, int], IntVar],
        pair_mode_constraint: PairModeConstraint,
    ):
        if pair_mode_constraint.allowed_mode_assignment is not None:
            for task1, task2 in pair_mode_constraint.allowed_mode_assignment:
                pairs_allowed = pair_mode_constraint.allowed_mode_assignment[
                    (task1, task2)
                ]
                all_modes_task1 = set([x[0] for x in pairs_allowed])
                all_modes_task2 = set([x[1] for x in pairs_allowed])
                for k in interval_per_tasks[task1]:
                    if k[1] not in all_modes_task1:
                        model.Add(is_present_var[k].Not())
                for k in interval_per_tasks[task2]:
                    if k[1] not in all_modes_task2:
                        model.Add(is_present_var[k].Not())
                for mode1, mode2 in pairs_allowed:
                    model.AddAllowedAssignments(
                        [
                            is_present_var[(task1, mode1)],
                            is_present_var[(task2, mode2)],
                        ],
                        [(1, 1), (0, 0)],
                    )
            return
        else:
            score_task = {}
            task_mode_score = pair_mode_constraint.score_mode
            for task in set([x[0] for x in task_mode_score]):
                min_score = min(task_mode_score[x] for x in interval_per_tasks[task])
                max_score = max(task_mode_score[x] for x in interval_per_tasks[task])
                score_task[task] = model.NewIntVar(
                    lb=min_score, ub=max_score, name=f"score_{task}"
                )
                model.Add(
                    score_task[task]
                    == sum(
                        [
                            task_mode_score[x] * is_present_var[x]
                            for x in interval_per_tasks[task]
                        ]
                    )
                )
            for task1, task2 in pair_mode_constraint.same_score_mode:
                # way 1
                model.Add(score_task[task1] == score_task[task2])

    def add_special_temporal_constraints(
        self,
        model: CpModel,
        starts_var: dict[Hashable, IntVar],
        ends_var: dict[Hashable, IntVar],
    ):
        """Add special temporal constraints to the CP model.

        Args:
            model: The CP-SAT model
            starts_var: Dictionary mapping tasks to their start time variables
            ends_var: Dictionary mapping tasks to their end time variables
        """
        # start_to_start_min_time_lag: start(t1) + offset <= start(t2) where offset >= 0 (minimum time lag)
        for (
            t1,
            t2,
            offset,
        ) in self.problem.special_constraints.start_to_start_min_time_lag:
            model.Add(starts_var[t1] + offset <= starts_var[t2])

        # start_to_start_max_time_lag: start(t2) <= start(t1) + offset where offset >= 0 (maximum time lag)
        for (
            t1,
            t2,
            offset,
        ) in self.problem.special_constraints.start_to_start_max_time_lag:
            model.Add(starts_var[t2] <= starts_var[t1] + offset)

        # start_together: start(t1) == start(t2)
        for t1, t2 in self.problem.special_constraints.start_together:
            model.Add(starts_var[t1] == starts_var[t2])

        # start_at_end: end(t1) == start(t2)
        for t1, t2 in self.problem.special_constraints.start_at_end:
            model.Add(ends_var[t1] == starts_var[t2])

        # start_at_end_plus_offset: end(t1) + offset <= start(t2)
        for t1, t2, offset in self.problem.special_constraints.start_at_end_plus_offset:
            model.Add(ends_var[t1] + offset <= starts_var[t2])

        # disjunctive_tasks: tasks cannot overlap (pairwise)
        # Either end(t1) <= start(t2) OR end(t2) <= start(t1)
        for t1, t2 in self.problem.special_constraints.disjunctive_tasks:
            b = model.NewBoolVar(f"disjunctive_{t1}_{t2}")
            model.Add(ends_var[t1] <= starts_var[t2]).OnlyEnforceIf(b)
            model.Add(ends_var[t2] <= starts_var[t1]).OnlyEnforceIf(b.Not())

        # start_times_window: lower_bound <= start(task) <= upper_bound
        for task in self.problem.special_constraints.start_times_window:
            lower, upper = self.problem.special_constraints.start_times_window[task]
            if lower is not None:
                model.Add(starts_var[task] >= lower)
            if upper is not None:
                model.Add(starts_var[task] <= upper)

        # end_times_window: lower_bound <= end(task) <= upper_bound
        for task in self.problem.special_constraints.end_times_window:
            lower, upper = self.problem.special_constraints.end_times_window[task]
            if lower is not None:
                model.Add(ends_var[task] >= lower)
            if upper is not None:
                model.Add(ends_var[task] <= upper)

    def init_model(self, **kwargs):
        """Init CP model."""
        include_special_constraints = kwargs.get(
            "include_special_constraints", self.problem.includes_special_constraint()
        )
        super().init_model(**kwargs)
        model = self.cp_model
        (
            starts_var,
            ends_var,
            is_present_var,
            interval_var,
            interval_per_tasks,
        ) = self.init_temporal_variable(model=model)
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "interval_var": interval_var,
        }
        self.add_one_mode_selected_per_task(
            model=model,
            is_present_var=is_present_var,
            interval_per_tasks=interval_per_tasks,
        )
        self.add_classical_precedence_constraints(
            model=model, starts_var=starts_var, ends_var=ends_var
        )
        resources = self.problem.resources_list
        for resource in resources:
            self.create_cumulative_constraint(
                resource=resource,
            )
        if include_special_constraints:
            if self.problem.special_constraints.pair_mode_constraint is not None:
                self.create_mode_pair_constraint(
                    model=model,
                    interval_per_tasks=interval_per_tasks,
                    is_present_var=is_present_var,
                    pair_mode_constraint=self.problem.special_constraints.pair_mode_constraint,
                )
            self.add_special_temporal_constraints(
                model=model,
                starts_var=starts_var,
                ends_var=ends_var,
            )
        objective = self.get_global_makespan_variable()
        self.minimize_variable(objective)

    def get_global_makespan_variable(self) -> Any:
        self.remove_constraints_on_objective()
        return self.variables["end"][self.problem.sink_task]

    def set_warm_start(self, solution: RcpspSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for task in self.variables["start"]:
            self.cp_model.AddHint(
                self.variables["start"][task],
                solution.rcpsp_schedule[task]["start_time"],
            )
            self.cp_model.AddHint(
                self.variables["end"][task], solution.rcpsp_schedule[task]["end_time"]
            )
        for task, mode in zip(self.problem.tasks_list_non_dummy, solution.rcpsp_modes):
            self.cp_model.AddHint(self.variables["is_present"][task, mode], 1)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RcpspSolution:
        """Construct a do solution from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.Value(VARIABLE_NAME)`.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the intermediate solution, at do format.

        """
        schedule = {}
        modes_dict = {}
        for task in self.variables["start"]:
            schedule[task] = {
                "start_time": cpsolvercb.Value(self.variables["start"][task]),
                "end_time": cpsolvercb.Value(self.variables["end"][task]),
            }
        for task, mode in self.variables["is_present"]:
            if cpsolvercb.Value(self.variables["is_present"][task, mode]):
                modes_dict[task] = mode
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )


class CpSatResourceRcpspSolver(CpSatRcpspSolver):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we don't sum up the resource at a given time, and it suits/makes sense mostly
    for disjunctive resource (machines)
    """

    def create_cumulative_constraint_and_used_resource(
        self,
        model: CpModel,
        resource: str,
        is_used_resource: dict[str, IntVar],
        is_present_var: dict[tuple[Hashable, int], IntVar],
    ):
        if resource in self.problem.non_renewable_resources:
            task_modes_consuming = [
                (
                    (task, mode),
                    int(self.problem.mode_details[task][mode].get(resource, 0)),
                )
                for task in self.problem.tasks_list
                for mode in self.problem.mode_details[task]
                if self.problem.mode_details[task][mode].get(resource, 0) > 0
            ]
            model.Add(
                sum([is_present_var[x[0]] * x[1] for x in task_modes_consuming])
                <= int(self.problem.get_max_resource_capacity(resource))
            )
            model.AddMaxEquality(
                is_used_resource[resource],
                [is_present_var[x[0]] for x in task_modes_consuming],
            )
        else:
            self.create_renewable_resources_constraint(resource=resource)

            resource_is_present_vars = [
                self.get_task_mode_is_present_variable(task=task, mode=mode)
                for task in self.problem.tasks_list
                for mode in self.problem.get_task_modes(task=task)
                if self.problem.get_renewable_resource_consumption(
                    resource=resource, task=task, mode=mode
                )
                > 0
            ]
            if len(resource_is_present_vars) > 0:
                model.add_max_equality(
                    is_used_resource[resource], resource_is_present_vars
                )

    def init_model(self, **kwargs):
        include_special_constraints = kwargs.get(
            "include_special_constraints", self.problem.includes_special_constraint()
        )
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)
        super().init_model(**kwargs)
        model = self.cp_model
        (
            starts_var,
            ends_var,
            is_present_var,
            interval_var,
            interval_per_tasks,
        ) = self.init_temporal_variable(model=model)
        resources = self.problem.resources_list
        is_used_resource = {res: model.NewBoolVar(f"used_{res}") for res in resources}
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "is_used_resource": is_used_resource,
            "interval_var": interval_var,
        }
        self.add_one_mode_selected_per_task(
            model=model,
            is_present_var=is_present_var,
            interval_per_tasks=interval_per_tasks,
        )
        self.add_classical_precedence_constraints(
            model=model, starts_var=starts_var, ends_var=ends_var
        )
        for resource in resources:
            self.create_cumulative_constraint_and_used_resource(
                model=model,
                resource=resource,
                is_used_resource=is_used_resource,
                is_present_var=is_present_var,
            )
        if include_special_constraints:
            if self.problem.special_constraints.pair_mode_constraint is not None:
                self.create_mode_pair_constraint(
                    model=model,
                    interval_per_tasks=interval_per_tasks,
                    is_present_var=is_present_var,
                    pair_mode_constraint=self.problem.special_constraints.pair_mode_constraint,
                )
            self.add_special_temporal_constraints(
                model=model,
                starts_var=starts_var,
                ends_var=ends_var,
            )
        model.Minimize(
            weight_on_used_resource
            * sum([is_used_resource[x] for x in is_used_resource])
            + weight_on_makespan * starts_var[self.problem.sink_task]
        )

    def _internal_used_resource(self) -> LinearExpr:
        return sum(self.variables["is_used_resource"].values())

    def _internal_objective(self, obj: str) -> ObjLinearExprT:
        if obj == "makespan":
            return self.get_global_makespan_variable()
        elif obj == "used_resource":
            return self._internal_used_resource()
        else:
            raise ValueError(f"Unknown objective '{obj}'.")

    def set_lexico_objective(self, obj: str) -> None:
        """Update internal model objective.

        Args:
            obj: a string representing the desired objective.
                Should be one of "makespan" or "used_resource".

        Returns:

        """
        self.cp_model.Minimize(self._internal_objective(obj))

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
        return [self.cp_model.Add(self._internal_objective(obj) <= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RcpspSolution:
        sol = super().retrieve_solution(cpsolvercb)
        sol._internal_objectives = {
            obj: cpsolvercb.value(self._internal_objective(obj))
            for obj in self.get_lexico_objectives_available()
        }
        return sol

    def get_lexico_objectives_available(self) -> list[str]:
        return ["makespan", "used_resource"]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [sol._internal_objectives[obj] for sol, fit in res.list_solution_fits]
        return min(values)


class CpSatCumulativeResourceRcpspSolver(CpSatRcpspSolver):
    """
    Specific solver to minimize the minimum resource amount needed to accomplish the scheduling problem.
    In this version we sum up the resource for each given time to do the resource optimisation.
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="use_overlap_for_disjunctive_resource",
            default=True,
            choices=[True, False],
        )
    ]

    def create_resource_capacity_var(self, model: CpModel):
        resource_capacity_var = {}
        for resource in self.problem.resources_list:
            resource_capacity_var[resource] = model.NewIntVar(
                lb=0,
                ub=self.problem.get_max_resource_capacity(resource),
                name=f"res_{resource}",
            )
        return resource_capacity_var

    def create_cumulative_constraint_and_resource_capa(
        self,
        model: CpModel,
        resource: str,
        resource_capacity_var: dict[str, IntVar],
        interval_var: dict[tuple[Hashable, int], IntervalVar],
        is_present_var: dict[tuple[Hashable, int], IntVar],
        use_overlap_for_disjunctive_resource: bool,
    ):
        if resource in self.problem.non_renewable_resources:
            task_modes_consuming = [
                (
                    (task, mode),
                    self.problem.mode_details[task][mode].get(resource, 0),
                )
                for task in self.problem.tasks_list
                for mode in self.problem.mode_details[task]
                if self.problem.mode_details[task][mode].get(resource, 0) > 0
            ]
            model.Add(
                sum([is_present_var[x[0]] * x[1] for x in task_modes_consuming])
                <= resource_capacity_var[resource]
            )
        else:
            self.create_renewable_resources_constraint(resource=resource)
            task_modes_consuming = [
                (
                    (task, mode),
                    self.problem.mode_details[task][mode].get(resource, 0),
                )
                for task in self.problem.tasks_list
                for mode in self.problem.mode_details[task]
                if self.problem.mode_details[task][mode].get(resource, 0) > 0
            ]
            if len(task_modes_consuming) == 0:
                # when empty, the constraints don't work !
                return
            capacity = self.problem.get_resource_max_capacity(resource)
            if capacity > 1 or (
                capacity == 1 and not use_overlap_for_disjunctive_resource
            ):
                model.AddCumulative(
                    [interval_var[x[0]] for x in task_modes_consuming],
                    demands=[x[1] for x in task_modes_consuming],
                    capacity=resource_capacity_var[resource],
                )
                for x in task_modes_consuming:
                    model.Add(
                        resource_capacity_var[resource] >= x[1] * is_present_var[x[0]]
                    )
            elif capacity == 1:
                model.AddMaxEquality(
                    resource_capacity_var[resource],
                    [is_present_var[x[0]] for x in task_modes_consuming],
                )

    def init_model(self, **kwargs):
        include_special_constraints = kwargs.get(
            "include_special_constraints", self.problem.includes_special_constraint()
        )
        weight_on_makespan = kwargs.get("weight_on_makespan", 1)
        weight_on_used_resource = kwargs.get("weight_on_used_resource", 10000)
        use_overlap_for_disjunctive_resource = kwargs.get(
            "use_overlap_for_disjunctive_resource", True
        )
        super().init_model(**kwargs)
        model = self.cp_model
        (
            starts_var,
            ends_var,
            is_present_var,
            interval_var,
            interval_per_tasks,
        ) = self.init_temporal_variable(model=model)
        resource_capacity_var = self.create_resource_capacity_var(model=model)
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "resource_capacity": resource_capacity_var,
            "interval_var": interval_var,
        }
        self.add_classical_precedence_constraints(
            model=model, starts_var=starts_var, ends_var=ends_var
        )
        self.add_one_mode_selected_per_task(
            model=model,
            is_present_var=is_present_var,
            interval_per_tasks=interval_per_tasks,
        )
        resources = self.problem.resources_list
        for resource in resources:
            self.create_cumulative_constraint_and_resource_capa(
                model=model,
                resource=resource,
                resource_capacity_var=resource_capacity_var,
                interval_var=interval_var,
                is_present_var=is_present_var,
                use_overlap_for_disjunctive_resource=use_overlap_for_disjunctive_resource,
            )
        if include_special_constraints:
            if self.problem.special_constraints.pair_mode_constraint is not None:
                self.create_mode_pair_constraint(
                    model=model,
                    interval_per_tasks=interval_per_tasks,
                    is_present_var=is_present_var,
                    pair_mode_constraint=self.problem.special_constraints.pair_mode_constraint,
                )
            self.add_special_temporal_constraints(
                model=model,
                starts_var=starts_var,
                ends_var=ends_var,
            )

        model.Minimize(
            weight_on_used_resource
            * sum([resource_capacity_var[x] for x in resource_capacity_var])
            + weight_on_makespan * starts_var[self.problem.sink_task]
        )

    def _internal_used_resource(self) -> LinearExpr:
        return sum(self.variables["resource_capacity"].values())

    def _internal_objective(self, obj: str) -> ObjLinearExprT:
        if obj == "makespan":
            return self.get_global_makespan_variable()
        elif obj == "used_resource":
            return self._internal_used_resource()
        else:
            raise ValueError(f"Unknown objective '{obj}'.")

    def set_lexico_objective(self, obj: str) -> None:
        """Update internal model objective.

        Args:
            obj: a string representing the desired objective.
                Should be one of "makespan" or "used_resource".

        Returns:

        """
        self.cp_model.Minimize(self._internal_objective(obj))

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
        return [self.cp_model.Add(self._internal_objective(obj) <= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RcpspSolution:
        sol = super().retrieve_solution(cpsolvercb)
        sol._internal_objectives = {
            obj: cpsolvercb.value(self._internal_objective(obj))
            for obj in self.get_lexico_objectives_available()
        }
        return sol

    def get_lexico_objectives_available(self) -> list[str]:
        return ["makespan", "used_resource"]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        values = [sol._internal_objectives[obj] for sol, fit in res.list_solution_fits]
        return min(values)
