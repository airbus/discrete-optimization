#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from typing import Dict, Hashable, List, Set, Tuple, Type

from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    WithoutAllocationProblem,
    WithoutAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    convert_calendar_to_availability_intervals,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.multimode import MultimodeSolution
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingSolution
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    NoSkill,
    Skill,
    WithoutSkillProblem,
    WithoutSkillSolution,
)
from discrete_optimization.generic_tools.do_problem import *

RESOURCE_KEY = Hashable
TASK_KEY = Hashable
GROUP_KEY = Hashable
Resource = RESOURCE_KEY
CumulativeResource = RESOURCE_KEY
NonRenewableResource = RESOURCE_KEY
Task = TASK_KEY


@dataclass
class TaskData:
    duration: int
    resource_consumption: Dict[RESOURCE_KEY, int]
    # Specify if the task can be paused during a resource break and resumed after,
    # this is a simplified preemption which is quite managable to deal with in mathematical models, in practice the task
    # will span over a longer duration than expected when there is a break inside.
    preemptive_on_resource_break: bool

    def get_res_consumption(self, res: RESOURCE_KEY):
        return self.resource_consumption.get(res, 0)


@dataclass
class TaskObject:
    id: TASK_KEY
    # more natural name if existing
    name: str
    # mode of execution of the task, it specifies different ways of doing the same task.
    modes: Dict[int, TaskData]
    # release date
    min_starting_date: Optional[int] = None
    # deadline on starting date
    max_starting_date: Optional[int] = None
    # deadline date
    max_ending_date: Optional[int] = None
    # "release" date of ending
    min_ending_date: Optional[int] = None

    # if max_ending_date should be considered really as hard.
    soft_max_end_date: bool = False

    # Price per unit time after finishing the task. Main parameter
    # to calculate WIP Objective as this price times the delta
    # of user end date and actual solution end date will be accounted
    # as wip.
    # To be improved: define this price per mode
    # Breakdown of the price:
    #    - Initial price: Price if the task has been already started
    #    - Delivery price: Fixed price to add at the end of the task
    #    - Resource price: This price should be variable, taking into account
    #                      actual subintervals where the task runs. Simplified
    #                      as the total price for the resources
    price: int = 1
    modes_id_to_index: dict[int, int] = field(init=False, default=None)

    def __post_init__(self):
        sorted_modes = sorted(list(self.modes.keys()))
        self.modes_id_to_index = {sorted_modes[i]: i for i in range(len(sorted_modes))}


class GroupType(Enum):
    SUBGROUP_TASK_FOR_OBJECTIVE = 0
    GROUP_TASK_NON_RELEASED_RESOURCE = 1
    TASK_RELEASE_MODE = 2
    GROUP_GENERIC_RELEASE = 3


@dataclass
class TasksGroups:
    # Typically a structure to group task of the same MSN aircraft, that could intervene in the objective functions !
    id: GROUP_KEY
    name: str
    tasks_group: Set[TASK_KEY]
    # if by chance we know the first task of this set of tasks
    first_task_if_any: Optional[TASK_KEY] = None
    # if by chance we know the last task of this set of tasks
    last_task_if_any: Optional[TASK_KEY] = None
    type_of_group: GroupType = GroupType.SUBGROUP_TASK_FOR_OBJECTIVE
    # if the group is of time "GROUP_TASK_NON_RELEASED_RESOURCE", then the span of group of tasks
    # consumes a given quantity of resource in this dictionary !
    res_not_released: Optional[Dict[RESOURCE_KEY, int]] = None
    # it the corresponding task should not overlap
    no_overlap: Optional[bool] = False

    # release date of the group
    min_starting_date: Optional[int] = None
    # "deadline" on starting date of the group
    max_starting_date: Optional[int] = None
    # deadline date of the task group
    max_ending_date: Optional[int] = None
    # "release" date of ending of the task group
    min_ending_date: Optional[int] = None

    #
    soft_max_end_date: bool = False


@dataclass
class TaskGroupAbstraction:
    is_a_task: bool
    task_id: TASK_KEY
    group_id: GROUP_KEY


@dataclass
class ResourceData:
    id: RESOURCE_KEY
    name: Optional[str]
    calendar_availability: np.ndarray
    renewable: bool
    max_capacity: int
    is_disjunctive: bool
    is_station: bool
    is_operator: bool
    child_resource: Set[RESOURCE_KEY] = None  # field(init=False, default=None)


@dataclass
class ConstraintsTask:
    # for each s in successors, for each succ in successors[s], start[succ] >= end[s]
    successors: Dict[TASK_KEY, Set[TASK_KEY]]

    # successors + non releasable resource relation : the resource of task is released at the start of the successor.
    # (t1, t2, {res: 1}) : res is consumed from end of task t1 to start of task t2
    successor_with_res_release_at_start_of_successor: list[
        Tuple[TASK_KEY, TASK_KEY, dict[RESOURCE_KEY, int]]
    ] = None
    # Initially, this could be
    # ((t1, mode1), t2, {res:1}) : if t1 is in mode1, then the res is consumed from end of t1 to start of task t2
    successor_with_res_release_at_start_of_successor_mode: list[
        Tuple[Tuple[TASK_KEY, int], TASK_KEY, dict[RESOURCE_KEY, int]]
    ] = None

    # object to handle resource blocked between end of task (or group of task)
    # and start of another task (or group of task)
    successor_generic_with_res_release_at_start_of_successor_generic: list[
        Tuple[TaskGroupAbstraction, TaskGroupAbstraction, dict[RESOURCE_KEY, int]]
    ] = None

    # successors link between group of Tasks
    successors_group_tasks: Optional[Dict[GROUP_KEY, Set[GROUP_KEY]]] = None

    # grouped task sharing some same resource that should not be released on the all span of the group of task
    # The set of resource that is specified will not be released over the set of task.
    # grouped_tasks: Optional[List[Tuple[Set[TASK_KEY], Set[RESOURCE_KEY]]]] = None

    # start at start data
    start_at_start: Optional[List[Tuple[TASK_KEY, TASK_KEY]]] = None

    # start at end
    start_at_end: Optional[List[Tuple[TASK_KEY, TASK_KEY]]] = None

    # st[e[0]] >= end[e[1]]+e[2]
    start_after_end_plus_offset: Optional[List[Tuple[TASK_KEY, TASK_KEY, int]]] = None
    # st[e[0]] == end[e[1]]+e[2]
    start_at_end_plus_offset: Optional[List[Tuple[TASK_KEY, TASK_KEY, int]]] = None
    # st[e[0]] >= st[e[1]]+e[2]
    start_after_start_plus_offset: Optional[List[Tuple[TASK_KEY, TASK_KEY, int]]] = None
    # st[e[0]] == st[e[1]]+e[2]
    start_at_start_plus_offset: Optional[List[Tuple[TASK_KEY, TASK_KEY, int]]] = None


class ObjectivesEnum(Enum):
    MAKESPAN = 0
    RESOURCE_COST = 1
    WORK_IN_PROGRESS = 2
    TARDINESS = 3
    EARLINESS = 4
    NON_RELEASE_DURATION = 5


@dataclass
class ObjectiveParamWIP:
    # Weight for the optimization
    weight: int
    weight_per_task: Dict[TASK_KEY, float]
    weights_per_group_task: Dict[GROUP_KEY, float]
    count_nb_group_in_progress: bool = True
    coefficient_on_nb_group_in_progress: float = 0


@dataclass
class ObjectiveParamResource:
    # here we want to minimize the resource capacity of some resources,
    # the unit cost of 1 capacity can be given with this dictionary
    weight_per_resource_unit: Dict[RESOURCE_KEY, float]

    # Here to discard or not some resource from our optimisation.
    # Typically some resources in the data like the station,
    # can be ignored from the optim, we have 1 and it's not flexible.
    consider_in_objectives: Dict[RESOURCE_KEY, bool]

    # Weight for the optimization
    weight: int


@dataclass
class ObjectiveParamTardiness:
    """Specify for meaningful task of group task some weight to put on the tardiness cost"""

    weight_per_task: Dict[TASK_KEY, float]
    weight_per_groups: Dict[GROUP_KEY, float]


@dataclass
class ObjectiveParamEarliness:
    """Specify for meaningful task of group task some weight to put on the tardiness cost"""

    weight_per_task: Dict[TASK_KEY, float]
    weight_per_groups: Dict[GROUP_KEY, float]


@dataclass
class ObjectiveParams:
    params_obj: Dict[
        ObjectivesEnum,
        Union[
            ObjectiveParamWIP,
            ObjectiveParamResource,
            ObjectiveParamEarliness,
            ObjectiveParamTardiness,
            float,
        ],
    ]


class ScheduleSolution(
    GenericSchedulingSolution[
        Task, NoUnaryResource, NoSkill, NonSkillCumulativeResource, NonRenewableResource
    ],
    WithoutSkillSolution[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutAllocationSolution[Task],
):
    problem: "FlexProblem"

    def __init__(self, problem: "FlexProblem", schedule: np.ndarray, modes: np.ndarray):
        super().__init__(problem)
        self.schedule = schedule
        self.modes = modes

    def get_end_time(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return int(self.schedule[index, 1])

    def get_start_time(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return int(self.schedule[index, 0])

    def get_mode(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return int(self.modes[index])

    def copy(self) -> "Solution":
        return ScheduleSolution(
            problem=self.problem,
            schedule=np.copy(self.schedule),
            modes=np.copy(self.modes),
        )


class ScheduleSolutionPreemptive(SchedulingSolution[Task], MultimodeSolution[Task]):
    problem: "FlexProblem"

    def __init__(
        self,
        problem: "FlexProblem",
        schedule: list[list[tuple[int, int]]],
        modes: np.ndarray,
    ):
        super().__init__(problem)
        self.schedule = schedule
        self.modes = modes

    def get_mode(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return self.modes[index]

    def get_end_time(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return self.schedule[index][-1][1]

    def get_start_time(self, task: Task) -> int:
        index = self.problem.task_id_to_index[task]
        return self.schedule[index][0][0]

    def copy(self) -> "Solution":
        return ScheduleSolutionPreemptive(
            problem=self.problem,
            schedule=deepcopy(self.schedule),
            modes=np.copy(self.modes),
        )


class FlexProblem(
    GenericSchedulingProblem[
        Task, NoUnaryResource, NoSkill, NonSkillCumulativeResource, NonRenewableResource
    ],
    WithoutSkillProblem[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutAllocationProblem[Task],
):
    def get_no_overlap(self) -> set[frozenset[Task]]:
        return set()

    @property
    def non_skill_cumulative_resources_list(self) -> list[Skill]:
        return [resource.id for resource in self.resources if resource.renewable]

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        if mode in self.get_task_modes(task):
            return (
                self.task_id_dict[task]
                .modes[mode]
                .resource_consumption.get(resource, 0)
            )
        return 0

    @cache
    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return convert_calendar_to_availability_intervals(
            calendar=self.resource_dict[resource].calendar_availability,
            horizon=self.horizon,
        )

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        if mode in self.get_task_modes(task):
            return self.task_id_dict[task].modes[mode].duration
        return 0

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return [r.id for r in self.resources if not r.renewable]

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        return self.resource_dict[resource].max_capacity

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        if mode in self.get_task_modes(task):
            return (
                self.task_id_dict[task]
                .modes[mode]
                .resource_consumption.get(resource, 0)
            )
        return 0

    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
        return self.constraints.successors

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        if start_or_end == StartOrEnd.START:
            return self.min_start_time[self.task_id_to_index[task]]
        if start_or_end == StartOrEnd.END:
            return self.min_end_time[self.task_id_to_index[task]]
        return 0

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        if start_or_end == StartOrEnd.START:
            return self.max_start_time[self.task_id_to_index[task]]
        if start_or_end == StartOrEnd.END:
            return self.max_end_time[self.task_id_to_index[task]]
        return self.get_makespan_upper_bound()

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.task_id_dict[task].modes)

    @property
    def tasks_list(self) -> list[Task]:
        return self.tasks_ids

    def __init__(
        self,
        resources: List[ResourceData],
        tasks: List[TaskObject],
        tasks_group: List[TasksGroups],
        constraints: ConstraintsTask,
        objective_params: ObjectiveParams,
        horizon: int,
    ):
        from .fsp_utils import (
            compute_duration_tasks_function_time,
            get_lb_ub_start_end_date,
        )

        self.resources = resources
        self.tasks = tasks
        self.tasks_group = tasks_group
        self.objective_params = objective_params
        self.horizon = horizon
        self.tasks_ids = [self.tasks[i].id for i in range(len(self.tasks))]
        assert len(set(self.tasks_ids)) == len(self.tasks)
        self.constraints = constraints
        self.nb_tasks = len(self.tasks)
        self.task_id_to_index = {self.tasks[i].id: i for i in range(self.nb_tasks)}
        self.index_to_task_id = {i: self.tasks[i].id for i in range(self.nb_tasks)}
        self.index_task_dict = {
            index: self.tasks[index] for index in self.index_to_task_id
        }
        self.task_id_dict: dict[TASK_KEY, TaskObject] = {
            self.tasks[index].id: self.tasks[index] for index in self.index_to_task_id
        }
        assert len(
            set([self.resources[i].id for i in range(len(self.resources))])
        ) == len(resources)
        self.nb_resources = len(resources)
        self.resource_dict: Dict[RESOURCE_KEY, ResourceData] = {
            r.id: r for r in self.resources
        }
        self.resource_id_to_index = {
            self.resources[i].id: i for i in range(self.nb_resources)
        }
        self.group_id_to_index = {
            self.tasks_group[i].id: i for i in range(len(self.tasks_group))
        }
        self.durations_data, self.res_arrays_data = (
            compute_duration_tasks_function_time(self)
        )
        (
            self.min_start_time,
            self.max_start_time,
            self.min_end_time,
            self.max_end_time,
        ) = get_lb_ub_start_end_date(self)

    def update_data_placeholders(self):
        """If data has been changed in place, we should change the other utils attributes"""
        from .fsp_utils import compute_duration_tasks_function_time

        self.tasks_ids = [self.tasks[i].id for i in range(len(self.tasks))]
        assert len(set(self.tasks_ids)) == len(self.tasks)
        self.nb_tasks = len(self.tasks)
        self.task_id_to_index = {self.tasks[i].id: i for i in range(self.nb_tasks)}
        self.index_to_task_id = {i: self.tasks[i].id for i in range(self.nb_tasks)}
        self.index_task_dict = {
            index: self.tasks[index] for index in self.index_to_task_id
        }
        self.task_id_dict: dict[TASK_KEY, TaskObject] = {
            self.tasks[index].id: self.tasks[index] for index in self.index_to_task_id
        }
        self.nb_resources = len(self.resources)
        self.resource_dict: Dict[RESOURCE_KEY, ResourceData] = {
            r.id: r for r in self.resources
        }
        self.resource_id_to_index = {
            self.resources[i].id: i for i in range(self.nb_resources)
        }
        self.group_id_to_index = {
            self.tasks_group[i].id: i for i in range(len(self.tasks_group))
        }
        self.get_resource_availabilities.cache_clear()
        self.durations_data, self.res_arrays_data = (
            compute_duration_tasks_function_time(self)
        )
        (
            self.min_start_time,
            self.max_start_time,
            self.min_end_time,
            self.max_end_time,
        ) = get_lb_ub_start_end_date(self)

    @staticmethod
    def _compute_resource_usage_preemptive(
        problem: "FlexProblem",
        schedule: list[list[tuple[int, int]]],
        modes: np.ndarray,
    ) -> Dict[RESOURCE_KEY, np.ndarray]:
        """Compute resource usage over time for a preemptive schedule."""
        resource_usage = {
            r.id: np.zeros(problem.horizon, dtype=int) for r in problem.resources
        }

        for task_idx in range(problem.nb_tasks):
            mode = int(modes[task_idx])
            task_data = problem.tasks[task_idx].modes[mode]

            # For each interval in the preemptive schedule
            for start, end in schedule[task_idx]:
                for t in range(start, end):
                    for res_id, consumption in task_data.resource_consumption.items():
                        resource_usage[res_id][t] += consumption

        return resource_usage

    def evaluate(self, variable: ScheduleSolution) -> Dict[str, Any]:
        """
        Evaluate the solution to compute KPIs matching the CP solver objectives.
        Returns a dictionary containing:
        - makespan
        - resource_consumption
        - tardiness (if applicable)
        - earliness (if applicable)
        - resource_cost (if applicable)
        - wip_cost (if applicable)
        """
        from .fsp_utils import SolutionDetails

        if isinstance(variable, ScheduleSolutionPreemptive):
            makespan = max(
                [variable.schedule[i][-1][1] for i in range(len(variable.schedule))]
            )
            resource_consumption = self._compute_resource_usage_preemptive(
                self, variable.schedule, variable.modes
            )

            # Initialize all objectives
            evaluation = {
                "makespan": makespan,
                "tardiness": 0.0,
                "earliness": 0.0,
                "resource_cost": 0.0,
                "wip_cost": 0.0,
            }

            # --- Tardiness ---
            if ObjectivesEnum.TARDINESS in self.objective_params.params_obj:
                obj_param = self.objective_params.params_obj[ObjectivesEnum.TARDINESS]
                tardiness = 0.0

                for t_id, weight in obj_param.weight_per_task.items():
                    if weight > 0:
                        idx = self.task_id_to_index[t_id]
                        deadline = self.tasks[idx].max_ending_date
                        if deadline is not None:
                            end_time = variable.schedule[idx][-1][1]
                            tardiness += max(0, end_time - deadline) * weight

                for g_id, weight in obj_param.weight_per_groups.items():
                    if weight > 0:
                        g_idx = self.group_id_to_index[g_id]
                        group = self.tasks_group[g_idx]
                        deadline = group.max_ending_date
                        if deadline is not None:
                            task_indices = [
                                self.task_id_to_index[t] for t in group.tasks_group
                            ]
                            group_end = max(
                                variable.schedule[idx][-1][1] for idx in task_indices
                            )
                            tardiness += max(0, group_end - deadline) * weight

                evaluation["tardiness"] = tardiness

            # --- Earliness ---
            if ObjectivesEnum.EARLINESS in self.objective_params.params_obj:
                obj_param = self.objective_params.params_obj[ObjectivesEnum.EARLINESS]
                earliness = 0.0

                for t_id, weight in obj_param.weight_per_task.items():
                    if weight > 0:
                        idx = self.task_id_to_index[t_id]
                        deadline = self.tasks[idx].max_ending_date
                        if deadline is not None:
                            end_time = variable.schedule[idx][-1][1]
                            earliness += max(0, deadline - end_time) * weight

                for g_id, weight in obj_param.weight_per_groups.items():
                    if weight > 0:
                        g_idx = self.group_id_to_index[g_id]
                        group = self.tasks_group[g_idx]
                        deadline = group.max_ending_date
                        if deadline is not None:
                            task_indices = [
                                self.task_id_to_index[t] for t in group.tasks_group
                            ]
                            group_end = max(
                                variable.schedule[idx][-1][1] for idx in task_indices
                            )
                            earliness += max(0, deadline - group_end) * weight

                evaluation["earliness"] = earliness

            # --- Resource Cost ---
            if ObjectivesEnum.RESOURCE_COST in self.objective_params.params_obj:
                obj_param = self.objective_params.params_obj[
                    ObjectivesEnum.RESOURCE_COST
                ]
                cost = 0.0
                for res_id, weight in obj_param.weight_per_resource_unit.items():
                    if weight == 0:
                        continue
                    if (
                        res_id in obj_param.consider_in_objectives
                        and not obj_param.consider_in_objectives[res_id]
                    ):
                        continue
                    max_usage = np.max(resource_consumption[res_id])
                    cost += max_usage * weight
                evaluation["resource_cost"] = cost

            # --- WIP Cost ---
            # WIP (Work In Progress) = maximum number of concurrent groups in progress
            if ObjectivesEnum.WORK_IN_PROGRESS in self.objective_params.params_obj:
                obj_param = self.objective_params.params_obj[
                    ObjectivesEnum.WORK_IN_PROGRESS
                ]
                wip_cost = 0.0

                if (
                    obj_param.count_nb_group_in_progress
                    and obj_param.coefficient_on_nb_group_in_progress != 0
                ):
                    relevant_groups = [
                        g
                        for g in self.tasks_group
                        if g.type_of_group == GroupType.SUBGROUP_TASK_FOR_OBJECTIVE
                    ]

                    if relevant_groups:
                        events = []
                        for g in relevant_groups:
                            t_indices = [
                                self.task_id_to_index[t] for t in g.tasks_group
                            ]
                            g_start = min(
                                variable.schedule[idx][0][0] for idx in t_indices
                            )
                            g_end = max(
                                variable.schedule[idx][-1][1] for idx in t_indices
                            )
                            events.append((g_start, 1))
                            events.append((g_end, -1))

                        events.sort(key=lambda x: (x[0], x[1]))

                        max_concurrent_groups = 0
                        current_concurrent = 0
                        for _, change in events:
                            current_concurrent += change
                            max_concurrent_groups = max(
                                max_concurrent_groups, current_concurrent
                            )

                        wip_cost = (
                            max_concurrent_groups
                            * obj_param.coefficient_on_nb_group_in_progress
                        )

                evaluation["wip_cost"] = wip_cost

            return evaluation

        # 1. Base Metrics
        makespan = np.max(variable.schedule[:, 1])
        sd = SolutionDetails(
            problem=self,
            solution=variable,
            durations_data=self.durations_data,
            res_arrays_data=self.res_arrays_data,
        )
        resource_consumption = sd.resource_usage
        sd.compute_details()
        # Initialize all objectives declared in get_objective_register()
        evaluation = {
            "makespan": makespan,
            "tardiness": 0.0,
            "earliness": 0.0,
            "resource_cost": 0.0,
            "wip_cost": 0.0,
        }
        # 2. Objective-based Metrics
        # We iterate through the objective parameters to calculate specific costs
        # matching the logic in the CP solvers (e.g., fsp_cpsat_solver.py)

        # --- Tardiness ---
        if ObjectivesEnum.TARDINESS in self.objective_params.params_obj:
            obj_param = self.objective_params.params_obj[ObjectivesEnum.TARDINESS]
            tardiness = 0.0

            # Task Tardiness: sum(max(0, end - deadline) * weight)
            for t_id, weight in obj_param.weight_per_task.items():
                if weight > 0:
                    idx = self.task_id_to_index[t_id]
                    deadline = self.tasks[idx].max_ending_date
                    if deadline is not None:
                        end_time = variable.schedule[idx, 1]
                        tardiness += max(0, end_time - deadline) * weight

            # Group Tardiness: sum(max(0, group_end - deadline) * weight)
            for g_id, weight in obj_param.weight_per_groups.items():
                if weight > 0:
                    g_idx = self.group_id_to_index[g_id]
                    group = self.tasks_group[g_idx]
                    deadline = group.max_ending_date
                    if deadline is not None:
                        # Group end is the max end time of all tasks in the group
                        task_indices = [
                            self.task_id_to_index[t] for t in group.tasks_group
                        ]
                        group_end = np.max(variable.schedule[task_indices, 1])
                        tardiness += max(0, group_end - deadline) * weight

            evaluation["tardiness"] = tardiness

        # --- Earliness ---
        if ObjectivesEnum.EARLINESS in self.objective_params.params_obj:
            obj_param = self.objective_params.params_obj[ObjectivesEnum.EARLINESS]
            earliness = 0.0

            # Task Earliness: sum(max(0, deadline - end) * weight)
            for t_id, weight in obj_param.weight_per_task.items():
                if weight > 0:
                    idx = self.task_id_to_index[t_id]
                    deadline = self.tasks[idx].max_ending_date
                    if deadline is not None:
                        end_time = variable.schedule[idx, 1]
                        earliness += max(0, deadline - end_time) * weight

            # Group Earliness
            for g_id, weight in obj_param.weight_per_groups.items():
                if weight > 0:
                    g_idx = self.group_id_to_index[g_id]
                    group = self.tasks_group[g_idx]
                    deadline = group.max_ending_date
                    if deadline is not None:
                        task_indices = [
                            self.task_id_to_index[t] for t in group.tasks_group
                        ]
                        group_end = np.max(variable.schedule[task_indices, 1])
                        earliness += max(0, deadline - group_end) * weight

            evaluation["earliness"] = earliness

        # --- Resource Cost ---
        if ObjectivesEnum.RESOURCE_COST in self.objective_params.params_obj:
            obj_param = self.objective_params.params_obj[ObjectivesEnum.RESOURCE_COST]
            cost = 0.0
            for res_id, weight in obj_param.weight_per_resource_unit.items():
                if weight == 0:
                    continue
                # Skip if resource is explicitly excluded from objectives
                if (
                    res_id in obj_param.consider_in_objectives
                    and not obj_param.consider_in_objectives[res_id]
                ):
                    continue
                max_usage = np.max(resource_consumption[res_id])
                cost += max_usage * weight
            evaluation["resource_cost"] = cost
        # --- WIP Cost ---
        # WIP (Work In Progress) = maximum number of concurrent groups in progress
        if ObjectivesEnum.WORK_IN_PROGRESS in self.objective_params.params_obj:
            obj_param = self.objective_params.params_obj[
                ObjectivesEnum.WORK_IN_PROGRESS
            ]
            wip_cost = 0.0

            if (
                obj_param.count_nb_group_in_progress
                and obj_param.coefficient_on_nb_group_in_progress != 0
            ):
                relevant_groups = [
                    g
                    for g in self.tasks_group
                    if g.type_of_group == GroupType.SUBGROUP_TASK_FOR_OBJECTIVE
                ]

                if relevant_groups:
                    # Collect start/end events for a sweep-line algorithm
                    events = []
                    for g in relevant_groups:
                        t_indices = [self.task_id_to_index[t] for t in g.tasks_group]
                        # Group spans from min start of its tasks to max end of its tasks
                        g_start = np.min(variable.schedule[t_indices, 0])
                        g_end = np.max(variable.schedule[t_indices, 1])

                        # Add events: +1 at start, -1 at end
                        events.append((g_start, 1))
                        events.append((g_end, -1))

                    # Sort events by time.
                    # If times are equal, process -1 (end) before +1 (start) to treat intervals as [s, e)
                    # This prevents false overlaps at boundaries.
                    events.sort(key=lambda x: (x[0], x[1]))

                    max_concurrent_groups = 0
                    current_concurrent = 0
                    for _, change in events:
                        current_concurrent += change
                        max_concurrent_groups = max(
                            max_concurrent_groups, current_concurrent
                        )

                    wip_cost = (
                        max_concurrent_groups
                        * obj_param.coefficient_on_nb_group_in_progress
                    )

            evaluation["wip_cost"] = wip_cost

        return evaluation

    def satisfy(self, variable: ScheduleSolution) -> bool:
        from .fsp_utils import SolutionDetails

        # TODO: implement proper satisfaction checking for preemptive schedules
        #  Should check:
        #  - Task durations (sum of intervals = required duration)
        #  - Resource constraints (preemptive intervals respect calendars)
        #  - Precedence constraints
        #  - Group constraints
        #  - Time windows (min/max start/end dates)
        if isinstance(variable, ScheduleSolutionPreemptive):
            return True

        l_violation = []
        sd = SolutionDetails(
            problem=self,
            solution=variable,
            durations_data=self.durations_data,
            res_arrays_data=self.res_arrays_data,
        )
        sd.compute_details()
        sat = sd.satisfy()
        if not sat:
            return sat
        # l_violation += satisfy_task_duration(variable, usage, self) # covered by the solution details
        l_violation += satisfy_precedence(variable, self)
        l_violation += satisfy_precedence_group(variable, self)
        l_violation += satisfy_release_and_date(variable, self)
        l_violation += satisfy_generic_precedence_constraint(variable, self)
        return len(l_violation) == 0

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister({})

    def get_solution_type(self) -> Type[Solution]:
        return ScheduleSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "tardiness": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "earliness": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "resource_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "wip_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
            },
        )

    def __str__(self):
        s = " ".join(
            [str(getattr(self, attr)) for attr in ["tasks", "tasks_group", "resources"]]
        )
        return s


def satisfy_release_and_date(solution: ScheduleSolution, problem: FlexProblem):
    l_violation = []
    for task in problem.tasks:
        index = problem.task_id_to_index[task.id]
        if task.min_starting_date is not None:
            if solution.schedule[index, 0] < task.min_starting_date:
                l_violation.append(
                    (
                        "start_before_min_starting_date",
                        task,
                        solution.schedule[index, 0],
                        task.min_starting_date,
                    )
                )
        if task.max_starting_date is not None:
            if solution.schedule[index, 0] > task.max_starting_date:
                l_violation.append(
                    (
                        "start_after_max_starting_date",
                        task,
                        solution.schedule[index, 0],
                        task.max_starting_date,
                    )
                )
        if task.min_ending_date is not None:
            if (
                solution.schedule[index, 1] < task.min_ending_date
                and not task.soft_max_end_date
            ):
                l_violation.append(
                    (
                        "end_before_min_ending_date",
                        task,
                        solution.schedule[index, 1],
                        task.min_ending_date,
                    )
                )
        if task.max_ending_date is not None:
            if (
                solution.schedule[index, 1] > task.max_ending_date
                and not task.soft_max_end_date
            ):
                l_violation.append(
                    (
                        "end_after_max_ending_date",
                        task,
                        solution.schedule[index, 1],
                        task.max_ending_date,
                    )
                )
    return l_violation


def satisfy_task_duration(solution, usage, problem):
    l_violation = []
    for task in problem.tasks:
        index = problem.task_id_to_index[task.id]
        mode = solution.modes[index]
        if task.modes[mode].duration != usage[index].sum():
            l_violation.append(
                ("task_duration", task, task.modes[mode].duration, usage[index].sum())
            )
    return l_violation


def satisfy_precedence(solution: ScheduleSolution, problem: FlexProblem):
    l_violation = []
    if problem.constraints.successors is not None:
        for task_id in problem.constraints.successors:
            index = problem.task_id_to_index[task_id]
            for succ_id in problem.constraints.successors[task_id]:
                succ_index = problem.task_id_to_index[succ_id]
                if solution.schedule[succ_index, 0] < solution.schedule[index, 1]:
                    l_violation.append(
                        (
                            "successor_violated",
                            (task_id, succ_id),
                            (
                                solution.schedule[index, 1],
                                solution.schedule[succ_index, 0],
                            ),
                        )
                    )
    return l_violation


def satisfy_precedence_group(solution: ScheduleSolution, problem: FlexProblem):
    l_violation = []
    if problem.constraints.successors_group_tasks is not None:
        for group_id in problem.constraints.successors_group_tasks:
            gr = problem.tasks_group[problem.group_id_to_index[group_id]]
            tasks_index_in_group = [problem.task_id_to_index[x] for x in gr.tasks_group]
            max_end_group = max([solution.schedule[x, 1] for x in tasks_index_in_group])
            for succ_group in problem.constraints.successors_group_tasks[group_id]:
                gr_succ = problem.tasks_group[problem.group_id_to_index[succ_group]]
                tasks_index_in_group_succ = [
                    problem.task_id_to_index[x] for x in gr_succ.tasks_group
                ]
                min_start_group = max(
                    [solution.schedule[x, 0] for x in tasks_index_in_group_succ]
                )
                if min_start_group < max_end_group:
                    l_violation.append(
                        (
                            "successor_group_violated",
                            (group_id, succ_group),
                            (max_end_group, min_start_group),
                        )
                    )
    return l_violation


def satisfy_generic_precedence_constraint(
    solution: ScheduleSolution, problem: FlexProblem
):
    l_violation = []
    if problem.constraints.start_at_end is not None:
        for t1, t2 in problem.constraints.start_at_end:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] != solution.schedule[i2, 0]:
                l_violation.append(
                    (
                        "start_at_start_violated",
                        (t1, t2),
                        (solution.schedule[i1, 0], solution.schedule[i2, 0]),
                    )
                )
    if problem.constraints.start_at_end is not None:
        for t1, t2 in problem.constraints.start_at_end:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] != solution.schedule[i2, 1]:
                l_violation.append(
                    (
                        "start_at_end_violated",
                        (t1, t2),
                        (solution.schedule[i1, 0], solution.schedule[i2, 1]),
                    )
                )
    if problem.constraints.start_after_end_plus_offset is not None:
        for t1, t2, delta in problem.constraints.start_after_end_plus_offset:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] < solution.schedule[i2, 1] + delta:
                l_violation.append(
                    (
                        "start_after_end_plus_offset_violated",
                        (t1, t2, delta),
                        (solution.schedule[i1, 0], solution.schedule[i2, 1]),
                    )
                )
    if problem.constraints.start_at_end_plus_offset is not None:
        for t1, t2, delta in problem.constraints.start_at_end_plus_offset:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] != solution.schedule[i2, 1] + delta:
                l_violation.append(
                    (
                        "start_at_end_plus_offset_violated",
                        (t1, t2, delta),
                        (solution.schedule[i1, 0], solution.schedule[i2, 1]),
                    )
                )
    if problem.constraints.start_after_start_plus_offset is not None:
        for t1, t2, delta in problem.constraints.start_after_start_plus_offset:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] < solution.schedule[i2, 0] + delta:
                l_violation.append(
                    (
                        "start_after_start_plus_offset_violated",
                        (t1, t2, delta),
                        (solution.schedule[i1, 0], solution.schedule[i2, 0]),
                    )
                )
    if problem.constraints.start_at_start_plus_offset is not None:
        for t1, t2, delta in problem.constraints.start_at_start_plus_offset:
            i1, i2 = problem.task_id_to_index[t1], problem.task_id_to_index[t2]
            if solution.schedule[i1, 0] != solution.schedule[i2, 0] + delta:
                l_violation.append(
                    (
                        "start_at_start_plus_offset_violated",
                        (t1, t2, delta),
                        (solution.schedule[i1, 0], solution.schedule[i2, 0]),
                    )
                )
    return l_violation


def get_lb_ub_start_end_date(problem: FlexProblem):
    min_start_time = {i: 0 for i in range(problem.nb_tasks)}
    max_start_time = {i: problem.horizon for i in range(problem.nb_tasks)}
    min_end_time = {i: 0 for i in range(problem.nb_tasks)}
    max_end_time = {i: problem.horizon for i in range(problem.nb_tasks)}
    for i in range(problem.nb_tasks):
        task = problem.tasks[i]
        if task.min_starting_date is not None:
            min_start_time[i] = task.min_starting_date
        if task.max_starting_date is not None:
            max_start_time[i] = task.max_starting_date
        if task.min_ending_date is not None:
            min_end_time[i] = task.min_ending_date
        if task.max_ending_date is not None:
            max_end_time[i] = task.max_ending_date
    return min_start_time, max_start_time, min_end_time, max_end_time
