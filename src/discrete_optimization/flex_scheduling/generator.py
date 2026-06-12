#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random
from typing import List, Optional

import numpy as np

from discrete_optimization.flex_scheduling.problem import (
    ConstraintsTask,
    FlexProblem,
    GroupType,
    ObjectiveParamEarliness,
    ObjectiveParamResource,
    ObjectiveParams,
    ObjectiveParamTardiness,
    ObjectivesEnum,
    ResourceData,
    TaskData,
    TaskGroupAbstraction,
    TaskObject,
    TasksGroups,
)


class FlexProblemGenerator:
    def __init__(
        self,
        nb_msn: int = 36,  # 36 Products
        horizon: Optional[int] = None,
        seed: int = 42,
        tardiness_weight: int = 20,
        earliness_weight: int = 1,
        tightness_factor: float = 1.3,
        nb_tools: int = 12,
        nb_stations: int = 32,
    ):
        self.nb_msn = nb_msn
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # --- RESOURCE CONFIGURATION (Strictly mimicking the parser pools) ---

        # 1. Operators (Variable Capacity)
        self.operator_pools = {"R1": 4, "R2": 2, "R35": 1}

        # 2. Stations (R3..R34): The main blocking resources.
        self.station_ids = [
            f"R{i}" for i in range(3, 3 + nb_stations) if f"R{i}" != "R35"
        ]

        # 3. Auxiliary Tools (mimicking R6, R24, etc. when combined with stations)
        # These are blocked ALONGSIDE stations.
        self.tools = [f"Tool_{i}" for i in range(nb_tools)]

        # 4. Transition Locks (mimicking Unique Resource blocks)
        # These are capacity-1 resources used purely for Task->Group or Task->Task blocking
        self.transition_locks = [f"Lock_{i}" for i in range(100)]

        # Horizon
        if horizon is None:
            raw_work = 36 * 6 * 2 * 15
            est_makespan = int(raw_work * 4.5 * 1.8 * 1.3)
            self.horizon = ((est_makespan // 5000) + 1) * 5000
        else:
            self.horizon = horizon

        self.tardiness_weight = int(tardiness_weight)
        self.earliness_weight = int(earliness_weight)
        self.tightness_factor = tightness_factor

    def _generate_calendar(self, kind: str, capacity: int) -> np.ndarray:
        calendar = np.zeros(self.horizon, dtype=int)
        week_len = 24 * 7
        offset = self.rng.randint(0, 50)

        for t in range(self.horizon):
            adj_t = t + offset
            hour_of_week = adj_t % week_len
            day = hour_of_week // 24
            hour = hour_of_week % 24

            val = 0
            if kind == "operator":
                # Mon-Fri 06:00-22:00
                if day < 5 and 6 <= hour < 22:
                    val = capacity
            elif kind == "station":
                # Mon-Sat 06:00-22:00
                if day < 6 and 6 <= hour < 22:
                    val = capacity

            calendar[t] = val
        return calendar

    def generate(self) -> FlexProblem:
        resources: List[ResourceData] = []
        tasks: List[TaskObject] = []
        groups: List[TasksGroups] = []

        constraints = ConstraintsTask(
            successors={},
            successors_group_tasks={},
            successor_generic_with_res_release_at_start_of_successor_generic=[],
            start_after_start_plus_offset=[],
        )

        # --- 1. RESOURCE GENERATION ---

        # Operators
        for name, cap in self.operator_pools.items():
            resources.append(
                ResourceData(
                    id=name,
                    name=name,
                    calendar_availability=self._generate_calendar("operator", cap),
                    renewable=True,
                    max_capacity=cap,
                    is_disjunctive=False,
                    is_station=False,
                    is_operator=True,
                )
            )

        # Stations
        for r_id in self.station_ids:
            resources.append(
                ResourceData(
                    id=r_id,
                    name=r_id,
                    calendar_availability=self._generate_calendar("station", 1),
                    renewable=True,
                    max_capacity=1,
                    is_disjunctive=True,
                    is_station=True,
                    is_operator=False,
                )
            )

        # Tools & Locks (Always Available physically, but blocked logically)
        for r_id in self.tools + self.transition_locks:
            resources.append(
                ResourceData(
                    id=r_id,
                    name=r_id,
                    calendar_availability=np.ones(self.horizon, dtype=int),
                    renewable=True,
                    max_capacity=1,
                    is_disjunctive=True,
                    is_station=False,
                    is_operator=False,
                )
            )

        # --- 2. PRODUCT GENERATION ---
        # 36 Products: 8 Large (7 stations), 28 Small (5 stations)
        product_configs = [("Large", 7) for _ in range(int(0.35 * self.nb_msn))] + [
            ("Small", 5) for _ in range(int(0.75 * self.nb_msn))
        ]
        self.rng.shuffle(product_configs)

        task_counter = 0
        task_tardiness_weights = {}
        task_earliness_weights = {}

        for msn_idx, (p_type, path_len) in enumerate(product_configs):
            msn_id = f"MSN{msn_idx + 1}"

            # Select Route (No Re-entry)
            route = sorted(self.rng.sample(self.station_ids, path_len))

            # State tracking for chaining
            prev_group_obj = None
            prev_last_task_id = None
            prev_blocked_res = None

            accumulated_work = 0
            all_msn_task_ids = []

            # Deterministic Release (No Stagger -> Max Contention)
            release_date = 0

            for step_idx, station_res_id in enumerate(route):
                # --- A. TASKS (Variable group size: 1-3) ---
                nb_tasks = self.rng.choices([1, 2, 3], weights=[0.2, 0.6, 0.2])[0]
                tasks_in_group_set = set()
                step_task_ids = []

                for _ in range(nb_tasks):
                    t_id = f"T_{task_counter}"
                    task_counter += 1
                    step_task_ids.append(t_id)
                    all_msn_task_ids.append(t_id)
                    tasks_in_group_set.add(t_id)

                    duration = self.rng.randint(5, 25)
                    accumulated_work += duration

                    # Operator Consumption
                    res_consumption = {}
                    if self.rng.random() < 0.7:
                        op = "R1" if self.rng.random() < 0.6 else "R2"
                        res_consumption[op] = 1
                    if self.rng.random() < 0.15:
                        res_consumption["R35"] = 1

                    new_task = TaskObject(
                        id=t_id,
                        name=f"{msn_id}_{t_id}",
                        modes={
                            1: TaskData(duration, res_consumption, True)
                        },  # Preemptive
                        min_starting_date=release_date if step_idx == 0 else None,
                    )
                    tasks.append(new_task)

                # Internal Precedence
                for i in range(len(step_task_ids) - 1):
                    c, n = step_task_ids[i], step_task_ids[i + 1]
                    if c not in constraints.successors:
                        constraints.successors[c] = set()
                    constraints.successors[c].add(n)

                # --- B. EXECUTION BLOCKING (Group Level) ---
                # "Group Blocks Resource"
                # Matches `task_block_group_resource.json`: 2 resources blocked (Station + Tool)
                blocked_res = {
                    station_res_id: 1,  # The Station
                    self.rng.choice(self.tools): 1,  # The Aux Tool
                }

                group_id = f"G_{msn_id}_{step_idx}"
                group_obj = TasksGroups(
                    id=group_id,
                    name=f"Group_{msn_id}_{station_res_id}",
                    tasks_group=tasks_in_group_set,
                    type_of_group=GroupType.GROUP_TASK_NON_RELEASED_RESOURCE,
                    res_not_released=blocked_res,
                    no_overlap=False,
                )
                groups.append(group_obj)

                # --- C. COMPLEX GAP BLOCKING (The "Parser" Logic) ---
                if prev_group_obj is not None:
                    # 1. Standard Precedence (Group -> Group)
                    if prev_group_obj.id not in constraints.successors_group_tasks:
                        constraints.successors_group_tasks[prev_group_obj.id] = set()
                    constraints.successors_group_tasks[prev_group_obj.id].add(
                        group_obj.id
                    )

                    # 2. RETENTION BLOCKING (Group -> Group)
                    # "Hold Previous Resources until Current Group Starts"
                    # CRITICAL FIX: No random reduction. We hold exactly what was held before.
                    # This ensures consistency: The product is still in the jig.
                    pred_abs = TaskGroupAbstraction(
                        is_a_task=False, task_id=0, group_id=prev_group_obj.id
                    )
                    succ_abs = TaskGroupAbstraction(
                        is_a_task=False, task_id=0, group_id=group_obj.id
                    )

                    constraints.successor_generic_with_res_release_at_start_of_successor_generic.append(
                        (pred_abs, succ_abs, prev_blocked_res.copy())  # COPY full dict
                    )

                    # 3. HANDSHAKE BLOCKING (Task -> Group)
                    # Matches `task_block_unique_resource.json` entries like {TASK_ID: X, SUCCESSOR_TASK_ID: [Y, Z]}
                    # Last Task of Prev -> Start of Current Group
                    # Blocks a separate "Lock" resource to enforce strict sequencing logic
                    lock_res = self.rng.choice(self.transition_locks)
                    pred_abs_t = TaskGroupAbstraction(
                        is_a_task=True, task_id=prev_last_task_id, group_id=None
                    )
                    # succ_abs is already Group (defined above)

                    constraints.successor_generic_with_res_release_at_start_of_successor_generic.append(
                        (pred_abs_t, succ_abs, {lock_res: 1})
                    )

                    # 4. BRIDGE BLOCKING (Task -> Task)
                    # Matches `task_block_unique_resource.json` entries like {TASK_ID: X, SUCCESSOR_TASK_ID: Y}
                    # Last Task of Prev -> First Task of Current
                    # Blocks YET ANOTHER resource (or reuse the lock) to tighten the coupling.
                    lock_res_2 = self.rng.choice(self.transition_locks)
                    succ_abs_t = TaskGroupAbstraction(
                        is_a_task=True, task_id=step_task_ids[0], group_id=None
                    )

                    constraints.successor_generic_with_res_release_at_start_of_successor_generic.append(
                        (pred_abs_t, succ_abs_t, {lock_res_2: 1})
                    )

                prev_group_obj = group_obj
                prev_blocked_res = blocked_res
                prev_last_task_id = step_task_ids[-1]

            # Objectives (Final Task)
            delivery_task_id = all_msn_task_ids[-1]
            deadline = int(
                release_date + (accumulated_work * 4.5 * self.tightness_factor)
            )
            t = next(x for x in tasks if x.id == delivery_task_id)
            t.max_ending_date = deadline
            t.soft_max_end_date = True

            task_tardiness_weights[delivery_task_id] = self.tardiness_weight
            task_earliness_weights[delivery_task_id] = self.earliness_weight

        # --- 3. OBJECTIVES ---
        obj_params = ObjectiveParams(
            params_obj={
                ObjectivesEnum.MAKESPAN: 1.0,
                ObjectivesEnum.TARDINESS: ObjectiveParamTardiness(
                    weight_per_task=task_tardiness_weights, weight_per_groups={}
                ),
                ObjectivesEnum.EARLINESS: ObjectiveParamEarliness(
                    weight_per_task=task_earliness_weights, weight_per_groups={}
                ),
                ObjectivesEnum.RESOURCE_COST: ObjectiveParamResource(
                    weight=1,
                    weight_per_resource_unit={"R1": 1, "R2": 1, "R35": 2},
                    consider_in_objectives={r.id: r.is_operator for r in resources},
                ),
                # WIP disabled by default - enable with count_nb_group_in_progress=True if needed
                # ObjectivesEnum.WORK_IN_PROGRESS: ObjectiveParamWIP(
                #     weight=1,
                #     weight_per_task={},  # Not used - WIP is only for concurrent groups
                #     weights_per_group_task={},
                #     count_nb_group_in_progress=True,
                #     coefficient_on_nb_group_in_progress=5.0,
                # ),
            }
        )

        return FlexProblem(
            resources, tasks, groups, constraints, obj_params, self.horizon
        )


# problem = gen.generate()
