#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import random
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
)

logger = logging.getLogger(__name__)


class HeuristicOvenSchedulingSolver(SolverDO):
    """
    This solver can be evolved to use different scheduling strategies:
    - Greedy batch construction
    - Priority-based task ordering
    - Attribute grouping strategies
    - Time window optimization
    - Setup cost minimization
    """

    problem: OvenSchedulingProblem
    hyperparameters = [
        IntegerHyperparameter(
            name="num_local_search_iterations", default=25, low=0, high=1000
        ),
        FloatHyperparameter(name="cooling_rate", default=0.97, low=0, high=1),
    ]

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function=params_objective_function, **kwargs
        )

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """
        Solve the OSP using a Large Neighborhood Search with a Simulated Annealing acceptance criterion.

        Strategy:
        1.  Initial constructive phase: Build a complete initial schedule using a greedy heuristic.
        2.  Local Search / Refinement Phase: Iteratively attempt to improve the schedule.
            a.  Ruin: Select a "bad" part of the current schedule to destroy using one of several
                diversified strategies (e.g., remove a tardy batch, a high-cost batch, or a random batch).
                The selected batch and all subsequent batches on that machine are removed.
            b.  Recreate: The tasks from the ruined batches are put back into the unscheduled pool
                and re-scheduled using the same powerful greedy constructor.
            c.  Acceptance (Simulated Annealing): A new schedule is always accepted if it's better
                than the previous one. Crucially, it may also be accepted if it is *worse*, based
                on a probability that depends on a decreasing "temperature". This allows the search
                to escape local optima.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        result_storage = self.create_result_storage([])

        # Objective weights (can be overridden by params_objective_function)
        default_weights = [1, 10000000, 1]  # processing_time, nb_late_jobs, setup_cost
        weights = (
            self.params_objective_function.weights
            if self.params_objective_function and self.params_objective_function.weights
            else default_weights
        )

        # Base weights for the greedy heuristic's scoring function
        tardiness_weight = weights[1]
        initial_lookahead_penalty_weight = 10000
        setup_cost_weight = weights[2]
        adaptive_lookahead_factor = 5

        # Pre-sort all tasks by criticality for efficient batch building
        all_tasks_sorted_by_criticality = sorted(
            range(self.problem.n_jobs),
            key=lambda t: (
                self.problem.tasks_data[t].latest_end,
                self.problem.tasks_data[t].earliest_start,
            ),
        )

        def build_batch(
            pool_of_tasks_to_draw_from: set[int],
            initial_tasks_in_batch_set: set[int],
            current_machine_data: Any,
        ):
            current_batch_tasks = set(initial_tasks_in_batch_set)
            current_size = sum(
                self.problem.tasks_data[t].size for t in current_batch_tasks
            )
            current_max_min_d = max(
                (self.problem.tasks_data[t].min_duration for t in current_batch_tasks),
                default=0,
            )
            current_min_max_d = min(
                (self.problem.tasks_data[t].max_duration for t in current_batch_tasks),
                default=float("inf"),
            )
            tasks_to_consider_sorted_for_batching = [
                t
                for t in all_tasks_sorted_by_criticality
                if t in pool_of_tasks_to_draw_from and t not in current_batch_tasks
            ]
            for task_id in tasks_to_consider_sorted_for_batching:
                task_data = self.problem.tasks_data[task_id]
                if current_size + task_data.size <= current_machine_data.capacity:
                    new_max_min_d = max(current_max_min_d, task_data.min_duration)
                    new_min_max_d = min(current_min_max_d, task_data.max_duration)
                    if new_max_min_d <= new_min_max_d:
                        current_batch_tasks.add(task_id)
                        current_size += task_data.size
                        current_max_min_d = new_max_min_d
                        current_min_max_d = new_min_max_d
            return current_batch_tasks

        def _run_greedy_scheduling_phase(
            initial_schedule_per_machine_state: dict[int, list[ScheduleInfo]],
            tasks_to_schedule_in_this_pass: set[int],
        ):
            schedule_per_machine = {
                m: list(s) for m, s in initial_schedule_per_machine_state.items()
            }
            unscheduled_tasks = set(tasks_to_schedule_in_this_pass)
            machine_end_times = [0] * self.problem.n_machines
            last_attribute_on_machine = [
                m.initial_attribute for m in self.problem.machines_data
            ]
            for m_id in range(self.problem.n_machines):
                if schedule_per_machine[m_id]:
                    last_batch = schedule_per_machine[m_id][-1]
                    machine_end_times[m_id] = last_batch.end_time
                    last_attribute_on_machine[m_id] = last_batch.task_attribute

            while unscheduled_tasks:
                num_total_tasks = self.problem.n_jobs
                num_unscheduled = len(unscheduled_tasks)
                progress_ratio = (
                    (num_total_tasks - num_unscheduled) / num_total_tasks
                    if num_total_tasks > 0
                    else 0
                )
                current_lookahead_penalty_weight = initial_lookahead_penalty_weight * (
                    1 + progress_ratio * adaptive_lookahead_factor
                )
                unscheduled_tasks_latest_ends = {
                    tid: self.problem.tasks_data[tid].latest_end
                    for tid in unscheduled_tasks
                }

                best_candidate = None
                best_score = float("inf")

                for machine_id in range(self.problem.n_machines):
                    machine_data = self.problem.machines_data[machine_id]
                    prev_attribute = last_attribute_on_machine[machine_id]
                    possible_attributes = {
                        self.problem.tasks_data[t].attribute
                        for t in unscheduled_tasks
                        if machine_id in self.problem.tasks_data[t].eligible_machines
                    }

                    for attribute in possible_attributes:
                        setup_time = self.problem.setup_times[prev_attribute][attribute]
                        setup_cost = self.problem.setup_costs[prev_attribute][attribute]
                        machine_ready_time_for_this_batch = (
                            machine_end_times[machine_id] + setup_time
                        )
                        eligible_tasks_for_attribute_machine = sorted(
                            [
                                t
                                for t in unscheduled_tasks
                                if self.problem.tasks_data[t].attribute == attribute
                                and machine_id
                                in self.problem.tasks_data[t].eligible_machines
                            ],
                            key=lambda t: (
                                self.problem.tasks_data[t].latest_end,
                                self.problem.tasks_data[t].earliest_start,
                            ),
                        )
                        if not eligible_tasks_for_attribute_machine:
                            continue

                        num_seed_tasks_to_consider = 3
                        unique_candidates = set()
                        for seed_task_index in range(
                            min(
                                num_seed_tasks_to_consider,
                                len(eligible_tasks_for_attribute_machine),
                            )
                        ):
                            seed_task_id = eligible_tasks_for_attribute_machine[
                                seed_task_index
                            ]
                            seed_task_data = self.problem.tasks_data[seed_task_id]
                            candidate_sets_for_this_seed = []
                            mf_tasks = build_batch(
                                set(eligible_tasks_for_attribute_machine),
                                {seed_task_id},
                                machine_data,
                            )
                            if mf_tasks:
                                candidate_sets_for_this_seed.append(mf_tasks)
                            candidate_sets_for_this_seed.append({seed_task_id})
                            q_tasks_filtered_pool = {seed_task_id}
                            urgency_delta = max(1, seed_task_data.max_duration) * 2
                            temporal_upper_bound = (
                                seed_task_data.latest_end + urgency_delta
                            )
                            for task_id in eligible_tasks_for_attribute_machine:
                                if (
                                    task_id != seed_task_id
                                    and self.problem.tasks_data[task_id].latest_end
                                    <= temporal_upper_bound
                                ):
                                    q_tasks_filtered_pool.add(task_id)
                            q_tasks = build_batch(
                                q_tasks_filtered_pool, {seed_task_id}, machine_data
                            )
                            if q_tasks:
                                candidate_sets_for_this_seed.append(q_tasks)
                            unique_candidates.update(
                                {
                                    frozenset(s)
                                    for s in candidate_sets_for_this_seed
                                    if s
                                }
                            )

                        for batch_tasks in unique_candidates:
                            batch_props = {
                                "max_min_d": max(
                                    (
                                        self.problem.tasks_data[t].min_duration
                                        for t in batch_tasks
                                    ),
                                    default=0,
                                ),
                                "min_max_d": min(
                                    (
                                        self.problem.tasks_data[t].max_duration
                                        for t in batch_tasks
                                    ),
                                    default=float("inf"),
                                ),
                                "max_es": max(
                                    (
                                        self.problem.tasks_data[t].earliest_start
                                        for t in batch_tasks
                                    ),
                                    default=0,
                                ),
                                "min_le": min(
                                    (
                                        self.problem.tasks_data[t].latest_end
                                        for t in batch_tasks
                                    ),
                                    default=float("inf"),
                                ),
                            }
                            b_min_duration = max(1, batch_props["max_min_d"])
                            b_max_duration = batch_props["min_max_d"]
                            if b_min_duration > b_max_duration:
                                continue
                            earliest_start_base = max(
                                machine_ready_time_for_this_batch, batch_props["max_es"]
                            )

                            found_slot_for_any_duration = False
                            for start_avail, end_avail in machine_data.availability:
                                potential_start = max(earliest_start_base, start_avail)
                                current_durations_to_check = {b_min_duration}
                                if (
                                    b_max_duration != float("inf")
                                    and b_max_duration > b_min_duration
                                ):
                                    current_durations_to_check.add(b_max_duration)
                                duration_to_hit_deadline = (
                                    batch_props["min_le"] - potential_start
                                )
                                if (
                                    b_min_duration
                                    <= duration_to_hit_deadline
                                    <= b_max_duration
                                ):
                                    current_durations_to_check.add(
                                        duration_to_hit_deadline
                                    )
                                if (
                                    b_max_duration != float("inf")
                                    and (b_max_duration - b_min_duration) > 4
                                ):
                                    step = (b_max_duration - b_min_duration) // 4
                                    for i in range(1, 4):
                                        intermediate_duration = (
                                            b_min_duration + i * step
                                        )
                                        if (
                                            b_min_duration
                                            < intermediate_duration
                                            < b_max_duration
                                        ):
                                            current_durations_to_check.add(
                                                intermediate_duration
                                            )
                                duration_filling_slot = end_avail - potential_start
                                if (
                                    duration_filling_slot > 0
                                    and b_min_duration
                                    <= duration_filling_slot
                                    <= b_max_duration
                                ):
                                    current_durations_to_check.add(
                                        duration_filling_slot
                                    )

                                for duration in sorted(
                                    list(current_durations_to_check)
                                ):
                                    if (
                                        duration <= 0
                                        or duration < b_min_duration
                                        or (
                                            b_max_duration != float("inf")
                                            and duration > b_max_duration
                                        )
                                    ):
                                        continue
                                    end_time = potential_start + duration
                                    if end_time <= end_avail:
                                        found_slot_for_any_duration = True
                                        tardiness = max(
                                            0, end_time - batch_props["min_le"]
                                        )
                                        lookahead_penalty_value = sum(
                                            max(0, end_time - le)
                                            for tid, le in unscheduled_tasks_latest_ends.items()
                                            if tid not in batch_tasks
                                        )
                                        score = (
                                            weights[0] * end_time
                                            + tardiness_weight * tardiness
                                            + current_lookahead_penalty_weight
                                            * lookahead_penalty_value
                                            + setup_cost_weight * setup_cost
                                        )
                                        if score < best_score:
                                            best_score = score
                                            best_candidate = {
                                                "tasks": batch_tasks,
                                                "attribute": attribute,
                                                "start_time": potential_start,
                                                "end_time": end_time,
                                                "machine_id": machine_id,
                                            }
                                if found_slot_for_any_duration:
                                    break

                if best_candidate:
                    machine_id = best_candidate["machine_id"]
                    new_batch = ScheduleInfo(
                        tasks=set(best_candidate["tasks"]),
                        task_attribute=best_candidate["attribute"],
                        start_time=best_candidate["start_time"],
                        end_time=best_candidate["end_time"],
                        machine_batch_index=(
                            machine_id,
                            len(schedule_per_machine[machine_id]),
                        ),
                    )
                    schedule_per_machine[machine_id].append(new_batch)
                    schedule_per_machine[machine_id].sort(key=lambda b: b.start_time)
                    unscheduled_tasks -= best_candidate["tasks"]
                    machine_end_times[machine_id] = new_batch.end_time
                    last_attribute_on_machine[machine_id] = new_batch.task_attribute
                else:
                    if unscheduled_tasks:
                        most_critical_task_id = sorted(
                            list(unscheduled_tasks),
                            key=lambda t: (
                                self.problem.tasks_data[t].latest_end,
                                self.problem.tasks_data[t].earliest_start,
                            ),
                        )[0]
                        logger.warning(
                            f"Could not schedule task {most_critical_task_id}. Removing it to proceed."
                        )
                        unscheduled_tasks.remove(most_critical_task_id)
                    else:
                        break
            return schedule_per_machine, unscheduled_tasks

        def _identify_worst_batch_and_affected_tasks(
            solution: OvenSchedulingSolution, strategy: str
        ):
            worst_batch_location = None
            if strategy == "tardiness":
                worst_tardiness_score, worst_end_time = -1, -1
                for m_id, schedule in solution.schedule_per_machine.items():
                    for b_idx, b_info in enumerate(schedule):
                        batch_tardiness = sum(
                            max(
                                0,
                                b_info.end_time
                                - self.problem.tasks_data[t_id].latest_end,
                            )
                            for t_id in b_info.tasks
                        )
                        if batch_tardiness > worst_tardiness_score or (
                            batch_tardiness == worst_tardiness_score
                            and b_info.end_time > worst_end_time
                        ):
                            (
                                worst_tardiness_score,
                                worst_end_time,
                                worst_batch_location,
                            ) = batch_tardiness, b_info.end_time, (m_id, b_idx)
            elif strategy == "latest_finish":
                latest_end_time = -1
                for m_id, schedule in solution.schedule_per_machine.items():
                    if schedule and schedule[-1].end_time > latest_end_time:
                        latest_end_time, worst_batch_location = (
                            schedule[-1].end_time,
                            (m_id, len(schedule) - 1),
                        )
            elif strategy == "setup_cost":
                max_setup_cost = -1
                for m_id, schedule in solution.schedule_per_machine.items():
                    last_attr = self.problem.machines_data[m_id].initial_attribute
                    for b_idx, b_info in enumerate(schedule):
                        current_setup_cost = self.problem.setup_costs[last_attr][
                            b_info.task_attribute
                        ]
                        if current_setup_cost > max_setup_cost:
                            max_setup_cost, worst_batch_location = (
                                current_setup_cost,
                                (m_id, b_idx),
                            )
                        last_attr = b_info.task_attribute
            elif strategy == "random":
                all_batches = [
                    (m_id, b_idx)
                    for m_id, schedule in solution.schedule_per_machine.items()
                    for b_idx in range(len(schedule))
                ]
                if all_batches:
                    worst_batch_location = random.choice(all_batches)

            if worst_batch_location is None:
                return None
            machine_id, batch_idx = worst_batch_location
            tasks_to_reschedule_from_removal = set().union(
                *(
                    b_info.tasks
                    for b_info in solution.schedule_per_machine[machine_id][batch_idx:]
                )
            )
            return machine_id, batch_idx, tasks_to_reschedule_from_removal

        # --- Main solve logic with Simulated Annealing ---
        initial_schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        initial_schedule, initial_unscheduled = _run_greedy_scheduling_phase(
            initial_schedule_per_machine, set(range(self.problem.n_jobs))
        )

        best_solution = OvenSchedulingSolution(
            problem=self.problem, schedule_per_machine=initial_schedule
        )
        best_fit = self.aggreg_from_sol(best_solution)
        result_storage.append((best_solution, best_fit))
        cb.on_step_end(1, result_storage, self)

        current_solution = best_solution
        current_fit = best_fit
        current_unscheduled = initial_unscheduled

        num_local_search_iterations = kwargs["num_local_search_iterations"]
        ruin_strategies = ["tardiness", "latest_finish", "setup_cost", "random"]
        initial_temperature = 0.01 * abs(best_fit) if best_fit != 0 else 1000.0
        cooling_rate = kwargs["cooling_rate"]
        temperature = initial_temperature

        for iter_ls in range(num_local_search_iterations):
            current_ruin_strategy = random.choice(ruin_strategies)
            worst_batch_info = _identify_worst_batch_and_affected_tasks(
                current_solution, strategy=current_ruin_strategy
            )
            if worst_batch_info is None:
                break

            machine_id_to_clear, batch_index_to_clear, tasks_to_reschedule = (
                worst_batch_info
            )
            temp_schedule = {
                m: list(s) for m, s in current_solution.schedule_per_machine.items()
            }
            temp_schedule[machine_id_to_clear] = temp_schedule[machine_id_to_clear][
                :batch_index_to_clear
            ]
            tasks_for_next_pass = current_unscheduled | tasks_to_reschedule

            new_schedule, new_unscheduled = _run_greedy_scheduling_phase(
                temp_schedule, tasks_for_next_pass
            )
            new_solution = OvenSchedulingSolution(
                problem=self.problem, schedule_per_machine=new_schedule
            )
            new_fit = self.aggreg_from_sol(new_solution)

            if new_fit < best_fit:
                best_solution = new_solution
                best_fit = new_fit
                result_storage.append((best_solution, best_fit))

            # SA Acceptance Criterion
            if new_fit < current_fit or (
                temperature > 1e-6
                and random.random() < math.exp(-(new_fit - current_fit) / temperature)
            ):
                current_solution = new_solution
                current_fit = new_fit
                current_unscheduled = new_unscheduled

            temperature *= cooling_rate
            stop = cb.on_step_end(iter_ls + 1, result_storage, self)
            if stop:
                break
        cb.on_solve_end(result_storage, self)
        return result_storage
