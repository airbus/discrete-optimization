#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""DIDPpy-based solver for the Oven Scheduling Problem."""

import re
from typing import Any

import didppy as dp

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
)


class DpOvenSchedulingSolver(DpSolver):
    """
    Dynamic Programming solver for the Oven Scheduling Problem using DIDPpy.

    This solver properly models batching by:
    - Tracking current open batch on each machine
    - Managing batch duration (fixed or dynamic based on allow_batch_duration_update)
    - Checking duration compatibility when adding subsequent jobs to a batch
    - Two transition types: add to batch vs. close batch and start new one
    - Respecting machine availability windows
    - Using weighted multi-objective cost function (tardiness + processing + setup)
    - Optional dual bounds to guide search (use_dual_bounds=True)
    - Optional transition dominance rules (use_dominance=True)

    Batch duration models:
    - Fixed (allow_batch_duration_update=False, default): Duration is set to first job's
      min_duration and never changes. Subsequent jobs must fit within this duration.
    - Dynamic (allow_batch_duration_update=True): Duration can increase when adding jobs
      with larger min_duration, up to the minimum max_duration of all jobs in batch.
      More flexible but may slightly misevaluate tardiness.

    The objective function correctly includes:
    - Tardiness costs (paid when each job is added to a batch)
    - Setup costs (paid when closing a batch and starting a new one)
    - Processing costs (paid when closing a batch + at base case for final batches)

    Transition dominance rules (3 rules when use_dominance=True):
    1. Add to batch > Close and start new (when job fits: avoids setup costs)
    2. Non-late job > Late job (when same attribute: prioritizes on-time delivery)
    3. Tighter deadline > Looser deadline (when both on time: reduces future conflicts)
    """

    problem: OvenSchedulingProblem

    def _get_objective_weights(self) -> tuple[int, int, int]:
        """Extract objective weights from params_objective_function."""
        weight_tardiness = 1
        weight_processing = 1
        weight_setup = 1

        if self.params_objective_function is not None:
            for i, obj_name in enumerate(self.params_objective_function.objectives):
                if obj_name == "nb_late_jobs":
                    weight_tardiness = self.params_objective_function.weights[i]
                elif obj_name == "processing_time":
                    weight_processing = self.params_objective_function.weights[i]
                elif obj_name == "setup_cost":
                    weight_setup = self.params_objective_function.weights[i]

        return int(weight_tardiness), int(weight_processing), int(weight_setup)

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the DIDPpy model with proper batching support.

        Args:
            use_dual_bounds: If True, add dual bounds to guide search (default: False)
            use_dominance: If True, add transition dominance rules (default: True)
            allow_batch_duration_update: If True, allow batch duration to increase when adding
                jobs with larger min_duration (default: False). This provides more flexibility
                but may slightly misevaluate tardiness.
        """
        use_dual_bounds = kwargs.get("use_dual_bounds", False)
        use_dominance = kwargs.get("use_dominance", True)
        allow_batch_duration_update = kwargs.get("allow_batch_duration_update", False)

        # Store for use in retrieve_solution
        self._allow_batch_duration_update = allow_batch_duration_update

        model = dp.Model()

        n_jobs = self.problem.n_jobs
        n_machines = self.problem.n_machines

        # Get objective weights
        weight_tardiness, weight_processing, weight_setup = (
            self._get_objective_weights()
        )

        # Object types
        job = model.add_object_type(number=n_jobs)
        n_attributes = len(self.problem.setup_costs)
        attribute = model.add_object_type(number=n_attributes)

        # State variables
        unscheduled = model.add_set_var(object_type=job, target=set(range(n_jobs)))

        # For each machine, track:
        current_time = [model.add_int_var(target=0) for _ in range(n_machines)]

        # Current open batch
        current_batch = [
            model.add_set_var(object_type=job, target=set()) for _ in range(n_machines)
        ]

        # Batch start time (fixed when batch is created with first task)
        batch_start_time = [model.add_int_var(target=0) for _ in range(n_machines)]

        # Batch duration (can be updated if allow_batch_duration_update=True)
        batch_duration = [model.add_int_var(target=0) for _ in range(n_machines)]

        # Batch max duration potential: minimum of max_durations of all jobs in batch
        # This constrains how much we can increase the batch duration
        # Initialized to large value (10000) when batch is empty
        batch_max_duration_potential = [
            model.add_int_var(target=10000) for _ in range(n_machines)
        ]

        # Batch cumulative size
        batch_size = [model.add_int_var(target=0) for _ in range(n_machines)]

        # Batch attribute (when batch is empty, use last closed batch attribute)
        batch_attribute = [
            model.add_element_var(
                object_type=attribute,
                target=self.problem.machines_data[m].initial_attribute,
            )
            for m in range(n_machines)
        ]

        # Store state variables as instance attributes for debugging/warmstart validation
        self.unscheduled = unscheduled
        self.current_batch = current_batch
        self.batch_start_time = batch_start_time
        self.batch_duration = batch_duration
        self.batch_max_duration_potential = batch_max_duration_potential
        self.batch_size = batch_size
        self.batch_attribute = batch_attribute
        self.current_time = current_time

        # Lookup tables
        job_attributes = [self.problem.tasks_data[j].attribute for j in range(n_jobs)]
        job_min_durations = [
            self.problem.tasks_data[j].min_duration for j in range(n_jobs)
        ]
        job_max_durations = [
            self.problem.tasks_data[j].max_duration for j in range(n_jobs)
        ]
        job_earliest_starts = [
            self.problem.tasks_data[j].earliest_start for j in range(n_jobs)
        ]
        job_latest_ends = [self.problem.tasks_data[j].latest_end for j in range(n_jobs)]
        job_sizes = [self.problem.tasks_data[j].size for j in range(n_jobs)]
        min_duration_table = model.add_int_table(job_min_durations)
        max_duration_table = model.add_int_table(job_max_durations)
        earliest_start_table = model.add_int_table(job_earliest_starts)
        latest_end_table = model.add_int_table(job_latest_ends)
        size_table = model.add_int_table(job_sizes)
        machine_capacities = [
            self.problem.machines_data[m].capacity for m in range(n_machines)
        ]
        capacity_table = model.add_int_table(machine_capacities)
        setup_cost_table = model.add_int_table(self.problem.setup_costs)
        setup_time_table = model.add_int_table(self.problem.setup_times)

        self.transitions = {}
        self.transition_ids = {}  # Store transition IDs for dominance rules

        # Sort jobs to bias transition order:
        # 1. Earlier earliest_start first (jobs that need to start soon)
        # 2. Larger min_duration first (to put them at the beginning of batches)
        # jobs_sorted = sorted(
        #    range(n_jobs),
        #    key=lambda j: (job_earliest_starts[j], -job_min_durations[j])
        # )
        jobs_sorted = range(n_jobs)

        # For each job and machine, create two types of transitions
        # Process jobs in sorted order to bias search toward better decisions
        for j in jobs_sorted:
            eligible_machines = self.problem.tasks_data[j].eligible_machines
            job_attr = job_attributes[j]
            job_min_dur = job_min_durations[j]
            job_max_dur = job_max_durations[j]
            job_size = job_sizes[j]

            for m in eligible_machines:
                machine_capacity = machine_capacities[m]

                # TRANSITION TYPE 1: Add job j to current open batch on machine m (same attribute)
                # We create this transition only for jobs that could be added to a batch with matching attribute
                # Since batch_attribute is a state variable, we create transitions for each possible attribute value

                if allow_batch_duration_update:
                    # Dynamic batch duration model:
                    # - Batch duration can increase to accommodate job's min_duration
                    # - End time uses the potentially updated duration
                    new_batch_duration = dp.max(batch_duration[m], job_min_dur)
                    new_max_potential = dp.min(
                        batch_max_duration_potential[m], job_max_dur
                    )
                    end_time = batch_start_time[m] + new_batch_duration
                    # Compute tardiness cost
                    is_late = (end_time - latest_end_table[j] > 0).if_then_else(1, 0)
                    tardiness_cost = is_late * weight_tardiness
                    add_to_batch = dp.Transition(
                        name=f"add_j{j}_m{m}_attr{job_attr}",
                        cost=tardiness_cost + dp.IntExpr.state_cost(),
                        effects=[
                            (unscheduled, unscheduled.remove(j)),
                            (current_batch[m], current_batch[m].add(j)),
                            (batch_size[m], batch_size[m] + job_size),
                            # Update batch duration to max of current and new job's min_duration
                            (batch_duration[m], new_batch_duration),
                            # Update max duration potential to min of current and new job's max_duration
                            (batch_max_duration_potential[m], new_max_potential),
                        ],
                        preconditions=[
                            unscheduled.contains(j),
                            ~current_batch[m].is_empty(),  # Batch must not be empty
                            batch_attribute[m]
                            == job_attr,  # Check batch has this attribute
                            job_min_dur <= batch_max_duration_potential[m],
                            job_max_dur >= batch_duration[m],
                            # Duration compatibility: new batch duration must not exceed the new potential max
                            # This ensures: max(current_duration, job_min_dur) <= min(current_potential, job_max_dur)
                            # new_batch_duration <= new_max_potential,
                            # Time compatibility: job can start at the fixed batch start time
                            earliest_start_table[j] <= batch_start_time[m],
                            # Capacity: current batch size + new job size must fit
                            batch_size[m] + job_size <= machine_capacity,
                        ],
                    )
                else:
                    # Fixed batch duration model (original behavior):
                    # - Batch duration is fixed when batch is created
                    # - Jobs must fit within this fixed duration
                    end_time = batch_start_time[m] + batch_duration[m]
                    # Compute tardiness cost
                    is_late = (end_time - latest_end_table[j] > 0).if_then_else(1, 0)
                    tardiness_cost = is_late * weight_tardiness

                    add_to_batch = dp.Transition(
                        name=f"add_j{j}_m{m}_attr{job_attr}",
                        cost=tardiness_cost + dp.IntExpr.state_cost(),
                        effects=[
                            (unscheduled, unscheduled.remove(j)),
                            (current_batch[m], current_batch[m].add(j)),
                            (batch_size[m], batch_size[m] + job_size),
                        ],
                        preconditions=[
                            unscheduled.contains(j),
                            ~current_batch[m].is_empty(),  # Batch must not be empty
                            batch_attribute[m]
                            == job_attr,  # Check batch has this attribute
                            # Duration compatibility: job's duration range must contain the fixed batch duration
                            # i.e., job.min_duration <= batch_duration <= job.max_duration
                            job_min_dur <= batch_duration[m],
                            batch_duration[m] <= job_max_dur,
                            # Time compatibility: job can start at the fixed batch start time
                            earliest_start_table[j] <= batch_start_time[m],
                            # Capacity: current batch size + new job size must fit
                            batch_size[m] + job_size <= machine_capacity,
                        ],
                    )

                transition_id = model.add_transition(add_to_batch)
                self.transitions[("add", j, m, job_attr)] = add_to_batch
                self.transition_ids[("add", j, m, job_attr)] = transition_id

                # TRANSITION TYPE 2: Close current batch and start new batch with job j
                # This pays the setup cost and updates time
                # For each availability slot
                machine_avail = self.problem.machines_data[m].availability

                for slot_idx, (avail_start, avail_end) in enumerate(machine_avail):
                    # Check static constraints before creating transition
                    if avail_end - avail_start < job_min_dur:
                        continue
                    if job_size > machine_capacity:
                        continue

                    setup_cost_expr = setup_cost_table[batch_attribute[m], job_attr]
                    setup_time_expr = setup_time_table[batch_attribute[m], job_attr]

                    # When closing a batch, we need to:
                    # 1. Pay processing cost for the closed batch (duration = batch_duration)
                    # 2. Pay setup cost for switching attributes
                    # 3. Start new batch with job j
                    # 4. Fix new batch duration = j.min_duration
                    # 5. Fix new batch start time in this availability window

                    # Batch duration when closing current batch
                    closed_batch_duration = batch_duration[m]

                    # Earliest we can start new batch after closing current one
                    # = batch_start_time + closed_batch_duration + setup_time
                    time_after_close = (
                        batch_start_time[m] + closed_batch_duration + setup_time_expr
                    )

                    # Earliest possible start for new batch in this availability window
                    earliest_possible = dp.max(
                        dp.max(time_after_close, earliest_start_table[j]), avail_start
                    )

                    # New batch duration = min_duration of first job (j)
                    new_batch_duration = job_min_dur
                    end_time = earliest_possible + new_batch_duration

                    # Compute tardiness cost for job j (first job of new batch)
                    is_late = (end_time - latest_end_table[j] > 0).if_then_else(1, 0)
                    tardiness_cost = is_late * weight_tardiness

                    # Cost = processing time of closed batch + setup cost + tardiness of new job
                    transition_cost = (
                        weight_processing * closed_batch_duration
                        + weight_setup * setup_cost_expr
                        + tardiness_cost
                        + dp.IntExpr.state_cost()
                    )

                    close_and_start = dp.Transition(
                        name=f"close_start_j{j}_m{m}_slot{slot_idx}",
                        cost=transition_cost,
                        effects=[
                            (unscheduled, unscheduled.remove(j)),
                            (
                                current_batch[m],
                                model.create_set_const(object_type=job, value={j}),
                            ),
                            (batch_duration[m], new_batch_duration),
                            (batch_start_time[m], earliest_possible),
                            (batch_size[m], job_size),
                            (batch_attribute[m], job_attr),
                            (current_time[m], earliest_possible),  # Track current time
                            # Initialize max duration potential for new batch
                            (batch_max_duration_potential[m], job_max_dur),
                        ],
                        preconditions=[
                            unscheduled.contains(j),
                            # Must fit in availability window
                            end_time <= avail_end,
                            earliest_possible >= avail_start,
                        ],
                    )
                    transition_id = model.add_transition(close_and_start)
                    self.transitions[("close_start", j, m, slot_idx)] = close_and_start
                    self.transition_ids[("close_start", j, m, slot_idx)] = transition_id

        # Transition dominance rules
        if use_dominance:
            # Rule 1: Adding to current batch dominates closing and starting new batch
            # If we can fit job j in the current batch on machine m, prefer that over
            # closing the batch and starting a new one (avoids setup costs)
            dominance_count = 0
            for j in jobs_sorted:  # Use sorted jobs list
                job_attr = job_attributes[j]
                job_min_dur = job_min_durations[j]
                job_max_dur = job_max_durations[j]
                job_size = job_sizes[j]
                eligible_machines = self.problem.tasks_data[j].eligible_machines

                for m in eligible_machines:
                    machine_capacity = machine_capacities[m]

                    # Check if add_to_batch transition exists for this job/machine/attribute
                    add_key = ("add", j, m, job_attr)
                    if add_key not in self.transition_ids:
                        continue

                    # For all close_start transitions of the same job on the same machine
                    for slot_idx in range(
                        len(self.problem.machines_data[m].availability)
                    ):
                        close_key = ("close_start", j, m, slot_idx)
                        if close_key not in self.transition_ids:
                            continue

                        # Conditions when adding to batch is better than closing:
                        # 1. Current batch is not empty
                        # 2. Batch has the same attribute as job j
                        # 3. Job fits in batch (duration compatibility)
                        # 4. Job fits in capacity
                        conditions = [
                            ~current_batch[m].is_empty(),
                            batch_attribute[m] == job_attr,
                            job_min_dur <= batch_duration[m],
                            batch_duration[m] <= job_max_dur,
                            earliest_start_table[j] <= batch_start_time[m],
                            batch_size[m] + job_size <= machine_capacity,
                        ]

                        model.add_transition_dominance(
                            self.transition_ids[add_key],  # Dominant: add to batch
                            self.transition_ids[
                                close_key
                            ],  # Dominated: close and start
                            conditions,
                        )
                        dominance_count += 1

            # Rule 2: Non-late job dominates late job for similar tasks
            # If two jobs have the same attribute and can both fit in the current batch,
            # prefer the one that won't be late
            dominance_count_2 = 0
            for j1 in jobs_sorted:
                for j2 in jobs_sorted:
                    if j1 == j2:
                        continue

                    # Only consider jobs with same attribute
                    if job_attributes[j1] != job_attributes[j2]:
                        continue

                    attr = job_attributes[j1]

                    # Check if both have overlapping duration ranges
                    # (can potentially fit in the same batch)
                    min_dur_overlap = max(job_min_durations[j1], job_min_durations[j2])
                    max_dur_overlap = min(job_max_durations[j1], job_max_durations[j2])
                    if min_dur_overlap > max_dur_overlap:
                        continue  # No overlap in duration ranges

                    # Find common eligible machines
                    common_machines = set(
                        self.problem.tasks_data[j1].eligible_machines
                    ) & set(self.problem.tasks_data[j2].eligible_machines)

                    for m in common_machines:
                        # Check if both add transitions exist
                        add_key_j1 = ("add", j1, m, attr)
                        add_key_j2 = ("add", j2, m, attr)

                        if (
                            add_key_j1 not in self.transition_ids
                            or add_key_j2 not in self.transition_ids
                        ):
                            continue

                        # Condition: j1 would not be late, but j2 would be late
                        # End time of batch: batch_start_time + batch_duration
                        end_time = batch_start_time[m] + batch_duration[m]

                        j1_not_late = end_time <= latest_end_table[j1]
                        j2_is_late = end_time > latest_end_table[j2]

                        # j1 dominates j2 if j1 is not late and j2 is late
                        model.add_transition_dominance(
                            self.transition_ids[add_key_j1],  # Dominant: non-late job
                            self.transition_ids[add_key_j2],  # Dominated: late job
                            [j1_not_late, j2_is_late],
                        )
                        dominance_count_2 += 1

            # Rule 3: Tighter deadline dominates when both would be on time
            # If we can schedule either j1 or j2, and both would be on time,
            # prefer the one with tighter deadline to avoid future conflicts
            dominance_count_3 = 0
            for j1 in jobs_sorted:
                for j2 in jobs_sorted:
                    if j1 == j2:
                        continue

                    if job_attributes[j1] != job_attributes[j2]:
                        continue

                    # j1 has tighter deadline (earlier latest_end)
                    if job_latest_ends[j1] >= job_latest_ends[j2]:
                        continue

                    # j2's deadline is loose enough that the difference is meaningful (at least 10 time units)
                    if job_latest_ends[j2] - job_latest_ends[j1] < 10:
                        continue

                    # Similar duration requirements
                    min_dur_overlap = max(job_min_durations[j1], job_min_durations[j2])
                    max_dur_overlap = min(job_max_durations[j1], job_max_durations[j2])
                    if min_dur_overlap > max_dur_overlap:
                        continue

                    attr = job_attributes[j1]
                    common_machines = set(
                        self.problem.tasks_data[j1].eligible_machines
                    ) & set(self.problem.tasks_data[j2].eligible_machines)

                    for m in common_machines:
                        add_key_j1 = ("add", j1, m, attr)
                        add_key_j2 = ("add", j2, m, attr)

                        if (
                            add_key_j1 not in self.transition_ids
                            or add_key_j2 not in self.transition_ids
                        ):
                            continue

                        # CRITICAL: Only dominate when BOTH would be on time
                        # This ensures we're not forcing a late job just because it has tight deadline
                        end_time = batch_start_time[m] + batch_duration[m]
                        both_on_time = (end_time <= latest_end_table[j1]) & (
                            end_time <= latest_end_table[j2]
                        )

                        model.add_transition_dominance(
                            self.transition_ids[
                                add_key_j1
                            ],  # Dominant: tighter deadline
                            self.transition_ids[
                                add_key_j2
                            ],  # Dominated: looser deadline
                            [both_on_time],
                        )
                        dominance_count_3 += 1

        # Dual bounds (lower bound on cost-to-go)
        if use_dual_bounds:
            # A dual bound is a lower bound on the remaining cost
            # We use the number of unscheduled jobs to estimate minimum remaining work

            # Count remaining jobs
            remaining_jobs_count = unscheduled.len()

            # Lower bound on processing time: use global minimum across ALL jobs
            # (can't use table.min(unscheduled) as it may panic in certain states)
            remaining_size = batch_size[m]

            global_min_duration = (remaining_jobs_count == 0).if_then_else(
                0, min_duration_table.min(unscheduled)
            )

            # Lower bound: remaining_jobs * min_duration * weight
            # This assumes we can schedule all jobs optimally (which we can't always do)
            processing_lower_bound = (
                remaining_jobs_count * global_min_duration * weight_processing
            )

            # For tardiness: very hard to estimate without knowing which jobs remain
            # Use 0 as conservative estimate (assumes we might schedule everything on time)
            tardiness_lower_bound = 0

            # Setup costs: also hard to predict, use 0
            setup_lower_bound = 0

            dual_bound = (
                processing_lower_bound + tardiness_lower_bound + setup_lower_bound
            )
            model.add_dual_bound(dual_bound)

        # Base case: all jobs scheduled
        # When we reach the base case, we need to pay processing cost for all remaining open batches
        final_processing_cost = sum(
            weight_processing * batch_duration[m] for m in range(n_machines)
        )
        model.add_base_case([unscheduled.is_empty()], cost=final_processing_cost)

        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        """Retrieve solution from DIDPpy by replaying transitions."""

        def extract_info(name: str) -> tuple:
            """Extract transition info from name."""
            if name.startswith("add_"):
                # add_j{j}_m{m}_attr{attr}
                numbers = [int(x) for x in re.findall(r"\d+", name)]
                if len(numbers) >= 3:
                    return ("add", numbers[0], numbers[1], numbers[2])  # j, m, attr
            elif name.startswith("close_start_"):
                # close_start_j{j}_m{m}_slot{slot}
                numbers = [int(x) for x in re.findall(r"\d+", name)]
                if len(numbers) >= 3:
                    return (
                        "close_start",
                        numbers[0],
                        numbers[1],
                        numbers[2],
                    )  # j, m, slot
            return None

        # Simulate the transitions to reconstruct batches
        # Track current state
        current_batches = {
            m: [] for m in range(self.problem.n_machines)
        }  # Current open batch jobs
        closed_batches = {
            m: [] for m in range(self.problem.n_machines)
        }  # List of closed batches
        batch_start_times = [0] * self.problem.n_machines  # Fixed when batch is created
        batch_durations = [0] * self.problem.n_machines  # Fixed when batch is created
        batch_attributes = [
            self.problem.machines_data[m].initial_attribute
            for m in range(self.problem.n_machines)
        ]

        for transition in sol.transitions:
            info = extract_info(transition.name)
            if not info:
                continue

            if info[0] == "add":
                _, j, m, _ = info
                # Add job to current open batch
                current_batches[m].append(j)

                # For dynamic duration model, update the batch duration
                # It becomes the maximum of min_durations of all jobs in the batch
                if (
                    hasattr(self, "_allow_batch_duration_update")
                    and self._allow_batch_duration_update
                ):
                    job_min_dur = self.problem.tasks_data[j].min_duration
                    batch_durations[m] = max(batch_durations[m], job_min_dur)

            elif info[0] == "close_start":
                _, j, m, slot_idx = info

                # Close current batch if not empty
                if current_batches[m]:
                    jobs_in_batch = current_batches[m]
                    attr = batch_attributes[m]

                    # For dynamic model, recompute final duration based on all jobs
                    if (
                        hasattr(self, "_allow_batch_duration_update")
                        and self._allow_batch_duration_update
                    ):
                        # Final duration = max of min_durations of all jobs in batch
                        duration = max(
                            self.problem.tasks_data[j_id].min_duration
                            for j_id in jobs_in_batch
                        )
                        # Verify it doesn't exceed min of max_durations (should already be satisfied by model)
                        max_duration_limit = min(
                            self.problem.tasks_data[j_id].max_duration
                            for j_id in jobs_in_batch
                        )
                        if duration > max_duration_limit:
                            print(
                                f"Warning: batch duration {duration} exceeds max limit {max_duration_limit}"
                            )
                            duration = max_duration_limit
                    else:
                        duration = batch_durations[m]

                    start_time = batch_start_times[m]
                    end_time = start_time + duration

                    closed_batches[m].append(
                        {
                            "jobs": set(jobs_in_batch),
                            "attribute": attr,
                            "start": start_time,
                            "end": end_time,
                        }
                    )

                # Start new batch with job j
                job_data = self.problem.tasks_data[j]

                # Calculate setup time
                setup_time = self.problem.setup_times[batch_attributes[m]][
                    job_data.attribute
                ]

                # Get availability window
                avail_windows = self.problem.machines_data[m].availability
                avail_start, avail_end = avail_windows[slot_idx]

                # Calculate start time for new batch (this is fixed for this batch)
                prev_end = (
                    batch_start_times[m] + batch_durations[m]
                    if current_batches[m]
                    else 0
                )
                start = max(prev_end + setup_time, job_data.earliest_start, avail_start)

                # Fix new batch parameters
                current_batches[m] = [j]
                batch_attributes[m] = job_data.attribute
                batch_start_times[m] = start
                batch_durations[m] = (
                    job_data.min_duration
                )  # Fixed to first job's min_duration

        # Close any remaining open batches
        for m in range(self.problem.n_machines):
            if current_batches[m]:
                jobs_in_batch = current_batches[m]
                attr = batch_attributes[m]

                # For dynamic model, recompute final duration based on all jobs
                if (
                    hasattr(self, "_allow_batch_duration_update")
                    and self._allow_batch_duration_update
                ):
                    duration = max(
                        self.problem.tasks_data[j_id].min_duration
                        for j_id in jobs_in_batch
                    )
                    max_duration_limit = min(
                        self.problem.tasks_data[j_id].max_duration
                        for j_id in jobs_in_batch
                    )
                    if duration > max_duration_limit:
                        print(
                            f"Warning: final batch duration {duration} exceeds max limit {max_duration_limit}"
                        )
                        duration = max_duration_limit
                else:
                    duration = batch_durations[m]

                start_time = batch_start_times[m]
                end_time = start_time + duration

                closed_batches[m].append(
                    {
                        "jobs": set(jobs_in_batch),
                        "attribute": attr,
                        "start": start_time,
                        "end": end_time,
                    }
                )

        # Convert to OvenSchedulingSolution format
        schedule_per_machine = {}
        for m in range(self.problem.n_machines):
            batches = []
            for batch_info in closed_batches[m]:
                batches.append(
                    ScheduleInfo(
                        tasks=batch_info["jobs"],
                        task_attribute=batch_info["attribute"],
                        start_time=batch_info["start"],
                        end_time=batch_info["end"],
                        machine_batch_index=(m, len(batches)),
                    )
                )
            schedule_per_machine[m] = batches
        sol = OvenSchedulingSolution(
            problem=self.problem, schedule_per_machine=schedule_per_machine
        )
        print(
            f"\nRelative performance: {self.aggreg_from_sol(sol) / self.problem.additional_data['ub']:.4f}"
        )
        return sol

    def set_warm_start(self, solution: OvenSchedulingSolution) -> None:
        """Convert an OvenSchedulingSolution into a sequence of transitions for warm start.

        Args:
            solution: An existing solution to use as warm start

        The conversion depends on the batch duration model:
        - Fixed duration model: Jobs in each batch are sorted by min_duration (descending)
          to ensure the first job sets an appropriate batch duration
        - Dynamic duration model: Jobs can be added in any feasible order
        """
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model must be initialized before setting warm start")

        transitions = []

        # Process each machine's schedule
        for machine_id in range(self.problem.n_machines):
            if machine_id not in solution.schedule_per_machine:
                continue

            batches = solution.schedule_per_machine[machine_id]

            for batch_idx, batch_info in enumerate(batches):
                jobs_in_batch = list(batch_info.tasks)

                if not jobs_in_batch:
                    continue

                # Sort jobs for optimal transition sequence
                # CRITICAL: The first job sets batch_start_time in the DP model.
                # All subsequent jobs must satisfy: earliest_start <= batch_start_time
                # Therefore, we must start with the job that has the LATEST earliest_start
                # to ensure all jobs in the batch can satisfy this constraint.
                # Secondary sort by min_duration (descending) for fixed duration model.
                jobs_sorted = sorted(
                    jobs_in_batch,
                    key=lambda j: (
                        -self.problem.tasks_data[
                            j
                        ].earliest_start,  # Latest earliest_start first
                        -self.problem.tasks_data[
                            j
                        ].min_duration,  # Then largest min_duration
                    ),
                )

                # First job: close previous batch and start new one
                first_job = jobs_sorted[0]
                job_attr = self.problem.tasks_data[first_job].attribute

                # Find appropriate availability slot for this batch
                slot_idx = self._find_slot_for_batch(machine_id, batch_info.start_time)

                # Create close_and_start transition
                transition_key = ("close_start", first_job, machine_id, slot_idx)
                if transition_key in self.transitions:
                    transitions.append(self.transitions[transition_key])

                # Remaining jobs: add to current batch
                for job_id in jobs_sorted[1:]:
                    job_attr = self.problem.tasks_data[job_id].attribute

                    # Create add_to_batch transition
                    transition_key = ("add", job_id, machine_id, job_attr)
                    if transition_key in self.transitions:
                        transitions.append(self.transitions[transition_key])

        # Set the initial solution in DIDPpy
        if transitions:
            self.initial_solution = transitions
            print(f"Warm start: converted solution to {len(transitions)} transitions")
        else:
            print(
                "Warning: could not generate any transitions from warm start solution"
            )

    def _find_slot_for_batch(self, machine_id: int, start_time: int) -> int:
        """Find the availability slot index that contains the given start time."""
        availability = self.problem.machines_data[machine_id].availability

        for slot_idx, (slot_start, slot_end) in enumerate(availability):
            if slot_start <= start_time < slot_end:
                return slot_idx

        # Fallback: return first slot
        return 0
