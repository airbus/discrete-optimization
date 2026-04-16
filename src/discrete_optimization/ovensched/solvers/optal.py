#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""OptalCP solver for the Oven Scheduling Problem."""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
)

try:
    import optalcp as cp

    optalcp_available = True
except ImportError:
    cp = None
    optalcp_available = False

logger = logging.getLogger(__name__)


class OvenObjectivesEnum(Enum):
    """Enumeration of available objectives for the oven scheduling problem."""

    NB_BATCH = "nb_batch"
    PROCESSING_TIME = "processing_time"
    SETUP_COST = "setup_cost"
    TARDINESS = "tardiness"
    MAKESPAN = "makespan"


class OvenSchedulingOptalSolver(OptalCpSolver, WarmstartMixin):
    """OptalCP solver for the Oven Scheduling Problem.

    This solver uses the OptalCP constraint programming solver to model
    the batching problem with machines, setup times/costs, and capacity constraints.
    """

    problem: OvenSchedulingProblem

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        max_nb_batch_per_machine: int | None = None,
        setup_modeling: str = "baseline",
        use_batch_count_vars: bool = False,
        use_explicit_batch_attrs: bool = False,
        use_incompatibility_constraints: bool = False,
        **kwargs: Any,
    ):
        """Initialize the OptalCP solver.

        Args:
            problem: The oven scheduling problem instance
            params_objective_function: Parameters for objective function
            max_nb_batch_per_machine: Maximum number of batches per machine.
                If None, estimated as 3 * n_jobs / n_machines
            setup_modeling: Setup cost modeling approach:
                - "baseline": Nested implies constraints (current approach)
                - "explicit_attrs": Explicit attribute variables per batch with element constraint
                - "direct_sequence": Compute from sequence variable (experimental)
            use_batch_count_vars: If True, add explicit integer variable for number of batches per machine
            use_explicit_batch_attrs: If True, add explicit attribute variables for better branching
            use_incompatibility_constraints: If True, add constraints forbidding incompatible jobs
                from being in the same batch. Jobs are incompatible if they have different attributes
                or non-overlapping duration ranges. Default False as these constraints may hurt
                performance by over-restricting the search space.
            **kwargs: Additional arguments passed to parent
        """
        if not optalcp_available:
            raise RuntimeError(
                "OptalCP is not available. Install it from: https://www.optalcp.com/"
            )
        super().__init__(problem, params_objective_function, **kwargs)

        self.max_nb_batch = max_nb_batch_per_machine
        if self.max_nb_batch is None:
            self.max_nb_batch = max(
                1, int(4 * self.problem.n_jobs / self.problem.n_machines)
            )

        # Modeling options
        self.setup_modeling = setup_modeling
        self.use_batch_count_vars = use_batch_count_vars
        self.use_explicit_batch_attrs = use_explicit_batch_attrs
        self.use_incompatibility_constraints = use_incompatibility_constraints

        # Index tasks by eligible machines
        self.tasks_per_machine = {
            m: [
                t
                for t in range(self.problem.n_jobs)
                if m in self.problem.tasks_data[t].eligible_machines
            ]
            for m in range(self.problem.n_machines)
        }

        self.variables = {}
        self.current_solution = None

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the OptalCP model."""
        self.cp_model = cp.Model()
        self.variables = {}

        # Create decision variables and constraints
        self._define_machine_calendar()
        self._define_machines_intervals()
        self._constraint_convention_machines()
        self._define_jobs_intervals()

        # Optionally add incompatibility constraints
        if self.use_incompatibility_constraints:
            self._add_job_incompatibility_constraints()

        # Setup cost modeling (choose based on option)
        if self.setup_modeling == "baseline":
            self._constraint_setup_time_baseline()
        elif self.setup_modeling == "explicit_attrs":
            self._constraint_setup_time_explicit_attrs()
        elif self.setup_modeling == "direct_sequence":
            self._constraint_setup_time_direct_sequence()
        else:
            raise ValueError(f"Unknown setup_modeling: {self.setup_modeling}")

        # Optional: explicit batch count variables
        if self.use_batch_count_vars:
            self._add_batch_count_variables()

        # Set objectives based on params_objective_function
        self._define_objectives_from_params()

    def _add_job_incompatibility_constraints(self):
        """Add constraints to forbid incompatible jobs from being in the same batch.

        Two jobs are incompatible if:
        1. They have different attributes
        2. Their duration ranges don't overlap (no common valid duration)

        Note: While these constraints are logically valid, they may hurt performance
        by over-restricting the search space and removing useful stepping stones for
        the LNS search. Use use_incompatibility_constraints=True to enable.
        """
        print("Adding job incompatibility constraints...")

        # Precompute incompatible pairs
        incompatible_pairs = []
        for i in range(self.problem.n_jobs):
            for j in range(i + 1, self.problem.n_jobs):
                task_i = self.problem.tasks_data[i]
                task_j = self.problem.tasks_data[j]

                # Different attributes => incompatible (can't be in same batch)
                if task_i.attribute != task_j.attribute:
                    incompatible_pairs.append((i, j))
                    continue

                # Check duration compatibility
                # Jobs are incompatible if their duration ranges don't overlap
                min_duration = max(task_i.min_duration, task_j.min_duration)
                max_duration = min(task_i.max_duration, task_j.max_duration)

                if min_duration > max_duration:
                    # No valid duration satisfies both jobs
                    incompatible_pairs.append((i, j))

        print(f"Found {len(incompatible_pairs)} incompatible job pairs")

        # Add constraints for incompatible pairs
        # If job i and job j are incompatible, they cannot be in the same batch on any machine
        for i, j in incompatible_pairs:
            task_i = self.problem.tasks_data[i]
            task_j = self.problem.tasks_data[j]

            # Get common eligible machines
            common_machines = set(task_i.eligible_machines) & set(
                task_j.eligible_machines
            )

            for m in common_machines:
                if (
                    i in self.variables["index_batch_machine_for_job"]
                    and m in self.variables["index_batch_machine_for_job"][i]
                    and j in self.variables["index_batch_machine_for_job"]
                    and m in self.variables["index_batch_machine_for_job"][j]
                ):
                    # Get batch index variables for jobs i and j on machine m
                    batch_index_i = self.variables["index_batch_machine_for_job"][i][m]
                    batch_index_j = self.variables["index_batch_machine_for_job"][j][m]

                    # If both jobs are on machine m, they must be in different batches
                    # This is expressed as: batch_index_i != batch_index_j
                    # when both jobs are present on machine m

                    interval_i_m = self.variables["interval_job_per_machine"][i][m]
                    interval_j_m = self.variables["interval_job_per_machine"][j][m]

                    # Constraint: if both are present on machine m, they must have different batch indices
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            self.cp_model.presence(interval_i_m)
                            & self.cp_model.presence(interval_j_m),
                            batch_index_i != batch_index_j,
                        )
                    )

    def _define_machine_calendar(self):
        """Define calendar constraints for machine availability."""
        self.calendar_step_function = {}
        for m in range(self.problem.n_machines):
            availability = self.problem.machines_data[m].availability
            list_values = []

            for start, end in availability:
                if len(list_values) > 0:
                    if start == list_values[-1][0]:
                        list_values[-1] = (start, 1)
                        list_values.append((end + 1, 0))
                        continue
                list_values.append((start, 1))
                list_values.append((end, 0))

            self.calendar_step_function[m] = self.cp_model.step_function(list_values)

    def _define_machines_intervals(self):
        """Create interval variables for batches on each machine."""
        horizon = max(
            interval[1]
            for machine_data in self.problem.machines_data
            for interval in machine_data.availability
        )

        intervals_per_machines = {m: [] for m in range(self.problem.n_machines)}
        intervals_per_machines_per_attr = {
            m: {} for m in range(self.problem.n_machines)
        }

        for m in range(self.problem.n_machines):
            tasks = self.tasks_per_machine[m]
            attributes = set(self.problem.tasks_data[t].attribute for t in tasks)

            # Get possible durations for this machine
            possible_durations = []
            for t in tasks:
                possible_durations.extend(
                    [
                        self.problem.tasks_data[t].min_duration,
                        self.problem.tasks_data[t].max_duration,
                    ]
                )
            possible_durations = sorted(set(possible_durations))

            # Create batch intervals for this machine
            for b in range(self.max_nb_batch):
                interval = self.cp_model.interval_var(
                    start=(0, horizon),
                    length=(min(possible_durations), max(possible_durations)),
                    end=(0, horizon),
                    optional=True,
                    name=f"interval_machine_{m}_{b}",
                )
                intervals_per_machines[m].append(interval)
                # Forbid extent outside calendar
                self.cp_model.forbid_extent(interval, self.calendar_step_function[m])
            # Create attribute-specific intervals
            for attr in attributes:
                attr_durations = []
                for t in tasks:
                    if self.problem.tasks_data[t].attribute == attr:
                        attr_durations.extend(
                            [
                                self.problem.tasks_data[t].min_duration,
                                self.problem.tasks_data[t].max_duration,
                            ]
                        )
                attr_durations = sorted(set(attr_durations))

                intervals_per_machines_per_attr[m][attr] = []
                for b in range(self.max_nb_batch):
                    interval = self.cp_model.interval_var(
                        start=(0, horizon),
                        length=(min(attr_durations), max(attr_durations)),
                        end=(0, horizon),
                        optional=True,
                        name=f"interval_machine_{m}_{b}_attr{attr}",
                    )
                    intervals_per_machines_per_attr[m][attr].append(interval)

            # Link main intervals with attribute intervals (alternative constraint)
            for b in range(self.max_nb_batch):
                self.cp_model.alternative(
                    intervals_per_machines[m][b],
                    [
                        intervals_per_machines_per_attr[m][attr][b]
                        for attr in attributes
                    ],
                )

        self.variables["intervals_per_machines"] = intervals_per_machines
        self.variables["intervals_per_machines_per_attr"] = (
            intervals_per_machines_per_attr
        )

    def _constraint_convention_machines(self):
        """Add ordering constraints on batches within each machine."""
        intervals_per_machines = self.variables["intervals_per_machines"]
        intervals_per_machines_per_attr = self.variables[
            "intervals_per_machines_per_attr"
        ]

        for m in intervals_per_machines:
            # Batches must be used in order (no gaps)
            for i in range(self.max_nb_batch - 1):
                self.cp_model.enforce(
                    self.cp_model.presence(intervals_per_machines[m][i])
                    >= self.cp_model.presence(intervals_per_machines[m][i + 1])
                )
                # Batch i+1 must start after batch i ends
                self.cp_model.end_before_start(
                    intervals_per_machines[m][i], intervals_per_machines[m][i + 1]
                )

            # Hidden constraint for presence chain
            self.cp_model._itv_presence_chain(intervals_per_machines[m])

            # Each batch has exactly one attribute (aids propagation even if redundant)
            for i in range(self.max_nb_batch):
                attributes = list(intervals_per_machines_per_attr[m].keys())
                self.cp_model.alternative(
                    intervals_per_machines[m][i],
                    [
                        intervals_per_machines_per_attr[m][attr][i]
                        for attr in attributes
                    ],
                )

    def _define_jobs_intervals(self):
        """Create interval variables for jobs and link them to batches."""
        horizon = max(
            interval[1]
            for machine_data in self.problem.machines_data
            for interval in machine_data.availability
        )
        interval_job = {}
        interval_job_per_machine = {}
        index_batch_machine_for_job = {t: {} for t in range(self.problem.n_jobs)}
        interval_batch_machine_for_job = {t: {} for t in range(self.problem.n_jobs)}
        # Create job intervals
        for t in self.problem.tasks_list:
            task_data = self.problem.tasks_data[t]
            # Main job interval
            interval_job[t] = self.cp_model.interval_var(
                start=(task_data.earliest_start, horizon),
                end=(
                    task_data.earliest_start + task_data.min_duration,
                    horizon,
                ),
                length=(task_data.min_duration, task_data.max_duration),
                optional=False,
                name=f"interval_job_{t}",
            )

            # Job intervals per machine
            interval_job_per_machine[t] = {}
            for m in task_data.eligible_machines:
                interval_job_per_machine[t][m] = self.cp_model.interval_var(
                    start=(task_data.earliest_start, horizon),
                    end=(
                        task_data.earliest_start + task_data.min_duration,
                        horizon,
                    ),
                    length=(task_data.min_duration, task_data.max_duration),
                    optional=True,
                    name=f"interval_job_{t}_{m}",
                )
            # Alternative: job must be on exactly one machine
            self.cp_model.alternative(
                interval_job[t],
                [interval_job_per_machine[t][m] for m in interval_job_per_machine[t]],
            )
        for m in range(self.problem.n_machines):
            tasks = self.tasks_per_machine[m]
            intervals_on_machine = [
                (
                    self.cp_model.interval_var(
                        start=(0, self.max_nb_batch - 1),
                        end=(1, self.max_nb_batch),
                        length=1,
                        optional=True,
                    ),
                    self.problem.tasks_data[t].size,
                )
                for t in tasks
            ]
            index = [self.cp_model.start(x[0]) for x in intervals_on_machine]
            self.cp_model._itv_mapping(
                [interval_job_per_machine[t][m] for t in tasks],
                [
                    self.variables["intervals_per_machines"][m][b]
                    for b in range(self.max_nb_batch)
                ],
                index,
            )
            # Capacity constraints using pulse (no _pack needed)
            self.cp_model.enforce(
                self.cp_model.sum(
                    [self.cp_model.pulse(x[0], x[1]) for x in intervals_on_machine]
                )
                <= self.problem.machines_data[m].capacity
            )
            self.cp_model.enforce(
                self.cp_model.sum(
                    [
                        self.cp_model.pulse(
                            interval_job_per_machine[t][m],
                            self.problem.tasks_data[t].size,
                        )
                        for t in tasks
                    ]
                )
                <= self.problem.machines_data[m].capacity
            )
            for i in range(len(tasks)):
                interval_batch_machine_for_job[tasks[i]][m] = intervals_on_machine[i][0]
                index_batch_machine_for_job[tasks[i]][m] = index[
                    i
                ]  # Store batch index variable
                attr = self.problem.tasks_data[tasks[i]].attribute
                self.cp_model.enforce(
                    self.cp_model.presence(intervals_on_machine[i][0])
                    == self.cp_model.presence(interval_job_per_machine[tasks[i]][m])
                )
                for val in range(self.max_nb_batch):
                    # Attribute presence constraint (simpler than implies)
                    self.cp_model.enforce(
                        self.cp_model.presence(
                            self.variables["intervals_per_machines_per_attr"][m][attr][
                                val
                            ]
                        )
                        >= (val == index[i])
                    )

                    # Duration constraints
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            val == index[i],
                            self.cp_model.length(
                                self.variables["intervals_per_machines"][m][val]
                            )
                            >= self.problem.tasks_data[tasks[i]].min_duration,
                        )
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            val == index[i],
                            self.cp_model.length(
                                self.variables["intervals_per_machines"][m][val]
                            )
                            <= self.problem.tasks_data[tasks[i]].max_duration,
                        )
                    )

        self.variables["interval_job"] = interval_job
        self.variables["interval_job_per_machine"] = interval_job_per_machine
        self.variables["index_batch_machine_for_job"] = index_batch_machine_for_job
        self.variables["interval_batch_machine_for_job"] = (
            interval_batch_machine_for_job
        )

    def _constraint_setup_time_baseline(self):
        """Add setup time and cost constraints (baseline approach with nested implies)."""
        possible_setup_times = sorted(
            set([0] + [x for row in self.problem.setup_times for x in row])
        )
        possible_setup_costs = sorted(
            set([x for row in self.problem.setup_costs for x in row])
        )

        setup_cost = {
            m: [
                self.cp_model.int_var(
                    min=min(possible_setup_costs),
                    max=max(possible_setup_costs),
                    optional=True,
                    name=f"setup_cost_m{m}_b{b}",
                )
                for b in range(self.max_nb_batch)
            ]
            for m in range(self.problem.n_machines)
        }

        # Store dummy initial intervals for warm start
        self.variables["dummy_initial_intervals"] = {}

        for m in range(self.problem.n_machines):
            machine_data = self.problem.machines_data[m]
            intervals_per_attr = self.variables["intervals_per_machines_per_attr"][m]

            # Create sequence with initial state - store dummy interval
            dummy_initial = self.cp_model.interval_var(-1, 0, 1)
            self.variables["dummy_initial_intervals"][m] = dummy_initial

            all_intervals = [(dummy_initial, machine_data.initial_attribute)]
            all_intervals += [
                (intervals_per_attr[attr][i], attr)
                for attr in intervals_per_attr
                for i in range(self.max_nb_batch)
            ]

            seq = self.cp_model.sequence_var(
                [x[0] for x in all_intervals], types=[x[1] for x in all_intervals]
            )

            # No overlap with setup times
            seq.no_overlap(self.problem.setup_times)

            # Setup cost constraints
            # First batch setup from initial state
            for attr in intervals_per_attr:
                iv = intervals_per_attr[attr][0]
                self.cp_model.enforce(
                    self.cp_model.implies(
                        self.cp_model.presence(iv),
                        setup_cost[m][0]
                        == self.problem.setup_costs[machine_data.initial_attribute][
                            attr
                        ],
                    )
                )

            # Subsequent batches
            for i in range(1, self.max_nb_batch):
                for prev_attr in intervals_per_attr:
                    iv_prev = intervals_per_attr[prev_attr][i - 1]
                    for curr_attr in intervals_per_attr:
                        iv_curr = intervals_per_attr[curr_attr][i]
                        self.cp_model.enforce(
                            self.cp_model.implies(
                                self.cp_model.presence(iv_prev)
                                & self.cp_model.presence(iv_curr),
                                setup_cost[m][i]
                                == self.problem.setup_costs[prev_attr][curr_attr],
                            )
                        )

            # Link setup cost presence to batch presence
            for i in range(self.max_nb_batch):
                self.cp_model.enforce(
                    self.cp_model.presence(
                        self.variables["intervals_per_machines"][m][i]
                    )
                    == self.cp_model.presence(setup_cost[m][i])
                )

        self.variables["setup_cost_per_machine"] = setup_cost

    def _constraint_setup_time_explicit_attrs(self):
        """Add setup constraints with explicit attribute variables per batch.

        This approach creates integer variables for the attribute assigned to each batch,
        allowing the solver to branch on high-level decisions before fixing detailed timings.
        """
        possible_setup_costs = sorted(
            set([x for row in self.problem.setup_costs for x in row])
        )

        setup_cost = {
            m: [
                self.cp_model.int_var(
                    min=min(possible_setup_costs),
                    max=max(possible_setup_costs),
                    optional=True,
                    name=f"setup_cost_m{m}_b{b}",
                )
                for b in range(self.max_nb_batch)
            ]
            for m in range(self.problem.n_machines)
        }

        # Explicit attribute variables per batch (for branching)
        batch_attribute = {}

        # Store dummy initial intervals for warm start
        self.variables["dummy_initial_intervals"] = {}

        for m in range(self.problem.n_machines):
            machine_data = self.problem.machines_data[m]
            intervals_per_attr = self.variables["intervals_per_machines_per_attr"][m]
            all_attributes = sorted(intervals_per_attr.keys())

            batch_attribute[m] = []
            for b in range(self.max_nb_batch):
                attr_var = self.cp_model.int_var(
                    min=min(all_attributes),
                    max=max(all_attributes),
                    optional=True,
                    name=f"batch_attr_m{m}_b{b}",
                )
                batch_attribute[m].append(attr_var)

                # Link attribute variable to attribute interval presence
                for attr in all_attributes:
                    self.cp_model.enforce(
                        self.cp_model.presence(intervals_per_attr[attr][b])
                        == (attr_var == attr)
                    )

            # Create sequence with initial state - store dummy interval
            dummy_initial = self.cp_model.interval_var(-1, 0, 1)
            self.variables["dummy_initial_intervals"][m] = dummy_initial

            all_intervals = [(dummy_initial, machine_data.initial_attribute)]
            all_intervals += [
                (intervals_per_attr[attr][i], attr)
                for attr in intervals_per_attr
                for i in range(self.max_nb_batch)
            ]

            seq = self.cp_model.sequence_var(
                [x[0] for x in all_intervals], types=[x[1] for x in all_intervals]
            )

            # No overlap with setup times
            seq.no_overlap(self.problem.setup_times)

            # Setup costs using element constraint
            # First batch: use element to get cost from initial_attribute to batch_attribute[0]
            if self.max_nb_batch > 0:
                costs_from_initial = [
                    self.problem.setup_costs[machine_data.initial_attribute][attr]
                    for attr in all_attributes
                ]
                # Use table constraint for first batch
                for attr_idx, attr in enumerate(all_attributes):
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            batch_attribute[m][0] == attr,
                            setup_cost[m][0] == costs_from_initial[attr_idx],
                        )
                    )

            # Subsequent batches: use element on prev_attr and curr_attr
            for i in range(1, self.max_nb_batch):
                # For each possible transition, add constraint
                for prev_attr in all_attributes:
                    for curr_attr in all_attributes:
                        self.cp_model.enforce(
                            self.cp_model.implies(
                                (batch_attribute[m][i - 1] == prev_attr)
                                & (batch_attribute[m][i] == curr_attr),
                                setup_cost[m][i]
                                == self.problem.setup_costs[prev_attr][curr_attr],
                            )
                        )

            # Link setup cost presence to batch presence
            for i in range(self.max_nb_batch):
                self.cp_model.enforce(
                    self.cp_model.presence(
                        self.variables["intervals_per_machines"][m][i]
                    )
                    == self.cp_model.presence(setup_cost[m][i])
                )
                self.cp_model.enforce(
                    self.cp_model.presence(
                        self.variables["intervals_per_machines"][m][i]
                    )
                    == self.cp_model.presence(batch_attribute[m][i])
                )

        self.variables["setup_cost_per_machine"] = setup_cost
        self.variables["batch_attribute"] = batch_attribute

    def _constraint_setup_time_direct_sequence(self):
        """Add setup constraints using cost-space sequence.

        Revolutionary approach: Create a parallel sequence in "cost space"!
        - Time dimension = cumulative setup cost (not real time)
        - Each batch is an interval at position = cumulative cost so far
        - Transitions use setup_costs as "setup times" in this cost space
        - The sequence.no_overlap with cost matrix automatically computes costs!
        - _same_sequence ensures time and cost sequences have identical ordering
        - Total setup cost = makespan of cost sequence

        This eliminates the O(n_attr² × n_batches) constraint explosion!
        """
        setup_cost = {m: [] for m in range(self.problem.n_machines)}

        # Store dummy initial intervals for warm start
        self.variables["dummy_initial_intervals"] = {}

        for m in range(self.problem.n_machines):
            machine_data = self.problem.machines_data[m]
            intervals_per_attr = self.variables["intervals_per_machines_per_attr"][m]
            all_attributes = sorted(intervals_per_attr.keys())

            # Standard time-based sequence for actual scheduling - store dummy interval
            dummy_initial = self.cp_model.interval_var(-1, 0, 1)
            self.variables["dummy_initial_intervals"][m] = dummy_initial

            time_intervals = [(dummy_initial, machine_data.initial_attribute)]
            time_intervals += [
                (intervals_per_attr[attr][i], attr)
                for attr in intervals_per_attr
                for i in range(self.max_nb_batch)
            ]

            seq_time = self.cp_model.sequence_var(
                [x[0] for x in time_intervals], types=[x[1] for x in time_intervals]
            )
            seq_time.no_overlap(self.problem.setup_times)

            # COST-SPACE SEQUENCE - this is the innovation!
            # Create intervals in cost-space (not time-space)
            max_total_cost = (
                sum(max(row) for row in self.problem.setup_costs) * self.max_nb_batch
            )

            # Initial interval at cost=0
            cost_initial = self.cp_model.interval_var(
                start=0, end=0, length=0, optional=False, name=f"cost_init_m{m}"
            )

            # Cost intervals: one per (batch, attribute) pair
            # These are positioned in cost-space, not time-space!
            cost_intervals_per_attr = {attr: [] for attr in all_attributes}

            for b in range(self.max_nb_batch):
                for attr in all_attributes:
                    # Position in cost-space = cumulative setup cost
                    cost_itv = self.cp_model.interval_var(
                        start=(0, max_total_cost),
                        length=0,  # Point in cost-space
                        end=(0, max_total_cost),
                        optional=True,
                        name=f"cost_m{m}_b{b}_attr{attr}",
                    )
                    cost_intervals_per_attr[attr].append(cost_itv)

                    # Link cost interval presence to actual batch-attribute
                    self.cp_model.enforce(
                        self.cp_model.presence(cost_itv)
                        == self.cp_model.presence(intervals_per_attr[attr][b])
                    )

            # Build cost-space sequence
            cost_sequence_intervals = [cost_initial]
            cost_sequence_types = [machine_data.initial_attribute]

            for attr in all_attributes:
                for b in range(self.max_nb_batch):
                    cost_sequence_intervals.append(cost_intervals_per_attr[attr][b])
                    cost_sequence_types.append(attr)

            seq_cost = self.cp_model.sequence_var(
                cost_sequence_intervals, types=cost_sequence_types
            )

            # THE MAGIC: Use setup_costs as transition matrix in COST-space!
            # This automatically positions intervals to satisfy: pos[i+1] = pos[i] + cost[attr[i]][attr[i+1]]
            seq_cost.no_overlap(self.problem.setup_costs)

            # Ensure time and cost sequences have the SAME ordering
            # This is critical - they must select the same batches in the same order
            self.cp_model._same_sequence(seq_time, seq_cost)

            # Total setup cost = end position of last batch in cost-space
            # Since batches are ordered and cost-space intervals have length=0,
            # the end of the last present interval = cumulative setup cost
            #
            # We take max over all batch-attribute cost intervals
            # Non-present intervals should not affect the max
            all_cost_ends = []
            for attr in all_attributes:
                for b in range(self.max_nb_batch):
                    cost_itv = cost_intervals_per_attr[attr][b]
                    all_cost_ends.append(self.cp_model.end(cost_itv))

            total_setup_cost = self.cp_model.int_var(
                min=0, max=max_total_cost, name=f"total_setup_cost_m{m}"
            )

            self.cp_model.enforce(total_setup_cost == self.cp_model.max(all_cost_ends))

            # Store as single-element list for compatibility with objective code
            setup_cost[m].append(total_setup_cost)

        self.variables["setup_cost_per_machine"] = setup_cost

    def _add_batch_count_variables(self):
        """Add explicit integer variables for number of batches per machine.

        Similar to the late job variables, this allows the solver to branch on
        high-level decisions (how many batches?) before fixing batch details.
        """
        nb_batches_per_machine = {}
        for m in range(self.problem.n_machines):
            nb_batches = self.cp_model.int_var(
                min=0, max=self.max_nb_batch, name=f"nb_batches_m{m}"
            )
            nb_batches_per_machine[m] = nb_batches

            # Link to batch presence
            self.cp_model.enforce(
                nb_batches
                == self.cp_model.sum(
                    [
                        self.cp_model.presence(
                            self.variables["intervals_per_machines"][m][i]
                        )
                        for i in range(self.max_nb_batch)
                    ]
                )
            )

        self.variables["nb_batches_per_machine"] = nb_batches_per_machine

    def _define_objectives_from_params(self):
        """Define objectives based on params_objective_function."""
        self.variables["objectives"] = {}
        objective_terms = []

        # Get weights from params_objective_function
        weights_dict = {
            obj_name: weight
            for obj_name, weight in zip(
                self.params_objective_function.objectives,
                self.params_objective_function.weights,
            )
        }

        horizon = max(
            interval[1]
            for machine_data in self.problem.machines_data
            for interval in machine_data.availability
        )

        # Processing time objective
        if "processing_time" in weights_dict:
            processing_time = self.cp_model.sum(
                [
                    self.cp_model.length(self.variables["intervals_per_machines"][m][i])
                    for m in self.variables["intervals_per_machines"]
                    for i in range(self.max_nb_batch)
                ]
            )
            self.variables["objectives"]["processing_time"] = processing_time
            if weights_dict["processing_time"] > 0:
                objective_terms.append(
                    weights_dict["processing_time"] * processing_time
                )

        # Setup cost objective
        if "setup_cost" in weights_dict and "setup_cost_per_machine" in self.variables:
            # For direct_sequence: setup_cost_per_machine[m] is a single total cost
            # For baseline/explicit_attrs: setup_cost_per_machine[m] is a list of per-batch costs
            setup_costs_flat = []
            for m in self.variables["setup_cost_per_machine"]:
                cost_entry = self.variables["setup_cost_per_machine"][m]
                if isinstance(cost_entry, list):
                    setup_costs_flat.extend(cost_entry)
                else:
                    setup_costs_flat.append(cost_entry)

            setup_cost = self.cp_model.sum(setup_costs_flat)
            self.variables["objectives"]["setup_cost"] = setup_cost
            if weights_dict["setup_cost"] > 0:
                objective_terms.append(weights_dict["setup_cost"] * setup_cost)

        # Tardiness objective (number of late jobs)
        if "nb_late_jobs" in weights_dict:
            late_jobs = {}
            for t in range(self.problem.n_jobs):
                task_data = self.problem.tasks_data[t]
                if task_data.latest_end < float("inf"):
                    late = self.cp_model.int_var(name=f"late_{t}")
                    late_jobs[t] = late
                    self.cp_model.enforce(
                        late
                        == (
                            self.cp_model.end(self.variables["interval_job"][t])
                            > task_data.latest_end
                        )
                    )

            if late_jobs:
                tardiness = self.cp_model.sum([late_jobs[t] for t in late_jobs])
                self.variables["objectives"]["nb_late_jobs"] = tardiness
                self.variables["late_jobs"] = late_jobs
                if weights_dict["nb_late_jobs"] > 0:
                    objective_terms.append(weights_dict["nb_late_jobs"] * tardiness)

        # Makespan objective (optional, for compatibility)
        if "makespan" in weights_dict:
            makespan = self.cp_model.int_var(
                min=max(
                    self.problem.tasks_data[t].earliest_start
                    for t in range(self.problem.n_jobs)
                ),
                max=horizon,
                name="makespan",
            )
            self.cp_model.enforce(
                self.cp_model.max(
                    [
                        self.cp_model.end(self.variables["interval_job"][t])
                        for t in self.variables["interval_job"]
                    ]
                )
                == makespan
            )
            self.variables["objectives"]["makespan"] = makespan
            if weights_dict["makespan"] > 0:
                objective_terms.append(weights_dict["makespan"] * makespan)

        # Number of batches objective (optional)
        if "nb_batch" in weights_dict:
            nb_batch = self.cp_model.sum(
                [
                    self.cp_model.presence(
                        self.variables["intervals_per_machines"][m][i]
                    )
                    for m in self.variables["intervals_per_machines"]
                    for i in range(self.max_nb_batch)
                ]
            )
            self.variables["objectives"]["nb_batch"] = nb_batch
            if weights_dict["nb_batch"] > 0:
                objective_terms.append(weights_dict["nb_batch"] * nb_batch)

        # Minimize weighted sum of objectives
        if objective_terms:
            self.cp_model.minimize(self.cp_model.sum(objective_terms))

    def retrieve_solution(self, result: "cp.SolutionEvent") -> OvenSchedulingSolution:
        """Construct a solution from OptalCP solver results.

        Args:
            result: The OptalCP solution event

        Returns:
            An OvenSchedulingSolution
        """
        if result.solution is None:
            # Return empty solution if no solution found
            return OvenSchedulingSolution(
                problem=self.problem,
                schedule_per_machine={m: [] for m in range(self.problem.n_machines)},
            )

        solution = result.solution

        # Log objective values
        logger.info(f"Objective: {solution.get_objective()}")
        # for obj_name in self.variables["objectives"]:
        #    obj_value = solution.get_value(self.variables["objectives"][obj_name])
        #    logger.info(f"{obj_name}: {obj_value}")

        # Extract schedule information
        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        batches_per_machine = {
            m: defaultdict(list) for m in range(self.problem.n_machines)
        }
        type_per_machine = {m: {} for m in range(self.problem.n_machines)}

        # Get batch timings
        intervals_per_machines = self.variables["intervals_per_machines"]
        for m in intervals_per_machines:
            for b in range(self.max_nb_batch):
                if solution.is_present(intervals_per_machines[m][b]):
                    start = solution.get_start(intervals_per_machines[m][b])
                    end = solution.get_end(intervals_per_machines[m][b])
                    schedule_per_machine[m].append((b, start, end))
        # Get job assignments to batches
        interval_batch_machine_for_job = self.variables[
            "interval_batch_machine_for_job"
        ]
        for job_id in interval_batch_machine_for_job:
            for machine in interval_batch_machine_for_job[job_id]:
                if solution.is_present(interval_batch_machine_for_job[job_id][machine]):
                    batch_idx = solution.get_start(
                        interval_batch_machine_for_job[job_id][machine]
                    )
                    type_per_machine[machine][batch_idx] = self.problem.tasks_data[
                        job_id
                    ].attribute
                    batches_per_machine[machine][batch_idx].append(job_id)
        # Build final schedule with ScheduleInfo objects
        final_schedule_per_machine = {}
        for m in range(self.problem.n_machines):
            final_schedule_per_machine[m] = []

            # Sort batches by their position index
            sorted_batches = sorted(schedule_per_machine[m], key=lambda x: x[0])

            for batch_idx, start_time, end_time in sorted_batches:
                if batch_idx in batches_per_machine[m]:
                    tasks_in_batch = set(batches_per_machine[m][batch_idx])
                    if tasks_in_batch:
                        task_attribute = type_per_machine[m][batch_idx]
                        schedule_info = ScheduleInfo(
                            tasks=tasks_in_batch,
                            task_attribute=task_attribute,
                            start_time=start_time,
                            end_time=end_time,
                            machine_batch_index=(m, len(final_schedule_per_machine[m])),
                        )
                        final_schedule_per_machine[m].append(schedule_info)
        oven_solution = OvenSchedulingSolution(
            problem=self.problem, schedule_per_machine=final_schedule_per_machine
        )

        # Log solution quality
        evaluation = self.problem.evaluate(oven_solution)
        if "ub" in self.problem.additional_data:
            gap = (
                self.aggreg_from_sol(oven_solution) / self.problem.additional_data["ub"]
            )
            logger.info(f"Gap to known upper bound: {gap:.4f}")
        logger.info(f"Feasible: {self.problem.satisfy(oven_solution)}")

        # Store current solution for warm start
        self.current_solution = result.solution

        return oven_solution

    def set_warm_start(self, solution: OvenSchedulingSolution) -> None:
        """Set warm start from a given solution.

        Args:
            solution: The solution to use for warm start
        """
        if self.cp_model is None:
            raise RuntimeError("Model must be initialized before setting warm start")

        # Create a CP solution object for warm start
        warm_start = cp.Solution()

        # OptalCP requires hints for ALL variables (present and absent)
        # Build complete warm start

        # Set dummy initial intervals (used in sequence constraints)
        if "dummy_initial_intervals" in self.variables:
            for m in self.variables["dummy_initial_intervals"]:
                # Dummy interval has fixed values: start=-1, length=1, end=0
                warm_start.set_value(
                    self.variables["dummy_initial_intervals"][m], -1, 0
                )

        for m in range(self.problem.n_machines):
            machine_batches = solution.schedule_per_machine.get(m, [])
            prev_attr = self.problem.machines_data[m].initial_attribute

            # Process present batches
            for batch_idx, sched_info in enumerate(machine_batches):
                if batch_idx < self.max_nb_batch:
                    # Set batch interval (present)
                    interval = self.variables["intervals_per_machines"][m][batch_idx]
                    warm_start.set_value(
                        interval, sched_info.start_time, sched_info.end_time
                    )

                    # Set attribute interval for used attribute (present)
                    attr = sched_info.task_attribute
                    attr_interval = self.variables["intervals_per_machines_per_attr"][
                        m
                    ][attr][batch_idx]
                    warm_start.set_value(
                        attr_interval, sched_info.start_time, sched_info.end_time
                    )

                    # Set absent all OTHER attribute intervals for this batch
                    for other_attr in self.variables["intervals_per_machines_per_attr"][
                        m
                    ]:
                        if other_attr != attr:
                            warm_start.set_absent(
                                self.variables["intervals_per_machines_per_attr"][m][
                                    other_attr
                                ][batch_idx]
                            )

                    # Set setup cost if applicable
                    if "setup_cost_per_machine" in self.variables:
                        if isinstance(
                            self.variables["setup_cost_per_machine"][m], list
                        ):
                            setup_cost = self.problem.setup_costs[prev_attr][attr]
                            warm_start.set_value(
                                self.variables["setup_cost_per_machine"][m][batch_idx],
                                setup_cost,
                            )

                    # Set batch attribute variable if using explicit_attrs
                    if "batch_attribute" in self.variables:
                        warm_start.set_value(
                            self.variables["batch_attribute"][m][batch_idx], attr
                        )

                    prev_attr = attr

            # Set absent batches beyond those used
            for batch_idx in range(len(machine_batches), self.max_nb_batch):
                warm_start.set_absent(
                    self.variables["intervals_per_machines"][m][batch_idx]
                )

                # Set absent ALL attribute intervals for unused batches
                for attr in self.variables["intervals_per_machines_per_attr"][m]:
                    warm_start.set_absent(
                        self.variables["intervals_per_machines_per_attr"][m][attr][
                            batch_idx
                        ]
                    )

                # Set absent setup cost for unused batches
                if "setup_cost_per_machine" in self.variables:
                    if isinstance(self.variables["setup_cost_per_machine"][m], list):
                        warm_start.set_absent(
                            self.variables["setup_cost_per_machine"][m][batch_idx]
                        )

        # Set job assignments - ALL jobs on ALL machines (present/absent)
        for task in range(self.problem.n_jobs):
            task_data = self.problem.tasks_data[task]

            # Find which machine this task is assigned to
            assigned_machine = None
            assigned_start = None
            assigned_end = None
            assigned_batch_idx = None

            for m in solution.schedule_per_machine:
                for batch_idx, sched_info in enumerate(
                    solution.schedule_per_machine[m]
                ):
                    if task in sched_info.tasks:
                        assigned_machine = m
                        assigned_start = sched_info.start_time
                        assigned_end = sched_info.end_time
                        assigned_batch_idx = batch_idx
                        break
                if assigned_machine is not None:
                    break

            # Set main job interval (always present)
            warm_start.set_value(
                self.variables["interval_job"][task], assigned_start, assigned_end
            )

            # Set job intervals on all eligible machines
            for m in self.variables["interval_job_per_machine"][task]:
                if m == assigned_machine:
                    # Present on assigned machine
                    warm_start.set_value(
                        self.variables["interval_job_per_machine"][task][m],
                        assigned_start,
                        assigned_end,
                    )
                else:
                    # Absent on other machines
                    warm_start.set_absent(
                        self.variables["interval_job_per_machine"][task][m]
                    )

            # Set batch assignment intervals on all eligible machines
            for m in self.variables["interval_batch_machine_for_job"][task]:
                if m == assigned_machine and assigned_batch_idx < self.max_nb_batch:
                    # Present on assigned machine
                    warm_start.set_value(
                        self.variables["interval_batch_machine_for_job"][task][m],
                        assigned_batch_idx,
                        assigned_batch_idx + 1,
                    )
                else:
                    # Absent on other machines
                    warm_start.set_absent(
                        self.variables["interval_batch_machine_for_job"][task][m]
                    )

        # Set A87RandomOvenSchedulingInstance-n250-k2-a5--2212-22.47.15.daLL late job variables (must set all, not just those with constraints)
        if "late_jobs" in self.variables:
            for task in self.variables["late_jobs"]:
                task_data = self.problem.tasks_data[task]
                # Find task's end time in solution
                task_end = None
                for m in solution.schedule_per_machine:
                    for sched_info in solution.schedule_per_machine[m]:
                        if task in sched_info.tasks:
                            task_end = sched_info.end_time
                            break
                    if task_end is not None:
                        break

                if task_end is not None:
                    is_late = 1 if task_end > task_data.latest_end else 0
                    warm_start.set_value(self.variables["late_jobs"][task], is_late)

        # Set batch count variables
        if "nb_batches_per_machine" in self.variables:
            for m in range(self.problem.n_machines):
                nb_batches = len(solution.schedule_per_machine.get(m, []))
                warm_start.set_value(
                    self.variables["nb_batches_per_machine"][m], nb_batches
                )

        # Set makespan if it exists as a variable
        if (
            "objectives" in self.variables
            and "makespan" in self.variables["objectives"]
        ):
            # Find maximum end time across all batches
            max_end = 0
            for m in solution.schedule_per_machine:
                for sched_info in solution.schedule_per_machine[m]:
                    max_end = max(max_end, sched_info.end_time)
            warm_start.set_value(self.variables["objectives"]["makespan"], max_end)

        # Set objective value hint
        objective_value = self.aggreg_from_sol(solution)
        warm_start.set_objective(int(objective_value))

        # Store warm start solution
        self.warm_start_solution = warm_start
        self.use_warm_start = True

        logger.info(f"Warm start set with objective value: {objective_value}")

    def set_warm_start_from_previous_run(self):
        """Set warm start from the previous solve run."""
        if self.current_solution is not None:
            self.warm_start_solution = self.current_solution
            self.use_warm_start = True
            logger.info("Warm start set from previous run")
