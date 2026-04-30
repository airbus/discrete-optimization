#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.ovensched.problem import (
    Machine,
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
    Task,
    UnaryResource,
)
from discrete_optimization.ovensched.solution_vector import VectorOvenSchedulingSolution

logger = logging.getLogger(__name__)


class OvenSchedulingCpSatSolver(
    SchedulingCpSatSolver[Task],
    AllocationCpSatSolver[Task, UnaryResource],
    WarmstartMixin,
):
    """CP-SAT solver for the Oven Scheduling Problem.

    This solver models:
    - Tasks that must be scheduled
    - Machines with availability windows and initial states
    - Batching: tasks with the same attribute can be batched together
    - Batch capacity constraints
    - Setup times and costs between different task attributes
    """

    problem: OvenSchedulingProblem

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        """Initialize the solver.

        Args:
            problem: The oven scheduling problem instance
            params_objective_function: Parameters for objective function (weights, handling, etc.)
            **kwargs: Additional arguments passed to parent constructors
        """
        super().__init__(problem, params_objective_function, **kwargs)
        self.max_nb_batch = None  # Will be set in init_model

        # Index tasks by eligible machines
        self.tasks_per_machine = {
            m: [
                t
                for t in range(self.problem.n_jobs)
                if m in self.problem.tasks_data[t].eligible_machines
            ]
            for m in range(self.problem.n_machines)
        }

        # Variables storage
        self.variables = {}

    def init_model(
        self, max_nb_batch_per_machine: int | None = None, **kwargs: Any
    ) -> None:
        """Initialize the CP-SAT model.

        Args:
            max_nb_batch_per_machine: Maximum number of batches per machine.
                If None, estimated as 4 * n_jobs / n_machines
            **kwargs: Additional arguments passed to parent init_model
        """
        super().init_model(**kwargs)
        self.variables = {}

        # Set max_nb_batch based on parameter or default heuristic
        if max_nb_batch_per_machine is not None:
            self.max_nb_batch = max_nb_batch_per_machine
        elif self.max_nb_batch is None:
            # Default heuristic
            self.max_nb_batch = max(
                1, int(4 * self.problem.n_jobs / self.problem.n_machines)
            )

        # Create decision variables
        self._create_batch_variables()
        self._create_task_variables()

        # Add constraints
        self._add_batch_ordering_constraints()
        self._calendar_constraints()
        self._add_task_allocation_constraints()
        self._add_batch_capacity_constraints()
        self._add_setup_constraints()

        # Set objective using params_objective_function
        self._set_objective_from_params()

    def _create_batch_variables(self):
        """Create variables for batches on each machine."""
        # For each machine and batch position, create:
        # - start/end/duration variables
        # - presence variable (is this batch used?)
        # - attribute variables (which task attribute is this batch processing?)

        self.variables["batch_start"] = {}
        self.variables["batch_end"] = {}
        self.variables["batch_duration"] = {}
        self.variables["batch_interval"] = {}
        self.variables["batch_present"] = {}
        self.variables["batch_attribute_is"] = {}

        for m in range(self.problem.n_machines):
            machine_data = self.problem.machines_data[m]
            calendar = machine_data.availability
            # Get possible attributes for this machine
            possible_attributes = set(
                self.problem.tasks_data[t].attribute for t in self.tasks_per_machine[m]
            )
            # Get possible durations
            possible_durations = {0}
            for t in self.tasks_per_machine[m]:
                possible_durations.add(self.problem.tasks_data[t].min_duration)
                possible_durations.add(self.problem.tasks_data[t].max_duration)
            self.variables["batch_start"][m] = []
            self.variables["batch_end"][m] = []
            self.variables["batch_duration"][m] = []
            self.variables["batch_present"][m] = []
            self.variables["batch_interval"][m] = []
            self.variables["batch_attribute_is"][m] = {}
            for attr in possible_attributes:
                self.variables["batch_attribute_is"][m][attr] = []
            for b in range(self.max_nb_batch):
                # Start/end must be in availability windows
                start_var = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromIntervals(calendar),
                    name=f"batch_start_m{m}_b{b}",
                )
                end_var = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromIntervals(calendar),
                    name=f"batch_end_m{m}_b{b}",
                )
                duration_var = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(sorted(possible_durations)),
                    name=f"batch_duration_m{m}_b{b}",
                )
                present_var = self.cp_model.NewBoolVar(name=f"batch_present_m{m}_b{b}")
                self.variables["batch_start"][m].append(start_var)
                self.variables["batch_end"][m].append(end_var)
                self.variables["batch_duration"][m].append(duration_var)
                self.variables["batch_present"][m].append(present_var)
                # Optional interval for this batch
                self.variables["batch_interval"][m].append(
                    self.cp_model.NewOptionalIntervalVar(
                        start=start_var,
                        size=duration_var,
                        end=end_var,
                        is_present=present_var,
                        name=f"batch_interval_m{m}_b{b}",
                    )
                )
                # Attribute indicator variables
                for attr in possible_attributes:
                    attr_var = self.cp_model.NewBoolVar(
                        name=f"batch_attr_m{m}_b{b}_a{attr}"
                    )
                    self.variables["batch_attribute_is"][m][attr].append(attr_var)

    def _create_task_variables(self):
        """Create variables for task scheduling and allocation."""
        # For each task:
        # - which machine is it allocated to?
        # - which batch on that machine?
        # - what are its start/end times?

        self.variables["task_start"] = {}
        self.variables["task_end"] = {}
        self.variables["task_duration"] = {}
        self.variables["task_interval"] = {}
        self.variables["task_on_machine"] = {}
        self.variables["task_on_machine_batch"] = {}

        horizon = max(
            interval[1]
            for machine_data in self.problem.machines_data
            for interval in machine_data.availability
        )

        for t in range(self.problem.n_jobs):
            task_data = self.problem.tasks_data[t]

            # Task timing variables
            start_var = self.cp_model.NewIntVar(
                lb=task_data.earliest_start, ub=horizon, name=f"task_start_{t}"
            )
            end_var = self.cp_model.NewIntVar(
                lb=task_data.earliest_start + task_data.min_duration,
                ub=horizon,
                name=f"task_end_{t}",
            )
            duration_var = self.cp_model.NewIntVar(
                lb=task_data.min_duration,
                ub=task_data.max_duration,
                name=f"task_duration_{t}",
            )

            self.variables["task_start"][t] = start_var
            self.variables["task_end"][t] = end_var
            self.variables["task_duration"][t] = duration_var

            # Link start + duration = end
            interval_var = self.cp_model.NewIntervalVar(
                start=start_var,
                size=duration_var,
                end=end_var,
                name=f"task_interval_m{t}",
            )
            self.variables["task_interval"][t] = interval_var
            # Machine allocation variables
            self.variables["task_on_machine"][t] = {}
            self.variables["task_on_machine_batch"][t] = {}

            for m in task_data.eligible_machines:
                # Binary: is task t on machine m?
                machine_var = self.cp_model.NewBoolVar(name=f"task_{t}_on_machine_{m}")
                self.variables["task_on_machine"][t][m] = machine_var
                # For each batch position on machine m
                self.variables["task_on_machine_batch"][t][m] = []
                for b in range(self.max_nb_batch):
                    batch_var = self.cp_model.NewBoolVar(name=f"task_{t}_on_m{m}_b{b}")
                    self.variables["task_on_machine_batch"][t][m].append(batch_var)

    def _add_batch_ordering_constraints(self):
        """Add constraints on batch ordering and presence."""
        for m in range(self.problem.n_machines):
            # Batches must be used in order (no gaps)
            for b in range(self.max_nb_batch - 1):
                self.cp_model.Add(
                    self.variables["batch_present"][m][b]
                    >= self.variables["batch_present"][m][b + 1]
                )

                # If batch b+1 is present, batch b+1 must start after batch b ends
                self.cp_model.Add(
                    self.variables["batch_start"][m][b + 1]
                    >= self.variables["batch_end"][m][b]
                ).OnlyEnforceIf(self.variables["batch_present"][m][b + 1])

            # Each present batch must have exactly one attribute
            possible_attributes = set(
                self.problem.tasks_data[t].attribute for t in self.tasks_per_machine[m]
            )
            for b in range(self.max_nb_batch):
                self.cp_model.AddAtMostOne(
                    [
                        self.variables["batch_attribute_is"][m][attr][b]
                        for attr in possible_attributes
                    ]
                )
                self.cp_model.Add(
                    sum(
                        self.variables["batch_attribute_is"][m][attr][b]
                        for attr in possible_attributes
                    )
                    == 1
                ).OnlyEnforceIf(self.variables["batch_present"][m][b])

    def _calendar_constraints(self):
        for m in range(self.problem.n_machines):
            availability = self.problem.machines_data[m].availability
            virtual_itv = [
                self.cp_model.new_fixed_size_interval_var(
                    start=av[1], size=1, name=f"avail_{m}_{av[1]}"
                )
                for av in availability
            ]
            self.cp_model.add_no_overlap(
                self.variables["batch_interval"][m] + virtual_itv
            )

    def _add_task_allocation_constraints(self):
        """Add constraints linking tasks to batches."""
        for t in range(self.problem.n_jobs):
            task_data = self.problem.tasks_data[t]

            # Each task must be on exactly one machine
            self.cp_model.AddExactlyOne(
                [
                    self.variables["task_on_machine"][t][m]
                    for m in task_data.eligible_machines
                ]
            )

            # Each task must be in exactly one batch
            for m in task_data.eligible_machines:
                # If task is on machine m, it must be in exactly one batch on m
                self.cp_model.Add(
                    sum(self.variables["task_on_machine_batch"][t][m]) == 1
                ).OnlyEnforceIf(self.variables["task_on_machine"][t][m])

                # If task is not on machine m, it's in no batch on m
                self.cp_model.Add(
                    sum(self.variables["task_on_machine_batch"][t][m]) == 0
                ).OnlyEnforceIf(self.variables["task_on_machine"][t][m].Not())

                # Link task-in-batch to batch presence and attributes
                for b in range(self.max_nb_batch):
                    task_in_batch = self.variables["task_on_machine_batch"][t][m][b]

                    # If task t is in batch b on machine m:
                    # 1. Batch b must be present
                    self.cp_model.Add(
                        self.variables["batch_present"][m][b] == 1
                    ).OnlyEnforceIf(task_in_batch)

                    # 2. Batch attribute must match task attribute
                    self.cp_model.Add(
                        self.variables["batch_attribute_is"][m][task_data.attribute][b]
                        == 1
                    ).OnlyEnforceIf(task_in_batch)

                    # 3. Task timing = batch timing
                    self.cp_model.Add(
                        self.variables["task_start"][t]
                        == self.variables["batch_start"][m][b]
                    ).OnlyEnforceIf(task_in_batch)
                    self.cp_model.Add(
                        self.variables["task_end"][t]
                        == self.variables["batch_end"][m][b]
                    ).OnlyEnforceIf(task_in_batch)
                    self.cp_model.Add(
                        self.variables["task_duration"][t]
                        == self.variables["batch_duration"][m][b]
                    ).OnlyEnforceIf(task_in_batch)

            # Each present batch must have at least one task
            for m in range(self.problem.n_machines):
                for b in range(self.max_nb_batch):
                    tasks_in_batch = [
                        self.variables["task_on_machine_batch"][t][m][b]
                        for t in self.tasks_per_machine[m]
                    ]
                    if tasks_in_batch:
                        self.cp_model.Add(sum(tasks_in_batch) >= 1).OnlyEnforceIf(
                            self.variables["batch_present"][m][b]
                        )
                        self.cp_model.Add(sum(tasks_in_batch) == 0).OnlyEnforceIf(
                            self.variables["batch_present"][m][b].Not()
                        )

    def _add_batch_capacity_constraints(self):
        """Add capacity constraints for each batch."""
        for m in range(self.problem.n_machines):
            capacity = self.problem.machines_data[m].capacity

            for b in range(self.max_nb_batch):
                # Sum of sizes of tasks in this batch <= capacity
                tasks_sizes = [
                    (
                        self.variables["task_on_machine_batch"][t][m][b],
                        self.problem.tasks_data[t].size,
                    )
                    for t in self.tasks_per_machine[m]
                ]

                if tasks_sizes:
                    self.cp_model.add(
                        LinearExpr.weighted_sum(
                            [x[0] for x in tasks_sizes], [x[1] for x in tasks_sizes]
                        )
                        <= capacity
                    )

    def _add_setup_constraints(self):
        """Add setup time and cost constraints."""
        self.variables["setup_time"] = {}
        self.variables["setup_cost"] = {}

        # Get all possible setup values
        all_setup_times = {0}
        all_setup_costs = {0}
        for row in self.problem.setup_times:
            all_setup_times.update(row)
        for row in self.problem.setup_costs:
            all_setup_costs.update(row)

        for m in range(self.problem.n_machines):
            machine_data = self.problem.machines_data[m]
            possible_attributes = set(
                self.problem.tasks_data[t].attribute for t in self.tasks_per_machine[m]
            )

            self.variables["setup_time"][m] = []
            self.variables["setup_cost"][m] = []

            for b in range(self.max_nb_batch):
                # Setup time/cost variables
                setup_time_var = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(sorted(all_setup_times)),
                    name=f"setup_time_m{m}_b{b}",
                )
                setup_cost_var = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(sorted(all_setup_costs)),
                    name=f"setup_cost_m{m}_b{b}",
                )

                self.variables["setup_time"][m].append(setup_time_var)
                self.variables["setup_cost"][m].append(setup_cost_var)

                if b == 0:
                    # Setup from initial state
                    prev_attr = machine_data.initial_attribute
                    for curr_attr in possible_attributes:
                        is_curr_attr = self.variables["batch_attribute_is"][m][
                            curr_attr
                        ][b]
                        setup_time = self.problem.setup_times[prev_attr][curr_attr]
                        setup_cost = self.problem.setup_costs[prev_attr][curr_attr]

                        self.cp_model.Add(setup_time_var == setup_time).OnlyEnforceIf(
                            [is_curr_attr, self.variables["batch_present"][m][b]]
                        )
                        self.cp_model.Add(setup_cost_var == setup_cost).OnlyEnforceIf(
                            [is_curr_attr, self.variables["batch_present"][m][b]]
                        )

                        # Start time must account for setup
                        self.cp_model.Add(
                            self.variables["batch_start"][m][b] >= setup_time
                        ).OnlyEnforceIf(
                            [is_curr_attr, self.variables["batch_present"][m][b]]
                        )
                else:
                    # Setup from previous batch
                    for prev_attr in possible_attributes:
                        for curr_attr in possible_attributes:
                            is_prev_attr = self.variables["batch_attribute_is"][m][
                                prev_attr
                            ][b - 1]
                            is_curr_attr = self.variables["batch_attribute_is"][m][
                                curr_attr
                            ][b]
                            setup_time = self.problem.setup_times[prev_attr][curr_attr]
                            setup_cost = self.problem.setup_costs[prev_attr][curr_attr]

                            both_present = [
                                self.variables["batch_present"][m][b - 1],
                                self.variables["batch_present"][m][b],
                                is_prev_attr,
                                is_curr_attr,
                            ]

                            self.cp_model.Add(
                                setup_time_var == setup_time
                            ).OnlyEnforceIf(both_present)
                            self.cp_model.Add(
                                setup_cost_var == setup_cost
                            ).OnlyEnforceIf(both_present)

                            # Start time must be after previous end + setup
                            self.cp_model.Add(
                                self.variables["batch_start"][m][b]
                                >= self.variables["batch_end"][m][b - 1] + setup_time
                            ).OnlyEnforceIf(both_present)

                # If batch not present, setup is 0
                self.cp_model.Add(setup_time_var == 0).OnlyEnforceIf(
                    self.variables["batch_present"][m][b].Not()
                )
                self.cp_model.Add(setup_cost_var == 0).OnlyEnforceIf(
                    self.variables["batch_present"][m][b].Not()
                )

    def _set_objective_from_params(self):
        """Set the objective function using params_objective_function."""
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

        # Processing time: sum of all batch durations
        if "processing_time" in weights_dict and weights_dict["processing_time"] > 0:
            total_processing = sum(
                self.variables["batch_duration"][m][b]
                for m in range(self.problem.n_machines)
                for b in range(self.max_nb_batch)
            )
            self.variables["objectives"]["processing_time"] = total_processing
            objective_terms.append(
                weights_dict["processing_time"]
                * self.variables["objectives"]["processing_time"]
            )

        # Tardiness: number of jobs finishing after their deadline
        if "nb_late_jobs" in weights_dict and weights_dict["nb_late_jobs"] > 0:
            late_jobs = []
            self.variables["late_tasks"] = {}
            for t in range(self.problem.n_jobs):
                task_data = self.problem.tasks_data[t]
                if task_data.latest_end < float("inf"):
                    is_late = self.cp_model.NewBoolVar(name=f"task_{t}_late")
                    self.cp_model.Add(
                        self.variables["task_end"][t] > task_data.latest_end
                    ).OnlyEnforceIf(is_late)
                    self.cp_model.Add(
                        self.variables["task_end"][t] <= task_data.latest_end
                    ).OnlyEnforceIf(is_late.Not())
                    self.variables["late_tasks"][t] = is_late
                    late_jobs.append(is_late)

            if late_jobs:
                self.variables["objectives"]["nb_late_jobs"] = LinearExpr.sum(late_jobs)
                objective_terms.append(
                    weights_dict["nb_late_jobs"]
                    * self.variables["objectives"]["nb_late_jobs"]
                )

        # Setup cost: sum of all setup costs
        if "setup_cost" in weights_dict and weights_dict["setup_cost"] > 0:
            total_setup_cost = LinearExpr.sum(
                [
                    self.variables["setup_cost"][m][b]
                    for m in range(self.problem.n_machines)
                    for b in range(self.max_nb_batch)
                ]
            )
            self.variables["objectives"]["setup_cost"] = total_setup_cost
            objective_terms.append(
                int(weights_dict["setup_cost"])
                * self.variables["objectives"]["setup_cost"]
            )

        if objective_terms:
            self.cp_model.Minimize(sum(objective_terms))

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> OvenSchedulingSolution:
        """Construct a solution from the CP-SAT solver's internal state.

        Args:
            cpsolvercb: The OR-Tools callback providing variable values

        Returns:
            An OvenSchedulingSolution
        """
        schedule_per_machine: dict[Machine, list[ScheduleInfo]] = {
            m: [] for m in range(self.problem.n_machines)
        }

        # Extract solution for each machine
        for m in range(self.problem.n_machines):
            for b in range(self.max_nb_batch):
                if cpsolvercb.Value(self.variables["batch_present"][m][b]) == 1:
                    # Find which tasks are in this batch
                    tasks_in_batch = set()
                    for t in self.tasks_per_machine[m]:
                        if (
                            cpsolvercb.Value(
                                self.variables["task_on_machine_batch"][t][m][b]
                            )
                            == 1
                        ):
                            tasks_in_batch.add(t)

                    if tasks_in_batch:
                        # Determine the attribute of this batch
                        task_attribute = self.problem.tasks_data[
                            next(iter(tasks_in_batch))
                        ].attribute

                        start_time = cpsolvercb.Value(
                            self.variables["batch_start"][m][b]
                        )
                        end_time = cpsolvercb.Value(self.variables["batch_end"][m][b])

                        schedule_info = ScheduleInfo(
                            tasks=tasks_in_batch,
                            task_attribute=task_attribute,
                            start_time=start_time,
                            end_time=end_time,
                            machine_batch_index=(m, b),
                        )
                        schedule_per_machine[m].append(schedule_info)

        solution = OvenSchedulingSolution(
            problem=self.problem, schedule_per_machine=schedule_per_machine
        )
        logger.info(
            f"{cpsolvercb.ObjectiveValue() / self.problem.additional_data.get('ub', 1)}"
        )
        return solution

    # Implement abstract methods from SchedulingCpSatSolver
    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        """Retrieve the variable storing the start or end time of given task."""
        if start_or_end == StartOrEnd.START:
            return self.variables["task_start"][task]
        else:
            return self.variables["task_end"][task]

    # Implement abstract methods from AllocationCpSatSolver
    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        """Return a 0-1 variable telling if the unary_resource is used for the task."""
        machine, batch_idx = unary_resource
        if machine in self.variables["task_on_machine_batch"][task]:
            if batch_idx < len(self.variables["task_on_machine_batch"][task][machine]):
                return self.variables["task_on_machine_batch"][task][machine][batch_idx]
        return 0  # Not compatible

    def set_warm_start(self, solution: OvenSchedulingSolution) -> None:
        self.cp_model.clear_hints()
        if isinstance(solution, VectorOvenSchedulingSolution):
            sol = solution.to_oven_scheduling_solution()
        else:
            sol = solution
        # Iterate over ALL machines, not just those with batches
        for m in range(self.problem.n_machines):
            # Number of batches on this machine in the solution
            nb_batches = len(sol.schedule_per_machine.get(m, []))

            if nb_batches > 0:
                # Machine has batches in solution - set hints for used batches
                prev_attr = self.problem.machines_data[m].initial_attribute
                for i_batch in range(nb_batches):
                    sched_info = sol.schedule_per_machine[m][i_batch]
                    attr = sched_info.task_attribute
                    start = sched_info.start_time
                    end = sched_info.end_time
                    self.cp_model.add_hint(
                        self.variables["setup_time"][m][i_batch],
                        self.problem.setup_times[prev_attr][attr],
                    )
                    self.cp_model.add_hint(
                        self.variables["setup_cost"][m][i_batch],
                        self.problem.setup_costs[prev_attr][attr],
                    )
                    self.cp_model.add_hint(
                        self.variables["batch_start"][m][i_batch], start
                    )
                    self.cp_model.add_hint(self.variables["batch_end"][m][i_batch], end)
                    self.cp_model.add_hint(
                        self.variables["batch_duration"][m][i_batch], end - start
                    )
                    self.cp_model.add_hint(
                        self.variables["batch_present"][m][i_batch], 1
                    )
                    for attribute in self.variables["batch_attribute_is"][m]:
                        if attribute == attr:
                            self.cp_model.add_hint(
                                self.variables["batch_attribute_is"][m][attribute][
                                    i_batch
                                ],
                                1,
                            )
                        else:
                            self.cp_model.add_hint(
                                self.variables["batch_attribute_is"][m][attribute][
                                    i_batch
                                ],
                                0,
                            )

                    for task in self.problem.tasks_list:
                        if task in sched_info.tasks:
                            self.cp_model.add_hint(
                                self.variables["task_on_machine"][task][m], 1
                            )
                            for mm in self.variables["task_on_machine"][task]:
                                if m != mm:
                                    self.cp_model.add_hint(
                                        self.variables["task_on_machine"][task][mm], 0
                                    )
                            self.cp_model.add_hint(
                                self.variables["task_on_machine_batch"][task][m][
                                    i_batch
                                ],
                                1,
                            )
                        else:
                            if m in self.variables["task_on_machine_batch"][task]:
                                self.cp_model.add_hint(
                                    self.variables["task_on_machine_batch"][task][m][
                                        i_batch
                                    ],
                                    0,
                                )
                    # Update prev_attr for next batch
                    prev_attr = attr
                last_start = sol.schedule_per_machine[m][-1].end_time

            # Set hints for unused batch positions (from nb_batches to max_nb_batch)
            # Use first availability window start to ensure hint is in domain
            safe_time = self.problem.machines_data[m].availability[0][0]
            for i_batch_after in range(nb_batches, self.max_nb_batch):
                self.cp_model.add_hint(
                    self.variables["setup_time"][m][i_batch_after], 0
                )
                self.cp_model.add_hint(
                    self.variables["setup_cost"][m][i_batch_after], 0
                )
                self.cp_model.add_hint(
                    self.variables["batch_start"][m][i_batch_after], safe_time
                )
                self.cp_model.add_hint(
                    self.variables["batch_end"][m][i_batch_after], safe_time
                )
                self.cp_model.add_hint(
                    self.variables["batch_duration"][m][i_batch_after], 0
                )
                self.cp_model.add_hint(
                    self.variables["batch_present"][m][i_batch_after], 0
                )
                for attribute in self.variables["batch_attribute_is"][m]:
                    self.cp_model.add_hint(
                        self.variables["batch_attribute_is"][m][attribute][
                            i_batch_after
                        ],
                        0,
                    )
                # Hint that no tasks are assigned to unused batch positions
                for task in self.tasks_per_machine[m]:
                    if m in self.variables["task_on_machine_batch"][task]:
                        if i_batch_after < len(
                            self.variables["task_on_machine_batch"][task][m]
                        ):
                            self.cp_model.add_hint(
                                self.variables["task_on_machine_batch"][task][m][
                                    i_batch_after
                                ],
                                0,
                            )
        for task in self.problem.tasks_list:
            st = sol.get_start_time(task)
            end = sol.get_end_time(task)
            self.cp_model.add_hint(
                self.get_task_start_or_end_variable(
                    task, start_or_end=StartOrEnd.START
                ),
                st,
            )
            self.cp_model.add_hint(
                self.get_task_start_or_end_variable(task, start_or_end=StartOrEnd.END),
                end,
            )
            self.cp_model.add_hint(self.variables["task_duration"][task], end - st)
            if "late_tasks" in self.variables:
                if task in self.variables["late_tasks"]:
                    if end > self.problem.tasks_data[task].latest_end:
                        self.cp_model.add_hint(self.variables["late_tasks"][task], 1)
                    else:
                        self.cp_model.add_hint(self.variables["late_tasks"][task], 0)
