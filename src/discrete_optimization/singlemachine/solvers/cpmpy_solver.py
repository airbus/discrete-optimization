#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List

import cpmpy
from cpmpy.expressions.globalconstraints import Cumulative, NoOverlap

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


class SingleMachineModel(Enum):
    """Enum to select the modeling formulation."""

    CP = 0
    LP = 1


class CpmpySingleMachineSolver(CpmpySolver):
    """
    CPMPy-based solver for the Single Machine Weighted Tardiness Problem.

    This solver can use two different formulations:
    - **CP (Constraint Programming):** A model based on temporal variables and disjunctive constraints. It's generally more efficient for scheduling problems.
    - **LP (Mixed-Integer Linear Programming):** A model based on relative ordering variables and "big-M" constraints.
    """

    def __init__(
        self,
        problem: WeightedTardinessProblem,
        **kwargs: Any,
    ):
        """
        Args:
            problem: The WeightedTardinessProblem instance to solve.
            model_type: The formulation to use (CP or LP). Defaults to CP.
        """
        super().__init__(problem=problem, **kwargs)
        self.problem: WeightedTardinessProblem = problem
        self.model_type: SingleMachineModel = None
        self.variables: Dict[str, Any] = {}
        self.meta_constraints: Dict[str, MetaCpmpyConstraint] = {}

    def init_model(
        self,
        model_type: SingleMachineModel = SingleMachineModel.CP,
        add_impossible_constraints: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Builds the CPMpy model based on the chosen formalism (CP or LP).
        It defines variables, constraints, meta-constraints, and the objective function.
        """
        self.model_type = model_type
        if self.model_type == SingleMachineModel.CP:
            self._init_cp_model()
        elif self.model_type == SingleMachineModel.LP:
            self._init_lp_model()
        else:
            raise ValueError("Unknown model type specified.")

        if add_impossible_constraints:
            if self.problem.num_jobs < 2:
                print("Warning: Need at least 2 jobs to add the impossible constraint.")
            else:
                # Impossible Deadline: The first two jobs must finish before the sum of their processing times.
                # This directly conflicts with the NoOverlap constraint.
                deadline = (
                    self.problem.processing_times[0]
                    + self.problem.processing_times[1]
                    - 1
                )
                impossible_constraints = []
                if self.model_type == SingleMachineModel.CP:
                    impossible_constraints.extend(
                        [
                            self.variables["ends"][0] <= deadline,
                            self.variables["ends"][1] <= deadline,
                        ]
                    )
                else:  # LP model
                    impossible_constraints.extend(
                        [
                            self.variables["completion_times"][0] <= deadline,
                            self.variables["completion_times"][1] <= deadline,
                        ]
                    )

                self.meta_constraints["impossible_deadline"] = MetaCpmpyConstraint(
                    name="impossible_deadline",
                    constraints=impossible_constraints,
                    metadata={
                        "description": f"deadline of {deadline} for task 0 and 1"
                    },
                )
                self.model += impossible_constraints

    def _init_cp_model(self) -> None:
        """Initializes the Constraint Programming (CP) model."""
        self.model = cpmpy.Model(minimize=True)

        num_jobs = self.problem.num_jobs
        p = self.problem.processing_times
        d = self.problem.due_dates
        r = self.problem.release_dates
        horizon = sum(p) + max(r) if num_jobs > 0 else 0
        # === Define Decision Variables ===
        starts = cpmpy.intvar(lb=0, ub=horizon, shape=num_jobs, name="starts")
        bound_constraints = []
        for i in range(self.problem.num_jobs):
            bound_constraints.append(starts[i] >= r[i])
        self.meta_constraints["release"] = MetaCpmpyConstraint(
            name="release", constraints=bound_constraints
        )
        self.model += bound_constraints
        ends = starts + p
        tardiness = cpmpy.intvar(lb=0, ub=horizon, shape=num_jobs, name="tardiness")
        self.variables = {"starts": starts, "ends": ends, "tardiness": tardiness}

        # === Define Meta-Constraints for Explanation ===
        # 1. No-Overlap Constraints: A machine can only process one job at a time.
        no_overlap_constraints = [
            NoOverlap(start=starts, dur=self.problem.processing_times, end=ends)
        ]
        self.meta_constraints["no_overlap"] = MetaCpmpyConstraint(
            name="no_overlap", constraints=no_overlap_constraints
        )

        self.model += no_overlap_constraints

        # 2. Tardiness Definition: Tardiness is the time a job is completed after its due date.
        tardiness_def_constraints = [
            tardiness[i] >= ends[i] - d[i] for i in range(num_jobs)
        ]
        self.meta_constraints["tardiness_def"] = MetaCpmpyConstraint(
            name="tardiness_def", constraints=tardiness_def_constraints
        )
        self.model += tardiness_def_constraints
        self.model.minimize(
            cpmpy.sum(
                [
                    self.problem.weights[i] * tardiness[i]
                    for i in range(self.problem.num_jobs)
                ]
            )
        )

    def _init_lp_model(self) -> None:
        """Initializes the Mixed-Integer Linear Programming (LP) model."""
        self.model = cpmpy.Model(minimize=True)
        num_jobs = self.problem.num_jobs
        p = self.problem.processing_times
        d = self.problem.due_dates
        r = self.problem.release_dates
        horizon = sum(p) + max(r) if num_jobs > 0 else 0
        big_m = horizon

        # === Define Decision Variables ===
        # x[i, j] = 1 if job i is scheduled before job j, 0 otherwise.
        x = cpmpy.boolvar(shape=(num_jobs, num_jobs), name="x")

        # C_i = completion time of job i.
        completion_times = cpmpy.intvar(
            lb=0, ub=horizon, shape=num_jobs, name="completion_times"
        )
        release_date_constraints = []
        for j in range(self.problem.num_jobs):
            release_date_constraints.append(
                completion_times[j]
                >= self.problem.release_dates[j] + self.problem.processing_times[j]
            )
        self.meta_constraints["release"] = MetaCpmpyConstraint(
            name="release", constraints=release_date_constraints
        )
        self.model += release_date_constraints
        # T_i = tardiness of job i.
        tardiness = cpmpy.intvar(lb=0, ub=horizon, shape=num_jobs, name="tardiness")
        self.variables = {
            "x": x,
            "completion_times": completion_times,
            "tardiness": tardiness,
        }

        # === Define Meta-Constraints for Explanation ===
        # 1. Ordering Constraints: For any two jobs, one must precede the other.
        ordering_constraints = [
            x[i, j] + x[j, i] == 1
            for i in range(num_jobs)
            for j in range(i + 1, num_jobs)
        ]
        self.meta_constraints["ordering"] = MetaCpmpyConstraint(
            name="ordering", constraints=ordering_constraints
        )
        self.model += ordering_constraints

        # 2. Completion Time Constraints (using big-M for non-overlap)
        completion_time_constraints = []
        for i in range(num_jobs):
            completion_time_i = []
            for j in range(num_jobs):
                if i == j:
                    continue
                # If x[i, j] is 1, then C_j must be at least C_i + p_j.
                completion_time_i.append(
                    completion_times[j]
                    >= completion_times[i] + p[j] - big_m * (1 - x[i, j])
                )
                completion_time_constraints.append(completion_time_i[-1])
            self.meta_constraints[f"completion_time_{i}"] = MetaCpmpyConstraint(
                name=f"completion_time_task_{i}", constraints=completion_time_i
            )
        self.meta_constraints["completion_time"] = MetaCpmpyConstraint(
            name="completion_time", constraints=completion_time_constraints
        )
        self.model += completion_time_constraints
        # 3. Tardiness Definition: T_i >= C_i - d_i
        tardiness_def_constraints = [
            tardiness[i] >= completion_times[i] - d[i] for i in range(num_jobs)
        ]
        self.meta_constraints["tardiness_def"] = MetaCpmpyConstraint(
            name="tardiness_def", constraints=tardiness_def_constraints
        )
        self.model += tardiness_def_constraints

        transitivity_constraints = [
            x[i, j] + x[j, k] + x[k, i] <= 2
            for i in range(num_jobs)
            for j in range(num_jobs)
            if i != j
            for k in range(num_jobs)
            if k not in {i, j}
        ]
        self.meta_constraints["transitivity"] = MetaCpmpyConstraint(
            name="transitivity", constraints=transitivity_constraints
        )
        self.model += transitivity_constraints
        self.model.minimize(
            cpmpy.sum(
                [
                    self.problem.weights[i] * tardiness[i]
                    for i in range(self.problem.num_jobs)
                ]
            )
        )

    def retrieve_current_solution(self) -> Solution:
        """Constructs a WTSolution from the solved CPMpy model."""
        schedule = [() for _ in range(self.problem.num_jobs)]

        if self.model_type == SingleMachineModel.CP:
            starts_val = self.variables["starts"].value()
            ends_val = self.variables["ends"].value()
            for i in range(self.problem.num_jobs):
                schedule[i] = (int(starts_val[i]), int(ends_val[i]))
        else:  # LP model
            completion_times_val = self.variables["completion_times"].value()
            for i in range(self.problem.num_jobs):
                end_time = int(completion_times_val[i])
                start_time = end_time - self.problem.processing_times[i]
                schedule[i] = (start_time, end_time)

        return WTSolution(problem=self.problem, schedule=schedule)

    def get_soft_meta_constraints(self) -> List[MetaCpmpyConstraint]:
        """
        Returns the list of soft meta-constraints for explanation purposes.
        These are constraints that can potentially be relaxed if the model is UNSAT.
        """
        if self.model is None:
            self.init_model()

        soft_constraints = []
        if self.model_type == SingleMachineModel.CP:
            soft_constraints.append(self.meta_constraints["no_overlap"])
        else:  # LP
            soft_constraints.append(self.meta_constraints["completion_time"])

        # The impossible constraint is also soft, as we might want to relax it.
        if "impossible_deadline" in self.meta_constraints:
            soft_constraints.append(self.meta_constraints["impossible_deadline"])

        return soft_constraints

    def get_hard_meta_constraints(self) -> List[MetaCpmpyConstraint]:
        """
        Returns the list of hard meta-constraints.
        These constraints define the core logic of the problem and should not be relaxed.
        """
        if self.model is None:
            self.init_model()

        if self.model_type == SingleMachineModel.CP:
            return [
                self.meta_constraints["tardiness_def"],
                self.meta_constraints["release"],
            ]
        else:  # LP
            return [
                self.meta_constraints["ordering"],
                self.meta_constraints["tardiness_def"],
                self.meta_constraints["transitivity"],
                self.meta_constraints["release"],
            ]
