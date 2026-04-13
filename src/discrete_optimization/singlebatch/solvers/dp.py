import math
from typing import Any

import didppy as dp

from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.singlebatch.problem import (
    BatchProcessingSolution,
    SingleBatchProcessingProblem,
)


class DpSingleBatchSolver(DpSolver):
    """Optimized Dynamic Programming Solver for Single Batch-Processing Machine using didppy."""

    problem: SingleBatchProcessingProblem
    sorted_indices: list[int]

    def init_model(self, **kwargs: Any) -> None:
        self.model = dp.Model()
        problem = self.problem
        n = problem.nb_jobs
        cap = problem.capacity

        # --- KEY OPTIMIZATION: PRE-SORTING ---
        # Sort jobs by processing time descending.
        # This guarantees the first job in any batch is the longest, so we pay the
        # cost upfront and subsequent jobs packed into that batch cost 0.
        self.sorted_indices = sorted(
            range(n), key=lambda i: problem.jobs[i].processing_time, reverse=True
        )
        job_sizes = [problem.jobs[i].size for i in self.sorted_indices]
        job_ps = [problem.jobs[i].processing_time for i in self.sorted_indices]

        job = self.model.add_object_type(n)
        job_and_dummy = self.model.add_object_type(n + 1)

        # --- State Variables ---
        # 1. Unassigned jobs (indices now refer to the sorted lists)
        unassigned = self.model.add_set_var(target=list(range(n)), object_type=job)

        # 2. Remaining capacity in the currently open batch
        rem_cap = self.model.add_int_resource_var(target=0, less_is_better=False)
        # current_batch_duration = self.model.add_int_resource_var(target=0,
        #                                                         less_is_better=True)
        # 3. Last added job index in the current batch (to break permutation symmetries)
        # target=n acts as our dummy state representing "no job added yet"
        last_added = self.model.add_element_var(target=n, object_type=job_and_dummy)
        job_sizes_table = self.model.add_int_table(job_sizes)
        # --- Base Case ---
        self.model.add_base_case([unassigned.is_empty()])
        last_added_lower_than_j = [
            self.model.add_bool_state_fun(last_added < j) for j in range(n)
        ]
        all_done_before = []
        for j in range(n):
            is_min_cond = unassigned.contains(j)
            for i in range(j):
                is_min_cond = is_min_cond & ~unassigned.contains(i)
            all_done_before.append(self.model.add_bool_state_fun(is_min_cond))
        min_sizes_unassigned = self.model.add_int_state_fun(
            job_sizes_table.min(unassigned)
        )
        # --- Transitions ---
        for j in range(n):
            # Transition A: Open a NEW batch with job j
            # Cost is incurred immediately because j is the longest job in this new batch
            t_open = dp.Transition(
                name=f"open_new_{j}",
                cost=dp.IntExpr.state_cost() + job_ps[j],
                effects=[
                    (unassigned, unassigned.remove(j)),
                    (rem_cap, cap - job_sizes[j]),
                    (last_added, j),
                    # (current_batch_duration, job_ps[j])
                ],
                preconditions=[
                    all_done_before[j],
                    unassigned.contains(j),
                    min_sizes_unassigned > rem_cap,
                ],
            )
            self.model.add_transition(t_open)
            # Transition B: Pack job j into the EXISTING open batch
            # Cost is 0 because the batch's cost is dictated by its first job (which is >= job_ps[j])
            # new_batch_duration = dp.max(current_batch_duration, job_ps[j])
            t_pack = dp.Transition(
                name=f"pack_{j}",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (unassigned, unassigned.remove(j)),
                    (rem_cap, rem_cap - job_sizes[j]),
                    (last_added, j),
                ],
                preconditions=[
                    last_added_lower_than_j[j],
                    unassigned.contains(j),
                    rem_cap >= job_sizes[j],
                ],
            )
            self.model.add_transition(t_pack)

        min_p = job_ps[-1]  # The shortest processing time among all jobs

        # 1. L1 Bound: Volume / Capacity
        weight_table = self.model.add_int_table(job_sizes)
        self.model.add_dual_bound(
            math.ceil((weight_table[unassigned] - rem_cap) / cap) * min_p
        )

        # 2. L2 Bound (Martello and Toth)
        weight_2_1 = [1 if job_sizes[i] > cap / 2 else 0 for i in range(n)]
        weight_2_2 = [0.5 if job_sizes[i] == cap / 2 else 0 for i in range(n)]

        weight_2_1_table = self.model.add_int_table(weight_2_1)
        weight_2_2_table = self.model.add_float_table(weight_2_2)

        self.model.add_dual_bound(
            (
                weight_2_1_table[unassigned]
                + math.ceil(weight_2_2_table[unassigned])
                - (rem_cap >= cap / 2).if_then_else(1, 0)
            )
            * min_p
        )

        # 3. L3 Bound (Finer grained capacities)
        weight_3 = [
            1.0
            if job_sizes[i] > cap * 2 / 3
            else 2 / 3
            if job_sizes[i] == cap * 2 / 3
            else 0.5
            if job_sizes[i] > cap / 3
            else 1 / 3
            if job_sizes[i] == cap / 3
            else 0.0
            for i in range(n)
        ]

        weight_3_table = self.model.add_float_table(weight_3)

        self.model.add_dual_bound(
            (
                math.ceil(weight_3_table[unassigned])
                - (rem_cap >= cap / 3).if_then_else(1, 0)
            )
            * min_p
        )

    def retrieve_solution(self, sol: dp.Solution) -> BatchProcessingSolution:
        """Parse the didppy state transition trace into a valid DO Solution."""
        job_to_batch = [-1] * self.problem.nb_jobs
        batch_id = -1

        for t in sol.transitions:
            name = t.name
            if name.startswith("open_new_"):
                batch_id += 1
                # Extract the sorted index and map it back to the original problem index
                sorted_j = int(name.split("_")[-1])
                original_j = self.sorted_indices[sorted_j]
                job_to_batch[original_j] = batch_id

            elif name.startswith("pack_"):
                sorted_j = int(name.split("_")[-1])
                original_j = self.sorted_indices[sorted_j]
                job_to_batch[original_j] = batch_id

        return BatchProcessingSolution(problem=self.problem, job_to_batch=job_to_batch)
