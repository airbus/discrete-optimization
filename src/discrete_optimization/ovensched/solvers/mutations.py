#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Mutation operators for the Oven Scheduling Problem.

This module provides mutation operators that work with the VectorOvenSchedulingSolution
using the permutation encoding..
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from discrete_optimization.generic_tools.do_mutation import (
    LocalMove,
    LocalMoveDefault,
    Mutation,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.ovensched import OvenSchedulingProblem
from discrete_optimization.ovensched.problem import OvenSchedulingProblem
from discrete_optimization.ovensched.solution_vector import VectorOvenSchedulingSolution


class OvenPermutationSwap(Mutation):
    """
    Mutation operator that swaps two random positions in the permutation.
    This is a simple but effective mutation for permutation-based solutions.
    We dont use common SwapMutation to handle the cache mecanism happening in VectorOvenSchedulingSolution
    """

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Apply swap mutation to a solution.

        Args:
            solution: The solution to mutate (must be VectorOvenSchedulingSolution)

        Returns:
            Tuple of (new solution, local move object)
        """
        solution, move = super().mutate(solution)

        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Solution must be VectorOvenSchedulingSolution, got {type(solution)}"
            )

        n = len(solution.permutation)

        # Choose two random distinct positions
        i, j = random.sample(range(n), 2)

        # Create new permutation with swapped elements
        new_perm = solution.permutation.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        # Create local move (for backtracking if needed)
        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)

        return new_solution, move


class OvenPermutationInsert(Mutation):
    """
    Mutation operator that moves an element from one position to another.
    """

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Apply insert mutation to a solution.

        Args:
            solution: The solution to mutate (must be VectorOvenSchedulingSolution)

        Returns:
            Tuple of (new solution, local move object)
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Solution must be VectorOvenSchedulingSolution, got {type(solution)}"
            )

        n = len(solution.permutation)

        # Choose source and target positions
        i, j = random.sample(range(n), 2)

        # Create new permutation with element moved
        new_perm = solution.permutation.copy()
        element = new_perm[i]

        # Remove from position i
        new_perm = np.delete(new_perm, i)

        # Insert at position j
        new_perm = np.insert(new_perm, j, element)

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        # Create local move
        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)

        return new_solution, move


class OvenPermutationPartialShuffle(Mutation):
    """
    Mutation operator that shuffles a random subsequence of the permutation.
    """

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        min_length: int = 2,
        max_length: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the partial shuffle mutation.

        Args:
            problem: The problem instance
            min_length: Minimum length of subsequence to shuffle
            max_length: Maximum length of subsequence to shuffle (default: n_jobs // 4)
        """
        super().__init__(problem=problem, **kwargs)
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else problem.n_jobs // 4

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Apply partial shuffle mutation to a solution.

        Args:
            solution: The solution to mutate (must be VectorOvenSchedulingSolution)

        Returns:
            Tuple of (new solution, local move object)
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Solution must be VectorOvenSchedulingSolution, got {type(solution)}"
            )

        n = len(solution.permutation)

        # Choose subsequence length
        max_len = min(self.max_length, n)
        subseq_length = random.randint(self.min_length, max_len)

        # Choose starting position
        start = random.randint(0, n - subseq_length)
        end = start + subseq_length

        # Create new permutation with shuffled subsequence
        new_perm = solution.permutation.copy()
        subseq = new_perm[start:end].copy()
        np.random.shuffle(subseq)
        new_perm[start:end] = subseq

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        # Create local move
        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)

        return new_solution, move


class OvenPermutationTwoOpt(Mutation):
    """
    2-opt mutation: reverses a random subsequence of the permutation.

    This is a classic neighborhood operator for permutation problems.
    """

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Apply 2-opt mutation to a solution.

        Args:
            solution: The solution to mutate (must be VectorOvenSchedulingSolution)

        Returns:
            Tuple of (new solution, local move object)
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Solution must be VectorOvenSchedulingSolution, got {type(solution)}"
            )

        n = len(solution.permutation)

        # Choose two distinct positions
        i, j = sorted(random.sample(range(n), 2))

        # Create new permutation with reversed subsequence
        new_perm = solution.permutation.copy()
        new_perm[i : j + 1] = new_perm[i : j + 1][::-1]

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        # Create local move
        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)

        return new_solution, move


class OvenPermutationMixedMutation(Mutation):
    """
    Mixed mutation that randomly applies one of several mutation operators.

    This provides diversity in the search process.
    """

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        swap_prob: float = 0.4,
        insert_prob: float = 0.3,
        two_opt_prob: float = 0.2,
        shuffle_prob: float = 0.1,
        **kwargs: Any,
    ):
        """
        Initialize mixed mutation.

        Args:
            problem: The problem instance
            swap_prob: Probability of applying swap mutation
            insert_prob: Probability of applying insert mutation
            two_opt_prob: Probability of applying 2-opt mutation
            shuffle_prob: Probability of applying partial shuffle mutation
        """
        super().__init__(problem=problem, **kwargs)

        # Normalize probabilities
        total = swap_prob + insert_prob + two_opt_prob + shuffle_prob
        self.swap_prob = swap_prob / total
        self.insert_prob = insert_prob / total
        self.two_opt_prob = two_opt_prob / total
        self.shuffle_prob = shuffle_prob / total

        # Create mutation operators
        self.mutations = {
            "swap": OvenPermutationSwap(problem),
            "insert": OvenPermutationInsert(problem),
            "two_opt": OvenPermutationTwoOpt(problem),
            "shuffle": OvenPermutationPartialShuffle(problem),
        }

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Apply a randomly selected mutation to a solution.

        Args:
            solution: The solution to mutate (must be VectorOvenSchedulingSolution)

        Returns:
            Tuple of (new solution, local move object)
        """
        # Choose mutation based on probabilities
        r = random.random()

        if r < self.swap_prob:
            return self.mutations["swap"].mutate(solution)
        elif r < self.swap_prob + self.insert_prob:
            return self.mutations["insert"].mutate(solution)
        elif r < self.swap_prob + self.insert_prob + self.two_opt_prob:
            return self.mutations["two_opt"].mutate(solution)
        else:
            return self.mutations["shuffle"].mutate(solution)


class SwapBatchesMutation(Mutation):
    """
    Swap Batches (SB) mutation from the paper.

    Swaps two batches on the same machine by reordering their tasks in the permutation.
    This changes the processing order of batches on a machine.
    """

    problem: OvenSchedulingProblem

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Swap two batches on the same machine.

        Strategy:
        1. Decode permutation to identify batches
        2. Choose a machine with at least 2 batches
        3. Pick two batches to swap
        4. Reorder permutation to swap these batches
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Expected VectorOvenSchedulingSolution, got {type(solution)}"
            )

        # Decode to get batch structure
        schedule = solution.get_schedule()

        # Find machines with at least 2 batches
        machines_with_batches = [
            m
            for m in range(self.problem.n_machines)
            if len(schedule.schedule_per_machine[m]) >= 2
        ]

        if not machines_with_batches:
            # No batches to swap, return unchanged
            return solution, LocalMoveDefault(solution, solution)

        # Choose random machine
        machine = random.choice(machines_with_batches)
        batches = schedule.schedule_per_machine[machine]

        # Choose two random batches to swap
        i, j = sorted(random.sample(range(len(batches)), 2))

        # Get tasks in each batch
        batch_i_tasks = list(batches[i].tasks)
        batch_j_tasks = list(batches[j].tasks)

        # Create new permutation by swapping batch task groups
        new_perm = solution.permutation.copy()

        # Find positions of tasks in permutation
        batch_i_positions = [np.where(new_perm == task)[0][0] for task in batch_i_tasks]
        batch_j_positions = [np.where(new_perm == task)[0][0] for task in batch_j_tasks]

        # Sort positions to maintain relative order within batches
        batch_i_positions.sort()
        batch_j_positions.sort()

        # Swap the task groups in the permutation
        # Extract tasks
        batch_i_extracted = [new_perm[pos] for pos in batch_i_positions]
        batch_j_extracted = [new_perm[pos] for pos in batch_j_positions]

        # Place batch_j tasks where batch_i was
        for idx, pos in enumerate(batch_i_positions):
            if idx < len(batch_j_extracted):
                new_perm[pos] = batch_j_extracted[idx]

        # Place batch_i tasks where batch_j was
        for idx, pos in enumerate(batch_j_positions):
            if idx < len(batch_i_extracted):
                new_perm[pos] = batch_i_extracted[idx]

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)
        return new_solution, move


class MoveJobToExistingBatchMutation(Mutation):
    """
    Moves a job closer to an existing compatible batch in the permutation.
    This encourages the decoder to batch them together.
    """

    problem: OvenSchedulingProblem

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Move a job close to a compatible batch.

        Strategy:
        1. Decode to identify batches
        2. Pick a random batch
        3. Find a job compatible with this batch (same attribute)
        4. Move job close to batch tasks in permutation
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Expected VectorOvenSchedulingSolution, got {type(solution)}"
            )

        # Decode to get batch structure
        schedule = solution.get_schedule()

        # Collect all batches
        all_batches = []
        for m in range(self.problem.n_machines):
            for batch in schedule.schedule_per_machine[m]:
                all_batches.append((m, batch))

        if not all_batches:
            return solution, LocalMoveDefault(solution, solution)

        # Choose random batch
        machine, target_batch = random.choice(all_batches)
        target_attr = target_batch.task_attribute
        target_tasks = list(target_batch.tasks)

        # Find jobs with same attribute not in this batch
        compatible_jobs = [
            job
            for job in range(self.problem.n_jobs)
            if self.problem.tasks_data[job].attribute == target_attr
            and job not in target_batch.tasks
        ]

        if not compatible_jobs:
            return solution, LocalMoveDefault(solution, solution)

        # Choose random compatible job
        job_to_move = random.choice(compatible_jobs)

        # Create new permutation: move job close to target batch
        new_perm = solution.permutation.copy()

        # Find position of job and position of first task in target batch
        job_pos = np.where(new_perm == job_to_move)[0][0]
        target_task = target_tasks[0]
        target_pos = np.where(new_perm == target_task)[0][0]

        # Remove job from current position
        new_perm = np.delete(new_perm, job_pos)

        # Insert near target batch (right after target_task)
        # Adjust target_pos if job was before it
        if job_pos < target_pos:
            target_pos -= 1

        new_perm = np.insert(new_perm, target_pos + 1, job_to_move)

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)
        return new_solution, move


class MoveJobToNewBatchMutation(Mutation):
    """
    Move Job to New Batch
    Moves a job away from its current batch neighbors to encourage
    creating a new batch.
    """

    problem: OvenSchedulingProblem

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Move a job away from its batch to create a new batch.

        Strategy:
        1. Decode to identify batches
        2. Pick a batch with at least 2 jobs
        3. Move one job far from its batch neighbors
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Expected VectorOvenSchedulingSolution, got {type(solution)}"
            )

        # Decode to get batch structure
        schedule = solution.get_schedule()

        # Find batches with at least 2 jobs
        splittable_batches = []
        for m in range(self.problem.n_machines):
            for batch in schedule.schedule_per_machine[m]:
                if len(batch.tasks) >= 2:
                    splittable_batches.append((m, batch))

        if not splittable_batches:
            return solution, LocalMoveDefault(solution, solution)

        # Choose random batch to split
        machine, source_batch = random.choice(splittable_batches)
        batch_tasks = list(source_batch.tasks)

        # Choose random job to move away
        job_to_move = random.choice(batch_tasks)

        # Create new permutation: move job to a random distant position
        new_perm = solution.permutation.copy()

        # Find current position
        job_pos = np.where(new_perm == job_to_move)[0][0]

        # Remove job
        new_perm = np.delete(new_perm, job_pos)

        # Insert at random position (prefer far from current position)
        n = len(new_perm)
        # Choose from positions at least n/4 away
        min_distance = max(1, n // 4)
        far_positions = [pos for pos in range(n) if abs(pos - job_pos) >= min_distance]

        if far_positions:
            new_pos = random.choice(far_positions)
        else:
            new_pos = random.randint(0, n)

        new_perm = np.insert(new_perm, new_pos, job_to_move)

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)
        return new_solution, move


class InvertBatchesOrderMutation(Mutation):
    """
    Invert Batches Order
    Reverses the order of a sequence of batches on a machine.
    """

    problem: OvenSchedulingProblem

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem=problem, **kwargs)

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Invert order of consecutive batches.

        Strategy:
        1. Decode to identify batches
        2. Choose machine and range of batches
        3. Reverse their order in the permutation
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Expected VectorOvenSchedulingSolution, got {type(solution)}"
            )

        # Decode to get batch structure
        schedule = solution.get_schedule()

        # Find machines with at least 2 batches
        machines_with_batches = [
            m
            for m in range(self.problem.n_machines)
            if len(schedule.schedule_per_machine[m]) >= 2
        ]

        if not machines_with_batches:
            return solution, LocalMoveDefault(solution, solution)

        # Choose random machine
        machine = random.choice(machines_with_batches)
        batches = schedule.schedule_per_machine[machine]

        # Choose range of batches to invert
        num_batches = len(batches)
        if num_batches < 2:
            return solution, LocalMoveDefault(solution, solution)

        # Choose start and end positions
        i, j = sorted(random.sample(range(num_batches), 2))

        # Get all tasks in the batches to invert
        tasks_to_invert = []
        for batch_idx in range(i, j + 1):
            tasks_to_invert.extend(list(batches[batch_idx].tasks))

        # Find positions of these tasks in permutation
        new_perm = solution.permutation.copy()
        task_positions = []
        for task in tasks_to_invert:
            pos = np.where(new_perm == task)[0][0]
            task_positions.append((pos, task))

        # Sort by position
        task_positions.sort()

        # Reverse the tasks
        reversed_tasks = [task for _, task in reversed(task_positions)]

        # Place reversed tasks back
        for idx, (pos, _) in enumerate(task_positions):
            new_perm[pos] = reversed_tasks[idx]

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)
        return new_solution, move


class MoveJobsToNewBatchMutation(Mutation):
    """
    Moves multiple jobs from one batch to create a new batch elsewhere.
    """

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        min_jobs_to_move: int = 2,
        max_jobs_to_move: int = 5,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, **kwargs)
        self.min_jobs_to_move = min_jobs_to_move
        self.max_jobs_to_move = max_jobs_to_move

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """
        Move multiple jobs together to form a new batch.

        Strategy:
        1. Find a batch with enough jobs
        2. Select subset of jobs (with same attribute)
        3. Move them together to a new position
        """
        if not isinstance(solution, VectorOvenSchedulingSolution):
            raise ValueError(
                f"Expected VectorOvenSchedulingSolution, got {type(solution)}"
            )

        # Decode to get batch structure
        schedule = solution.get_schedule()

        # Find batches with enough jobs
        suitable_batches = []
        for m in range(self.problem.n_machines):
            for batch in schedule.schedule_per_machine[m]:
                if len(batch.tasks) >= self.min_jobs_to_move:
                    suitable_batches.append((m, batch))

        if not suitable_batches:
            return solution, LocalMoveDefault(solution, solution)

        # Choose random batch
        machine, source_batch = random.choice(suitable_batches)
        batch_tasks = list(source_batch.tasks)

        # Determine how many jobs to move
        num_to_move = random.randint(
            self.min_jobs_to_move, min(self.max_jobs_to_move, len(batch_tasks))
        )

        # Choose random jobs to move
        jobs_to_move = random.sample(batch_tasks, num_to_move)

        # Create new permutation
        new_perm = solution.permutation.copy()

        # Find positions of jobs to move
        job_positions = []
        for job in jobs_to_move:
            pos = np.where(new_perm == job)[0][0]
            job_positions.append((pos, job))

        # Sort by position
        job_positions.sort()

        # Remove jobs from current positions (in reverse order to maintain indices)
        for pos, job in reversed(job_positions):
            new_perm = np.delete(new_perm, pos)

        # Insert jobs together at new position
        n = len(new_perm)
        new_pos = random.randint(0, n)

        for job in jobs_to_move:
            new_perm = np.insert(new_perm, new_pos, job)
            new_pos += 1  # Keep them together

        # Create new solution
        new_solution = VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=new_perm,
        )

        move = LocalMoveDefault(prev_solution=solution, new_solution=new_solution)
        return new_solution, move


class ScheduleAwareMixedMutation(Mutation):
    """
    Mixed mutation combining all schedule-aware operators.

    Randomly applies one of the schedule-aware mutations from the paper.
    """

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        swap_batches_prob: float = 0.25,
        mjeb_prob: float = 0.25,
        mjnb_prob: float = 0.20,
        invert_prob: float = 0.15,
        move_jobs_prob: float = 0.15,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, **kwargs)

        # Normalize probabilities
        total = swap_batches_prob + mjeb_prob + mjnb_prob + invert_prob + move_jobs_prob
        self.swap_batches_prob = swap_batches_prob / total
        self.mjeb_prob = mjeb_prob / total
        self.mjnb_prob = mjnb_prob / total
        self.invert_prob = invert_prob / total
        self.move_jobs_prob = move_jobs_prob / total

        # Create mutation operators
        self.mutations = {
            "swap_batches": SwapBatchesMutation(problem),
            "mjeb": MoveJobToExistingBatchMutation(problem),
            "mjnb": MoveJobToNewBatchMutation(problem),
            "invert": InvertBatchesOrderMutation(problem),
            "move_jobs": MoveJobsToNewBatchMutation(problem),
        }

    def mutate(self, solution: Solution) -> tuple[Solution, LocalMove]:
        """Apply randomly selected schedule-aware mutation."""
        r = random.random()

        if r < self.swap_batches_prob:
            return self.mutations["swap_batches"].mutate(solution)
        elif r < self.swap_batches_prob + self.mjeb_prob:
            return self.mutations["mjeb"].mutate(solution)
        elif r < self.swap_batches_prob + self.mjeb_prob + self.mjnb_prob:
            return self.mutations["mjnb"].mutate(solution)
        elif (
            r
            < self.swap_batches_prob
            + self.mjeb_prob
            + self.mjnb_prob
            + self.invert_prob
        ):
            return self.mutations["invert"].mutate(solution)
        else:
            return self.mutations["move_jobs"].mutate(solution)
