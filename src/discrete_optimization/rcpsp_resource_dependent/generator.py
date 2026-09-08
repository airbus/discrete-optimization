#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Utilities for generating resource-dependent RCPSP problems from standard RCPSP problems.

This module provides functions to transform regular RCPSP problems into resource-dependent
variants by adding mode-dependent resource consumption patterns.
"""

import random
from copy import deepcopy
from enum import Enum
from typing import Hashable, Optional

from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
)


class DependencyStrategy(Enum):
    """Strategy for adding resource dependencies."""

    RANDOM = "random"
    PREDECESSOR_BASED = "predecessor_based"
    RESOURCE_CONTENTION = "resource_contention"
    MIXED = "mixed"


def generate_resource_dependent_problem(
    base_problem: RcpspProblem,
    dependency_strategy: DependencyStrategy = DependencyStrategy.PREDECESSOR_BASED,
    dependency_probability: float = 0.3,
    variation_factor: float = 0.5,
    seed: Optional[int] = None,
) -> RcpspResourceDependentProblem:
    """Generate a resource-dependent RCPSP problem from a standard RCPSP problem.

    Args:
        base_problem: Base RCPSP problem to transform
        dependency_strategy: Strategy for selecting which tasks have dependencies
        dependency_probability: Probability that a task-resource pair becomes dependent (0.0-1.0)
        variation_factor: Factor controlling variation in resource consumption (0.0-1.0).
            Higher values mean more variation between different mode configurations.
        seed: Random seed for reproducibility

    Returns:
        RcpspResourceDependentProblem with added resource dependencies

    Example:
        >>> from discrete_optimization.rcpsp.parser import get_data_available, parse_file
        >>> from discrete_optimization.rcpsp_resource_dependent.generator import (
        ...     generate_resource_dependent_problem,
        ...     DependencyStrategy
        ... )
        >>> # Load a standard RCPSP instance
        >>> files = get_data_available()
        >>> rcpsp_problem = parse_file(files[0])
        >>> # Generate resource-dependent variant
        >>> rd_problem = generate_resource_dependent_problem(
        ...     base_problem=rcpsp_problem,
        ...     dependency_strategy=DependencyStrategy.PREDECESSOR_BASED,
        ...     dependency_probability=0.3,
        ...     variation_factor=0.5,
        ...     seed=42
        ... )
    """
    if seed is not None:
        random.seed(seed)

    # Deep copy the mode_details to avoid modifying the original
    new_mode_details = deepcopy(base_problem.mode_details)

    # Get task dependency relationships based on strategy
    task_dependencies = _select_dependencies(
        base_problem, dependency_strategy, dependency_probability
    )

    # Transform selected task-resource pairs to be mode-dependent
    for task, dependent_tasks in task_dependencies.items():
        if task == base_problem.source_task or task == base_problem.sink_task:
            continue

        for mode in new_mode_details[task].keys():
            # Get all resources for this task-mode
            resources_to_process = [
                r
                for r in new_mode_details[task][mode].keys()
                if r != "duration" and new_mode_details[task][mode][r] > 0
            ]

            for resource in resources_to_process:
                if random.random() < dependency_probability:
                    original_value = new_mode_details[task][mode][resource]

                    # Create dependency mapping
                    dependency_mapping = _create_dependency_mapping(
                        base_problem=base_problem,
                        dependent_tasks=dependent_tasks,
                        original_value=original_value,
                        variation_factor=variation_factor,
                    )

                    # Replace the fixed value with the dependency mapping
                    new_mode_details[task][mode][resource] = dependency_mapping

    return RcpspResourceDependentProblem(
        resources=base_problem.resources,
        non_renewable_resources=base_problem.non_renewable_resources,
        mode_details=new_mode_details,
        successors=base_problem.successors,
        horizon=base_problem.horizon,
        tasks_list=base_problem.tasks_list,
        source_task=base_problem.source_task,
        sink_task=base_problem.sink_task,
    )


def _select_dependencies(
    problem: RcpspProblem,
    strategy: DependencyStrategy,
    probability: float,
) -> dict[Hashable, list[Hashable]]:
    """Select which tasks depend on which other tasks based on strategy.

    Returns:
        Dictionary mapping each task to list of tasks it may depend on
    """
    dependencies = {}

    if strategy == DependencyStrategy.PREDECESSOR_BASED:
        # Tasks depend on their direct predecessors
        predecessors = _compute_predecessors(problem.successors)
        for task in problem.tasks_list:
            if task in predecessors and len(predecessors[task]) > 0:
                dependencies[task] = list(predecessors[task])

    elif strategy == DependencyStrategy.RESOURCE_CONTENTION:
        # Tasks depend on other tasks that use the same resources
        resource_users = _compute_resource_users(problem)
        for task in problem.tasks_list:
            task_resources = _get_task_resources(problem, task)
            dependent_tasks = set()
            for resource in task_resources:
                if resource in resource_users:
                    dependent_tasks.update(resource_users[resource])
            dependent_tasks.discard(task)  # Don't depend on self
            if dependent_tasks:
                dependencies[task] = list(dependent_tasks)

    elif strategy == DependencyStrategy.RANDOM:
        # Random dependencies on any predecessor in the graph
        predecessors = _compute_predecessors(problem.successors)
        all_predecessors = _compute_all_predecessors(problem.successors)
        for task in problem.tasks_list:
            if task in all_predecessors and len(all_predecessors[task]) > 0:
                # Randomly select subset of all predecessors
                potential_deps = list(all_predecessors[task])
                num_deps = max(1, int(len(potential_deps) * probability))
                dependencies[task] = random.sample(potential_deps, num_deps)

    elif strategy == DependencyStrategy.MIXED:
        # Combine predecessor-based and resource contention
        predecessors = _compute_predecessors(problem.successors)
        resource_users = _compute_resource_users(problem)

        for task in problem.tasks_list:
            dependent_tasks = set()

            # Add direct predecessors
            if task in predecessors:
                dependent_tasks.update(predecessors[task])

            # Add some tasks competing for same resources
            task_resources = _get_task_resources(problem, task)
            for resource in task_resources:
                if resource in resource_users:
                    # Add a subset of resource competitors
                    competitors = [t for t in resource_users[resource] if t != task]
                    if competitors:
                        num_to_add = max(1, len(competitors) // 3)
                        dependent_tasks.update(random.sample(competitors, num_to_add))

            if dependent_tasks:
                dependencies[task] = list(dependent_tasks)

    return dependencies


def _create_dependency_mapping(
    base_problem: RcpspProblem,
    dependent_tasks: list[Hashable],
    original_value: int,
    variation_factor: float,
) -> dict[frozenset[tuple[Hashable, int]], int]:
    """Create a mapping from mode configurations to resource consumption values.

    Args:
        base_problem: Base RCPSP problem
        dependent_tasks: Tasks whose modes affect this consumption
        original_value: Original fixed resource consumption
        variation_factor: How much variation to introduce (0.0-1.0)

    Returns:
        Mapping from frozenset of (task, mode) pairs to consumption values
    """
    if not dependent_tasks:
        return {frozenset([]): original_value}

    # For simplicity, focus on the first dependent task's modes
    # (Can be extended to combinations of multiple tasks)
    primary_dependent = dependent_tasks[0]
    modes = list(base_problem.mode_details[primary_dependent].keys())

    mapping = {}
    for mode in modes:
        # Vary consumption based on mode
        # Lower modes get values closer to original
        # Higher modes get more variation
        mode_index = modes.index(mode)
        variation = int(original_value * variation_factor * (mode_index / len(modes)))

        # Randomly increase or decrease
        if random.random() < 0.5:
            new_value = max(0, original_value - variation)
        else:
            new_value = original_value + variation

        mapping[frozenset([(primary_dependent, mode)])] = new_value

    return mapping


def _compute_predecessors(
    successors: dict[Hashable, list[Hashable]],
) -> dict[Hashable, set[Hashable]]:
    """Compute direct predecessors from successor relationships."""
    predecessors = {task: set() for task in successors.keys()}
    for task, succs in successors.items():
        for succ in succs:
            if succ not in predecessors:
                predecessors[succ] = set()
            predecessors[succ].add(task)
    return predecessors


def _compute_all_predecessors(
    successors: dict[Hashable, list[Hashable]],
) -> dict[Hashable, set[Hashable]]:
    """Compute all predecessors (transitive closure) from successor relationships."""
    all_preds = {task: set() for task in successors.keys()}

    # Topological traversal
    def visit(task, visited):
        if task in visited:
            return visited[task]
        visited[task] = set()
        for succ in successors.get(task, []):
            visited[task].add(succ)
            visited[task].update(visit(succ, visited))
        return visited[task]

    # Invert: for each task, find what can reach it
    all_succs = {}
    for task in successors.keys():
        all_succs[task] = visit(task, {})

    for task in successors.keys():
        for other_task, succs in all_succs.items():
            if task in succs:
                all_preds[task].add(other_task)

    return all_preds


def _compute_resource_users(problem: RcpspProblem) -> dict[str, set[Hashable]]:
    """Compute which tasks use each resource."""
    resource_users = {r: set() for r in problem.resources.keys()}

    for task, modes in problem.mode_details.items():
        for mode_id, mode_data in modes.items():
            for resource, consumption in mode_data.items():
                if resource != "duration" and consumption > 0:
                    resource_users[resource].add(task)

    return resource_users


def _get_task_resources(problem: RcpspProblem, task: Hashable) -> set[str]:
    """Get all resources used by a task across all its modes."""
    resources = set()
    for mode_id, mode_data in problem.mode_details[task].items():
        for resource, consumption in mode_data.items():
            if resource != "duration" and consumption > 0:
                resources.add(resource)
    return resources


def add_simple_resource_dependency(
    base_problem: RcpspProblem,
    task: Hashable,
    resource: str,
    dependent_task: Hashable,
    mode_to_value: dict[int, int],
) -> RcpspResourceDependentProblem:
    """Add a simple resource dependency to a specific task-resource pair.

    This is a low-level utility for manually creating specific dependencies.

    Args:
        base_problem: Base RCPSP problem
        task: Task whose resource consumption should depend on another task
        resource: Resource name
        dependent_task: Task whose mode affects the consumption
        mode_to_value: Mapping from dependent_task's modes to resource consumption values

    Returns:
        New RcpspResourceDependentProblem with the dependency added

    Example:
        >>> # Make task "2" resource "R1" consumption depend on task "1" mode
        >>> rd_problem = add_simple_resource_dependency(
        ...     base_problem=rcpsp_problem,
        ...     task="2",
        ...     resource="R1",
        ...     dependent_task="1",
        ...     mode_to_value={1: 2, 2: 5}  # mode 1 -> 2 units, mode 2 -> 5 units
        ... )
    """
    new_mode_details = deepcopy(base_problem.mode_details)

    # Update all modes of the task
    for mode in new_mode_details[task].keys():
        # Convert mode_to_value to the frozenset format
        dependency_mapping = {
            frozenset([(dependent_task, dep_mode)]): value
            for dep_mode, value in mode_to_value.items()
        }
        new_mode_details[task][mode][resource] = dependency_mapping

    return RcpspResourceDependentProblem(
        resources=base_problem.resources,
        non_renewable_resources=base_problem.non_renewable_resources,
        mode_details=new_mode_details,
        successors=base_problem.successors,
        horizon=base_problem.horizon,
        tasks_list=base_problem.tasks_list,
        source_task=base_problem.source_task,
        sink_task=base_problem.sink_task,
    )


def validate_resource_dependent_problem(
    problem: RcpspResourceDependentProblem,
) -> dict[str, any]:
    """Validate a resource-dependent problem and return statistics.

    Args:
        problem: Problem to validate

    Returns:
        Dictionary with validation statistics including:
        - num_tasks: Total number of tasks
        - num_dependent_consumptions: Number of dependent resource consumptions
        - num_fixed_consumptions: Number of fixed resource consumptions
        - dependency_ratio: Ratio of dependent to total consumptions
        - tasks_with_dependencies: Set of tasks having at least one dependency

    Example:
        >>> stats = validate_resource_dependent_problem(rd_problem)
        >>> print(f"Dependency ratio: {stats['dependency_ratio']:.2%}")
    """
    num_fixed = 0
    num_dependent = 0
    tasks_with_deps = set()

    for task, modes in problem.mode_details.items():
        for mode, mode_data in modes.items():
            for key, value in mode_data.items():
                if key == "duration":
                    continue

                if isinstance(value, int):
                    num_fixed += 1
                elif isinstance(value, dict):
                    num_dependent += 1
                    tasks_with_deps.add(task)

    total = num_fixed + num_dependent
    dependency_ratio = num_dependent / total if total > 0 else 0.0

    return {
        "num_tasks": len(problem.tasks_list),
        "num_dependent_consumptions": num_dependent,
        "num_fixed_consumptions": num_fixed,
        "dependency_ratio": dependency_ratio,
        "tasks_with_dependencies": tasks_with_deps,
        "num_tasks_with_dependencies": len(tasks_with_deps),
    }
