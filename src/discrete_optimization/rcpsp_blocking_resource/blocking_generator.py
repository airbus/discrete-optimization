#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Generator for RCPSP instances with resource blocking constraints.

This module provides utilities to generate RCPSP problems with blocking constraints
from standard RCPSP instances. It creates realistic scenarios such as:
- Setup times between tasks using same resources
- Changeover periods between task modes
- Project-level resource reservations
"""

import random
from typing import Optional

from discrete_optimization.generic_tasks_tools.entities import GroupEntity, TaskEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
    FlexibleGapBlockingConstraint,
    SpanBlockingConstraint,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_blocking_resource.problem_with_blocking import (
    RcpspWithResourceBlocking,
)


def generate_setup_time_blocking(
    base_problem: RcpspProblem,
    setup_ratio: float = 0.2,
    resource_subset: Optional[list[str]] = None,
    seed: Optional[int] = None,
    blocking_intensity: float = 0.5,
) -> RcpspWithResourceBlocking:
    """Generate RCPSP with setup time blocking between consecutive tasks.

    Creates blocking constraints representing setup/changeover times between tasks
    that consume the same resources. Since blocking is ADDITIVE (task + blocking <= capacity),
    the blocking amount is carefully chosen to avoid infeasibilities.

    Args:
        base_problem: Original RCPSP problem
        setup_ratio: Ratio of setup time to average task duration (default 0.2 = 20%)
        resource_subset: Resources affected by setup (None = all renewable resources)
        seed: Random seed for reproducibility
        blocking_intensity: Fraction of AVAILABLE capacity to block (default 0.5 = 50%)
                          Available capacity = total capacity - max task consumption

    Returns:
        RcpspWithResourceBlocking with gap blocking constraints for setup times

    Example:
        >>> from discrete_optimization.rcpsp.parser import parse_file, get_data_available
        >>> from discrete_optimization.rcpsp.blocking_generator import generate_setup_time_blocking
        >>> files = get_data_available()
        >>> base_problem = parse_file(files[0])
        >>> problem_with_setup = generate_setup_time_blocking(base_problem, blocking_intensity=0.7)
    """
    if seed is not None:
        random.seed(seed)

    # Determine which resources have setup times
    if resource_subset is None:
        resource_subset = [
            r
            for r in base_problem.resources_list
            if r not in base_problem.non_renewable_resources
        ]

    # Compute max task consumption per resource to avoid infeasibilities
    max_consumption = {}
    for resource in resource_subset:
        max_cons = 0
        for task in base_problem.tasks_list:
            if task in base_problem.mode_details:
                task_mode = base_problem.mode_details[task].get(1, {})
                max_cons = max(max_cons, task_mode.get(resource, 0))
        max_consumption[resource] = max_cons

    blocking_constraints: list[FlexibleGapBlockingConstraint] = []

    # Generate setup constraints for tasks with precedence relationships
    for task, successors in base_problem.successors.items():
        # Skip dummy tasks
        if task == base_problem.source_task or task == base_problem.sink_task:
            continue

        for successor in successors:
            if successor == base_problem.sink_task:
                continue

            # Check if tasks share resources
            task_mode = base_problem.mode_details[task][1]  # Use mode 1
            succ_mode = base_problem.mode_details[successor][1]

            shared_resources = {}
            for resource in resource_subset:
                task_usage = task_mode.get(resource, 0)
                succ_usage = succ_mode.get(resource, 0)
                if task_usage > 0 and succ_usage > 0:
                    capacity = base_problem.resources[resource]
                    if isinstance(capacity, list):
                        capacity = min(capacity)

                    # ADDITIVE blocking: ensure task + blocking <= capacity
                    # Available capacity = total - max task that could run during gap
                    available = capacity - max_consumption.get(resource, 0)
                    if available <= 0:
                        continue  # No room for blocking

                    # Block a fraction of available capacity
                    setup_amount = max(1, int(available * blocking_intensity))
                    shared_resources[resource] = setup_amount

            if shared_resources:
                # Create gap blocking constraint for setup time
                constraint: FlexibleGapBlockingConstraint = (
                    TaskEntity(task),
                    StartOrEnd.END,
                    TaskEntity(successor),
                    StartOrEnd.START,
                    shared_resources,
                    BlockingConstraintMetadata(
                        mode=BlockingMode.RESERVATION,
                        description=f"Setup time between task {task} and {successor}",
                    ),
                )
                blocking_constraints.append(constraint)

    return RcpspWithResourceBlocking(
        resources=base_problem.resources,
        non_renewable_resources=base_problem.non_renewable_resources,
        mode_details=base_problem.mode_details,
        successors=base_problem.successors,
        horizon=base_problem.horizon,
        tasks_list=base_problem.tasks_list,
        source_task=base_problem.source_task,
        sink_task=base_problem.sink_task,
        name_task=base_problem.name_task,
        calendar_details=base_problem.calendar_details,
        flexible_gap_blocking_constraints=blocking_constraints,
    )


def generate_batch_blocking(
    base_problem: RcpspProblem,
    batch_size: int = 3,
    resource_name: Optional[str] = None,
    seed: Optional[int] = None,
    blocking_intensity: float = 0.2,
) -> RcpspWithResourceBlocking:
    """Generate RCPSP with span blocking for batch processing.

    Creates span blocking constraints where groups of tasks must reserve
    a portion of resources for their entire execution span. Since blocking is
    ADDITIVE (task + blocking <= capacity), the blocking amount is carefully
    chosen to avoid infeasibilities.

    Args:
        base_problem: Original RCPSP problem
        batch_size: Number of tasks per batch
        resource_name: Resource to block (None = first renewable resource)
        seed: Random seed for batch generation
        blocking_intensity: Fraction of AVAILABLE capacity to block (default 0.4 = 40%)
                          Available capacity = total capacity - max batch task consumption

    Returns:
        RcpspWithResourceBlocking with span blocking constraints for batches

    Example:
        #>>> from discrete_optimization.rcpsp.parser import parse_file, get_data_available
        #>>> from discrete_optimization.rcpsp.blocking_generator import generate_batch_blocking
        #>>> files = get_data_available()
        #>>> base_problem = parse_file(files[0])
        #>>> problem_with_batches = generate_batch_blocking(base_problem, blocking_intensity=0.6)
    """
    if seed is not None:
        random.seed(seed)

    # Select resource for batch blocking
    if resource_name is None:
        renewable_resources = [
            r
            for r in base_problem.resources_list
            if r not in base_problem.non_renewable_resources
        ]
        if not renewable_resources:
            raise ValueError("No renewable resources available for batch blocking")
        resource_name = renewable_resources[0]

    # Get non-dummy tasks
    non_dummy_tasks = [
        t
        for t in base_problem.tasks_list
        if t != base_problem.source_task and t != base_problem.sink_task
    ]

    # Create batches
    random.shuffle(non_dummy_tasks)
    batches = [
        non_dummy_tasks[i : i + batch_size]
        for i in range(0, len(non_dummy_tasks), batch_size)
    ]

    blocking_constraints: list[SpanBlockingConstraint] = []

    # Get resource capacity
    capacity = base_problem.resources[resource_name]
    if isinstance(capacity, list):
        capacity = min(capacity)

    # Create span blocking for each batch
    for batch_idx, batch_tasks in enumerate(batches):
        if len(batch_tasks) < 2:
            continue  # Skip single-task batches

        # Compute max consumption within this batch
        max_batch_consumption = 0
        for task in batch_tasks:
            if task in base_problem.mode_details:
                task_mode = base_problem.mode_details[task].get(1, {})
                max_batch_consumption = max(
                    max_batch_consumption, task_mode.get(resource_name, 0)
                )

        # ADDITIVE blocking: ensure batch_tasks + blocking <= capacity
        # Available capacity = total - max task in batch
        available = capacity - max_batch_consumption
        if available <= 0:
            continue  # No room for blocking

        # Block a fraction of available capacity
        blocking_amount = max(1, int(available * blocking_intensity))

        constraint: SpanBlockingConstraint = (
            GroupEntity(frozenset(batch_tasks)),
            {resource_name: blocking_amount},
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION,
                description=f"Batch {batch_idx + 1} reservation for {resource_name}",
            ),
        )
        blocking_constraints.append(constraint)

    return RcpspWithResourceBlocking(
        resources=base_problem.resources,
        non_renewable_resources=base_problem.non_renewable_resources,
        mode_details=base_problem.mode_details,
        successors=base_problem.successors,
        horizon=base_problem.horizon,
        tasks_list=base_problem.tasks_list,
        source_task=base_problem.source_task,
        sink_task=base_problem.sink_task,
        name_task=base_problem.name_task,
        calendar_details=base_problem.calendar_details,
        span_blocking_constraints=blocking_constraints,
    )


def generate_combined_blocking(
    base_problem: RcpspProblem,
    setup_ratio: float = 0.2,
    batch_size: int = 3,
    seed: Optional[int] = None,
) -> RcpspWithResourceBlocking:
    """Generate RCPSP with both setup time and batch blocking constraints.

    Combines gap blocking (setup times) and span blocking (batches) to create
    more complex and realistic scheduling scenarios.

    Args:
        base_problem: Original RCPSP problem
        setup_ratio: Setup time ratio for gap blocking
        batch_size: Batch size for span blocking
        seed: Random seed

    Returns:
        RcpspWithResourceBlocking with both types of blocking constraints

    Example:
        #>>> from discrete_optimization.rcpsp.parser import parse_file, get_data_available
        #>>> from discrete_optimization.rcpsp.blocking_generator import generate_combined_blocking
        #>>> files = get_data_available()
        #>>> base_problem = parse_file(files[0])
        #>>> problem = generate_combined_blocking(base_problem)
    """
    if seed is not None:
        random.seed(seed)

    # Generate both types of constraints
    setup_problem = generate_setup_time_blocking(base_problem, setup_ratio, seed=seed)
    batch_problem = generate_batch_blocking(base_problem, batch_size, seed=seed)

    return RcpspWithResourceBlocking(
        resources=base_problem.resources,
        non_renewable_resources=base_problem.non_renewable_resources,
        mode_details=base_problem.mode_details,
        successors=base_problem.successors,
        horizon=base_problem.horizon,
        tasks_list=base_problem.tasks_list,
        source_task=base_problem.source_task,
        sink_task=base_problem.sink_task,
        name_task=base_problem.name_task,
        calendar_details=base_problem.calendar_details,
        flexible_gap_blocking_constraints=setup_problem.get_flexible_gap_blocking_constraints(),
        span_blocking_constraints=batch_problem.get_span_blocking_constraints(),
    )
