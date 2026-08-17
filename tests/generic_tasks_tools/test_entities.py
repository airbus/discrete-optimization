#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for scheduling entity abstraction and hierarchical composition."""

import pytest

from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    GroupEntity,
    TaskEntity,
    TaskModeEntity,
)
from discrete_optimization.generic_tasks_tools.multimode import MultimodeSolution
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingSolution


class MockTask:
    """Mock task for testing."""

    def __init__(self, task_id):
        self.id = task_id

    def __eq__(self, other):
        return isinstance(other, MockTask) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return str(self.id) < str(other.id)

    def __repr__(self):
        return f"Task({self.id})"


class MockSchedulingSolution(SchedulingSolution):
    """Mock solution for testing entity queries."""

    def __init__(self, task_times, task_modes=None):
        """
        Args:
            task_times: dict mapping task -> (start, end)
            task_modes: dict mapping task -> mode (optional)
        """
        self.task_times = task_times
        self.task_modes = task_modes or {}

    def get_start_time(self, task):
        return self.task_times[task][0]

    def get_end_time(self, task):
        return self.task_times[task][1]

    def copy(self):
        """Implement required copy method."""
        return MockSchedulingSolution(
            task_times=dict(self.task_times), task_modes=dict(self.task_modes)
        )


class MockMultimodeSolution(MultimodeSolution):
    """Mock multimode solution for testing mode-aware entities."""

    def __init__(self, task_times, task_modes):
        """
        Args:
            task_times: dict mapping task -> (start, end)
            task_modes: dict mapping task -> mode
        """
        self.task_times = task_times
        self.task_modes = task_modes

    def get_start_time(self, task):
        return self.task_times[task][0]

    def get_end_time(self, task):
        return self.task_times[task][1]

    def get_mode(self, task):
        return self.task_modes[task]

    def copy(self):
        """Implement required copy method."""
        return MockMultimodeSolution(
            task_times=dict(self.task_times), task_modes=dict(self.task_modes)
        )


# === TaskEntity Tests ===


def test_task_entity_basic():
    """Test basic TaskEntity functionality."""
    task = MockTask(1)
    entity = TaskEntity(task)

    solution = MockSchedulingSolution({task: (10, 20)})

    assert entity.get_start_time(solution) == 10
    assert entity.get_end_time(solution) == 20
    assert entity.is_active(solution)
    assert entity.get_tasks() == frozenset({task})
    assert entity.entity_id == ("task", task)


def test_task_entity_hashable():
    """Test that TaskEntity can be used as dict key."""
    task1 = MockTask(1)
    task2 = MockTask(2)
    entity1 = TaskEntity(task1)
    entity2 = TaskEntity(task1)
    entity3 = TaskEntity(task2)

    # Same task -> same entity
    assert entity1 == entity2
    assert hash(entity1) == hash(entity2)

    # Different task -> different entity
    assert entity1 != entity3

    # Can use as dict key
    d = {entity1: "value1", entity3: "value3"}
    assert d[entity2] == "value1"


# === GroupEntity Tests ===


def test_group_entity_basic():
    """Test basic GroupEntity functionality."""
    task1 = MockTask(1)
    task2 = MockTask(2)
    task3 = MockTask(3)

    entity = GroupEntity(frozenset({task1, task2, task3}), group_id="batch_A")

    solution = MockSchedulingSolution({task1: (5, 10), task2: (8, 15), task3: (12, 20)})

    # Start = min start, End = max end
    assert entity.get_start_time(solution) == 5
    assert entity.get_end_time(solution) == 20
    assert entity.is_active(solution)
    assert entity.get_tasks() == frozenset({task1, task2, task3})
    assert entity.entity_id == ("group", "batch_A")


def test_group_entity_empty_raises():
    """Test that empty GroupEntity raises ValueError."""
    with pytest.raises(ValueError, match="must contain at least one task"):
        GroupEntity(frozenset())


def test_group_entity_deterministic_id():
    """Test that GroupEntity without explicit ID has deterministic entity_id."""
    task1 = MockTask(1)
    task2 = MockTask(2)

    entity1 = GroupEntity(frozenset({task1, task2}))
    entity2 = GroupEntity(frozenset({task2, task1}))  # Different order

    # Should have same ID (order-independent)
    assert entity1.entity_id == entity2.entity_id
    assert entity1 == entity2


# === TaskModeEntity Tests ===


def test_task_mode_entity_active():
    """Test TaskModeEntity when task is in specified mode."""
    task = MockTask(1)
    entity = TaskModeEntity(task, mode=2)

    solution = MockMultimodeSolution({task: (10, 20)}, task_modes={task: 2})

    assert entity.is_active(solution)
    assert entity.get_start_time(solution) == 10
    assert entity.get_end_time(solution) == 20
    assert entity.get_tasks() == frozenset({task})
    assert entity.entity_id == ("task_mode", task, 2)


def test_task_mode_entity_inactive():
    """Test TaskModeEntity when task is in different mode."""
    task = MockTask(1)
    entity = TaskModeEntity(task, mode=2)

    solution = MockMultimodeSolution({task: (10, 20)}, task_modes={task: 1})

    # Entity is inactive
    assert not entity.is_active(solution)

    # Querying times should raise
    with pytest.raises(ValueError, match="not active"):
        entity.get_start_time(solution)

    with pytest.raises(ValueError, match="not active"):
        entity.get_end_time(solution)


def test_task_mode_entity_no_mode_support():
    """Test TaskModeEntity when solution doesn't support modes."""
    task = MockTask(1)
    entity = TaskModeEntity(task, mode=2)

    solution = MockSchedulingSolution({task: (10, 20)})

    # Without mode support, entity is inactive
    assert not entity.is_active(solution)


# === CompositeEntity Tests ===


def test_composite_entity_basic():
    """Test basic CompositeEntity with TaskEntity children."""
    task1 = MockTask(1)
    task2 = MockTask(2)

    entity1 = TaskEntity(task1)
    entity2 = TaskEntity(task2)

    composite = CompositeEntity(frozenset({entity1, entity2}), composite_id="phase1")

    solution = MockSchedulingSolution({task1: (5, 10), task2: (8, 15)})

    assert composite.get_start_time(solution) == 5  # min of children
    assert composite.get_end_time(solution) == 15  # max of children
    assert composite.is_active(solution)
    assert composite.get_tasks() == frozenset({task1, task2})
    assert composite.entity_id == ("composite", "phase1")


def test_composite_entity_hierarchical():
    """Test hierarchical composition: CompositeEntity containing other CompositeEntity."""
    task1 = MockTask(1)
    task2 = MockTask(2)
    task3 = MockTask(3)
    task4 = MockTask(4)

    # Level 1: Tasks
    e1 = TaskEntity(task1)
    e2 = TaskEntity(task2)
    e3 = TaskEntity(task3)
    e4 = TaskEntity(task4)

    # Level 2: Phases
    phase1 = CompositeEntity(frozenset({e1, e2}), composite_id="phase1")
    phase2 = CompositeEntity(frozenset({e3, e4}), composite_id="phase2")

    # Level 3: Project
    project = CompositeEntity(frozenset({phase1, phase2}), composite_id="project_alpha")

    solution = MockSchedulingSolution(
        {
            task1: (0, 5),
            task2: (5, 10),
            task3: (10, 15),
            task4: (15, 20),
        }
    )

    # Project spans from start of phase1 to end of phase2
    assert project.get_start_time(solution) == 0
    assert project.get_end_time(solution) == 20
    assert project.is_active(solution)

    # Should collect all tasks recursively
    assert project.get_tasks() == frozenset({task1, task2, task3, task4})


def test_composite_entity_mixed_types():
    """Test CompositeEntity with mixed entity types."""
    task1 = MockTask(1)
    task2 = MockTask(2)
    task3 = MockTask(3)
    task4 = MockTask(4)

    # Mix different entity types
    task_entity = TaskEntity(task1)
    group_entity = GroupEntity(frozenset({task2, task3}), group_id="batch")
    mode_entity = TaskModeEntity(task4, mode=2)

    composite = CompositeEntity(
        frozenset({task_entity, group_entity, mode_entity}), composite_id="mixed"
    )

    solution = MockMultimodeSolution(
        {
            task1: (0, 10),
            task2: (5, 15),
            task3: (10, 20),
            task4: (20, 30),
        },
        task_modes={task1: 1, task2: 1, task3: 1, task4: 2},
    )

    # All children active
    assert composite.is_active(solution)
    assert composite.get_start_time(solution) == 0
    assert composite.get_end_time(solution) == 30
    assert composite.get_tasks() == frozenset({task1, task2, task3, task4})


def test_composite_entity_partial_activation():
    """Test CompositeEntity where some children are inactive."""
    task1 = MockTask(1)
    task2 = MockTask(2)
    task3 = MockTask(3)

    e1 = TaskEntity(task1)
    e2 = TaskModeEntity(task2, mode=2)  # Will be inactive
    e3 = TaskEntity(task3)

    composite = CompositeEntity(frozenset({e1, e2, e3}))

    solution = MockMultimodeSolution(
        {task1: (0, 10), task2: (5, 15), task3: (10, 20)},
        task_modes={task1: 1, task2: 1, task3: 1},  # task2 in mode 1, so e2 is inactive
    )

    # Composite is still active (e1 and e3 are active)
    assert composite.is_active(solution)

    # Times only consider active children
    assert composite.get_start_time(solution) == 0  # min(e1, e3)
    assert composite.get_end_time(solution) == 20  # max(e1, e3)

    # get_tasks() returns all tasks (even from inactive entities)
    assert composite.get_tasks() == frozenset({task1, task2, task3})


def test_composite_entity_all_inactive():
    """Test CompositeEntity where all children are inactive."""
    task1 = MockTask(1)
    task2 = MockTask(2)

    e1 = TaskModeEntity(task1, mode=2)
    e2 = TaskModeEntity(task2, mode=3)

    composite = CompositeEntity(frozenset({e1, e2}))

    solution = MockMultimodeSolution(
        {task1: (0, 10), task2: (5, 15)}, task_modes={task1: 1, task2: 1}
    )

    # All children inactive -> composite is inactive
    assert not composite.is_active(solution)

    # Querying times should raise
    with pytest.raises(ValueError, match="no active children"):
        composite.get_start_time(solution)

    with pytest.raises(ValueError, match="no active children"):
        composite.get_end_time(solution)


def test_composite_entity_empty_raises():
    """Test that empty CompositeEntity raises ValueError."""
    with pytest.raises(ValueError, match="must contain at least one entity"):
        CompositeEntity(frozenset())


def test_composite_entity_deterministic_id():
    """Test CompositeEntity without explicit ID has deterministic entity_id."""
    task1 = MockTask(1)
    task2 = MockTask(2)

    e1 = TaskEntity(task1)
    e2 = TaskEntity(task2)

    composite1 = CompositeEntity(frozenset({e1, e2}))
    composite2 = CompositeEntity(frozenset({e2, e1}))

    # Should have same ID (order-independent)
    assert composite1.entity_id == composite2.entity_id
    assert composite1 == composite2


def test_composite_entity_deep_hierarchy():
    """Test deeply nested hierarchy (3+ levels)."""
    tasks = [MockTask(i) for i in range(8)]

    # Level 1: Individual tasks
    task_entities = [TaskEntity(t) for t in tasks]

    # Level 2: Pairs
    pairs = [
        CompositeEntity(
            frozenset({task_entities[i], task_entities[i + 1]}), f"pair{i // 2}"
        )
        for i in range(0, 8, 2)
    ]

    # Level 3: Quads
    quads = [
        CompositeEntity(frozenset({pairs[i], pairs[i + 1]}), f"quad{i // 2}")
        for i in range(0, 4, 2)
    ]

    # Level 4: Octet
    octet = CompositeEntity(frozenset(quads), "octet")

    solution = MockSchedulingSolution(
        {tasks[i]: (i * 10, (i + 1) * 10) for i in range(8)}
    )

    # Should recursively collect all tasks
    assert octet.get_tasks() == frozenset(tasks)
    assert octet.get_start_time(solution) == 0
    assert octet.get_end_time(solution) == 80
    assert octet.is_active(solution)


# === Entity Equality and Hashing ===


def test_entity_equality_cross_type():
    """Test that different entity types are not equal."""
    task = MockTask(1)

    task_entity = TaskEntity(task)
    group_entity = GroupEntity(frozenset({task}))
    mode_entity = TaskModeEntity(task, mode=1)

    # All represent the same task, but are different entities
    assert task_entity != group_entity
    assert task_entity != mode_entity
    assert group_entity != mode_entity


def test_composite_entity_hashable():
    """Test that CompositeEntity can be used as dict key."""
    task1 = MockTask(1)
    task2 = MockTask(2)

    e1 = TaskEntity(task1)
    e2 = TaskEntity(task2)

    composite1 = CompositeEntity(frozenset({e1, e2}), "comp")
    composite2 = CompositeEntity(frozenset({e1, e2}), "comp")
    composite3 = CompositeEntity(frozenset({e1, e2}), "other")

    # Same ID -> equal
    assert composite1 == composite2
    assert hash(composite1) == hash(composite2)

    # Different ID -> not equal
    assert composite1 != composite3

    # Can use as dict key
    d = {composite1: "value1", composite3: "value3"}
    assert d[composite2] == "value1"


# === Real-world Scenario Tests ===


def test_hierarchical_project_structure():
    """Test realistic project → phase → task hierarchy."""
    # Project with 3 phases, each with 2 tasks
    tasks = {
        f"phase{p}_task{t}": MockTask(f"{p}_{t}") for p in range(3) for t in range(2)
    }

    # Build hierarchy
    phases = []
    for p in range(3):
        phase_tasks = [TaskEntity(tasks[f"phase{p}_task{t}"]) for t in range(2)]
        phase = CompositeEntity(frozenset(phase_tasks), composite_id=f"phase{p}")
        phases.append(phase)

    project = CompositeEntity(frozenset(phases), composite_id="project")

    # Create schedule where phases are sequential
    solution = MockSchedulingSolution(
        {
            tasks["phase0_task0"]: (0, 10),
            tasks["phase0_task1"]: (5, 15),
            tasks["phase1_task0"]: (15, 25),
            tasks["phase1_task1"]: (20, 30),
            tasks["phase2_task0"]: (30, 40),
            tasks["phase2_task1"]: (35, 45),
        }
    )

    # Verify project spans entire timeline
    assert project.get_start_time(solution) == 0
    assert project.get_end_time(solution) == 45

    # Verify each phase
    assert phases[0].get_start_time(solution) == 0
    assert phases[0].get_end_time(solution) == 15
    assert phases[1].get_start_time(solution) == 15
    assert phases[1].get_end_time(solution) == 30
    assert phases[2].get_start_time(solution) == 30
    assert phases[2].get_end_time(solution) == 45

    # All tasks should be collected
    assert len(project.get_tasks()) == 6
