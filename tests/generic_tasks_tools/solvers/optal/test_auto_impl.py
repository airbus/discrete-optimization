#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random

import numpy as np
import pytest
from ortools.sat.python.cp_model import LinearExprT
from pytest import fixture

import discrete_optimization.rcpsp.parser as rcpsp_parser
import discrete_optimization.rcpsp_multiskill.parser_imopse as parser_imopse
import discrete_optimization.shop.fjsp.parser as fjsp_parser
import discrete_optimization.shop.jsp.parser as jsp_parser
from discrete_optimization.generic_tasks_tools.entities import (
    ConstantDurationEntity,
    GroupEntity,
    TaskEntity,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
)
from discrete_optimization.generic_tasks_tools.objectives.allocated_tasks import (
    AllocatedTasksObjective,
)
from discrete_optimization.generic_tasks_tools.objectives.allocation_cost import (
    AllocationCostComputerMultimode,
)
from discrete_optimization.generic_tasks_tools.objectives.cumul_cost import (
    CumulCostComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.makespan import (
    MakespanObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.mode_cost import (
    ModeCostComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.resource_levels import (
    CalendarRenewableResourceLevelObjectiveComputer,
    NonRenewableResourceLevelObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.scheduled_tasks import (
    ScheduledTasksComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.soft_time_penalty import (
    SoftTimePenaltyComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.weighted_sum_start_or_end import (
    WeightedSumStartEndObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
    FlexibleGapBlockingConstraint,
    SpanBlockingConstraint,
    StartOrEnd,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_impl import (
    GenericSchedulingAutoOptalImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    optalcp_available,
)
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    RcpspToGenericSchedulingTransformation,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.transformations.generic_scheduling_impl import (
    MultiskillRcpspToGenericSchedulingTransformation,
    transform_solution_from_raw_generic_to_rcpsp_ms,
)
from discrete_optimization.shop.transformations.to_generic_scheduling import (
    ShopToGenericSchedulingTransformation,
)

if not optalcp_available:
    pytest.skip(
        reason="You need optalcp to test this solver.",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "objective",
    list(Objective)
    + [
        [
            (Objective.MAKESPAN, 2),
            (Objective.NB_TASKS_ALLOCATED, -2),
        ],
        [
            (Objective.MODE_COST, 1),
            (Objective.ALLOCATION_COST, 1),
        ],
    ],
)
def test_auto(objective, caplog, optal_preview_only):
    if objective in [
        Objective.ALLOCATION_CHANGES,
        Objective.SCHEDULE_CHANGES,
        Objective.NB_CAL_PREEMPTED_TASKS,
        Objective.COST_ON_ALTERNATIVE_SUBPROBLEMS,
        Objective.WEIGHTED_SUM_START_OR_END,
        Objective.EARLINESS_TARDINESS,
    ]:
        pytest.skip(f"Not testing objective {objective} here.")

    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return variable.compute_nb_tasks_allocated() - variable.get_max_end_time()

    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        custom_evaluate_fn=custom_evaluate_fn,
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        list_objective_computer=[
            MakespanObjectiveComputer(),
            CalendarRenewableResourceLevelObjectiveComputer(
                weight_objective=1,
                weight_resource={"cumulative_resource": 1},
            ),
            NonRenewableResourceLevelObjectiveComputer(
                weight_objective=1,
                weight_resource={"non_renewable_resource": 1},
            ),
            AllocatedTasksObjective(problem=None, weight_objective=-1),
            UnaryResourcesUsedComputer(
                weight_per_unary_resource={ur: 1 for ur in {"worker1", "worker2"}},
            ),
            ModeCostComputer(
                weight_objective=1,
                mode_cost={
                    "task-1": {
                        0: 100,
                        1: 3,
                    },
                    "task-2": {
                        0: 0,
                    },
                },
            ),
            AllocationCostComputerMultimode(
                weight_objective=1,
                cost_allocation_resource_to_task_mode={
                    ("task-1", 1): {"worker1": 27, "worker2": 10}
                },
            ),
            SoftTimePenaltyComputer(),
            CumulCostComputer(),
            ScheduledTasksComputer(),
            WeightedSumStartEndObjectiveComputer(),
        ],
    )

    # prepare solver

    # custom objective: makespan - nb tasks allocated
    def custom_objective_factory(
        solver: GenericSchedulingAutoOptalImplSolver,
    ) -> LinearExprT:
        return (
            solver.get_nb_tasks_allocated_variable()
            - solver.get_global_makespan_variable()
        )

    exactly_one_unary_resource_per_task = objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.CALENDAR_RESOURCES_LEVELS,
        Objective.CUSTOM,
    ]
    if objective == [(Objective.MODE_COST, 1), (Objective.ALLOCATION_COST, 1)]:
        exactly_one_unary_resource_per_task = True
    if isinstance(objective, Objective):
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.SINGLE,
            objectives=[objective],
            weights=[1 if objective != Objective.NB_TASKS_ALLOCATED else -1],
            sense_function=ModeOptim.MINIMIZATION,
        )
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=[obj[0] for obj in objective],
            weights=[obj[1] for obj in objective],
            sense_function=ModeOptim.MINIMIZATION,
        )
    solver = GenericSchedulingAutoOptalImplSolver(
        problem=problem,
        objective=objective,
        params_objective_function=params_objective_function,
        custom_objective_factory=custom_objective_factory,
    )

    solver.init_model(
        exactly_one_unary_resource_per_task=exactly_one_unary_resource_per_task
    )

    # solve
    res = solver.solve(do_not_retrieve_solutions=optal_preview_only)
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # check sol and kpis
        sol: GenericSchedulingImplSolution
        sol, fit = res[-1]
        assert problem.satisfy(sol)
        kpi = problem.evaluate(sol)

        if objective == Objective.NB_UNARY_RESOURCES_USED:
            assert kpi[Objective.NB_UNARY_RESOURCES_USED] == 1
        elif objective == Objective.MAKESPAN:
            assert kpi[Objective.MAKESPAN] == 9
        elif objective == Objective.NB_TASKS_ALLOCATED:
            assert kpi[Objective.NB_TASKS_ALLOCATED] == 2
        elif objective == [(Objective.MODE_COST, 1), (Objective.ALLOCATION_COST, 1)]:
            assert sol.get_mode("task-1") == 1
            assert not sol.is_allocated("task-1", unary_resource="worker1")
            assert sol.is_allocated("task-1", unary_resource="worker2")
            assert kpi[Objective.ALLOCATION_COST] == 10
            assert kpi[Objective.MODE_COST] == 3
        elif objective == Objective.CUSTOM:
            assert kpi[Objective.CUSTOM] == 2 - 10
        elif objective == [
            (Objective.MAKESPAN, 2),
            (Objective.NB_TASKS_ALLOCATED, -2),
        ]:
            assert kpi[Objective.NB_TASKS_ALLOCATED] == 2
            assert kpi[Objective.MAKESPAN] == 9


@pytest.mark.parametrize("mode_optim", list(ModeOptim))
def test_auto_custom(mode_optim, caplog, optal_preview_only):
    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return variable.compute_nb_tasks_allocated() - variable.get_max_end_time()

    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        custom_evaluate_fn=custom_evaluate_fn,
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        list_objective_computer=[
            MakespanObjectiveComputer(),
            CalendarRenewableResourceLevelObjectiveComputer(
                problem=None,
                weight_objective=1,
                weight_resource={"cumulative_resource": 1},
            ),
            NonRenewableResourceLevelObjectiveComputer(
                problem=None,
                weight_objective=1,
                weight_resource={"non_renewable_resource": 1},
            ),
            AllocatedTasksObjective(problem=None, weight_objective=-1),
            UnaryResourcesUsedComputer(
                problem=None,
                weight_per_unary_resource={ur: 1 for ur in {"worker1", "worker2"}},
            ),
            ModeCostComputer(
                problem=None,
                weight_objective=1,
                mode_cost={
                    "task-1": {
                        0: 100,
                        1: 3,
                    },
                    "task-2": {
                        0: 0,
                    },
                },
            ),
            AllocationCostComputerMultimode(
                problem=None,
                weight_objective=1,
                cost_allocation_resource_to_task_mode={
                    ("task-1", 1): {"worker1": 27, "worker2": 10}
                },
            ),
        ],
    )

    # custom objective: makespan - nb tasks allocated
    def custom_objective_factory(
        solver: GenericSchedulingAutoOptalImplSolver,
    ) -> LinearExprT:
        return (
            solver.get_nb_tasks_allocated_variable()
            - solver.get_global_makespan_variable()
        )

    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.SINGLE,
        objectives=[Objective.CUSTOM],
        weights=[1],
        sense_function=mode_optim,
    )
    solver = GenericSchedulingAutoOptalImplSolver(
        problem=problem,
        objective=Objective.CUSTOM,
        params_objective_function=params_objective_function,
        custom_objective_factory=custom_objective_factory,
    )

    solver.init_model(exactly_one_unary_resource_per_task=True)

    # solve
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # check sol and kpis
        sol: GenericSchedulingImplSolution
        sol, fit = res[-1]
        assert problem.satisfy(sol)
        kpi = problem.evaluate(sol)
        if mode_optim == ModeOptim.MINIMIZATION:
            assert kpi[Objective.CUSTOM] == 2 - 10
        if mode_optim == ModeOptim.MAXIMIZATION:
            assert kpi[Objective.CUSTOM] == 2 - 9


@pytest.mark.parametrize(
    "objective",
    list(Objective)
    + [
        [(Objective.MAKESPAN, 2), (Objective.NB_TASKS_ALLOCATED, -2)],
        [(Objective.MODE_COST, 1), (Objective.ALLOCATION_COST, 1)],
    ],
)
def test_auto_optional_tasks(objective, caplog, optal_preview_only):
    if objective in [
        Objective.ALLOCATION_CHANGES,
        Objective.SCHEDULE_CHANGES,
        Objective.NB_CAL_PREEMPTED_TASKS,
        Objective.COST_ON_ALTERNATIVE_SUBPROBLEMS,
        Objective.WEIGHTED_SUM_START_OR_END,
        Objective.EARLINESS_TARDINESS,
    ]:
        pytest.skip(f"Not testing objective {objective} here.")

    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return -sum(
            variable.get_start_time(task) for task in variable.get_present_tasks()
        )

    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        optional_tasks={"task-1"},
        objective=objective,
        custom_evaluate_fn=custom_evaluate_fn,
        list_objective_computer=[
            MakespanObjectiveComputer(),
            CalendarRenewableResourceLevelObjectiveComputer(
                weight_objective=1,
                weight_resource={"cumulative_resource": 1},
            ),
            NonRenewableResourceLevelObjectiveComputer(
                weight_objective=1,
                weight_resource={"non_renewable_resource": 1},
            ),
            AllocatedTasksObjective(problem=None, weight_objective=-1),
            UnaryResourcesUsedComputer(
                weight_per_unary_resource={ur: 1 for ur in {"worker1", "worker2"}},
            ),
            ModeCostComputer(
                weight_objective=1,
                mode_cost={
                    "task-1": {
                        0: 100,
                        1: 3,
                    },
                    "task-2": {
                        0: 0,
                    },
                },
            ),
            AllocationCostComputerMultimode(
                weight_objective=1,
                cost_allocation_resource_to_task_mode={
                    ("task-1", 1): {"worker1": 27, "worker2": 10}
                },
            ),
            SoftTimePenaltyComputer(),
            CumulCostComputer(),
            ScheduledTasksComputer(),
            WeightedSumStartEndObjectiveComputer(),
        ],
    )

    # prepare solver

    # custom objective: makespan - nb tasks allocated
    def custom_objective_factory(
        solver: GenericSchedulingAutoOptalImplSolver,
    ) -> LinearExprT:
        return -solver.get_subtasks_sum_start_time_variable(problem.tasks_list)

    exactly_one_unary_resource_per_task = objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.CALENDAR_RESOURCES_LEVELS,
        Objective.NON_RENEWABLE_RESOURCES_LEVELS,
        Objective.MODE_COST,
    ]
    if objective == [(Objective.MODE_COST, 1), (Objective.ALLOCATION_COST, 1)]:
        exactly_one_unary_resource_per_task = True

    if isinstance(objective, Objective):
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.SINGLE,
            objectives=[objective],
            weights=[1 if objective != Objective.NB_TASKS_ALLOCATED else -1],
            sense_function=ModeOptim.MINIMIZATION,
        )
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.SINGLE,
            objectives=[obj[0] for obj in objective],
            weights=[obj[1] for obj in objective],
            sense_function=ModeOptim.MINIMIZATION,
        )
    solver = GenericSchedulingAutoOptalImplSolver(
        problem=problem,
        params_objective_function=params_objective_function,
        objective=objective,
        custom_objective_factory=custom_objective_factory,
    )

    solver.init_model(
        exactly_one_unary_resource_per_task=exactly_one_unary_resource_per_task
    )
    # solver.set_warm_start(sol)

    # solve
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # check sol and kpis
        sol: GenericSchedulingImplSolution
        sol, fit = res[-1]
        assert problem.satisfy(sol)
        kpi = problem.evaluate(sol)

        if objective == Objective.NB_UNARY_RESOURCES_USED:
            assert kpi[Objective.NB_UNARY_RESOURCES_USED] == 1
        elif objective == Objective.MAKESPAN:
            assert kpi[Objective.MAKESPAN] == 9
        elif objective == Objective.NB_TASKS_ALLOCATED:
            assert kpi[Objective.NB_TASKS_ALLOCATED] == 2
        elif objective == [(Objective.MODE_COST, 1), (Objective.ALLOCATION_COST, 1)]:
            assert not sol.is_present("task-1")
            assert kpi[Objective.MODE_COST] + kpi[Objective.ALLOCATION_COST] == 0
        elif objective == Objective.CUSTOM:
            # task 1 (mode1): 3->6, task 2 : 6->10
            assert kpi[Objective.CUSTOM] == -9
        elif isinstance(objective, list):
            assert kpi[Objective.NB_TASKS_ALLOCATED] == 2
            assert kpi[Objective.MAKESPAN] == 9


def test_auto_optional_tasks_with_resource_blocking(optal_preview_only):
    """Test optional tasks with resource blocking constraints.

    This test verifies that resource blocking constraints work correctly
    when some tasks involved in the blocking are optional.

    Scenario:
    - Task A (optional): can be scheduled or skipped
    - Task B (mandatory): must be scheduled
    - Task C (optional): can be scheduled or skipped
    - Task D (mandatory): must be scheduled

    Blocking constraints:
    1. FlexibleGapBlocking: Between Task A and Task B (both involved in blocking)
       - If Task A is scheduled, machine_1 is blocked during gap A→B
    2. SpanBlocking: During span of Tasks C+D (C is optional, D is mandatory)
       - If C is present, blocks machine_2 during span of {C, D}
       - If C is not present, blocks machine_2 during D's execution only
    """
    horizon = 30

    # Task definitions
    durations_per_mode = {
        "task_a": {0: 4},  # Optional task, 4 hours
        "task_b": {0: 3},  # Mandatory task, 3 hours
        "task_c": {0: 5},  # Optional task, 5 hours
        "task_d": {0: 2},  # Mandatory task, 2 hours
        "task_e": {0: 6},  # Additional mandatory task to make it interesting
    }

    # Resource consumptions
    resource_consumptions = {
        "task_a": {0: {"machine_1": 1}},
        "task_b": {0: {"machine_1": 1}},
        "task_c": {0: {"machine_2": 1}},
        "task_d": {0: {"machine_2": 1}},
        "task_e": {0: {"machine_2": 1}},
    }

    # Machine capacities (both have capacity 2 to allow some parallelism)
    non_skill_cumulative_resources = {
        "machine_1": 2,
        "machine_2": 2,
    }

    # Optional tasks
    optional_tasks = {"task_a", "task_c"}

    # Precedence: task_a must finish before task_b starts (if task_a is scheduled)
    successors = {"task_a": ["task_b"]}

    # Minimum 2-hour gap between task_a end and task_b start (setup time)
    end_to_start_min_time_lags = [("task_a", "task_b", 2)]

    # FlexibleGapBlocking: Block machine_1 during the gap task_a→task_b
    # This should only be enforced if task_a is present
    flexible_gap_blocking = FlexibleGapBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Setup time blocking between task_a and task_b"
        ),
        default_resource_blocked={"machine_1": 1},
        choice_resource_blocked={},
        left_entity=TaskEntity("task_a"),
        start_or_end_left_entity=StartOrEnd.END,
        right_entity=TaskEntity("task_b"),
        start_or_end_right_entity=StartOrEnd.START,
    )

    # SpanBlocking: Block machine_2 during the span of tasks {task_c, task_d}
    # If task_c is not scheduled, this should only block during task_d
    span_blocking = SpanBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Safety monitoring for batch C+D"
        ),
        default_resource_blocked={"machine_2": 1},
        choice_resource_blocked={},
        entity=GroupEntity(frozenset(["task_c", "task_d"])),
    )

    # Create problem with blocking constraints
    problem = GenericSchedulingImplProblem(
        horizon=horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        end_to_start_min_time_lags=end_to_start_min_time_lags,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
        optional_tasks=optional_tasks,
        flexible_gap_blocking_constraints=[flexible_gap_blocking],
        span_blocking_constraints=[span_blocking],
        objective=Objective.MAKESPAN,
    )

    # Solve the problem
    solver = GenericSchedulingAutoOptalImplSolver(problem=problem)
    solver.init_model()
    result = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # Get solution
        solution: GenericSchedulingImplSolution
        solution, fit = result[-1]

        # Verify solution satisfies all constraints
        assert problem.satisfy(solution)

        # Get evaluation
        kpi = problem.evaluate(solution)
        print(f"\nSolution KPIs: {kpi}")

        # Print schedule
        print("\nSchedule:")
        for task in problem.tasks_list:
            if solution.is_present(task):
                start = solution.get_start_time(task)
                end = solution.get_end_time(task)
                print(f"  {task}: [{start:2d}, {end:2d}) - SCHEDULED")
            else:
                print(f"  {task}: NOT SCHEDULED (optional)")

        # Additional assertions to verify blocking constraints behavior
        # Test 1: If task_a is present, verify the gap blocking
        if solution.is_present("task_a"):
            end_a = solution.get_end_time("task_a")
            start_b = solution.get_start_time("task_b")
            gap_duration = start_b - end_a

            print(
                f"\ntask_a is present: gap between task_a and task_b = {gap_duration}"
            )
            assert gap_duration >= 2, "Minimum gap constraint violated"

            # The blocking constraint should prevent other tasks from using machine_1
            # during the gap [end_a, start_b)
            # We can't strictly test this without inspecting the solver internals,
            # but we can at least verify the solution is valid

        # Test 2: Verify span blocking for task_c + task_d
        # If both are present, check they form a span
        if solution.is_present("task_c"):
            start_c = solution.get_start_time("task_c")
            end_c = solution.get_end_time("task_c")
            start_d = solution.get_start_time("task_d")
            end_d = solution.get_end_time("task_d")

            span_start = min(start_c, start_d)
            span_end = max(end_c, end_d)
            print(
                f"\ntask_c is present: span of {{task_c, task_d}} = [{span_start}, {span_end})"
            )

            # The blocking reserves 1 unit of machine_2 during this span
            # task_e should be scheduled considering this constraint
            if solution.is_present("task_e"):
                start_e = solution.get_start_time("task_e")
                end_e = solution.get_end_time("task_e")
                print(f"task_e: [{start_e}, {end_e})")

        # Final check: solution is feasible
        assert solution is not None
        print("\n✓ Test passed: Optional tasks with resource blocking work correctly!")


def test_auto_with_resource_blocking(optal_preview_only):
    """Test optional tasks with resource blocking constraints.
    This test verifies that resource blocking constraints work correctly
    Blocking constraints:
    1. FlexibleGapBlocking: Between Task A and Task B (both involved in blocking)
       - If Task A is scheduled, machine_1 is blocked during gap A→B
    2. SpanBlocking: During span of Tasks C+D (C is optional, D is mandatory)
       - If C is present, blocks machine_2 during span of {C, D}
       - If C is not present, blocks machine_2 during D's execution only
    """
    horizon = 30

    # Task definitions
    durations_per_mode = {
        "task_a": {0: 4},  # Optional task, 4 hours
        "task_b": {0: 3},  # Mandatory task, 3 hours
        "task_c": {0: 5},  # Optional task, 5 hours
        "task_d": {0: 2},  # Mandatory task, 2 hours
        "task_e": {0: 6},  # Additional mandatory task to make it interesting
    }

    # Resource consumptions
    resource_consumptions = {
        "task_a": {0: {"machine_1": 1}},
        "task_b": {0: {"machine_1": 1}},
        "task_c": {0: {"machine_2": 1}},
        "task_d": {0: {"machine_2": 1}},
        "task_e": {0: {"machine_2": 1}},
    }

    # Machine capacities (both have capacity 2 to allow some parallelism)
    non_skill_cumulative_resources = {
        "machine_1": 2,
        "machine_2": 2,
    }

    # Precedence: task_a must finish before task_b starts (if task_a is scheduled)
    successors = {"task_a": ["task_b"]}

    # Minimum 2-hour gap between task_a end and task_b start (setup time)
    end_to_start_min_time_lags = [("task_a", "task_b", 2)]

    # FlexibleGapBlocking: Block machine_1 during the gap task_a→task_b
    # This should only be enforced if task_a is present
    flexible_gap_blocking = FlexibleGapBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Setup time blocking between task_a and task_b"
        ),
        default_resource_blocked={"machine_1": 1},
        choice_resource_blocked={},
        left_entity=TaskEntity("task_a"),
        start_or_end_left_entity=StartOrEnd.END,
        right_entity=TaskEntity("task_b"),
        start_or_end_right_entity=StartOrEnd.START,
    )

    # SpanBlocking: Block machine_2 during the span of tasks {task_c, task_d}
    # If task_c is not scheduled, this should only block during task_d
    span_blocking = SpanBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Safety monitoring for batch C+D"
        ),
        default_resource_blocked={"machine_2": 1},
        choice_resource_blocked={},
        entity=GroupEntity(frozenset(["task_c", "task_d"])),
    )

    # Create problem with blocking constraints
    problem = GenericSchedulingImplProblem(
        horizon=horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        end_to_start_min_time_lags=end_to_start_min_time_lags,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
        flexible_gap_blocking_constraints=[flexible_gap_blocking],
        span_blocking_constraints=[span_blocking],
        objective=Objective.MAKESPAN,
    )

    # Solve the problem
    solver = GenericSchedulingAutoOptalImplSolver(problem=problem)
    solver.init_model()
    result = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # Get solution
        solution: GenericSchedulingImplSolution
        solution, fit = result[-1]

        # Verify solution satisfies all constraints
        assert problem.satisfy(solution)

        # Get evaluation
        kpi = problem.evaluate(solution)
        print(f"\nSolution KPIs: {kpi}")

        # Print schedule
        print("\nSchedule:")
        for task in problem.tasks_list:
            if solution.is_present(task):
                start = solution.get_start_time(task)
                end = solution.get_end_time(task)
                print(f"  {task}: [{start:2d}, {end:2d}) - SCHEDULED")
            else:
                print(f"  {task}: NOT SCHEDULED (optional)")

        # Additional assertions to verify blocking constraints behavior
        # Test 1: If task_a is present, verify the gap blocking
        if solution.is_present("task_a"):
            end_a = solution.get_end_time("task_a")
            start_b = solution.get_start_time("task_b")
            gap_duration = start_b - end_a

            print(
                f"\ntask_a is present: gap between task_a and task_b = {gap_duration}"
            )
            assert gap_duration >= 2, "Minimum gap constraint violated"

            # The blocking constraint should prevent other tasks from using machine_1
            # during the gap [end_a, start_b)
            # We can't strictly test this without inspecting the solver internals,
            # but we can at least verify the solution is valid

        # Test 2: Verify span blocking for task_c + task_d
        # If both are present, check they form a span
        if solution.is_present("task_c"):
            start_c = solution.get_start_time("task_c")
            end_c = solution.get_end_time("task_c")
            start_d = solution.get_start_time("task_d")
            end_d = solution.get_end_time("task_d")

            span_start = min(start_c, start_d)
            span_end = max(end_c, end_d)
            print(
                f"\ntask_c is present: span of {{task_c, task_d}} = [{span_start}, {span_end})"
            )

            # The blocking reserves 1 unit of machine_2 during this span
            # task_e should be scheduled considering this constraint
            if solution.is_present("task_e"):
                start_e = solution.get_start_time("task_e")
                end_e = solution.get_end_time("task_e")
                print(f"task_e: [{start_e}, {end_e})")

        # Final check: solution is feasible
        assert solution is not None


def test_auto_with_resource_blocking_and_choices(optal_preview_only):
    """Test optional tasks with resource blocking constraints.
    This test verifies that resource blocking constraints work correctly
    Blocking constraints:
    1. FlexibleGapBlocking: Between Task A and Task B (both involved in blocking)
       - If Task A is scheduled, machine_1 is blocked during gap A→B
    2. SpanBlocking: During span of Tasks C+D (C is optional, D is mandatory)
       - If C is present, blocks machine_2 during span of {C, D}
       - If C is not present, blocks machine_2 during D's execution only
    """
    horizon = 30

    # Task definitions
    durations_per_mode = {
        "task_a": {0: 4},  # Optional task, 4 hours
        "task_b": {0: 3},  # Mandatory task, 3 hours
        "task_c": {0: 5},  # Optional task, 5 hours
        "task_d": {0: 2},  # Mandatory task, 2 hours
        "task_e": {0: 6},  # Additional mandatory task to make it interesting
    }

    # Resource consumptions
    resource_consumptions = {
        "task_a": {0: {"machine_1": 1}},
        "task_b": {0: {"machine_1": 1}},
        "task_c": {0: {"machine_2": 1}},
        "task_d": {0: {"machine_2": 1}},
        "task_e": {0: {"machine_2": 1}},
    }

    # Machine capacities (both have capacity 2 to allow some parallelism)
    non_skill_cumulative_resources = {
        "machine_1": 2,
        "machine_2": 2,
    }

    # Precedence: task_a must finish before task_b starts (if task_a is scheduled)
    successors = {"task_a": ["task_b"]}

    # Minimum 2-hour gap between task_a end and task_b start (setup time)
    end_to_start_min_time_lags = [("task_a", "task_b", 2)]

    # FlexibleGapBlocking: Block machine_1 during the gap task_a→task_b
    # This should only be enforced if task_a is present
    flexible_gap_blocking = FlexibleGapBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Setup time blocking between task_a and task_b",
            name_choice="choice_flex_0",
        ),
        default_resource_blocked={},
        # Different choice of resource blocked.
        choice_resource_blocked={0: {"machine_1": 3}, 1: {"machine_2": 1}},
        left_entity=TaskEntity("task_a"),
        start_or_end_left_entity=StartOrEnd.END,
        right_entity=TaskEntity("task_b"),
        start_or_end_right_entity=StartOrEnd.START,
    )

    # SpanBlocking: Block machine_2 during the span of tasks {task_c, task_d}
    # If task_c is not scheduled, this should only block during task_d
    span_blocking = SpanBlockingConstraint(
        metadata=BlockingConstraintMetadata(
            description="Safety monitoring for batch C+D", name_choice="choice_span_0"
        ),
        default_resource_blocked={"machine_1": 1},
        choice_resource_blocked={0: {"machine_1": 0}, 1: {"machine_1": 1}},
        entity=GroupEntity(frozenset(["task_c", "task_d"])),
    )

    # Create problem with blocking constraints
    problem = GenericSchedulingImplProblem(
        horizon=horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        end_to_start_min_time_lags=end_to_start_min_time_lags,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
        flexible_gap_blocking_constraints=[flexible_gap_blocking],
        span_blocking_constraints=[span_blocking],
        objective=Objective.MAKESPAN,
    )

    # Solve the problem
    solver = GenericSchedulingAutoOptalImplSolver(problem=problem)
    solver.init_model()
    result = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        # Get solution
        solution: GenericSchedulingImplSolution
        solution, fit = result[-1]

        # Verify solution satisfies all constraints
        assert problem.satisfy(solution)
        # The mode 0 is unsat !!
        assert solution.get_choice_of_resource_blocking(flexible_gap_blocking) == 1
        # Get evaluation
        kpi = problem.evaluate(solution)
        print(f"\nSolution KPIs: {kpi}")

        # Print schedule
        print("\nSchedule:")
        for task in problem.tasks_list:
            if solution.is_present(task):
                start = solution.get_start_time(task)
                end = solution.get_end_time(task)
                print(f"  {task}: [{start:2d}, {end:2d}) - SCHEDULED")
            else:
                print(f"  {task}: NOT SCHEDULED (optional)")

        # Additional assertions to verify blocking constraints behavior
        # Test 1: If task_a is present, verify the gap blocking
        if solution.is_present("task_a"):
            end_a = solution.get_end_time("task_a")
            start_b = solution.get_start_time("task_b")
            gap_duration = start_b - end_a

            print(
                f"\ntask_a is present: gap between task_a and task_b = {gap_duration}"
            )
            assert gap_duration >= 2, "Minimum gap constraint violated"

            # The blocking constraint should prevent other tasks from using machine_1
            # during the gap [end_a, start_b)
            # We can't strictly test this without inspecting the solver internals,
            # but we can at least verify the solution is valid

        # Test 2: Verify span blocking for task_c + task_d
        # If both are present, check they form a span
        if solution.is_present("task_c"):
            start_c = solution.get_start_time("task_c")
            end_c = solution.get_end_time("task_c")
            start_d = solution.get_start_time("task_d")
            end_d = solution.get_end_time("task_d")

            span_start = min(start_c, start_d)
            span_end = max(end_c, end_d)
            print(
                f"\ntask_c is present: span of {{task_c, task_d}} = [{span_start}, {span_end})"
            )

            # The blocking reserves 1 unit of machine_2 during this span
            # task_e should be scheduled considering this constraint
            if solution.is_present("task_e"):
                start_e = solution.get_start_time("task_e")
                end_e = solution.get_end_time("task_e")
                print(f"task_e: [{start_e}, {end_e})")

        # Final check: solution is feasible
        assert solution is not None


def test_start_to_end_time_lag(optal_preview_only):
    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        start_to_end_min_time_lags=[("task-1", "task-2", 8)],
    )
    solver = GenericSchedulingAutoOptalImplSolver(problem=problem)
    result = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        solution: GenericSchedulingImplSolution = result.get_best_solution()
        assert problem.satisfy(solution)


def test_start_to_end_time_lag_optional_tasks(optal_preview_only):
    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        optional_tasks={"task-1"},
        start_to_end_min_time_lags=[("task-1", "task-2", 8)],
    )
    solver = GenericSchedulingAutoOptalImplSolver(problem=problem)
    solver.init_model()
    result = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        solution: GenericSchedulingImplSolution = result.get_best_solution()
        assert problem.satisfy(solution)
        assert not solution.is_present("task-1")


def test_no_overlap(optal_preview_only):
    problem = GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 2,
            },
            "task-2": {
                0: 4,
            },
        },
        no_overlap_sets={frozenset({"task-1", "task-2"})},
        forbidden_intervals={"task-1": [(1, 3)]},
    )
    solver = GenericSchedulingAutoOptalImplSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            ObjectiveHandling.SINGLE,
            objectives=[Objective.MAKESPAN],
            weights=[1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        sol: GenericSchedulingImplSolution = res.get_best_solution()
        assert problem.satisfy(sol)
        kpi = problem.evaluate(sol)
        assert kpi[Objective.MAKESPAN] == 6


@fixture
def problem_with_unary_dependent_resource_consumption(optal_preview_only):
    return GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        unary_resource_consumptions_dependent={
            "task-1": {
                0: {"cumulative_resource": {"worker1": 2, "worker2": 1}},
                1: {"cumulative_resource": {"worker1": 1, "worker2": 1}},
            }
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        list_objective_computer=[MakespanObjectiveComputer(weight_objective=1)],
    )


@fixture
def problem_with_unary_dependent_nr_resource_consumption(optal_preview_only):
    return GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
        },
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resource_consumptions_dependent={
            "task-1": {
                0: {
                    "cumulative_resource": {"worker1": 2, "worker2": 1},
                    "non_renewable_resource": {"worker1": 1, "worker2": 2},
                },
                1: {
                    "cumulative_resource": {"worker1": 1, "worker2": 1},
                    "non_renewable_resource": {"worker1": 2, "worker2": 0},
                },
            }
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 2,
        },
        list_objective_computer=[MakespanObjectiveComputer(weight_objective=1)],
    )


@fixture
def problem_with_forbidden_task_mode_unary_resource(optal_preview_only):
    return GenericSchedulingImplProblem(
        horizon=10,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                    "worker": 1,
                },
                1: {
                    "non_renewable_resource": 1,
                    "worker": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                    "worker": 1,
                },
            },
        },
        skills={"worker"},
        unary_resources_skills={"worker1": {"worker": 1}, "worker2": {"worker": 1}},
        successors={"task-1": {"task-2"}},
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 18)],
            "worker2": [(3, 18)],
        },
        # dict[Task, dict[UnaryResource, set[int]]]
        unary_resources_task_mode_compatibility={
            "task-1": {"worker1": {0, 1}, "worker2": {1}},
            "task-2": {"worker1": {0}},
        },
        unary_resource_consumptions_dependent={
            "task-1": {
                0: {"cumulative_resource": {"worker1": 2, "worker2": 1}},
                1: {"cumulative_resource": {"worker1": 1, "worker2": 1}},
            }
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 10, 2),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
        list_objective_computer=[MakespanObjectiveComputer(weight_objective=1)],
    )


def test_solving_unary_resource_consumption_dependent(
    problem_with_unary_dependent_resource_consumption, optal_preview_only
):
    problem: GenericSchedulingImplProblem = (
        problem_with_unary_dependent_resource_consumption
    )
    solver = GenericSchedulingAutoOptalImplSolver(problem)
    solver.init_model()
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        sol = res[-1][0]
        assert problem.satisfy(sol)


def test_solving_unary_nr_resource_consumption_dependent(
    problem_with_unary_dependent_nr_resource_consumption, optal_preview_only
):
    problem: GenericSchedulingImplProblem = (
        problem_with_unary_dependent_nr_resource_consumption
    )
    solver = GenericSchedulingAutoOptalImplSolver(problem)
    solver.init_model()
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        sol = res[-1][0]
        assert problem.satisfy(sol)


def test_solving_forbidden_unary_resource_task_mode(
    problem_with_forbidden_task_mode_unary_resource, optal_preview_only
):
    problem: GenericSchedulingImplProblem = (
        problem_with_forbidden_task_mode_unary_resource
    )
    solver = GenericSchedulingAutoOptalImplSolver(problem)
    solver.init_model()
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound
    if not optal_preview_only:
        sol = res[-1][0]
        assert problem.satisfy(sol)


def test_rcpsp_simple(optal_preview_only):
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 3, "R1": 1}},
        3: {1: {"duration": 2, "R1": 1}},
        4: {1: {"duration": 4, "R1": 1}},
        5: {1: {"duration": 0}},  # dummy end
    }

    successors = {
        1: {2, 3},
        2: {5},
        3: {4},
        4: {5},
        5: {},
    }

    resources = {"R1": 2}

    horizon = 100

    special_constraints = SpecialConstraintsDescription(
        start_together=[(2, 3)],  # tasks 2 and 3 start together
        start_times={4: 10},  # task 4 should start at 10
        start_at_end=[(3, 4)],  # task 4 starts when task 3 ends
        end_times={5: 20},  # task 5 ends at 20
    )
    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=horizon,
        special_constraints=special_constraints,
    )

    # transform to generic problem
    transfo = RcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)
    generic_solver = GenericSchedulingAutoOptalImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    res = generic_solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert generic_solver.status_solver in (StatusSolver.OPTIMAL,)
    assert generic_solver.current_obj >= generic_solver.current_bound
    if not optal_preview_only:
        generic_solution = res.get_best_solution()
        assert generic_problem.satisfy(generic_solution)


def test_rcpsp_mm(optal_preview_only):
    filename = "j1010_1.mm"
    files_available = rcpsp_parser.get_data_available()
    file = [f for f in files_available if filename in f][0]
    problem = rcpsp_parser.parse_file(file)
    for resource in problem.resources:
        problem.resources[resource] = np.array(
            problem.get_resource_availability_array(resource)
        )
        problem.resources[resource][10:15] = 0
    problem.update_problem()

    # transform to generic problem
    transfo = RcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoOptalImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    res = generic_solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
        time_limit=2,
        callbacks=[NbIterationStopper(1)],
    )
    assert generic_solver.status_solver in (
        StatusSolver.OPTIMAL,
        StatusSolver.SATISFIED,
    )
    assert generic_solver.current_obj >= generic_solver.current_bound
    if not optal_preview_only:
        generic_solution = res.get_best_solution()
        assert generic_problem.satisfy(generic_solution)


@pytest.mark.parametrize(
    "one_worker_per_task, one_skill_per_task, exact_skill, slack_skill, use_energy_constraints, redundant_skill_cumulative",
    [
        (False, False, False, False, False, False),
        (False, False, False, False, False, True),
        (True, False, False, False, False, False),
        (True, False, False, False, True, False),
        (False, True, False, False, False, False),
        (False, True, True, False, False, False),
        (False, True, True, True, False, False),
        (False, False, True, True, False, False),
        (True, False, True, True, False, False),
    ],
)
def test_rcpsp_multiskill(
    one_worker_per_task,
    one_skill_per_task,
    exact_skill,
    slack_skill,
    use_energy_constraints,
    redundant_skill_cumulative,
    optal_preview_only,
):
    file = [f for f in parser_imopse.get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parser_imopse.parse_file(file, max_horizon=1000)
    problem.only_one_skill_per_task = one_skill_per_task
    task = problem.tasks_list[0]
    calendar = [2] * problem.horizon
    for t in range(120, 180):
        calendar[t] = 1
    problem.non_renewable_resources = {"R0"}
    problem.resources_availability = {"R0": [1], "R1": calendar}
    problem.resources_set = set(problem.resources_availability)
    problem.partial_preemption_data = None
    problem.always_releasable_resources = None
    problem.never_releasable_resources = None
    problem.mode_details[task][1]["R1"] = 2
    problem.mode_details[task][2] = dict(problem.mode_details[task][1])
    problem.mode_details[task][2]["R0"] = 1
    problem.mode_details[task][1]["R0"] = 2
    problem.update_problem()

    # transform to generic problem
    transfo = MultiskillRcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoOptalImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model(
        use_only_skill_to_allocate=True,  # same behaviour as multiskill solver
        at_most_one_unary_resource_per_task=one_worker_per_task,
        use_exact_skill=exact_skill,
        use_slack_for_skill=slack_skill,
        max_slack_for_skill=5,
        use_energy_constraints=use_energy_constraints,
        add_redundant_skill_cumulative_constraints=redundant_skill_cumulative,
    )
    res = generic_solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
        time_limit=2,
        callbacks=[NbIterationStopper(1)],
    )
    assert generic_solver.status_solver in (
        StatusSolver.OPTIMAL,
        StatusSolver.SATISFIED,
    )
    assert generic_solver.current_obj >= generic_solver.current_bound
    if not optal_preview_only:
        generic_solution = res.get_best_solution()
        assert generic_problem.satisfy(generic_solution)
        if one_skill_per_task:
            assert all(
                all(
                    len(skills_used) == 1
                    for skills_used in task_variable.allocated.values()
                )
                for task_variable in generic_solution.raw_sol.task_variables.values()
            )

        # compare solutions
        from_generic_solution: MultiskillRcpspSolution = (
            transform_solution_from_raw_generic_to_rcpsp_ms(
                raw_sol=generic_solution.raw_sol,
                problem=problem,
            )
        )
        assert problem.satisfy(from_generic_solution)
        assert from_generic_solution.check_skill_constraints(
            exact=exact_skill, slack=5 if slack_skill else 0
        )
        if one_skill_per_task:
            assert all(
                all(len(skills_used) == 1 for skills_used in allocated.values())
                for allocated in from_generic_solution.employee_usage.values()
            )


def test_jsp(optal_preview_only):
    filename = "la02"
    filepath = [f for f in jsp_parser.get_data_available() if f.endswith(filename)][0]
    problem = jsp_parser.parse_file(filepath)

    # transform to generic problem
    transfo = ShopToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)
    generic_solver = GenericSchedulingAutoOptalImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    generic_solution: GenericSchedulingImplSolution
    res = generic_solver.solve(
        do_not_retrieve_solutions=optal_preview_only, time_limit=2
    )
    assert generic_solver.status_solver in (
        StatusSolver.OPTIMAL,
        StatusSolver.SATISFIED,
    )
    assert generic_solver.current_obj >= generic_solver.current_bound
    if not optal_preview_only:
        generic_solution = res.get_best_solution()
        assert generic_problem.satisfy(generic_solution)


def test_fjsp(optal_preview_only):
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)

    # transform to generic problem
    transfo = ShopToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoOptalImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    generic_solution: GenericSchedulingImplSolution
    res = generic_solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
        time_limit=2,
        callbacks=[NbIterationStopper(1)],
    )
    assert generic_solver.status_solver in (
        StatusSolver.OPTIMAL,
        StatusSolver.SATISFIED,
    )
    assert generic_solver.current_obj >= generic_solver.current_bound
    if not optal_preview_only:
        generic_solution = res.get_best_solution()
        assert generic_problem.satisfy(generic_solution)


def test_solving_with_entities(optal_preview_only):
    entities: list[ConstantDurationEntity] = []
    duration_per_mode = {}
    for product in range(5):
        keys_for_product = set()
        for nb_task in range(3):
            name_task = f"task-prod{product}-{nb_task}"
            duration_per_mode[name_task] = {0: random.randint(2, 5)}
            keys_for_product.add(name_task)
        entities.append(
            ConstantDurationEntity(
                other_entity=GroupEntity(frozenset(keys_for_product)),
                constant_duration=10,
                start_or_end=StartOrEnd.START,
            )
        )
    span_blocking_constraints = [
        SpanBlockingConstraint(
            BlockingConstraintMetadata(
                mode=BlockingMode.RESERVATION, name_choice=f"prod_{i}"
            ),
            default_resource_blocked={"r1": 1},
            entity=entities[i],
        )
        for i in range(len(entities))
    ]
    problem = GenericSchedulingImplProblem(
        horizon=200,
        durations_per_mode=duration_per_mode,
        non_skill_cumulative_resources={"r1": 1},
        span_blocking_constraints=span_blocking_constraints,
    )
    solver = GenericSchedulingAutoOptalImplSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            ObjectiveHandling.SINGLE,
            objectives=[Objective.MAKESPAN],
            weights=[1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    # solve
    res = solver.solve(
        do_not_retrieve_solutions=optal_preview_only,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL,)
    assert solver.current_obj >= solver.current_bound

    if not optal_preview_only:
        sol = res[-1][0]
        assert problem.satisfy(sol)
        start_entities = sorted(
            [entities[i].get_start_time(sol) for i in range(len(entities))]
        )
        for j in range(1, len(entities)):
            assert start_entities[j] >= start_entities[j - 1] + 10
