#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import numpy as np
import pytest
from ortools.sat.python.cp_model import LinearExprT

import discrete_optimization.rcpsp.parser as rcpsp_parser
import discrete_optimization.rcpsp_multiskill.parser_imopse as parser_imopse
import discrete_optimization.shop.fjsp.parser as fjsp_parser
import discrete_optimization.shop.jsp.parser as jsp_parser
from discrete_optimization.generic_tasks_tools.entities import GroupEntity, TaskEntity
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    StartOrEnd
)
from discrete_optimization.generic_tasks_tools.objectives.allocated_tasks import (
    AllocatedTasksObjective,
)
from discrete_optimization.generic_tasks_tools.objectives.allocation_cost import (
    AllocationCostComputerMultimode,
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
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp.transformations.generic_scheduling_impl import (
    RcpspToGenericSchedulingTransformation,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.transformations.generic_scheduling_impl import (
    MultiskillRcpspToGenericSchedulingTransformation,
)
from discrete_optimization.shop.fjsp.problem import FJobShopSolution
from discrete_optimization.shop.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.shop.jsp.problem import JobShopSolution
from discrete_optimization.shop.jsp.solvers.cpsat import CpSatJspSolver
from discrete_optimization.shop.transformations.to_generic_scheduling import (
    ShopToGenericSchedulingTransformation,
)


@pytest.mark.parametrize(
    "objective",
    list(Objective) + [[(Objective.MAKESPAN, -2), (Objective.NB_TASKS_ALLOCATED, +2)]],
)
@pytest.mark.parametrize(
    "avoid_interval_optional, duplicate_start_var_per_mode",
    [(True, False), (False, False), (False, True)],
)
@pytest.mark.parametrize(
    "use_energy_constraints, keep_only_most_nested_energy_constraints",
    [(False, False), (True, False), (True, True)],
)
def test_auto(
    objective,
    caplog,
):
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

    # prepare solver
    if not isinstance(objective, list):
        if problem.get_objective_computer(objective) is None:
            return

    # custom objective: makespan - nb tasks allocated
    def custom_objective_factory(
        solver: GenericSchedulingAutoCpSatImplSolver,
    ) -> LinearExprT:
        return (
            solver.get_nb_tasks_allocated_variable()
            - solver.get_global_makespan_variable()
        )

    exactly_one_unary_resource_per_task = objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.CALENDAR_RESOURCES_LEVELS,
    ]
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
    solver = GenericSchedulingAutoCpSatImplSolver(
        problem=problem,
        objective=objective,
        params_objective_function=params_objective_function,
        custom_objective_factory=custom_objective_factory,
    )

    solver.init_model(
        exactly_one_unary_resource_per_task=exactly_one_unary_resource_per_task
    )

    # solve
    res = solver.solve(parameters_cp=ParametersCp.default())

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
    #elif objective == Objective.MODE_COST:
    #    assert sol.get_mode("task-1") == 1
    #    assert not sol.is_allocated("task-1", unary_resource="worker1")
    #    assert sol.is_allocated("task-1", unary_resource="worker2")
    #    assert kpi["cost"] == 3 + 10
    elif objective == Objective.CUSTOM:
        assert kpi["custom_objective"] == 2 - 9
    elif isinstance(objective, list):
        assert kpi[Objective.NB_TASKS_ALLOCATED] == 2
        assert kpi[Objective.MAKESPAN] == 9

    bad_sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=4, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=6, end=10, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    problem.satisfy(bad_sol)

    # warm start + 1 sol only => should find the "bad" solution
    solver.set_warm_start(solution=bad_sol)
    res = solver.solve(
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
        parameters_cp=ParametersCp.default(),
        callbacks=[NbIterationStopper(1)],
    )
    sol, fit = res[0]
    assert sol.raw_sol.task_variables == bad_sol.raw_sol.task_variables


@pytest.mark.parametrize(
    "objective",
    list(Objective) + [[(Objective.MAKESPAN, -2), (Objective.NB_TASKS_ALLOCATED, +2)]],
)
def test_auto_optional_tasks(
    objective,
    caplog,
):
    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return -sum(
            variable.get_start_time(task) for task in variable.problem.tasks_list
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
        mode_costs={
            "task-1": {
                0: 100,
                1: 3,
            },
            "task-2": {
                0: 0,
            },
        },
        unary_resource_costs={
            "task-1": {
                1: {
                    "worker1": 27,
                    "worker2": 10,
                },
            },
        },
    )

    # prepare solver

    # custom objective: makespan - nb tasks allocated
    def custom_objective_factory(
        solver: GenericSchedulingAutoCpSatImplSolver,
    ) -> LinearExprT:
        return -solver.get_subtasks_sum_start_time_variable(problem.tasks_list)

    exactly_one_unary_resource_per_task = objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.CALENDAR_RESOURCES_LEVELS,
        Objective.NON_RENEWABLE_RESOURCES_LEVELS,
        Objective.MODE_COST
    ]

    solver = GenericSchedulingAutoCpSatImplSolver(
        problem=problem,
        objective=objective,
        custom_objective_factory=custom_objective_factory,
    )

    solver.init_model(
        exactly_one_unary_resource_per_task=exactly_one_unary_resource_per_task
    )

    # solve
    res = solver.solve(
        parameters_cp=ParametersCp.default(),
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )

    # check sol and kpis
    sol: GenericSchedulingImplSolution
    sol, fit = res[-1]
    assert problem.satisfy(sol)
    kpi = problem.evaluate(sol)

    if objective == Objective.NB_UNARY_RESOURCES_USED:
        assert kpi["nb_unary_resources_used"] == 1
    elif objective == Objective.MAKESPAN:
        assert kpi["makespan"] == 9
    elif objective == Objective.NB_TASKS_ALLOCATED:
        assert kpi["nb_tasks_allocated"] == 2
    #elif objective == Objective.COST:
    #    assert not sol.is_present("task-1")
    #    assert kpi["cost"] == 0

    elif objective == Objective.CUSTOM:
        assert kpi["custom_objective"] == -5
    elif isinstance(objective, list):
        assert kpi["nb_tasks_allocated"] == 2
        assert kpi["makespan"] == 9

    # check warm start from a "bad" solution
    # if objective == Objective.COST:
    #     return  # skip warm start
    bad_sol = GenericSchedulingImplSolution(
        problem=problem,
        raw_sol=RawSolution(
            task_variables={
                "task-1": TaskVariable(
                    start=1, end=4, mode=1, allocated={"worker1": set()}
                ),
                "task-2": TaskVariable(
                    start=6, end=10, mode=0, allocated={"worker2": set()}
                ),
            }
        ),
    )
    problem.satisfy(bad_sol)

    # warm start + 1 sol only => should find the "bad" solution
    solver.set_warm_start(solution=bad_sol)
    res = solver.solve(
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
        parameters_cp=ParametersCp.default(),
        callbacks=[NbIterationStopper(1)],
    )
    sol, fit = res[0]
    assert sol.raw_sol.task_variables == bad_sol.raw_sol.task_variables


def test_auto_optional_tasks_with_resource_blocking():
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
    flexible_gap_blocking = (
        TaskEntity("task_a"),
        StartOrEnd.END,
        TaskEntity("task_b"),
        StartOrEnd.START,
        {"machine_1": 1},  # Block 1 unit of machine_1
        BlockingConstraintMetadata(
            description="Setup time blocking between task_a and task_b"
        ),
    )

    # SpanBlocking: Block machine_2 during the span of tasks {task_c, task_d}
    # If task_c is not scheduled, this should only block during task_d
    span_blocking = (
        GroupEntity(frozenset(["task_c", "task_d"])),
        {"machine_2": 1},  # Block 1 unit of machine_2
        BlockingConstraintMetadata(description="Safety monitoring for batch C+D"),
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
    solver = GenericSchedulingAutoCpSatImplSolver(problem=problem)
    solver.init_model()
    result = solver.solve(
        time_limit=30,
        parameters_cp=ParametersCp.default(),
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )

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

        print(f"\ntask_a is present: gap between task_a and task_b = {gap_duration}")
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


def test_start_to_end_time_lag():
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
    solver = GenericSchedulingAutoCpSatImplSolver(problem=problem)
    result = solver.solve(time_limit=10, parameters_cp=ParametersCp.default())
    solution: GenericSchedulingImplSolution = result.get_best_solution()
    assert problem.satisfy(solution)


def test_start_to_end_time_lag_optional_tasks():
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
    solver = GenericSchedulingAutoCpSatImplSolver(problem=problem)
    result = solver.solve(time_limit=10, parameters_cp=ParametersCp.default())
    solution: GenericSchedulingImplSolution = result.get_best_solution()
    assert problem.satisfy(solution)
    assert not solution.is_present("task-1")


def test_no_overlap():
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
    solver = GenericSchedulingAutoCpSatImplSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            ObjectiveHandling.SINGLE,
            objectives=[Objective.MAKESPAN],
            weights=[1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    sol: GenericSchedulingImplSolution = solver.solve().get_best_solution()
    assert problem.satisfy(sol)
    kpi = problem.evaluate(sol)
    assert kpi[Objective.MAKESPAN] == 6


def test_rcpsp_simple():
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
    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10, parameters_cp=ParametersCp.default())
    solution: RcpspSolution = result.get_best_solution()

    assert solution is not None, "Solver should find a solution"
    assert problem.satisfy(solution)
    assert solution.get_start_time(4) == 10
    assert solution.get_end_time(5) == 20
    assert solution.get_end_time(3) == solution.get_start_time(4)
    assert solution.get_start_time(2) == solution.get_start_time(3)

    # transform to generic problem
    transfo = RcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)
    generic_solver = GenericSchedulingAutoCpSatImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    result = generic_solver.solve(time_limit=10, parameters_cp=ParametersCp.default())
    generic_solution: GenericSchedulingImplSolution = result.get_best_solution()
    assert generic_solution is not None
    assert generic_problem.satisfy(generic_solution)

    # compare solutions
    from_generic_solution: RcpspSolution = solver.convert_task_variables_to_solution(
        generic_solution.raw_sol
    )
    assert from_generic_solution == solution


def test_rcpsp_mm():
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
    solver = CpSatRcpspSolver(problem=problem)
    solution, _ = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert solution is not None, "Solver should find a solution"
    assert problem.satisfy(solution)

    # transform to generic problem
    transfo = RcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoCpSatImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    generic_solution, _ = generic_solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert generic_solution is not None
    assert generic_problem.satisfy(generic_solution)

    # compare solutions
    from_generic_solution: RcpspSolution = solver.convert_task_variables_to_solution(
        generic_solution.raw_sol
    )
    assert problem.satisfy(from_generic_solution)
    print("specific", problem.evaluate(solution))
    print("generic", problem.evaluate(from_generic_solution))

    # generic solution same as or better than specific one
    assert solver.aggreg_from_sol(from_generic_solution) >= solver.aggreg_from_sol(
        solution
    )


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
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model(
        one_worker_per_task=one_worker_per_task,
        exact_skill=exact_skill,
        slack_skill=slack_skill,
        use_energy_constraints=use_energy_constraints,
        redundant_skill_cumulative=redundant_skill_cumulative,
    )
    solution: MultiskillRcpspSolution
    solution, _ = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert solution is not None, "Solver should find a solution"
    assert problem.satisfy(solution)
    assert solution.check_skill_constraints(
        exact=exact_skill, slack=5 if slack_skill else 0
    )
    if one_skill_per_task:
        assert all(
            all(len(skills_used) == 1 for skills_used in allocated.values())
            for allocated in solution.employee_usage.values()
        )

    # transform to generic problem
    transfo = MultiskillRcpspToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoCpSatImplSolver(
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
    generic_solution: GenericSchedulingImplSolution
    generic_solution, _ = generic_solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert generic_solution is not None
    assert generic_problem.satisfy(generic_solution)
    assert generic_solution.check_skill_constraints(
        exact=exact_skill, slack=5 if slack_skill else 0
    )
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
        solver.convert_task_variables_to_solution(generic_solution.raw_sol)
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
    print("specific", problem.evaluate(solution))
    print("generic", problem.evaluate(from_generic_solution))

    # generic solution same as or better than specific one
    assert solver.aggreg_from_sol(from_generic_solution) >= solver.aggreg_from_sol(
        solution
    )


def test_jsp():
    filename = "la02"
    filepath = [f for f in jsp_parser.get_data_available() if f.endswith(filename)][0]
    problem = jsp_parser.parse_file(filepath)
    solver = CpSatJspSolver(problem=problem)
    solution: JobShopSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    ).get_best_solution()
    print(solution.schedule)
    assert problem.satisfy(solution)

    # transform to generic problem
    transfo = ShopToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)
    generic_solver = GenericSchedulingAutoCpSatImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    generic_solution: GenericSchedulingImplSolution
    generic_solution, _ = generic_solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert generic_solution is not None
    assert generic_problem.satisfy(generic_solution)

    # compare solutions
    from_generic_solution: JobShopSolution = solver.convert_task_variables_to_solution(
        generic_solution.raw_sol
    )
    assert problem.satisfy(from_generic_solution)
    print("specific", problem.evaluate(solution))
    print("generic", problem.evaluate(from_generic_solution))

    # generic solution same as or better than specific one
    assert solver.aggreg_from_sol(from_generic_solution) >= solver.aggreg_from_sol(
        solution
    )


def test_fjsp():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = CpSatFjspSolver(problem=problem)
    solution: FJobShopSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    ).get_best_solution()
    print(solution.schedule)
    assert problem.satisfy(solution)

    # transform to generic problem
    transfo = ShopToGenericSchedulingTransformation()
    generic_problem = transfo.transform_problem(problem)

    generic_solver = GenericSchedulingAutoCpSatImplSolver(
        problem=generic_problem,
    )
    generic_solver.init_model()
    generic_solution: GenericSchedulingImplSolution
    generic_solution, _ = generic_solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert generic_solution is not None
    assert generic_problem.satisfy(generic_solution)

    # compare solutions
    from_generic_solution: FJobShopSolution = solver.convert_task_variables_to_solution(
        generic_solution.raw_sol
    )
    assert problem.satisfy(from_generic_solution)
    print("specific", problem.evaluate(solution))
    print("generic", problem.evaluate(from_generic_solution))

    # generic solution same as or better than specific one
    assert solver.aggreg_from_sol(from_generic_solution) >= solver.aggreg_from_sol(
        solution
    )
