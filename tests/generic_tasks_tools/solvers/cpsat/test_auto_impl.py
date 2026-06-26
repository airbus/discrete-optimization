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
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.solvers.cpsat_auto import (
    CpSatAutoMultiskillRcpspSolver,
)
from discrete_optimization.shop.fjsp.problem import FJobShopSolution
from discrete_optimization.shop.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.shop.jsp.problem import JobShopSolution
from discrete_optimization.shop.jsp.solvers.cpsat import CpSatJspSolver


@pytest.mark.parametrize(
    "objective",
    list(Objective) + [[(Objective.MAKESPAN, -2), (Objective.NB_TASKS_DONE, +2)]],
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
    avoid_interval_optional,
    duplicate_start_var_per_mode,
    use_energy_constraints,
    keep_only_most_nested_energy_constraints,
    caplog,
):
    def custom_evaluate_fn(variable: GenericSchedulingImplSolution):
        return variable.compute_nb_tasks_done() - variable.get_max_end_time()

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
        successors={"task-1": ["task-2"]},
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
        return (
            solver.get_nb_tasks_done_variable() - solver.get_global_makespan_variable()
        )

    exactly_one_unary_resource_per_task = objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.NB_RESOURCES_USED,
        Objective.RESOURCES_LEVELS,
        Objective.COST,
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
    res = solver.solve(parameters_cp=ParametersCp.default())

    # check sol and kpis
    sol: GenericSchedulingImplSolution
    sol, fit = res[-1]
    assert problem.satisfy(sol)
    kpi = problem.evaluate(sol)

    if objective == Objective.NB_UNARY_RESOURCES_USED:
        assert kpi["nb_unary_resources_used"] == 1
    elif objective == Objective.NB_RESOURCES_USED:
        assert kpi["nb_resources_used"] == 3
    elif objective == Objective.MAKESPAN:
        assert kpi["makespan"] == 9
    elif objective == Objective.NB_TASKS_DONE:
        assert kpi["nb_tasks_done"] == 2
    elif objective == Objective.COST:
        assert sol.get_mode("task-1") == 1
        assert not sol.is_allocated("task-1", unary_resource="worker1")
        assert sol.is_allocated("task-1", unary_resource="worker2")
        assert kpi["cost"] == 3 + 10

    elif objective == Objective.CUSTOM:
        assert kpi["custom_objective"] == 2 - 9
    elif isinstance(objective, list):
        assert kpi["nb_tasks_done"] == 2
        assert kpi["makespan"] == 9

    # check warm start from a "bad" solution
    if objective == Objective.COST:
        return  # skip warm start
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


def test_rcpsp_simple():
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 3, "R1": 1}},
        3: {1: {"duration": 2, "R1": 1}},
        4: {1: {"duration": 4, "R1": 1}},
        5: {1: {"duration": 0}},  # dummy end
    }

    successors = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [],
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
    solver = CpSatAutoRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10, parameters_cp=ParametersCp.default())
    solution: RcpspSolution = result.get_best_solution()

    assert solution is not None, "Solver should find a solution"
    assert problem.satisfy(solution)
    assert solution.get_start_time(4) == 10
    assert solution.get_end_time(5) == 20
    assert solution.get_end_time(3) == solution.get_start_time(4)
    assert solution.get_start_time(2) == solution.get_start_time(3)

    # transform to generic problem
    time_windows = {}
    windowed_tasks = set()
    windowed_tasks.update(special_constraints.start_times)
    windowed_tasks.update(special_constraints.end_times)
    for task in windowed_tasks:
        start_lb = start_ub = special_constraints.start_times.get(task, None)
        end_lb = end_ub = special_constraints.end_times.get(task, None)
        time_windows[task] = (start_lb, end_lb, start_ub, end_ub)

    start_to_start_min_time_lags = problem.get_start_to_start_min_time_lags() + [
        (t2, t1, -offset)
        for t1, t2, offset in problem.get_start_to_start_max_time_lags()
    ]
    start_to_end_min_time_lags = problem.get_start_to_end_min_time_lags() + [
        (t2, t1, -offset) for t1, t2, offset in problem.get_end_to_start_max_time_lags()
    ]
    end_to_start_min_time_lags = problem.get_end_to_start_min_time_lags() + [
        (t2, t1, -offset) for t1, t2, offset in problem.get_start_to_end_max_time_lags()
    ]
    end_to_end_min_time_lags = problem.get_end_to_end_min_time_lags() + [
        (t2, t1, -offset) for t1, t2, offset in problem.get_end_to_end_max_time_lags()
    ]
    durations_per_mode = {
        task: {
            mode: problem.get_task_mode_duration(task=task, mode=mode)
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    resource_consumptions = {
        task: {
            mode: {
                **{
                    resource: problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.cumulative_resources_list
                },
                **{
                    resource: problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.non_renewable_resources_list
                },
            }
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    generic_problem = GenericSchedulingImplProblem(
        horizon=horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        non_skill_cumulative_resources=resources,
        time_windows=time_windows,
        start_to_start_min_time_lags=start_to_start_min_time_lags,
        end_to_start_min_time_lags=end_to_start_min_time_lags,
        start_to_end_min_time_lags=start_to_end_min_time_lags,
        end_to_end_min_time_lags=end_to_end_min_time_lags,
    )
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
    solver = CpSatAutoRcpspSolver(problem=problem)
    solution, _ = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    )[-1]
    assert solution is not None, "Solver should find a solution"
    assert problem.satisfy(solution)

    # transform to generic problem
    non_renewable_resources = {
        resource: problem.get_non_renewable_resource_capacity(resource)
        for resource in problem.non_renewable_resources_list
    }
    non_skill_cumulative_resources = {
        resource: problem.get_resource_availabilities(resource)
        for resource in problem.non_skill_cumulative_resources_list
    }
    durations_per_mode = {
        task: {
            mode: problem.get_task_mode_duration(task=task, mode=mode)
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    resource_consumptions = {
        task: {
            mode: {
                **{
                    resource: problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.cumulative_resources_list
                },
                **{
                    resource: problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.non_renewable_resources_list
                },
            }
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    generic_problem = GenericSchedulingImplProblem(
        horizon=problem.horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=problem.successors,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
        non_renewable_resources=non_renewable_resources,
    )

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
    solver = CpSatAutoMultiskillRcpspSolver(
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
    non_renewable_resources = {
        resource: problem.get_non_renewable_resource_capacity(resource)
        for resource in problem.non_renewable_resources_list
    }
    non_skill_cumulative_resources = {
        resource: problem.get_resource_availabilities(resource)
        for resource in problem.non_skill_cumulative_resources_list
    }
    skills = set(problem.skills_list)
    unary_resources = set(problem.unary_resources_list)
    unary_resources_skills = {
        unary_resource: {
            skill: detail.skill_value
            for skill, detail in problem.employees[unary_resource].dict_skill.items()
        }
        for unary_resource in unary_resources
    }
    unary_resources_availabilities = {
        unary_resource: [
            (start, end)
            for start, end, _ in problem.get_resource_availabilities(
                resource=unary_resource
            )
        ]
        for unary_resource in unary_resources
    }
    durations_per_mode = {
        task: {
            mode: problem.get_task_mode_duration(task=task, mode=mode)
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    resource_consumptions = {
        task: {
            mode: {
                **{
                    resource: problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.cumulative_resources_list
                },
                **{
                    resource: problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in problem.non_renewable_resources_list
                },
            }
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    generic_problem = GenericSchedulingImplProblem(
        horizon=problem.horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=problem.successors,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
        non_renewable_resources=non_renewable_resources,
        skills=skills,
        unary_resources=unary_resources,
        unary_resources_skills=unary_resources_skills,
        unary_resources_availabilities=unary_resources_availabilities,
    )

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
    non_skill_cumulative_resources = {
        resource: problem.get_resource_availabilities(resource)
        for resource in problem.non_skill_cumulative_resources_list
    }
    durations_per_mode = {
        task: {
            mode: problem.get_task_mode_duration(task=task, mode=mode)
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    resource_consumptions = {
        task: {
            mode: {
                resource: problem.get_cumulative_resource_consumption(
                    resource=resource, task=task, mode=mode
                )
                for resource in problem.cumulative_resources_list
            }
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    successors = problem.get_precedence_constraints()
    generic_problem = GenericSchedulingImplProblem(
        horizon=problem.horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
    )

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
    non_skill_cumulative_resources = {
        resource: problem.get_resource_availabilities(resource)
        for resource in problem.non_skill_cumulative_resources_list
    }
    durations_per_mode = {
        task: {
            mode: problem.get_task_mode_duration(task=task, mode=mode)
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    resource_consumptions = {
        task: {
            mode: {
                resource: problem.get_cumulative_resource_consumption(
                    resource=resource, task=task, mode=mode
                )
                for resource in problem.cumulative_resources_list
            }
            for mode in problem.get_task_modes(task)
        }
        for task in problem.tasks_list
    }
    successors = problem.get_precedence_constraints()
    generic_problem = GenericSchedulingImplProblem(
        horizon=problem.horizon,
        durations_per_mode=durations_per_mode,
        resource_consumptions=resource_consumptions,
        successors=successors,
        non_skill_cumulative_resources=non_skill_cumulative_resources,
    )

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
