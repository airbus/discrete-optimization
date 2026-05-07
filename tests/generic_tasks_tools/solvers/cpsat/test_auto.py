#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import logging
from copy import deepcopy
from typing import Iterable, Union

import pytest

from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    Objective,
    TaskVariable,
    TemporarySolution,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)

UnaryResource = str
CumulativeResource = str
Resource = Union[UnaryResource, CumulativeResource]
NonRenewableResource = str
Task = str


class MySolution(
    GenericSchedulingSolution[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]
):
    problem: MyProblem

    def __init__(
        self,
        problem: MyProblem,
        temp_sol: TemporarySolution[Task, UnaryResource],
    ):
        super().__init__(problem)
        self.temp_sol = temp_sol

    def get_end_time(self, task: Task) -> int:
        return self.temp_sol.task_variables[task].end

    def get_start_time(self, task: Task) -> int:
        return self.temp_sol.task_variables[task].start

    def get_mode(self, task: Task) -> int:
        return self.temp_sol.task_variables[task].mode

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return unary_resource in self.temp_sol.task_variables[task].allocated

    def copy(self) -> Solution:
        return MySolution(problem=self.problem, temp_sol=deepcopy(self.temp_sol))


class MyProblem(
    GenericSchedulingProblem[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]
):
    horizon = 10
    non_renewable_resources = ["non_renewable_resource"]
    cumulative_resources = ["cumulative_resource"]
    unary_resources = ["worker1", "worker2"]
    resource_availabilities = {
        "non_renewable_resource": 1,
        "cumulative_resource": [
            (3, 5, 1),
            (5, 10, 2),
        ],
        "worker1": [(1, 4, 1)],
        "worker2": [(3, 18, 1)],
    }
    mode_details = {
        "task-1": {
            0: {"non_renewable_resource": 2, "duration": 1},
            1: {"non_renewable_resource": 1, "duration": 3},
        },
        "task-2": {
            0: {"cumulative_resource": 2, "duration": 4},
        },
    }
    successors = {"task-1": ["task-2"]}

    @property
    def cumulative_resources_list(self) -> list[CumulativeResource]:
        return self.cumulative_resources

    def get_renewable_resource_consumption(
        self, resource: Resource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return self.resource_availabilities[resource]

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.mode_details[task][mode]["duration"]

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return self.non_renewable_resources

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        return self.resource_availabilities[resource]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        return self.mode_details[task][mode].get(resource, 0)

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return self.unary_resources

    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
        return self.successors

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    def get_task_modes(self, task: Task) -> set[int]:
        return set(self.mode_details[task])

    @property
    def tasks_list(self) -> list[Task]:
        return list(self.mode_details)

    def evaluate(self, variable: Solution) -> dict[str, float]:
        variable: MySolution
        return dict(
            makespan=variable.get_max_end_time(),
            nb_allocated=variable.compute_nb_unary_resources_used(),
            nb_resources_used=variable.compute_nb_renewable_resources_used()
            + variable.compute_nb_non_renewable_resources_used(),
            resources_consumptions=variable.compute_aggregated_renewable_resources_consumptions()
            + variable.compute_aggregated_non_renewable_resources_consumptions(),
            nb_tasks_done=variable.compute_nb_tasks_done(),
        )

    def satisfy(self, variable: Solution) -> bool:
        variable: MySolution
        return (
            variable.check_precedence_constraints()
            and variable.check_all_non_renewable_resource_capacity_constraints()
            and variable.check_all_renewable_resource_capacity_constraints()
            and variable.check_duration_constraints()
        )

    def get_solution_type(self) -> type[Solution]:
        return MySolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc=dict(
                makespan=ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1,
                ),
                nb_allocated=ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1,
                ),
                nb_resources_used=ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1,
                ),
                resources_consumptions=ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1,
                ),
                nb_tasks_done=ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=-1,
                ),
            ),
        )


class MyAutoCpsatSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]
):
    problem: MyProblem

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> MySolution:
        return MySolution(problem=self.problem, temp_sol=temp_sol)


def test_problem(caplog):
    problem = MyProblem()
    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=1, end=4, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=5, end=9, mode=0, allocated=["worker2"]),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    assert d["makespan"] == 9
    assert d["nb_tasks_done"] == 2
    assert d["nb_allocated"] == 2
    assert d["nb_resources_used"] == 4
    assert d["resources_consumptions"] == 5

    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=3, end=6, mode=1, allocated=["worker2"]),
                "task-2": TaskVariable(start=6, end=10, mode=0, allocated=["worker2"]),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    assert d["makespan"] == 10
    assert d["nb_tasks_done"] == 2
    assert d["nb_allocated"] == 1

    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(
                    start=3,
                    end=6,
                    mode=1,
                ),
                "task-2": TaskVariable(
                    start=6,
                    end=10,
                    mode=0,
                ),
            }
        ),
    )
    problem.satisfy(sol)
    d = problem.evaluate(sol)
    assert d["makespan"] == 10
    assert d["nb_tasks_done"] == 0
    assert d["nb_allocated"] == 0

    # nok: worker not available
    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=3, end=6, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=6, end=10, mode=0, allocated=["worker1"]),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "worker1" in caplog.text

    # nok: cumulative resource not available
    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=1, end=4, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=4, end=8, mode=0, allocated=["worker2"]),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "cumulative_resource" in caplog.text

    # nok: duration not correct
    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=1, end=2, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=5, end=9, mode=0, allocated=["worker2"]),
            }
        ),
    )
    with caplog.at_level(level=logging.DEBUG):
        assert not problem.satisfy(sol)
    assert "Duration" in caplog.text

    # lower cumulative_resource requirement => better sol
    problem.mode_details = copy.deepcopy(problem.mode_details)
    problem.mode_details["task-2"][0]["cumulative_resource"] = 1
    sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=1, end=4, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=4, end=8, mode=0, allocated=["worker2"]),
            }
        ),
    )
    assert problem.satisfy(sol)


@pytest.mark.parametrize("objective", Objective)
def test_auto(objective):
    problem = MyProblem()

    # prepare solver
    solver = MyAutoCpsatSolver(problem=problem)
    solver.objective = objective
    if objective in [
        Objective.NB_UNARY_RESOURCES_USED,
        Objective.NB_RESOURCES_USED,
        Objective.RESOURCES_CONSUMPTION,
    ]:
        solver.exactly_one_unary_resource_per_task = True
    solver.init_model()
    if objective == Objective.CUSTOM:
        objective_var = (
            solver.get_global_makespan_variable() - solver.get_nb_tasks_done_variable()
        )
        solver.cp_model.minimize(objective_var)

    # solve
    res = solver.solve()

    # check sol and kpis
    sol: MySolution
    sol, fit = res[-1]
    assert problem.satisfy(sol)
    kpi = problem.evaluate(sol)
    print(kpi)
    print(sol.temp_sol.task_variables)

    if objective == Objective.NB_UNARY_RESOURCES_USED:
        assert kpi["nb_allocated"] == 1
    elif objective == Objective.NB_RESOURCES_USED:
        assert kpi["nb_resources_used"] == 3
    elif objective == Objective.MAKESPAN:
        assert kpi["makespan"] == 9
    elif objective == Objective.NB_TASKS_DONE:
        assert kpi["nb_tasks_done"] == 2
    elif objective == Objective.CUSTOM:
        assert kpi["nb_tasks_done"] == 2
        assert kpi["makespan"] == 9
        assert kpi["nb_allocated"] == 2

    # check warm start from a "bad" solution
    bad_sol = MySolution(
        problem=problem,
        temp_sol=TemporarySolution(
            task_variables={
                "task-1": TaskVariable(start=1, end=4, mode=1, allocated=["worker1"]),
                "task-2": TaskVariable(start=6, end=10, mode=0, allocated=["worker2"]),
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
    assert sol.temp_sol.task_variables == bad_sol.temp_sol.task_variables
