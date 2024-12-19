#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Iterable, Optional

from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborBuilderTimeWindow,
    NeighborRandom,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.toulbar_tools import (
    ToulbarSolver,
    to_lns_toulbar,
)
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solvers.rcpsp_solver import RcpspSolver

try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = False
import logging

logger = logging.getLogger(__name__)


class ToulbarRcpspSolver(ToulbarSolver, RcpspSolver, WarmstartMixin):
    start_task_to_index: dict[int, int]

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule={
                self.problem.tasks_list[i]: {
                    "start_time": solution_from_toulbar2[0][i],
                    "end_time": solution_from_toulbar2[0][i]
                    + self.problem.mode_details[self.problem.tasks_list[i]][1][
                        "duration"
                    ],
                }
                for i in range(self.problem.n_jobs)
            },
        )

    def init_model(self, **kwargs: Any) -> None:
        try:
            assert not self.problem.is_rcpsp_multimode()
        except AssertionError as exc:
            logging.exception(
                f"Your problem is multimode, and this toulbar model can't tackle it. "
                f"Please go with ToulbarMultimodeRcpspSolver instead."
            )
            raise exc
        if "vns" in kwargs:
            model = pytoulbar2.CFN(
                ubinit=kwargs.get("ub", self.problem.horizon), vns=kwargs["vns"]
            )
        else:
            model = pytoulbar2.CFN(ubinit=kwargs.get("ub", self.problem.horizon))
        n_jobs = self.problem.n_jobs
        horizon = self.problem.horizon
        index_var = 0
        self.start_task_to_index = {}
        for i in range(n_jobs):
            model.AddVariable(f"x_{i}", range(horizon + 1))
            self.start_task_to_index[i] = index_var
            index_var += 1
        # force start of first task
        model.AddFunction(
            [self.problem.index_task[self.problem.source_task]],
            [0 if a == 0 else horizon for a in range(horizon + 1)],
        )
        # makespan optimisation
        model.AddFunction(
            [self.problem.index_task[self.problem.sink_task]],
            [a for a in range(horizon + 1)],
        )
        # precedence constraints
        for task in self.problem.successors:
            i_task = self.problem.index_task[task]
            duration = self.problem.mode_details[task][1]["duration"]
            for succ in self.problem.successors[task]:
                i_succ = self.problem.index_task[succ]
                model.AddFunction(
                    [i_task, i_succ],
                    [
                        (0 if a + duration <= b else horizon)
                        for a in range(horizon + 1)
                        for b in range(horizon + 1)
                    ],
                )
        for resource in self.problem.resources_list:
            capacity = self.problem.get_max_resource_capacity(resource)
            tasks = [
                (
                    self.problem.index_task[t],
                    self.problem.mode_details[t][1].get(resource, 0),
                    self.problem.mode_details[t][1]["duration"],
                )
                for t in self.problem.tasks_list
                if self.problem.mode_details[t][1].get(resource, 0) > 0
            ]
            for time_a in range(horizon + 1):
                params = ""
                scope = []
                for (i_t, val, dur) in tasks:
                    paramval = ""
                    nbval = 0
                    for time_b in range(horizon + 1):
                        if time_b <= time_a < time_b + dur:
                            nbval += 1
                            paramval += " " + str(time_b) + " " + str(-val)
                    if nbval > 0:
                        params += " " + str(nbval) + paramval
                        scope.append(i_t)
                if len(scope) > 0:
                    model.CFN.wcsp.postKnapsackConstraint(
                        scope, str(-capacity) + params, False, True
                    )
        self.model = model

    def set_warm_start(self, solution: RcpspSolution) -> None:
        for i in range(self.problem.n_jobs):
            st = solution.get_start_time(self.problem.tasks_list[i])
            self.model.CFN.wcsp.setBestValue(i, st)


class ToulbarMultimodeRcpspSolver(ToulbarSolver, RcpspSolver, WarmstartMixin):
    modes: list[list[dict]]
    name_var_to_index: dict[str, int]
    start_task_to_index: dict[int, int]

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        modes_dict = {t: 1 for t in self.problem.tasks_list}
        duration_dict = {
            t: self.problem.mode_details[t][1]["duration"]
            for t in self.problem.tasks_list
        }
        for i in range(len(self.modes)):
            if len(self.modes[i]) > 1:
                mode = solution_from_toulbar2[0][self.name_var_to_index[f"mode_{i}"]]
                modes_dict[self.problem.tasks_list[i]] = mode + 1
                duration_dict[self.problem.tasks_list[i]] = self.modes[i][int(mode)][
                    "duration"
                ]
        return RcpspSolution(
            problem=self.problem,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
            rcpsp_schedule={
                self.problem.tasks_list[i]: {
                    "start_time": solution_from_toulbar2[0][
                        self.name_var_to_index[f"start_{i}"]
                    ],
                    "end_time": solution_from_toulbar2[0][
                        self.name_var_to_index[f"start_{i}"]
                    ]
                    + duration_dict[self.problem.tasks_list[i]],
                }
                for i in range(self.problem.n_jobs)
            },
        )

    def init_model(self, **kwargs: Any) -> None:
        model = pytoulbar2.CFN(ubinit=self.problem.horizon)
        modes = [
            [
                self.problem.mode_details[t][m]
                for m in sorted(self.problem.mode_details[t])
            ]
            for t in self.problem.tasks_list
        ]
        self.modes = modes
        n_jobs = self.problem.n_jobs
        horizon = self.problem.horizon
        name_var_to_index = {}
        self.start_task_to_index = {}
        index = 0
        for i in range(n_jobs):
            if len(modes[i]) > 1:
                model.AddVariable(f"start_{i}", range(horizon + 1))
                name_var_to_index[f"start_{i}"] = index
                self.start_task_to_index[i] = index
                index += 1
                model.AddVariable(f"mode_{i}", range(len(modes[i])))
                name_var_to_index[f"mode_{i}"] = index
                index += 1
                for i_mode in range(len(modes[i])):
                    model.AddVariable(f"start_{i}_{i_mode}", range(-1, horizon + 1))
                    name_var_to_index[f"start_{i}_{i_mode}"] = index
                    index += 1
                    # when the mode is not selected, then the start of this task/mode is -1.
                    model.AddFunction(
                        [f"mode_{i}", f"start_{i}_{i_mode}"],
                        [
                            1000
                            if (
                                (j_mode == i_mode and time == -1)
                                or (j_mode != i_mode and time != -1)
                            )
                            else 0
                            for j_mode in range(len(modes[i]))
                            for time in range(-1, horizon + 1)
                        ],
                    )
                # the real start time is the sum of all starts per mode + (len(modes)-1) (counting the -1)
                model.AddLinearConstraint(
                    coefs=[-1 for _ in range(len(modes[i]))] + [1],
                    scope=[f"start_{i}_{i_mode}" for i_mode in range(len(modes[i]))]
                    + [f"start_{i}"],
                    operand="==",
                    rightcoef=len(modes[i]) - 1,
                )
                durs = [modes[i][j]["duration"] for j in range(len(modes[i]))]
                model.AddVariable(f"dur_{i}", range(max(durs) + 1))
                name_var_to_index[f"dur_{i}"] = index
                index += 1
                model.AddFunction(
                    [f"mode_{i}", f"dur_{i}"],
                    costs=[
                        1000 if durs[mode] != dur else 0
                        for mode in range(len(modes[i]))
                        for dur in range(max(durs) + 1)
                    ],
                )
            else:
                model.AddVariable(f"start_{i}", range(horizon + 1))
                name_var_to_index[f"start_{i}"] = index
                self.start_task_to_index[i] = index
                index += 1

        # force start of first task
        model.AddFunction(
            [f"start_{self.problem.index_task[self.problem.source_task]}"],
            [0 if a == 0 else horizon for a in range(horizon + 1)],
        )
        # makespan optimisation
        model.AddFunction(
            [f"start_{self.problem.index_task[self.problem.sink_task]}"],
            [a for a in range(horizon + 1)],
        )
        # precedence constraints
        for task in self.problem.successors:
            i_task = self.problem.index_task[task]
            duration = self.problem.mode_details[task][1]["duration"]
            # duration = max(m["duration"] for m in modes[i_task])
            for succ in self.problem.successors[task]:
                i_succ = self.problem.index_task[succ]
                if f"dur_{i_task}" in name_var_to_index:
                    model.AddLinearConstraint(
                        [1, 1, -1],
                        scope=[
                            name_var_to_index[f"start_{i_task}"],
                            name_var_to_index[f"dur_{i_task}"],
                            name_var_to_index[f"start_{i_succ}"],
                        ],
                        operand="<=",
                        rightcoef=0,
                    )
                else:
                    model.AddFunction(
                        [f"start_{i_task}", f"start_{i_succ}"],
                        [
                            (0 if a + duration <= b else horizon)
                            for a in range(horizon + 1)
                            for b in range(horizon + 1)
                        ],
                    )
        for resource in self.problem.resources_list:
            capacity = self.problem.get_max_resource_capacity(resource)
            if resource in self.problem.non_renewable_resources:
                constant_cons = 0
                params = ""
                scopes = []
                for i_t in range(len(modes)):
                    if len(modes[i_t]) == 1:
                        if modes[i_t][0].get(resource, 0) > 0:
                            constant_cons += modes[i_t][0][resource]
                    else:
                        j_s = [
                            (j, modes[i_t][j].get(resource, 0))
                            for j in range(len(modes[i_t]))
                        ]
                        if len(j_s) > 0:
                            params += (
                                " "
                                + str(len(j_s))
                                + " "
                                + " ".join(f" {x[0]} {-x[1]}" for x in j_s)
                            )
                            scopes.append(name_var_to_index[f"mode_{i_t}"])
                if len(scopes) > 0:
                    # print(params, scope)
                    model.CFN.wcsp.postKnapsackConstraint(
                        scopes, str(-(capacity + constant_cons)) + params, False, True
                    )
            else:
                tasks = [
                    (
                        i_t,
                        i_mode,
                        modes[i_t][i_mode].get(resource, 0),
                        modes[i_t][i_mode]["duration"],
                    )
                    for i_t in range(len(modes))
                    for i_mode in range(len(modes[i_t]))
                ]
                for time_a in range(horizon + 1):
                    params = ""
                    scope = []
                    for (i_t, i_mode, val, dur) in tasks:
                        paramval = ""
                        nbval = 0
                        for time_b in range(horizon + 1):
                            if time_b <= time_a < time_b + dur:
                                nbval += 1
                                paramval += " " + str(time_b) + " " + str(-val)
                        if nbval > 0:
                            params += " " + str(nbval) + paramval
                            if len(modes[i_t]) > 1:
                                scope.append(name_var_to_index[f"start_{i_t}_{i_mode}"])
                            else:
                                scope.append(name_var_to_index[f"start_{i_t}"])
                    if len(scope) > 0:
                        # print(params, scope)
                        model.CFN.wcsp.postKnapsackConstraint(
                            scope, str(-capacity) + params, False, True
                        )
        self.name_var_to_index = name_var_to_index
        self.model = model

    def set_warm_start(self, solution: RcpspSolution) -> None:
        for i in range(self.problem.n_jobs):
            st = solution.get_start_time(self.problem.tasks_list[i])
            self.model.CFN.wcsp.setBestValue(self.name_var_to_index[f"start_{i}"], st)


ToulbarRcpspSolverForLns = to_lns_toulbar(ToulbarRcpspSolver)
ToulbarMultimodeRcpspSolverForLns = to_lns_toulbar(ToulbarMultimodeRcpspSolver)


class RcpspConstraintHandlerToulbar(ConstraintHandler):
    def __init__(self, problem: RcpspProblem, fraction_task: float = 0.5):
        self.fraction_task = fraction_task
        self.problem = problem
        self.graph = problem.compute_graph()
        neighbors_1 = NeighborBuilderSubPart(
            problem=self.problem, graph=self.graph, nb_cut_part=3
        )
        neighbors_2 = NeighborRandom(
            problem=self.problem, graph=self.graph, fraction_subproblem=0.3
        )
        neighbors_3 = NeighborBuilderTimeWindow(
            problem=self.problem, graph=self.graph, time_window_length=20
        )
        self.neighbors = NeighborBuilderMix(
            [neighbors_1, neighbors_2, neighbors_3], [1 / 3, 1 / 3, 1 / 3]
        )

    def remove_constraints_from_previous_iteration(
        self, solver: SolverDO, previous_constraints: Iterable[Any], **kwargs: Any
    ) -> None:
        pass

    def adding_constraint_from_results_store(
        self,
        solver: ToulbarRcpspSolverForLns,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        sol: RcpspSolution = result_storage.get_best_solution_fit()[0]
        ms = sol.get_start_time(self.problem.sink_task)
        tasks_1, tasks_2 = self.neighbors.find_subtasks(current_solution=sol)
        solver.model.CFN.timer(100)
        try:
            for t in tasks_1:
                st = sol.get_start_time(t)
                ind = self.problem.index_task[t]
                solver.model.AddLinearConstraint(
                    [1],
                    [solver.start_task_to_index[ind]],
                    operand="<=",
                    rightcoef=min(st + 100, ms),
                )
                solver.model.AddLinearConstraint(
                    [1],
                    [solver.start_task_to_index[ind]],
                    operand=">=",
                    rightcoef=max(0, st - 100),
                )
            for t in tasks_2:
                st = sol.get_start_time(t)
                ind = self.problem.index_task[t]
                solver.model.AddLinearConstraint(
                    [1],
                    [solver.start_task_to_index[ind]],
                    operand="<=",
                    rightcoef=min(st + 5, ms),
                )
                solver.model.AddLinearConstraint(
                    [1],
                    [solver.start_task_to_index[ind]],
                    operand=">=",
                    rightcoef=max(0, st - 5),
                )
        except:
            pass
        solver.set_warm_start(solution=sol)
