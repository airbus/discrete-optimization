#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from functools import reduce

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp_alternative.problem import (
    AlternativeSchedulingSubProblem,
    RcpspWithAlternativePath,
)


class CpsatRcpspWithAlternativePathSolver(CpSatRcpspSolver):
    hyperparameters = [
        CategoricalHyperparameter(
            name="strict_alternative_path", choices=[True, False], default=True
        )
    ]
    problem: RcpspWithAlternativePath
    additional_variables: dict

    def create_is_done_variable(self):
        is_done = {}
        for t in self.problem.alternative_tasks:
            is_done[t] = self.cp_model.NewBoolVar(name=f"is_done_{t}")
        return is_done

    def link_is_done_and_modes(self, is_present_var, is_done, interval_per_tasks):
        for t in interval_per_tasks:
            self.cp_model.add_max_equality(
                is_done[t], [is_present_var[key] for key in interval_per_tasks[t]]
            )

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        include_special_constraints = kwargs.get(
            "include_special_constraints", self.problem.includes_special_constraint()
        )
        strict_alternative_path = kwargs["strict_alternative_path"]
        SchedulingCpSatSolver.init_model(self, **kwargs)
        model = self.cp_model
        (
            starts_var,
            ends_var,
            is_present_var,
            interval_var,
            interval_per_tasks,
        ) = self.init_temporal_variable(model=model)
        self.variables = {
            "start": starts_var,
            "end": ends_var,
            "is_present": is_present_var,
            "interval_var": interval_var,
        }
        self.add_one_mode_selected_per_task(
            model=model,
            is_present_var=is_present_var,
            interval_per_tasks={
                t: interval_per_tasks[t] for t in self.problem._tasks_list
            },
        )
        is_done = self.create_is_done_variable()
        self.additional_variables = {"is_done": is_done}
        self.link_is_done_and_modes(
            is_present_var=is_present_var,
            is_done=is_done,
            interval_per_tasks={t: interval_per_tasks[t] for t in is_done},
        )
        self.create_precedence_constraints()
        self.create_precedence_constraints_alternative()
        resources = self.problem.resources_list
        for resource in resources:
            self.create_cumulative_constraint(
                resource=resource,
            )
        for i in range(len(self.problem.list_alternative_subproblem)):
            self.create_alternative_path_constraint(
                alternative_problem=self.problem.list_alternative_subproblem[i],
                tag_alternative_problem=str(i),
                strict_alternative_path=strict_alternative_path,
            )
        if include_special_constraints:
            if self.problem.special_constraints.pair_mode_constraint is not None:
                self.create_mode_pair_constraint(
                    model=model,
                    interval_per_tasks=interval_per_tasks,
                    is_present_var=is_present_var,
                    pair_mode_constraint=self.problem.special_constraints.pair_mode_constraint,
                )
            self.add_special_temporal_constraints(
                model=model,
                starts_var=starts_var,
                ends_var=ends_var,
            )
        objective = self.get_global_makespan_variable()
        self.minimize_variable(objective)

    def create_precedence_constraints_alternative(self):
        for t in self.problem.alternative_successors:
            for tt in self.problem.alternative_successors[t]:
                if tt in self.problem._tasks_list:
                    (
                        self.cp_model.add(
                            self.variables["end"][t] <= self.variables["start"][tt]
                        ).only_enforce_if(self.additional_variables["is_done"][t])
                    )
                if tt in self.problem.alternative_tasks:
                    (
                        self.cp_model.add(
                            self.variables["end"][t] <= self.variables["start"][tt]
                        ).only_enforce_if(
                            self.additional_variables["is_done"][t],
                            self.additional_variables["is_done"][tt],
                        )
                    )

    def create_alternative_path_constraint(
        self,
        alternative_problem: AlternativeSchedulingSubProblem,
        tag_alternative_problem: str,
        strict_alternative_path: bool,
    ):
        ps = [
            [p for p in path if p in self.problem.alternative_tasks]
            for path in alternative_problem.list_paths
        ]
        sum_len = sum([len(p) for p in ps])
        merged = reduce(lambda x, y: x.union(set(y)), ps, set())
        len_merged = len(merged)
        if len_merged == sum_len:
            # print("Disjoint paths")
            # Disjoint paths, nominal case.
            for p in ps:
                for p0, p1 in zip(p[:-1], p[1:]):
                    self.cp_model.add_implication(
                        self.additional_variables["is_done"][p0],
                        self.additional_variables["is_done"][p1],
                    )
            nb_to_do = alternative_problem.nb_path_to_do
            if len(ps) >= nb_to_do:
                if nb_to_do == 1:
                    self.cp_model.add_exactly_one(
                        [self.additional_variables["is_done"][p[0]] for p in ps]
                    )
                else:
                    self.cp_model.add(
                        sum([self.additional_variables["is_done"][p[0]] for p in ps])
                        == nb_to_do
                    )
            if alternative_problem.is_path_successors:
                for p in ps:
                    for p0, p1 in zip(p[:-1], p[1:]):
                        # print(p0, p1)
                        self.cp_model.add(
                            self.variables["start"][p1] >= self.variables["end"][p0]
                        )
                for p in alternative_problem.list_paths:
                    for p0, p1 in zip(p[:-1], p[1:]):
                        # print(p0, p1)
                        self.cp_model.add(
                            self.variables["start"][p1] >= self.variables["end"][p0]
                        )
                    if p[0] != alternative_problem.source_task:
                        self.cp_model.add(
                            self.variables["start"][p[0]]
                            >= self.variables["end"][alternative_problem.source_task]
                        )
                    if p[-1] != alternative_problem.sink_task:
                        self.cp_model.add(
                            self.variables["start"][alternative_problem.sink_task]
                            >= self.variables["end"][p[-1]]
                        )
        else:
            path_taken = [
                self.cp_model.new_bool_var(name=f"{tag_alternative_problem}_{i}")
                for i in range(len(ps))
            ]
            for i in range(len(path_taken)):
                self.cp_model.add_min_equality(
                    path_taken[i],
                    [self.additional_variables["is_done"][p] for p in ps[i]],
                )
            nb_to_do = alternative_problem.nb_path_to_do
            if nb_to_do == 1:
                self.cp_model.add_exactly_one(path_taken)
            else:
                self.cp_model.add(sum(path_taken) == nb_to_do)
            if alternative_problem.is_path_successors:
                for i in range(len(alternative_problem.list_paths)):
                    path = alternative_problem.list_paths[i]
                    for p0, p1 in zip(path[:-1], path[1:]):
                        (
                            self.cp_model.add(
                                self.variables["start"][p1] >= self.variables["end"][p0]
                            ).only_enforce_if(path_taken[i])
                        )
                    if path[0] != alternative_problem.source_task:
                        (
                            self.cp_model.add(
                                self.variables["start"][path[0]]
                                >= self.variables["end"][
                                    alternative_problem.source_task
                                ]
                            ).only_enforce_if(path_taken[i])
                        )
                    if path[-1] != alternative_problem.sink_task:
                        (
                            self.cp_model.add(
                                self.variables["start"][alternative_problem.sink_task]
                                >= self.variables["end"][path[-1]]
                            ).only_enforce_if(path_taken[i])
                        )
        if strict_alternative_path:
            path_used = [
                self.cp_model.new_bool_var(name=f"{tag_alternative_problem}_{i}")
                for i in range(len(ps))
            ]
            for i in range(len(path_used)):
                self.cp_model.add_max_equality(
                    path_used[i],
                    [self.additional_variables["is_done"][p] for p in ps[i]],
                )
            self.cp_model.add(sum(path_used) <= alternative_problem.nb_path_to_do)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RcpspSolution:
        schedule = {}
        modes_dict = {}
        for task in self.variables["start"]:
            schedule[task] = {
                "start_time": cpsolvercb.Value(self.variables["start"][task]),
                "end_time": cpsolvercb.Value(self.variables["end"][task]),
            }
        for task, mode in self.variables["is_present"]:
            if cpsolvercb.Value(self.variables["is_present"][task, mode]):
                modes_dict[task] = mode
        for t in self.problem.alternative_tasks:
            if not cpsolvercb.Value(self.additional_variables["is_done"][t]):
                schedule[t]["start_time"] = 0
                schedule[t]["end_time"] = 0
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=schedule,
            rcpsp_modes=[
                modes_dict.get(t, None) for t in self.problem.tasks_list_non_dummy
            ],
        )
