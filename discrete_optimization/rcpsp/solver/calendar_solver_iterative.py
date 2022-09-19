#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Iterable, Optional, Union

import numpy as np
from deprecation import deprecated
from minizinc import Instance

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import (
    LNS_CP,
    ConstraintHandler,
    SolverDO,
)
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    PartialSolution,
    RCPSPModel,
    RCPSPModelCalendar,
    RCPSPSolution,
)
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN, CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import (
    OptionNeighbor,
    build_neighbor_operator,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)


@deprecated(deprecated_in="0.1")
def get_ressource_breaks(
    problem_calendar: RCPSPModel,
    problem_no_calendar: RCPSPModel,
    solution: RCPSPSolution,
):
    ressources = problem_calendar.resources_list
    ressource_arrays = {}
    ressource_arrays_usage = {}
    for r in ressources:
        ressource_arrays[r] = np.zeros((solution.get_max_end_time() + 1))
        ressource_arrays_usage[r] = np.zeros(
            (solution.get_max_end_time() + 1, len(solution.rcpsp_schedule))
        )
    sorted_keys_schedule = problem_calendar.tasks_list
    for ji in range(len(sorted_keys_schedule)):
        j = sorted_keys_schedule[ji]
        for r in problem_calendar.mode_details[j][1]:
            if r == "duration":
                continue
            if r is None:
                continue
            if r == "None":
                continue
            if problem_calendar.mode_details[j][1][r] == 0:
                continue
            if (
                solution.rcpsp_schedule[j]["end_time"]
                == solution.rcpsp_schedule[j]["start_time"]
            ):
                continue
            ressource_arrays_usage[r][
                int(solution.rcpsp_schedule[j]["start_time"]) : int(
                    solution.rcpsp_schedule[j]["end_time"]
                ),
                ji,
            ] = 1
            ressource_arrays[r][
                int(solution.rcpsp_schedule[j]["start_time"]) : int(
                    solution.rcpsp_schedule[j]["end_time"]
                )
            ] += problem_calendar.mode_details[j][1][r]
    index_ressource = {}
    task_concerned = {}
    constraints = {}
    for r in ressource_arrays:
        index = np.argwhere(ressource_arrays[r] > problem_calendar.resources[r])
        index_ressource[r] = index
        task_concerned[r] = [
            j
            for j in range(ressource_arrays_usage[r].shape[1])
            if any(
                ressource_arrays_usage[r][ind[0], j] == 1
                for ind in index
                if problem_calendar.resources[r][ind[0]] == 0
            )
        ]
        task_concerned[r] = [sorted_keys_schedule[rr] for rr in task_concerned[r]]
        constraints[r] = {}
        for t in task_concerned[r]:
            current_start = solution.rcpsp_schedule[t]["start_time"]
            first_possible_start_future = next(
                (
                    st
                    for st in range(current_start, len(problem_calendar.resources[r]))
                    if problem_calendar.resources[r][st]
                    >= problem_calendar.mode_details[t][1][r]
                ),
                None,
            )
            first_possible_start_before = next(
                (
                    st - problem_calendar.mode_details[t][1]["duration"] + 1
                    for st in range(solution.rcpsp_schedule[t]["end_time"], -1, -1)
                    if problem_calendar.resources[r][st - 1]
                    >= problem_calendar.mode_details[t][1][r]
                    and problem_calendar.resources[r][
                        max(0, st - problem_calendar.mode_details[t][1]["duration"] + 1)
                    ]
                    >= problem_calendar.mode_details[t][1][r]
                ),
                None,
            )
            constraints[r][t] = (
                first_possible_start_before,
                first_possible_start_future,
            )
    return index_ressource, constraints


@deprecated(deprecated_in="0.1")
class PostProcessSolutionNonFeasible(PostProcessSolution):
    def __init__(
        self,
        problem_calendar: RCPSPModelCalendar,
        problem_no_calendar: RCPSPModel,
        partial_solution: PartialSolution = None,
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = problem_no_calendar
        self.partial_solution = partial_solution
        if self.partial_solution is None:

            def check_solution(problem, solution):
                return True

        else:

            def check_solution(problem, solution):
                start_together = partial_solution.start_together
                start_at_end = partial_solution.start_at_end
                start_at_end_plus_offset = partial_solution.start_at_end_plus_offset
                start_after_nunit = partial_solution.start_after_nunit
                for (t1, t2) in start_together:
                    b = (
                        solution.rcpsp_schedule[t1]["start_time"]
                        == solution.rcpsp_schedule[t2]["start_time"]
                    )
                    if not b:
                        return False
                for (t1, t2) in start_at_end:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        == solution.rcpsp_schedule[t1]["end_time"]
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_at_end_plus_offset:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["end_time"] + off
                    )
                    if not b:
                        return False
                for (t1, t2, off) in start_after_nunit:
                    b = (
                        solution.rcpsp_schedule[t2]["start_time"]
                        >= solution.rcpsp_schedule[t1]["start_time"] + off
                    )
                    if not b:
                        return False
                return True

        self.check_sol = check_solution

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        for sol in list(result_storage.list_solution_fits):
            if "satisfy" not in sol[0].__dict__.keys():
                rb, constraints = get_ressource_breaks(
                    self.problem_calendar, self.problem_no_calendar, sol[0]
                )
                sol[0].satisfy = not (any(len(rb[r]) > 0 for r in rb))
                sol[0].constraints = constraints
            if self.partial_solution is None:
                solution = RCPSPSolution(
                    problem=self.problem_calendar,
                    rcpsp_permutation=sol[0].rcpsp_permutation,
                    rcpsp_modes=sol[0].rcpsp_modes,
                )
                solution.satisfy = self.check_sol(self.problem_calendar, solution)
                result_storage.list_solution_fits += [
                    (
                        solution,
                        -self.problem_calendar.evaluate(solution)["makespan"],
                    )
                ]
        return result_storage


@deprecated(deprecated_in="0.1")
class ConstraintHandlerAddCalendarConstraint(ConstraintHandler):
    def __init__(
        self,
        problem_calendar: RCPSPModelCalendar,
        problem_no_calendar: RCPSPModel,
        other_constraint: ConstraintHandler,
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = problem_no_calendar
        self.other_constraint = other_constraint
        self.store_constraints = set()

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is None:
            raise ValueError("This constraint need last_result_store to be not None.")
        for s, f in list(result_storage.list_solution_fits):
            if "satisfy" in s.__dict__.keys() and s.satisfy:
                last_result_store.list_solution_fits += [(s, f)]
        r = random.random()
        if r <= 0.2:
            solution, fit = last_result_store.get_last_best_solution()
        elif r <= 0.4:
            solution, fit = last_result_store.get_best_solution_fit()
        elif r <= 0.99:
            solution, fit = last_result_store.get_random_best_solution()
        else:
            solution, fit = last_result_store.get_random_solution()
        for s in self.store_constraints:
            child_instance.add_string(s)
        if "satisfy" in solution.__dict__.keys() and solution.satisfy:
            return self.other_constraint.adding_constraint_from_results_store(
                cp_solver, child_instance, last_result_store, last_result_store
            )
        ressource_breaks, constraints = get_ressource_breaks(
            self.problem_calendar, self.problem_no_calendar, solution
        )
        list_strings = []
        for r in ressource_breaks:
            index_ressource = cp_solver.resources_index.index(r)
            if len(ressource_breaks[r]) == 0:
                continue
            for index in [
                ressource_breaks[r][i] for i in range(0, len(ressource_breaks[r]), 20)
            ] + [ressource_breaks[r][-1]]:
                ind = index[0]
                rq = self.problem_calendar.resources[r][ind]
                s = ""
                if isinstance(cp_solver, CP_MRCPSP_MZN):
                    s = (
                        """constraint """
                        + str(rq)
                        + """>=sum( i in Act ) (
                                    bool2int(start[i] <="""
                        + str(ind)
                        + """ /\ """
                        + str(ind)
                        + """< start[i] + adur[i]) * arreq["""
                        + str(index_ressource + 1)
                        + """,i]);\n"""
                    )
                elif isinstance(cp_solver, CP_RCPSP_MZN):
                    s = (
                        """constraint """
                        + str(rq)
                        + """>=sum( i in Tasks ) (
                                                        bool2int(s[i] <="""
                        + str(ind)
                        + """ /\ """
                        + str(ind)
                        + """< s[i] + d[i]) * rr["""
                        + str(index_ressource + 1)
                        + """,i]);\n"""
                    )
                child_instance.add_string(s)
                list_strings += [s]
        satisfiable = [
            (s, f)
            for s, f in last_result_store.list_solution_fits
            if "satisfy" in s.__dict__.keys() and s.satisfy
        ]
        if len(satisfiable) > 0:
            res = ResultStorage(
                list_solution_fits=satisfiable, mode_optim=result_storage.mode_optim
            )
            self.other_constraint.adding_constraint_from_results_store(
                cp_solver, child_instance, res, last_result_store
            )
        self.store_constraints.update(list_strings)
        return self.store_constraints

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_RCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


@deprecated(deprecated_in="0.1")
class SolverWithCalendarIterative(SolverDO):
    def __init__(
        self,
        problem_calendar: RCPSPModelCalendar,
        partial_solution: PartialSolution = None,
        option_neighbor: OptionNeighbor = OptionNeighbor.MIX_ALL,
        **kwargs
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = RCPSPModel(
            resources={
                r: int(self.problem_calendar.get_max_resource_capacity(r))
                for r in self.problem_calendar.resources
            },
            non_renewable_resources=self.problem_calendar.non_renewable_resources,
            mode_details=self.problem_calendar.mode_details,
            successors=self.problem_calendar.successors,
            horizon=self.problem_calendar.horizon,
            name_task=self.problem_calendar.name_task,
        )
        solver = CP_MRCPSP_MZN(
            rcpsp_model=self.problem_no_calendar, cp_solver_name=CPSolverName.CHUFFED
        )
        solver.init_model(
            output_type=True, model_type="multi", partial_solution=partial_solution
        )
        params_objective_function = get_default_objective_setup(
            problem=self.problem_no_calendar
        )
        constraint_handler = build_neighbor_operator(
            option_neighbor=option_neighbor, rcpsp_model=self.problem_no_calendar
        )
        constraint_handler = ConstraintHandlerAddCalendarConstraint(
            self.problem_calendar, self.problem_no_calendar, constraint_handler
        )
        initial_solution_provider = InitialSolutionRCPSP(
            problem=self.problem_calendar,
            initial_method=InitialMethodRCPSP.DUMMY,
            params_objective_function=params_objective_function,
        )
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.params_objective_function = params_objective_function
        self.cp_solver = solver
        self.post_process_solution = PostProcessSolutionNonFeasible(
            self.problem_calendar,
            self.problem_no_calendar,
            partial_solution=partial_solution,
        )
        self.lns_solver = LNS_CP(
            problem=self.problem_calendar,
            cp_solver=self.cp_solver,
            post_process_solution=self.post_process_solution,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            params_objective_function=params_objective_function,
        )

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        skip_first_iteration: bool = False,
        **args
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return self.lns_solver.solve_lns(
            parameters_cp=parameters_cp,
            max_time_seconds=max_time_seconds,
            skip_first_iteration=skip_first_iteration,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            nb_iteration_lns=nb_iteration_lns,
        )
