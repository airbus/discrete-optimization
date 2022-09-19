#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
import time
from datetime import timedelta
from typing import Any, Iterable, Optional, Union

import networkx as nx
import numpy as np
from deprecation import deprecated
from minizinc import Instance, Status

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler, SolverDO
from discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    schedule_solution_to_variant,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    PartialSolution,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_cp_lns_solver import (
    OptionNeighbor,
    build_neighbor_operator,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionMS_RCPSP,
)

logger = logging.getLogger(__name__)


@deprecated(deprecated_in="0.1")
def get_ressource_breaks(problem_calendar: MS_RCPSPModel, solution: MS_RCPSPSolution):
    ressources = problem_calendar.resources_availability
    ressource_arrays = {}
    ressource_arrays_usage = {}
    employees_arrays = {}
    employees_arrays_usage = {}
    for r in ressources:
        ressource_arrays[r] = np.zeros(
            (len(problem_calendar.resources_availability[r]))
        )
        ressource_arrays_usage[r] = np.zeros(
            (len(problem_calendar.resources_availability[r]), len(solution.schedule))
        )
    for emp in problem_calendar.employees:
        employees_arrays[emp] = np.zeros(
            (len(problem_calendar.employees[emp].calendar_employee))
        )
        employees_arrays_usage[emp] = np.zeros(
            (
                len(problem_calendar.employees[emp].calendar_employee),
                len(solution.schedule),
            )
        )
    sorted_keys_schedule = sorted(solution.schedule.keys())
    for ji in range(len(sorted_keys_schedule)):
        j = sorted_keys_schedule[ji]
        for r in problem_calendar.mode_details[j][1]:
            if r == "duration":
                continue
            if r is None:
                continue
            if r == "None":
                continue
            if r not in ressource_arrays_usage:
                continue
            if problem_calendar.mode_details[j][1][r] == 0:
                continue
            if solution.schedule[j]["end_time"] == solution.schedule[j]["start_time"]:
                continue
            ressource_arrays_usage[r][
                int(solution.schedule[j]["start_time"]) : int(
                    solution.schedule[j]["end_time"]
                ),
                ji,
            ] = 1
            ressource_arrays[r][
                int(solution.schedule[j]["start_time"]) : int(
                    solution.schedule[j]["end_time"]
                )
            ] += problem_calendar.mode_details[j][1][r]

    for ji in range(len(sorted_keys_schedule)):
        j = sorted_keys_schedule[ji]
        emp_usage = solution.employee_usage.get(j, {})
        for emp in emp_usage:
            if len(emp_usage[emp]) > 0:
                employees_arrays_usage[emp][
                    int(solution.schedule[j]["start_time"]) : int(
                        solution.schedule[j]["end_time"]
                    ),
                    ji,
                ] = 1
                employees_arrays[emp][
                    int(solution.schedule[j]["start_time"]) : int(
                        solution.schedule[j]["end_time"]
                    )
                ] += 1
    index_ressource = {}
    task_concerned = {}
    constraints = {}
    for r in ressource_arrays:
        index = np.argwhere(
            ressource_arrays[r] > problem_calendar.resources_availability[r]
        )
        index_ressource[r] = index
        task_concerned[r] = [
            j
            for j in range(ressource_arrays_usage[r].shape[1])
            if any(
                ressource_arrays_usage[r][ind[0], j] == 1
                for ind in index
                if problem_calendar.resources_availability[r][ind[0]] == 0
            )
        ]
        task_concerned[r] = [sorted_keys_schedule[rr] for rr in task_concerned[r]]
        constraints[r] = {}
        for t in task_concerned[r]:
            current_start = solution.schedule[t]["start_time"]
            first_possible_start_future = next(
                (
                    st
                    for st in range(
                        current_start, len(problem_calendar.resources_availability[r])
                    )
                    if problem_calendar.resources_availability[r][st]
                    >= problem_calendar.mode_details[t][1][r]
                ),
                None,
            )
            first_possible_start_before = next(
                (
                    st - problem_calendar.mode_details[t][1]["duration"] + 1
                    for st in range(solution.schedule[t]["end_time"], -1, -1)
                    if problem_calendar.resources_availability[r][st - 1]
                    >= problem_calendar.mode_details[t][1][r]
                    and problem_calendar.resources_availability[r][
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

    constraints_employee = {}
    for emp in employees_arrays:
        index = np.argwhere(
            employees_arrays[emp]
            > 1 * problem_calendar.employees[emp].calendar_employee
        )
        index_ressource[emp] = index
        task_concerned[emp] = [
            j
            for j in range(employees_arrays_usage[emp].shape[1])
            if any(
                employees_arrays_usage[emp][ind[0], j] == 1
                for ind in index
                if not problem_calendar.employees[emp].calendar_employee[ind[0]]
            )
        ]
        task_concerned[emp] = [sorted_keys_schedule[rr] for rr in task_concerned[emp]]
        constraints_employee[emp] = {}
        for t in task_concerned[emp]:
            current_start = solution.schedule[t]["start_time"]
            first_possible_start_future = next(
                (
                    st
                    for st in range(
                        current_start,
                        len(problem_calendar.employees[emp].calendar_employee),
                    )
                    if problem_calendar.employees[emp].calendar_employee[st]
                ),
                None,
            )
            first_possible_start_before = next(
                (
                    st
                    for st in range(current_start, -1, -1)
                    if problem_calendar.employees[emp].calendar_employee[st]
                ),
                None,
            )
            constraints_employee[emp] = (
                first_possible_start_before,
                first_possible_start_future,
            )
    return index_ressource, constraints, constraints_employee


@deprecated(deprecated_in="0.1")
class PostProcessSolutionNonFeasible(PostProcessSolution):
    def __init__(
        self,
        problem_calendar: MS_RCPSPModel,
        problem_no_calendar: MS_RCPSPModel,
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
        self.graph = self.problem_calendar.compute_graph()
        self.successors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.predecessors = {
            n: nx.algorithms.descendants(self.graph.graph_nx, n)
            for n in self.graph.graph_nx.nodes()
        }
        self.immediate_predecessors = {
            n: self.graph.get_predecessors(n) for n in self.graph.nodes_name
        }

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        for sol in list(result_storage.list_solution_fits):
            if "satisfy" not in sol[0].__dict__.keys():
                rb, constraints, constraint_emp = get_ressource_breaks(
                    self.problem_calendar, sol[0]
                )
                sol[0].satisfy = not (any(len(rb[r]) > 0 for r in rb))
                sol[0].satisfy = self.problem_calendar.satisfy(sol[0])
                sol[0].constraints = constraints
            if self.partial_solution is None:
                s: MS_RCPSPSolution = sol[0]
                s.problem = self.problem_calendar
                solution = schedule_solution_to_variant(s)
                solution.satisfy = self.check_sol(self.problem_calendar, solution)
                solution.satisfy = self.problem_calendar.satisfy(solution)
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
        problem_calendar: MS_RCPSPModel,
        problem_no_calendar: MS_RCPSPModel,
        other_constraint: ConstraintHandler,
    ):
        self.problem_calendar: MS_RCPSPModel = problem_calendar
        self.problem_no_calendar = problem_no_calendar
        self.other_constraint = other_constraint

    def adding_constraint_from_results_store(
        self,
        cp_solver: Union[CP_MS_MRCPSP_MZN],
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_storage: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        solution, fit = result_storage.get_best_solution_fit()
        solution: MS_RCPSPSolution = solution
        if "satisfy" in solution.__dict__.keys() and solution.satisfy:
            return self.other_constraint.adding_constraint_from_results_store(
                cp_solver,
                child_instance,
                ResultStorage(
                    list_solution_fits=[(solution, fit)],
                    mode_optim=result_storage.mode_optim,
                ),
                last_result_storage,
            )
        ressource_breaks, constraints, constraints_employee = get_ressource_breaks(
            self.problem_calendar, solution
        )
        list_strings = []
        tasks = sorted(self.problem_calendar.mode_details.keys())
        for r in constraints:
            for t in constraints[r]:
                index = tasks.index(t)
                s = None
                if isinstance(cp_solver, CP_MS_MRCPSP_MZN):
                    if (
                        constraints[r][t][0] is not None
                        and constraints[r][t][1] is not None
                    ):
                        s = (
                            """constraint start["""
                            + str(index + 1)
                            + """]<="""
                            + str(constraints[r][t][0])
                            + " \/ "
                            "start["
                            ""
                            + str(index + 1)
                            + """]>="""
                            + str(constraints[r][t][1])
                            + """;\n"""
                        )
                    elif (
                        constraints[r][t][0] is None
                        and constraints[r][t][1] is not None
                    ):
                        s = (
                            """constraint start["""
                            + str(index + 1)
                            + """]>="""
                            + str(constraints[r][t][1])
                            + """;\n"""
                        )
                    elif (
                        constraints[r][t][0] is not None
                        and constraints[r][t][1] is None
                    ):
                        s = (
                            """constraint start["""
                            + str(index + 1)
                            + """]<="""
                            + str(constraints[r][t][0])
                            + """;\n"""
                        )
                if s is not None:
                    child_instance.add_string(s)
                    list_strings += [s]
        for r in ressource_breaks:
            for index in ressource_breaks[r]:
                if random.random() < 0.3:
                    continue
                ind = index[0]
                if r in self.problem_calendar.resources_availability:
                    index_ressource = cp_solver.resources_index.index(r)
                    rq = self.problem_calendar.resources_availability[r][ind]
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
                    child_instance.add_string(s)
                    list_strings += [s]
                if r in self.problem_calendar.employees:
                    index_ressource = cp_solver.employees_position.index(r)
                    rq = int(self.problem_calendar.employees[r].calendar_employee[ind])
                    s = (
                        """constraint """
                        + str(rq)
                        + """>=sum( i in Act ) (
                                        bool2int(start[i] <="""
                        + str(ind)
                        + """ /\ """
                        + str(ind)
                        + """< start[i] + adur[i]) * unit_used["""
                        + str(index_ressource + 1)
                        + """,i]);\n"""
                    )
                    child_instance.add_string(s)
                    list_strings += [s]

        satisfiable = [
            (s, f)
            for s, f in result_storage.list_solution_fits
            if "satisfy" in s.__dict__.keys() and s.satisfy
        ]
        if len(satisfiable) > 0:
            res = ResultStorage(
                list_solution_fits=satisfiable, mode_optim=result_storage.mode_optim
            )
            self.other_constraint.adding_constraint_from_results_store(
                cp_solver, child_instance, res, last_result_storage
            )
        return ["req"] + list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CP_MS_MRCPSP_MZN,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


@deprecated(deprecated_in="0.1")
class SolverWithCalendarIterative(SolverDO):
    def __init__(
        self,
        problem_calendar: MS_RCPSPModel,
        partial_solution: PartialSolution = None,
        option_neighbor: OptionNeighbor = OptionNeighbor.MIX_FAST,
        **kwargs,
    ):
        self.problem_calendar = problem_calendar
        self.problem_no_calendar = self.problem_calendar.copy()
        self.problem_no_calendar = MS_RCPSPModel(
            skills_set=self.problem_calendar.skills_set,
            resources_set=self.problem_calendar.resources_set,
            non_renewable_resources=self.problem_calendar.non_renewable_resources,
            resources_availability={
                r: [int(max(self.problem_calendar.resources_availability[r]))]
                * len(self.problem_calendar.resources_availability[r])
                for r in self.problem_calendar.resources_availability
            },
            employees={
                emp: self.problem_calendar.employees[emp].copy()
                for emp in self.problem_calendar.employees
            },
            employees_availability=self.problem_calendar.employees_availability,
            mode_details=self.problem_calendar.mode_details,
            successors=self.problem_calendar.successors,
            horizon=self.problem_calendar.horizon,
            horizon_multiplier=self.problem_calendar.horizon_multiplier,
        )
        for emp in self.problem_calendar.employees:
            self.problem_no_calendar.employees[emp].calendar_employee = [
                max(self.problem_calendar.employees[emp].calendar_employee)
            ] * len(self.problem_calendar.employees[emp].calendar_employee)
        solver = CP_MS_MRCPSP_MZN(
            rcpsp_model=self.problem_no_calendar,
            cp_solver_name=CPSolverName.CHUFFED,
            **kwargs,
        )
        solver.init_model(
            output_type=True,
            model_type="single",
            partial_solution=partial_solution,
            add_calendar_constraint_unit=False,
            exact_skills_need=False,
            **kwargs,
        )
        parameters_cp = ParametersCP.default()
        parameters_cp.time_limit = 500
        parameters_cp.time_limit_iter0 = 500
        params_objective_function = get_default_objective_setup(
            problem=self.problem_no_calendar
        )
        constraint_handler = build_neighbor_operator(
            option_neighbor=option_neighbor, rcpsp_model=self.problem_no_calendar
        )
        constraint_handler = ConstraintHandlerAddCalendarConstraint(
            self.problem_calendar, self.problem_no_calendar, constraint_handler
        )
        initial_solution_provider = InitialSolutionMS_RCPSP(
            problem=self.problem_calendar,
            initial_method=InitialMethodRCPSP.PILE_CALENDAR,
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

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        skip_first_iteration: bool = False,
        **args,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        sense = self.params_objective_function.sense_function
        if max_time_seconds is None:
            max_time_seconds = 3600 * 24  # One day
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2 * nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        deb_time = time.time()
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            store_lns = self.post_process_solution.build_other_solution(store_lns)
            store_with_all = ResultStorage(
                list(store_lns.list_solution_fits), mode_optim=store_lns.mode_optim
            )
            init_solution, objective = store_lns.get_best_solution_fit()
            best_solution = init_solution.copy()
            satisfy = self.problem_calendar.satisfy(init_solution)
            logger.debug(f"Satisfy {satisfy}")
            best_objective = objective
        else:
            best_objective = (
                float("inf") if sense == ModeOptim.MINIMIZATION else -float("inf")
            )
            best_solution = None
            store_lns = None
            store_with_all = None
        constraint_to_keep = set()
        for iteration in range(nb_iteration_lns):
            logger.debug(
                (
                    "Starting iteration n°",
                    iteration,
                    " current objective ",
                    best_objective,
                )
            )
            try:
                logger.debug(
                    (
                        "Best feasible solution ",
                        max(
                            [
                                f
                                for s, f in store_with_all.list_solution_fits
                                if "satisfy" in s.__dict__.keys() and s.satisfy
                            ]
                        ),
                    )
                )
            except:
                logger.warning("No Feasible solution yet")
            with self.cp_solver.instance.branch() as child:
                if iteration == 0 and not skip_first_iteration or iteration >= 1:
                    for c in constraint_to_keep:
                        if random.random() < 1.0:
                            child.add_string(c)
                    constraint_iterable = (
                        self.constraint_handler.adding_constraint_from_results_store(
                            cp_solver=self.cp_solver,
                            child_instance=child,
                            result_storage=store_lns,
                        )
                    )
                    if constraint_iterable[0] == "req":
                        constraint_to_keep.update(
                            set([c for c in constraint_iterable[1:]])
                        )
                try:
                    if iteration == 0:
                        result = child.solve(
                            timeout=timedelta(seconds=parameters_cp.time_limit_iter0),
                            intermediate_solutions=parameters_cp.intermediate_solution,
                        )
                    else:
                        result = child.solve(
                            timeout=timedelta(seconds=parameters_cp.time_limit),
                            intermediate_solutions=parameters_cp.intermediate_solution,
                        )
                    result_store = self.cp_solver.retrieve_solutions(
                        result, parameters_cp=parameters_cp
                    )
                    logger.debug(f"iteration n°{iteration} Solved !!!")
                    logger.debug(result.status)
                    if len(result_store.list_solution_fits) > 0:
                        logger.debug("Solved !!!")
                        bsol, fit = result_store.get_best_solution_fit()
                        logger.debug(f"Fitness = {fit}")
                        logger.debug("Post Process..")
                        logger.debug("Satisfy best current sol : ")
                        logger.debug(self.problem_calendar.satisfy(bsol))
                        result_store = self.post_process_solution.build_other_solution(
                            result_store
                        )
                        bsol, fit = result_store.get_best_solution_fit()
                        logger.debug(f"After postpro = {fit}")
                        if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                            if fit > best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_solution = bsol
                            best_objective = fit
                        elif sense == ModeOptim.MAXIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        elif sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                            if fit < best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_solution = bsol
                            best_objective = fit
                        elif sense == ModeOptim.MINIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        if skip_first_iteration and iteration == 0:
                            store_lns = result_store
                            store_with_all = ResultStorage(
                                list_solution_fits=list(store_lns.list_solution_fits),
                                mode_optim=store_lns.mode_optim,
                            )
                        store_lns = result_store
                        for s, f in store_lns.list_solution_fits:
                            store_with_all.list_solution_fits += [(s, f)]
                        for s, f in store_with_all.list_solution_fits:
                            if s.satisfy:
                                store_lns.list_solution_fits += [(s, f)]
                    else:
                        current_nb_iteration_no_improvement += 1
                    if (
                        skip_first_iteration
                        and result.status == Status.OPTIMAL_SOLUTION
                        and iteration == 0
                        and best_solution.satisfy
                    ):
                        logger.info("Finish LNS because found optimal solution")
                        break
                except Exception as e:
                    current_nb_iteration_no_improvement += 1
                    logger.warning(f"Failed ! reason : {e}")
                if time.time() - deb_time > max_time_seconds:
                    logger.info("Finish LNS with time limit reached")
                    break
                logger.debug(
                    f"{current_nb_iteration_no_improvement} / {nb_iteration_no_improvement}"
                )
                if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                    logger.info("Finish LNS with maximum no improvement iteration ")
                    break
        return store_with_all
