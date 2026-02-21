#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Union

import numpy as np

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cp_mzn import CpMultimodeRcpspSolver
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRcpsp
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    PreemptiveMultiskillRcpspSolution,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    CpMultiskillRcpspSolver,
    CpPreemptiveMultiskillRcpspSolver,
    CpSolverName,
    stick_to_solution,
    stick_to_solution_preemptive,
)

logger = logging.getLogger(__name__)


class MultimodeTranspositionMultiskillRcpspSolver(SolverDO):
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        multimode_problem: Union[RcpspProblem, PreemptiveRcpspProblem] = None,
        worker_type_to_worker: dict[str, set[Union[str, int]]] = None,
        params_objective_function: ParamsObjectiveFunction = None,
        solver_multimode_rcpsp: SolverDO = None,
        limit_number_of_mode_per_task: bool = True,
        max_number_of_mode: int = 3,
        check_resource_compliance: bool = True,
        reconstruction_cp_time_limit: int = 3600,
        **kwargs,
    ):
        """Initialize the multimode transposition solver.
        
        Args:
            problem: The multi-skill RCPSP problem to solve
            multimode_problem: Pre-computed multimode RCPSP problem (optional)
            worker_type_to_worker: Pre-computed worker type to worker mapping (optional)
            params_objective_function: Parameters for objective function
            solver_multimode_rcpsp: Solver for the multimode RCPSP problem
            limit_number_of_mode_per_task: Whether to limit modes per task in transformation
            max_number_of_mode: Maximum number of modes per task during transformation
            check_resource_compliance: Whether to check resource compliance during transformation
            reconstruction_cp_time_limit: Time limit (seconds) for CP-based reconstruction
            **kwargs: Additional arguments for parent class
        """
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.multimode_problem = multimode_problem
        self.worker_type_to_worker = worker_type_to_worker
        self.solver_multimode_rcpsp = solver_multimode_rcpsp
        self.limit_number_of_mode_per_task = limit_number_of_mode_per_task
        self.max_number_of_mode = max_number_of_mode
        self.check_resource_compliance = check_resource_compliance
        self.reconstruction_cp_time_limit = reconstruction_cp_time_limit

    def solve(self, **kwargs) -> ResultStorage:
        """Solve the multi-skill RCPSP using two-stage decomposition.
        The idea is to transform MS-RCPSP to multi-mode RCPSP by abstracting employees as worker types.

        Returns:
            ResultStorage containing the reconstructed multi-skill RCPSP solutions
        """
        # Construct the multimode problem if not already given
        if self.multimode_problem is None or self.worker_type_to_worker is None:
            logger.info("Construct RCPSP problem by abstracting employees as worker types")
            algo = MultiSkillToRcpsp(self.problem)
            rcpsp_problem = algo.construct_rcpsp_by_worker_type(
                limit_number_of_mode_per_task=self.limit_number_of_mode_per_task,
                max_number_of_mode=self.max_number_of_mode,
                check_resource_compliance=self.check_resource_compliance,
            )
            logger.info(f"RCPSP Problem Info: {rcpsp_problem}")
            self.multimode_problem = rcpsp_problem
            self.worker_type_to_worker = algo.worker_type_to_worker

        if self.solver_multimode_rcpsp is None:
            # Create an underlying solver for the multimode RCPSP problem if not provided
            self.solver_multimode_rcpsp = CpSatRcpspSolver(
                problem=self.multimode_problem
            )
            logger.info("Create an underlying solver for the multimode RCPSP problem")
        else:
            # Set the problem on the underlying solver if not already set
            if self.solver_multimode_rcpsp.problem is None:
                self.solver_multimode_rcpsp.problem = self.multimode_problem
                logger.info("Set the problem on the given underlying solver")

        # Solve the multimode problem
        logger.info("Solving the multi-mode RCPSP problem...")
        result_store = self.solver_multimode_rcpsp.solve(**kwargs)
        solution, fit = result_store.get_best_solution_fit()
        
        # Check if a solution was found
        if solution is None:
            logger.warning("No solution found by the multi-mode RCPSP solver. Returning empty result store.")
            return ResultStorage(
                list_solution_fits=[],
                mode_optim=self.problem.get_objective_register().objective_sense,
            )
        
        logger.info(f"Multi-mode RCPSP solution obtained with objective/fit: {fit}")
        
        # Rebuild the solution for the multiskill problem using the solution of the rcpsp problem
        logger.info("Rebuild solution for the multiskill problem")
        res = rebuild_multiskill_solution_cp_based(
            multiskill_rcpsp_problem=self.problem,
            multimode_rcpsp_problem=self.multimode_problem,
            worker_type_to_worker=self.worker_type_to_worker,
            solution_rcpsp=solution,
            time_limit=self.reconstruction_cp_time_limit,
        )
        return res


def rebuild_multiskill_solution(
    multiskill_rcpsp_problem: MultiskillRcpspProblem,
    multimode_rcpsp_problem: Union[RcpspProblem, PreemptiveRcpspProblem],
    worker_type_to_worker: dict[str, set[Union[str, int]]],
    solution_rcpsp: Union[RcpspSolution, PreemptiveRcpspSolution],
) -> Union[MultiskillRcpspSolution, PreemptiveMultiskillRcpspSolution]:
    """
    This function takes the schedule from the RCPSP solution and rebuilds the solution for the multiskill problem.
    NOTE: need review, this function is currently not used.
    """
    new_horizon = multimode_rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in multimode_rcpsp_problem.resources_list:
        if multimode_rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = multimode_rcpsp_problem.resources[res][
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, multimode_rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    worker_avail_in_time = {}
    for i in multiskill_rcpsp_problem.employees:
        worker_avail_in_time[i] = np.array(
            multiskill_rcpsp_problem.employees[i].calendar_employee[: new_horizon + 1],
            dtype=np.bool_,
        )
    rcpsp_schedule = solution_rcpsp.rcpsp_schedule
    employee_usage = {}
    modes_dict = multimode_rcpsp_problem.build_mode_dict(solution_rcpsp.rcpsp_modes)
    sorted_tasks = sorted(rcpsp_schedule, key=lambda x: solution_rcpsp.get_end_time(x))
    for task in sorted_tasks:
        employee_usage[task] = {}
        ressource_requirements = multimode_rcpsp_problem.mode_details[task][
            modes_dict[task]
        ]
        non_zeros_ressource_requirements = set(
            [
                k
                for k in ressource_requirements
                if k in worker_type_to_worker and ressource_requirements[k] > 0
            ]
        )
        if len(non_zeros_ressource_requirements) >= 1:
            active_times = solution_rcpsp.get_active_time(task)
            for k in non_zeros_ressource_requirements:
                number_worker = ressource_requirements[k]
                workers_available = [
                    w
                    for w in worker_type_to_worker[k]
                    if all(worker_avail_in_time[w][i] for i in active_times)
                ]
                if len(workers_available) >= number_worker:
                    wavail = workers_available[:number_worker]
                    skills_needed_by_task = [
                        s
                        for s in multiskill_rcpsp_problem.mode_details[task][1]
                        if s in multiskill_rcpsp_problem.skills_set
                        and multiskill_rcpsp_problem.mode_details[task][1][s] > 0
                    ]
                    non_zeros = multiskill_rcpsp_problem.employees[
                        wavail[0]
                    ].get_non_zero_skills()
                    skills_interest = [
                        s for s in non_zeros if s in skills_needed_by_task
                    ]
                    employee_usage[task].update(
                        {emp: set(skills_interest) for emp in wavail}
                    )
                    for i in active_times:
                        for w in wavail:
                            worker_avail_in_time[w][i] = False
                else:
                    if isinstance(solution_rcpsp, PreemptiveRcpspSolution):
                        for s, e in zip(
                            solution_rcpsp.rcpsp_schedule[task]["starts"],
                            solution_rcpsp.rcpsp_schedule[task]["ends"],
                        ):
                            at = range(s, e)
                            workers_available = [
                                w
                                for w in worker_type_to_worker[k]
                                if all(worker_avail_in_time[w][i] for i in at)
                            ]
                            if len(workers_available) >= number_worker:
                                wavail = workers_available[:number_worker]
                                skills_needed_by_task = [
                                    s
                                    for s in multiskill_rcpsp_problem.mode_details[
                                        task
                                    ][1]
                                    if s in multiskill_rcpsp_problem.skills_set
                                    and multiskill_rcpsp_problem.mode_details[task][1][
                                        s
                                    ]
                                    > 0
                                ]
                                non_zeros = multiskill_rcpsp_problem.employees[
                                    wavail[0]
                                ].get_non_zero_skills()
                                skills_interest = [
                                    s for s in non_zeros if s in skills_needed_by_task
                                ]
                                for emp in wavail:
                                    if emp not in employee_usage[task]:
                                        employee_usage[task][emp] = {
                                            "skills": [],
                                            "times": [],
                                        }
                                    employee_usage[task][emp]["skills"] += [
                                        set(skills_interest)
                                    ]
                                    employee_usage[task][emp]["times"] += [(s, e)]
                                for i in at:
                                    for w in wavail:
                                        worker_avail_in_time[w][i] = False
                            else:
                                logger.warning("Problem finding a worker")

    if isinstance(solution_rcpsp, PreemptiveRcpspSolution):
        return PreemptiveMultiskillRcpspSolution(
            problem=multiskill_rcpsp_problem,
            modes={task: 1 for task in multiskill_rcpsp_problem.tasks_list},
            employee_usage=employee_usage,
            schedule=rcpsp_schedule,
        )
    else:
        return MultiskillRcpspSolution(
            problem=multiskill_rcpsp_problem,
            modes={task: 1 for task in multiskill_rcpsp_problem.tasks_list},
            employee_usage=employee_usage,
            schedule=rcpsp_schedule,
        )


def rebuild_multiskill_solution_cp_based(
    multiskill_rcpsp_problem: MultiskillRcpspProblem,
    multimode_rcpsp_problem: Union[RcpspProblem, PreemptiveRcpspProblem],   # TODO: need review, currently unused
    worker_type_to_worker: dict[str, set[Union[str, int]]],                 # TODO: need review, currently unused
    solution_rcpsp: Union[RcpspSolution, PreemptiveRcpspSolution],
    time_limit: int = 3600,
) -> ResultStorage:
    """
    Reconstructs a valid multi-skill RCPSP solution from the schedule from an RCPSP solution 
    by assigning specific employees to tasks.
    
    Args:
        multiskill_rcpsp_problem: The multi-skill RCPSP problem
        multimode_rcpsp_problem: The multimode RCPSP problem (for reference)
        worker_type_to_worker: Mapping from worker types to employee sets
        solution_rcpsp: The RCPSP solution to constrain to
        time_limit: Time limit in seconds for CP solver
        
    Returns:
        ResultStorage: containing the reconstructed multi-skill RCPSP solutions
    """
    if isinstance(solution_rcpsp, RcpspSolution):
        model = CpMultiskillRcpspSolver(
            problem=multiskill_rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED
        )
        model.init_model(
            add_calendar_constraint_unit=False,
            fake_tasks=True,
            one_ressource_per_task=False,
            exact_skills_need=False,
            output_type=True,
        )
        # Constraint to stick to the solution of the RCPSP problem
        strings = stick_to_solution(solution_rcpsp, model)
        for s in strings:
            model.instance.add_string(s)
    else:
        model = CpPreemptiveMultiskillRcpspSolver(
            problem=multiskill_rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED
        )
        model.init_model(
            add_calendar_constraint_unit=False,
            fake_tasks=True,
            exact_skills_need=False,
            one_ressource_per_task=False,
            output_type=True,
            nb_preemptive=10,
            unit_usage_preemptive=True,
            max_preempted=100,
        )
        # Constraint to stick to the solution of the RCPSP problem
        strings = stick_to_solution_preemptive(solution_rcpsp, model)
        for s in strings:
            model.instance.add_string(s)

    # Solve the CP model
    result_store = model.solve(time_limit=time_limit)
    return result_store
