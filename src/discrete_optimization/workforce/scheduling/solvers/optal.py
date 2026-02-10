#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
try:
    import optalcp as cp
    if TYPE_CHECKING:
        from optalcp import Model as OptalModel, Solution as OptalSolution  # type: ignore
        from optalcp import IntExpr as OptalIntExpr 
except ImportError:
    cp = None
    
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import (
    OptalPythonSolver,
    OptalSolver,
)
from discrete_optimization.generic_tools.hub_solver.optal.model_collections import (
    DoProblemEnum,
    problem_to_script_path,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    export_scheduling_problem_json,
)


def alloc_scheduling_to_dict(problem: AllocSchedulingProblem):
    return export_scheduling_problem_json(problem=problem)


class OptalAllocSchedulingSolverNode(OptalSolver):
    """Solver for Workforce Scheduling using the OptalCP TypeScript API (fallback if Python API is not available)"""
    problem: AllocSchedulingProblem

    def __init__(
        self,
        problem: AllocSchedulingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = problem_to_script_path[DoProblemEnum.WORKFORCE_SCHEDULING]
        self.model_dispersion: bool = False
        self.run_lexico: bool = False

    def init_model(
        self, model_dispersion: bool = False, run_lexico: bool = False, **args: Any
    ) -> None:
        output = alloc_scheduling_to_dict(self.problem)
        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.json")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            json.dump(output, f, indent=4)
        self._file_input = file_input_path
        super().init_model(**args)
        self.model_dispersion = model_dispersion
        self.run_lexico = run_lexico

    def build_command(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        **args: Any,
    ):
        if "model_dispersion" in args:
            args.pop("model_dispersion")
        if "run_lexico" in args:
            args.pop("run_lexico")
        command = super().build_command(
            parameters_cp=parameters_cp, time_limit=time_limit, **args
        )
        if self.model_dispersion:
            command.append("--model-dispersion")
        if self.run_lexico:
            command.append("--run-lexico")
        command += ["--output-json", self._result_path]
        return command

    def retrieve_current_solution(self, dict_results: dict) -> AllocSchedulingSolution:
        starts = dict_results["startTimes"]
        ends = dict_results["endTimes"]
        allocation = dict_results["teamAssignments"]
        return AllocSchedulingSolution(
            problem=self.problem,
            schedule=np.array([[s, e] for s, e in zip(starts, ends)], dtype=int),
            allocation=np.array([self.problem.teams_to_index[t] for t in allocation]),
        )


class OptalAllocSchedulingSolver(OptalPythonSolver):
    """Solver for Workforce Scheduling using the OptalCP Python API (default if OptalCP is installed)"""
    problem: AllocSchedulingProblem

    def __init__(
        self,
        problem: AllocSchedulingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.model_dispersion: bool = False
        self.run_lexico: bool = False

    def init_model(
        self, model_dispersion: bool = False, run_lexico: bool = False, **args: Any
    ) -> None:
        self.model_dispersion = model_dispersion
        self.run_lexico = run_lexico
        super().init_model(**args)

    def build_model(self, **kwargs: Any) -> "OptalModel":
        """Builds the OptalCP model for the Workforce Scheduling problem."""
        model = cp.Model()
        tasks = self.problem.tasks_list
        teams = self.problem.team_names
        tasks_data = self.problem.tasks_data
        compatible_teams = {
            t: set(self.problem.available_team_for_activity[t])
            if t in self.problem.available_team_for_activity
            else set(teams)
            for t in tasks
        }
        successors = self.problem.precedence_constraints
        same_allocation = self.problem.same_allocation
        horizon = self.problem.horizon

        task_vars = {}

        # 1. Create task and mode variables
        for task in tasks:
            duration = tasks_data[task].duration_task
            start_window = self.problem.start_window.get(task, (0, horizon))
            start_window = (int(start_window[0]), int(start_window[1]))
            end_window = self.problem.end_window.get(task, (duration, horizon))
            end_window = (int(end_window[0]), int(end_window[1]))
            # print(f"DEBUG: task={task}, start_window={start_window}, end_window={end_window}")

            # Interval variable for the task itself (regardless of team assignment)
            task_vars[task] = {
                "main": model.interval_var(
                    start=start_window,
                    end=end_window,
                    length=duration,
                    name=f"task_{task}",
                    optional=False,
                ),
                "modes": [],
                "team_vars": {},
            }

            # Optional interval variables for each compatible team (mode)
            compatible = compatible_teams[task]
            for team in compatible:
                mode = model.interval_var(
                    length=duration, optional=True, name=f"task_{task}_team_{team}"
                )
                task_vars[task]["modes"].append(mode)
                task_vars[task]["team_vars"][team] = mode

        # 2. Add constraints
        
        # Alternative constraints: each task must be assigned to exactly one team (mode)
        for task in tasks:
            model.alternative(task_vars[task]["main"], task_vars[task]["modes"])

        # Precedence constraints
        for task in successors:
            for succ in successors[task]:
                model.end_before_start(task_vars[task]["main"], task_vars[succ]["main"])

        # Same allocation constraints: tasks in the same group must be assigned to the same team
        for group in same_allocation:
            group_list = list(group)
            for i in range(len(group_list) - 1):
                task1 = group_list[i]
                task2 = group_list[i + 1]
                common_teams = compatible_teams[task1].intersection(
                    compatible_teams[task2]
                )
                for team in common_teams:
                    model.enforce(
                        model.presence(task_vars[task1]["team_vars"][team])
                        == model.presence(task_vars[task2]["team_vars"][team])
                    )

        # Resource constraints: tasks assigned to the same team cannot overlap, and must respect team availability
        team_optional_tasks = {team: [] for team in teams}
        for task in tasks:
            for team, mode_var in task_vars[task]["team_vars"].items():
                team_optional_tasks[team].append(mode_var)

        # Add no-overlap constraints for each team, considering their availability calendar
        for team in teams:
            intervals_for_team = list(team_optional_tasks[team])
            calendar = self.problem.calendar_team.get(team)
            if calendar:
                last_available_end = 0
                for available_slot in calendar:
                    available_start, available_end = available_slot
                    if available_start > last_available_end:
                        unavailability = model.interval_var(
                            start=(int(last_available_end), int(last_available_end)),
                            length=int(available_start - last_available_end),
                            name=f"unavail_{team}_{last_available_end}",
                            optional=False,
                        )
                        intervals_for_team.append(unavailability)
                    last_available_end = available_end

                if last_available_end < horizon:
                    unavailability = model.interval_var(
                        start=(int(last_available_end), int(last_available_end)),
                        length=int(horizon - last_available_end),
                        name=f"unavail_{team}_{last_available_end}_final",
                        optional=False,
                    )
                    intervals_for_team.append(unavailability)
            model.no_overlap(intervals_for_team)

        # 3. Define the objective: Minimize the number of teams used
        teams_used_vars = []
        for team in teams:
            # Define a binary variable that indicates whether the team is used (at least one task assigned)
            team_used = model.int_var(min=0, max=1, name=f"used_{team}")
            team_tasks = team_optional_tasks[team]
            # model.max([itv.presence() for itv in team_tasks]) == team_used
            # But max with list of expressions might be tricky if some are bool expressions (?)
            # Better use model.enforce(team_used == model.max([itv.presence() for itv in team_tasks]))
            # if optalcp supports max over BoolExpr...
            if team_tasks:
                model.enforce(
                    team_used == model.max([model.presence(itv) for itv in team_tasks])
                )
            else:
                model.enforce(team_used == 0)
            teams_used_vars.append(team_used)
        # Compute the total number of teams used
        nb_teams_used = model.sum(teams_used_vars)

        self._task_vars = task_vars
        self._teams_used_vars = teams_used_vars
        self._nb_teams_used = nb_teams_used
        
        # Store model for solve() in case we need Stage 2
        self.model = model

        # TODO: Need review, since we assume that the run_lexico option only makes sense if model_dispersion is True
        # Logic for objectives:
        # 1. model_dispersion=False: Minimize nb_teams_used (single Stage).
        # 2. model_dispersion=True, run_lexico=False: Weighted Sum (single Stage).
        # 3. model_dispersion=True, run_lexico=True: Lexicographical Stage 1 (min teams only).
        
        if self.model_dispersion and not self.run_lexico:
            # Weighted Sum approach (Soft Lexicographical)
            dispersion = self._get_workload_dispersion(model)
            model.minimize(nb_teams_used * 10000 + dispersion)
        else:
            # Regular minimization (either primary objective or Stage 1 of Lexico)
            model.minimize(nb_teams_used)

        return model

    def _get_workload_dispersion(self, model: "OptalModel") -> "OptalIntExpr":
        """Defines workload variables and returns the dispersion expression (max - min).
        
        Workload for a team is defined as the sum of durations of tasks assigned to it.
        Dispersion is the difference between the max and min workload across all teams.
        """
        teams = self.problem.team_names
        tasks = self.problem.tasks_list
        tasks_data = self.problem.tasks_data
        horizon = self.problem.horizon
        
        workload_per_team = []
        for i, team in enumerate(teams):
            workload = model.int_var(min=0, max=horizon, name=f"workload_{team}")
            
            # Sum of durations for tasks where this team is present
            team_modes_with_durations = [
                (self._task_vars[task]["team_vars"][team], tasks_data[task].duration_task)
                for task in tasks if team in self._task_vars[task]["team_vars"]
            ]
            
            workload_expr = model.sum(
                [model.presence(itv) * dur for itv, dur in team_modes_with_durations]
            )
            
            # Constrain workload: if team is used, workload matches the sum.
            # If team is NOT used, workload becomes 0 (sum over empty/absent modes).
            model.enforce(workload == workload_expr)
            workload_per_team.append(workload)
            
        return model.max(workload_per_team) - model.min(workload_per_team)

    def solve(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Optimizes the problem using either single-stage or two-stage (Lexicographical) strategy.
        
        If model_dispersion and run_lexico are both True:
        - Stage 1: Minimize number of teams used.
        - Stage 2: Minimize workload dispersion while maintaining the minimum teams from Stage 1.
        """
        # If not Lexicographical Workload Balancing, solve once with the model built in build_model()
        if not (self.run_lexico and self.model_dispersion):
            return super().solve(parameters_cp=parameters_cp, **kwargs)
        
        # --- Stage 1: Minimize nb_teams_used ---
        res1 = super().solve(parameters_cp=parameters_cp, **kwargs) 
        if not res1.list_solution_fits:
            return res1 # No solution found in Stage 1, return results as is.
                
        # --- Stage 2: Minimize dispersion subject to min_teams ---
        if self._stats["objectiveHistory"]:
            # Add constraintes to enforce the minimum number of teams found in Stage 1
            min_team = int(self._stats["objectiveHistory"][-1]["objective"])
            print(f"Stage 1 optimal number of teams: {min_team}")
            self.model.enforce(self._nb_teams_used <= min_team)
        # Add dispersion constrainst and set new objective
        dispersion = self._get_workload_dispersion(self.model)
        self.model.minimize(dispersion)
        
        # Final solve for the secondary objective
        return super().solve(parameters_cp=parameters_cp, **kwargs)
    
    def retrieve_current_solution(self, solution: "OptalSolution") -> AllocSchedulingSolution:
        """Extracts the schedule from the OptalCP solution and constructs an AllocSchedulingSolution."""
        tasks = self.problem.tasks_list
        teams = self.problem.team_names
        
        start_times = []
        end_times = []
        team_assignments = []
        
        for task in tasks:
            assigned_team = -1
            start_time = -1
            end_time = -1
            
            for i, team in enumerate(teams):
                mode_var = self._task_vars[task]["team_vars"].get(team)
                if mode_var and solution.is_present(mode_var):
                    assigned_team = i
                    start_time = solution.get_start(mode_var)
                    end_time = solution.get_end(mode_var)
                    break
            
            start_times.append(start_time)
            end_times.append(end_time)
            team_assignments.append(assigned_team)
            
        return AllocSchedulingSolution(
            problem=self.problem,
            schedule=np.array([[s, e] for s, e in zip(start_times, end_times)], dtype=int),
            allocation=np.array(team_assignments, dtype=int),
        )
