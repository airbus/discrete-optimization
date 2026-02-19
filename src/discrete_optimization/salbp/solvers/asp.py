#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional

import clingo

from discrete_optimization.generic_tools.asp_tools import AspClingoSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.salbp.problem import SalbpProblem, SalbpSolution

logger = logging.getLogger(__name__)


class AspSalbpSolver(AspClingoSolver, WarmstartMixin):
    """Solver based on Answer Set Programming for the Simple Assembly Line Balancing Problem."""

    problem: SalbpProblem
    upper_bound: int

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.basic_model = ""
        self.string_data_input = ""

    def init_model(self, **kwargs: Any) -> None:
        # A safe upper bound for stations is the number of tasks
        self.upper_bound = kwargs.get("upper_bound", self.problem.number_of_tasks)

        # 1. Define the ASP program for SALBP
        self.basic_model = """
        % --- Domains ---
        station(0..nb_stations-1).

        % --- Generator ---
        % Assign every task to exactly one station
        1 { assign(T, S) : station(S) } 1 :- task(T, _).

        % --- Constraints ---

        % 1. Cycle Time Constraint: Sum of durations in a station <= cycle_time
        :- station(S), #sum { D, T : assign(T, S), task(T, D) } > max_cycle_time.

        % 2. Precedence Constraint: Pred station <= Succ station
        :- precede(T1, T2), assign(T1, S1), assign(T2, S2), S1 > S2.

        % --- Auxiliary predicates for optimization ---
        used(S) :- assign(_, S).

        % Symmetry Breaking: Use lower-indexed stations first
        :- used(S), not used(S-1), S > 0.

        % --- Optimization ---
        % Minimize the number of used stations
        #minimize { 1, S : used(S) }.

        #show assign/2.
        """

        string_data_input = self.build_string_data_input()

        # Warmstart/Heuristics handling
        solution = kwargs.get("solution", None)
        flags_clingo = [
            "--warn=no-atom-undefined",
            "--opt-mode=optN",
            "--parallel-mode=10",
        ]

        if solution is not None:
            heuristic_str = self.build_heuristic_input(solution)
            string_data_input += "\n" + heuristic_str
            flags_clingo.append("--heuristic=Domain")

        self.ctl = clingo.Control(flags_clingo)
        self.ctl.add("base", [], self.basic_model)
        self.ctl.add("base", [], string_data_input)
        self.string_data_input = string_data_input

    def build_string_data_input(self) -> str:
        """Converts SalbpProblem instance data into ASP facts."""
        facts = []

        # Constants
        facts.append(f"#const max_cycle_time={self.problem.cycle_time}.")
        facts.append(f"#const nb_tasks={self.problem.number_of_tasks}.")
        facts.append(f"#const nb_stations={self.upper_bound}.")

        # Task facts: task(TaskID, Duration)
        for t_id, duration in self.problem.task_times.items():
            facts.append(f"task({t_id}, {duration}).")

        # Precedence facts: precede(PredID, SuccID)
        for p, s in self.problem.precedence:
            facts.append(f"precede({p}, {s}).")

        return "\n".join(facts)

    def build_heuristic_input(self, solution: SalbpSolution) -> str:
        """Builds clingo heuristics based on an existing solution."""
        heuristics = []
        for i, task_id in enumerate(self.problem.tasks):
            station_idx = solution.allocation_to_station[i]
            heuristics.append(
                f"#heuristic assign({task_id}, {station_idx}):station({station_idx}). [100000, true]"
            )
        return "\n".join(heuristics)

    def set_warm_start(self, solution: SalbpSolution) -> None:
        """Re-initializes the control object with a solution hint."""
        self.init_model(solution=solution)

    def retrieve_solution(self, model: clingo.Model) -> SalbpSolution:
        """Parses the Clingo model to construct a SalbpSolution."""
        symbols = model.symbols(atoms=True)

        # Map back to the list-based allocation in SalbpSolution
        # We initialize with a dummy value
        allocation_map = {}

        for s in symbols:
            if s.name == "assign":
                task_id = s.arguments[0].number
                station_idx = s.arguments[1].number
                allocation_map[task_id] = station_idx

        # SalbpSolution expects a list ordered by self.problem.tasks
        ordered_allocation = [allocation_map[task_id] for task_id in self.problem.tasks]

        return SalbpSolution(
            problem=self.problem, allocation_to_station=ordered_allocation
        )
