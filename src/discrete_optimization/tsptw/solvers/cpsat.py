#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, Optional

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution

logger = logging.getLogger(__name__)


class CpSatTSPTWSolver(OrtoolsCpSatSolver, WarmstartMixin):
    """
    CP-SAT solver for the Traveling Salesman Problem with Time Windows.

    This solver uses a circuit constraint to ensure a valid tour and enforces
    time window constraints through implications based on the selected arcs.

    Attributes:
        problem (TSPTWProblem): The TSP-TW problem instance.
        variables (Dict[str, Any]): A dictionary to store the CP-SAT model variables,
                                    including arc variables ('x_arc'), time variables ('t_time'),
                                    and the makespan variable.
    """

    problem: TSPTWProblem

    def __init__(
        self,
        problem: TSPTWProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: Dict[str, Any] = {}

    def init_model(
        self, scaling_factor: float = 10.0, relaxed: bool = False, **kwargs: Any
    ) -> None:
        """Initialise the CP-SAT model."""
        super().init_model(**kwargs)
        n = self.problem.nb_nodes
        depot = self.problem.depot_node
        if relaxed:
            horizon = int(
                10
                * max([int(scaling_factor * x[1]) for x in self.problem.time_windows])
            )
            bounds = [
                (int(scaling_factor * x[0]), horizon) for x in self.problem.time_windows
            ]
        else:
            bounds = [
                (int(scaling_factor * x[0]), int(scaling_factor * x[1]))
                for x in self.problem.time_windows
            ]
        # --- Create variables ---

        # Arc variables: x_arc[i, j] is true if the tour includes the arc from i to j
        x_arc = {
            (i, j): self.cp_model.NewBoolVar(f"x_{i},{j}")
            for i in range(n)
            for j in range(n)
            if i != j
            and bounds[j][1]
            >= bounds[i][0] + int(scaling_factor * self.problem.distance_matrix[i][j])
        }

        # Time variables: t_time[i] is the time at which service starts at node i
        t_time = [
            self.cp_model.NewIntVar(
                lb=bounds[i][0],
                ub=bounds[i][1],
                name=f"t_{i}",
            )
            for i in range(n)
        ]
        t_time_return_depot = self.cp_model.NewIntVar(
            lb=bounds[depot][0],
            ub=bounds[depot][1],
            name=f"t_return_depot",
        )

        # Makespan variable: represents the total tour duration (arrival back at depot)
        makespan_ub = bounds[depot][1]
        makespan = self.cp_model.NewIntVar(0, makespan_ub, "makespan")

        self.variables = {
            "x_arc": x_arc,
            "t_time": t_time,
            "t_return": t_time_return_depot,
            "makespan": makespan,
        }

        # --- Add constraints ---

        # Build arcs list for the circuit constraint
        arcs = [(i, j, x_arc[(i, j)]) for i, j in x_arc]

        # Add a single tour visiting all nodes
        self.cp_model.AddCircuit(arcs)

        # Fix the start time at the depot
        self.cp_model.Add(
            t_time[depot] == int(scaling_factor * self.problem.time_windows[depot][0])
        )

        # Time window propagation constraints
        for i in range(n):
            for j in range(n):
                if (i, j) not in x_arc:
                    continue

                # The time to get from i to j, including service time at i
                travel_and_service_time = int(
                    scaling_factor * self.problem.distance_matrix[i, j]
                )

                # If arc (i,j) is taken, then t_j must be after t_i + travel
                if j == self.problem.depot_node:
                    self.cp_model.Add(
                        t_time_return_depot >= t_time[i] + travel_and_service_time
                    ).OnlyEnforceIf(x_arc[i, j])
                else:
                    self.cp_model.Add(
                        t_time[j] >= t_time[i] + travel_and_service_time
                    ).OnlyEnforceIf(x_arc[i, j])

        # Makespan constraints: if arc (i, depot) is taken, makespan is at least t_i + travel
        for i in range(n):
            if i == depot:
                continue
            travel_to_depot = int(
                scaling_factor * self.problem.distance_matrix[i, depot]
            )
            self.cp_model.Add(makespan >= t_time[i] + travel_to_depot).OnlyEnforceIf(
                x_arc[i, depot]
            )

        # --- Set objective ---
        if not relaxed:
            self.cp_model.Minimize(makespan)
        else:
            horizon = int(
                10
                * max([int(scaling_factor * x[1]) for x in self.problem.time_windows])
            )
            lateness = {
                i: self.cp_model.NewIntVar(
                    lb=0,
                    ub=horizon - int(scaling_factor * self.problem.time_windows[i][1]),
                    name=f"lateness_{i}",
                )
                for i in range(self.problem.nb_nodes)
            }
            self.variables["lateness"] = lateness
            for i in lateness:
                if i == self.problem.depot_node:
                    self.cp_model.AddMaxEquality(
                        lateness[i],
                        [0, t_time_return_depot - self.problem.time_windows[i][1]],
                    )
                else:
                    self.cp_model.AddMaxEquality(
                        lateness[i], [0, t_time[i] - self.problem.time_windows[i][1]]
                    )
            self.cp_model.Minimize(
                10000 * sum([lateness[x] for x in lateness]) + makespan
            )
        logger.info("CP-SAT model initialized.")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> TSPTWSolution:
        """
        Build a TSPTWSolution from the CP-SAT solver's callback.

        Args:
            cpsolvercb: The ortools callback object containing the current solution.

        Returns:
            The current solution in the TSPTWSolution format.
        """
        # Reconstruct the path from the active arc variables
        permutation = []

        current_node = self.problem.depot_node
        if "lateness" in self.variables:
            sum_ = sum(
                [
                    cpsolvercb.Value(self.variables["lateness"][x])
                    for x in self.variables["lateness"]
                ]
            )
            logger.info(f"Lateness : {sum_}")
        for _ in range(self.problem.nb_nodes - 1):
            for j in range(self.problem.nb_nodes):
                if current_node == j:
                    continue
                if (current_node, j) in self.variables["x_arc"] and cpsolvercb.Value(
                    self.variables["x_arc"][current_node, j]
                ):
                    permutation.append(j)
                    current_node = j
                    break

        return TSPTWSolution(problem=self.problem, permutation=permutation)

    def set_warm_start(self, solution: TSPTWSolution) -> None:
        """
        Provides a warm start hint to the CP-SAT solver from an existing solution.

        Args:
            solution: A TSPTWSolution object.
        """
        if self.cp_model is None:
            self.init_model()

        self.cp_model.ClearHints()
        logger.info("Setting warm start from solution.")

        # Hint arc variables
        path = (
            [self.problem.depot_node] + solution.permutation + [self.problem.depot_node]
        )
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in self.variables["x_arc"]:
                self.cp_model.AddHint(self.variables["x_arc"][u, v], 1)

        # Hint time variables if they have been calculated
        if solution.start_service_times:
            for i in range(self.problem.nb_nodes):
                if i in solution.start_service_times:
                    self.cp_model.AddHint(
                        self.variables["t_time"][i],
                        int(solution.start_service_times[i]),
                    )
