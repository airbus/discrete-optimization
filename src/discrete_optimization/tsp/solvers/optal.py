#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
from enum import Enum

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.tsp.problem import Node, TspProblem, TspSolution
from discrete_optimization.tsp.utils import build_matrice_distance


class ModelingTspEnum(Enum):
    V0 = "V0"
    V1 = "V1"


logger = logging.getLogger(__name__)


class OptalTspSolver(SchedulingOptalSolver[Node]):
    problem: TspProblem
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingTspEnum, default=ModelingTspEnum.V0
        )
    ]

    def __init__(
        self,
        problem: TspProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.problem.end_index, self.problem.start_index] = 0
        self.modeling: ModelingTspEnum = None
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        if args["modeling"] == ModelingTspEnum.V0:
            self.modeling = ModelingTspEnum.V0
            self.init_model_v1(**args)
        if args["modeling"] == ModelingTspEnum.V1:
            self.init_model_v1(scaling=args.get("scaling", 100), **args)
            self.modeling = ModelingTspEnum.V1

    def init_model_v0(self, **args: Any) -> None:
        """Model from gpd"""
        self.cp_model = cp.Model()
        upper_bound = int(sum(self.distance_matrix.max(axis=1)))
        visits = [
            self.cp_model.interval_var(
                start=(0, None),
                end=(0, upper_bound),
                length=0,
                optional=False,
                name=f"visit_{i}",
            )
            for i in range(self.problem.node_count)
        ]
        self.variables["visits"] = visits
        seq = self.cp_model.sequence_var(visits)
        self.cp_model.enforce(
            self.cp_model.start(visits[self.problem.start_index]) == 0
        )
        seq.no_overlap()
        self.cp_model.no_overlap(
            seq,
            [
                [
                    int(self.distance_matrix[i][j])
                    for j in range(self.problem.node_count)
                ]
                for i in range(self.problem.node_count)
            ],
        )
        if self.problem.start_index == self.problem.end_index:
            come_back_base = self.cp_model.interval_var(
                start=(0, None),
                end=(0, upper_bound),
                length=0,
                optional=False,
                name=f"come_back_base",
            )
            self.variables["come_back_base"] = come_back_base
            for i in range(self.problem.node_count):
                self.cp_model.end_before_start(
                    visits[i],
                    come_back_base,
                    int(self.distance_matrix[i, self.problem.start_index]),
                )
            self.cp_model.minimize(self.cp_model.end(come_back_base))
        else:
            for i in range(self.problem.node_count):
                if i != self.problem.end_index:
                    self.cp_model.end_before_start(
                        visits[i], visits[self.problem.end_index]
                    )
            self.cp_model.minimize(self.cp_model.end(visits[self.problem.end_index]))

    def init_model_v1(self, scaling: float = 100.0, **kwargs: Any):
        """Builds the OptalCP model for the TSP problem,
        model proposed by @thtran97"""
        self.cp_model = cp.Model()
        node_count = self.problem.node_count
        start_index = self.problem.start_index
        end_index = self.problem.end_index
        # Check if the distance matrix is already computed, otherwise compute it
        distance_matrix = (
            (scaling * np.asarray(self.distance_matrix)).astype(int).tolist()
        )
        # Create interval variables for each node (except start and end)
        visit_intervals = [
            self.cp_model.interval_var(length=0, name=f"Visit_{i}")
            for i in range(node_count)
        ]
        tour_end = self.cp_model.interval_var(length=0, name="TourEnd")
        # Define the sequence of visits, excluding the start and end nodes
        nodes_in_sequence = [
            i for i in range(node_count) if i != start_index and i != end_index
        ]
        sequence_intervals = [visit_intervals[i] for i in nodes_in_sequence]
        # Create a sequence variable that will enforce the order of visits
        sequence = self.cp_model.sequence_var(sequence_intervals, nodes_in_sequence)

        # Constraints to ensure the tour starts at start_index and ends at end_index
        self.cp_model.enforce(self.cp_model.start(visit_intervals[start_index]) == 0)
        self.cp_model.no_overlap(sequence, distance_matrix)

        # Add constraints to ensure the correct travel times between nodes based on the distance matrix
        for ni in nodes_in_sequence:
            # the visit of a node "ni" should be after the visit of the first node + delta time.
            self.cp_model.end_before_start(
                visit_intervals[start_index],
                visit_intervals[ni],
                distance_matrix[start_index][ni],
            )
            # Specific additional constraint for the tour_end
            self.cp_model.end_before_start(
                visit_intervals[ni], tour_end, distance_matrix[ni][end_index]
            )
        if not nodes_in_sequence and start_index != end_index:
            # case with only 2 nodes ??!
            self.cp_model.end_before_start(
                visit_intervals[start_index],
                tour_end,
                distance_matrix[start_index][end_index],
            )

        # Objective: minimize the total tour length (end time of the tour)
        self.cp_model.minimize(tour_end.start())
        self.variables["visits"] = visit_intervals

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["visits"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> TspSolution:
        logger.info(f"Current obj {result.solution.get_objective()}")
        starts = [
            result.solution.get_start(self.get_task_interval_variable(i))
            for i in range(self.problem.node_count)
        ]
        ordered = sorted(range(len(starts)), key=lambda i: starts[i])
        solution = TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=[
                o
                for o in ordered
                if o not in {self.problem.start_index, self.problem.end_index}
            ],
        )
        eval_ = self.problem.evaluate(solution)
        logger.info(f"{eval_}")
        return solution
