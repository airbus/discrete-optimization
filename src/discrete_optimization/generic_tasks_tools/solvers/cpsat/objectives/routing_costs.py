#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.routing_costs import (
    RoutingCostComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import Task
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class RoutingCostsCpSatModeler(ObjectiveModelerCpSat, Generic[Task]):
    objective_computer: RoutingCostComputer

    def get_objective_expr(self) -> LinearExpr:
        terms = []
        for r in self.objective_computer.weight_per_routing_entities:
            int_ = (
                self.solver.problem.get_routing_entities()
                + self.solver.problem.get_ordered_routing_entities()
            ).index(r)
            vars = self.solver.cumul_addition_variable_routing[int_]
            for c in self.objective_computer.weight_per_routing_entities[r]:
                weight = self.objective_computer.weight_per_routing_entities[r][c]
                for entity in vars[c]:
                    terms.append(weight * vars[c][entity])
        return sum(terms)
