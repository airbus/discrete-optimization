#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.objectives.weighted_sum_start_or_end import (
    WeightedSumStartEndObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import Task
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class WeightedSumStartEndCpSatModeler(ObjectiveModelerCpSat, Generic[Task]):
    """CP-SAT modeler for weighted sum of task/entity start and end times.

    Computes: sum(weight[task/entity][start_or_end] * time_variable)

    Supports both tasks and scheduling entities (batches, groups, etc.).
    """

    objective_computer: WeightedSumStartEndObjectiveComputer[Task]

    def get_objective_expr(self) -> LinearExpr:
        """Build the weighted sum expression for CP-SAT optimization.

        Returns:
            LinearExpr: Sum of weighted start/end times for tasks and entities
        """
        terms = []

        for item in self.objective_computer.dict_weight:
            weights_dict = self.objective_computer.dict_weight[item]

            # Get weights for start and end (default to 0 if not specified)
            weight_start = weights_dict.get(StartOrEnd.START, 0)
            weight_end = weights_dict.get(StartOrEnd.END, 0)

            if weight_start == 0 and weight_end == 0:
                continue

            if isinstance(item, SchedulingEntity):
                # Handle entity (batch, group, etc.)
                is_active = self.solver.get_entity_is_active_variable(item)

                if isinstance(is_active, int) and is_active == 1:
                    # Entity is always active - direct terms
                    if weight_start != 0:
                        start_var = self.solver.get_entity_start_or_end_variable(
                            entity=item, start_or_end=StartOrEnd.START
                        )
                        terms.append(weight_start * start_var)
                    if weight_end != 0:
                        end_var = self.solver.get_entity_start_or_end_variable(
                            entity=item, start_or_end=StartOrEnd.END
                        )
                        terms.append(weight_end * end_var)
                else:
                    # Entity is optional - create one aux var for combined contribution
                    # aux_var = weight_start * start + weight_end * end when active, 0 otherwise

                    # Compute bounds for the weighted sum
                    horizon = self.solver.get_makespan_upper_bound()

                    # Conservative bounds considering both positive and negative weights
                    min_contrib = min(
                        0,
                        weight_start * horizon,
                        weight_end * horizon,
                        (weight_start + weight_end) * horizon,
                    )
                    max_contrib = max(
                        0,
                        weight_start * horizon,
                        weight_end * horizon,
                        (weight_start + weight_end) * horizon,
                    )

                    aux_var = self.solver.cp_model.new_int_var(
                        lb=int(min_contrib),
                        ub=int(max_contrib),
                        name=f"weighted_sum_{item.entity_id}",
                    )

                    # Build the weighted expression: weight_start * start + weight_end * end
                    weighted_expr_terms = []
                    if weight_start != 0:
                        start_var = self.solver.get_entity_start_or_end_variable(
                            entity=item, start_or_end=StartOrEnd.START
                        )
                        weighted_expr_terms.append(weight_start * start_var)
                    if weight_end != 0:
                        end_var = self.solver.get_entity_start_or_end_variable(
                            entity=item, start_or_end=StartOrEnd.END
                        )
                        weighted_expr_terms.append(weight_end * end_var)

                    weighted_expr = (
                        sum(weighted_expr_terms)
                        if len(weighted_expr_terms) > 1
                        else weighted_expr_terms[0]
                    )

                    # If active: aux_var = weighted_expr, else: aux_var = 0
                    self.solver.cp_model.add(aux_var == weighted_expr).only_enforce_if(
                        is_active
                    )
                    self.solver.cp_model.add(aux_var == 0).only_enforce_if(
                        is_active.Not()
                    )

                    terms.append(aux_var)
            else:
                # Handle task
                if self.solver.problem.is_optional(item):
                    # Task is optional - create one aux var for combined contribution
                    is_present = self.solver.get_task_is_present_variable(task=item)

                    # Compute bounds
                    horizon = self.solver.get_makespan_upper_bound()
                    min_contrib = min(
                        0,
                        weight_start * horizon,
                        weight_end * horizon,
                        (weight_start + weight_end) * horizon,
                    )
                    max_contrib = max(
                        0,
                        weight_start * horizon,
                        weight_end * horizon,
                        (weight_start + weight_end) * horizon,
                    )

                    aux_var = self.solver.cp_model.new_int_var(
                        lb=int(min_contrib),
                        ub=int(max_contrib),
                        name=f"weighted_sum_{item}",
                    )

                    # Build weighted expression
                    weighted_expr_terms = []
                    if weight_start != 0:
                        start_var = self.solver.get_task_start_or_end_variable(
                            task=item, start_or_end=StartOrEnd.START
                        )
                        weighted_expr_terms.append(weight_start * start_var)
                    if weight_end != 0:
                        end_var = self.solver.get_task_start_or_end_variable(
                            task=item, start_or_end=StartOrEnd.END
                        )
                        weighted_expr_terms.append(weight_end * end_var)

                    weighted_expr = (
                        sum(weighted_expr_terms)
                        if len(weighted_expr_terms) > 1
                        else weighted_expr_terms[0]
                    )

                    # If present: aux_var = weighted_expr, else: aux_var = 0
                    self.solver.cp_model.add(aux_var == weighted_expr).only_enforce_if(
                        is_present
                    )
                    self.solver.cp_model.add(aux_var == 0).only_enforce_if(
                        is_present.Not()
                    )

                    terms.append(aux_var)
                else:
                    # Task is always present - direct terms
                    if weight_start != 0:
                        start_var = self.solver.get_task_start_or_end_variable(
                            task=item, start_or_end=StartOrEnd.START
                        )
                        terms.append(weight_start * start_var)
                    if weight_end != 0:
                        end_var = self.solver.get_task_start_or_end_variable(
                            task=item, start_or_end=StartOrEnd.END
                        )
                        terms.append(weight_end * end_var)

        if not terms:
            # No terms - return constant 0
            return 0

        return sum(terms)
