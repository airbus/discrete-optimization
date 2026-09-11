#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Util module to share some modeling routine that may be used in several part of the code
#  For example in cumulative and non-renewable resource mixin.
from enum import Enum

from ortools.sat.python.cp_model import CpModel, Domain, LinearExprT

from discrete_optimization.generic_tasks_tools.generic_scheduling import Task
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode import (
    MultimodeCpSatSolver,
)


class ModeToValueModeling(Enum):
    """
    This is some option to define constraint between
    x=[list of N boolean variables] with a sum <= 1
    vals=[list of N int constants]
    and a variable Y that should value vals[i] when x[i] is True.
    """

    LINEAR_SUM = 0
    ENFORCE_IF = 1
    TABLE = 2


def create_variable_function_of_mode_on_solver(
    solver: MultimodeCpSatSolver,
    name: str,
    mode2value: dict[int, int],
    mode2var: dict[int, LinearExprT],
    modeling: ModeToValueModeling,
) -> LinearExprT:
    return create_variable_function_of_mode(
        cp_model=solver.cp_model,
        name_var=name,
        mode2value=mode2value,
        mode2var=mode2var,
        modeling=modeling,
    )


def create_variable_function_of_mode(
    cp_model: CpModel,
    name_var: str,
    mode2value: dict[int, int],
    mode2var: dict[int, LinearExprT],
    modeling: ModeToValueModeling = ModeToValueModeling.ENFORCE_IF,
) -> LinearExprT:
    possible_values = set(mode2value.values())
    if len(possible_values) == 1:
        var = next(iter(possible_values))
        return var
    match modeling:
        case ModeToValueModeling.LINEAR_SUM:
            return sum(
                mode2value[m] * mode2var[m] for m in mode2value if mode2value[m] != 0
            )
        case ModeToValueModeling.ENFORCE_IF:
            var = cp_model.new_int_var_from_domain(
                Domain.from_values(list(possible_values)), name=name_var
            )
            for mode, value in mode2value.items():
                cp_model.add(var == value).only_enforce_if(mode2var[mode])
        case ModeToValueModeling.TABLE:
            # WARNING : experimental feature.
            var = cp_model.new_int_var_from_domain(
                Domain.from_values(list(possible_values)), name=name_var
            )
            for mode, value in mode2value.items():
                cp_model.add_allowed_assignments(
                    [mode2var[mode], var],
                    [(1, value)] + [(0, v) for v in possible_values],
                )
            modes = list(mode2var)
            modes_vars = [mode2var[m] for m in modes]
            values = [mode2value[m] for m in modes]
            expressions = modes_vars + [var]
            nb_modes_var = len(modes_vars)
            possible_values = [
                tuple([0] * i + [1] + [0] * (nb_modes_var - i - 1) + [values[i]])
                for i in range(nb_modes_var)
            ]
            cp_model.add_allowed_assignments(expressions, possible_values)
        case _:
            raise NotImplementedError()
    return var


def create_resource_dependent_variable(
    cp_model: CpModel,
    name_var: str,
    task: Task,
    task_mode_var: dict[tuple[Task, int], LinearExprT],
    mode2mapping: dict[int, dict[frozenset[tuple[Task, int]], int] | None],
    possible_values: set[int] = None,
):
    if possible_values is None:
        possible_values = set(
            [x for mode in mode2mapping for x in mode2mapping[mode].values()] + [0]
        )
    demand_var = cp_model.new_int_var_from_domain(
        domain=Domain.FromValues(list(possible_values)), name=name_var
    )
    for mode in mode2mapping:
        mapping = mode2mapping[mode]
        for set_task_mode in mapping:
            value = mapping[set_task_mode]
            modes_var = [task_mode_var[tt, mm] for tt, mm in set_task_mode]
            (
                cp_model.add(demand_var == value).only_enforce_if(
                    *([task_mode_var[task, mode]] + modes_var)
                )
            )
    return demand_var
