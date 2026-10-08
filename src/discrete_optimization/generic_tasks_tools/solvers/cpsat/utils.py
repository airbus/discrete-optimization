#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Util module to share some modeling routine that may be used in several part of the code
#  For example in cumulative and non-renewable resource mixin.
from collections.abc import Iterable
from enum import Enum

from ortools.sat.python.cp_model import Constraint, CpModel, Domain, LinearExprT

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import Task
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingProblem
from discrete_optimization.generic_tasks_tools.solvers.cpsat.base import (
    TasksCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode import (
    MultimodeCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
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


class SpanModeling(Enum):
    INEQUALITIES = 0
    EXACT = 1


def create_variable_function_of_mode_on_solver(
    solver: MultimodeCpSatSolver,
    name: str,
    mode2value: dict[int, int],
    task: Task,
    modeling: ModeToValueModeling = ModeToValueModeling.ENFORCE_IF,
    conditional_var: LinearExprT | None = None,
    no_constraint: bool = False,
) -> LinearExprT:
    """Create a variable whose values depend on chosen mode

     If the task is optional, we add the value 0 if no mode is chosen.
     The new variable can also be conditioned to another boolean variable
     (typically a given unary resource is allocated), which means its value will be 0
     if the conditioning variable is false.

     Args:
         solver:
         name:
         mode2value:
         task: task for which the variable is created
         modeling:
         conditional_var: (optional) conditioning boolean variable implying the new variable to be 0 if false.
         no_constraint: if True, create a variable with proper domain without constraining
            values on mode (e.g. because the constraints are created elsewhere via interval variables)

    Returns:
        The new variable

    """
    if isinstance(conditional_var, int):
        if conditional_var == 0:
            # conditional variable always false
            return 0
        elif conditional_var == 1:
            # always true => no conditioning
            conditional_var = None
    optional_task = solver.problem.is_optional(task)
    possible_values = set(mode2value.values())
    if conditional_var is not None or optional_task:
        possible_values.add(0)  # for the case conditional_var == 0 or task is absent
    if len(possible_values) == 1:
        return next(iter(possible_values))
    if no_constraint:
        return solver.cp_model.new_int_var_from_domain(
            Domain.from_values(list(possible_values)), name=name
        )
    match modeling:
        case ModeToValueModeling.LINEAR_SUM:
            if conditional_var is not None:
                raise ValueError(
                    "Cannot model as a linear combination if conditioned by an other variable (conditional_var is not None)"
                )
            return sum(
                mode2value[mode]
                * solver.get_task_mode_is_present_variable(task=task, mode=mode)
                for mode in mode2value
                if mode2value[mode] != 0
            )
        case ModeToValueModeling.ENFORCE_IF:
            if possible_values == {0, 1}:
                # Corner case, might happen.
                var = solver.cp_model.new_bool_var(name=name)
            else:
                var = solver.cp_model.new_int_var_from_domain(
                    Domain.from_values(list(possible_values)), name=name
                )
            for mode, value in mode2value.items():
                enforce_mode_value_vars = (
                    solver.get_task_mode_is_present_variable(task=task, mode=mode),
                )
                if conditional_var is not None:
                    solver.cp_model.add(var == 0).only_enforce_if(~conditional_var)
                    enforce_mode_value_vars = enforce_mode_value_vars + (
                        conditional_var,
                    )
                solver.cp_model.add(var == value).only_enforce_if(
                    *enforce_mode_value_vars
                )
            if optional_task:
                # no mode chosen => var == 0
                solver.cp_model.add(var == 0).only_enforce_if(
                    *(
                        ~solver.get_task_mode_is_present_variable(task=task, mode=mode)
                        for mode in mode2value
                    )
                )
        case ModeToValueModeling.TABLE:
            # WARNING : experimental feature.
            values = list(possible_values)
            var = solver.cp_model.new_int_var_from_domain(
                Domain.from_values(values), name=name
            )
            for mode, value in mode2value.items():
                if conditional_var is None:
                    solver.cp_model.add_allowed_assignments(
                        [
                            solver.get_task_mode_is_present_variable(
                                task=task, mode=mode
                            ),
                            var,
                        ],
                        [(1, value)] + [(0, v) for v in possible_values],
                    )
                else:
                    solver.cp_model.add_allowed_assignments(
                        [
                            solver.get_task_mode_is_present_variable(
                                task=task, mode=mode
                            ),
                            conditional_var,
                            var,
                        ],
                        [
                            (1, 1, value)
                        ]  # mode chosen + conditional var true => mode value
                        + [(0, 1, v) for v in possible_values]  # other mode chosen
                        + [(1, 0, 0), (0, 0, 0)],  # conditional var false => 0
                    )
                    solver.cp_model.add_allowed_assignments(
                        [conditional_var, var],
                        [
                            (1, v) for v in possible_values
                        ]  # conditional var true => any value
                        + [(0, 0)],  # conditional var false => 0
                    )
            # assigments minxing all possible modes
            modes = list(mode2value)
            modes_vars = [
                solver.get_task_mode_is_present_variable(task=task, mode=m)
                for m in modes
            ]
            values = [mode2value[mode] for mode in modes]
            expressions = modes_vars + [var]
            nb_modes_var = len(modes_vars)
            tuples_list = [
                (0,) * i + (1,) + (0,) * (nb_modes_var - i - 1) + (values[i],)
                for i in range(nb_modes_var)
            ]
            if optional_task:
                tuples_list.append((0,) * (nb_modes_var + 1))
            if conditional_var is not None:
                expressions.append(conditional_var)
                tuples_list = [tuple_values + (1,) for tuple_values in tuples_list]
                tuples_list += [
                    tuple_values[:nb_modes_var] + (0, 0) for tuple_values in tuples_list
                ]
            solver.cp_model.add_allowed_assignments(expressions, tuples_list)
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


def enforce_only_if_tasks_present(
    constraint: Constraint, tasks: Iterable[Task], solver: TasksCpSatSolver[Task]
):
    """Enforce given constraints only if all given tasks are present.

    Do nothing if all tasks are mandatory.

    Args:
        constraint:
        tasks:
        solver:

    Returns:

    """
    is_present_tasks_variables = tuple(
        solver.get_task_is_present_variable(task)
        for task in tasks
        if solver.problem.is_optional(task)
    )
    if len(is_present_tasks_variables) > 0:
        return constraint.only_enforce_if(*is_present_tasks_variables)
    else:
        return constraint


def create_span_start_end_variables(
    solver: SchedulingCpSatSolver[Task],
    set_tasks: set[Task],
    name_span: str,
    start_span: LinearExprT | None = None,
    end_span: LinearExprT | None = None,
    span_modeling: SpanModeling = SpanModeling.INEQUALITIES,
):
    problem: SchedulingProblem[Task] = solver.problem
    if start_span is None:
        # TODO: use better bound for span.
        start_span = solver.cp_model.new_int_var(
            lb=0, ub=problem.get_makespan_upper_bound(), name=f"start_{name_span}"
        )
    if end_span is None:
        end_span = solver.cp_model.new_int_var(
            lb=0, ub=problem.get_makespan_upper_bound(), name=f"end_{name_span}"
        )
    all_mandatory = all(not problem.is_optional(t) for t in set_tasks)
    if all_mandatory:
        # Use min/max
        starts = [
            solver.get_task_start_or_end_variable(t, start_or_end=StartOrEnd.START)
            for t in set_tasks
        ]
        ends = [
            solver.get_task_start_or_end_variable(t, start_or_end=StartOrEnd.END)
            for t in set_tasks
        ]
        solver.cp_model.add_min_equality(start_span, starts)
        solver.cp_model.add_max_equality(end_span, ends)
        return start_span, end_span
    else:
        starts_dict = {
            t: solver.get_task_start_or_end_variable(t, start_or_end=StartOrEnd.START)
            for t in set_tasks
        }
        ends_dict = {
            t: solver.get_task_start_or_end_variable(t, start_or_end=StartOrEnd.END)
            for t in set_tasks
        }
        if span_modeling == SpanModeling.INEQUALITIES:
            for t in set_tasks:
                if problem.is_optional(t):
                    solver.cp_model.add(start_span <= starts_dict[t]).only_enforce_if(
                        solver.get_task_is_present_variable(task=t)
                    )
                    solver.cp_model.add(end_span >= ends_dict[t]).only_enforce_if(
                        solver.get_task_is_present_variable(task=t)
                    )
                else:
                    solver.cp_model.add(start_span <= starts_dict[t])
                    solver.cp_model.add(end_span >= ends_dict[t])
        if span_modeling == SpanModeling.EXACT:
            start_min_array = []
            # Dummy/safe upper bound
            upper_bound = 2 * solver.get_makespan_upper_bound() + 1
            for t in set_tasks:
                start = solver.get_task_start_or_end_variable(
                    t, start_or_end=StartOrEnd.START
                )
                end = solver.get_task_start_or_end_variable(
                    t, start_or_end=StartOrEnd.END
                )
                if problem.is_optional(t):
                    smin = solver.cp_model.new_int_var(
                        lb=0, ub=upper_bound, name=f"start_min_{name_span}_{t}"
                    )
                    is_present = solver.get_task_is_present_variable(t)
                    solver.cp_model.add(smin == upper_bound).only_enforce_if(
                        ~is_present
                    )
                    solver.cp_model.add(smin == start).only_enforce_if(is_present)
                    start_min_array.append(smin)
                else:
                    start_min_array.append(start)
            solver.cp_model.add_min_equality(start_span, start_min_array)
            solver.cp_model.add_max_equality(
                end_span, [ends_dict[t] for t in set_tasks]
            )
    return start_span, end_span


def create_span_start_end_from_vars(
    solver: SchedulingCpSatSolver[Task],
    starts_list: list[LinearExprT],
    ends_list: list[LinearExprT],
    is_present_list: list[LinearExprT],
    name_span: str,
    start_span: LinearExprT | None = None,
    end_span: LinearExprT | None = None,
    span_modeling: SpanModeling = SpanModeling.INEQUALITIES,
):
    problem: SchedulingProblem[Task] = solver.problem
    if start_span is None:
        start_span = solver.cp_model.new_int_var(
            lb=0, ub=problem.get_makespan_upper_bound(), name=f"start_{name_span}"
        )
    if end_span is None:
        end_span = solver.cp_model.new_int_var(
            lb=0, ub=problem.get_makespan_upper_bound(), name=f"end_{name_span}"
        )
    if all(isinstance(x, int) and x == 1 for x in is_present_list):
        # Use min/max
        starts = starts_list
        ends = ends_list

        solver.cp_model.add_min_equality(start_span, starts)
        solver.cp_model.add_max_equality(end_span, ends)
        return start_span, end_span
    else:
        starts = starts_list
        ends = ends_list
        if span_modeling == SpanModeling.INEQUALITIES:
            for i in range(len(starts)):
                is_present = is_present_list[i]
                solver.cp_model.add(start_span <= starts[i]).only_enforce_if(is_present)
                solver.cp_model.add(end_span >= ends[i]).only_enforce_if(is_present)
        if span_modeling == SpanModeling.EXACT:
            start_min_array = []
            # Dummy/safe upper bound
            upper_bound = 2 * solver.get_makespan_upper_bound() + 1
            for i in range(len(starts)):
                start = starts[i]
                is_present = is_present_list[i]
                if not (isinstance(is_present, int) and is_present == 1):
                    smin = solver.cp_model.new_int_var(
                        lb=0, ub=upper_bound, name=f"start_min_{name_span}_{i}"
                    )
                    solver.cp_model.add(smin == upper_bound).only_enforce_if(
                        ~is_present
                    )
                    solver.cp_model.add(smin == start).only_enforce_if(is_present)
                    start_min_array.append(smin)
                else:
                    start_min_array.append(start)
            solver.cp_model.add_min_equality(start_span, start_min_array)
            solver.cp_model.add_max_equality(end_span, ends)
    return start_span, end_span
