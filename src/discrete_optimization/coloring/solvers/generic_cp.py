#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Iterable, Optional

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.generic_tasks_tools.generic_allocation_impl import (
    GenericAllocationImplProblem,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_allocation_impl import (
    GenericAllocationAutoCpImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_impl import (
    GenericSchedulingAutoCpImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import LinearExpr
from discrete_optimization.generic_tasks_tools.solvers.cp.cpmpy.auto_allocation_impl import (
    GenericAllocationAutoCpmpyImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.cpsat.auto_allocation_impl import (
    GenericAllocationAutoCpSatImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_allocation_impl import (
    GenericAllocationAutoOptalImplSolver,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_allocation_impl import (
    FromPureToGenericAllocationImpl,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import (
    TransformationSolver,
)


class ColoringToGenericAllocationTransformation(
    FromPureToGenericAllocationImpl[
        ColoringProblem,
        ColoringSolution,
    ]
):
    """Transform a coloring problem into a generic allocation problem.

    Lose constraints on the way, to be added to the generic solver:
    - one color per node
    - no neighbour nodes can share a color

    """

    def __init__(
        self,
        list_objective_computer: list[ObjectiveComputer] | None = None,
        **kwargs,
    ):
        super().__init__(list_objective_computer, **kwargs)

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: ColoringProblem,
    ) -> ColoringSolution:
        colors = [-1] * len(source_problem.tasks_list)
        for i, item in enumerate(source_problem.tasks_list):
            task_variable = raw_sol.task_variables[item]
            assert len(task_variable.allocated) == 1, (
                "One color and one only must be allocated to the item."
            )
            color: int = next(iter(task_variable.allocated))
            colors[i] = color
        return ColoringSolution(problem=source_problem, colors=colors)

    def transform_problem(
        self, source_problem: ColoringProblem
    ) -> GenericAllocationImplProblem:
        generic_problem = super().transform_problem(source_problem)
        need_update = False
        if source_problem.has_constraints_coloring:
            generic_problem.unary_resources_task_compatibility = {
                item: {color}
                for item, color in source_problem.constraints_coloring.color_constraint.items()
            }
            need_update = True
        if need_update:
            generic_problem.update_problem()

        return generic_problem


class _ColoringGenericAutoCpImplSolver(GenericAllocationAutoCpImplSolver):
    """Coloring cp generic solver to derive for each cp framework.

    Will add the missing constraints from the generic problem.

    """

    def __init__(
        self,
        problem: GenericSchedulingImplProblem,
        nb_colors: int | None = None,
        subset_tasks: set[Task] | None = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        custom_objective_factory: Optional[
            Callable[[GenericSchedulingAutoCpImplSolver], LinearExpr]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function, custom_objective_factory, **kwargs
        )
        if nb_colors is not None:
            self.unary_resources_allowed = [
                color for color in problem.unary_resources_list if color < nb_colors
            ]
        else:
            self.unary_resources_allowed = problem.unary_resources_list

        if subset_tasks is None:
            self.subset_tasks = self.problem.tasks_list
        else:
            self.subset_tasks = subset_tasks

    @property
    def subset_tasks_of_interest(self) -> Iterable[Task]:
        return self.subset_tasks

    @property
    def subset_unaryresources_allowed(self) -> Iterable[UnaryResource]:
        return self.unary_resources_allowed

    def init_model(
        self, exactly_one_unary_resource_per_task: bool = True, **kwargs: Any
    ) -> None:
        super().init_model(
            exactly_one_unary_resource_per_task=exactly_one_unary_resource_per_task,
            **kwargs,
        )


class ColoringGenericAutoCpSatImplSolver(
    _ColoringGenericAutoCpImplSolver, GenericAllocationAutoCpSatImplSolver
): ...


class ColoringGenericAutoCpmpyImplSolver(
    _ColoringGenericAutoCpImplSolver, GenericAllocationAutoCpmpyImplSolver
): ...


class ColoringGenericAutoOptalImplSolver(
    _ColoringGenericAutoCpImplSolver, GenericAllocationAutoOptalImplSolver
): ...


class _ColoringAutoCpImplSolver(TransformationSolver):
    generic_solver_cls: type[_ColoringGenericAutoCpImplSolver]

    def __init__(
        self,
        problem: ColoringProblem,
        nb_colors: int | None = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        transformation = ColoringToGenericAllocationTransformation(**kwargs)

        solver_brick = SubBrick(
            cls=self.generic_solver_cls,
            kwargs=dict(
                subset_tasks=problem.subset_nodes,
                nb_colors=nb_colors,
                **kwargs,
            ),
        )
        super().__init__(
            transformation=transformation,
            solver_brick=solver_brick,
            source_problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )


class ColoringAutoCpSatImplSolver(_ColoringAutoCpImplSolver):
    """Coloring solver using transformation to generic allocation problem + cpsat auto."""

    generic_solver_cls = ColoringGenericAutoCpSatImplSolver


class ColoringAutoCpmpyImplSolver(_ColoringAutoCpImplSolver):
    """Coloring solver using transformation to generic allocation problem + cpmpy auto."""

    generic_solver_cls = ColoringGenericAutoCpSatImplSolver


class ColoringAutoOptalImplSolver(_ColoringAutoCpImplSolver):
    """Coloring solver using transformation to generic allocation problem + optal auto."""

    generic_solver_cls = ColoringGenericAutoCpSatImplSolver
