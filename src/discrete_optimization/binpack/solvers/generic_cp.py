#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Iterable, Optional

from discrete_optimization.binpack.problem import (
    BinPack,
    BinPackProblem,
    BinPackProblemBinType,
    BinPackSolution,
)
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


class BinPackToGenericAllocationTransformation(
    FromPureToGenericAllocationImpl[
        BinPackProblemBinType,
        BinPackSolution,
    ]
):
    """Transform a binpack problem into a generic allocation problem.

    Lose constraints on the way, to be added to the generic solver:
    - one color per node
    - no neighbour nodes can share a color

    Args:
        nb_colors: nb max of colors allowed

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
        source_problem: BinPackProblemBinType,
    ) -> BinPackSolution:
        allocation = [-1] * source_problem.nb_items
        for i_item, task_variable in raw_sol.task_variables.items():
            assert len(task_variable.allocated) == 1, (
                "One bin and one only must be allocated to the item."
            )
            bin_alloc: int = next(iter(task_variable.allocated))
            allocation[i_item] = bin_alloc
        return BinPackSolution(problem=source_problem, allocation=allocation)

    def transform_problem(
        self, source_problem: BinPackProblemBinType
    ) -> GenericAllocationImplProblem:
        generic_problem = super().transform_problem(source_problem)
        # bin max_capacity => non-renewable resource for each bin weight
        ind_bin_to_nr_resource: dict[BinPack, str] = {
            i: f"weight_{i}" for i in range(source_problem.nb_bins)
        }
        generic_problem.non_renewable_resources = {
            ind_bin_to_nr_resource[i]: bin_instance.bin_type.capacity
            for i, bin_instance in enumerate(source_problem.list_bin_instances)
        }
        generic_problem.unary_resource_consumptions_dependent = {
            # task: item
            i_item: {
                # mode
                0: {
                    # non-renewable resource (bin weight)
                    ind_bin_to_nr_resource[i]: {
                        # unary resource: bin -> nr conso = item weight
                        i: int(item.weight)
                    }
                    for i in range(source_problem.nb_bins)
                }
            }
            for i_item, item in enumerate(source_problem.list_items)
        }
        generic_problem.update_problem()

        return generic_problem


class _BinPackGenericAutoCpImplSolver(GenericAllocationAutoCpImplSolver):
    """BinPack cp generic solver to derive for each cp framework.

    Will add the missing constraints from the generic problem.

    """

    def __init__(
        self,
        problem: GenericSchedulingImplProblem,
        nb_bins: int | None = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        custom_objective_factory: Optional[
            Callable[[GenericSchedulingAutoCpImplSolver], LinearExpr]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function, custom_objective_factory, **kwargs
        )
        if nb_bins is not None:
            self.unary_resources_allowed = [
                i_bin for i_bin in problem.unary_resources_list if i_bin < nb_bins
            ]
        else:
            self.unary_resources_allowed = problem.unary_resources_list

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


class BinPackGenericAutoCpSatImplSolver(
    _BinPackGenericAutoCpImplSolver, GenericAllocationAutoCpSatImplSolver
): ...


class BinPackGenericAutoCpmpyImplSolver(
    _BinPackGenericAutoCpImplSolver, GenericAllocationAutoCpmpyImplSolver
): ...


class BinPackGenericAutoOptalImplSolver(
    _BinPackGenericAutoCpImplSolver, GenericAllocationAutoOptalImplSolver
): ...


class _BinPackAutoCpImplSolver(TransformationSolver):
    generic_solver_cls: type[_BinPackGenericAutoCpImplSolver]

    def __init__(
        self,
        problem: BinPackProblem,
        nb_bins: int | None = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        transformation = BinPackToGenericAllocationTransformation(**kwargs)

        solver_brick = SubBrick(
            cls=self.generic_solver_cls,
            kwargs=dict(
                nb_bins=nb_bins,
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


class BinPackAutoCpSatImplSolver(_BinPackAutoCpImplSolver):
    """BinPack solver using transformation to generic allocation problem + cpsat auto."""

    generic_solver_cls = BinPackGenericAutoCpSatImplSolver


class BinPackAutoCpmpyImplSolver(_BinPackAutoCpImplSolver):
    """BinPack solver using transformation to generic allocation problem + cpmpy auto."""

    generic_solver_cls = BinPackGenericAutoCpSatImplSolver


class BinPackAutoOptalImplSolver(_BinPackAutoCpImplSolver):
    """BinPack solver using transformation to generic allocation problem + optal auto."""

    generic_solver_cls = BinPackGenericAutoCpSatImplSolver
