#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.generic_allocation import (
    GenericAllocationProblem,
    GenericAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.generic_allocation_impl import (
    GenericAllocationImplProblem,
    GenericAllocationImplSolution,
    NonRenewableResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_scheduling_impl import (
    extract_resource_consumptions,
)
from discrete_optimization.generic_tools.transformation import ProblemTransformation

SpecificAllocationProblem = TypeVar(
    "SpecificAllocationProblem", bound=GenericAllocationProblem
)
SpecificAllocationSolution = TypeVar(
    "SpecificAllocationSolution", bound=GenericAllocationSolution
)

PureAllocationProblem = TypeVar("PureAllocationProblem", bound=AllocationProblem)
PureAllocationSolution = TypeVar("PureAllocationSolution", bound=AllocationSolution)


class FromPureToGenericAllocationImpl(
    ProblemTransformation[
        PureAllocationProblem,
        PureAllocationSolution,
        GenericAllocationImplProblem,
        GenericAllocationImplSolution,
    ],
    Generic[
        PureAllocationProblem,
        PureAllocationSolution,
    ],
):
    """Transform a pure allocation problem (with no other task mixin) into the generic implementation.

    This is still an abstract class, as `convert_solution_from_raw_generic_to_specific()` remains to implement.


    """

    def __init__(
        self, list_objective_computer: list[ObjectiveComputer] | None = None, **kwargs
    ):
        self._list_objective_computer = list_objective_computer

    @abstractmethod
    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: PureAllocationProblem,
    ) -> PureAllocationSolution:
        """Convert a raw solution (from generic problem) into a specific solution to the source problem.

        Args:
            raw_sol: Raw solution to convert to a specific solution
            source_problem: Target Problem we want to build a solution for.

        Returns:

        """
        ...

    def back_transform_solution(
        self,
        solution: GenericAllocationImplSolution,
        source_problem: PureAllocationProblem,
    ) -> PureAllocationSolution:
        return self.transform_solution_from_raw_generic_to_specific(
            raw_sol=solution.raw_sol, source_problem=source_problem
        )

    def get_list_objective_computer(
        self,
        source_problem: PureAllocationProblem,
    ) -> list[ObjectiveComputer]:
        """From the source problem, compute a list of objective computers to be used in `GenericAllocationImplProblem`.
        To be overriden in child classes.

        Returns:

        """
        if self._list_objective_computer is not None:
            return self._list_objective_computer
        else:
            return [UnaryResourcesUsedComputer()]

    def transform_problem(
        self,
        source_problem: PureAllocationProblem,
    ) -> GenericAllocationImplProblem:
        tasks_modes: dict[Task, set[int]] = {
            task: {0} for task in source_problem.tasks_list
        }
        unary_resources: set[UnaryResource] = set(source_problem.unary_resources_list)
        unary_resources_task_compatibility: dict[Task, set[UnaryResource]] = {
            task: {
                unary_resource
                for unary_resource in source_problem.unary_resources_list
                if source_problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                )
            }
            for task in source_problem.tasks_list
        }
        optional_tasks = set(source_problem.optional_tasks_list)
        list_objective_computer = self.get_list_objective_computer(source_problem)
        return GenericAllocationImplProblem(
            tasks_modes=tasks_modes,
            unary_resources=unary_resources,
            unary_resources_task_compatibility=unary_resources_task_compatibility,
            same_unary_allocation=source_problem.get_same_unary_allocation(),
            optional_tasks=optional_tasks,
            list_objective_computer=list_objective_computer,
        )

    def forward_transform_solution(
        self,
        solution: PureAllocationSolution,
        target_problem: GenericAllocationImplProblem,
    ) -> Optional[GenericAllocationImplSolution]:
        return convert_solution_from_pure_to_generic(
            solution=solution,
            generic_problem=target_problem,
        )

    def is_bidirectional(self, source_problem: PureAllocationProblem) -> bool:
        return True


class FromSpecificToGenericAllocationImpl(
    ProblemTransformation[
        SpecificAllocationProblem,
        SpecificAllocationSolution,
        GenericAllocationImplProblem,
        GenericAllocationImplSolution,
    ],
    Generic[
        SpecificAllocationProblem,
        SpecificAllocationSolution,
    ],
):
    """Transform a specific allocation problem (with all tasks mixins) into the generic implementation.

    This is still an abstract class, as `convert_solution_from_raw_generic_to_specific()` remains to implement.


    """

    def __init__(
        self, list_objective_computer: list[ObjectiveComputer] | None = None, **kwargs
    ):
        self._list_objective_computer = list_objective_computer

    @abstractmethod
    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: SpecificAllocationProblem,
    ) -> SpecificAllocationSolution:
        """Convert a raw solution (from generic problem) into a specific solution to the source problem.

        Args:
            raw_sol: Raw solution to convert to a specific solution
            source_problem: Target Problem we want to build a solution for.

        Returns:

        """
        ...

    def back_transform_solution(
        self,
        solution: GenericAllocationImplSolution,
        source_problem: SpecificAllocationProblem,
    ) -> SpecificAllocationSolution:
        return self.transform_solution_from_raw_generic_to_specific(
            raw_sol=solution.raw_sol, source_problem=source_problem
        )

    def get_list_objective_computer(
        self,
        source_problem: SpecificAllocationProblem,
    ) -> list[ObjectiveComputer]:
        """From the source problem, compute a list of objective computers to be used in `GenericAllocationImplProblem`.
        To be overriden in child classes.

        Returns:

        """
        if self._list_objective_computer is not None:
            return self._list_objective_computer
        return source_problem.get_list_objective_computer()

    def transform_problem(
        self,
        source_problem: SpecificAllocationProblem,
    ) -> GenericAllocationImplProblem:
        tasks_modes: dict[Task, set[int]] = {
            task: source_problem.get_task_modes(task=task)
            for task in source_problem.tasks_list
        }
        (
            resources_consumption,
            resources_consumption_dependent,
        ) = extract_resource_consumptions(source_problem)
        unary_resources: set[UnaryResource] = set(source_problem.unary_resources_list)
        unary_resources_skills: dict[UnaryResource, dict[Skill, int]] = {
            unary_resource: {
                skill: value
                for skill in source_problem.skills_list
                if (
                    value := source_problem.get_unary_resource_skill_value(
                        unary_resource=unary_resource, skill=skill
                    )
                )
                > 0
            }
            for unary_resource in source_problem.unary_resources_list
        }
        unary_resources_task_compatibility: dict[Task, set[UnaryResource]] = {
            task: {
                unary_resource
                for unary_resource in source_problem.unary_resources_list
                if source_problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                )
            }
            for task in source_problem.tasks_list
        }
        skills: set[Skill] = set(source_problem.skills_list)
        non_renewable_resources: dict[NonRenewableResource, int] = {
            resource: source_problem.get_non_renewable_resource_capacity(
                resource=resource
            )
            for resource in source_problem.non_renewable_resources_list
        }
        optional_tasks = set(source_problem.optional_tasks_list)
        list_objective_computer = self.get_list_objective_computer(source_problem)
        exclusion_resource_capacity = {
            r: source_problem.get_capacity_exclusion_resource(r)
            for r in source_problem.exclusion_resources_list
        }
        exclusion_resource_consumptions = {
            t: {
                m: {
                    r: source_problem.get_task_consumption_exclusion_resource(
                        resource=r, task=t, mode=m
                    )
                    for r in source_problem.exclusion_resources_list
                }
                for m in source_problem.get_task_modes(t)
            }
            for t in source_problem.tasks_list
        }
        exclusion_resource_boolean = {
            t: {
                m: {
                    r: source_problem.is_task_mode_excluding_others(
                        task=t, mode=m, resource=r
                    )
                    for r in source_problem.exclusion_resources_list
                }
                for m in source_problem.get_task_modes(t)
            }
            for t in source_problem.tasks_list
        }
        return GenericAllocationImplProblem(
            tasks_modes=tasks_modes,
            resource_consumptions=resources_consumption,
            resource_consumptions_dependent=resources_consumption_dependent,
            unary_resources=unary_resources,
            unary_resources_skills=unary_resources_skills,
            unary_resources_task_compatibility=unary_resources_task_compatibility,
            skills=skills,
            non_renewable_resources=non_renewable_resources,
            mode_constraints=source_problem.get_mode_constraints(),
            same_unary_allocation=source_problem.get_same_unary_allocation(),
            optional_tasks=optional_tasks,
            list_objective_computer=list_objective_computer,
            alternative_scheduling_subproblems=source_problem.get_alternative_scheduling_subproblem(),
            exclusion_resource_capacity=exclusion_resource_capacity,
            exclusion_resource_consumptions=exclusion_resource_consumptions,
            exclusion_resource_boolean=exclusion_resource_boolean,
        )

    def forward_transform_solution(
        self,
        solution: SpecificAllocationSolution,
        target_problem: GenericAllocationImplProblem,
    ) -> Optional[GenericAllocationImplSolution]:
        return convert_solution_from_specific_to_generic(
            solution=solution,
            generic_problem=target_problem,
        )

    def is_bidirectional(self, source_problem: SpecificAllocationProblem) -> bool:
        return True


def convert_solution_from_pure_to_generic(
    solution: PureAllocationSolution,
    generic_problem: GenericAllocationImplProblem,
) -> GenericAllocationImplSolution:
    return GenericAllocationImplSolution(
        problem=generic_problem,
        raw_sol=RawSolution(
            task_variables={
                task: TaskVariable(
                    start=0,
                    end=1,
                    mode=0,
                    is_present=solution.is_present(task),
                    allocated={
                        unary_resource: set()
                        for unary_resource in solution.get_task_allocation(task)
                    },
                )
                for task in generic_problem.tasks_list
            }
        ),
    )


def convert_solution_from_specific_to_generic(
    solution: SpecificAllocationSolution,
    generic_problem: GenericAllocationImplProblem,
) -> GenericAllocationImplSolution:
    return GenericAllocationImplSolution(
        problem=generic_problem,
        raw_sol=RawSolution(
            task_variables={
                task: TaskVariable(
                    start=0,
                    end=1,
                    mode=solution.get_mode(task),
                    is_present=solution.is_present(task),
                    allocated={
                        unary_resource: {
                            skill
                            for skill in generic_problem.skills_list
                            if solution.is_skill_used(
                                task=task,
                                skill=skill,
                                unary_resource=unary_resource,
                            )
                        }
                        for unary_resource in solution.get_task_allocation(task)
                    },
                )
                for task in generic_problem.tasks_list
            }
        ),
    )


class FromGenericAllocationImpl(
    ProblemTransformation[
        GenericAllocationImplProblem,
        GenericAllocationImplSolution,
        SpecificAllocationProblem,
        SpecificAllocationSolution,
    ],
    Generic[SpecificAllocationProblem, SpecificAllocationSolution],
):
    """Transform the generic implementation of a allocation problem into a specific one.

    This is still an abstract class, as the `transform_problem()` remains to implement.

    """

    def back_transform_solution(
        self,
        solution: SpecificAllocationSolution,
        source_problem: GenericAllocationImplProblem,
    ) -> GenericAllocationImplSolution:
        return convert_solution_from_specific_to_generic(
            solution=solution,
            generic_problem=source_problem,
        )
