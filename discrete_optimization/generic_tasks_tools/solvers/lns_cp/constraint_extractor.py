from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Generic, Union

import numpy as np

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationCpSolver,
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeCpSolver,
    MultimodeProblem,
    MultimodeSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingCpSolver,
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)


class ParamsConstraintExtractor(Hyperparametrizable):
    hyperparameters = [
        # scheduling
        IntegerHyperparameter(name="minus_delta_primary", low=0, default=5, high=20),
        IntegerHyperparameter(name="plus_delta_primary", low=0, default=5, high=20),
        IntegerHyperparameter(name="minus_delta_secondary", low=0, default=0, high=10),
        IntegerHyperparameter(name="plus_delta_secondary", low=0, default=0, high=10),
        # scheduling preemptive
        IntegerHyperparameter(
            name="minus_delta_primary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="plus_delta_primary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="minus_delta_secondary_duration", default=5, low=0, high=10
        ),
        IntegerHyperparameter(
            name="plus_delta_secondary_duration", default=5, low=0, high=10
        ),
        # chaining extractor
        CategoricalHyperparameter(
            name="chaining",
            choices=[True, False],
            default=False,
        ),
        FloatHyperparameter(
            name="frac_fixed_chaining",
            default=0.25,
            low=0.0,
            high=1.0,
            depends_on=("chaining", [True]),
        ),
        # multimode
        CategoricalHyperparameter(
            name="fix_primary_tasks_modes",
            choices=[True, False],
            default=False,
        ),
        CategoricalHyperparameter(
            name="fix_secondary_tasks_modes",
            choices=[True, False],
            default=True,
        ),
        # allocation
        CategoricalHyperparameter(
            name="allocation_subtasks",
            choices=[True, False],
            default=True,
        ),
        CategoricalHyperparameter(
            name="fix_secondary_tasks_allocation",
            choices=[True, False],
            default=False,
            depends_on=("allocation_subtasks", [True]),
        ),
        FloatHyperparameter(
            name="frac_random_fixed_tasks",
            default=0.6,
            low=0.0,
            high=1.0,
            depends_on=("allocation_subtasks", [True]),
        ),
        CategoricalHyperparameter(
            name="allocation_subresources",
            choices=[True, False],
            default=False,
        ),
        FloatHyperparameter(
            name="frac_random_fixed_unary_resources",
            default=0.5,
            low=0.0,
            high=1.0,
            depends_on=("allocation_subresources", [True]),
        ),
        CategoricalHyperparameter(
            name="nb_changes",
            choices=[True, False],
            default=False,
        ),
        IntegerHyperparameter(
            name="nb_changes_max",
            default=10,
            low=0,
            high=20,
            depends_on=("nb_changes", [True]),
        ),
        CategoricalHyperparameter(
            name="nb_usages",
            choices=[True, False],
            default=False,
        ),
        IntegerHyperparameter(
            name="plus_delta_nb_usages_total",
            default=5,
            low=0,
            high=10,
            depends_on=("nb_usages", [True]),
        ),
        IntegerHyperparameter(
            name="plus_delta_nb_usages_per_unary_resource",
            default=3,
            low=0,
            high=5,
            depends_on=("nb_usages", [True]),
        ),
        IntegerHyperparameter(
            name="minus_delta_nb_usages_per_unary_resource",
            default=3,
            low=0,
            high=5,
            depends_on=("nb_usages", [True]),
        ),
    ]

    def __init__(
        self,
        # scheduling
        minus_delta_primary: int = 5,
        plus_delta_primary: int = 5,
        minus_delta_secondary: int = 0,
        plus_delta_secondary: int = 0,
        constraint_to_current_solution_makespan: bool = True,
        margin_rel_to_current_solution_makespan: float = 0.05,
        # scheduling preemptive
        minus_delta_primary_duration: int = 5,
        plus_delta_primary_duration: int = 5,
        minus_delta_secondary_duration: int = 5,
        plus_delta_secondary_duration: int = 5,
        # multimode
        fix_primary_tasks_modes: bool = False,
        fix_secondary_tasks_modes: bool = True,
        # chaining
        chaining: bool = False,
        frac_fixed_chaining: float = 0.25,
        # allocation
        allocation_subtasks: bool = True,
        fix_secondary_tasks_allocation: bool = False,
        frac_random_fixed_tasks: float = 0.6,
        allocation_subresources: bool = False,
        frac_random_fixed_unary_resources: float = 0.5,
        nb_changes: bool = False,
        nb_changes_max: int = 10,
        nb_usages: bool = False,
        plus_delta_nb_usages_total: int = 5,
        plus_delta_nb_usages_per_unary_resource: int = 3,
        minus_delta_nb_usages_per_unary_resource: int = 3,
    ):
        self.minus_delta_primary = minus_delta_primary
        self.plus_delta_primary = plus_delta_primary
        self.minus_delta_secondary = minus_delta_secondary
        self.plus_delta_secondary = plus_delta_secondary
        self.constraint_to_current_solution_makespan = (
            constraint_to_current_solution_makespan
        )
        self.margin_rel_to_current_solution_makespan = (
            margin_rel_to_current_solution_makespan
        )
        self.minus_delta_primary_duration = minus_delta_primary_duration
        self.plus_delta_primary_duration = plus_delta_primary_duration
        self.minus_delta_secondary_duration = minus_delta_secondary_duration
        self.plus_delta_secondary_duration = plus_delta_secondary_duration
        self.fix_primary_tasks_modes = fix_primary_tasks_modes
        self.fix_secondary_tasks_modes = fix_secondary_tasks_modes
        self.chaining = chaining
        self.frac_fixed_chaining = frac_fixed_chaining
        self.allocation_subtasks = allocation_subtasks
        self.fix_secondary_tasks_allocation = fix_secondary_tasks_allocation
        self.frac_random_fixed_tasks = frac_random_fixed_tasks
        self.allocation_subresources = allocation_subresources
        self.frac_random_fixed_unary_resources = frac_random_fixed_unary_resources
        self.nb_changes = nb_changes
        self.nb_changes_max = nb_changes_max
        self.nb_usages = nb_usages
        self.plus_delta_nb_usages_total = plus_delta_nb_usages_total
        self.plus_delta_nb_usages_per_unary_resource = (
            plus_delta_nb_usages_per_unary_resource
        )
        self.minus_delta_nb_usages_per_unary_resource = (
            minus_delta_nb_usages_per_unary_resource
        )


class BaseConstraintExtractor(ABC, Generic[Task]):
    """Base class for constraint extractor.

    The constraints are extracted from a current solution + tasks subset.

    """

    @abstractmethod
    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        """Extract constraints and add them to the cp model.

        Args:
            current_solution:
            solver:
            tasks_primary:
            tasks_secondary:
            **kwargs:

        Returns:

        """
        ...


class DummyConstraintExtractor(BaseConstraintExtractor[Task]):
    """Does not add any constraint.
    Can be useful during LNS to let run the solver without neighborhood constraint"""

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        return []


class ConstraintExtractorList(BaseConstraintExtractor[Task]):
    """Extractor adding constraints from multiple sub-extractors."""

    def __init__(
        self,
        extractors: list[BaseConstraintExtractor[Task]],
    ):
        self.extractors = extractors

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        constraints = []
        for extractor in self.extractors:
            constraints += extractor.add_constraints(
                current_solution=current_solution,
                solver=solver,
                tasks_primary=tasks_primary,
                tasks_secondary=tasks_secondary,
                **kwargs,
            )
        return constraints


class ConstraintExtractorPortfolio(BaseConstraintExtractor[Task]):
    """Extractor adding constraints from multiple sub-extractors."""

    def __init__(
        self,
        extractors: list[BaseConstraintExtractor[Task]],
        weights: Union[list[float], np.array] = None,
        verbose: bool = False,
    ):
        self.extractors = extractors
        self.weights = weights
        if self.weights is None:
            self.weights = [
                1 / len(self.extractors) for _ in range(len(self.extractors))
            ]
        if isinstance(self.weights, list):
            self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
        self.index_np = np.array(range(len(self.extractors)), dtype=np.int_)
        self.verbose = verbose

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        choice = np.random.choice(self.index_np, size=1, p=self.weights)[0]
        return self.extractors[choice].add_constraints(
            current_solution=current_solution,
            solver=solver,
            tasks_primary=tasks_primary,
            tasks_secondary=tasks_secondary,
            **kwargs,
        )


class SchedulingConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        minus_delta_primary: int = 5,
        plus_delta_primary: int = 5,
        minus_delta_secondary: int = 0,
        plus_delta_secondary: int = 0,
    ):
        self.plus_delta_primary = plus_delta_primary
        self.minus_delta_secondary = minus_delta_secondary
        self.plus_delta_secondary = plus_delta_secondary
        self.minus_delta_primary = minus_delta_primary

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, SchedulingSolution)
            and isinstance(solver, SchedulingCpSolver)
        ):
            raise ValueError(
                f"{self.__class__.__name__} extract constraints only "
                f"if solution and solver are related to a scheduling problem."
            )
        makespan_ub = solver.get_makespan_upper_bound()
        constraints = []
        for task in tasks_primary:
            start_time_j = current_solution.get_start_time(task)
            constraints += solver.add_constraint_on_task(
                task=task,
                start_or_end=StartOrEnd.START,
                sign=SignEnum.UEQ,
                time=max(0, start_time_j - self.minus_delta_primary),
            )
            constraints += solver.add_constraint_on_task(
                task=task,
                start_or_end=StartOrEnd.START,
                sign=SignEnum.LEQ,
                time=min(makespan_ub, start_time_j + self.plus_delta_primary),
            )
        for task in tasks_secondary:
            if task in tasks_primary:
                continue
            start_time_j = current_solution.get_start_time(task)
            if self.minus_delta_secondary == 0 and self.plus_delta_secondary == 0:
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.EQUAL,
                    time=start_time_j,
                )
            else:
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.UEQ,
                    time=max(
                        0,
                        start_time_j - self.minus_delta_secondary,
                    ),
                )
                constraints += solver.add_constraint_on_task(
                    task=task,
                    start_or_end=StartOrEnd.START,
                    sign=SignEnum.LEQ,
                    time=min(
                        makespan_ub,
                        start_time_j + self.plus_delta_secondary,
                    ),
                )
        return constraints


class ChainingConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        frac_fixed_chaining: float = 0.25,
    ):
        self.frac_fixed_chaining = frac_fixed_chaining

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, SchedulingSolution)
            and isinstance(solver, SchedulingCpSolver)
        ):
            raise ValueError(
                f"{self.__class__.__name__} extract constraints only "
                f"if solution and solver are related to a scheduling problem."
            )

        all_tasks = current_solution.problem.tasks_list
        tasks = random.sample(all_tasks, int(self.frac_fixed_chaining * len(all_tasks)))
        constraints = []
        for task1 in tasks:
            for task2 in tasks:
                if current_solution.get_end_time(
                    task1
                ) == current_solution.get_start_time(task2):
                    constraints += solver.add_constraint_chaining_tasks(
                        task1=task1, task2=task2
                    )
        return constraints


class MultimodeConstraintExtractor(BaseConstraintExtractor[Task]):
    """Extractor adding constraints on modes."""

    def __init__(
        self,
        fix_primary_tasks_modes: bool = False,
        fix_secondary_tasks_modes: bool = True,
    ):
        self.fix_primary_tasks_modes = fix_primary_tasks_modes
        self.fix_secondary_tasks_modes = fix_secondary_tasks_modes

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, MultimodeSolution)
            and isinstance(solver, MultimodeCpSolver)
        ):
            raise ValueError("current_solution and solver must manage tasks modes.")

        constraints = []
        if self.fix_primary_tasks_modes:
            for task in tasks_primary:
                constraints += solver.add_constraint_on_task_mode(
                    task=task, mode=current_solution.get_mode(task)
                )
        if self.fix_secondary_tasks_modes:
            for task in tasks_secondary:
                constraints += solver.add_constraint_on_task_mode(
                    task=task, mode=current_solution.get_mode(task)
                )
        return constraints


class SubtasksAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        fix_secondary_tasks_allocation: bool = False,
        frac_random_fixed_tasks: float = 0.6,
    ):
        self.frac_random_fixed_tasks = frac_random_fixed_tasks
        self.fix_secondary_tasks_modes = fix_secondary_tasks_allocation

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        if self.fix_secondary_tasks_modes:
            tasks = tasks_secondary
        else:
            all_tasks = current_solution.problem.tasks_list
            tasks = set(
                random.sample(
                    all_tasks,
                    int(self.frac_random_fixed_tasks * len(all_tasks)),
                )
            )
        return solver.add_constraint_same_allocation_as_ref(
            ref=current_solution, tasks=tasks
        )


class SubresourcesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        frac_random_fixed_unary_resources: float = 0.5,
    ):
        self.frac_random_fixed_unary_resources = frac_random_fixed_unary_resources

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        all_unary_resources = current_solution.problem.unary_resources_list
        unary_resources = set(
            random.sample(
                all_unary_resources,
                int(self.frac_random_fixed_unary_resources * len(all_unary_resources)),
            )
        )
        return solver.add_constraint_same_allocation_as_ref(
            ref=current_solution, unary_resources=unary_resources
        )


class NbChangesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        nb_changes_max: int = 10,
    ):
        self.nb_changes = nb_changes_max

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        return solver.add_constraint_on_nb_allocation_changes(
            ref=current_solution, nb_changes=self.nb_changes
        )


class NbUsagesAllocationConstraintExtractor(BaseConstraintExtractor[Task]):
    def __init__(
        self,
        plus_delta_nb_usages_total: int = 5,
        plus_delta_nb_usages_per_unary_resource: int = 3,
        minus_delta_nb_usages_per_unary_resource: int = 3,
    ):
        self.plus_delta_nb_usages_per_unary_resource = (
            plus_delta_nb_usages_per_unary_resource
        )
        self.minus_delta_nb_usages_per_unary_resource = (
            minus_delta_nb_usages_per_unary_resource
        )
        self.plus_delta_nb_usages_total = plus_delta_nb_usages_total

    def add_constraints(
        self,
        current_solution: TasksSolution[Task],
        solver: TasksCpSolver[Task],
        tasks_primary: set[Task],
        tasks_secondary: set[Task],
        **kwargs: Any,
    ) -> list[Any]:
        if not (
            isinstance(current_solution, AllocationSolution)
            and isinstance(solver, AllocationCpSolver)
        ):
            raise ValueError(
                "current_solution and solver must manage resource allocation."
            )
        constraints = []
        nb_usages_total = current_solution.compute_nb_unary_resource_usages()
        constraints += solver.add_constraint_on_total_nb_usages(
            SignEnum.LEQ, nb_usages_total + self.plus_delta_nb_usages_total
        )
        for unary_resource in current_solution.problem.unary_resources_list:
            nb_usages = current_solution.compute_nb_unary_resource_usages(
                unary_resources=(unary_resource,)
            )
            constraints += solver.add_constraint_on_unary_resource_nb_usages(
                unary_resource=unary_resource,
                sign=SignEnum.LEQ,
                target=nb_usages + self.plus_delta_nb_usages_per_unary_resource,
            )
            constraints += solver.add_constraint_on_unary_resource_nb_usages(
                unary_resource=unary_resource,
                sign=SignEnum.UEQ,
                target=nb_usages - self.minus_delta_nb_usages_per_unary_resource,
            )

        return constraints


def build_default_constraint_extractor(
    problem: TasksProblem, params_constraint_extractor: ParamsConstraintExtractor
) -> BaseConstraintExtractor:
    extractors = []
    if isinstance(problem, SchedulingProblem):
        extractors.append(
            SchedulingConstraintExtractor(
                plus_delta_primary=params_constraint_extractor.plus_delta_primary,
                minus_delta_primary=params_constraint_extractor.minus_delta_primary,
                plus_delta_secondary=params_constraint_extractor.plus_delta_secondary,
                minus_delta_secondary=params_constraint_extractor.minus_delta_secondary,
            )
        )
        if params_constraint_extractor.chaining:
            extractors.append(
                ChainingConstraintExtractor(
                    frac_fixed_chaining=params_constraint_extractor.frac_fixed_chaining
                )
            )
    if isinstance(problem, MultimodeProblem) and problem.is_multimode:
        extractors.append(
            MultimodeConstraintExtractor(
                fix_primary_tasks_modes=params_constraint_extractor.fix_primary_tasks_modes,
                fix_secondary_tasks_modes=params_constraint_extractor.fix_secondary_tasks_allocation,
            )
        )
    if isinstance(problem, AllocationProblem):
        if params_constraint_extractor.allocation_subtasks:
            extractors.append(
                SubtasksAllocationConstraintExtractor(
                    fix_secondary_tasks_allocation=params_constraint_extractor.fix_secondary_tasks_allocation,
                    frac_random_fixed_tasks=params_constraint_extractor.frac_random_fixed_tasks,
                )
            )
        if params_constraint_extractor.allocation_subresources:
            extractors.append(
                SubresourcesAllocationConstraintExtractor(
                    frac_random_fixed_unary_resources=params_constraint_extractor.frac_random_fixed_unary_resources
                )
            )
        if params_constraint_extractor.nb_changes:
            extractors.append(
                NbChangesAllocationConstraintExtractor(
                    nb_changes_max=params_constraint_extractor.nb_changes_max
                )
            )
        if params_constraint_extractor.nb_usages:
            extractors.append(
                NbUsagesAllocationConstraintExtractor(
                    plus_delta_nb_usages_total=params_constraint_extractor.plus_delta_nb_usages_total,
                    plus_delta_nb_usages_per_unary_resource=params_constraint_extractor.plus_delta_nb_usages_per_unary_resource,
                    minus_delta_nb_usages_per_unary_resource=params_constraint_extractor.minus_delta_nb_usages_per_unary_resource,
                )
            )
    return ConstraintExtractorList(extractors=extractors)
