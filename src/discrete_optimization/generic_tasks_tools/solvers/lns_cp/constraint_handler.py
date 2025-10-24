from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Generic, Iterable, Optional

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
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingCpSolver,
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    BaseConstraintExtractor,
    ParamsConstraintExtractor,
    build_default_constraint_extractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilder,
    build_default_neighbor_builder,
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
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class ObjectiveSubproblem(Enum):
    # default
    INITIAL_OBJECTIVE = 0  # keep initial objective of the solver
    # scheduling
    MAKESPAN_SUBTASKS = 1  # makespan on tasks subset
    SUM_START_SUBTASKS = 2  # sum of start time on tasks subset
    SUM_END_SUBTASKS = 3  # sum of end time on tasks subset
    GLOBAL_MAKESPAN = 4  # global makespan
    # allocation
    NB_TASKS_DONE = 5  # number of tasks with at least a resource allocated
    NB_UNARY_RESOURCES_USED = (
        6  # number of unary resources allocated to at least one task
    )


SCHEDULING_OBJECTIVES = (
    ObjectiveSubproblem.MAKESPAN_SUBTASKS,
    ObjectiveSubproblem.SUM_START_SUBTASKS,
    ObjectiveSubproblem.SUM_END_SUBTASKS,
    ObjectiveSubproblem.GLOBAL_MAKESPAN,
)

ALLOCATION_OBJECTIVES = (
    ObjectiveSubproblem.NB_TASKS_DONE,
    ObjectiveSubproblem.NB_UNARY_RESOURCES_USED,
)


class TasksConstraintHandler(ConstraintHandler, Generic[Task]):
    """Generic constraint handler for tasks related problems.

    Include constraints for scheduling, multimode, and allocation features if present.

    """

    def __init__(
        self,
        problem: TasksProblem,
        neighbor_builder: Optional[NeighborBuilder[Task]] = None,
        constraints_extractor: Optional[BaseConstraintExtractor] = None,
        params_constraint_extractor: Optional[ParamsConstraintExtractor] = None,
        objective_subproblem: ObjectiveSubproblem = ObjectiveSubproblem.INITIAL_OBJECTIVE,
    ):
        self.problem = problem
        if neighbor_builder is None:
            self.neighbor_builder = build_default_neighbor_builder(problem=problem)
        else:
            self.neighbor_builder = neighbor_builder
        if params_constraint_extractor is None:
            self.params_constraint_extractor = ParamsConstraintExtractor()
        else:
            self.params_constraint_extractor = params_constraint_extractor
        if constraints_extractor is None:
            self.constraints_extractor = build_default_constraint_extractor(
                problem=problem,
                params_constraint_extractor=self.params_constraint_extractor,
            )
        else:
            self.constraints_extractor = constraints_extractor
        self.objective_subproblem = objective_subproblem

        if self.objective_subproblem in SCHEDULING_OBJECTIVES and not isinstance(
            problem, SchedulingProblem
        ):
            raise ValueError(
                f"{self.objective_subproblem} objective possible only for a scheduling problem."
            )
        if self.objective_subproblem in ALLOCATION_OBJECTIVES and not isinstance(
            problem, AllocationProblem
        ):
            raise ValueError(
                f"{self.objective_subproblem} objective possible only for an allocation problem."
            )

    def adding_constraint_from_results_store(
        self,
        solver: TasksCpSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            **kwargs:

        Returns:
            list of added constraints

        """
        # current solution
        current_solution: TasksSolution[Task] = (
            self.extract_best_solution_from_last_iteration(
                result_storage=result_storage,
                result_storage_last_iteration=result_storage_last_iteration,
            )
        )

        # split tasks
        (tasks_primary, tasks_secondary) = self.neighbor_builder.find_subtasks(
            current_solution=current_solution
        )
        logger.debug(self.__class__.__name__)
        logger.debug(
            f"{len(tasks_primary)} in first set, {len(tasks_secondary)} in second set"
        )

        constraints = self.constraints_extractor.add_constraints(
            current_solution=current_solution,
            solver=solver,
            tasks_primary=tasks_primary,
            tasks_secondary=tasks_secondary,
        )

        # change objective and add constraint on it
        objective: Optional[Any] = None
        if self.objective_subproblem == ObjectiveSubproblem.INITIAL_OBJECTIVE:
            # keep current objective
            pass
        elif self.objective_subproblem == ObjectiveSubproblem.MAKESPAN_SUBTASKS:
            if isinstance(current_solution, SchedulingSolution) and isinstance(
                solver, SchedulingCpSolver
            ):
                objective = solver.get_subtasks_makespan_variable(
                    subtasks=tasks_primary
                )
        elif self.objective_subproblem == ObjectiveSubproblem.GLOBAL_MAKESPAN:
            if isinstance(current_solution, SchedulingSolution) and isinstance(
                solver, SchedulingCpSolver
            ):
                objective = solver.get_global_makespan_variable()
        elif self.objective_subproblem == ObjectiveSubproblem.SUM_START_SUBTASKS:
            if isinstance(current_solution, SchedulingSolution) and isinstance(
                solver, SchedulingCpSolver
            ):
                objective = solver.get_subtasks_sum_start_time_variable(
                    subtasks=tasks_primary
                )
        elif self.objective_subproblem == ObjectiveSubproblem.SUM_END_SUBTASKS:
            if isinstance(current_solution, SchedulingSolution) and isinstance(
                solver, SchedulingCpSolver
            ):
                objective = solver.get_subtasks_sum_end_time_variable(
                    subtasks=tasks_primary
                )
        elif self.objective_subproblem == ObjectiveSubproblem.NB_TASKS_DONE:
            if isinstance(current_solution, AllocationSolution) and isinstance(
                solver, AllocationCpSolver
            ):
                objective = -solver.get_nb_tasks_done_variable()
        elif self.objective_subproblem == ObjectiveSubproblem.NB_UNARY_RESOURCES_USED:
            if isinstance(current_solution, AllocationSolution) and isinstance(
                solver, AllocationCpSolver
            ):
                objective = solver.get_nb_unary_resources_used_variable()
        if objective is not None:
            solver.minimize_variable(objective)

        return constraints
