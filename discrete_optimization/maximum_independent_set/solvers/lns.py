import random
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver


class OrtoolsCpSatMisConstraintHandler(OrtoolsCpSatConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: MisProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: CpSatMisSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        if not isinstance(solver, CpSatMisSolver):
            raise ValueError("solver must a CpSatMisSolver for this constraint.")
        lns_constraints = []
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, MisSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a MisSolution."
            )
        nb_chosen = sum(current_solution.chosen)
        idx_chosen = list(np.where(current_solution.chosen)[0])
        subpart_chosen = set(
            random.sample(
                idx_chosen,
                int(self.fraction_to_fix * nb_chosen),
            )
        )
        in_set = solver.variables["in_set"]
        for idx in subpart_chosen:
            lns_constraints.append(solver.cp_model.Add(in_set[idx] == 1))
        return lns_constraints


class DestroyOrtoolsCpSatMisConstraintHandler(OrtoolsCpSatConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: MisProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: CpSatMisSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        if not isinstance(solver, CpSatMisSolver):
            raise ValueError("solver must a CpSatMisSolver for this constraint.")
        lns_constraints = []
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, MisSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a MisSolution."
            )
        nb_chosen = sum(current_solution.chosen)
        idx_chosen = list(np.where(current_solution.chosen)[0])
        subpart_chosen = set(
            random.sample(
                idx_chosen,
                int(self.fraction_to_fix * nb_chosen),
            )
        )
        in_set = solver.variables["in_set"]
        for idx in subpart_chosen:
            lns_constraints.append(solver.cp_model.Add(in_set[idx] == 0))

        # subpart_chosen_2 = set(
        #     random.sample(
        #         range(self.problem.number_nodes),
        #         int(0.2 * self.problem.number_nodes),
        #     )
        # )
        # for idx in subpart_chosen_2:
        #     if idx not in subpart_chosen:
        #         lns_constraints.append(
        #             solver.cp_model.Add(in_set[idx] == current_solution.chosen[idx])
        #         )
        return lns_constraints


class AllVarsOrtoolsCpSatMisConstraintHandler(OrtoolsCpSatConstraintHandler):
    """
    Fix fraction of all variables, not only the ones already chosen.
    """

    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: MisProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: CpSatMisSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        if not isinstance(solver, CpSatMisSolver):
            raise ValueError("solver must a CpSatMisSolver for this constraint.")
        lns_constraints = []
        current_solution, _ = result_storage.list_solution_fits[-1]
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, MisSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a MisSolution."
            )
        subpart_chosen = set(
            random.sample(
                range(self.problem.number_nodes),
                int(self.fraction_to_fix * self.problem.number_nodes),
            )
        )
        in_set = solver.variables["in_set"]
        for idx in subpart_chosen:
            lns_constraints.append(
                solver.cp_model.Add(in_set[idx] == current_solution.chosen[idx])
            )
        return lns_constraints


class LocalMovesOrtoolsCpSatMisConstraintHandler(OrtoolsCpSatConstraintHandler):
    hyperparameters = [
        IntegerHyperparameter(name="nb_moves", default=1, low=0, high=10),
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(
        self, problem: MisProblem, nb_moves: int = 5, fraction_to_fix: float = 0.9
    ):
        self.problem = problem
        self.nb_moves = nb_moves
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: CpSatMisSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Constraint]:
        if not isinstance(solver, CpSatMisSolver):
            raise ValueError("solver must a CpSatMisSolver for this constraint.")
        lns_constraints = []
        current_solution, _ = result_storage.get_last_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, MisSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a MisSolution."
            )
        change = [
            solver.cp_model.NewBoolVar(name=f"change_{k}")
            for k in range(self.problem.number_nodes)
        ]
        in_set = solver.variables["in_set"]
        for k in range(self.problem.number_nodes):
            if current_solution.chosen[k] == 1:
                lns_constraints.append(
                    solver.cp_model.AddImplication(change[k], in_set[k].Not())
                )
                lns_constraints.append(
                    solver.cp_model.AddImplication(change[k].Not(), in_set[k])
                )
                # lns_constraints.append(solver.cp_model.AddImplication(in_set[k].Not(), change[k]))
            else:
                lns_constraints.append(
                    solver.cp_model.AddImplication(change[k], in_set[k])
                )
                lns_constraints.append(
                    solver.cp_model.AddImplication(change[k].Not(), in_set[k].Not())
                )

        idx_chosen = list(np.where(current_solution.chosen)[0])
        nb_chosen = sum(current_solution.chosen)
        subpart_chosen = set(
            random.sample(
                idx_chosen,
                int(self.fraction_to_fix * nb_chosen),
            )
        )
        in_set = solver.variables["in_set"]
        for idx in subpart_chosen:
            lns_constraints.append(solver.cp_model.Add(in_set[idx] == 1))
        lns_constraints.append(solver.cp_model.Add(sum(change) <= self.nb_moves))
        return lns_constraints

    # def remove_constraints_from_previous_iteration(
    #     self,
    #     solver: OrtoolsCpSatSolver,
    #     previous_constraints: Iterable[Constraint],
    #     **kwargs: Any,
    # ) -> None:
    #     solver.init_model()
