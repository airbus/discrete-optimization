import random
from typing import Any, Iterable, Optional

import numpy as np
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import OrtoolsCPSatConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)


class MisOrtoolsCPSatConstraintHandler(OrtoolsCPSatConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: MisProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MisOrtoolsSolver,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any
    ) -> Iterable[Constraint]:
        if not isinstance(solver, MisOrtoolsSolver):
            raise ValueError("solver must a MisOrtoolsSolver for this constraint.")
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
