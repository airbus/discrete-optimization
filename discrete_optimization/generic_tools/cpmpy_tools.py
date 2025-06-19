#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Thanks to Leuven university for the cpmpy library.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import cpmpy
from cpmpy.expressions.core import Expression
from cpmpy.expressions.variables import NDVarArray
from cpmpy.model import Model
from cpmpy.solvers.solver_interface import ExitStatus, SolverStatus
from cpmpy.tools import make_assump_model
from cpmpy.tools.explain.mcs import mcs_grow, mcs_opt
from cpmpy.tools.explain.mus import (
    mus,
    mus_naive,
    optimal_mus,
    optimal_mus_naive,
    quickxplain,
    quickxplain_naive,
    smus,
)

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import CpSolver
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.unsat_tools import MetaConstraint

map_exitstatus2statussolver = {
    ExitStatus.NOT_RUN: StatusSolver.UNKNOWN,
    ExitStatus.OPTIMAL: StatusSolver.OPTIMAL,
    ExitStatus.FEASIBLE: StatusSolver.SATISFIED,
    ExitStatus.UNSATISFIABLE: StatusSolver.UNSATISFIABLE,
    ExitStatus.ERROR: StatusSolver.UNKNOWN,
    ExitStatus.UNKNOWN: StatusSolver.UNKNOWN,
}


class CpmpyExplainUnsatMethod(Enum):
    core = "core"
    mus = "mus"
    quickxplain = "quickxplain"
    optimal_mus = "optimal_mus"
    smus = "smus"
    mus_naive = "mus_naive"
    quickxplain_naive = "quickxplain_naive"
    optimal_mus_naive = "optimal_mus_naive"


def get_core_constraints(soft, hard, **kwargs):
    (m, soft, assump) = make_assump_model(soft, hard=hard)
    s = cpmpy.SolverLookup.get(kwargs.get("solver", "ortools"), m)
    # create dictionary from assump to soft
    dmap = dict(zip(assump, soft))
    assert not s.solve(assumptions=assump, **kwargs), "MUS: model must be UNSAT"
    core = set(s.get_core())
    return [dmap[avar] for avar in core]


map_explainunsatmethod2fun = {
    CpmpyExplainUnsatMethod.core: get_core_constraints,
    CpmpyExplainUnsatMethod.mus: mus,
    CpmpyExplainUnsatMethod.quickxplain: quickxplain,
    CpmpyExplainUnsatMethod.optimal_mus: optimal_mus,
    CpmpyExplainUnsatMethod.smus: smus,
    CpmpyExplainUnsatMethod.mus_naive: mus_naive,
    CpmpyExplainUnsatMethod.quickxplain_naive: quickxplain_naive,
    CpmpyExplainUnsatMethod.optimal_mus_naive: optimal_mus_naive,
}


class CpmpyCorrectUnsatMethod(Enum):
    mcs_opt = "mcs_opt"
    mcs_grow = "mcs_grow"


map_correctunsatmethod2fun = {
    CpmpyCorrectUnsatMethod.mcs_opt: mcs_opt,
    CpmpyCorrectUnsatMethod.mcs_grow: mcs_grow,
}


class CpmpySolver(CpSolver):
    """Generic cpmpy solver."""

    model: Optional[Model] = None
    cpm_status: SolverStatus = SolverStatus("Model")

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any,
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        solver = kwargs.get("solver", "ortools")
        self.model.solve(solver, time_limit=time_limit, **kwargs)
        self.cpm_status = self.model.cpm_status
        self.status_solver = map_exitstatus2statussolver[self.cpm_status.exitstatus]

        if self.cpm_status.exitstatus in [ExitStatus.UNSATISFIABLE, ExitStatus.ERROR]:
            res = self.create_result_storage([])
        else:
            sol = self.retrieve_current_solution()
            fit = self.aggreg_from_sol(sol)
            res = self.create_result_storage(
                [(sol, fit)],
            )

        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def explain_unsat_meta(
        self,
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        cpmpy_method: CpmpyExplainUnsatMethod = CpmpyExplainUnsatMethod.mus,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """Explain unsatisfiability of the problem via meta-constraints.

        Meta-constraints are gathering several finer constraints in order to be more human-readable.
        We construct the corresponding cpmpy constraint via `cpmpy.all(meta.constraints)`.

        Args:
            soft: list of soft meta-constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard meta-constraints (that have no sense to be removed
                like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mus` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mus.mus`)

        Returns:
            minimal subset of soft meta-constraints leading to unsatisfiability.

        Note:
            running several times may lead to a different (minimal) subset of meta-constraints.

        """
        fine_method = self.explain_unsat_fine
        fine_method_kwargs = dict(cpmpy_method=cpmpy_method, **kwargs)
        return (
            self._compute_meta_minimal_meta_constraint_set_using_fine_constraint_method(
                fine_method=fine_method, soft=soft, hard=hard, **fine_method_kwargs
            )
        )

    def explain_unsat_deduced_meta(
        self,
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        cpmpy_method: CpmpyExplainUnsatMethod = CpmpyExplainUnsatMethod.mus,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """Explain unsatisfiability of the problem via meta-constraints.

        Meta-constraints are gathering several finer constraints in order to be more human readable.
        We compute the minimal subset of fine constraints included in the metaconstraints and deduce
        the corresponding meta-constraints impacted.

        Args:
            soft: list of soft meta-constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard meta-constraints (that have no sense to be removed
                like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mus` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mus.mus`)

        Returns:
            subset of soft meta-constraints leading to unsatisfiability (corresponding fine constraints subset being minimal).

        Note:
            running several times may lead to a different subset of meta-constraints.

        """
        fine_method = self.explain_unsat_fine
        fine_method_kwargs = dict(cpmpy_method=cpmpy_method, **kwargs)
        return (
            self._deduce_meta_minimal_constraint_set_from_minimal_fine_constraint_set(
                fine_method=fine_method, soft=soft, hard=hard, **fine_method_kwargs
            )
        )

    def explain_unsat_fine(
        self,
        soft: Optional[list[Expression]] = None,
        hard: Optional[list[Expression]] = None,
        cpmpy_method: CpmpyExplainUnsatMethod = CpmpyExplainUnsatMethod.mus,
        **kwargs: Any,
    ) -> list[Expression]:
        """Explain unsatisfiability of the problem via fine cpmpy constraints.

        Args:
            soft: list of soft constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard constraints (that have no sense to be removed like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mus` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mus.mus`)

        Returns:
            subset minimal list of soft constraints leading to unsatisfiability.

        Note:
            running several times may lead to a different (minimal) subset of constraints.

        """
        assert self.status_solver == StatusSolver.UNSATISFIABLE, (
            "self.solve() must have been run "
            "and self.status_solver must be SolverStatus.UNSATISFIABLE"
        )
        if soft is None:
            soft = self.get_soft_constraints()
        if hard is None:
            hard = self.get_hard_constraints()
        fun = map_explainunsatmethod2fun[cpmpy_method]
        return fun(soft=soft, hard=hard, **kwargs)

    def correct_unsat_meta(
        self,
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        cpmpy_method: CpmpyCorrectUnsatMethod = CpmpyCorrectUnsatMethod.mcs_opt,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """Correct unsatisfiability of the problem with a minimal set of meta-constraints.

        Meta-constraints are gathering several finer constraints in order to be more human-readable.
        We construct the corresponding cpmpy constraint via `cpmpy.all(meta.constraints)`.

        Args:
            soft: list of soft meta-constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard meta-constraints (that have no sense to be removed
                like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mcs` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mcs.mcs_opt`)

        Returns:
            minimal subset of soft meta-constraints leading to unsatisfiability.

        Note:
            running several times may lead to a different (minimal) subset of meta-constraints.

        """
        fine_method = self.correct_unsat_fine
        fine_method_kwargs = dict(cpmpy_method=cpmpy_method, **kwargs)
        return (
            self._compute_meta_minimal_meta_constraint_set_using_fine_constraint_method(
                fine_method=fine_method, soft=soft, hard=hard, **fine_method_kwargs
            )
        )

    def correct_unsat_deduced_meta(
        self,
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        cpmpy_method: CpmpyCorrectUnsatMethod = CpmpyCorrectUnsatMethod.mcs_opt,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """Correct unsatisfiability of the problem with a set of meta-constraints.

        Meta-constraints are gathering several finer constraints in order to be more human-readable.
        We compute the minimal subset of fine constraints included in the metaconstraints and deduce
        the corresponding meta-constraints impacted.

        Args:
            soft: list of soft meta-constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard meta-constraints (that have no sense to be removed
                like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mcs` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mcs.mcs_opt`)

        Returns:
            subset of soft meta-constraints leading to unsatisfiability (corresponding fine constraints subset being minimal).

        Note:
            running several times may lead to a different subset of meta-constraints.

        """
        fine_method = self.correct_unsat_fine
        fine_method_kwargs = dict(cpmpy_method=cpmpy_method, **kwargs)
        return (
            self._deduce_meta_minimal_constraint_set_from_minimal_fine_constraint_set(
                fine_method=fine_method, soft=soft, hard=hard, **fine_method_kwargs
            )
        )

    def correct_unsat_fine(
        self,
        soft: Optional[list[Expression]] = None,
        hard: Optional[list[Expression]] = None,
        cpmpy_method: CpmpyCorrectUnsatMethod = CpmpyCorrectUnsatMethod.mcs_opt,
        **kwargs: Any,
    ) -> list[Expression]:
        """Correct unsatisfiability of the problem with a minimal set of (fine) cpmpy constraints.

        Args:
            soft: list of soft constraints (that could be removed from model).
                Default to the ones returned by `get_default_soft_meta_constraints()`.
            hard: list of hard constraints (that have no sense to be removed like physical facts or modelling choices).
                Default to the ones returned by `get_default_hard_meta_constraints()`.
            cpmpy_method: corresponding to the function of `cpmpy.tools.explain.mcs` to be used
            **kwargs: passed to the cpmpy_method (like `cpmpy.tools.explain.mcs.mcs_opt`)

        Returns:
            subset minimal list of soft constraints leading to unsatisfiability.


        """
        assert self.status_solver == StatusSolver.UNSATISFIABLE, (
            "self.solve() must have been run "
            "and self.status_solver must be SolverStatus.UNSATISFIABLE"
        )
        if soft is None:
            soft = self.get_soft_constraints()
        if hard is None:
            hard = self.get_hard_constraints()
        fun = map_correctunsatmethod2fun[cpmpy_method]
        return fun(soft=soft, hard=hard, **kwargs)

    def _deduce_meta_minimal_constraint_set_from_minimal_fine_constraint_set(
        self,
        fine_method: Callable[[...], list[Expression]],
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """

        Args:
            fine_method:  method used to extract the minimal set of fine constraints
            soft: soft meta-constraints
            hard: hard meta-constraints
            **kwargs: kwargs for fine_method (apart from soft and hard parameters)

        Returns:

        """
        if soft is None:
            soft = self.get_soft_meta_constraints()
        if hard is None:
            hard = self.get_hard_meta_constraints()
        soft_normalized = _normalize_metaconstraints(soft)
        hard_normalized = _normalize_metaconstraints(hard)
        soft_fine = _convert_normalized_metaconstraints_to_constraints(soft_normalized)
        hard_fine = _convert_normalized_metaconstraints_to_constraints(hard_normalized)
        ms_constraints = fine_method(soft=soft_fine, hard=hard_fine, **kwargs)
        ms_meta_constraints = set()
        for c in ms_constraints:
            for meta, meta_normalized in zip(soft, soft_normalized):
                if c in set(meta_normalized):
                    ms_meta_constraints.add(meta)
        return list(ms_meta_constraints)

    def _compute_meta_minimal_meta_constraint_set_using_fine_constraint_method(
        self,
        fine_method: Callable[[...], list[Expression]],
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """

        Args:
            fine_method:  method used to extract the minimal set of fine constraints
            soft: soft meta-constraints
            hard: hard meta-constraints
            **kwargs: kwargs for fine_method (apart from soft and hard parameters)

        Returns:

        """
        if soft is None:
            soft = self.get_soft_meta_constraints()
        if hard is None:
            hard = self.get_hard_meta_constraints()
        soft_normalized = _normalize_metaconstraints(soft)
        hard_normalized = _normalize_metaconstraints(hard)
        soft_cpmpy = [cpmpy.all(meta.constraints) for meta in soft_normalized]
        hard_cpmpy = [cpmpy.all(meta.constraints) for meta in hard_normalized]
        cstr2meta = {cstr: meta for cstr, meta in zip(soft_cpmpy, soft)}
        ms_constraints = fine_method(soft=soft_cpmpy, hard=hard_cpmpy, **kwargs)
        return [cstr2meta[cstr] for cstr in ms_constraints]

    def get_others_meta_constraint(
        self, meta_constraints: Optional[list[MetaCpmpyConstraint]] = None
    ) -> MetaCpmpyConstraint:
        """Create a meta-constraint gathering all remaining constraints.

        Create a meta-constraint named "others" containing all model constraints not already
        taken into account by the given meta-constraints.

        Args:
            meta_constraints: meta constraints to consider. By default,
                `self.get_soft_meta_constraints() + self.get_hard_meta_constraints()`.

        Returns:
            a meta constraint gathering remaining constraints

        """
        if meta_constraints is None:
            meta_constraints = (
                self.get_soft_meta_constraints() + self.get_hard_meta_constraints()
            )
        solver_constraints = _get_normalized_constraints(self.model.constraints)
        constraints_from_meta = _convert_normalized_metaconstraints_to_constraints(
            _normalize_metaconstraints(meta_constraints)
        )
        constraints_from_meta_ids = {id(c) for c in constraints_from_meta}
        remaining_constraints = [
            c for c in solver_constraints if id(c) not in constraints_from_meta_ids
        ]
        return MetaCpmpyConstraint(name="others", constraints=remaining_constraints)

    def get_soft_constraints(self) -> list[Expression]:
        """Get soft fine constraints defining the internal model.

        To be used to explain unsatisfiability. See `explain_unsat_fine()`.
        Default implementation: all constraints from `self.model`.
        To be overriden according to problems.

        Returns:
            default set of soft constraints defining the problem

        """
        return self.model.constraints

    def get_hard_constraints(self) -> list[Expression]:
        """Get hard fine constraints defining the internal model.

        To be used to explain unsatisfiability. See `explain_unsat_fine()`.

        Returns:
            default set of hard constraints defining the problem

        """
        return []

    def get_soft_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        """Get soft meta-constraints defining the internal model.

        To be used to explain unsatisfiability. See `explain_unsat_meta()`.
        Default implementation: all constraints from `self.model`.
        To be overriden according to problems.

        Returns:
            default set of soft meta-constraints defining the problem

        """
        raise NotImplementedError("No meta constraints defined for this model.")

    def get_hard_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        """Get hard fine meta-constraints defining the internal model.

        To be used to explain unsatisfiability. See `explain_unsat_meta()`.

        Returns:
            default set of hard meta-constraints defining the problem

        """
        return [
            MetaCpmpyConstraint(
                name="hard constraints", constraints=self.get_hard_constraints()
            )
        ]

    @abstractmethod
    def retrieve_current_solution(self) -> Solution:
        """Construct a do solution from the cpmpy solver internal solution.

        It will be called after self.model.solve()

        Returns:
            the solution, at do format.

        """
        ...


class MetaCpmpyConstraint(MetaConstraint[Expression]):
    def normalize(self) -> None:
        """Split NDVarArray constraints into atomic ones."""
        self.constraints = _get_normalized_constraints(self.constraints)

    def to_normalized(self) -> MetaCpmpyConstraint:
        """Create a new meta constraints with NDVarArray constraints splitted into atomic ones."""
        return MetaCpmpyConstraint(
            name=self.name, constraints=_get_normalized_constraints(self.constraints)
        )


def _get_normalized_constraints(
    original_constraints: list[Expression],
) -> list[Expression]:
    constraints = []
    for c in original_constraints:
        if isinstance(c, NDVarArray):
            constraints.extend(c.ravel().tolist())
        else:
            constraints.append(c)
    return constraints


def _convert_normalized_metaconstraints_to_constraints(
    meta_constraints: list[MetaCpmpyConstraint],
) -> list[Expression]:
    return list(set(c for meta in meta_constraints for c in meta.constraints))


def _normalize_metaconstraints(
    meta_constraints: list[MetaCpmpyConstraint],
) -> list[MetaCpmpyConstraint]:
    return [meta.to_normalized() for meta in meta_constraints]
