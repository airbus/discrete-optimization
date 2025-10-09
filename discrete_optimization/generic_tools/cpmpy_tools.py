#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Thanks to Leuven university for the cpmpy library.
from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import cpmpy
from cpmpy import SolverLookup
from cpmpy.expressions.core import BoolVal, Expression
from cpmpy.expressions.variables import NDVarArray, _BoolVarImpl
from cpmpy.model import Model
from cpmpy.solvers.ortools import OrtSolutionPrinter
from cpmpy.solvers.solver_interface import ExitStatus, SolverInterface, SolverStatus
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
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    ParametersCp,
    SignEnum,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
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


class _CpmpyCbClass:
    def __init__(self, do_solver: "CpmpySolver", callback: Callback):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def callback_func(self):
        sol = self.do_solver.retrieve_current_solution()
        fit = self.do_solver.aggreg_from_sol(sol)
        self.res.append((sol, fit))
        self.nb_solutions += 1
        # end of step callback: stopping?
        stopping = self.callback.on_step_end(
            step=self.nb_solutions, res=self.res, solver=self.do_solver
        )


class CpmpySolver(CpSolver):
    """Generic cpmpy solver."""

    model: Optional[Model] = None
    cpm_status: SolverStatus = SolverStatus("Model")
    cpm_solver: Optional[SolverInterface] = None

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        solver_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function=params_objective_function, **kwargs
        )
        if solver_name is None:
            self.solver_name = "ortools"
        else:
            self.solver_name = solver_name
        self.cb_object: _CpmpyCbClass = None

    def minimize_variable(self, var: Any) -> None:
        self.model.minimize(var)

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        if sign == SignEnum.LEQ:
            cstr = var <= value
        elif sign == SignEnum.UEQ:
            cstr = var >= value
        elif sign == SignEnum.EQUAL:
            cstr = var == value
        elif sign == SignEnum.LESS:
            cstr = var < value
        elif sign == SignEnum.UP:
            cstr = var > value
        else:
            raise NotImplementedError(sign)
        self.model.add(cstr)
        return [cstr]

    def create_callback_function(self, callback: Callback):
        self.cb_object = _CpmpyCbClass(do_solver=self, callback=callback)
        return self.cb_object.callback_func

    def reset_cpm_solver(self):
        """Reset wrapped solver.

        Call it so that modifications on `self.model` will be taken into account by next call to `self.solve()`.
        Else, the cpmpy wrapped solver initialized at first solve will be used and not taken into account model changes.

        """
        self.cpm_solver = None

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        solver_kwargs = dict(kwargs)
        if self.cpm_solver is None:  # this is the first solve call
            self.cpm_solver = SolverLookup.get(self.solver_name, self.model)
        solver_allowed_params = inspect.signature(self.cpm_solver.solve).parameters
        if "display" in solver_allowed_params.keys():
            solver_kwargs["display"] = self.create_callback_function(
                callback=callbacks_list
            )
            if self.solver_name == "exact":
                solver_kwargs.pop("display")

        if self.solver_name == "ortools":
            solver_kwargs["solution_callback"] = _OrtoolsCpSatCallbackViaCpmpy(
                do_solver=self,
                solver_obj=self.cpm_solver,
                callback=callbacks_list,
                retrieve_stats=True,
            )
            verbose = solver_kwargs.pop("verbose", False)
            solver_kwargs.update(
                dict(
                    num_search_workers=parameters_cp.nb_process
                    if parameters_cp.multiprocess
                    else 1,
                    log_search_progress=verbose,
                )
            )
        elif self.solver_name == "gurobi":
            solver_kwargs.update(
                dict(
                    Threads=parameters_cp.nb_process
                    if parameters_cp.multiprocess
                    else 1
                )
            )
        self.cpm_solver.solve(time_limit=time_limit, **solver_kwargs)
        self.cpm_status = self.cpm_solver.status()
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
                fine_method=fine_method,
                soft=soft,
                hard=hard,
                include_all_trivial_false_constraints=True,
                **fine_method_kwargs,
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
                elif isinstance(c, BoolVal):  # manage False <-> BoolVal(False)
                    if c.value() in set(meta_normalized):
                        ms_meta_constraints.add(meta)
        return list(ms_meta_constraints)

    def _compute_meta_minimal_meta_constraint_set_using_fine_constraint_method(
        self,
        fine_method: Callable[[...], list[Expression]],
        soft: Optional[list[MetaCpmpyConstraint]] = None,
        hard: Optional[list[MetaCpmpyConstraint]] = None,
        include_all_trivial_false_constraints: bool = False,
        **kwargs: Any,
    ) -> list[MetaCpmpyConstraint]:
        """

        Args:
            fine_method:  method used to extract the minimal set of fine constraints
            soft: soft meta-constraints
            hard: hard meta-constraints
            include_all_trivial_false_constraints: whether to include all trivial False constraints (for mcs) or only
                first one (for mus)
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
        # handle trivial False constraints
        soft_with_trivial_false = [
            meta for cstr, meta in zip(soft_cpmpy, soft) if is_trivially_false(cstr)
        ]
        soft_wo_trivial_false = [
            meta for meta in soft if meta not in soft_with_trivial_false
        ]
        soft_cpmpy_wo_trivial_false = [
            cstr for cstr in soft_cpmpy if not (is_trivially_false(cstr))
        ]
        if (
            len(soft_with_trivial_false) > 0
            and not include_all_trivial_false_constraints
        ):
            return soft_with_trivial_false[:1]  # only first meta containing False
        cstr2meta = dict(zip(soft_cpmpy, soft))
        ms_constraints = fine_method(
            soft=soft_cpmpy_wo_trivial_false, hard=hard_cpmpy, **kwargs
        )
        return soft_with_trivial_false + [cstr2meta[cstr] for cstr in ms_constraints]

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


class _OrtoolsCpSatCallbackViaCpmpy(CpSolverSolutionCallback):
    def __init__(
        self,
        do_solver: CpmpySolver,
        solver_obj: SolverInterface,
        callback: Callback,
        retrieve_stats: bool = False,
    ):
        super().__init__()
        self._varmap = solver_obj._varmap
        self._cpm_vars = solver_obj.user_vars
        self.do_solver = do_solver
        self.callback = callback
        self.retrieve_stats = retrieve_stats
        self.res = do_solver.create_result_storage()
        if retrieve_stats:
            self.res.stats = []
        self.nb_solutions = 0

    def on_solution_callback(self) -> None:
        self.store_current_solution()
        self.nb_solutions += 1
        # end of step callback: stopping?
        try:
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        if stopping:
            self.StopSearch()

    def store_current_solution(self):
        # Store the cpsat values in the cpm vars, before calling retrieve current solution (taken from
        if len(self._cpm_vars):
            # populate values before printing
            for cpm_var in self._cpm_vars:
                # it might be an NDVarArray
                if hasattr(cpm_var, "flat"):
                    for cpm_subvar in cpm_var.flat:
                        cpm_subvar._value = self.Value(self._varmap[cpm_subvar])
                elif isinstance(cpm_var, _BoolVarImpl):
                    cpm_var._value = bool(self.Value(self._varmap[cpm_var]))
                else:
                    cpm_var._value = self.Value(self._varmap[cpm_var])
        sol = self.do_solver.retrieve_current_solution()
        fit = self.do_solver.aggreg_from_sol(sol)
        self.res.append((sol, fit))
        if self.retrieve_stats:
            self.res.stats.append(
                {
                    "bound": self.BestObjectiveBound(),
                    "obj": self.ObjectiveValue(),
                    "time": self.UserTime(),
                    "num_conflicts": self.NumConflicts(),
                }
            )
        # update current bound and value
        self.do_solver._current_internal_objective_best_value = self.ObjectiveValue()
        self.do_solver._current_internal_objective_best_bound = (
            self.BestObjectiveBound()
        )


def is_trivially_false(cstr: Expression) -> bool:
    """Check if a cpmpy constraint is trivially False.

    This means it is always equal to False.

    """
    if isinstance(cstr, BoolVal) or isinstance(cstr, bool):
        return not cstr
    else:
        return False
