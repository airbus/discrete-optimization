from copy import copy, deepcopy
from datetime import datetime
from typing import List, Optional

import cpmpy
from cpmpy import SolverLookup

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.allocation.problem import (
    TeamAllocationProblem,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.solvers.cpmpy import (
    CPMpyTeamAllocationSolverStoreConstraintInfo,
    MetaCpmpyConstraint,
    ModelisationAllocationCP,
)


class InteractSolve:
    def __init__(self, problem: TeamAllocationProblem):
        self.problem = problem
        self.current_problem = copy(problem)
        self.solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem)
        self.solver.init_model(modelisation_allocation=ModelisationAllocationCP.BINARY)
        soft = [
            copy(mc)
            for mc in self.solver.meta_constraints
            if mc.metadata["type"] in {"allocated_task", "same_allocation"}
        ]
        hard = [
            copy(mc)
            for mc in self.solver.meta_constraints
            if mc.metadata["type"] not in {"allocated_task", "same_allocation"}
        ]
        self.soft = soft
        self.hard = hard
        self.current_soft = soft
        self.current_hard = hard
        self.current_hard_str = set([str(c) for c in self.current_hard])
        self.current_soft_str = set([str(c) for c in self.current_soft])
        self.dropped_const = []
        self.dropped_tasks = []
        self.solver.model.constraints = []
        for m in self.solver.meta_constraints:
            self.solver.model.constraints.extend(m.constraints)

    def get_current_soft_constraints(self):
        return self.current_soft

    def drop_constraints(self, constraints: list[MetaCpmpyConstraint]):
        cstrs = []
        csoft = []
        for m in self.current_soft:
            if m in constraints:
                continue
            cstrs.extend(m.constraints)
            csoft.append(m)
        self.current_soft = csoft
        chard = []
        for h in self.current_hard:
            cstrs.extend(h.constraints)
            chard.append(h)
        self.current_hard = chard
        self.solver.model.constraints = cstrs
        for c in constraints:
            if c.metadata.get("type", None) == "allocated_task":
                self.dropped_tasks.append(c.metadata["task_index"])

    def put_as_hard(self, constraints: list[MetaCpmpyConstraint]):
        self.current_soft = [c for c in self.current_soft if c not in constraints]
        # self.current_soft_str = set([str(c) for c in self.current_soft])
        self.current_hard += [c for c in constraints if c not in self.current_hard]
        self.current_hard_str = set([str(c) for c in self.current_hard])
        cstrs = []
        for m in self.current_soft:
            cstrs.extend(m.constraints)
        for m in self.current_hard:
            cstrs.extend(m.constraints)
        self.solver.model.constraints = cstrs

    def solve_current_problem(self, **kwargs) -> tuple[StatusSolver, ResultStorage]:
        # temporary fix
        self.solver.cpm_solver = SolverLookup.get(
            self.solver.solver_name, self.solver.model
        )
        res = self.solver.solve(**kwargs)
        status = self.solver.status_solver
        # sol, fit = res.get_best_solution_fit()
        return status, res

    def solve_relaxed_problem(
        self, base_solution: Optional[TeamAllocationSolution] = None
    ):
        model = cpmpy.Model()
        allocation_binary = self.solver.variables["allocation_binary"]
        is_allocated = cpmpy.boolvar(
            shape=(self.problem.number_of_activity,), name="is_allocated"
        )
        for i in range(self.problem.number_of_activity):
            model += [
                is_allocated[i].implies(
                    sum([allocation_binary[i][j] for j in allocation_binary[i]]) == 1
                )
            ]
        weight = [1000 for i in range(self.problem.number_of_activity)]
        for t in self.dropped_tasks:
            weight[t] = -10
        cstrs = []
        for m in self.current_soft:
            if m.metadata["type"] != "allocated_task":
                cstrs.extend(m.constraints)
        for m in self.current_hard:
            cstrs.extend(m.constraints)
        delta_objective = 0
        if base_solution is not None:
            for i in range(self.problem.number_of_activity):
                if i not in self.dropped_tasks:
                    for j in allocation_binary[i]:
                        if j == base_solution.allocation[i]:
                            delta_objective += 1 - allocation_binary[i][j]
        model += cstrs
        model.maximize(
            -100 * delta_objective
            + sum(
                [
                    is_allocated[i] * weight[i]
                    for i in range(self.problem.number_of_activity)
                ]
            )
        )
        # cpm_solver = SolverLookup.get(self.solver.solver_name, model)
        res = model.solve(solver="ortools", time_limit=2)
        sol = self.solver.retrieve_current_solution()
        return sol

    def reset(self):
        self.current_soft = deepcopy(self.soft)
        self.current_hard = deepcopy(self.hard)
        self.current_hard_str = set([str(c) for c in self.current_hard])
        self.current_soft_str = set([str(c) for c in self.current_soft])
        self.solver.model.constraints = self.current_soft + self.current_hard
