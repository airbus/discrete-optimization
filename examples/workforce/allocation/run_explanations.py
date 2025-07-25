#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from copy import copy, deepcopy
from datetime import datetime
from typing import Optional

import cpmpy
import plotly.io as pio
from cpmpy import SolverLookup

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpyCorrectUnsatMethod,
    CpmpyExplainUnsatMethod,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.workforce.allocation.parser import (
    build_allocation_problem_from_scheduling,
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
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.allocation.utils import plot_allocation_solution
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    generate_allocation_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.utils import (
    alloc_solution_to_alloc_sched_solution,
    build_scheduling_problem_from_allocation,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.


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

        # solver.
        # solver.model.constraints.extend(h.constraints)
        # all_names = {str(constraint.name) for constraint in constraints
        #             for c in constraints}
        # self.current_soft = [c for c in self.current_soft
        #                      if str(c) not in all_names]
        # self.solver.model.constraints = self.current_soft + self.current_hard

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
            print(m.metadata["type"])
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


def run_disruption_creation():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    nb_teams = allocation_problem.evaluate(sol)["nb_teams"]
    # fig = plot_allocation_solution(problem=allocation_problem, sol=sol, display=False,
    #                                ref_date=datetime(year=2024, month=1, day=1))
    # fig.show()
    ds = [
        generate_allocation_disruption(
            original_allocation_problem=allocation_problem,
            original_solution=sol,
            params_randomness=ParamsRandomness(
                lower_nb_disruption=1,
                upper_nb_disruption=1,
                lower_nb_teams=1,
                upper_nb_teams=2,
                lower_time=0,
                upper_time=600,
                duration_discrete_distribution=(
                    [15, 30, 60, 120],
                    [0.25, 0.25, 0.25, 0.25],
                ),
            ),
        )
        for i in range(10)
    ]
    # problem = ds[0]["new_allocation_problem"]
    allocation_problem.allocation_additional_constraint.nb_max_teams = 9
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem=allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationCP.BINARY)
    res = solver.solve(time_limit=2)
    print(solver.status_solver)
    print(solver.get_types_of_meta_constraints())
    soft = [
        mc
        for mc in solver.meta_constraints
        if mc.metadata["type"] in {"nb_max_teams", "same_allocation", "clique_overlap"}
    ]
    hard = [
        mc
        for mc in solver.meta_constraints
        if mc.metadata["type"]
        not in {"nb_max_teams", "same_allocation", "clique_overlap"}
    ]
    explanations = solver.explain_unsat_meta(
        soft=soft,
        hard=hard,
        cpmpy_method=CpmpyExplainUnsatMethod.mus,
        solver="exact",
    )
    start_shift = datetime(day=1, month=7, year=2025).timestamp()
    sched_problem = build_scheduling_problem_from_allocation(
        problem=allocation_problem, horizon_start_shift=start_shift
    )
    sched_scheduling = alloc_solution_to_alloc_sched_solution(
        problem=sched_problem, solution=sol
    )
    tasks_in_conflict = []
    for mc in explanations:
        if mc.metadata["type"] == "allocated_task":
            print(mc.metadata)
            tasks_in_conflict.append(mc.metadata["task_index"])
        if mc.metadata["type"] == "clique_overlap":
            print(mc.metadata)
            tasks_in_conflict.extend(list(mc.metadata["tasks_index"]))
        print(mc.metadata)
    plotly_schedule_comparison(
        base_solution=sched_scheduling,
        updated_solution=sched_scheduling,
        problem=sched_problem,
        use_color_map_per_task=True,
        plot_team_breaks=True,
        color_map_per_task={i: "red" for i in tasks_in_conflict},
        display=True,
    )
    for exp in explanations:
        print(exp)

    soft = [
        mc
        for mc in solver.meta_constraints
        if mc.metadata["type"] in {"same_allocation", "allocated_task"}
    ]
    hard = [
        mc
        for mc in solver.meta_constraints
        if mc.metadata["type"] not in {"same_allocation", "allocated_task"}
    ]
    mcs = solver.correct_unsat_meta(
        soft=soft,
        hard=hard,
        cpmpy_method=CpmpyCorrectUnsatMethod.mcs_grow,
        solver="exact",
    )
    for mc in mcs:
        print(mc.metadata)


def interactive_solving():
    kwargs_mus = {"solver": "exact", "cpmpy_method": CpmpyExplainUnsatMethod.mus}
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    allocation_problem.allocation_additional_constraint.nb_max_teams = 8
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem=allocation_problem)

    start_shift = datetime(day=1, month=7, year=2025).timestamp()
    sched_problem = build_scheduling_problem_from_allocation(
        problem=allocation_problem, horizon_start_shift=start_shift
    )
    sched_scheduling = alloc_solution_to_alloc_sched_solution(
        problem=sched_problem, solution=sol
    )
    solver.init_model(modelisation_allocation=ModelisationAllocationCP.BINARY)
    # copy meta constraints as we will update them later
    meta_constraints = [
        copy(mc)
        for mc in solver.meta_constraints
        if mc.metadata["type"]
        in {"allocated_task", "same_allocation", "clique_overlap"}
    ]
    hard = [
        copy(mc)
        for mc in solver.meta_constraints
        if mc.metadata["type"]
        not in {"allocated_task", "same_allocation", "clique_overlap"}
    ]
    done = False
    removed_meta = []
    while not done:
        print("Solving...")
        result_store = solver.solve()
        if solver.status_solver == StatusSolver.OPTIMAL:
            plot_allocation_solution(
                problem=allocation_problem, sol=result_store[-1][0], display=True
            )
            print(allocation_problem.evaluate(result_store[-1][0]))
        print(solver.status_solver)
        if solver.status_solver == StatusSolver.UNSATISFIABLE:
            meta_mus = solver.explain_unsat_meta(
                soft=meta_constraints, hard=hard, **kwargs_mus
            )
            if True:
                tasks_in_conflict = []
                for mc in meta_mus:
                    if mc.metadata["type"] == "allocated_task":
                        print(mc.metadata)
                        tasks_in_conflict.append(mc.metadata["task_index"])
                    if mc.metadata["type"] == "clique_overlap":
                        print(mc.metadata)
                        tasks_in_conflict.extend(list(mc.metadata["tasks_index"]))
                    print(mc.metadata)
                plotly_schedule_comparison(
                    base_solution=sched_scheduling,
                    updated_solution=sched_scheduling,
                    problem=sched_problem,
                    plot_team_breaks=True,
                    use_color_map_per_task=True,
                    color_map_per_task={i: "red" for i in tasks_in_conflict},
                    display=True,
                )
            print("The problem is unsatisfiable.")
            str_list_meta = "\n".join(
                f"{i_meta}: {meta.name}, {meta.metadata}"
                for i_meta, meta in enumerate(meta_mus)
            )
            i_meta = int(
                input(
                    "Choose a meta-constraint to remove among this minimal unsatisfiable subset:\n"
                    + str_list_meta
                    + "\n> "
                )
            )
            # remove meta-constraint:
            # - remove subconstraints from model
            # - remove subconstraints from other meta-constraints
            # - remove meta-constraint from list of meta-constraints
            meta = meta_mus[i_meta]
            subconstraints_ids = {id(c_) for c_ in meta.constraints}
            print("Before", len(solver.model.constraints))

            solver.model.constraints = []
            for m in meta_constraints:
                if m is meta:
                    continue
                solver.model.constraints.extend(m.constraints)
            for h in hard:
                solver.model.constraints.extend(h.constraints)
            print("After", len(solver.model.constraints))
            print("Before", len(meta_constraints))
            meta_constraints = [m for m in meta_constraints if m is not meta]
            print("After", len(meta_constraints))
            # for other_meta in meta_constraints:
            #    other_meta.constraints = [
            #        c for c in other_meta.constraints if id(c) not in subconstraints_ids
            #    ]
            removed_meta.append(meta)
        elif len(result_store) > 0:
            print(f"The problem was solved with status {solver.status_solver.value}.")
            if len(removed_meta) > 0:
                print(
                    f"We removed {len(removed_meta)} meta-constraint{'s' if len(removed_meta) > 0 else ''}: "
                    f"{[meta.name for meta in removed_meta]}"
                )
            break
        else:
            print(f"No solution found and status is {solver.status_solver}. Exiting.")
            break


def interactive_solving_mcs():
    kwargs_mcs = {"solver": "exact", "cpmpy_method": CpmpyCorrectUnsatMethod.mcs_grow}
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    allocation_problem.allocation_additional_constraint.nb_max_teams = 8
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem=allocation_problem)

    start_shift = datetime(day=1, month=7, year=2025).timestamp()
    sched_problem = build_scheduling_problem_from_allocation(
        problem=allocation_problem, horizon_start_shift=start_shift
    )
    sched_scheduling = alloc_solution_to_alloc_sched_solution(
        problem=sched_problem, solution=sol
    )
    solver.init_model(modelisation_allocation=ModelisationAllocationCP.BINARY)
    # copy meta constraints as we will update them later
    meta_constraints = [
        copy(mc)
        for mc in solver.meta_constraints
        if mc.metadata["type"] in {"allocated_task", "same_allocation"}
    ]
    hard = [
        copy(mc)
        for mc in solver.meta_constraints
        if mc.metadata["type"] not in {"allocated_task", "same_allocation"}
    ]
    done = False
    removed_meta = []
    while not done:
        print("Solving...")
        result_store = solver.solve()
        if solver.status_solver == StatusSolver.OPTIMAL:
            plot_allocation_solution(
                problem=allocation_problem, sol=result_store[-1][0], display=True
            )
            print(allocation_problem.evaluate(result_store[-1][0]))
        print(solver.status_solver)
        if solver.status_solver == StatusSolver.UNSATISFIABLE:
            meta_mus = solver.correct_unsat_meta(
                soft=meta_constraints, hard=hard, **kwargs_mcs
            )
            if True:
                tasks_in_conflict = []
                for mc in meta_mus:
                    if mc.metadata["type"] == "allocated_task":
                        print(mc.metadata)
                        tasks_in_conflict.append(mc.metadata["task_index"])
                    # if mc.metadata["type"] == "clique_overlap":
                    #     print(mc.metadata)
                    #     tasks_in_conflict.extend(list(mc.metadata["tasks_index"]))
                    print(mc.metadata)
                plotly_schedule_comparison(
                    base_solution=sched_scheduling,
                    updated_solution=sched_scheduling,
                    problem=sched_problem,
                    plot_team_breaks=True,
                    use_color_map_per_task=True,
                    color_map_per_task={i: "red" for i in tasks_in_conflict},
                    display=True,
                )
            print("The problem is unsatisfiable.")
            str_list_meta = "\n".join(
                f"{i_meta}: {meta.name}, {meta.metadata}"
                for i_meta, meta in enumerate(meta_mus)
            )
            i_meta = int(
                input(
                    "Choose a meta-constraint to remove among this minimal unsatisfiable subset:\n"
                    + str_list_meta
                    + "\n> "
                )
            )
            # remove meta-constraint:
            # - remove subconstraints from model
            # - remove subconstraints from other meta-constraints
            # - remove meta-constraint from list of meta-constraints
            meta = meta_mus[i_meta]
            subconstraints_ids = {id(c_) for c_ in meta.constraints}
            print("Before", len(solver.model.constraints))

            solver.model.constraints = []
            for m in meta_constraints:
                if m is meta:
                    continue
                solver.model.constraints.extend(m.constraints)
            for h in hard:
                solver.model.constraints.extend(h.constraints)
            print("After", len(solver.model.constraints))
            print("Before", len(meta_constraints))
            meta_constraints = [m for m in meta_constraints if m is not meta]
            print("After", len(meta_constraints))
            # for other_meta in meta_constraints:
            #    other_meta.constraints = [
            #        c for c in other_meta.constraints if id(c) not in subconstraints_ids
            #    ]
            removed_meta.append(meta)
        elif len(result_store) > 0:
            print(f"The problem was solved with status {solver.status_solver.value}.")
            if len(removed_meta) > 0:
                print(
                    f"We removed {len(removed_meta)} meta-constraint{'s' if len(removed_meta) > 0 else ''}: "
                    f"{[meta.name for meta in removed_meta]}"
                )
            break
        else:
            print(f"No solution found and status is {solver.status_solver}. Exiting.")
            break


def interactive_solving_with_interact_obj():
    kwargs_mus = {"solver": "exact", "cpmpy_method": CpmpyExplainUnsatMethod.mus}
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    scheduling_problem = parse_json_to_problem(instance)
    allocation_problem = build_allocation_problem_from_scheduling(
        problem=scheduling_problem
    )
    solver = CpsatTeamAllocationSolver(allocation_problem)
    solver.init_model(modelisation_allocation=ModelisationAllocationOrtools.BINARY)
    sol = solver.solve(time_limit=5).get_best_solution()
    # allocation_problem.allocation_additional_constraint.nb_max_teams = 8
    from discrete_optimization.workforce.generators.resource_scenario import (
        generate_allocation_disruption,
    )

    d = generate_allocation_disruption(
        original_allocation_problem=allocation_problem,
        original_solution=sol,
        params_randomness=ParamsRandomness(
            upper_nb_disruption=2, lower_nb_teams=1, upper_nb_teams=2
        ),
    )
    allocation_problem = d["new_allocation_problem"]
    sol = d["new_solution"]
    solver = CPMpyTeamAllocationSolverStoreConstraintInfo(problem=allocation_problem)
    start_shift = datetime(day=1, month=7, year=2025).timestamp()
    sched_problem = build_scheduling_problem_from_allocation(
        problem=allocation_problem, horizon_start_shift=start_shift
    )
    sched_scheduling = alloc_solution_to_alloc_sched_solution(
        problem=sched_problem, solution=sol
    )

    interact_solve = InteractSolve(problem=allocation_problem)
    solver = interact_solve.solver

    done = False
    removed_meta = []
    while not done:
        print("Solving...")
        status_solver, result_store = interact_solve.solve_current_problem()
        if status_solver == StatusSolver.OPTIMAL:
            plot_allocation_solution(
                problem=allocation_problem, sol=result_store[-1][0], display=True
            )
            print(allocation_problem.evaluate(result_store[-1][0]))
        print(status_solver)
        if status_solver == StatusSolver.UNSATISFIABLE:
            meta_mus = solver.explain_unsat_meta(
                soft=interact_solve.current_soft,
                hard=interact_solve.current_hard,
                **kwargs_mus,
            )
            if True:
                tasks_in_conflict = []
                for mc in meta_mus:
                    if mc.metadata["type"] == "allocated_task":
                        print(mc.metadata)
                        tasks_in_conflict.append(mc.metadata["task_index"])
                    if mc.metadata["type"] == "clique_overlap":
                        print(mc.metadata)
                        tasks_in_conflict.extend(list(mc.metadata["tasks_index"]))
                    print(mc.metadata)
                rel_sol = interact_solve.solve_relaxed_problem(base_solution=sol)
                upd_sol = alloc_solution_to_alloc_sched_solution(
                    problem=sched_problem, solution=rel_sol
                )
                color_map = {i: "red" for i in tasks_in_conflict}
                for t in interact_solve.dropped_tasks:
                    color_map[t] = "orange"
                plotly_schedule_comparison(
                    base_solution=sched_scheduling,
                    updated_solution=upd_sol,
                    problem=sched_problem,
                    plot_team_breaks=True,
                    use_color_map_per_task=True,
                    color_map_per_task=color_map,
                    display=True,
                )
            print("The problem is unsatisfiable.")
            str_list_meta = "\n".join(
                f"{i_meta}: {meta.name}, {meta.metadata}"
                for i_meta, meta in enumerate(meta_mus)
            )
            i_meta = int(
                input(
                    "Choose a meta-constraint to remove among this minimal unsatisfiable subset:\n"
                    + str_list_meta
                    + "\n> "
                )
            )
            # remove meta-constraint:
            # - remove subconstraints from model
            # - remove subconstraints from other meta-constraints
            # - remove meta-constraint from list of meta-constraints
            meta = meta_mus[i_meta]
            print(len(interact_solve.current_soft), " before (soft)")
            print(len(interact_solve.current_hard), " before (hard)")
            interact_solve.drop_constraints(constraints=[meta])
            print(len(interact_solve.current_soft), " after (soft)")
            print(len(interact_solve.current_hard), " after (hard)")

        elif len(result_store) > 0:
            print(f"The problem was solved with status {solver.status_solver.value}.")
            if len(removed_meta) > 0:
                print(
                    f"We removed {len(removed_meta)} meta-constraint{'s' if len(removed_meta) > 0 else ''}: "
                    f"{[meta.name for meta in removed_meta]}"
                )
            break
        else:
            print(f"No solution found and status is {solver.status_solver}. Exiting.")
            break


if __name__ == "__main__":
    interactive_solving_with_interact_obj()
