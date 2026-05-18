#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from copy import deepcopy
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from discrete_optimization.flex_scheduling.problem import (
    TASK_KEY,
    ConstraintsTask,
    FlexProblem,
    ScheduleSolution,
)
from discrete_optimization.flex_scheduling.solvers.cpsat import CpSatFlexSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import (
    ParamsObjectiveFunction,
    SolverDO,
    StatusSolver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class SequentialFlexSolver(SolverDO):
    def __init__(
        self,
        problem: FlexProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.original_problem = problem

    def solve(
        self, nb_batches: int = 5, time_limit_per_batch: float = 60.0, **kwargs
    ) -> ResultStorage:
        """
        Solves the problem iteratively by growing the problem scope batch by batch.

        Ensures that the final solution is re-ordered to match the original problem's task order.
        """

        # 1. Batching
        ordered_task_batches = self._create_task_batches(nb_batches)
        logger.info(f"Problem split into {len(ordered_task_batches)} atomic batches.")

        # 2. State
        accumulated_task_ids = []
        previous_solution: ScheduleSolution = None
        previous_sub_problem: FlexProblem = None

        # 3. Iterative Loop
        for i, batch_task_ids in enumerate(ordered_task_batches):
            logger.info(f"--- Iteration {i + 1}/{len(ordered_task_batches)} ---")

            # Grow the problem size
            accumulated_task_ids.extend(batch_task_ids)
            logger.info(f"Current Problem Size: {len(accumulated_task_ids)} tasks")

            # A. Create Sub-Problem (Includes all tasks so far)
            sub_problem = self._create_sub_problem(accumulated_task_ids)

            # B. Setup Solver
            solver = CpSatFlexSolver(sub_problem)
            solver.init_model(**kwargs)

            # C. Fix Previous Solution
            if previous_solution is not None:
                logger.info("Fixing variables for tasks from previous iterations...")
                self._apply_fixing_constraints(
                    solver, sub_problem, previous_solution, previous_sub_problem
                )

            # D. Solve
            solver.set_lexico_objective("makespan")

            params_cp = ParametersCp.default_cpsat()
            params_cp.nb_process = kwargs.get("nb_process", 16)

            res = solver.solve(
                time_limit=time_limit_per_batch,
                parameters_cp=params_cp,
                ortools_cpsat_solver_kwargs={"log_search_progress": True},
            )
            if (
                solver.status_solver == StatusSolver.UNKNOWN
                or solver.status_solver == StatusSolver.UNSATISFIABLE
            ):
                logger.warning("Solver returned UNKNOWN or UNSATISFIABLE status.")
            best_sol_storage = res.get_best_solution()

            if best_sol_storage is None:
                logger.error(f"Failed to solve iteration {i + 1}. Stopping sequence.")
                break

            # Update State for next iteration
            previous_solution = best_sol_storage
            previous_sub_problem = sub_problem

        # 4. Rebuild Final Solution in Original Order
        if previous_solution is None:
            logger.error("No solution found.")
            return ResultStorage(
                list_solution_fits=[],
                mode_optim=self.original_problem.objective_params.params_obj,
            )

        # Extract schedule map from the last (full or partial) sub-problem solution
        # The sub-problem tasks are likely ordered differently than original problem
        schedule_map = {}
        mode_map = {}

        for i in range(previous_sub_problem.nb_tasks):
            tid = previous_sub_problem.index_to_task_id[i]
            schedule_map[tid] = (
                previous_solution.schedule[i, 0],
                previous_solution.schedule[i, 1],
            )
            mode_map[tid] = previous_solution.modes[i]

        # Reorder to match original problem
        final_solution = self._rebuild_final_solution(schedule_map, mode_map)
        final_solution._intern_obj = previous_solution._intern_obj

        # Final Evaluation
        try:
            fitness = self.original_problem.evaluate(final_solution)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            fitness = float("inf")

        return ResultStorage(
            list_solution_fits=[(final_solution, fitness)],
            mode_optim=self.original_problem.objective_params.params_obj,
        )

    def _rebuild_final_solution(
        self,
        global_schedule: Dict[TASK_KEY, Tuple[int, int]],
        global_modes: Dict[TASK_KEY, int],
    ) -> ScheduleSolution:
        """
        Constructs a ScheduleSolution where arrays are indexed according to self.original_problem.
        """
        nb_tasks = self.original_problem.nb_tasks
        schedule = -np.ones((nb_tasks, 2), dtype=int)
        modes = np.zeros(nb_tasks, dtype=int)

        for i in range(nb_tasks):
            t_id = self.original_problem.index_to_task_id[i]
            if t_id in global_schedule:
                schedule[i, 0] = global_schedule[t_id][0]
                schedule[i, 1] = global_schedule[t_id][1]
                modes[i] = global_modes[t_id]
            else:
                # This might happen if the loop broke early or tasks were missing
                logger.warning(f"Task {t_id} was not scheduled in the final result.")

        return ScheduleSolution(self.original_problem, schedule, modes)

    def _apply_fixing_constraints(
        self,
        current_solver: CpSatFlexSolver,
        current_problem: FlexProblem,
        prev_sol: ScheduleSolution,
        prev_problem: FlexProblem,
    ):
        cp_model = current_solver.cp_model

        for prev_idx, task in enumerate(prev_problem.tasks):
            tid = task.id
            if tid not in current_problem.task_id_to_index:
                continue

            curr_idx = current_problem.task_id_to_index[tid]

            prev_start = prev_sol.schedule[prev_idx, 0]
            prev_end = prev_sol.schedule[prev_idx, 1]
            prev_mode = prev_sol.modes[prev_idx]

            # Fix Start/End/Mode
            cp_model.Add(
                current_solver.variables["starts"][curr_idx] == int(prev_start)
            )
            cp_model.Add(current_solver.variables["ends"][curr_idx] == int(prev_end))

            if curr_idx in current_solver.variables["is_present"]:
                mode_vars = current_solver.variables["is_present"][curr_idx]
                if prev_mode in mode_vars:
                    cp_model.Add(mode_vars[prev_mode] == 1)

    def _create_sub_problem(self, task_ids_subset: List[TASK_KEY]) -> FlexProblem:
        task_set = set(task_ids_subset)

        # Tasks
        tasks = []
        for t in self.original_problem.tasks:
            if t.id in task_set:
                tasks.append(deepcopy(t))

        # Groups (Partial)
        groups = []
        group_ids_subset = set()
        for grp in self.original_problem.tasks_group:
            common = set(grp.tasks_group).intersection(task_set)
            if common:
                new_grp = deepcopy(grp)
                new_grp.tasks_group = common
                groups.append(new_grp)
                group_ids_subset.add(grp.id)

        # Constraints & Objectives
        new_constraints = self._filter_constraints(
            self.original_problem.constraints, task_set, group_ids_subset
        )
        new_objectives = self._filter_objectives(
            self.original_problem.objective_params, task_set, group_ids_subset
        )

        return FlexProblem(
            resources=self.original_problem.resources,
            tasks=tasks,
            tasks_group=groups,
            constraints=new_constraints,
            objective_params=new_objectives,
            horizon=self.original_problem.horizon,
        )

    def _filter_constraints(self, original, tasks, groups):
        nc = ConstraintsTask(
            successors={},
            start_at_start=[],
            start_at_end=[],
            start_at_start_plus_offset=[],
            start_at_end_plus_offset=[],
            start_after_start_plus_offset=[],
            start_after_end_plus_offset=[],
        )

        if original.successors:
            for t1, succs in original.successors.items():
                if t1 in tasks:
                    valid = {t2 for t2 in succs if t2 in tasks}
                    if valid:
                        nc.successors[t1] = valid

        # General Time Constraints
        for attr in ["start_at_start", "start_at_end"]:
            original_list = getattr(original, attr)
            if original_list:
                filtered = [
                    pair
                    for pair in original_list
                    if pair[0] in tasks and pair[1] in tasks
                ]
                setattr(nc, attr, filtered)

        for attr in [
            "start_at_start_plus_offset",
            "start_at_end_plus_offset",
            "start_after_start_plus_offset",
            "start_after_end_plus_offset",
        ]:
            original_list = getattr(original, attr)
            if original_list:
                filtered = [
                    triple
                    for triple in original_list
                    if triple[0] in tasks and triple[1] in tasks
                ]
                setattr(nc, attr, filtered)

        if original.successor_with_res_release_at_start_of_successor:
            nc.successor_with_res_release_at_start_of_successor = [
                x
                for x in original.successor_with_res_release_at_start_of_successor
                if x[0] in tasks and x[1] in tasks
            ]

        if original.successor_generic_with_res_release_at_start_of_successor_generic:
            filtered = []
            for item in original.successor_generic_with_res_release_at_start_of_successor_generic:
                src, tgt, _ = item

                def is_in(abs_item):
                    if abs_item.is_a_task:
                        tid = abs_item.task_id
                        # Check against ID map if needed, assuming task_id is the key
                        return tid in tasks or (
                            self.original_problem.index_to_task_id.get(tid) in tasks
                        )
                    else:
                        return abs_item.group_id in groups

                if is_in(src) and is_in(tgt):
                    filtered.append(item)
            nc.successor_generic_with_res_release_at_start_of_successor_generic = (
                filtered
            )
        return nc

    def _filter_objectives(self, params, tasks, groups):
        new_p = deepcopy(params)
        for obj in new_p.params_obj.values():
            if hasattr(obj, "weight_per_task"):
                obj.weight_per_task = {
                    t: w for t, w in obj.weight_per_task.items() if t in tasks
                }
            if hasattr(obj, "weights_per_group_task"):
                obj.weights_per_group_task = {
                    g: w for g, w in obj.weights_per_group_task.items() if g in groups
                }
            if hasattr(obj, "weight_per_groups"):
                obj.weight_per_groups = {
                    g: w for g, w in obj.weight_per_groups.items() if g in groups
                }
        return new_p

    def _create_task_batches(self, nb_batches: int) -> List[List[TASK_KEY]]:
        # Build blocking graph
        blocking_graph = nx.Graph()
        blocking_graph.add_nodes_from(self.original_problem.tasks_ids)

        for grp in self.original_problem.tasks_group:
            if grp.res_not_released and len(grp.tasks_group) > 1:
                tasks = list(grp.tasks_group)
                for k in range(len(tasks) - 1):
                    blocking_graph.add_edge(tasks[k], tasks[k + 1])

        if self.original_problem.constraints.successor_with_res_release_at_start_of_successor:
            for (
                t1,
                t2,
                _,
            ) in self.original_problem.constraints.successor_with_res_release_at_start_of_successor:
                blocking_graph.add_edge(t1, t2)

        def get_ids(abstraction):
            if abstraction.is_a_task:
                tid = abstraction.task_id
                if tid in self.original_problem.task_id_dict:
                    return [tid]
                if tid in self.original_problem.index_to_task_id:
                    return [self.original_problem.index_to_task_id[tid]]
                return [tid]
            else:
                gid = abstraction.group_id
                if gid in self.original_problem.group_id_to_index:
                    return list(
                        self.original_problem.tasks_group[
                            self.original_problem.group_id_to_index[gid]
                        ].tasks_group
                    )
                return []

        if self.original_problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic:
            for (
                src,
                tgt,
                _,
            ) in self.original_problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic:
                for s in get_ids(src):
                    for t in get_ids(tgt):
                        blocking_graph.add_edge(s, t)

        # Add edges for Start-Time Constraints (Must be in same atom/batch if strongly coupled)
        # For 'equal' constraints, they should definitely be together.
        # For 'offset' constraints, maybe less critical, but safe to group.

        strong_constraints = [
            self.original_problem.constraints.start_at_start,
            self.original_problem.constraints.start_at_end,
            self.original_problem.constraints.start_at_start_plus_offset,
            self.original_problem.constraints.start_at_end_plus_offset,
        ]

        for constr_list in strong_constraints:
            if constr_list:
                for item in constr_list:
                    t1, t2 = item[0], item[1]
                    blocking_graph.add_edge(t1, t2)

        atoms = [list(c) for c in nx.connected_components(blocking_graph)]
        task_to_atom = {}
        for idx, atom in enumerate(atoms):
            for tid in atom:
                task_to_atom[tid] = idx

        atom_graph = nx.DiGraph()
        atom_graph.add_nodes_from(range(len(atoms)))

        if self.original_problem.constraints.successors:
            for t1, succs in self.original_problem.constraints.successors.items():
                if t1 not in task_to_atom:
                    continue
                a1 = task_to_atom[t1]
                for t2 in succs:
                    if t2 not in task_to_atom:
                        continue
                    a2 = task_to_atom[t2]
                    if a1 != a2:
                        atom_graph.add_edge(a1, a2)

        # Add Precedence Edges for other constraint types (Start After...)
        weak_constraints = [
            self.original_problem.constraints.start_after_start_plus_offset,
            self.original_problem.constraints.start_after_end_plus_offset,
        ]
        for constr_list in weak_constraints:
            if constr_list:
                for item in constr_list:
                    t1, t2 = item[0], item[1]
                    if t1 in task_to_atom and t2 in task_to_atom:
                        a1, a2 = task_to_atom[t1], task_to_atom[t2]
                        if a1 != a2:
                            atom_graph.add_edge(
                                a2, a1
                            )  # t1 >= t2 + offset => t2 precedes t1

        if not nx.is_directed_acyclic_graph(atom_graph):
            sorted_atoms = list(range(len(atoms)))
        else:
            sorted_atoms = list(nx.topological_sort(atom_graph))

        total = self.original_problem.nb_tasks
        if nb_batches is None:
            nb_batches = 1
        target = total / nb_batches
        batches = []
        cur_batch = []
        cur_size = 0

        for a_idx in sorted_atoms:
            atom = atoms[a_idx]
            if cur_batch and (cur_size + len(atom) > target * 1.2):
                batches.append(cur_batch)
                cur_batch = []
                cur_size = 0
            cur_batch.extend(atom)
            cur_size += len(atom)
        if cur_batch:
            batches.append(cur_batch)
        return batches
