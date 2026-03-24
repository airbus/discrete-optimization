#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcalbp_l.problem import RCALBPLProblem, RCALBPLSolution
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver

# Assuming CpSatRCALBPLSolver is defined in the same module or imported:
# from discrete_optimization.rcalbp_l.cpsat import CpSatRCALBPLSolver

logger = logging.getLogger(__name__)


class SequentialRCALBPLSolver(SolverDO):
    """
    Forward Sequential Solver.
    Solves the problem iteratively by chunking the periods from 0 to N.

    Features:
    - Extrapolates solutions from period p to p+1 for perfect warm-starting (thanks to the learning effect).
    - Early Stopping: Once the target cycle time (c_target) is reached, the schedule is frozen
      and extrapolated to the end of the timeline instantly.
    """

    problem: RCALBPLProblem

    def __init__(
        self,
        problem: RCALBPLProblem,
        chunk_size: int = 5,
        time_limit_per_iteration: int = 60,
        stop_early: bool = True,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, **kwargs)
        self.chunk_size = chunk_size
        self.time_limit_per_iteration = time_limit_per_iteration
        self.stop_early = stop_early

    def _build_subproblem(self, p_start: int, p_end: int) -> RCALBPLProblem:
        """Creates a clone of the problem, restricted to a specific window of periods."""
        return RCALBPLProblem(
            c_target=self.problem.c_target,
            c_max=self.problem.c_max,
            nb_stations=self.problem.nb_stations,
            nb_periods=self.problem.nb_periods,  # Keep total periods intact
            nb_tasks=self.problem.nb_tasks,
            precedences=self.problem.precedences,
            durations=self.problem.durations,
            nb_resources=self.problem.nb_resources,
            capa_resources=self.problem.capa_resources,
            cons_resources=self.problem.cons_resources,
            nb_zones=self.problem.nb_zones,
            capa_zones=self.problem.capa_zones,
            cons_zones=self.problem.cons_zones,
            neutr_zones=self.problem.neutr_zones,
            p_start=p_start,
            p_end=p_end,
        )

    def _extrapolate_warm_start(
        self, prev_sol: RCALBPLSolution, target_periods: int
    ) -> RCALBPLSolution:
        """
        Takes a solution from a smaller subproblem and projects it forward.
        Because tasks shrink due to the learning curve, copying the start times
        from the last solved period to future periods is mathematically guaranteed
        to remain a valid schedule!
        """
        wks = dict(prev_sol.wks)
        raw = dict(prev_sol.raw)
        start = dict(prev_sol.start)
        cyc = dict(prev_sol.cyc)

        last_p = max(prev_sol.cyc.keys())

        # Extrapolate the schedule to the target periods
        for p in range(last_p + 1, target_periods):
            cyc[p] = cyc[last_p]
            for t in self.problem.tasks:
                if (t, last_p) in start:
                    start[(t, p)] = start[(t, last_p)]

        # Create a new dummy problem to hold the extrapolated solution bounds
        sub_prob = self._build_subproblem(p_start=0, p_end=target_periods)
        return RCALBPLSolution(sub_prob, wks, raw, start, cyc)

    def solve(self, **kwargs: Any) -> ResultStorage:
        current_p = max(
            min(self.chunk_size, self.problem.nb_periods), self.problem.nb_stations
        )
        prev_sol: Optional[RCALBPLSolution] = None

        while current_p <= self.problem.nb_periods:
            logger.info(
                f"Forward: Solving up to period {current_p} / {self.problem.nb_periods}"
            )

            # 1. Create problem truncated to current_p
            sub_prob = self._build_subproblem(p_start=0, p_end=current_p)
            sub_solver = CpSatRCALBPLSolver(problem=sub_prob)
            sub_solver.init_model()

            # 2. Apply Warm Start
            if prev_sol is not None:
                extrapolated_ws = self._extrapolate_warm_start(prev_sol, current_p)
                sub_solver.set_warm_start(extrapolated_ws)
                logger.info("Warm-start extrapolated and applied.")

            # 3. Solve the chunk
            res = sub_solver.solve(time_limit=self.time_limit_per_iteration, **kwargs)
            if len(res) == 0:
                logger.error(
                    f"Solver failed to find a solution for period subset {current_p}. Aborting."
                )
                break

            prev_sol = res.get_best_solution_fit()[0]
            current_cyc = prev_sol.cyc[current_p - 1]
            logger.info(
                f"Subproblem {current_p} solved. Current max cycle time: {current_cyc}"
            )

            # 4. Early Stopping condition
            if self.stop_early and current_cyc <= self.problem.c_target:
                logger.info(
                    f"Target cycle time reached at period {current_p - 1}! Extrapolating to the end and stopping early."
                )
                final_sol = self._extrapolate_warm_start(
                    prev_sol, self.problem.nb_periods
                )
                final_sol.change_problem(self.problem)
                return self.create_result_storage(
                    [(final_sol, self.aggreg_from_sol(final_sol))]
                )

            if current_p == self.problem.nb_periods:
                break

            # Increment chunk
            current_p = min(current_p + self.chunk_size, self.problem.nb_periods)

        # Re-wrap the final solution back into the original full problem context
        if prev_sol is not None:
            prev_sol.change_problem(self.problem)
            return self.create_result_storage(
                [(prev_sol, self.aggreg_from_sol(prev_sol))]
            )

        return self.create_result_storage([])


class BackwardSequentialRCALBPLSolver(SolverDO):
    """
    Independent Chunk Backward Reasoning Solver with SGS Warm-Starting.
    """

    problem: RCALBPLProblem

    def __init__(
        self,
        problem: RCALBPLProblem,
        future_chunk_size: int = 5,
        phase2_chunk_size: int = 10,
        time_limit_phase1: int = 120,
        time_limit_phase2: int = 30,
        use_sgs_warm_start: bool = True,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, **kwargs)
        self.future_chunk_size = future_chunk_size
        self.phase2_chunk_size = phase2_chunk_size
        self.time_limit_phase1 = time_limit_phase1
        self.time_limit_phase2 = time_limit_phase2
        self.use_sgs_warm_start = use_sgs_warm_start

    def _build_subproblem(self, p_start: int, p_end: int) -> RCALBPLProblem:
        return RCALBPLProblem(
            c_target=self.problem.c_target,
            c_max=self.problem.c_max,
            nb_stations=self.problem.nb_stations,
            nb_periods=self.problem.nb_periods,
            nb_tasks=self.problem.nb_tasks,
            precedences=self.problem.precedences,
            durations=self.problem.durations,
            nb_resources=self.problem.nb_resources,
            capa_resources=self.problem.capa_resources,
            cons_resources=self.problem.cons_resources,
            nb_zones=self.problem.nb_zones,
            capa_zones=self.problem.capa_zones,
            cons_zones=self.problem.cons_zones,
            neutr_zones=self.problem.neutr_zones,
            p_start=p_start,
            p_end=p_end,
        )

    def _generate_sgs_warm_start(
        self, base_sol: RCALBPLSolution, target_p_start: int, target_p_end: int
    ) -> RCALBPLSolution:
        """
        Uses the Serial Generation Scheme to generate a mathematically valid
        warm start for earlier periods by replaying the optimal future task sequence.
        """
        wks = dict(base_sol.wks)
        raw = dict(base_sol.raw)
        start = {}
        cyc = {}

        # 1. Extract the optimal topological task order from the earliest known solved period
        earliest_p_solved = min(base_sol.cyc.keys())
        target_starts = {
            t: base_sol.start.get((t, earliest_p_solved), 0) for t in self.problem.tasks
        }

        # 2. To strictly enforce cycle time monotonicity backwards, track the lower bound
        min_allowed_cyc = base_sol.cyc[earliest_p_solved]

        # 3. Generate schedules stepping backwards into the target chunk
        for p in range(target_p_end - 1, target_p_start - 1, -1):
            start_p, sgs_cyc = self.problem.build_sgs_schedule_for_period(
                wks=wks, raw=raw, target_starts=target_starts, period=p
            )

            for t in self.problem.tasks:
                start[(t, p)] = start_p[t]

            # Clamp the cycle time to ensure it satisfies monotonicity constraints backwards
            actual_cyc = max(sgs_cyc, min_allowed_cyc)
            cyc[p] = actual_cyc
            min_allowed_cyc = actual_cyc  # Feed constraint to the next earlier period

        sub_prob = self._build_subproblem(p_start=target_p_start, p_end=target_p_end)
        return RCALBPLSolution(sub_prob, wks, raw, start, cyc)

    def solve(self, **kwargs: Any) -> ResultStorage:
        # --- PHASE 1: Find optimal steady-state allocation ---
        p_end = self.problem.nb_periods
        current_p_start = max(
            self.problem.nb_stations, self.problem.nb_periods - self.future_chunk_size
        )

        logger.info(
            f"Backward Phase 1: Solving future window [{current_p_start}, {p_end})"
        )

        future_prob = self._build_subproblem(p_start=current_p_start, p_end=p_end)
        future_solver = CpSatRCALBPLSolver(problem=future_prob)
        future_solver.init_model(
            minimize_used_cycle_time=True, add_heuristic_constraint=False
        )  # Squeeze layout tightly
        res1 = future_solver.solve(time_limit=self.time_limit_phase1, **kwargs)

        if len(res1) == 0:
            logger.error("Phase 1 failed. Aborting.")
            return self.create_result_storage([])
        future_sol: RCALBPLSolution = res1[-1][0]
        optimal_wks = future_sol.wks
        optimal_raw = future_sol.raw

        # Memory storage for final reconstructed solution
        merged_start = dict(future_sol.start)
        merged_cyc = dict(future_sol.cyc)
        latest_chunk_sol = future_sol

        # --- PHASE 2: Solve independent chunks backwards ---
        current_p_end = current_p_start

        logger.info(
            "Backward Phase 2: Solving independent chunks backwards with locked layout."
        )

        while current_p_end > 0:
            current_p_start = max(0, current_p_end - self.phase2_chunk_size)

            # Keep unstable periods (0 to W-1) strictly together in the final chunk
            if current_p_start < self.problem.nb_stations:
                current_p_start = 0

            logger.info(
                f"Phase 2: Solving independent chunk [{current_p_start}, {current_p_end})"
            )

            sub_prob = self._build_subproblem(
                p_start=current_p_start, p_end=current_p_end
            )
            sub_solver = CpSatRCALBPLSolver(problem=sub_prob)
            sub_solver.init_model(minimize_used_cycle_time=False)

            # 1. Lock the layout
            if hasattr(sub_solver, "fix_allocations_and_resources"):
                sub_solver.fix_allocations_and_resources(optimal_wks, optimal_raw)

            # 2. Chain Cycle Times backward
            next_chunk_first_cyc = merged_cyc[current_p_end]
            if hasattr(sub_solver, "add_cycle_time_lower_bound"):
                sub_solver.add_cycle_time_lower_bound(
                    current_p_end - 1, next_chunk_first_cyc
                )

            # 3. Inject SGS Warm Start!
            if self.use_sgs_warm_start:
                sgs_ws = self._generate_sgs_warm_start(
                    latest_chunk_sol, current_p_start, current_p_end
                )
                sub_solver.set_warm_start(sgs_ws)
                logger.info("SGS Warm Start successfully applied!")

            res2 = sub_solver.solve(time_limit=self.time_limit_phase2, **kwargs)

            if len(res2) == 0:
                logger.error(
                    f"Phase 2 failed at chunk [{current_p_start}, {current_p_end})."
                )
                break

            latest_chunk_sol: RCALBPLSolution = res2[-1][0]

            # Merge results
            merged_start.update(latest_chunk_sol.start)
            merged_cyc.update(latest_chunk_sol.cyc)
            current_p_end = current_p_start

        # --- FINALIZATION ---
        final_sol = RCALBPLSolution(
            problem=self.problem,
            wks=optimal_wks,
            raw=optimal_raw,
            start=merged_start,
            cyc=merged_cyc,
        )
        return self.create_result_storage(
            [(final_sol, self.aggreg_from_sol(final_sol))]
        )
