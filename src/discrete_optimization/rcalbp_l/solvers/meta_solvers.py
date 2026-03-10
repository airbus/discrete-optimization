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

logger = logging.getLogger(__name__)


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
        wks, raw, start, cyc = dict(base_sol.wks), dict(base_sol.raw), {}, {}
        earliest_p_solved = min(base_sol.cyc.keys())
        target_starts = {
            t: base_sol.start.get((t, earliest_p_solved), 0) for t in self.problem.tasks
        }
        min_allowed_cyc = base_sol.cyc[earliest_p_solved]
        for p in range(target_p_end - 1, target_p_start - 1, -1):
            start_p, sgs_cyc = self.problem.build_sgs_schedule_for_period(
                wks=wks, raw=raw, target_starts=target_starts, period=p
            )
            for t in self.problem.tasks:
                start[(t, p)] = start_p[t]
            actual_cyc = max(sgs_cyc, min_allowed_cyc)
            cyc[p] = actual_cyc
            min_allowed_cyc = actual_cyc
        sub_prob = self._build_subproblem(p_start=target_p_start, p_end=target_p_end)
        return RCALBPLSolution(sub_prob, wks, raw, start, cyc)

    def _run_phase_1(self, **kwargs: Any) -> Optional[RCALBPLSolution]:
        """Original Phase 1: Solves a contiguous future chunk."""
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
        )

        res = future_solver.solve(time_limit=self.time_limit_phase1, **kwargs)
        if len(res) == 0:
            logger.error("Phase 1 failed. Aborting.")
            return None
        return res[-1][0]

    def _run_phase_2(
        self, phase1_sol: RCALBPLSolution, current_p_end: int = None, **kwargs: Any
    ) -> ResultStorage:
        """Phase 2: Locks the allocation from Phase 1 and solves chunks backwards."""
        optimal_wks, optimal_raw = phase1_sol.wks, phase1_sol.raw
        merged_start, merged_cyc = dict(phase1_sol.start), dict(phase1_sol.cyc)
        latest_chunk_sol = phase1_sol
        if current_p_end is None:
            current_p_end = min(phase1_sol.cyc.keys())
        logger.info(
            "Backward Phase 2: Solving independent chunks backwards with locked layout."
        )

        while current_p_end > 0:
            current_p_start = max(0, current_p_end - self.phase2_chunk_size)
            print(current_p_start)
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

            if hasattr(sub_solver, "fix_allocations_and_resources"):
                sub_solver.fix_allocations_and_resources(optimal_wks, optimal_raw)

            if hasattr(sub_solver, "add_cycle_time_lower_bound"):
                sub_solver.add_cycle_time_lower_bound(
                    current_p_end - 1, merged_cyc[current_p_end]
                )

            if self.use_sgs_warm_start:
                sgs_ws = self._generate_sgs_warm_start(
                    latest_chunk_sol, current_p_start, current_p_end
                )
                sub_solver.set_warm_start(sgs_ws)

            res2 = sub_solver.solve(time_limit=self.time_limit_phase2, **kwargs)

            if len(res2) == 0:
                logger.error(
                    f"Phase 2 failed at chunk [{current_p_start}, {current_p_end})."
                )
                break

            latest_chunk_sol = res2[-1][0]
            merged_start.update(latest_chunk_sol.start)
            merged_cyc.update(latest_chunk_sol.cyc)
            current_p_end = current_p_start

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

    def solve(self, **kwargs: Any) -> ResultStorage:
        phase1_sol = self._run_phase_1(**kwargs)
        if phase1_sol is None:
            return self.create_result_storage([])
        return self._run_phase_2(phase1_sol, **kwargs)


class BackwardSequentialRCALBPLSolverSGS(BackwardSequentialRCALBPLSolver):
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

    def _run_phase_2(
        self, phase1_sol: RCALBPLSolution, current_p_end: int = None, **kwargs: Any
    ) -> ResultStorage:
        """Phase 2: Locks the allocation from Phase 1 and solves chunks backwards."""
        merged_start, merged_cyc = dict(phase1_sol.start), dict(phase1_sol.cyc)
        latest_chunk_sol = phase1_sol
        if current_p_end is None:
            current_p_end = min(phase1_sol.cyc.keys())
        logger.info(
            "Backward Phase 2: Solving independent chunks backwards with locked layout."
        )
        sgs_sol = self._generate_sgs_warm_start(latest_chunk_sol, 0, current_p_end)
        sgs_sol.start.update(merged_start)
        sgs_sol.cyc.update(merged_cyc)
        return self.create_result_storage([(sgs_sol, self.aggreg_from_sol(sgs_sol))])


class BalancedBackwardSequentialRCALBPLSolver(BackwardSequentialRCALBPLSolver):
    """
    Upgraded Backward Solver using the 'Two-Period Extremes' heuristic.
    Phase 1 isolates ONLY the first unstable period and the final steady-state period.
    """

    def _run_phase_1(self, **kwargs: Any) -> Optional[RCALBPLSolution]:
        p_first = max(0, self.problem.nb_stations - 1)
        p_last = self.problem.nb_periods - 1
        logger.info(
            f"Balanced Phase 1: Co-optimizing extremes (Period {p_first} and Period {p_last})"
        )
        phase1_prob = self._build_subproblem(p_start=0, p_end=self.problem.nb_periods)
        phase1_prob.periods = list(range(self.problem.nb_stations)) + [p_last]
        phase1_solver = CpSatRCALBPLSolver(problem=phase1_prob)
        phase1_solver.init_model(
            minimize_used_cycle_time=True, add_heuristic_constraint=False
        )
        res = phase1_solver.solve(time_limit=self.time_limit_phase1, **kwargs)
        if len(res) == 0:
            logger.error("Phase 1 failed. Aborting.")
            return None
        sol: RCALBPLSolution = res[-1][0]
        sol.cyc = self.problem.compute_actual_cycle_time_per_period(sol)
        sol.cyc = {p_last: max(self.problem.c_target, sol.cyc[p_last])}
        sol.start = {(t, p_last): sol.start[(t, p_last)] for t in self.problem.tasks}
        logger.info(f"Balanced layout locked! Final Cyc: {sol.cyc.get(p_last, 'N/A')}")
        return sol

    def solve(self, **kwargs: Any) -> ResultStorage:
        phase1_sol = self._run_phase_1(**kwargs)
        if phase1_sol is None:
            return self.create_result_storage([])
        return self._run_phase_2(
            phase1_sol, current_p_end=self.problem.nb_periods - 1, **kwargs
        )
