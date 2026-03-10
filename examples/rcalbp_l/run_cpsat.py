import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcalbp_l.problem import (
    RCALBPLProblem,
    RCALBPLSolution,
    parse_rcalbpl_json,
)
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver
from discrete_optimization.rcalbp_l.solvers.sequential_solver import (
    BackwardSequentialRCALBPLSolver,
)
from examples.rcalbp_l.plot import plot_rcalbpl_dashboard

logging.basicConfig(level=logging.INFO)


def main():
    problem = parse_rcalbpl_json("instances/187_2_26_2880.json")
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    solver = CpSatRCALBPLSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(
        time_limit=300,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        parameters_cp=p,
    )
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


def main_sequential():
    problem = parse_rcalbpl_json("instances/795_2_26_2880.json")
    # problem.nb_periods = 5
    # problem.periods = range(problem.nb_periods)
    from discrete_optimization.generic_tools.sequential_metasolver import (
        SequentialMetasolver,
        SubBrick,
    )

    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    brick1 = SubBrick(
        BackwardSequentialRCALBPLSolver,
        kwargs=dict(
            future_chunk_size=1,
            phase2_chunk_size=5,
            time_limit_phase1=200,
            time_limit_phase2=200,
            use_sgs_warm_start=True,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        ),
    )
    brick2 = SubBrick(
        CpSatRCALBPLSolver,
        dict(
            add_heuristic_constraint=False,
            parameters_cp=p,
            ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
            time_limit=200,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    fig, slider = plot_rcalbpl_dashboard(problem, sol)
    print(problem.evaluate(sol), problem.satisfy(sol))


def _build_subproblem(
    problem: RCALBPLProblem, p_start: int, p_end: int
) -> RCALBPLProblem:
    """Helper to clone the problem restricted to a specific window of periods."""
    return RCALBPLProblem(
        c_target=problem.c_target,
        c_max=problem.c_max,
        nb_stations=problem.nb_stations,
        nb_periods=problem.nb_periods,
        nb_tasks=problem.nb_tasks,
        precedences=problem.precedences,
        durations=problem.durations,
        nb_resources=problem.nb_resources,
        capa_resources=problem.capa_resources,
        cons_resources=problem.cons_resources,
        nb_zones=problem.nb_zones,
        capa_zones=problem.capa_zones,
        cons_zones=problem.cons_zones,
        neutr_zones=problem.neutr_zones,
        p_start=p_start,
        p_end=p_end,
    )


def main_test_sgs():
    """
    Test script to verify the Serial Generation Scheme (SGS).
    Solves only the last period, extracts the layout and sequence,
    and rebuilds the full timeline purely via SGS logic.
    """
    # Load the problem
    problem = parse_rcalbpl_json("instances/795_4_50_1440.json")

    # 1. Build a subproblem for ONLY the last period
    last_p = problem.nb_periods - 1
    sub_prob = _build_subproblem(problem, p_start=last_p, p_end=problem.nb_periods)

    # 2. Solve this last period optimally to get a good packing/order
    logging.info(f"Solving ONLY the last period ({last_p})...")
    solver = CpSatRCALBPLSolver(sub_prob)

    # Use minimize_used_cycle_time=True to pack it as tightly as possible
    solver.init_model(minimize_used_cycle_time=True, add_heuristic_constraint=False)

    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = solver.solve(
        time_limit=20,
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )

    if len(res) == 0:
        logging.error("Failed to find a solution for the last period!")
        return

    sol_last: RCALBPLSolution = res[-1][0]
    logging.info(
        f"Solution found for the last period. Max cycle time: {sol_last.cyc[last_p]}"
    )
    # 3. Extract wks, raw, and task order
    wks = sol_last.wks
    raw = sol_last.raw
    # Sort tasks by their start time in the solved period
    target_starts = {t: sol_last.start.get((t, last_p), 0) for t in problem.tasks}
    # 4. Rebuild the schedule for all periods using SGS
    logging.info("Rebuilding full schedule using Serial Generation Scheme (SGS)...")
    full_start = {}
    used_cyc = {}
    for p_idx in problem.periods:
        print("Running sgs..", p_idx)
        start_p, cyc_p = problem.build_sgs_schedule_for_period(
            wks, raw, target_starts, p_idx
        )
        used_cyc[p_idx] = cyc_p
        for t in problem.tasks:
            full_start[(t, p_idx)] = start_p[t]
    # 5. Fix the chosen cycle times to respect monotonicity and unstable period rules
    chosen_cyc = {}
    min_allowed = problem.c_target

    # Go backwards for stable periods (ensure cyc[p] >= cyc[p+1])
    for p_idx in range(problem.nb_periods - 1, problem.nb_stations - 1, -1):
        chosen_cyc[p_idx] = max(min_allowed, used_cyc[p_idx])
        min_allowed = chosen_cyc[p_idx]  # Next earlier period must be at least this big

    # Unstable periods (0 to W-1) must share the same cycle time
    if problem.nb_stations > 0:
        max_unstable = max([used_cyc[p_idx] for p_idx in range(problem.nb_stations)])
        if problem.nb_periods > problem.nb_stations:
            # Must also be >= the first stable period
            max_unstable = max(max_unstable, chosen_cyc[problem.nb_stations])
        for p_idx in range(problem.nb_stations):
            chosen_cyc[p_idx] = max_unstable

    # 6. Assemble the full solution
    full_sol = RCALBPLSolution(
        problem=problem, wks=wks, raw=raw, start=full_start, cyc=chosen_cyc
    )

    # 7. Evaluate and verify validity
    evals = problem.evaluate(full_sol)
    is_valid = problem.satisfy(full_sol)

    logging.info(f"SGS Reconstructed Solution Evaluation: {evals}")
    print(is_valid)
    print(evals)
    if is_valid:
        logging.info(
            "SUCCESS! The SGS successfully reconstructed a 100% valid schedule."
        )
    else:
        logging.error(
            "FAILED. The SGS generated an invalid schedule. Check capacity/precedence logic."
        )

    # Plot the result
    fig, slider = plot_rcalbpl_dashboard(problem, full_sol)


if __name__ == "__main__":
    main_sequential()
