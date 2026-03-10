import logging
import os
import pickle
import time

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.pareto_tools import CpsatParetoSolver
from discrete_optimization.rcalbp_l.problem import parse_rcalbpl_json
from discrete_optimization.rcalbp_l.solvers.cpsat import CpSatRCALBPLSolver
from discrete_optimization.rcalbp_l.solvers.pareto_postprocess import (
    DpRCALBPLPostProSolver,
    RampUpParetoSolverPostpro,
)
from discrete_optimization.rcalbp_l.solvers.sequential_solver import (
    BackwardSequentialRCALBPLSolver,
)

logging.basicConfig(level=logging.INFO)


def main_pareto(instance="187_2_26_2880.json"):
    problem = parse_rcalbpl_json("instances/" + instance)
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
            time_limit_phase2=50,
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
            time_limit=500,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    res_dict = {"instance": instance, "sol": sol}
    pickle.dump(
        res_dict, open(f"sol_{instance}_cp_more_{time.process_time()}.pkl", "wb")
    )

    postpro_solver = RampUpParetoSolverPostpro(problem=problem)
    postpro_solver.init_model(from_solution=sol)
    pareto_solver = CpsatParetoSolver(
        problem=problem,
        solver=postpro_solver,
        objective_names=["change_cost", "ramp_up_cost"],
        dict_function={
            "change_cost": lambda sol: problem.evaluate(sol)["nb_adjustments"],
            "ramp_up_cost": lambda sol: problem.evaluate(sol)["ramp_up_duration"],
        },
        delta_ref_improvement=[0, 0],
        delta_abs_improvement=[1, 1],
    )
    front = pareto_solver.solve(
        obj_vars=[
            postpro_solver.variables["objectives"][c]
            for c in pareto_solver.objective_names
        ],
        time_limit=100,
        subsolver_kwargs={
            "time_limit": 4,
            "parameters_cp": ParametersCp.default_cpsat(),
        },
    )
    f1s, f2s = [], []
    for sol, fit in front:
        print(f"  Obj: {fit} | Sol: {sol}")
        if pareto_solver.dict_function["change_cost"](sol) >= 1:
            f1s.append(pareto_solver.dict_function["change_cost"](sol))
            f2s.append(pareto_solver.dict_function["ramp_up_cost"](sol))
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"pareto_rcsalbp_{instance[:-5]}_more.png")
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plt.show()
    # fig, slider = plot_rcalbpl_dashboard(problem, sol)


def main_pareto_dp(instance="187_2_26_2880.json"):
    problem = parse_rcalbpl_json("instances/" + instance)
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
            time_limit_phase2=50,
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
            time_limit=500,
        ),
    )
    solver = SequentialMetasolver(list_subbricks=[brick1, brick2], problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    res_dict = {"instance": instance, "sol": sol}
    pickle.dump(
        res_dict, open(f"sol_{instance}_dp_more_{time.process_time()}.pkl", "wb")
    )
    import didppy as dp

    postpro_solver = DpRCALBPLPostProSolver(problem=problem)
    front = postpro_solver.create_result_storage([])
    postpro_solver.init_model(from_solution=sol, max_nb_adjustments=1)
    for i in range(1, len(postpro_solver.decision_step) + 1):
        postpro_solver.init_model(from_solution=sol, max_nb_adjustments=i)
        res = postpro_solver.solve(solver=dp.CABS, time_limit=5, threads=10)
        front.extend(res[-1:])
        print(problem.evaluate(res[-1][0]))

    f1s, f2s = [], []
    for sol, fit in front:
        eval_ = problem.evaluate(sol)
        dur_rampup = eval_["ramp_up_duration"]
        nb_adjustments = eval_["nb_adjustments"]
        print(f"  Obj: {fit} | Sol: {sol}")
        if nb_adjustments >= 1:
            f1s.append(nb_adjustments)
            f2s.append(dur_rampup)
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(f1s, f2s, c="green", s=100, label="Pareto Front")
    # Known optima for Example 9 are (1, 2) and (3, 0)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Pareto Front (Epsilon Constraint via Add/Remove)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"pareto_rcsalbp_{instance[:-5]}_dp_more.png")
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plt.show()
    # fig, slider = plot_rcalbpl_dashboard(problem, sol)


def main_script():
    # main_pareto_dp("187_2_26_2880.json")
    instance_folder = "instances/"
    for instance in os.listdir(instance_folder)[::-1]:
        main_pareto_dp(instance)
        main_pareto(instance)


if __name__ == "__main__":
    main_script()
