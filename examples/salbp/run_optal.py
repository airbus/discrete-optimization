import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.salbp.parser import get_data_available, parse_alb_file
from discrete_optimization.salbp.solvers.optal import (
    OptalSalbp12Solver,
    OptalSalbpSolver,
    SalbpProblem_1_2,
)

logging.basicConfig(level=logging.INFO)


def run_optal():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_337.alb" in f][0]
    problem = parse_alb_file(file)
    solver = OptalSalbpSolver(problem)
    solver.init_model(use_lb=True)
    # greedy = GreedySalbpSolver(problem)
    # sol = greedy.solve()[-1][0]
    # solver.set_warm_start(sol)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=100, preset="Default")
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_optal_on_salbp2():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_337.alb" in f][0]
    problem = parse_alb_file(file)
    problem = SalbpProblem_1_2.from_salbp1(problem)
    solver = OptalSalbp12Solver(
        problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["cycle_time", "nb_stations"],
            weights=[1, 0],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=100,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_optal_on_lexico():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_337.alb" in f][0]
    problem = parse_alb_file(file)
    problem = SalbpProblem_1_2.from_salbp1(problem)
    solver = OptalSalbp12Solver(
        problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["cycle_time", "nb_stations"],
            weights=[1, 0],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    lex = LexicoSolver(problem=problem, subsolver=solver)
    res = lex.solve(
        objectives=["cycle_time", "nb_stations"],
        subsolver_callbacks=[ProblemEvaluateLogger(step_verbosity_level=logging.INFO)],
        time_limit=30,
        parameters_cp=p,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def compute_pareto_front():
    files = get_data_available()
    file = [f for f in files if "instance_n=1000_337.alb" in f][0]
    problem = parse_alb_file(file)
    problem = SalbpProblem_1_2.from_salbp1(problem)
    solver = OptalSalbp12Solver(problem)
    solver.init_model()
    up_number_station = 300
    low_number_station = 10
    time_limit_per_run = 10
    granularity = 2
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    merged_res = solver.create_result_storage([])
    for value in range(up_number_station, low_number_station, -granularity):
        solver.add_bound_constraint(
            var=solver.variables["objs"]["nb_stations"], sign=SignEnum.LEQ, value=value
        )
        solver.set_lexico_objective("cycle_time")
        res = solver.solve(parameters_cp=p, time_limit=time_limit_per_run)
        merged_res.extend(res)

    import pickle

    with open("pareto_salbp.pkl", "wb") as f:
        pickle.dump(merged_res, f)
    kpis = [problem.evaluate(sol) for sol, _ in merged_res]
    nb_stations = [k["nb_stations"] for k in kpis]
    cycle_time = [k["cycle_time"] for k in kpis]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    ax.scatter(x=nb_stations, y=cycle_time, marker="o", c="black", s=100)
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("Cycle time")
    ax.set_title("Found solutions")
    plt.show()


if __name__ == "__main__":
    run_optal()
