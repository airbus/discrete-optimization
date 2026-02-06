from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import (
    RcpspProblem,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp.problem_preemptive import (
    get_rcpsp_problemp_preemptive,
)
from discrete_optimization.rcpsp.solvers.preemptive.cpsat import (
    CpSatCalendarPreemptiveSolver,
    CpSatPreemptiveRcpspSolver,
    transform_calendar_preemptive_solution_to_preemptive,
)
from discrete_optimization.rcpsp.solvers.preemptive.optal import (
    OptalCalendarPreemptiveRcpspSolver,
    OptalPreemptiveRcpspSolver,
)
from discrete_optimization.rcpsp.utils_preemptive import (
    plot_ressource_view,
    plot_task_gantt,
)


def load_preemptive_rcpsp_problem(problem: RcpspProblem = None):
    # file = get_data_available()[1]
    if problem is None:
        file = [f for f in get_data_available() if "j1201_1" in f][0]
        problem = parse_file(file)
    for r in problem.resources_list:
        if r not in problem.non_renewable_resources:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
            if True:
                for i in range(len(problem.resources[r])):
                    if i % 5 == 0:
                        problem.resources[r][i] = 0
                    if i % 5 == 1:
                        problem.resources[r][i] = max_capa - 1
        else:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
    problem.horizon = problem.horizon * 3
    problem.update_functions()
    preemptive = get_rcpsp_problemp_preemptive(problem)
    return preemptive


def main_optal():
    preemptive = load_preemptive_rcpsp_problem()
    solver = OptalPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    import optalcp as cp

    res = solver.solve(
        parameters_cp=p,
        preset="Default",
        workers=[
            cp.WorkerParameters(searchType="FDS"),
            cp.WorkerParameters(searchType="FDSDual"),
        ],
        time_limit=10,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_optal_cal_preemptive():
    problem = parse_file([f for f in get_data_available() if "j1010_8.mm" in f][0])
    preemptive = load_preemptive_rcpsp_problem(problem)
    solver = OptalCalendarPreemptiveRcpspSolver(preemptive)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    import optalcp as cp

    res = solver.solve(
        parameters_cp=p,
        preset="Default",
        workers=[
            cp.WorkerParameters(searchType="FDS"),
            cp.WorkerParameters(searchType="FDSDual"),
        ],
        time_limit=10,
    )
    sol = res[-1][0]
    sol = transform_calendar_preemptive_solution_to_preemptive(sol, preemptive)
    print(preemptive.satisfy(sol), preemptive.evaluate(sol))
    plot_task_gantt(preemptive, sol)
    plot_ressource_view(preemptive, sol)
    plt.show()


def main_optal_lexico():
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver

    preemptive = load_preemptive_rcpsp_problem()
    solver = OptalPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    solver.use_warm_start = True
    lexico_solver = LexicoSolver(problem=preemptive, subsolver=solver)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    import optalcp as cp

    res = lexico_solver.solve(
        parameters_cp=p,
        objectives=["makespan", "nb_preemption"],
        preset="Default",
        workers=[
            cp.WorkerParameters(searchType="FDS"),
            # cp.WorkerParameters(searchType="SetTimes"),
            cp.WorkerParameters(searchType="FDSDual"),
        ],
        time_limit=30,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_cpsat():
    preemptive = load_preemptive_rcpsp_problem()
    solver = CpSatPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        preset="Default",
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=100,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


def main_cpsat_cal_preemptive():
    problem = parse_file([f for f in get_data_available() if "j1010_8.mm" in f][0])
    preemptive = load_preemptive_rcpsp_problem(problem)
    solver = CpSatCalendarPreemptiveSolver(preemptive)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=10,
    )
    sol = res[-1][0]
    sol = transform_calendar_preemptive_solution_to_preemptive(sol, preemptive)
    print(preemptive.satisfy(sol), preemptive.evaluate(sol))
    plot_task_gantt(preemptive, sol)
    plot_ressource_view(preemptive, sol)
    plt.show()


def main_cpsat_lexico():
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver

    class WSLexico(Callback):
        def on_step_end(self, step, res, solver: LexicoSolver):
            solver.subsolver.set_warm_start_from_previous_run()

    preemptive = load_preemptive_rcpsp_problem()
    solver = CpSatPreemptiveRcpspSolver(preemptive)
    solver.init_model(max_nb_preemption=10)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    lexico = LexicoSolver(problem=preemptive, subsolver=solver)
    res = lexico.solve(
        parameters_cp=p,
        callbacks=[WSLexico()],
        objectives=["makespan", "nb_preemption"],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=30,
    )
    plot_task_gantt(preemptive, res[-1][0])
    plot_ressource_view(preemptive, res[-1][0])
    plt.show()


if __name__ == "__main__":
    # main_optal_cal_preemptive()
    main_cpsat_cal_preemptive()
