import os

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_parser import (
    get_data_available,
    get_results_available,
    parse_file,
    parse_results_file,
)
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import GreedyChoice, PileSolverRCPSP


def compare_results():
    results_available = get_results_available()
    result_file = [f for f in results_available if "J30_DH1m.csv" in f]
    result_file = [f for f in results_available if "j60" in f]
    result_file = [f for f in results_available if "J120" in f]
    folder_image = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images/")
    print(folder_image)
    if not os.path.exists(folder_image):
        os.makedirs(folder_image)
    results = parse_results_file(result_file[0])
    for result in results:
        file_problem = result["file_problem"]
        files_available = get_data_available()
        if file_problem in files_available:
            print(file_problem)
            rcpsp_model = parse_file(file_problem)
            solver = PileSolverRCPSP(rcpsp_model=rcpsp_model)
            result_store = solver.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
            best_solution = result_store.get_best_solution()
            solution_baseline: RCPSPSolution = result["Solution"]
            schedule = solution_baseline.rcpsp_schedule
            # we can recompute the end times...
            for task in schedule:
                schedule[task]["end_time"] = (
                    schedule[task]["start_time"]
                    + rcpsp_model.mode_details[task][1]["duration"]
                )
            solution_baseline = RCPSPSolution(
                problem=rcpsp_model,
                rcpsp_schedule=schedule,
                rcpsp_modes=solution_baseline.rcpsp_modes,
                rcpsp_schedule_feasible=True,
            )
            fit_pile = rcpsp_model.evaluate(best_solution)
            fit_baseline = rcpsp_model.evaluate(solution_baseline)
            base_name = os.path.basename(file_problem)
            fig_pil = plot_ressource_view(
                rcpsp_model=rcpsp_model,
                rcpsp_sol=best_solution,
                title_figure="greedy_results_" + str(base_name),
            )
            fig_baseline = plot_ressource_view(
                rcpsp_model=rcpsp_model,
                rcpsp_sol=solution_baseline,
                title_figure="external_results_" + str(base_name),
            )
            fig_pil.savefig(
                os.path.join(folder_image, str(base_name) + "_greedy_results.png")
            )
            fig_baseline.savefig(
                os.path.join(folder_image, str(base_name) + "_external_results.png")
            )
            print("Pile ", fit_pile)
            print("Baseline ", fit_baseline)


if __name__ == "__main__":
    compare_results()
