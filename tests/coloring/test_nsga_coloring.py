import matplotlib.pyplot as plt
from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    TypeAttribute,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
)
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    plot_pareto_2d,
    plot_storage_2d,
)


def test_coloring_nsga_1():

    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    objectives = ["nb_colors", "nb_violations"]
    ga_solver = Nsga(
        color_problem,
        encoding="colors",
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    plt.show()


def test_coloring_nsga_2():

    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem: ColoringProblem = parse_file(file)

    encoding = {
        "name": "colors",
        "type": [TypeAttribute.LIST_INTEGER],
        "n": 70,
        "arrity": 10,
    }

    objectives = ["nb_colors", "nb_violations"]
    ga_solver = Nsga(
        color_problem,
        encoding=encoding,
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    plt.show()


if __name__ == "__main__":
    test_coloring_nsga_1()
    test_coloring_nsga_2()
