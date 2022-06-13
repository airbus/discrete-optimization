from discrete_optimization.tsp.solver.tsp_cp_solver import TSP_CP_Solver, TSP_CPModel
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_int_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = TSP_CPModel.INT_VERSION
    cp_solver = TSP_CP_Solver(model, model_type=model_type)
    cp_solver.init_model(solver="chuffed")
    var, fit = cp_solver.solve(max_time_seconds=100).get_best_solution_fit()
    print(var)


def test_float_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = TSP_CPModel.FLOAT_VERSION
    cp_solver = TSP_CP_Solver(model, model_type=model_type)
    cp_solver.init_model(solver="gecode")
    var, fit = cp_solver.solve(max_time_seconds=100).get_best_solution_fit()
    print(var)


if __name__ == "__main__":
    test_int_cp()
