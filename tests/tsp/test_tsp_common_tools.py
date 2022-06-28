from discrete_optimization.tsp.common_tools_tsp import closest_greedy
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_common_tools():
    file_location = [f for f in get_data_available() if f.endswith("tsp_574_1")][0]
    model = parse_file(file_location)
    sol, length_circuit, opt = closest_greedy(model.node_count, model.list_points)
    assert len(sol) == 574
