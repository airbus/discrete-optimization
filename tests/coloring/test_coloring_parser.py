from discrete_optimization.coloring.coloring_parser import files_available, parse_file


def parsing():
    file_path = files_available[0]
    coloring_model = parse_file(file_path)
    print(coloring_model)
    print(coloring_model.graph)
    print(coloring_model.number_of_nodes)
    print(coloring_model.graph.nodes_name)


if __name__ == "__main__":
    parsing()
