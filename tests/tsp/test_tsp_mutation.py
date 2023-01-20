#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.tsp.mutation.mutation_tsp import MutationSwapTSP
from discrete_optimization.tsp.tsp_model import compute_length
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_mutation():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    model = parse_file(files[0])
    mutation = MutationSwapTSP(model)
    solution = model.get_dummy_solution()
    sol = mutation.mutate_and_compute_obj(solution)
    lengths, obj = compute_length(
        start_index=model.start_index,
        end_index=model.end_index,
        solution=sol[0].permutation,
        list_points=model.list_points,
        node_count=model.node_count,
        length_permutation=model.length_permutation,
    )
    assert len(sol[0].lengths) == 51
    assert len(lengths) == 51
    sol_back = sol[1].backtrack_local_move(sol[0])
    assert sol_back.length == sol[0].length
