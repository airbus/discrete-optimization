#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.datasets import get_data_home
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file


def test_parsing_sm():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]

    rcpsp_model = parse_file(file_path)
    assert rcpsp_model.n_jobs == 32
    assert rcpsp_model.resources_list == ["R1", "R2", "R3", "R4"]
    assert rcpsp_model.successors == {
        1: [2, 3, 4],
        2: [6, 11, 15],
        3: [7, 8, 13],
        4: [5, 9, 10],
        5: [20],
        6: [30],
        7: [27],
        8: [12, 19, 27],
        9: [14],
        10: [16, 25],
        11: [20, 26],
        12: [14],
        13: [17, 18],
        14: [17],
        15: [25],
        16: [21, 22],
        17: [22],
        18: [20, 22],
        19: [24, 29],
        20: [23, 25],
        21: [28],
        22: [23],
        23: [24],
        24: [30],
        25: [30],
        26: [31],
        27: [28],
        28: [31],
        29: [32],
        30: [32],
        31: [32],
        32: [],
    }
    assert rcpsp_model.mode_details == {
        1: {1: {"duration": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0}},
        2: {1: {"duration": 8, "R1": 4, "R2": 0, "R3": 0, "R4": 0}},
        3: {1: {"duration": 4, "R1": 10, "R2": 0, "R3": 0, "R4": 0}},
        4: {1: {"duration": 6, "R1": 0, "R2": 0, "R3": 0, "R4": 3}},
        5: {1: {"duration": 3, "R1": 3, "R2": 0, "R3": 0, "R4": 0}},
        6: {1: {"duration": 8, "R1": 0, "R2": 0, "R3": 0, "R4": 8}},
        7: {1: {"duration": 5, "R1": 4, "R2": 0, "R3": 0, "R4": 0}},
        8: {1: {"duration": 9, "R1": 0, "R2": 1, "R3": 0, "R4": 0}},
        9: {1: {"duration": 2, "R1": 6, "R2": 0, "R3": 0, "R4": 0}},
        10: {1: {"duration": 7, "R1": 0, "R2": 0, "R3": 0, "R4": 1}},
        11: {1: {"duration": 9, "R1": 0, "R2": 5, "R3": 0, "R4": 0}},
        12: {1: {"duration": 2, "R1": 0, "R2": 7, "R3": 0, "R4": 0}},
        13: {1: {"duration": 6, "R1": 4, "R2": 0, "R3": 0, "R4": 0}},
        14: {1: {"duration": 3, "R1": 0, "R2": 8, "R3": 0, "R4": 0}},
        15: {1: {"duration": 9, "R1": 3, "R2": 0, "R3": 0, "R4": 0}},
        16: {1: {"duration": 10, "R1": 0, "R2": 0, "R3": 0, "R4": 5}},
        17: {1: {"duration": 6, "R1": 0, "R2": 0, "R3": 0, "R4": 8}},
        18: {1: {"duration": 5, "R1": 0, "R2": 0, "R3": 0, "R4": 7}},
        19: {1: {"duration": 3, "R1": 0, "R2": 1, "R3": 0, "R4": 0}},
        20: {1: {"duration": 7, "R1": 0, "R2": 10, "R3": 0, "R4": 0}},
        21: {1: {"duration": 2, "R1": 0, "R2": 0, "R3": 0, "R4": 6}},
        22: {1: {"duration": 7, "R1": 2, "R2": 0, "R3": 0, "R4": 0}},
        23: {1: {"duration": 2, "R1": 3, "R2": 0, "R3": 0, "R4": 0}},
        24: {1: {"duration": 3, "R1": 0, "R2": 9, "R3": 0, "R4": 0}},
        25: {1: {"duration": 3, "R1": 4, "R2": 0, "R3": 0, "R4": 0}},
        26: {1: {"duration": 7, "R1": 0, "R2": 0, "R3": 4, "R4": 0}},
        27: {1: {"duration": 8, "R1": 0, "R2": 0, "R3": 0, "R4": 7}},
        28: {1: {"duration": 3, "R1": 0, "R2": 8, "R3": 0, "R4": 0}},
        29: {1: {"duration": 7, "R1": 0, "R2": 7, "R3": 0, "R4": 0}},
        30: {1: {"duration": 2, "R1": 0, "R2": 7, "R3": 0, "R4": 0}},
        31: {1: {"duration": 2, "R1": 0, "R2": 0, "R3": 2, "R4": 0}},
        32: {1: {"duration": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0}},
    }
    assert rcpsp_model.resources == {"R1": 12, "R2": 13, "R3": 4, "R4": 12}
    assert rcpsp_model.non_renewable_resources == []
    assert rcpsp_model.horizon == 158


def test_parsing_mm():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Single mode RCPSP
    file_path = files[0]

    rcpsp_model = parse_file(file_path)
    assert rcpsp_model.n_jobs == 12
    assert rcpsp_model.resources_list == ["R1", "R2", "N1", "N2"]
    assert rcpsp_model.successors == {
        1: [2, 3, 4],
        2: [10, 11],
        3: [5, 9],
        4: [7, 10],
        5: [6],
        6: [7, 8],
        7: [11],
        8: [10, 11],
        9: [12],
        10: [12],
        11: [12],
        12: [],
    }
    assert rcpsp_model.mode_details == {
        1: {1: {"duration": 0, "R1": 0, "R2": 0, "N1": 0, "N2": 0}},
        2: {
            1: {"duration": 2, "R1": 6, "R2": 0, "N1": 9, "N2": 0},
            2: {"duration": 5, "R1": 0, "R2": 3, "N1": 0, "N2": 4},
            3: {"duration": 7, "R1": 0, "R2": 3, "N1": 9, "N2": 0},
        },
        3: {
            1: {"duration": 4, "R1": 0, "R2": 8, "N1": 0, "N2": 8},
            2: {"duration": 9, "R1": 7, "R2": 0, "N1": 3, "N2": 0},
            3: {"duration": 10, "R1": 6, "R2": 0, "N1": 0, "N2": 6},
        },
        4: {
            1: {"duration": 1, "R1": 0, "R2": 9, "N1": 0, "N2": 5},
            2: {"duration": 3, "R1": 6, "R2": 0, "N1": 3, "N2": 0},
            3: {"duration": 6, "R1": 5, "R2": 0, "N1": 2, "N2": 0},
        },
        5: {
            1: {"duration": 5, "R1": 0, "R2": 7, "N1": 0, "N2": 10},
            2: {"duration": 7, "R1": 0, "R2": 6, "N1": 6, "N2": 0},
            3: {"duration": 10, "R1": 0, "R2": 6, "N1": 0, "N2": 9},
        },
        6: {
            1: {"duration": 4, "R1": 0, "R2": 6, "N1": 4, "N2": 0},
            2: {"duration": 7, "R1": 0, "R2": 5, "N1": 4, "N2": 0},
            3: {"duration": 9, "R1": 0, "R2": 5, "N1": 0, "N2": 6},
        },
        7: {
            1: {"duration": 2, "R1": 9, "R2": 0, "N1": 0, "N2": 5},
            2: {"duration": 2, "R1": 0, "R2": 6, "N1": 10, "N2": 0},
            3: {"duration": 7, "R1": 9, "R2": 0, "N1": 10, "N2": 0},
        },
        8: {
            1: {"duration": 2, "R1": 10, "R2": 0, "N1": 7, "N2": 0},
            2: {"duration": 5, "R1": 7, "R2": 0, "N1": 0, "N2": 6},
            3: {"duration": 5, "R1": 0, "R2": 4, "N1": 6, "N2": 0},
        },
        9: {
            1: {"duration": 1, "R1": 6, "R2": 0, "N1": 0, "N2": 2},
            2: {"duration": 2, "R1": 0, "R2": 2, "N1": 0, "N2": 1},
            3: {"duration": 4, "R1": 1, "R2": 0, "N1": 3, "N2": 0},
        },
        10: {
            1: {"duration": 7, "R1": 0, "R2": 3, "N1": 0, "N2": 7},
            2: {"duration": 7, "R1": 0, "R2": 7, "N1": 4, "N2": 0},
            3: {"duration": 9, "R1": 7, "R2": 0, "N1": 2, "N2": 0},
        },
        11: {
            1: {"duration": 4, "R1": 0, "R2": 10, "N1": 8, "N2": 0},
            2: {"duration": 10, "R1": 5, "R2": 0, "N1": 8, "N2": 0},
            3: {"duration": 10, "R1": 0, "R2": 8, "N1": 0, "N2": 7},
        },
        12: {1: {"duration": 0, "R1": 0, "R2": 0, "N1": 0, "N2": 0}},
    }
    assert rcpsp_model.resources == {"R1": 10, "R2": 13, "N1": 29, "N2": 30}
    assert rcpsp_model.non_renewable_resources == ["N1", "N2"]
    assert rcpsp_model.horizon == 77


def test_parsing_rcp():
    data_folder = f"{get_data_home()}/rcpsp/RG30/Set 1/"
    files = get_data_available(data_folder=data_folder)
    files = [f for f in files if "Pat8.rcp" in f]
    assert len(files) > 0
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    # Checking expected values to check at least partially the parsing of rcp files.
    assert len(rcpsp_model.resources_list) == 4
    assert all(rcpsp_model.resources[r] == 10 for r in rcpsp_model.resources_list)
    assert rcpsp_model.source_task == 1
    assert rcpsp_model.sink_task == 32
    assert len(rcpsp_model.successors[rcpsp_model.source_task]) == 18
    successors_source = [
        2,
        3,
        4,
        5,
        7,
        10,
        11,
        12,
        13,
        14,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ]
    assert all(
        s in rcpsp_model.successors[rcpsp_model.source_task] for s in successors_source
    )
