#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import os
from typing import Optional

from discrete_optimization.datasets import ERROR_MSG_MISSING_DATASETS, get_data_home
from discrete_optimization.rcalbp_l.problem import RCALBPLProblem


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
) -> list[str]:
    """Get datasets available for rcpsp.

    Params:
        data_folder: folder where datasets for rcpsp whould be find.
            If None, we look in "rcpsp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/rcalb_l"

    try:
        files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e) + ERROR_MSG_MISSING_DATASETS)
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def parse_rcalbpl_json(file_path: str) -> RCALBPLProblem:
    """
    Parses the RC-ALBP/L JSON data and constructs the Problem instance.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # 1. Base Variables
    c_target = data["c_target"]
    c_max = data["c_max"]
    nb_stations = data["nb_stations"]
    nb_periods = data["nb_periods"]
    nb_tasks = data["nb_tasks"]

    # 2. Precedences
    precedences = [(p[0] - 1, p[1] - 1) for p in data.get("precedences", [])]

    # 3. Durations matrix
    durations = data.get("durations", [])

    # 4. Resources
    nb_resources = data.get("nb_resources", 0)
    capa_resources = data.get("capa_resources", [])
    cons_resources = data.get("cons_resources", [])

    # 5. Zones
    nb_zones = data.get("nb_zones", 0)
    capa_zones = data.get("capa_zones", [])
    cons_zones = data.get("cons_zones", [])
    neutr_zones = data.get("neutr_zones", [])
    neutr_zones = [[i - 1 for i in x] for x in neutr_zones]

    return RCALBPLProblem(
        c_target=c_target,
        c_max=c_max,
        nb_stations=nb_stations,
        nb_periods=nb_periods,
        nb_tasks=nb_tasks,
        precedences=precedences,
        durations=durations,
        nb_resources=nb_resources,
        capa_resources=capa_resources,
        cons_resources=cons_resources,
        nb_zones=nb_zones,
        capa_zones=capa_zones,
        cons_zones=cons_zones,
        neutr_zones=neutr_zones,
    )
