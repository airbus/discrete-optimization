#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from enum import Enum

this_path = os.path.dirname(os.path.abspath(__file__))
files_mzn = {
    "single": os.path.join(this_path, "../minizinc/rcpsp_single_mode_mzn.mzn"),
    "single-preemptive": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_mzn_preemptive.mzn"
    ),
    "multi-preemptive": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_preemptive.mzn"
    ),
    "multi-preemptive-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_preemptive_calendar.mzn"
    ),
    "single-preemptive-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_mzn_preemptive_calendar.mzn"
    ),
    "multi": os.path.join(this_path, "../minizinc/rcpsp_multi_mode_mzn.mzn"),
    "multi-faketasks": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_with_faketasks.mzn"
    ),
    "multi-no-bool": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_no_bool.mzn"
    ),
    "multi-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_calendar.mzn"
    ),
    "multi-calendar-boxes": os.path.join(
        this_path, "../minizinc/rcpsp_mzn_calendar_boxes.mzn"
    ),
    "multi-resource-feasibility": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_resource_feasibility_mzn.mzn"
    ),
    "modes": os.path.join(this_path, "../minizinc/mrcpsp_mode_satisfy.mzn"),
}


class CPModelEnum(Enum):
    SINGLE = "single"
    SINGLE_PREEMPTIVE = "single-preemptive"
    SINGLE_PREEMPTIVE_CALENDAR = "single-preemptive-calendar"
    MULTI_PREEMPTIVE = "multi-preemptive"
    MULTI_PREEMPTIVE_CALENDAR = "multi-preemptive-calendar"
    MULTI = "multi"
    MULTI_FAKETASKS = "multi-faketasks"
    MULTI_NO_BOOL = "multi-no-bool"
    MULTI_CALENDAR = "multi-calendar"
    MULTI_CALENDAR_BOXES = "multi-calendar-boxes"
    MULTI_RESOURCE_FEASIBILITY = "multi-resource-feasibility"
    MODES = "modes"
