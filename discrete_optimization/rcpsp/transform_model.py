#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Transform a model without special constraints into one with those.
#  Also permits to pass from a classic RCPSP to a preemptive version
from discrete_optimization.rcpsp import RCPSPModelSpecialConstraintsPreemptive
from discrete_optimization.rcpsp.rcpsp_model import (
    RCPSPModel,
    SpecialConstraintsDescription,
)


def from_rcpsp_model(
    rcpsp_model: RCPSPModel,
    constraints: SpecialConstraintsDescription,
    preemptive=False,
):
    if preemptive:
        return RCPSPModelSpecialConstraintsPreemptive(
            resources=rcpsp_model.resources,
            non_renewable_resources=rcpsp_model.non_renewable_resources,
            mode_details=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            horizon=rcpsp_model.horizon,
            special_constraints=constraints,
            tasks_list=rcpsp_model.tasks_list,
            source_task=rcpsp_model.source_task,
            sink_task=rcpsp_model.sink_task,
            name_task=rcpsp_model.name_task,
        )
    return RCPSPModel(
        resources=rcpsp_model.resources,
        non_renewable_resources=rcpsp_model.non_renewable_resources,
        mode_details=rcpsp_model.mode_details,
        successors=rcpsp_model.successors,
        horizon=rcpsp_model.horizon,
        special_constraints=constraints,
        tasks_list=rcpsp_model.tasks_list,
        source_task=rcpsp_model.source_task,
        sink_task=rcpsp_model.sink_task,
        name_task=rcpsp_model.name_task,
    )
