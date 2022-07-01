from typing import Union

from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPModelCalendar,
    SingleModeRCPSPModel,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
)

GENERIC_CLASS = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
]
