from typing import Union
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, SingleModeRCPSPModel,\
    MultiModeRCPSPModel, RCPSPModelCalendar
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import  RCPSPModelSpecialConstraints, \
    RCPSPModelSpecialConstraintsPreemptive
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive

GENERIC_CLASS = Union[RCPSPModel,
                      RCPSPModelPreemptive,
                      RCPSPModelSpecialConstraints,
                      RCPSPModelSpecialConstraintsPreemptive]
