#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional, Union

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    BasicConstraintBuilder,
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborConstraintBreaks,
    NeighborRandomAndNeighborGraph,
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_rcpsp_tools.solution_repair import (
    NeighborRepairProblems,
)
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lns_cp import LNS_CP
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import PartialSolutionPreemptive
from discrete_optimization.rcpsp.solver.cp_lns_methods_clean import NeighborSubproblem
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialMethodRCPSP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import (
    PostProMSRCPSP,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    InitialSolutionMS_RCPSP,
)


class ConstraintHandlerType(Enum):
    MIX_SUBPROBLEMS = 0
    SOLUTION_REPAIR = 1


GENERIC_CLASS_MULTISKILL = Union[MS_RCPSPModel, MS_RCPSPModel_Variant]


def build_default_cp_model(
    rcpsp_problem: GENERIC_CLASS_MULTISKILL, partial_solution=None, **kwargs
):
    if rcpsp_problem.preemptive:
        solver = CP_MS_MRCPSP_MZN_PREEMPTIVE(
            rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
        )
        solver.init_model(
            output_type=True,
            model_type="ms_rcpsp_preemptive",
            nb_preemptive=kwargs.get("nb_preemptive", 13),
            max_preempted=kwargs.get("max_preempted", 100),
            partial_solution=partial_solution,
            possibly_preemptive=[
                rcpsp_problem.preemptive_indicator[t] for t in rcpsp_problem.tasks_list
            ],
            unit_usage_preemptive=kwargs.get("unit_usage_preemptive", True),
            exact_skills_need=False,
            add_calendar_constraint_unit=False,
            add_partial_solution_hard_constraint=False,
            **kwargs
        )
        return solver
    else:
        solver = CP_MS_MRCPSP_MZN(
            rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
        )
        solver.init_model(
            output_type=True,
            model_type="multi-calendar",
            partial_solution=partial_solution,
            exact_skills_need=False,
            add_calendar_constraint_unit=False,
            add_partial_solution_hard_constraint=False,
            **kwargs
        )
        return solver


def build_constraint_handler(rcpsp_problem, graph, **kwargs):
    constraint_handler_type = kwargs.get(
        "constraint_handler_type", ConstraintHandlerType.MIX_SUBPROBLEMS
    )
    if constraint_handler_type == ConstraintHandlerType.MIX_SUBPROBLEMS:
        n1 = NeighborBuilderSubPart(
            problem=rcpsp_problem,
            graph=graph,
            nb_cut_part=kwargs.get("nb_cut_part", 10),
        )
        n2 = NeighborRandomAndNeighborGraph(
            problem=rcpsp_problem,
            graph=graph,
            fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
        )
        n3 = NeighborConstraintBreaks(
            problem=rcpsp_problem,
            graph=graph,
            fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
            other_constraint_handler=n1,
        )
        n_mix = NeighborBuilderMix(
            list_neighbor=[n1, n2, n3], weight_neighbor=[0.2, 0.5, 0.3]
        )
        basic_constraint_builder = BasicConstraintBuilder(
            params_constraint_builder=ParamsConstraintBuilder(
                minus_delta_primary=6000,
                plus_delta_primary=6000,
                minus_delta_secondary=400,
                plus_delta_secondary=400,
                constraint_max_time_to_current_solution=False,
            ),
            neighbor_builder=n_mix,
            preemptive=kwargs.get("preemptive", False),
            multiskill=kwargs.get("multiskill", False),
        )
        params_list = kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=400,
                    plus_delta_secondary=400,
                    constraint_max_time_to_current_solution=False,
                ),
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=0,
                    plus_delta_secondary=0,
                    constraint_max_time_to_current_solution=False,
                ),
            ],
        )
        constraint_handler = NeighborSubproblem(
            problem=rcpsp_problem,
            basic_constraint_builder=basic_constraint_builder,
            params_list=params_list,
        )
    elif constraint_handler_type == ConstraintHandlerType.SOLUTION_REPAIR:
        params_list = kwargs.get(
            "params_list",
            [
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=400,
                    plus_delta_secondary=400,
                    constraint_max_time_to_current_solution=False,
                ),
                ParamsConstraintBuilder(
                    minus_delta_primary=5000,
                    plus_delta_primary=5000,
                    minus_delta_secondary=2,
                    plus_delta_secondary=2,
                    constraint_max_time_to_current_solution=False,
                ),
            ],
        )
        constraint_handler = NeighborRepairProblems(
            problem=rcpsp_problem, params_list=params_list
        )
    return constraint_handler


class LargeNeighborhoodSearchMSRCPSP(SolverDO):
    def __init__(
        self,
        rcpsp_problem: GENERIC_CLASS_MULTISKILL,
        partial_solution: PartialSolutionPreemptive = None,
        **kwargs
    ):
        self.rcpsp_problem = rcpsp_problem
        graph = build_graph_rcpsp_object(self.rcpsp_problem)
        solver = build_default_cp_model(
            rcpsp_problem=rcpsp_problem, partial_solution=partial_solution, **kwargs
        )
        self.cp_solver = solver
        params_objective_function = get_default_objective_setup(
            problem=self.rcpsp_problem
        )
        self.parameters_cp = kwargs.get("parameters_cp", ParametersCP.default())
        self.constraint_handler = build_constraint_handler(
            rcpsp_problem=self.rcpsp_problem,
            graph=graph,
            multiskill=isinstance(self.rcpsp_problem, MS_RCPSPModel),
            preemptive=self.rcpsp_problem.preemptive,
            **kwargs
        )
        self.post_pro = None
        if not self.rcpsp_problem.preemptive:
            self.post_pro = PostProMSRCPSP(
                problem=self.rcpsp_problem,
                params_objective_function=params_objective_function,
            )
        self.initial_solution_provider = kwargs.get("initial_solution_provider", None)
        if self.initial_solution_provider is None:
            self.initial_solution_provider = InitialSolutionMS_RCPSP(
                problem=self.rcpsp_problem,
                initial_method=InitialMethodRCPSP.DUMMY,
                params_objective_function=params_objective_function,
            )
        self.lns_solver = LNS_CP(
            problem=self.rcpsp_problem,
            cp_solver=self.cp_solver,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            post_process_solution=self.post_pro,
            params_objective_function=params_objective_function,
        )

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        skip_first_iteration: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        **args
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return self.lns_solver.solve_lns(
            parameters_cp=parameters_cp,
            max_time_seconds=max_time_seconds,
            skip_first_iteration=skip_first_iteration,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            nb_iteration_lns=nb_iteration_lns,
        )
