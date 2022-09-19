#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver as rcpsp_lns
from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_builder import (
    OptionNeighborRandom,
    build_neighbor_mixing_cut_parts,
    build_neighbor_mixing_methods,
    build_neighbor_random,
    mix_both,
)
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import (
    ParamsConstraintBuilder,
)
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LNS_CP, SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import PartialSolutionPreemptive
from discrete_optimization.rcpsp.solver.cp_lns_methods_clean import (
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
)
from discrete_optimization.rcpsp.solver.cp_lns_methods_preemptive import (
    PostProLeftShift,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_PREEMMPTIVE,
    CP_RCPSP_MZN,
    CP_RCPSP_MZN_PREEMMPTIVE,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)

GENERIC_CLASS = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraints,
    RCPSPModelSpecialConstraintsPreemptive,
]


def build_default_cp_model(
    rcpsp_problem: GENERIC_CLASS, partial_solution=None, **kwargs
):
    if isinstance(
        rcpsp_problem, (RCPSPModelPreemptive, RCPSPModelSpecialConstraintsPreemptive)
    ):
        if rcpsp_problem.is_rcpsp_multimode():
            solver = CP_MRCPSP_MZN_PREEMMPTIVE(
                rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            solver.init_model(
                output_type=True,
                model_type="multi-preemptive",
                nb_preemptive=12,
                max_preempted=100,
                partial_solution=partial_solution,
                **kwargs
            )
        else:
            solver = CP_RCPSP_MZN_PREEMMPTIVE(
                rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            solver.init_model(
                output_type=True,
                model_type="single-preemptive",
                nb_preemptive=12,
                max_preempted=100,
                partial_solution=partial_solution,
                **kwargs
            )
        return solver
    else:
        if rcpsp_problem.is_rcpsp_multimode():
            solver = CP_MRCPSP_MZN(
                rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            solver.init_model(
                output_type=True,
                model_type="multi",
                partial_solution=partial_solution,
                **kwargs
            )
        else:
            solver = CP_RCPSP_MZN(
                rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            if "model_type" not in kwargs:
                kwargs["model_type"] = "single"  # you can use "single-no-search" too.
            solver.init_model(
                output_type=True, partial_solution=partial_solution, **kwargs
            )
        return solver


class LargeNeighborhoodSearchRCPSP(SolverDO):
    def __init__(
        self,
        rcpsp_problem: GENERIC_CLASS,
        partial_solution: PartialSolutionPreemptive = None,
        **kwargs
    ):
        self.rcpsp_problem = rcpsp_problem
        graph = build_graph_rcpsp_object(self.rcpsp_problem)
        solver = build_default_cp_model(
            rcpsp_problem=rcpsp_problem, partial_solution=partial_solution, **kwargs
        )
        params_objective_function = get_default_objective_setup(
            problem=self.rcpsp_problem
        )
        constraint_handler = None
        option = kwargs.get("option_neighbor_operator", 2)
        if option == 0:
            constraint_handler = build_neighbor_random(
                option_neighbor=kwargs.get(
                    "option_neighbor_random", OptionNeighborRandom.MIX_ALL
                ),
                rcpsp_model=self.rcpsp_problem,
            )
        if option == 1:
            constraint_handler = mix_both(
                rcpsp_model=rcpsp_problem,
                option_neighbor_random=kwargs.get(
                    "option_neighbor_random", OptionNeighborRandom.MIX_ALL
                ),
                graph=graph,
                fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
                cut_part=kwargs.get("cut_part", 10),
                params_list=kwargs.get(
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
                ),
            )
        if option == 2:
            constraint_handler = build_neighbor_mixing_methods(
                rcpsp_model=self.rcpsp_problem,
                graph=graph,
                fraction_subproblem=kwargs.get("fraction_subproblem", 0.05),
                cut_part=kwargs.get("cut_part", 10),
                params_list=kwargs.get(
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
                ),
            )
        if option == 3:
            constraint_handler = build_neighbor_mixing_cut_parts(
                rcpsp_model=rcpsp_problem,
                graph=graph,
                params_list=kwargs.get(
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
                ),
            )
        initial_solution_provider = kwargs.get("initial_solution_provider", None)
        if initial_solution_provider is None:
            initial_solution_provider = InitialSolutionRCPSP(
                problem=self.rcpsp_problem,
                initial_method=InitialMethodRCPSP.DUMMY,
                params_objective_function=params_objective_function,
            )
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.params_objective_function = params_objective_function
        self.cp_solver = solver
        if isinstance(
            self.rcpsp_problem,
            (RCPSPModelPreemptive, RCPSPModelSpecialConstraintsPreemptive),
        ):
            self.post_process_solution = PostProLeftShift(
                problem=self.rcpsp_problem,
                params_objective_function=params_objective_function,
                do_ls=kwargs.get("do_ls", False),
            )
        else:
            self.post_process_solution = rcpsp_lns.PostProcessLeftShift(
                rcpsp_problem=self.rcpsp_problem, partial_solution=None
            )
        self.lns_solver = LNS_CP(
            problem=self.rcpsp_problem,
            cp_solver=self.cp_solver,
            post_process_solution=self.post_process_solution,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
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
