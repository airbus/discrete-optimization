#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

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
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lns_cp import LNS_CP
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import PartialSolutionPreemptive
from discrete_optimization.rcpsp.solver.cp_lns_methods_clean import (
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
)
from discrete_optimization.rcpsp.solver.cp_lns_methods_preemptive import (
    PostProLeftShift,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN,
    CP_RCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP

GENERIC_CLASS = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
]


def build_default_cp_model(
    rcpsp_problem: GENERIC_CLASS, partial_solution=None, **kwargs
):
    if isinstance(
        rcpsp_problem, (RCPSPModelPreemptive, RCPSPModelSpecialConstraintsPreemptive)
    ):
        if rcpsp_problem.is_rcpsp_multimode():
            solver = CP_MRCPSP_MZN_PREEMPTIVE(
                problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
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
            solver = CP_RCPSP_MZN_PREEMPTIVE(
                problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
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
                problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            solver.init_model(
                output_type=True,
                model_type="multi",
                partial_solution=partial_solution,
                **kwargs
            )
        else:
            solver = CP_RCPSP_MZN(
                problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
            )
            if "model_type" not in kwargs:
                kwargs["model_type"] = "single"  # you can use "single-no-search" too.
            solver.init_model(
                output_type=True, partial_solution=partial_solution, **kwargs
            )
        return solver


class LargeNeighborhoodSearchRCPSP(SolverRCPSP):
    def __init__(
        self,
        problem: GENERIC_CLASS,
        partial_solution: PartialSolutionPreemptive = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        graph = build_graph_rcpsp_object(problem)
        solver = build_default_cp_model(
            rcpsp_problem=problem, partial_solution=partial_solution, **kwargs
        )
        constraint_handler = None
        option = kwargs.get("option_neighbor_operator", 2)
        if option == 0:
            constraint_handler = build_neighbor_random(
                option_neighbor=kwargs.get(
                    "option_neighbor_random", OptionNeighborRandom.MIX_ALL
                ),
                rcpsp_model=problem,
            )
        if option == 1:
            constraint_handler = mix_both(
                rcpsp_model=problem,
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
                rcpsp_model=problem,
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
                rcpsp_model=problem,
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
                problem=problem,
                initial_method=InitialMethodRCPSP.DUMMY,
                params_objective_function=self.params_objective_function,
            )
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.params_objective_function = params_objective_function
        self.cp_solver = solver
        if isinstance(
            problem,
            (RCPSPModelPreemptive, RCPSPModelSpecialConstraintsPreemptive),
        ):
            self.post_process_solution = PostProLeftShift(
                problem=problem,
                params_objective_function=self.params_objective_function,
                do_ls=kwargs.get("do_ls", False),
            )
        else:
            self.post_process_solution = rcpsp_lns.PostProcessLeftShift(
                rcpsp_problem=problem, partial_solution=None
            )
        self.lns_solver = LNS_CP(
            problem=problem,
            cp_solver=self.cp_solver,
            post_process_solution=self.post_process_solution,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            params_objective_function=self.params_objective_function,
        )

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_first_iteration: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **kwargs
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return self.lns_solver.solve_lns(
            parameters_cp=parameters_cp,
            skip_first_iteration=skip_first_iteration,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            nb_iteration_lns=nb_iteration_lns,
            callbacks=callbacks,
            **kwargs
        )
