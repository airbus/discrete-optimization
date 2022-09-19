#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

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
    ANY_RCPSP,
    BasicConstraintBuilder,
    ConstraintHandlerScheduling,
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
from discrete_optimization.generic_tools.lns_mip import InitialSolutionFromSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
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
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.solvers.lns_post_process_rcpsp import (
    PostProMSRCPSP,
)


class ConstraintHandlerType(Enum):
    MIX_SUBPROBLEMS = 0
    SOLUTION_REPAIR = 1


def build_default_cp_model(rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
    if rcpsp_problem.is_preemptive():
        if rcpsp_problem.is_multiskill():
            solver = CP_MS_MRCPSP_MZN_PREEMPTIVE(
                rcpsp_model=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
            )
            solver.init_model(
                output_type=True,
                model_type="ms_rcpsp_preemptive",
                nb_preemptive=kwargs.get("nb_preemptive", 13),
                max_preempted=kwargs.get("max_preempted", 100),
                partial_solution=partial_solution,
                possibly_preemptive=[
                    rcpsp_problem.preemptive_indicator[t]
                    for t in rcpsp_problem.tasks_list
                ],
                unit_usage_preemptive=kwargs.get("unit_usage_preemptive", True),
                exact_skills_need=False,
                add_calendar_constraint_unit=False,
                add_partial_solution_hard_constraint=kwargs.get(
                    "add_partial_solution_hard_constraint", False
                ),
                **{
                    x: kwargs[x]
                    for x in kwargs
                    if x
                    not in [
                        "nb_preemptive",
                        "max_preempted",
                        "fake_tasks",
                        "add_partial_solution_hard_constraint",
                    ]
                }
            )
            return solver
    if rcpsp_problem.is_preemptive():
        if not rcpsp_problem.is_multiskill():
            if rcpsp_problem.is_rcpsp_multimode():
                solver = CP_MRCPSP_MZN_PREEMMPTIVE(
                    rcpsp_model=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    model_type="multi-preemptive",
                    nb_preemptive=kwargs.get("nb_preemptive", 13),
                    max_preempted=kwargs.get("max_preempted", 100),
                    fake_tasks=kwargs.get("fake_tasks", True),
                    partial_solution=partial_solution,
                    possibly_preemptive=[
                        rcpsp_problem.preemptive_indicator[t]
                        for t in rcpsp_problem.tasks_list
                    ],
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x
                        not in [
                            "nb_preemptive",
                            "max_preempted",
                            "fake_tasks",
                            "add_partial_solution_hard_constraint",
                        ]
                    }
                )
                return solver
            if not rcpsp_problem.is_rcpsp_multimode():
                solver = CP_RCPSP_MZN_PREEMMPTIVE(
                    rcpsp_model=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    model_type="single-preemptive",
                    nb_preemptive=kwargs.get("nb_preemptive", 13),
                    max_preempted=kwargs.get("max_preempted", 100),
                    fake_tasks=kwargs.get("fake_tasks", True),
                    partial_solution=partial_solution,
                    possibly_preemptive=[
                        rcpsp_problem.preemptive_indicator[t]
                        for t in rcpsp_problem.tasks_list
                    ],
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x
                        not in [
                            "nb_preemptive",
                            "max_preempted",
                            "fake_tasks",
                            "add_partial_solution_hard_constraint",
                        ]
                    }
                )
                return solver
    if not rcpsp_problem.is_preemptive():
        if rcpsp_problem.is_multiskill():
            solver = CP_MS_MRCPSP_MZN(
                rcpsp_model=rcpsp_problem,
                cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                one_ressource_per_task=kwargs.get(
                    "one_ressource_per_task", rcpsp_problem.one_unit_per_task_max
                ),
            )
            solver.init_model(
                output_type=True,
                exact_skills_need=False,
                partial_solution=partial_solution,
                add_partial_solution_hard_constraint=kwargs.get(
                    "add_partial_solution_hard_constraint", False
                ),
                **{
                    x: kwargs[x]
                    for x in kwargs
                    if x not in ["add_partial_solution_hard_constraint"]
                }
            )
            return solver
        if not rcpsp_problem.is_multiskill():
            if rcpsp_problem.is_rcpsp_multimode():
                solver = CP_MRCPSP_MZN(
                    rcpsp_model=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    partial_solution=partial_solution,
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x not in ["add_partial_solution_hard_constraint"]
                    }
                )
                return solver
            if not rcpsp_problem.is_rcpsp_multimode():
                solver = CP_RCPSP_MZN(
                    rcpsp_model=rcpsp_problem,
                    cp_solver_name=kwargs.get("cp_solver_name", CPSolverName.CHUFFED),
                )
                solver.init_model(
                    output_type=True,
                    partial_solution=partial_solution,
                    add_partial_solution_hard_constraint=kwargs.get(
                        "add_partial_solution_hard_constraint", False
                    ),
                    **{
                        x: kwargs[x]
                        for x in kwargs
                        if x not in ["add_partial_solution_hard_constraint"]
                    }
                )
                return solver


def build_default_postpro(rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
    if not rcpsp_problem.is_multiskill():
        if rcpsp_problem.is_preemptive():
            post_process_solution = PostProLeftShift(
                problem=rcpsp_problem,
                params_objective_function=None,
                do_ls=kwargs.get("do_ls", False),
            )
            return post_process_solution
        if partial_solution is not None:
            return None
        if not rcpsp_problem.is_preemptive():
            post_process_solution = rcpsp_lns.PostProcessLeftShift(
                rcpsp_problem=rcpsp_problem, partial_solution=partial_solution
            )
            return post_process_solution
    if rcpsp_problem.is_multiskill():
        if rcpsp_problem.is_preemptive():
            return None
        else:
            post_process_solution = PostProMSRCPSP(
                problem=rcpsp_problem, params_objective_function=None
            )
            return post_process_solution


def build_default_initial_solution(rcpsp_problem: ANY_RCPSP, **kwargs):
    if rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LS_RCPSP_Solver(model=rcpsp_problem, ls_solver=LS_SOLVER.SA),
            nb_iteration_max=500,
        )
        return initial_solution_provider
    if not rcpsp_problem.is_multiskill():
        initial_solution_provider = InitialSolutionFromSolver(
            LS_RCPSP_Solver(model=rcpsp_problem, ls_solver=LS_SOLVER.SA),
            nb_iteration_max=200,
        )
        return initial_solution_provider


def build_constraint_handler(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    constraint_handler_type = kwargs.get(
        "constraint_handler_type", ConstraintHandlerType.MIX_SUBPROBLEMS
    )
    constraint_handler = None
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
        constraint_handler = ConstraintHandlerScheduling(
            problem=rcpsp_problem,
            basic_constraint_builder=basic_constraint_builder,
            params_list=params_list,
            use_makespan_of_subtasks=kwargs.get("use_makespan_of_subtasks", False),
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


def build_constraint_handler_helper(rcpsp_problem: ANY_RCPSP, graph, **kwargs):
    option = kwargs.get("option_neighbor_operator", 2)
    constraint_handler = None
    if option == 0:
        constraint_handler = build_neighbor_random(
            option_neighbor=kwargs.get(
                "option_neighbor_random", OptionNeighborRandom.MIX_ALL
            ),
            rcpsp_model=rcpsp_problem,
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
            rcpsp_model=rcpsp_problem,
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
    return constraint_handler


class LargeNeighborhoodSearchScheduling(SolverDO):
    def __init__(self, rcpsp_problem: ANY_RCPSP, partial_solution=None, **kwargs):
        self.rcpsp_problem = rcpsp_problem
        graph = build_graph_rcpsp_object(self.rcpsp_problem)
        solver = kwargs.get("cp_solver", None)
        if solver is None:
            solver = build_default_cp_model(
                rcpsp_problem=rcpsp_problem, partial_solution=partial_solution, **kwargs
            )
        self.cp_solver = solver
        params_objective_function = get_default_objective_setup(
            problem=self.rcpsp_problem
        )
        self.parameters_cp = kwargs.get("parameters_cp", ParametersCP.default())
        self.constraint_handler = kwargs.get("constraint_handler", None)
        if self.constraint_handler is None:
            self.constraint_handler = build_constraint_handler(
                rcpsp_problem=self.rcpsp_problem,
                graph=graph,
                multiskill=self.rcpsp_problem.is_multiskill(),
                preemptive=self.rcpsp_problem.is_preemptive(),
                **kwargs
            )
        self.post_pro = kwargs.get("post_process_solution", None)
        if self.post_pro is None:
            self.post_pro = build_default_postpro(
                rcpsp_problem=self.rcpsp_problem,
                partial_solution=partial_solution,
                **kwargs
            )
        self.initial_solution_provider = kwargs.get("initial_solution_provider", None)
        if self.initial_solution_provider is None:
            self.initial_solution_provider = build_default_initial_solution(
                rcpsp_problem=self.rcpsp_problem, **kwargs
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
