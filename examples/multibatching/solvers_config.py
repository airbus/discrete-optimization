#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lp_tools import mathopt
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.generic_tools.study import SolverConfig
from discrete_optimization.multibatching.solvers.cpsat import (
    CpsatMultibatchingSolver,
    ModelingMultiBatch,
)
from discrete_optimization.multibatching.solvers.lp import (
    GurobiMultibatchingSolver,
    GurobiMultibatchingSolverUnitFlow,
    MathOptMultibatchingSolver,
    MathOptMultibatchingSolverUnitFlow,
)
from discrete_optimization.multibatching.solvers.netx import NetxMultibatchingSolver
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    CpsatPackingSubproblem,
    GreedyPackingForMultibatching,
    PackingViaBinPacking,
)
from discrete_optimization.multibatching.solvers.two_steps import (
    TwoStepMultibatchingSolver,
)

time_limit = 100


p = ParametersCp.default_cpsat()
p.nb_process = 8


def subbricks_packing(timeout, add_asp: bool = True):
    greedy = SubBrick(GreedyPackingForMultibatching, {})
    cpsat = SubBrick(
        CpsatPackingSubproblem, {"time_limit": int(timeout), "parameters_cp": p}
    )
    cpsat_binpack = SubBrick(
        cls=PackingViaBinPacking,
        kwargs={
            "bin_packing_solver": SubBrick(
                cls=CpSatBinPackSolver,
                kwargs=dict(modeling=ModelingBinPack.SCHEDULING, parameters_cp=p),
            ),
            "time_limit_per_link": 1,
        },
    )
    solvers = {"greedy": greedy, "cp": cpsat, "cp-binpack": cpsat_binpack}
    if add_asp:
        asp_binpack = SubBrick(
            cls=PackingViaBinPacking,
            kwargs={
                "time_limit_per_link": 1,
                "bin_packing_solver": SubBrick(cls=AspBinPackingSolver, kwargs={}),
            },
        )
        solvers["asp-binpack"] = asp_binpack
    return solvers


def subbricks_flow(timeout):
    c = dict(
        networkx=SubBrick(cls=NetxMultibatchingSolver, kwargs={}),
        networkx_short=SubBrick(
            cls=NetxMultibatchingSolver,
            kwargs={
                "restrict_to_shortest_paths": True,
                "shortest_path_tolerance": 1.2,
            },
        ),
        cp=SubBrick(
            cls=CpsatMultibatchingSolver,
            kwargs={
                "modeling": ModelingMultiBatch.FLOW,
                "time_limit": int(timeout),
                "parameters_cp": p,
            },
        ),
        cp_short=SubBrick(
            cls=CpsatMultibatchingSolver,
            kwargs={
                "modeling": ModelingMultiBatch.FLOW,
                "time_limit": int(timeout),
                "restrict_to_shortest_paths": True,
                "shortest_path_tolerance": 1.2,
                "parameters_cp": p,
            },
        ),
        milp=SubBrick(
            cls=GurobiMultibatchingSolver, kwargs={"time_limit": int(timeout)}
        ),
        milp_short=SubBrick(
            cls=GurobiMultibatchingSolver,
            kwargs={
                "time_limit": int(timeout),
                "restrict_to_shortest_paths": True,
                "shortest_path_tolerance": 1.2,
            },
        ),
        mathopt=SubBrick(
            cls=MathOptMultibatchingSolver,
            kwargs={
                "time_limit": int(timeout),
                "mathopt_solver_type": mathopt.SolverType.GSCIP,
            },
        ),
        mathopt_short=SubBrick(
            cls=MathOptMultibatchingSolver,
            kwargs={
                "time_limit": int(timeout),
                "mathopt_solver_type": mathopt.SolverType.GSCIP,
                "restrict_to_shortest_paths": True,
                "shortest_path_tolerance": 1.2,
            },
        ),
    )
    return c


def subbricks_third_step(timeout, verbose: bool = False):
    return dict(
        cp_no_delta=SubBrick(
            CpsatMultibatchingSolver,
            kwargs=dict(
                modeling=ModelingMultiBatch.UNIT_FLOW,
                time_limit=int(timeout),
                ortools_cpsat_solver_kwargs=dict(log_search_progress=verbose),
                parameters_cp=p,
            ),
            kwargs_from_solution={
                "max_trips": lambda sol: sol.problem.get_max_nb_trips(sol)
            },
        ),
        cp_delta=SubBrick(
            CpsatMultibatchingSolver,
            dict(
                modeling=ModelingMultiBatch.UNIT_FLOW,
                time_limit=int(timeout),
                ortools_cpsat_solver_kwargs=dict(log_search_progress=verbose),
                parameters_cp=p,
                delta_to_solution=2,
            ),
            kwargs_from_solution={"solution": lambda sol: sol},
        ),
        milp_no_delta=SubBrick(
            GurobiMultibatchingSolverUnitFlow,
            kwargs=dict(time_limit=int(timeout)),
            kwargs_from_solution={
                "max_trips_per_link": lambda sol: sol.problem.get_max_nb_trips(sol)
            },
        ),
        milp_delta=SubBrick(
            GurobiMultibatchingSolverUnitFlow,
            dict(time_limit=int(timeout), delta_to_solution=2),
            kwargs_from_solution={"solution": lambda sol: sol},
        ),
        mathopt_no_delta=SubBrick(
            MathOptMultibatchingSolverUnitFlow,
            kwargs=dict(
                time_limit=int(timeout), mathopt_solver_type=mathopt.SolverType.GSCIP
            ),
            kwargs_from_solution={
                "max_trips_per_link": lambda sol: sol.problem.get_max_nb_trips(sol)
            },
        ),
        mathopt_delta=SubBrick(
            MathOptMultibatchingSolverUnitFlow,
            dict(
                time_limit=int(timeout),
                mathopt_solver_type=mathopt.SolverType.GSCIP,
                delta_to_solution=2,
            ),
            kwargs_from_solution={"solution": lambda sol: sol},
        ),
    )


def configs_direct_solve(timeout):
    c = dict()
    c["flow-milp"] = SolverConfig(
        GurobiMultibatchingSolverUnitFlow, {"time_limit": timeout}
    )
    c["flow-cpsat"] = SolverConfig(
        CpsatMultibatchingSolver,
        {
            "modeling": ModelingMultiBatch.UNIT_FLOW,
            "time_limit": timeout,
            "parameters_cp": p,
        },
    )
    return c


def config_2step(timeout, add_asp: bool = True):
    c = {}
    flows = subbricks_flow(int(0.75 * timeout))
    flows_full_timeout = subbricks_flow(timeout)
    packings = subbricks_packing(int(0.25 * timeout), add_asp=add_asp)
    for flow_method in flows:
        for packing_method in packings:
            if packing_method == "greedy":
                c[(flow_method, packing_method)] = SolverConfig(
                    cls=TwoStepMultibatchingSolver,
                    kwargs={
                        "flow_solver": flows_full_timeout[flow_method],
                        "packing_solver": packings[packing_method],
                    },
                )
            else:
                c[(flow_method, packing_method)] = SolverConfig(
                    cls=TwoStepMultibatchingSolver,
                    kwargs={
                        "flow_solver": flows[flow_method],
                        "packing_solver": packings[packing_method],
                    },
                )
    return c


def config_3steps(timeout, verbose=False):
    c = config_2step(timeout // 2)
    c_3steps = subbricks_third_step(timeout // 2, verbose=verbose)
    configs = {}
    for flow in ["cp", "milp", "cp_short", "milp_short"]:
        for third_step in c_3steps:
            config: SolverConfig = c[(flow, "greedy")]
            configs[flow + "-greedy-" + third_step] = SolverConfig(
                cls=SequentialMetasolver,
                kwargs=dict(
                    list_subbricks=[
                        SubBrick(config.cls, config.kwargs),
                        c_3steps[third_step],
                    ]
                ),
            )
    return configs


def all_configs(
    timeout,
    add_1step: bool = True,
    add_2step: bool = True,
    add_asp: bool = True,
    add_3steps: bool = True,
    verbose: bool = False,
):
    c = dict()
    if add_1step:
        c.update(configs_direct_solve(timeout))
    if add_2step:
        c.update(config_2step(timeout, add_asp=add_asp))
    if add_3steps:
        c.update(config_3steps(timeout, verbose))
    return c


if __name__ == "__main__":
    configs = all_configs(10)
    print(len(configs))
    print(configs.keys())
