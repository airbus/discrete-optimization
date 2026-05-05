#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from collections import Counter, defaultdict
from typing import Any, Optional

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.binpack.problem import (
    BinPackProblem,
    BinPackSolution,
    ItemBinPack,
)
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportLink,
)
from discrete_optimization.multibatching.solvers import MultibatchingSolver

logger = logging.getLogger(__name__)


class PackingSubproblemSolver(MultibatchingSolver):
    base_solution: MultibatchingSolution
    problem: MultibatchingProblem

    def init_from_solution(self, solution: MultibatchingSolution):
        self.base_solution = solution

    def helper_analyse_function(self):
        nb_trips = {}
        flow_per_link = {}
        for x in self.base_solution.list_flows:
            if x.transport_link not in nb_trips:
                nb_trips[x.transport_link] = 0
                flow_per_link[x.transport_link] = defaultdict(lambda: 0)
            nb_trips[x.transport_link] += x.nb_packing
            for p in x.product_packing:
                flow_per_link[x.transport_link][p] += (
                    x.product_packing[p] * x.nb_packing
                )
        for x in flow_per_link:
            for p in flow_per_link[x]:
                flow_per_link[x][p] = round(flow_per_link[x][p])
        return nb_trips, flow_per_link

    def analyse_solution(self, new_solution: MultibatchingSolution):
        nb_trips_per_link_base_solution, flow_per_link = self.helper_analyse_function()
        nb_trips_per_link_new = defaultdict(lambda: 0)
        for flow in new_solution.list_flows:
            link = flow.transport_link
            nb_trips_per_link_new[link] += flow.nb_packing
        changes = []
        for link in nb_trips_per_link_new:
            if nb_trips_per_link_base_solution[link] != nb_trips_per_link_new[link]:
                changes.append(
                    (
                        link,
                        flow_per_link[link],
                        nb_trips_per_link_new[link],
                        nb_trips_per_link_base_solution[link],
                    )
                )
        return changes


class CpsatPackingSubproblem(OrtoolsCpSatSolver, PackingSubproblemSolver):
    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        # self.modeling: ModelingMultiBatch = None
        self.base_solution = None
        self.scaling_factor = kwargs.get("scaling_factor", 10**3)

    def init_model(self, **args: Any) -> None:
        if self.base_solution is None:
            if "solution" in args:
                self.init_from_solution(solution=args["solution"])
        model = cp_model.CpModel()
        solution = self.base_solution
        transport_links = [x.transport_link for x in solution.list_flows]
        transport_link_to_index = {
            transport_links[i]: i for i in range(len(transport_links))
        }
        flow_per_link: dict[TransportLink, dict[Product, int]] = {
            x.transport_link: {
                p: round(x.product_packing[p] * x.nb_packing) for p in x.product_packing
            }
            for x in solution.list_flows
        }
        sum_size_per_link = {
            x: sum([p.size * flow_per_link[x][p] for p in flow_per_link[x]])
            for x in flow_per_link
        }
        approx_num_trip_per_link = {
            x: math.ceil(sum_size_per_link[x] / x.transport_type.capacity)
            for x in flow_per_link
        }
        # --- Determine Max Trips per Link for Variable Indexing ---
        # This ensures we create enough variables for the maximum number of trips any link might have.
        packed_on_trip = defaultdict(lambda: defaultdict(dict))
        # is_trip_used[link_obj][trip_idx]: Binary variable, 1 if trip is used, 0 otherwise
        is_trip_used = defaultdict(lambda: {})
        for link in transport_links:
            # Number of trips to actually attempt packing into for this specific link
            # Use the number of trips allocated by the main model.
            num_trips_for_this_link = 2 * approx_num_trip_per_link[link]
            # Only create variables if trips are allocated to avoid unnecessary complexity
            if num_trips_for_this_link > 0:
                for t_idx in range(num_trips_for_this_link):
                    is_trip_used[link][t_idx] = model.NewBoolVar(
                        f"is_trip_used_{transport_link_to_index[link]}_{t_idx}"
                    )
                    for product in flow_per_link[link]:
                        # Upper bound for product units in a single trip
                        ub_per_product = int(
                            link.transport_type.capacity // product.size
                        )
                        packed_on_trip[link][t_idx][product] = model.NewIntVar(
                            lb=0,
                            ub=ub_per_product,
                            name=f"packed_{transport_link_to_index[link]}_trip{t_idx}_P{product.id}",
                        )
        # --- Constraints ---
        for link in transport_links:
            # 1. Total Flow Fulfillment (per link, per product):
            for product in flow_per_link[link]:
                total_product_flow_on_link = flow_per_link[link][product]
                model.Add(
                    sum(
                        packed_on_trip[link][t_idx][product]
                        for t_idx in packed_on_trip[link]
                    )
                    == total_product_flow_on_link
                )
            # 2. Trip Capacity Constraint (per individual trip for this link):
            for t_idx in packed_on_trip[link]:
                # Total volume packed on this specific trip
                packed_volume_on_this_trip = sum(
                    int(product.size) * packed_on_trip[link][t_idx][product]
                    for product in packed_on_trip[link][t_idx]
                )
                # If trip is used, its volume must respect capacity. If not used, its volume must be 0.
                model.Add(
                    packed_volume_on_this_trip <= int(link.transport_type.capacity)
                )
                model.Add(packed_volume_on_this_trip > 0).OnlyEnforceIf(
                    is_trip_used[link][t_idx]
                )
                model.Add(packed_volume_on_this_trip == 0).OnlyEnforceIf(
                    is_trip_used[link][t_idx].Not()
                )
        nb_trips_link = {
            link: sum(is_trip_used[link][i] for i in is_trip_used[link])
            for link in is_trip_used
        }
        transport_cost = self.scaling_factor * sum(
            [
                nb_trips_link[link] * link.distance * link.transport_type.cost
                for link in is_trip_used
            ]
        )
        emission_cost = self.scaling_factor * sum(
            [
                nb_trips_link[link] * link.distance * link.transport_type.emissions
                for link in is_trip_used
            ]
        )
        model.Minimize(transport_cost + emission_cost)
        self.cp_model = model
        self.variables["packed_on_trip"] = packed_on_trip
        self.variables["trip_used"] = is_trip_used
        self.variables["objs"] = {
            "transport": transport_cost,
            "emission": emission_cost,
        }

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> MultibatchingSolution:
        logger.info(f"Obj {cpsolvercb.ObjectiveValue()}")
        if "objs" in self.variables:
            for obj in self.variables["objs"]:
                if obj == "capital":
                    continue
                try:
                    logger.info(
                        f"{obj}: {cpsolvercb.value(self.variables['objs'][obj])}"
                    )
                except Exception as e:
                    pass
        list_flows = []
        for link in self.variables["packed_on_trip"]:
            packings = []
            for i in sorted(self.variables["trip_used"][link]):
                used = self.variables["trip_used"][link][i]
                if cpsolvercb.value(used) > 0:
                    current_packing = set()
                    for product in self.variables["packed_on_trip"][link][i]:
                        val = cpsolvercb.value(
                            self.variables["packed_on_trip"][link][i][product]
                        )
                        if val > 0:
                            current_packing.add((product, int(val)))
                    cpack = frozenset(current_packing)
                    packings.append(cpack)
            if len(packings) > 0:
                c = Counter(packings)
                for key, val in c.items():
                    packing_transport = PackingTransport(
                        transport_link=link,
                        product_packing={x[0]: x[1] for x in key},
                        nb_packing=val,
                    )
                    list_flows.append(packing_transport)
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)


class GreedyPackingForMultibatching(PackingSubproblemSolver):
    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        single_batching: bool = False,
        **kwargs: Any,
    ) -> ResultStorage:
        solution = self.base_solution
        transport_link_flows: dict[TransportLink, list[dict[Product, int]]] = (
            defaultdict(lambda: [])
        )
        aggregated_link_flow: dict[TransportLink, dict[Product, int]] = defaultdict(
            lambda: dict()
        )
        for flow in solution.list_flows:
            tl = flow.transport_link
            nb_packing = flow.nb_packing
            transport_link_flows[tl].append(
                {p: nb_packing * flow.product_packing[p] for p in flow.product_packing}
            )
            if tl not in aggregated_link_flow:
                aggregated_link_flow[tl] = {}
            for p in flow.product_packing:
                if flow.product_packing[p] > 0:
                    if p not in aggregated_link_flow[tl]:
                        aggregated_link_flow[tl][p] = 0
                    aggregated_link_flow[tl][p] += nb_packing * flow.product_packing[p]
        packings = []
        for tl in aggregated_link_flow:
            packings_for_tl: list[PackingTransport] = []
            capacity = tl.transport_type.capacity
            current_packing = PackingTransport(
                transport_link=tl, product_packing=defaultdict(lambda: 0), nb_packing=1
            )
            current_packing.current_size = 0
            packings_for_tl.append(current_packing)
            for p in aggregated_link_flow[tl]:
                for _ in range(round(aggregated_link_flow[tl][p])):
                    success = False
                    for j in range(len(packings_for_tl)):
                        if single_batching:
                            if (
                                len(packings_for_tl[j].product_packing) > 0
                                and p not in packings_for_tl[j].product_packing
                            ):
                                continue
                        if packings_for_tl[j].current_size + p.size <= capacity:
                            packings_for_tl[j].product_packing[p] += 1
                            packings_for_tl[j].current_size += p.size
                            success = True
                            break
                    if not success:
                        current_packing = PackingTransport(
                            transport_link=tl,
                            product_packing=defaultdict(lambda: 0),
                            nb_packing=1,
                        )
                        current_packing.product_packing[p] += 1
                        current_packing.current_size = p.size
                        packings_for_tl.append(current_packing)

            packings.extend(packings_for_tl)
        sol = MultibatchingSolution(problem=self.problem, list_flows=packings)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])


class PackingViaBinPacking(PackingSubproblemSolver):
    # from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
    hyperparameters = [
        CategoricalHyperparameter(
            "round_flow", choices=["int", "round"], default="round"
        ),
        SubBrickHyperparameter(
            name="bin_packing_solver",
            choices=[
                CpSatBinPackSolver,
                # AspBinPackingSolver
            ],
            default=SubBrick(
                cls=CpSatBinPackSolver,
                kwargs=dict(
                    modeling=ModelingBinPack.SCHEDULING,
                    parameters_cp=ParametersCp.default_cpsat(),
                ),
            ),
        ),
    ]

    def __init__(
        self,
        problem: MultibatchingProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **args,
    ):
        super().__init__(problem, params_objective_function, **args)
        self.cache = dict()

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit_per_link=2,
        **kwargs: Any,
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        bin_packing_solver: SubBrick = kwargs["bin_packing_solver"]
        solution = self.base_solution
        transport_link_flows: dict[TransportLink, list[dict[Product, int]]] = (
            defaultdict(lambda: [])
        )
        aggregated_link_flow: dict[TransportLink, dict[Product, int]] = defaultdict(
            lambda: dict()
        )
        for flow in solution.list_flows:
            tl = flow.transport_link
            nb_packing = flow.nb_packing
            transport_link_flows[tl].append(
                {p: nb_packing * flow.product_packing[p] for p in flow.product_packing}
            )
            if tl not in aggregated_link_flow:
                aggregated_link_flow[tl] = {}
            for p in flow.product_packing:
                if flow.product_packing[p] > 0:
                    if p not in aggregated_link_flow[tl]:
                        aggregated_link_flow[tl][p] = 0
                    aggregated_link_flow[tl][p] += nb_packing * flow.product_packing[p]
                    if (
                        flow.product_packing[p] > 0
                        and int(nb_packing * flow.product_packing[p]) == 0
                    ):
                        logger.info(f"Potential issue on flow of product {p.name}")
        for tl in aggregated_link_flow:
            for p in aggregated_link_flow[tl]:
                if kwargs["round_flow"] == "int":
                    aggregated_link_flow[tl][p] = int(aggregated_link_flow[tl][p])
                if kwargs["round_flow"] == "round":
                    aggregated_link_flow[tl][p] = int(
                        round(aggregated_link_flow[tl][p])
                    )
        all_packings = []
        for tl in aggregated_link_flow:
            capacity = tl.transport_type.capacity
            key = (
                tl,
                frozenset(
                    [(p, aggregated_link_flow[tl][p]) for p in aggregated_link_flow[tl]]
                ),
            )
            list_items = []
            index_to_product = []
            cur_index = 0
            for p in aggregated_link_flow[tl]:
                list_items.extend(
                    [
                        ItemBinPack(cur_index + j, int(p.size))
                        for j in range(aggregated_link_flow[tl][p])
                    ]
                )
                cur_index += aggregated_link_flow[tl][p]
                index_to_product.extend([p for _ in range(aggregated_link_flow[tl][p])])
            if len(list_items) == 0:
                continue
            if key in self.cache:
                logger.info(f"Found cached solution")
                (sol, nb_bins) = self.cache[key]
            else:
                problem_bin_pack = BinPackProblem(
                    list_items=list_items, capacity_bin=int(capacity)
                )
                solver = bin_packing_solver.cls(problem=problem_bin_pack)
                solver.init_model(**bin_packing_solver.kwargs)
                res = solver.solve(
                    time_limit=time_limit_per_link, **bin_packing_solver.kwargs
                )
                sol: BinPackSolution = res[-1][0]
                nb_bins = problem_bin_pack.evaluate(sol)["nb_bins"]
                self.cache[key] = (sol, nb_bins)
                logger.info(f"Found {nb_bins} bins")
            packings = [
                PackingTransport(
                    transport_link=tl,
                    product_packing=defaultdict(lambda: 0),
                    nb_packing=1,
                )
                for i in range(nb_bins)
            ]
            for i in range(len(sol.allocation)):
                bin_i = sol.allocation[i]
                packings[bin_i].product_packing[index_to_product[i]] += 1
            all_packings.extend(packings)
        sol = MultibatchingSolution(problem=self.problem, list_flows=all_packings)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
