import logging
from collections import defaultdict
from typing import Any, Optional, List, Dict

import clingo
from clingo.ast import parse_string, ProgramBuilder
from clingcon import ClingconTheory
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportLink
)

import sys
import pandas as pd
import os
import csv
import time
import math
import networkx as nx

logger = logging.getLogger(__name__)

# Logic Program Content
ASP_MODEL_CONTENT = """
#const cap_size_divide = 100.
#const max_freq = 30.

ship(tR8128955e62abfa10;tR37f7026257afb729;tR04573f34dc9e3dfe;tRfbf64cc727baf133;tR464448f7af6f430b;tRab2d245aa3a02226;tR57d1727d1540f494).
harbor(lOCac0636559af29d3c;lOC03f36a95cb36cdf5;lOCaf8ae38dce404ace;lOC67caa3aa77b472a5;lOCf72ff506715c40bf;lOC12d769ce3d0b8ae5;lOC61275990ac7ab35f;lOCa38e748fb941ca99;lOC7401ba2d22d623a1;lOC379bb10472d4a147;lOC33caf5d710aa7bbc;lOCf7b0dbbac96e2b31;lOCcbaad3cf4a6507e5;lOC2df188aa7389c304;lOC3696c4b6a4f4e9de).

totalSupply(P, Max) :- Max = #sum {V, L : demandOffer(P,L,V), V > 0}, product(P).

hasDemand(P, L) :- demandOffer(P, L, V), V < 0.
hasOffer(P, L)  :- demandOffer(P, L, V), V > 0.

demand(P, L, Val * -1) :- demandOffer(P, L, Val), Val < 0.
demand(P, L, 0)        :- product(P), location(L), not hasDemand(P, L).

offer(P, L, Val) :- demandOffer(P, L, Val), Val > 0.
offer(P, L, 0)   :- product(P), location(L), not hasOffer(P, L).

requiredNet(P, L, Net) :- demand(P,L,D), offer(P,L,O), Net = D - O.
requiredNet(P, L, 0) :- not demand(P,L,_), not offer(P,L,_), location(L), product(P).
noShipBetweenHarbors(From, To, TR) :- harbor(From), harbor(To), not ship(TR), transportResource(TR).

flow(From, To, TR, P) :- possibleFlow(From, To, TR,_), 
not noShipBetweenHarbors(From, To, TR),
tRCapSmall(TR, Cap), productSizeSmall(P,S).

tRCapSmall(TR,Cap) :- transportCapacity(TR,CapB), Cap=CapB/cap_size_divide.
productSizeSmall(P,S) :- productSize(P,SB), S = SB/cap_size_divide.

&dom { 0..Max } = load(From, To, TR, P) :- flow(From, To, TR,P), totalSupply(P, Max).

&sum { load(From, L, TR, P): flow(From, L, TR, P)} = flow_in(P, L) :-  location(L), product(P).
&sum { load(L, To, TR, P): flow(L, To, TR, P)} = flow_out(P, L) :-  location(L), product(P).
&sum { flow_in(P,L); -1 * flow_out(P,L)} = Net :- requiredNet(P, L, Net).

activeFlow(From,To,TR,P) :-  &sum { load(From, To, TR, P):flow(From, To, TR, P)} > 0,flow(From, To, TR, P).

freq(1..max_freq).
1{routeFreq(From, To, TR, Freq, C*Freq) : freq(Freq), tRCapSmall(TR,C) }1 :- activeFlow(From,To,TR,_).

% upper bound
&sum{S*load(F, T, TR, P) : flow(F, T, TR, P),productSizeSmall(P,S)} <= TotalCap :-
routeFreq(F, T, TR, Freq, TotalCap).
% lower bound
:- &sum{S*load(F, T, TR, P) :flow(F, T, TR, P),productSizeSmall(P,S)} <= Minus1 , 
routeFreq(F, T, TR, Freq, TotalCap), tRCapSmall(TR,C), Minus1=TotalCap-C, Freq>1.

:- &sum{flow_in(P,L)}>0, demandOffer(P,L,N), N>0.
:- &sum{flow_out(P,From)} > Max, totalSupply(P, Max), flow(From, _, _,P).
:- &sum{flow_in(P,To)} > Max, totalSupply(P, Max), flow(_, To, _,P).

smallDistance(From,To,TR,D) :- flow(From,To,TR,_), route(From,To,TR,Dis,_), D=Dis/10.
smallEmission(TR,E) :- transportCO2(TR,Em), route(From,To,TR,_,_), E=Em/10.

#minimize{(D*Freq*E)/10000,From,To,TR: routeFreq(From, To, TR, Freq, _),smallDistance(From, To, TR,D),smallEmission(TR,E)}.
#minimize{(C*Freq),From,To,TR: routeFreq(From, To, TR, Freq, _),route(From, To, TR,D, C)}.

"""


class ClingconMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem

    all_dataframes = {}
    modelIndex = 0
    start_solving = 0

    modelAtomTemplates = [
                {'name':'totalSupply', 'filter':lambda s: s.name=='totalSupply', 'columns':['product' ,'amount']},
                {'name':'route', 'filter':lambda s: s.name=='route', 'columns':['idx','from' ,'to', 'transportresource', 'distance','cost']},
                {'name':'routeCostTotal', 'filter':lambda s: s.name=='routeCostTotal', 'columns':['from' ,'to', 'transportresource', 'cost']},
                {'name':'routeFreq','filter':lambda s: s.name=='routeFreq', 'columns':['from', 'to', 'transportresource','freq','maxcapacity']}, #,'maxcapacity', 'minus1'
                {'name':'activeFlow','filter':lambda s: s.name=='activeFlow', 'columns':['from', 'to', 'transportresource',"product"]},
                {'name':'demandOffer','filter':lambda s: s.name=='demandOffer', 'columns':['product', 'location', 'amount']},
                {'name':'requiredNet','filter':lambda s: s.name=='requiredNet', 'columns':['product', 'location', 'amount']},
                {'name':'productTR','filter':lambda s: s.name=='productTR', 'columns':['product', 'transportresource']},
                {'name':'productSize','filter':lambda s: s.name=='productSize', 'columns':['product', 'size']},
                {'name':'possibleFlow','filter':lambda s: s.name=='possibleFlow', 'columns':['from', 'to', 'transportresource','product']},
                {'name':'transportCapacity','filter':lambda s: s.name=='transportCapacity', 'columns':['transportresource', 'capacity']},
                {'name':'transportCost','filter':lambda s: s.name=='transportCost', 'columns':['transportresource', 'transportationCost']},
                {'name':'transportCO2','filter':lambda s: s.name=='transportCO2', 'columns':['transportresource', 'CO2']},
                {'name':'productOnFlow','filter':lambda s: s.name=='productOnFlow', 'columns':['from', 'to', 'transportresource','product','amount']}
            ]

    modelAggregateTemplates = [ 
                            {'name':'route_freq_cap','filter':lambda s: s.name=='route_freq_cap', 'columns':['from', 'to', 'transportresource','freq', 'total']},
                             {'name':'flow_out', 'filter':lambda s: s.name=='flow_out', 'columns':['product' ,'location', 'amount']},
                            {'name':'flow_in', 'filter':lambda s: s.name=='flow_in', 'columns':['product' ,'location', 'amount']},
                            {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['product' ,'location', 'amount']},
                            {'name':'route_cost_total', 'filter':lambda s: s.name=='route_cost_total', 'columns':['from' ,'to', 'transportresource', 'cost']},
                            {'name':'load', 'filter':lambda s: s.name=='load', 'columns':['from' ,'to', 'transportresource', 'product', 'amount']},
                            {'name':'active_flow', 'filter':lambda s: s.name=='active_flow', 'columns':['from' ,'to', 'transportresource','val']}
                            ]

    def __init__(
            self,
            problem: MultibatchingProblem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            **kwargs: Any, 
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ctl: Optional[clingo.Control] = None
        self.theory: Optional[ClingconTheory] = None
        self.name_to_location: Dict[str, Any] = {}
        self.name_to_transport_type: Dict[str, Any] = {}
        self.name_to_product: Dict[str, Any] = {}
        self.route_key_to_link: Dict[tuple, TransportLink] = {}

    def sanitize(self, s: str) -> str:
            s = str(s).replace(".", "").replace(" ", "").replace("-", "").replace("_", "")
            return s[0].lower() + s[1:]

    def sanitize_num(self, val) -> int:
        """Ensures input is a valid integer for Clingcon constraints."""
        try:
            return int(float(val)) # float handles "10.0" strings better than int()
        except (ValueError, TypeError):
            print("ValueError/ TypeError", ValueError, TypeError)

    def precompute_valid_links(self, tolerance: float = 0.0) -> Dict[int, set]:
        """
        Robust Heuristic: Identifies which transport links are relevant for each product.

        It builds a specific graph G_p for each product P, containing only
        the TransportLinks compatible with P (i.e. link.transport_type in p.valid_transports).

        A link (u, v) is relevant for product P if there exists ANY pair of
        (Source s, Sink d) for P such that:
           dist_p(s, u) + link_len(u, v) + dist_p(v, d) <= (1 + tolerance) * dist_p(s, d)
        """
        valid_links_per_product = defaultdict(set)

        for p in self.problem.products:
            # 1. Identify Producers and Consumers for this specific product
            sources = [l.id for l in self.problem.locations if l.net_supply.get(p, 0) > 1e-6]
            sinks = [l.id for l in self.problem.locations if l.net_supply.get(p, 0) < -1e-6]

            if not sources or not sinks:
                continue

            # 2. Build the Product-Specific Graph G_p
            # Only include links where the transport type is allowed for this product
            G_p = nx.DiGraph()

            # We also map (u, v) -> list of (link_index, weight)
            # because there might be multiple transport modes between u and v (e.g. Truck vs Train)
            # and we need to check them individually.
            edges_to_indices = defaultdict(list)

            for i, tl in enumerate(self.problem.transport_links):
                # KEY CHECK: Is this transport type valid for this product?
                # Also check capacity constraints if relevant (size <= capacity)
                if (tl.transport_type in p.valid_transports and
                        p.size <= tl.transport_type.capacity):

                    u, v = tl.location_l1.id, tl.location_l2.id
                    w = 1  # tl.distance

                    # For NetworkX, we usually want the *best* weight between u and v
                    # if we just want shortest path distance.
                    # But we must store all parallel edges to evaluate them later.
                    edges_to_indices[(u, v)].append((i, w))

                    # Update graph with the shortest edge between u and v found so far
                    if G_p.has_edge(u, v):
                        if w < G_p[u][v]['weight']:
                            G_p[u][v]['weight'] = w
                    else:
                        G_p.add_edge(u, v, weight=w)

            # 3. Compute All-Pairs Shortest Paths on G_p
            try:
                # This gives the shortest path distance matrix for valid moves only
                path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_p, weight='weight'))
            except nx.NetworkXNoPath:
                continue  # Disconnected graph for this product

            # 4. Filter Links
            # We iterate over all physically existing links again, but only check valid ones
            for (u, v), link_list in edges_to_indices.items():

                # Pre-check: Can any source reach u?
                valid_sources = [s for s in sources if s in path_lengths and u in path_lengths[s]]
                if not valid_sources:
                    continue

                # Pre-check: Can v reach any sink?
                if v not in path_lengths:
                    continue
                valid_sinks = [d for d in sinks if d in path_lengths[v]]
                if not valid_sinks:
                    continue

                # Check each specific transport mode between u and v
                for (index_tl, w) in link_list:
                    is_useful = False

                    for s in valid_sources:
                        dist_s_u = path_lengths[s][u]

                        for d in valid_sinks:
                            if d not in path_lengths[s]:
                                continue  # Should not happen if graph is connected, but safety check

                            dist_s_d_opt = path_lengths[s][d]
                            dist_v_d = path_lengths[v][d]

                            # Check Path Quality
                            path_via_link = dist_s_u + w + dist_v_d

                            # Relaxed triangle inequality
                            if path_via_link <= dist_s_d_opt * (1.0 + tolerance) + 1e-5:
                                is_useful = True
                                break

                        if is_useful:
                            break

                    if is_useful:
                        valid_links_per_product[p.id].add(index_tl)

        return valid_links_per_product
        

    def processModelAtoms(self, model: clingo.Model):
        path_symbols = []
        for name in [next(iter(d.values())) for d in self.modelAtomTemplates]:
            path_symbols += list(filter(lambda s: s.name == name, model.symbols(atoms=True)))

            with open(f"output/model_{self.modelIndex}/model.txt", "w") as model_file:
                model_file.write(str(model))

        for modelAtomTemplate in self.modelAtomTemplates:
            dataName = modelAtomTemplate['name']
            filteredSymbols = list(filter(modelAtomTemplate['filter'], path_symbols))
            arguments = list(map(lambda s: s.arguments, filteredSymbols))
            df = pd.DataFrame(data=arguments, columns= modelAtomTemplate['columns'])
            self.all_dataframes[dataName] = df

            output_dir = f"output/model_{self.modelIndex}"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(f"{output_dir}/{dataName}.csv", index=False)


    def init_model(self, **kwargs: Any) -> None:
        # Check if we should use shortest path heuristic
        use_shortest_path = kwargs.get("restrict_to_shortest_paths", False)
        sp_tolerance = kwargs.get("shortest_path_tolerance", 0.0)  # 0.0 = Strict, 0.2 = +20% length allowed

        # Default options
        clingo_args = kwargs.get("clingo_args", 
                                 ["--models=0", "--opt-mode=opt", "--parallel-mode=4,split",
                        "--configuration=crafty", "--restart-on-model", "--opt-strategy=bb"])
                              #    [f"--models=0", "--opt-mode=opt", "--configuration=handy", "--parallel-mode=4,split",
                                               #  "--parallel-mode=4,compete",  # Use 4 CPU cores (adjust to your machine) competing against each other
  # "--configuration=jumpy" #,       # Equip each core with a different solving strategy
   # "--restart-on-model",         # THE MAGIC BULLET: Restart the search tree when a better bound is found
  #  "--opt-strategy=bb",          # Use Branch and Bound for cost optimization
                               #                  ])
      #  clingo_args = kwargs.get("clingo_args", 
                               #  [
    #"--parallel-mode=4,split",   # 'split' is better for optimization than 'compete'
    #"--opt-mode=opt",            # Forces search for the best
    #"--restart-on-model",        # CRITICAL: Restarts search with a better bound
    #"--opt-strategy=bb",         # Branch-and-Bound (ensures cost only goes down)
    #"--configuration=many"       # Keeps diverse strategies but they will share bounds
#])
                               #  ["--opt-mode=optN", "-t", "4", "--stats", "--restart-on-model"])
                                 
                                # ["--opt-mode=optN",  "--stats", "--restart-on-model", "--heuristic=Vsids"])
        #"--heuristic=Domain", # Tells clingo to actually use your #heuristic rules
        #"--opt-strategy=bb,dec",
        #"--parallel-mode=4",
        #"--configuration=many"
        #"--restart-on-model",
        #"--rand-freq=0.05", # 5% random moves to escape rigid symmetry rules
        #"--opt-strategy=bb,dec",
        #"--parallel-mode=4"
        #])
        
        
        self.ctl = clingo.Control(clingo_args)
        self.theory = ClingconTheory()

        # 1. Register Clingcon theory
        self.theory.register(self.ctl)

        # 2. Add Logic Program using the rewriter
        # This fixes the "no definition found for theory atom" error
        with ProgramBuilder(self.ctl) as bld:
            # Parse main model
            parse_string(ASP_MODEL_CONTENT, lambda ast: self.theory.rewrite_ast(ast, bld.add))

            # Generate and parse facts
            valid_links_map = None
            if use_shortest_path:
                logger.info(f"Computing valid links (Shortest Path Heuristic, tol={sp_tolerance})...")
                valid_links_map = self.precompute_valid_links(tolerance=sp_tolerance)
                logger.info(f"Valid links computed for shortest path filtering")

            facts_str = self._generate_facts(scale_emission_cost=kwargs.get("scale_emission_cost", 100000),
                                             valid_links_map=valid_links_map)
            logger.debug("Generated ASP Facts:\n" + facts_str)
            parse_string(facts_str, lambda ast: self.theory.rewrite_ast(ast, bld.add))

        # 3. Ground

        start_time_grounding = time.process_time()
        logger.info("Grounding logic program...")
        self.ctl.ground([("base", [])])
        end_time_grounding = time.process_time()
        logger.info(f"Grounding done ({round((end_time_grounding - start_time_grounding) / 60, 2)} minutes).")
    
        # 4. Prepare theory (must happen after grounding)
        self.theory.prepare(self.ctl)

    def solve(
            self,
            callbacks: Optional[List[Callback]] = None,
            time_limit: Optional[float] = None,
            **kwargs: Any
    ) -> ResultStorage:

        if self.ctl is None:
            self.init_model(**kwargs)

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        res = self.create_result_storage()

        def on_model_wrapper(model: clingo.Model):
            self.modelIndex += 1
            logger.info(f"Model [{self.modelIndex}] found after {round((time.process_time() - self.start_solving) / 60, 2)} minutes of solving.")

            #self.processModelAtoms(model)
            #self.compute_active_flows()
            #self.compute_costs()
            sol = self.retrieve_solution(model)
            fit = self.aggreg_from_sol(sol)

            res.append((sol, fit))
            stopping = callbacks_list.on_step_end(step=len(res), res=res, solver=self)

            #print("total costs (estimate without frequencies):", fit, f"({fit:.2e})")
            #print('satisfy', self.problem.satisfy(sol))
            #print("Check flows ", self.problem.check_flows(sol))
            #print("stopping", stopping)
            
            return not stopping
        

        logger.info("Starting Clingcon solve...")

        # Timeout handling via async handle
        self.start_solving = time.process_time()
        if time_limit is None:
            self.ctl.solve(on_model=on_model_wrapper)
        else:
            with self.ctl.solve(on_model=on_model_wrapper, async_=True) as handle:
                finished = handle.wait(time_limit)
                if not finished:
                    logger.info(f"Time limit of {time_limit}s reached.")
                    handle.cancel()
        if len(res) > 0:
            self.status_solver = StatusSolver.SATISFIED
        else:
            self.status_solver = StatusSolver.UNSATISFIABLE

        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def retrieve_solution(self, model: clingo.Model) -> MultibatchingSolution:
        logger.info(f"Optimality proven : {model.optimality_proven}")
        assignment = list(self.theory.assignment(model.thread_id))
        flows_aggregation = defaultdict(lambda: defaultdict(int))
        #print([f'{key}={val}' for key, val in assignment if key.name == "active_flow" and val != 0])
        for template in self.modelAggregateTemplates:
            df = pd.DataFrame(data={}, columns=template["columns"])
            self.all_dataframes[template["name"]] = df

        for name, value in assignment:
            if name.name == "load" and value > 0:
                try:
                    args = name.arguments
                    from_name, to_name, tr_name, p_name = map(str, args)
                    route_key = (from_name, to_name, tr_name)
                    link = self.route_key_to_link.get(route_key)
                    product = self.name_to_product.get(p_name)
                    if link and product:
                        flows_aggregation[link][product] += value
                        df = self.all_dataframes["load"]
                        df.loc[len(df)]  = {"from": from_name, "to":to_name , "transportresource": tr_name, "product":p_name, "amount":value}
                        
                    output_dir = f"output/model_{self.modelIndex}"
                    os.makedirs(output_dir, exist_ok=True)
                    df.to_csv(f"{output_dir}/load.csv", index=False)

                except Exception as e:
                    logger.error(f"Error parsing variable {name}: {e}")
            
            if name.name == "flow_in" and value > 0:
                df = self.all_dataframes["flow_in"]
                args = name.arguments
                p_name, to_name = map(str, args)
                product = self.name_to_product.get(p_name)
                df.loc[len(df)] = {"product": p_name, "location":to_name, "amount":value}
            
            if name.name == "flow_out" and value > 0:
                df = self.all_dataframes["flow_out"]
                args = name.arguments
                p_name, from_name = map(str, args)
                product = self.name_to_product.get(p_name)
                df.loc[len(df)] = {"product": p_name, "location":from_name, "amount":value}

            if name.name == "route_cost_total" and value > 0:
                df = self.all_dataframes["route_cost_total"]
                args = name.arguments 
                from_name, to_name, tr_name = map(str, args)
               
                df.loc[len(df)]= {"from": from_name, "to":to_name, "transportresource": tr_name, "cost":value}
                
            if name.name == "req" and value != 0:
                df = self.all_dataframes["req"]
                args = name.arguments
                p_name, loc_name  = map(str, args)
                df.loc[len(df)] = {"product": p_name, "location":loc_name, "amount":value}

        list_flows = []

        for link, product_quantities in flows_aggregation.items():
            #packing = {p: q for p, q in product_quantities.items() if q > 0}
            from_loc = self.sanitize(link.location_l1.name)
            to_loc = self.sanitize(link.location_l2.name)
            mode = self.sanitize(link.transport_type.name)
            
            #if self.all_dataframes["routeFreq"]:
            #    routeFreq_df = self.all_dataframes["routeFreq"]
            #    routeFreq_df['from','to','transportresource'] = routeFreq_df['from'].astype(str)
            #    routeFreq_df['from','to','transportresource'] = routeFreq_df['to'].astype(str)
            #    routeFreq_df['from','to','transportresource'] = routeFreq_df['transportresource'].astype(str)
            #    mask = (routeFreq_df['from'].astype(str) == from_loc) & \
            #        (routeFreq_df['to'].astype(str) == to_loc) & \
            #        (routeFreq_df['transportresource'].astype(str) == mode)
            #    
            #    freq = (routeFreq_df.loc[mask, 'freq'].values[0]).number
            #else:
            freq = 1
            packing = {}
            for p, q in product_quantities.items():
                if q > 0:
                    items_per_trip = q #math.ceil(q / freq) 
                    if items_per_trip > 0:
                        packing[p] = items_per_trip
            list_flows.append(PackingTransport(link, packing, freq))

        sol = MultibatchingSolution(problem=self.problem, list_flows=list_flows)
        return sol 
        
    def _generate_facts(self, scale_emission_cost: float=100000., valid_links_map: Dict[int, set] = None) -> str:
        # Same generation logic as before, ensure it returns a valid ASP string
        facts = []
        possibleFlow_facts = []

        # Locations
        self.name_to_location = {}
        for loc in self.problem.locations:
            s_name = self.sanitize(loc.name)
            self.name_to_location[s_name] = loc
            facts.append(f"location({s_name}).")

        # Transport Resources
        self.name_to_transport_type = {}
        for tt in self.problem.transport_types:
            s_name = self.sanitize(tt.name if tt.name else f"tr{tt.id}")
            self.name_to_transport_type[s_name] = tt
            facts.append(f"transportResource({s_name}).")
            # 1000000
            co2 = int(tt.emissions * scale_emission_cost)
            facts.append(f"transportCO2({s_name},{co2}).")
            facts.append(f"transportCost({s_name},{max(0,self.sanitize_num(tt.cost))}).")
            scaled_cap = max(0, int(tt.capacity))
            facts.append(f"transportCapacity({s_name},{scaled_cap}).")
            facts.append(f"transportSpeed({s_name},{self.sanitize_num(tt.speed)}).")

        # Routes
        self.route_key_to_link = {}
        distNew = 6000
        for idx, tl in enumerate(self.problem.transport_links):
            source = self.sanitize(tl.location_l1.name)
            dest = self.sanitize(tl.location_l2.name)
            tr_name = self.sanitize(tl.transport_type.name if tl.transport_type.name else f"tr{tl.transport_type.id}")
            self.route_key_to_link[(source, dest, tr_name)] = tl
            dist = int(tl.distance)
            cost = int(self.sanitize_num(tl.transport_type.cost) * dist)
            facts.append(f"route({source},{dest},{tr_name},{dist},{cost}).")

        # Products
        self.name_to_product = {}
        for p in self.problem.products:
            s_name = self.sanitize(p.name if p.name else f"p{p.id}")
            self.name_to_product[s_name] = p
            facts.append(f"product({s_name}).")
            scaled_size = int(p.size)
            facts.append(f"productSize({s_name},{scaled_size}).")
            for valid_tr in p.valid_transports:
                tr_s_name = self.sanitize(valid_tr.name if valid_tr.name else f"tr{valid_tr.id}")
                facts.append(f"productTR({s_name},{tr_s_name}).")

        # Generate possibleFlow facts (with optional filtering)
        if valid_links_map is not None:
            logger.info("Generating possibleFlow facts with shortest path filtering...")
            for p in self.problem.products:
                p_name = self.sanitize(p.name if p.name else f"p{p.id}")
                valid_link_indices = valid_links_map.get(p.id, set())
                logger.debug(f"Product {p_name}: {len(valid_link_indices)} valid links")

                for index_tl in valid_link_indices:
                    tl = self.problem.transport_links[index_tl]
                    source = self.sanitize(tl.location_l1.name)
                    dest = self.sanitize(tl.location_l2.name)
                    tr_name = self.sanitize(tl.transport_type.name if tl.transport_type.name else f"tr{tl.transport_type.id}")
                    possibleFlow_facts.append(f"possibleFlow({source},{dest},{tr_name},{p_name}).")

            logger.info(f"Generated {len(possibleFlow_facts)} possibleFlow facts using shortest path heuristic")

            # Write possibleFlow facts to separate file
            possibleFlow_file = os.path.join(os.path.dirname(__file__), 'possibleFlows_shortestPath.lp')
            with open(possibleFlow_file, 'w') as f:
                f.write("\n".join(possibleFlow_facts))
            logger.info(f"possibleFlow facts written to {possibleFlow_file}")
        else:
            # Generate all possible flows when not using shortest path heuristic
            logger.info("Generating all possibleFlow facts (no filtering)...")
            for p in self.problem.products:
                p_name = self.sanitize(p.name if p.name else f"p{p.id}")
                for tl in self.problem.transport_links:
                    # Check if product can use this transport type
                    if tl.transport_type in p.valid_transports and p.size <= tl.transport_type.capacity:
                        source = self.sanitize(tl.location_l1.name)
                        dest = self.sanitize(tl.location_l2.name)
                        tr_name = self.sanitize(tl.transport_type.name if tl.transport_type.name else f"tr{tl.transport_type.id}")
                        possibleFlow_facts.append(f"possibleFlow({source},{dest},{tr_name},{p_name}).")

            logger.info(f"Generated {len(possibleFlow_facts)} possibleFlow facts (all valid combinations)")

        # Demand/Offers
        for loc in self.problem.locations:
            loc_s_name = self.sanitize(loc.name)
            for p, amount in loc.net_supply.items():
                if amount != 0:
                    p_s_name = self.sanitize(p.name if p.name else f"p{p.id}")
                    facts.append(f"demandOffer({p_s_name},{loc_s_name},{int(amount)}).")

        # Combine all facts (including possibleFlow if generated)
        all_facts = facts + possibleFlow_facts

        with open(os.path.join(os.path.dirname(__file__),'factsASP.lp'), 'w') as file_out:
            file_out.writelines("\n".join(all_facts))
        return "\n".join(all_facts)
    


    def extract_clingo_number(self, val):
        # Case A: It's a Clingo Symbol (has .number attribute)
        if hasattr(val, 'number'):  
            return val.number
        # Case B: It's already a Python int or float
        if isinstance(val, (int, float)):
            return val
        # Case C: It's a string that looks like a number ("6")
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0  # Default fallback if data is bad
            
    def compute_active_flows(self):
        
        #active_flows_df = self.all_dataframes["activeFlow"]
        #print(f"Total active flows (wrt individual product) for model: {active_flows_df.shape[0]}")
        route_freq_df = self.all_dataframes["routeFreq"]
        route_freq_df['freq'] = route_freq_df['freq'].values
        route_freq_df['freq'] = route_freq_df['freq'].apply(self.extract_clingo_number)
        print(f"Total active flows including frequencies for model: {route_freq_df['freq'].sum()}")

    def compute_costs(self):
        

        route_df = self.all_dataframes["route"]
        route_df['cost'] = route_df['cost'].values
        route_df['cost'] = route_df['cost'].apply(self.extract_clingo_number)

        route_df['distance'] = route_df['distance'].values
        route_df['distance'] = route_df['distance'].apply(self.extract_clingo_number)

        route_freq_df = self.all_dataframes["routeFreq"]
        route_freq_df['freq'] = route_freq_df['freq'].values
        route_freq_df['freq'] = route_freq_df['freq'].apply(self.extract_clingo_number)
        
        transportCost_df = self.all_dataframes["transportCost"]
        transportCost_df['transportationCost'] = transportCost_df['transportationCost'].values
        transportCost_df['transportationCost'] = transportCost_df['transportationCost'].apply(self.extract_clingo_number)

        co2Cost_df = self.all_dataframes["transportCO2"]
        co2Cost_df['CO2'] = co2Cost_df['CO2'].values
        co2Cost_df['CO2'] = co2Cost_df['CO2'].apply(self.extract_clingo_number)

        merged_df1 = pd.merge(
            route_freq_df, 
            route_df, 
            on=['from', 'to', 'transportresource'], 
            how='inner'
        )

        merged_df1['total_transportation_costs'] = merged_df1['freq'] * merged_df1['cost']

        merged_df2 = pd.merge(
            merged_df1, 
            co2Cost_df, 
            on=['transportresource'], 
            how='inner'
        )

        merged_df2['total_co2_costs'] = merged_df2['freq'] * merged_df2['CO2'] * merged_df2['distance']
    
        transport_costs = merged_df1['total_transportation_costs'].sum()
        co2_costs = merged_df2['total_co2_costs'].sum()

        total_costs = transport_costs + co2_costs
        print("total costs:", total_costs, f"({total_costs:.2e}) [transport: {transport_costs} ({transport_costs:.2e}), emission: {co2_costs}, ({co2_costs:.2e})]") 

        return total_costs