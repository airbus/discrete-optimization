include "globals.mzn";
include "circuit.mzn";
include "increasing.mzn";

int: number_vehicle;
set of int: VEHICLES=1..number_vehicle;

int: number_resource;
set of int: RESOURCES=1..number_resource;
array[ALL_NODES, RESOURCES] of int: resource_flow_node;
array[ALL_NODES, ALL_NODES, RESOURCES] of int: resource_flow_edges;

array[VEHICLES, RESOURCES] of int: max_capacity_resource;
array[VEHICLES, RESOURCES] of int: min_capacity_resource;
array[RESOURCES] of bool: cut_to_max;
bool: consider_resource;
array[VEHICLES, RESOURCES, TRAJECTORY_INDEX] of var int: resource_leaving_node;
array[VEHICLES, RESOURCES, TRAJECTORY_INDEX] of var int: resource_arriving_node;

% Capacity constraints
constraint if consider_resource then
                forall(v in VEHICLES, r in RESOURCES, i in TRAJECTORY_INDEX)(
                    resource_leaving_node[v, r, i]<=max_capacity_resource[v, r] /\
                    resource_leaving_node[v, r, i]>=min_capacity_resource[v, r] /\
                    resource_arriving_node[v, r, i]<=max_capacity_resource[v, r] /\
                    resource_arriving_node[v, r, i]>=min_capacity_resource[v, r])
            endif;

constraint forall(v in VEHICLES, r in RESOURCES)(resource_leaving_node[v, r, 1]==resource_flow_node[starting_node[v], r]);
constraint forall(v in VEHICLES, r in RESOURCES)(resource_arriving_node[v, r, 1]==resource_flow_node[starting_node[v], r]);
constraint
    if consider_resource then
        forall(v in VEHICLES, r in RESOURCES, i in 2..trajectory_max_index)(
            let{var ALL_NODES: prev = trajectories[v, i-1],
                var ALL_NODES: cur = trajectories[v, i],
                var int: newres = resource_leaving_node[v, r, i-1]+resource_flow_edges[prev, cur, r]}
                in (
                if cut_to_max[r] then
                    resource_arriving_node[v, r, i] = min(newres,
                                                          max_capacity_resource[v, r])
                else
                    resource_arriving_node[v, r, i] = newres
                endif
            ))
    endif;
constraint if consider_resource then
                forall(v in VEHICLES, r in RESOURCES, i in 2..trajectory_max_index)(
                let{var ALL_NODES: cur = trajectories[v, i],
                    var int: newres = resource_arriving_node[v, r, i] + resource_flow_node[cur, r]}
                 in (
                    if cut_to_max[r] then
                        resource_leaving_node[v, r, i] = min(newres, max_capacity_resource[v, r])
                    else
                        resource_leaving_node[v, r, i] = newres
                    endif))
           endif;


% Initial values


int: number_of_nodes_transportation;
int: number_of_nodes = number_of_nodes_transportation+2*number_vehicle;
set of int: ALL_NODES=1..number_of_nodes;
set of int: NODES_TRANSPORTATION=1..number_of_nodes_transportation;
set of int: NODES_ORIGIN=number_of_nodes_transportation+1..number_of_nodes_transportation+number_vehicle;
set of int: NODES_TARGET=number_of_nodes_transportation+number_vehicle+1..number_of_nodes_transportation+2*number_vehicle;
array[VEHICLES] of NODES_ORIGIN: starting_node = [number_of_nodes_transportation+i | i in 1..number_vehicle];
array[VEHICLES] of NODES_TARGET: target_node = [number_of_nodes_transportation+number_vehicle+i | i in 1..number_vehicle];

array[ALL_NODES, ALL_NODES] of int: distance_delta;
array[ALL_NODES, ALL_NODES] of int: time_delta;
array[ALL_NODES] of var VEHICLES: vehicle_for_node;
constraint forall(v in VEHICLES, index in TRAJECTORY_INDEX)(vehicle_for_node[trajectories[v, index]]==v);
constraint forall(v in VEHICLES)(vehicle_for_node[starting_node[v]]==v);
constraint forall(v in VEHICLES)(vehicle_for_node[target_node[v]]==v);
%constraint forall(i in ALL_NODES)(sum(j in TRAJECTORY_INDEX)(trajectories[vehicle_for_node[i], j]==i)>=1);

int: nb_request_transportation;
bool: include_request_transportation;
set of int: REQUEST_TRANSPORTATION = 1..nb_request_transportation;
array[REQUEST_TRANSPORTATION, 1..2] of set of ALL_NODES: precedence_pickup;
constraint if include_request_transportation then
                forall(request in REQUEST_TRANSPORTATION)(
                       all_equal([vehicle_for_node[n]| i in 1..2, n in precedence_pickup[request, i]]))
           endif;
%constraint redundant_constraint(forall(request in REQUEST_TRANSPORTATION)(
%                                max([reached_time[n]| n in precedence_pickup[request, 1]])<=
%                                min([reached_time[n]| n in precedence_pickup[request, 2]])));
constraint if include_request_transportation then
        forall(request in REQUEST_TRANSPORTATION)(
             forall(n in precedence_pickup[request, 1], n2 in precedence_pickup[request, 2])(
                value_precede(n, n2, [trajectories[v, i]|v in VEHICLES, i in TRAJECTORY_INDEX])))
           endif;
int: trajectory_max_index = number_of_nodes_transportation+2;
set of int: TRAJECTORY_INDEX = 1..trajectory_max_index;
TRAJECTORY_INDEX: last_index = max(TRAJECTORY_INDEX);
array[VEHICLES, TRAJECTORY_INDEX] of var ALL_NODES: trajectories;
array[ALL_NODES] of var int: reached_time;
var int: final_duration;
%constraint forall(v in VEHICLES)(nvalue([trajectories[v, i]| i in TRAJECTORY_INDEX])>=
%                                 0.5 * (number_of_nodes_transportation div number_vehicle));
constraint final_duration=max(reached_time);
constraint forall(n in NODES_ORIGIN)(reached_time[n] == 0);
constraint forall(v in VEHICLES, i in 2..last_index)(
 let{var ALL_NODES: prev_node=trajectories[v, i-1],
     var ALL_NODES: cur_node=trajectories[v, i]} in
     (reached_time[cur_node] == reached_time[prev_node]+time_delta[prev_node, cur_node]));


constraint forall(v in VEHICLES)(trajectories[v, 1]==starting_node[v]);
constraint forall(v in VEHICLES)(trajectories[v, 2]!=starting_node[v]);
constraint forall(v in VEHICLES)(trajectories[v, last_index]==target_node[v]);
constraint nvalue(number_of_nodes, [trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX]);
constraint forall(n in NODES_TRANSPORTATION)(count_eq([trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX], n, 1));
constraint forall(n in NODES_ORIGIN)(count_eq([trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX], n, 1));

var int: distance;
constraint forall(v in VEHICLES)(alldifferent_except_0([trajectories[v,j]-target_node[v]|
                                                        j in TRAJECTORY_INDEX]));
constraint if number_vehicle == 1 then alldifferent([trajectories[1,j]|
                                                     j in TRAJECTORY_INDEX])
           endif;

constraint sum(v in VEHICLES, i in 1..last_index-1)(distance_delta[trajectories[v, i], trajectories[v, i+1]])==distance;
constraint forall(v in VEHICLES, i in 2..last_index-1)(trajectories[v, i]==target_node[v]->trajectories[v, i+1]==target_node[v]);
constraint forall(v in VEHICLES, vprime in VEHICLES, i in TRAJECTORY_INDEX)(
    if v != vprime then
    trajectories[vprime, i] != starting_node[v] /\ trajectories[vprime, i] != target_node[v]
    endif
);

array[NODES_TRANSPORTATION] of var 2..number_of_nodes_transportation+1: index_node;
constraint forall(v in VEHICLES, i in TRAJECTORY_INDEX)(
    if trajectories[v,i]!=target_node[v] /\ trajectories[v,i]!=starting_node[v] then
        index_node[trajectories[v,i]]==i
    endif);

var int: objective_to_consider;
array[1..2] of int: weight_objective;
constraint objective_to_consider = weight_objective[1]*final_duration+weight_objective[2]*distance;

bool: force_all_use_of_vehicle;
constraint if force_all_use_of_vehicle then
                forall(v in VEHICLES)(trajectories[v, 2]!=target_node[v])
           endif;

solve
    :: seq_search([%int_search(vehicle_for_node, smallest, indomain_min),
                   %int_search([final_duration], dom_w_deg, indomain_min),
                   %int_search([distance], input_order, indomain_min)
                   int_search([trajectories[v, i]| i in TRAJECTORY_INDEX, v in VEHICLES],
                               smallest, indomain_min),
                   %int_search([distance], first_fail, indomain_min, complete),
                   %int_search([final_duration], first_fail, indomain_min)
                   ])
    %::restart_constant(1000)
    minimize(objective_to_consider);
output [
    "x = ", show(trajectories), "\n",
    "final time = ", show(final_duration),"\n",
    "cumulated distance = ", show(distance), "\n",
    "resource =", show(resource_arriving_node), "\n",
    "resourceleaving =", show(resource_leaving_node), "\n",

];
