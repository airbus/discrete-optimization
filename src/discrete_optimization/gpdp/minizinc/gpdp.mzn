include "globals.mzn";
include "circuit.mzn";
include "increasing.mzn";

int: number_vehicle;
set of int: VEHICLES=1..number_vehicle;

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

set of int: TRAJECTORY_INDEX = 1..number_of_nodes_transportation+2;
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
     (reached_time[cur_node] >= reached_time[prev_node]+time_delta[prev_node, cur_node]));


constraint forall(v in VEHICLES)(trajectories[v, 1]==starting_node[v]);
constraint forall(v in VEHICLES)(trajectories[v, 2]!=starting_node[v]);
constraint forall(v in VEHICLES)(trajectories[v, last_index]==target_node[v]);
constraint nvalue(number_of_nodes, [trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX]);
constraint forall(n in NODES_TRANSPORTATION)(count_eq([trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX], n, 1));
constraint forall(n in NODES_ORIGIN)(count_eq([trajectories[v, j]| v in VEHICLES, j in TRAJECTORY_INDEX], n, 1));

var int: distance;
constraint forall(v in VEHICLES)(alldifferent_except_0([trajectories[v,j]-target_node[v]|
                                                        j in TRAJECTORY_INDEX]));
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

%constraint forall(v in VEHICLES)(let {var set of NODES_TRANSPORTATION:
% s={trajectories[v,i]|i in TRAJECTORY_INDEX where not (trajectories[v,i]
%                                                 in {starting_node[v],
%                                                     target_node[v]})}} in
%                                (
%
%                               );

solve
    :: seq_search([int_search([trajectories[v, i]| i in TRAJECTORY_INDEX, v in VEHICLES],
                               dom_w_deg, indomain_min),
                   %int_search([distance], first_fail, indomain_min, complete),
                   %int_search([final_duration], first_fail, indomain_min)
                   ])
    %::restart_constant(1000)
    minimize(10*final_duration+distance);
output [
    "x = ", show(trajectories), "\n",
    "final time = ", show(final_duration),"\n",
    "cumulated distance = ", show(distance), "\n"
];
