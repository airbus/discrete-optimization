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
bool: adding_subtour_elimination=false;


array[VEHICLES, ALL_NODES, ALL_NODES] of var bool: flow;
array[ALL_NODES] of var 0..1000: time_reached;
constraint forall(v in VEHICLES)(time_reached[starting_node[v]]==0);

% Indegree
constraint
    forall(j in NODES_TRANSPORTATION)(
        sum(v in VEHICLES, j1 in ALL_NODES)(flow[v, j1, j]) = 1
    );

% Outdegree
constraint
    forall(j in NODES_TRANSPORTATION)(
        sum(v in VEHICLES, j1 in ALL_NODES)(flow[v, j, j1]) = 1
    );

constraint
    forall(v in VEHICLES, j in ALL_NODES)(flow[v, j, j]=0);


constraint forall(v in VEHICLES)(
    sum(j1 in ALL_NODES)(flow[v, starting_node[v], j1])==1
    );

array[NODES_TRANSPORTATION] of var VEHICLES: allocation;
constraint forall(j in NODES_TRANSPORTATION)(
allocation[j] == sum(v in VEHICLES, j1 in ALL_NODES)(v*flow[v, j1, j])
);

constraint forall(v in VEHICLES, n in NODES_TRANSPORTATION)(
sum(j in ALL_NODES)(flow[v, j, n])==sum(j in ALL_NODES)(flow[v, n, j])
);



array[VEHICLES, ALL_NODES] of var ALL_NODES: y;

constraint
    if adding_subtour_elimination then
        forall(v in VEHICLES, i in ALL_NODES, j in ALL_NODES)(
            flow[v, i, j]==1 -> (y[v, i]+1<=y[v, j]))
    endif;

constraint forall(v in VEHICLES)(y[v, starting_node[v]]==1);
constraint forall(v in VEHICLES, n in ALL_NODES)(
if n!=target_node[v] then
    y[v, target_node[v]]>y[v, n]
endif);

constraint
    if adding_subtour_elimination then
        forall(v in VEHICLES, i in ALL_NODES, j in ALL_NODES)(
            flow[v, i, j]==1 -> (time_reached[i]+time_delta[i, j]<=time_reached[j]))
    endif;


var int: distance =
    sum(v in VEHICLES, i in ALL_NODES, j in ALL_NODES)
        (distance_delta[i, j] * flow[v, i, j]);
var 0..1000: final_duration;
%constraint final_duration=max(time_reached);
constraint forall(n in ALL_NODES)(final_duration>=time_reached[n]);
solve
    :: int_search([flow[v, i, j] | i in ALL_NODES, j in ALL_NODES, v in VEHICLES],
                   first_fail, indomain_min, complete)
    minimize distance;

output [
    "x = ", show(flow), "\n",
    "y = ", show(y), "\n",
    "final duration = ", show(final_duration),"\n",
    "distance = ", show(distance)
];
