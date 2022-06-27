include "globals.mzn";
include "circuit.mzn";
include "increasing.mzn";

int: nb_nodes;
int: nb_vehicle;
set of int: VEHICLES = 1..nb_vehicle;
set of int: NODES = 1..nb_nodes;
int: nb_edges;
set of int: EDGES = 1..nb_edges;
array[EDGES] of NODES: from_node; # define origin of edges
array[EDGES] of NODES: to_node; # define destination of edges
array[NODES, NODES] of bool: adjacency_graph = array2d(NODES, NODES,
                                                       [exists(i in EDGES)(from_node[i]==j /\ to_node[i]==k) |
                                                       j in NODES, k in NODES]);
array[NODES, NODES] of int: index_of_edges = array2d(NODES, NODES,
                                                     [sum(i in EDGES)(i*(from_node[i]==j /\ to_node[i]==k)) |
                                                     j in NODES, k in NODES]);
array[EDGES] of int: time_travel;
array[EDGES] of int: distance;
array[VEHICLES] of NODES: origin_vehicle;
array[VEHICLES] of NODES: destination_vehicle;
array[customer_range] of int: demand;
array[NODES] of NODES: orbit;
constraint value_precede_chain(origin_vehicle,
                               orbit); # The orbit is in the vehicle order.
constraint value_precede_chain(destination_vehicle,
                               orbit);
constraint forall(v in VEHICLES)(value_precede(origin_vehicle[v], destination_vehicle[v], orbit);
array[NODES] of var bool: nodes_taken;
array[EDGES] of var bool: edges_taken;
int: max_distance = sum(distance);
var of 0..max_distance: cumulated_distance;
constraint forall(i in 1..nb_nodes-1)(adjacency_graph[orbit[i], orbit[i+1]] /\
edges_taken[index_of_edges[orbit[i], orbit[i+1]]);
constraint bounded_path(from_node
                        to_node,
                        distance,
                        origin_vehicle[1],
                        destination_vehicle[nb_vehicle],
                        [true | n in NODES],
                        edges_taken,
                        cumulated_distance);

constraint all_different(orbit);
array[VEHICLES] of var customer_range: index_end_subcircuit;
constraint increasing(index_end_subcircuit);
constraint decreasing([index_end_subcircuit[i+1]-index_end_subcircuit[i] | i in 1..nb_vehicle-1]);
constraint forall(v in VEHICLES)(orbit[index_end_subcircuit[v]]==destination_vehicle[v]);
constraint forall(v in VEHICLES)(orbit[index_end_subcircuit[v]]==destination_vehicle[v]);

array[VEHICLES] of int: capacity_vehicle;
int: max_capacity=max(capacity_vehicle);
set of int: capacity_per_vehicle=0..max_capacity;
array[1..nb_vehicle] of var capacity_per_vehicle: capacity_used;

constraint forall(v in 2..nb_vehicle)(
    (sum(i in index_end_subcircuit[v-1]+1..index_end_subcircuit[v])(demand[orbit[i]]))==capacity_used[v]);
constraint sum(i in 1..index_end_subcircuit[1])(demand[circuit_vrp[i]]) == capacity_used[1];
constraint forall(v in VEHICLES)(capacity_used[v]<=capacity_vehicle[v]);
constraint orbit[1] == origin_vehicle[1];
constraint orbit[nb_nodes] == destination_vehicle[nb_vehicle];

solve
    :: int_search(edges_taken, first_fail, indomain_min, complete)
    minimize cumulated_distance;

output [
    "x = ", show(orbit), "\n",
    "objective = ", show(cumulated_distance),"\n"
];
