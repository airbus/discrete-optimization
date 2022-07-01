include "globals.mzn";
include "circuit.mzn";
include "increasing.mzn";
int: nb_customers_without_depot;
int: nb_vehicle;
int: nb_customers_virtual = nb_customers_without_depot+nb_vehicle+1;
int: max_capacity_vehicle;
array[nb_vehicle] of int: capacity_vehicles;
set of int: capacity_per_vehicle=0..max_capacity_vehicle;
set of int: customer_range=1..nb_customers_virtual;


int: nb_edges;
set of int: EDGES=1..nb_edges;
int: nb_nodes=nb_customers_virtual;
bool: undirected = false;
array[EDGES] of customer_range: origin_edge;
array[EDGES] of customer_range: destination_edge;
array[EDGES] of int: weight_edges;

array[customer_range] of var customer_range: circuit_vrp;
array[1..nb_vehicle] of var customer_range: index_end_subcircuit;

array[1..nb_vehicle+1] of customer_range: virtual_nodes = [nb_customers_without_depot+i | i in 1..nb_vehicle+1];
constraint forall(v in 1..nb_vehicle)(circuit_vrp[index_end_subcircuit[v]]==virtual_nodes[v+1]);


array[1..nb_vehicle] of var capacity_per_vehicle: capa;
constraint forall(v in 2..nb_vehicle)(
    (sum(i in index_end_subcircuit[v-1]..index_end_subcircuit[v])(demand[circuit_vrp[i]]))==capa[v]);
