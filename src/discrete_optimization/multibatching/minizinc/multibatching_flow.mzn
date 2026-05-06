% Multibatching Flow Formulation using MiniZinc
% Solves the multibatching problem using network flow constraints

include "globals.mzn";

% ==================== Parameters ====================

% Problem dimensions
int: nb_locations;
int: nb_products;
int: nb_transport_links;

% Sets
set of int: LOCATIONS = 1..nb_locations;
set of int: PRODUCTS = 1..nb_products;
set of int: LINKS = 1..nb_transport_links;

% Product characteristics
array[PRODUCTS] of int: product_size;

% Transport link characteristics
array[LINKS] of int: link_from;        % Origin location (1..nb_locations)
array[LINKS] of int: link_to;          % Destination location (1..nb_locations)
array[LINKS] of int: link_capacity;    % Capacity of transport
array[LINKS] of int: link_distance;    % Distance
array[LINKS] of int: link_cost;        % Cost per unit distance
array[LINKS] of int: link_emissions;   % Emissions per unit distance

% Arc structure for network_flow predicate (from, to pairs)
array[LINKS, 1..2] of LOCATIONS: arcs = array2d(LINKS, 1..2, [
    if j == 1 then link_from[i] else link_to[i] endif
    | i in LINKS, j in 1..2
]);

% Supply/demand at each location for each product
% Positive = supply (source), Negative = demand (sink), 0 = transshipment
array[LOCATIONS, PRODUCTS] of int: net_supply;

% Product compatibility with transport links
% 1 if product p can use link l, 0 otherwise
array[LINKS, PRODUCTS] of 0..1: product_can_use_link;

% Upper bounds for flows (total supply of each product)
array[PRODUCTS] of int: max_flow_per_product;

% ==================== Decision Variables ====================

% Number of trips on each transport link
array[LINKS] of var int: nb_trips;

% Flow of each product on each transport link
array[LINKS, PRODUCTS] of var int: flow;

% ==================== Constraints ====================

% 1. Network flow conservation for each product
constraint forall(p in PRODUCTS)(
    network_flow(
        arcs,
        [net_supply[loc, p] | loc in LOCATIONS],
        [flow[link, p] | link in LINKS]
    )
);

% 2. Flow bounds: only non-negative flows, bounded by total supply
constraint forall(link in LINKS, p in PRODUCTS)(
    flow[link, p] >= 0 /\
    flow[link, p] <= max_flow_per_product[p]
);

% 3. Product compatibility: flow can only exist on compatible links
constraint forall(link in LINKS, p in PRODUCTS)(
    product_can_use_link[link, p] == 0 -> flow[link, p] == 0
);

% 4. Capacity constraint: total volume cannot exceed trips * capacity
constraint forall(link in LINKS)(
    sum(p in PRODUCTS)(flow[link, p] * product_size[p])
    <= nb_trips[link] * link_capacity[link]
);

% 5. Number of trips must be non-negative
constraint forall(link in LINKS)(
    nb_trips[link] >= 0
);

% 6. If no flow on a link, no trips needed (optional but helps solver)
constraint forall(link in LINKS)(
    sum(p in PRODUCTS)(flow[link, p]) == 0 -> nb_trips[link] == 0
);

% ==================== Objective ====================

% Total transport cost = distance * (cost + emissions) * nb_trips
var int: total_cost = sum(link in LINKS)(
    nb_trips[link] * link_distance[link] * (link_cost[link] + link_emissions[link])
);

solve minimize total_cost;

% ==================== Output ====================

output [
    "total_cost = ", show(total_cost), "\n",
    "nb_trips = ", show(nb_trips), "\n",
    "flow = ", show(flow), "\n"
];
