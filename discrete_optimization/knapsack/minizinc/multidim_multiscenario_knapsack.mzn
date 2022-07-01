%------------------------------------------------------------------------------%
% Inspired by https://github.com/MiniZinc/minizinc-benchmarks/blob/master/multi-knapsack/mknapsack_global.mzn
% Multi Dimensional Knapsack Problem
%------------------------------------------------------------------------------%

include "knapsack.mzn";

%------------------------------------------------------------------------------%
% Parameters

int: nb_items; % number of variables
int: nb_dimension; % number of weight dimension
int: nb_scenario;

set of int: ITEMS = 1..nb_items;
set of int: DIMENSIONS = 1..nb_dimension;
set of int: SCENARIOS = 1..nb_scenario;

array[ITEMS, DIMENSIONS, SCENARIOS] of int: weights;   % Weight of items per bin
array[DIMENSIONS, SCENARIOS] of int: max_capacity;     % capacity of each dimension
array[ITEMS, SCENARIOS] of int: values;            % Profit of items
array[SCENARIOS] of int: max_values_per_scenario = [sum(i in ITEMS)(values[i, k])|k in SCENARIOS];
int: max_value = max(max_values_per_scenario);

%------------------------------------------------------------------------------%
% Variables

array[ITEMS] of var 0..1: taken;               % Whether an item is packed
array[DIMENSIONS, SCENARIOS] of var 0..ub_array(max_capacity): weights_var;  % Total weight in a bin


array[SCENARIOS] of var 0..max_value: objective_per_scenario;
constraint forall(k in SCENARIOS)(sum(i in ITEMS)(values[i,k]*taken[i])==objective_per_scenario[k]);
var 0..sum(max_values_per_scenario): objective = sum(objective_per_scenario);

%------------------------------------------------------------------------------%
% Constraints
% Constraining the size of the bins
%
constraint
	forall(i in DIMENSIONS, j in SCENARIOS)( weights_var[i, j] >= 0 /\
	weights_var[i, j] <= max_capacity[i, j]);

% Knapsack constraint
%
constraint
	forall(j in DIMENSIONS, k in SCENARIOS)(
        knapsack([weights[i, j, k] | i in ITEMS],
                 [values[i, k] | i in ITEMS],
                 taken,
                 weights_var[j, k],
                 objective_per_scenario[k])
    );

%------------------------------------------------------------------------------%
% Some integrety check for the (input) data

constraint
    forall(i in ITEMS,j in DIMENSIONS, k in SCENARIOS)(
        assert(weights[i, j, k] >= 0, "negative values in a")
    );
constraint
    forall(i in ITEMS, k in SCENARIOS)(assert(values[i, k] >= 0, "negative values in values"));
constraint
    forall(j in DIMENSIONS, k in SCENARIOS)(assert(max_capacity[j, k] >= 0, "negative values in max_capacity") );

%------------------------------------------------------------------------------%
% Search

solve
    :: int_search(taken, input_order, indomain_max)
    maximize objective;

%------------------------------------------------------------------------------%
% Output

output [
    "x = ", show(taken), ";\n",
    "objective = ", show(objective), ";\n"
];

%------------------------------------------------------------------------------%
