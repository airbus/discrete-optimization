%------------------------------------------------------------------------------%
% Inspired by https://github.com/MiniZinc/minizinc-benchmarks/blob/master/multi-knapsack/mknapsack_global.mzn
% Multi Dimensional Knapsack Problem
%------------------------------------------------------------------------------%

include "knapsack.mzn";

%------------------------------------------------------------------------------%
% Parameters

int: nb_items; % number of variables
int: nb_dimension; % number of constraints

set of int: ITEMS = 1..nb_items;
set of int: DIMENSIONS = 1..nb_dimension;

array[ITEMS, DIMENSIONS] of int: weights;   % Weight of items per bin
array[DIMENSIONS] of int: max_capacity;            % capacity of each dimension
array[ITEMS] of int: values;            % Profit of items


%------------------------------------------------------------------------------%
% Variables

array[ITEMS] of var 0..1: taken;               % Whether an item is packed
array[DIMENSIONS] of var 0..ub_array(max_capacity): weights_var;  % Total weight in a bin

var 0..sum(values): objective = sum(i in ITEMS)(values[i] * taken[i]);   % Total profit

%------------------------------------------------------------------------------%
% Constraints
% Constraining the size of the bins
%
constraint
	forall(i in DIMENSIONS)( weights_var[i] >= 0 /\ weights_var[i] <= max_capacity[i] );

% Knapsack constraint
%
constraint
	forall(j in DIMENSIONS)(
        knapsack([weights[i, j] | i in ITEMS], values, taken, weights_var[j], objective)
    );

%------------------------------------------------------------------------------%
% Some integrety check for the (input) data

constraint
    forall(i in ITEMS,j in DIMENSIONS)(
        assert(weights[i,j] >= 0, "negative values in a")
    );
constraint
    forall(i in ITEMS)( assert(values[i] >= 0, "negative values in values") );
constraint
    forall(j in DIMENSIONS)( assert(max_capacity[j] >= 0, "negative values in max_capacity") );

%------------------------------------------------------------------------------%
% Search

solve
    %:: int_search(taken, input_order, indomain_max, complete)
    maximize objective;

%------------------------------------------------------------------------------%
% Output

output [
    "x = ", show(taken), ";\n",
    "objective = ", show(objective), ";\n"
];

%------------------------------------------------------------------------------%
