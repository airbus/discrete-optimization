%include "globals.mzn";
include "increasing.mzn";
include "decreasing.mzn";
include "my_cumulative.mzn";
%-----------------------------------------------------------------------------%
% Model parameters.


% Resources
%
int: n_res;                     % The number of resources
set of int: Res = 1..n_res;     % The set of all resources
array [Res] of int: rc;         % The resource capabilities
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)
set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};


% Tasks
%
int: n_tasks;                           % The number of tasks
set of int: Tasks = 1..n_tasks;         % The set of all tasks
array [Tasks] of set of int: suc;  % The task successors


% Modes
% Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;
set of int: Mod = 1..nb_mod;    % Set of modes
array[Tasks] of set of Mod: modes;
array[Tasks] of set of Mod: mode = modes;

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;
array[Res, Mod] of int: rreq;
array[Res, Mod] of int: mrreq = rreq;
array[Mod] of Tasks: mact = [sum(i in Tasks where m in mode[i])(i) | m in Mod];
int: min_mrreq = min(k in Res, m in Mod)(mrreq[k,m]);
int: max_mrreq = max(k in Res, m in Mod)(mrreq[k,m]);

array[Mod] of var bool: mrun;
array[Tasks] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Tasks];
array[Res, Tasks] of var min_mrreq..max_mrreq: arreq = array2d(Res,Tasks,[let {var {mrreq[k,m] | m in mode[i]}: x} in x | k in Res, i in Tasks]);

%---------------------------------------------------------------------------%
% Constraints

    % Activities and mode constraints
    %
constraint
    forall(i in Tasks)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  sum(m in mode[i])(mrun[m]) = 1
        /\  adur[i] = mdur[mi]
        /\  forall(k in Res)( arreq[k,i] = mrreq[k, mi] )
        )
    );


array[Tasks] of bool: possibly_preemptive; % boolean indicating if the task can be preempted.



% Planning horizon
int: max_time;
set of int: Times = 0..max_time-1;     % Possible start times
set of int: Times1 = 1..max_time;
array[Res, Times1] of int: ressource_capacity_time;


%-----------------------------------------------------------------------------%
% Model variables.
int: nb_preemptive;
set of int: PREEMPTIVE=1..nb_preemptive;


array [Tasks] of var Times: s;  % The start times
array [Tasks, PREEMPTIVE] of var Times: s_preemptive;  % The start times
array [Tasks, PREEMPTIVE] of var int: d_preemptive;

constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]>=0);
constraint forall(i in Tasks)(d_preemptive[i, nb_preemptive]==0);
constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]<=adur[i]);
constraint forall(i in Tasks)(increasing([s_preemptive[i, j]| j in PREEMPTIVE]));
constraint forall(i in Tasks)(s[i]==s_preemptive[i,1]);
constraint forall(i in Tasks)(if adur[i]>0 then d_preemptive[i, 1]>0 endif);
constraint forall(i in Tasks)(decreasing([d_preemptive[i, j]|j in PREEMPTIVE]));
constraint forall(i in Tasks)(sum(j in PREEMPTIVE)(d_preemptive[i, j])==adur[i]);
constraint forall(i in Tasks)(disjunctive([s_preemptive[i, j]|j in PREEMPTIVE], [d_preemptive[i, j]|j in PREEMPTIVE]));
constraint forall(i in Tasks)(forall(j in 2..nb_preemptive)(
    (d_preemptive[i, j]==0) -> s_preemptive[i, j]==s_preemptive[i, j-1]+d_preemptive[i, j-1]
));

array[Tasks] of var bool: is_paused;
constraint forall(i in Tasks)(
        if possibly_preemptive[i]==false then
            d_preemptive[i, 1]==adur[i] /\ is_paused[i]==false
        endif
);


constraint forall(i in Tasks)(is_paused[i]<->d_preemptive[i, 1]<adur[i]);
var int: nb_paused = sum(i in Tasks)(is_paused[i]);


int: max_preempted;
constraint nb_paused<=max_preempted;
% constraint nb_paused>=5;
% constraint nb_paused>=3;

constraint forall(i in Tasks)(forall(j in 1..nb_preemptive-1)(
    (d_preemptive[i, j+1]>0 <-> s_preemptive[i,j+1]>s_preemptive[i,j]+d_preemptive[i, j]))); %avoid dummy solution
constraint forall(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>=s_preemptive[i, 1]+adur[i]);
%var int: nb_preemptivity;
%constraint sum(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>s_preemptive[i, 1]+adur[i])==nb_preemptivity;
var 0..max_time: objective      ;  % The project duration (makespan)


%-----------------------------------------------------------------------------%
% Constraints.
% Precedence constraints
%
constraint
   forall ( i in Tasks, j in suc[i] )
   (
         s_preemptive[i, nb_preemptive] + d_preemptive[i, nb_preemptive] <= s_preemptive[j, 1]
   );

% Redundant non-overlapping constraints
%
constraint
    forall ( i, j in Tasks where i < j )
    (
        if exists(r in Res)(rreq[r, i] + rreq[r, j] > rc[r]) then
            s_preemptive[i, 1] + d_preemptive[i, 1] <= s_preemptive[j, 1]   \/
            s_preemptive[j, 1] + d_preemptive[j, 1] <= s_preemptive[i, 1]
        else
            true
        endif
    );


% Cumulative resource constraints
%
constraint forall(r in Res)(my_cumulative([s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                                          [d_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                                          [arreq[r, i] | i in Tasks, j in PREEMPTIVE],
                                          [ressource_capacity_time[r, t] | t in Times1],
                                           max_time-1));

% Non-renewable resource constraints    %
constraint forall(k in NRes)(sum(i in Tasks)(arreq[k, i]) <= rc[k]);

% Objective constraints
%
bool: add_objective_makespan; % by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)
constraint
 if add_objective_makespan then
        forall(i in Tasks where suc[i]={})(s_preemptive[i, nb_preemptive] + d_preemptive[i, nb_preemptive]
                                            <= objective)
 else
    true
 endif;

%-----------------------------------------------------------------------------%
% Second Objective
bool: ignore_sec_objective;
var int: sec_objective;
constraint sec_objective>=0; % trick to add a second part in the objective.
constraint (if ignore_sec_objective then
                sec_objective = 0
            else
                true
            endif);

%-----------------------------------------------------------------------------%
% Objective.

%solve satisfy;
ann: durThenStartThenMode = seq_search([
        %int_search(d_preemptive, smallest, indomain_max, complete),
        %int_search( [nb_preemptivity], smallest, indomain_min),
        %bool_search(is_paused, input_order, indomain_min, complete),
        %int_search([nb_paused], input_order, indomain_min),
        int_search([s_preemptive[i, 1]| i in Tasks], smallest, indomain_min, complete),
        int_search([d_preemptive[i, 1]| i in Tasks], input_order, indomain_max),
        %int_search(s ++ [objective], smallest, indomain_min, complete)
    ]);

solve
   :: durThenStartThenMode
   %:: int_search( s ++ [objective], smallest, indomain_min, complete) %, complete )
   minimize objective+sec_objective;

%-----------------------------------------------------------------------------%
output [
    "Start times = ", show(s), "\n",
    "nb_paused= ", show(nb_paused), "\n",
    "makespan = ", show(objective), "\n",
    "sec objective = ", show(sec_objective), "\n"];

%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%
