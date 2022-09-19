%-----------------------------------------------------------------------------%
% Model example for Resource-Constrained Project Scheduling Problems with multiscenario
%-----------------------------------------------------------------------------%
include "globals.mzn";
%-----------------------------------------------------------------------------%
% Model parameters.


% Resources
%
int: n_res;                     % The number of resources
set of int: Res = 1..n_res;     % The set of all resources
array [Res] of int: rc;         % The resource capabilities

% Scenario
int: n_scenario;
set of int: SCENARIS = 1..n_scenario;

    % Tasks
    %
int: n_tasks;                           % The number of tasks
set of int: Tasks = 1..n_tasks;         % The set of all tasks
array [Tasks, SCENARIS]      of int       : d  ;  % The task durations
array [Res, Tasks, SCENARIS] of int       : rr ;  % The resource requirements
array [Tasks]      of set of int: suc;  % The task successors


% Fake Tasks (to handle notably the varying resource availability)
bool: include_fake_tasks;
int: max_duration_fake_task;
int: n_fake_tasks;
set of int: FakeAct=1..n_fake_tasks;
array[FakeAct] of 0..max_time: fakestart;
array[FakeAct] of 1..max_duration_fake_task: fakedur;
array[Res,FakeAct] of 0..max(rc): fakereq;

% Planning horizon
% int: t_max = sum(i in Tasks)(d[i]);     % End time of the planning horizon
% set of int: Times = 0..t_max-1;     % Possible start times
int: max_time;
set of int: Times = 0..max_time;     % Possible start times

array[Tasks] of var Tasks: ordering;
% constraint ordering = argsort(s);
constraint alldifferent(ordering);
array[Tasks] of var Tasks: position;
constraint inverse(ordering, position);
constraint
   forall (i in Tasks, j in suc[i] )
   (
         position[j]>position[i]
   ); % coherent position with successors (can help pruning hopefully)

var 0..n_tasks*n_scenario: nb_incoherence;
int: nb_incoherence_limit;
constraint nb_incoherence <= nb_incoherence_limit;
bool: relax_ordering;
constraint
    if relax_ordering then
        sum(j in SCENARIS, t in 1..n_tasks-1)(s[ordering[t], j]>s[ordering[t+1], j])==nb_incoherence
    else
        nb_incoherence=0
    endif;

constraint if relax_ordering==false then
                forall(j in SCENARIS, t in 1..n_tasks-1)(s[ordering[t], j]<=s[ordering[t+1], j])
           else
                true
           endif;


%constraint forall(j in SCENARIS, t in 1..n_tasks-1)(s[ordering[t], j]<=s[ordering[t+1], j]);

%-----------------------------------------------------------------------------%
% Model variables.

array [Tasks, SCENARIS] of var Times: s;  % The start times
var 0..n_scenario*max_time: objective      ;  % The project duration (makespan)

%-----------------------------------------------------------------------------%
% Constraints.

    % Precedence constraints
    %
constraint
   forall (sc in SCENARIS, i in Tasks, j in suc[i] )
   (
         s[i, sc] + d[i, sc] <= s[j, sc]
   );

    % Redundant non-overlapping constraints
    %
constraint
    forall (sc in SCENARIS)(
    forall ( i, j in Tasks where i < j )
    (
        if exists(r in Res)(rr[r, i, sc] + rr[r, j, sc] > rc[r]) then
            s[i, sc] + d[i, sc] <= s[j, sc]   \/ s[j, sc] + d[j, sc] <= s[i, sc]
        else
            true
        endif
    )
    );


constraint forall(sc in SCENARIS)(
             forall(r in Res)(
               forall(i in Tasks)(
                    if d[i, sc]>0 then
                      assert(rr[r, i, sc]<=rc[r], "Task \(i) consuming more resource \(r) than available")
                      %rr[r,i]<=rc[r]
                    else
                        true
                    endif
           )));

%constraint
%  forall(r in Res)(cumulative(
%                  [s[i] | i in Tasks ],
%                  [d[i] | i in Tasks ],
%                  [rr[r, i] | i in Tasks ],
%                  rc[r]));

% Cumulative resource constraints
%
constraint
     forall (sc in SCENARIS)(
     forall ( r in Res )
     (
         let {
             set of int: RTasks =
                             { i | i in Tasks
                             where rr[r, i, sc] > 0 /\ d[i, sc] > 0 },
             int: sum_rr = sum(i in RTasks)(rr[r, i, sc])
         } in (
             if RTasks != {} then
                 if include_fake_tasks  then
                    let{set of FakeAct: FTasks = {j | j in FakeAct where fakereq[r, j]>0}}
                      in(
                        cumulative([ s[i, sc] | i in RTasks ]++[fakestart[p] | p in FTasks],
                                   [ d[i, sc] | i in RTasks ]++[fakedur[p] | p in FTasks],
                                   [ rr[r, i, sc] | i in RTasks ]++[fakereq[r, p] | p in FTasks],
                                   rc[r])
                        )
                 else if sum_rr > rc[r] then
                        cumulative([ s[i, sc] | i in RTasks ],
                                   [ d[i, sc] | i in RTasks ],
                                   [ rr[r, i, sc] | i in RTasks ],
                                   rc[r])
                      else
                        true
                      endif
                 endif
             endif
         )
     ));

% Makespan constraints
bool: add_objective_makespan; % by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)
Tasks: sink_task;
constraint(if add_objective_makespan then
               sum(sc in SCENARIS)(s[sink_task, sc])==objective
           else
                true
           endif);


%-----------------------------------------------------------------------------%
% Objective.
bool: ignore_sec_objective;
var int: sec_objective;
constraint sec_objective>=0; % trick to add a second part in the objective.
constraint (if ignore_sec_objective then
                sec_objective = 0
            else
                true
            endif);



%solve satisfy;
solve
   :: int_search(ordering, input_order, indomain_min)
   %:: int_search(position, input_order, indomain_min)
   %:: int_search( [s[j, sc]| j in Tasks, sc in SCENARIS] ++ [objective], input_order, indomain_min) %, complete )
   %:: int_search( s ++ [objective], smallest, indomain_min, complete) %, complete )
   minimize objective+sec_objective;

%-----------------------------------------------------------------------------%
output [
    "incoherence =", show(nb_incoherence), "\n",
    "Ordering =", show(ordering),"\n",
    "Start times = ", show(s), "\n",
    "makespan = ", show(objective), "\n",
    "sec objective = ", show(sec_objective), "\n",
    "rr=", show(rr)];

%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%
