%include "globals.mzn";
include "cumulative.mzn";
include "increasing.mzn";
include "disjunctive.mzn";

%-----------------------------------------------------------------------------%
% Model parameters.


% Resources
%
int: n_res;                     % The number of resources
set of int: Res = 1..n_res;     % The set of all resources
array [Res] of int: rc;         % The resource capabilities


% Tasks
%
int: n_tasks;                           % The number of tasks
set of int: Tasks = 1..n_tasks;         % The set of all tasks
array [Tasks]      of int       : d  ;  % The task durations
array [Res, Tasks] of int       : rr ;  % The resource requirements
array [Tasks]      of set of int: suc;  % The task successors


% Fake Tasks (to handle notably the varying resource availability)
bool: include_fake_tasks;
int: max_duration_fake_task;
int: n_fake_tasks;
set of int: FakeAct=1..n_fake_tasks;
array[FakeAct] of 0..max_time: fakestart;
array[FakeAct] of 1..max_duration_fake_task: fakedur;
array[Res,FakeAct] of 0..max(rc): fakereq;

array[Tasks] of bool: possibly_preemptive; % boolean indicating if the task can be preempted.

% Planning horizon
% int: t_max = sum(i in Tasks)(d[i]);     % End time of the planning horizon
int: max_time;
set of int: Times = 0..max_time;     % Possible start times
% set of int: Times = 0..t_max-1;     % Possible start times



%-----------------------------------------------------------------------------%
% Model variables.
int: nb_preemptive;
set of int: PREEMPTIVE=1..nb_preemptive;


array [Tasks] of var Times: s;  % The start times
array [Tasks, PREEMPTIVE] of var Times: s_preemptive;  % The start times
int: max_dur = max(d);
array [Tasks, PREEMPTIVE] of var 0..max_dur: d_preemptive;

constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]>=0);
constraint forall(i in Tasks)(d_preemptive[i, nb_preemptive]==0);
constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]<=d[i]);
constraint forall(i in Tasks)(increasing([s_preemptive[i, j]| j in PREEMPTIVE]));
constraint forall(i in Tasks)(s[i]==s_preemptive[i,1]);
constraint forall(i in Tasks)(if d[i]>0 then d_preemptive[i, 1]>0 endif);
% constraint forall(i in Tasks)(decreasing([d_preemptive[i, j]|j in PREEMPTIVE]));
constraint forall(i in Tasks, j in 1..nb_preemptive-1)(d_preemptive[i, j]==0->d_preemptive[i, j+1]==0);

constraint forall(i in Tasks)(sum(j in PREEMPTIVE)(d_preemptive[i, j])==d[i]);
constraint forall(i in Tasks)(disjunctive([s_preemptive[i, j]|j in PREEMPTIVE], [d_preemptive[i, j]|j in PREEMPTIVE]));
constraint forall(i in Tasks)(forall(j in 2..nb_preemptive)(
    (d_preemptive[i, j]==0) -> s_preemptive[i, j]==s_preemptive[i, j-1]+d_preemptive[i, j-1]
));

array[Tasks] of var bool: is_paused;
constraint forall(i in Tasks)(
        if possibly_preemptive[i]==false then
            d_preemptive[i, 1]==d[i] /\ is_paused[i]==false
        endif
);


constraint forall(i in Tasks)(is_paused[i]<->d_preemptive[i, 1]<d[i]);
var 0..n_tasks: nb_paused = sum(i in Tasks)(is_paused[i]);


int: max_preempted;
constraint nb_paused<=max_preempted;
constraint forall(i in Tasks)(forall(j in 1..nb_preemptive-1)(
    (d_preemptive[i, j+1]>0 <-> s_preemptive[i,j+1]>s_preemptive[i,j]+d_preemptive[i, j]))); %avoid dummy solution
constraint forall(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>=s_preemptive[i, 1]+d[i]);
%var int: nb_preemptivity;
%constraint sum(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>s_preemptive[i, 1]+d[i])==nb_preemptivity;

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
        if exists(r in Res)(rr[r, i] + rr[r, j] > rc[r]) then
            s_preemptive[i, 1] + d_preemptive[i, 1] <= s_preemptive[j, 1]   \/
            s_preemptive[j, 1] + d_preemptive[j, 1] <= s_preemptive[i, 1]
        else
            true
        endif
    );


constraint forall(r in Res)(
               forall(i in Tasks)(
                    if d[i]>0 then
                      assert(rr[r, i]<=rc[r], "Task \(i) consuming more resource \(r) than available")
                      %rr[r,i]<=rc[r]
                    else
                        true
                    endif
           ));

%constraint
%  forall(r in Res)(cumulative(
%                  [s[i] | i in Tasks ],
%                  [d[i] | i in Tasks ],
%                  [rr[r, i] | i in Tasks ],
%                  rc[r]));

% Cumulative resource constraints
%
constraint
     forall ( r in Res )
     (
         let {
             set of int: RTasks =
                             { i | i in Tasks
                              where rr[r, i] > 0 /\ d[i] > 0 },
             int: sum_rr = sum(i in RTasks)(rr[r, i])
         } in (
             if RTasks != {} then
                 if include_fake_tasks then
                    let{set of FakeAct: FTasks = {j | j in FakeAct where fakereq[r, j]>0}}
                      in(
                        cumulative([s_preemptive[i, j] | i in RTasks, j in PREEMPTIVE]++[fakestart[p] | p in FTasks],
                                   [d_preemptive[i, j] | i in RTasks, j in PREEMPTIVE]++[fakedur[p] | p in FTasks],
                                   [rr[r, i] | i in RTasks, j in PREEMPTIVE]++[fakereq[r, p] | p in FTasks],
                                   rc[r])
                        )
                 else if sum_rr > rc[r] then
                     cumulative([s_preemptive[i, j] | i in RTasks, j in PREEMPTIVE],
                               [d_preemptive[i, j] | i in RTasks, j in PREEMPTIVE],
                               [rr[r, i] | i in RTasks, j in PREEMPTIVE],
                               rc[r])
                     else
                        true
                     endif
                 endif
             else
                 true
             endif
         )
     );

% Makespan constraints
%


bool: add_objective_makespan; % by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)

constraint (if add_objective_makespan then
                forall ( i in Tasks where suc[i] == {} )
                    (s_preemptive[i, nb_preemptive] + d_preemptive[i, nb_preemptive] <= objective)
            else
                true
            endif);
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
set of Tasks: TasksPreempted = {i | i in Tasks where possibly_preemptive[i]==true};
set of Tasks: TasksNotPreempted = {i | i in Tasks where possibly_preemptive[i]==false};
set of int: PREEMP2 = 2..nb_preemptive;
array[1..length(TasksPreempted)] of var Times: ss = [s_preemptive[i, 2]| i in TasksPreempted];


ann: durThenStartThenMode = seq_search([
        int_search([s_preemptive[i, 1]| i in Tasks], smallest, indomain_min),
        int_search([d_preemptive[i, 1]| i in Tasks], input_order, indomain_max),
        bool_search(is_paused, input_order, indomain_min),
        %int_search([s_preemptive[i, 2]| i in Tasks], first_fail, indomain_min),
        int_search([nb_paused], input_order, indomain_min),
        %int_search([d_preemptive[i, 1]| i in Tasks], input_order, indomain_max),
        int_search([sec_objective], smallest, indomain_min),
        int_search([objective], smallest, indomain_min),
    ]);

solve
  :: durThenStartThenMode
  % :: restart_constant(1000)
  minimize objective+sec_objective;


%-----------------------------------------------------------------------------%
output ["Start times = ", show(s), "\n",
        "nb_paused= ", show(nb_paused), "\n",
        "makespan = ", show(objective), "\n",
        "sec objective = ", show(sec_objective), "\n",
        "rr=", show(rr)];

%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%
