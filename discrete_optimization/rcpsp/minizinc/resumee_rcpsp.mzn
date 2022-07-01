

% Planning horizon
% int: t_max = sum(i in Tasks)(d[i]);     % End time of the planning horizon
% set of int: Times = 0..t_max-1;     % Possible start times

int: max_time;
set of int: Times = 0..max_time;     % Possible start times
-----------------------------------------------------------------------------%
% Model variables.

array [Tasks] of var Times: s;  % The start times
var 0..t_max: objective      ;  % The project duration (makespan)

%-----------------------------------------------------------------------------%
% Constraints.

% Precedence constraints    %
constraint
   forall ( i in Tasks, j in suc[i] )
   (
         s[i] + d[i] <= s[j]
   );

bool: include_fake_tasks;
int: max_duration_fake_task;
int: n_fake_tasks;
set of int: FakeAct=1..n_fake_tasks;
array[FakeAct] of 0..max_time: fakestart;
array[FakeAct] of 1..max_duration_fake_task: fakedur;
array[Res,FakeAct] of 0..max(rcap): fakereq;

constraint
  forall(r in Res)(cumulative(
                  [s[i] | i in Tasks ],
                  [d[i] | i in Tasks ],
                  [rr[r, i] | i in Tasks ],
                  rc[r]));
constraint
    forall(r in Res)(cumulative([ s[i] | i in Tasks ]++[fakestart[p] | p in FakeAct],
                                [ d[i] | i in Tasks ]++[fakedur[p] | p in FakeAct],
                                [ rr[r, i] | i in Tasks ]++[fakereq[r, p] | p in FakeAct],
                                rc[r]));

constraint forall ( i in Tasks where suc[i] == {} )
               (
                  s[i] + d[i] <= objective
               );

solve
   :: int_search( s ++ [objective], input_order, indomain_min) %, complete )
   %:: int_search( s ++ [objective], smallest, indomain_min, complete) %, complete )
   minimize objective+sec_objective;
