%include "globals.mzn";
include "cumulative.mzn";
include "increasing.mzn";
include "disjunctive.mzn";
% include "chuffed.mzn";

%-----------------------------------------------------------------------------%
% Model parameters.


% Resources
%
int: n_res;                     % The number of resources
set of int: Res = 1..n_res;     % The set of all resources
array [Res] of int: rc;         % The resource capacities
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)
set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};


% Tasks
%
int: n_tasks;                           % The number of tasks
set of int: Tasks = 1..n_tasks;         % The set of all tasks
array [Tasks]      of set of int: suc;  % The task successors

int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills
array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Tasks] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Tasks,
                                                                                       [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                        k in Skill, i in Tasks]);
int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;
array[Units, Tasks] of var bool: unit_used;
array[Units, Tasks, PREEMPTIVE] of var bool: unit_used_preemptive;
bool: unit_usage_preemptive;

% Prune.
constraint forall(w in Units, t in Tasks)(
    let{set of Skill: sk = {skl | skl in Skill where skillunits[w, skl]>0 /\
       sum(m in modes[t])(skillreq[skl, m])>0}}
        in(
            sk=={}->unit_used[w, t]=false
        )
);

% Prune.
constraint forall(w in Units, i in Tasks, j in PREEMPTIVE)(
    if mask_unit_task[w, i] == true then
        unit_used_preemptive[w, i, j]=false
    endif
);
array[Units, Tasks] of bool: mask_unit_task = array2d(Units, Tasks,
        [let{set of Skill: sk = {skl | skl in Skill where skillunits[w, skl]>0 /\
             sum(m in modes[i])(skillreq[skl, m])>0}}
             in (sk=={}) | w in Units, i in Tasks]);


array[Res, Tasks] of bool: mask_res_task = array2d(Res, Tasks,
        [sum(m in modes[i])(mrreq[res,m])==0
         | res in Res, i in Tasks]);


constraint forall(w in Units, t in Tasks, j in PREEMPTIVE)(
                if unit_usage_preemptive=false then
                    unit_used_preemptive[w, t, j] = false
                endif);

constraint forall(w in Units, t in Tasks)(
                if unit_usage_preemptive then
                    unit_used[w, t] = false
                endif);


constraint
    forall(i in Tasks)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  forall(k in Skill)( array_skills_required[k,i] = skillreq[k, mi])
        )
    );


% skills used overachieve
% the need of the skill.
constraint forall(sk in Skill, i in Tasks)(
    if unit_usage_preemptive=false then
        if array_skills_required[sk, i]>0 then
            sum(w in Units)(skillunits[w, sk]*unit_used[w, i])>=array_skills_required[sk, i]
        endif
    endif);

constraint forall(sk in Skill, i in Tasks, n in PREEMPTIVE)(
    if unit_usage_preemptive then
        if array_skills_required[sk, i]>0 then
            d_preemptive[i, n] > 0 -> sum(w in Units)(skillunits[w, sk]*unit_used_preemptive[w, i, n])>=array_skills_required[sk, i]
        endif
    endif);



% Simplification (not necessary but cut search space for useless solution.)
% constraint forall(w in Units, i in Tasks, n in PREEMPTIVE)(
%         if unit_usage_preemptive then
%             d_preemptive[i, n] == 0 -> unit_used_preemptive[w, i, n]=false
%         endif);

bool: include_cumulative=true;
constraint forall(w in Units)(
     if include_cumulative /\ unit_usage_preemptive==false then
         if include_fake_tasks_unit then
                let{set of FakeActUnit: FTasks = {j | j in FakeActUnit where fakereq_unit[w, j]>0}}
                      in(
                         cumulative([s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE]++[fakestart_unit[p] | p in FTasks],
                                    [d_preemptive[i, j] | i in Tasks, j in PREEMPTIVE]++[fakedur_unit[p] | p in FTasks],
                                    [unit_used[w, i] | i in Tasks, j in PREEMPTIVE]++[fakereq_unit[w, p] | p in FTasks],
                                    1)
                      )
         else
                cumulative([s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                           [d_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                           [unit_used[w, i] | i in Tasks, j in PREEMPTIVE],
                           1)
        endif
     else
        true
     endif);

constraint forall(w in Units)(
     if include_cumulative /\ unit_usage_preemptive then
         if include_fake_tasks_unit then
                let{set of FakeActUnit: FTasks = {j | j in FakeActUnit where fakereq_unit[w, j]>0},
                    set of Tasks: TasksW = {i| i in Tasks where mask_unit_task[w, i]==false}}
                      in(
                         cumulative([s_preemptive[i, j] | i in TasksW, j in PREEMPTIVE]++[fakestart_unit[p] | p in FTasks],
                                    [d_preemptive[i, j] | i in TasksW, j in PREEMPTIVE]++[fakedur_unit[p] | p in FTasks],
                                    [unit_used_preemptive[w, i, j] | i in TasksW, j in PREEMPTIVE]++[fakereq_unit[w, p] | p in FTasks],
                                    1)
                      )
         else
                cumulative([s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                           [d_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                           [unit_used_preemptive[w, i, j] | i in Tasks, j in PREEMPTIVE],
                           1)
        endif
     else
        true
     endif);


% Fake Tasks (to handle notably the varying resource availability)
bool: include_fake_tasks_resource;
int: max_duration_fake_task_resource;
int: n_fake_task_resource;
set of int: FakeActRes=1..n_fake_task_resource;
array[FakeActRes] of 0..max_time: fakestart_resource;
array[FakeActRes] of 1..max_duration_fake_task_resource: fakedur_resource;
array[Res, FakeActRes] of 0..20: fakereq_resource;


bool: include_fake_tasks_unit;
int: max_duration_fake_task_unit;
int: n_fake_task_unit;
set of int: FakeActUnit=1..n_fake_task_unit;
array[FakeActUnit] of 0..max_time: fakestart_unit;
array[FakeActUnit] of 1..max_duration_fake_task_unit: fakedur_unit;
array[Units, FakeActUnit] of 0..2: fakereq_unit;


% Modes
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
int: min_mrreq = if length(Res)>0 then
                    min(k in Res, m in Mod)(mrreq[k,m])
                 else 0
                 endif;
int: max_mrreq = if length(Res)>0 then
                    max(k in Res, m in Mod)(mrreq[k,m])
                 else 0
                 endif;

array[Mod] of var bool: mrun;


array[Tasks] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Tasks];
array[Res, Tasks] of var min_mrreq..max_mrreq: arreq = array2d(Res, Tasks,
 [let {var {mrreq[k,m] | m in mode[i]}: x} in x | k in Res, i in Tasks]);

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
set of int: Times = 0..max_time;     % Possible start times


%-----------------------------------------------------------------------------%
% Model variables.
int: nb_preemptive;
set of int: PREEMPTIVE=1..nb_preemptive;

array [Tasks] of var Times: s;  % The start times
array [Tasks, PREEMPTIVE] of var Times: s_preemptive;  % The start times
array [Tasks, PREEMPTIVE] of var 0..max(mdur): d_preemptive;
constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]>=0);
constraint forall(i in Tasks)(d_preemptive[i, nb_preemptive]==0);

constraint forall(i in Tasks, j in 1..nb_preemptive-1)(d_preemptive[i, j]==0->d_preemptive[i, j+1]==0);

constraint forall(i in Tasks, j in PREEMPTIVE)(d_preemptive[i, j]<=adur[i]);
constraint forall(i in Tasks)(increasing(row(s_preemptive, i)));
constraint forall(i in Tasks)(s[i]==s_preemptive[i,1]);
constraint forall(i in Tasks)(if adur[i]>0 then d_preemptive[i, 1]>0 endif);
%constraint forall(i in Tasks)(decreasing([d_preemptive[i, j]|j in PREEMPTIVE]));
constraint forall(i in Tasks)(sum(row(d_preemptive, i))==adur[i]);
constraint forall(i in Tasks)(disjunctive(row(s_preemptive, i), row(d_preemptive, i)));
constraint forall(i in Tasks)(forall(j in 2..nb_preemptive)(
    (d_preemptive[i, j]==0) -> s_preemptive[i, j]==s_preemptive[i, j-1]+d_preemptive[i, j-1]
));

array[Tasks] of var bool: is_paused;
constraint forall(i in Tasks)(
        if possibly_preemptive[i]==false then
            d_preemptive[i, 1]==adur[i] /\ is_paused[i]==false
        endif
);
constraint forall(i in Tasks)(if possibly_preemptive[i]==false then
                    forall(j in 2..nb_preemptive)(d_preemptive[i, j]==0) /\ forall(j in 2..nb_preemptive, w in Units)(unit_used_preemptive[w, i, j]==0)
                    endif);


constraint forall(i in Tasks)(is_paused[i]<->d_preemptive[i, 1] < adur[i]);
constraint redundant_constraint(forall(i in Tasks)(is_paused[i]<->sum(j in 2..nb_preemptive)(d_preemptive[i,j])>0));
var int: nb_paused = sum(i in Tasks)(is_paused[i]);


int: max_preempted;
constraint nb_paused<=max_preempted;
% constraint nb_paused>=1; % debug purposes.
% constraint nb_paused>=3;

bool: strictly_disjunctive;
constraint if strictly_disjunctive then
            forall(i in Tasks)(forall(j in 1..nb_preemptive-1)(
                    (d_preemptive[i, j+1]>0 <-> s_preemptive[i,j+1]>s_preemptive[i,j]+d_preemptive[i, j])))
            endif; %avoid dummy solution

constraint forall(i in Tasks)(forall(j in 1..nb_preemptive-1)(
    (d_preemptive[i, j+1]>0 <-> s_preemptive[i,j+1]>s_preemptive[i,j]+d_preemptive[i, j]))); %avoid dummy solution
constraint forall(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>=s_preemptive[i, 1]+adur[i]);
% var int: nb_preemptivity;
% constraint sum(i in Tasks)(s_preemptive[i,nb_preemptive]+d_preemptive[i, nb_preemptive]>s_preemptive[i, 1]+adur[i])==nb_preemptivity;
var 0..max_time: objective;  % The project duration (makespan)


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
    (if exists(r in Res)(arreq[r, i] + arreq[r, j] > rc[r]) then
            s_preemptive[i, 1] + d_preemptive[i, 1] <= s_preemptive[j, 1]   \/
            s_preemptive[j, 1] + d_preemptive[j, 1] <= s_preemptive[i, 1]
     else
         true
     endif
    );


% Cumulative resource constraints
%
constraint forall(r in Res)(
    if include_cumulative then
        if include_fake_tasks_resource then
            let{set of FakeActRes: FTasks = {j | j in FakeActRes where fakereq_resource[r, j]>0},
                set of Tasks: TasksR = {i| i in Tasks where mask_res_task[r, i]=false}} in (
            cumulative([s_preemptive[i, j] | i in TasksR, j in PREEMPTIVE]++[fakestart_resource[j]|j in FTasks],
                       [d_preemptive[i, j] | i in TasksR, j in PREEMPTIVE]++[fakedur_resource[j]|j in FTasks],
                       [arreq[r, i] | i in TasksR, j in PREEMPTIVE]++[fakereq_resource[r, j] | j in FTasks],
                       rc[r])
            )
        else
            cumulative([s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                       [d_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
                       [arreq[r, i] | i in Tasks, j in PREEMPTIVE],
                       rc[r])
        endif
    else true

    endif);

% Non-renewable resource constraints    %
constraint forall(k in NRes)(sum(i in Tasks)(arreq[k, i]) <= rc[k]);


%-----------------------------------------------------------------------------%
% Objective.

%solve satisfy;



% Makespan constraints
%

bool: add_objective_makespan;
% by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)

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


constraint (if add_objective_makespan then
                forall ( i in Tasks where suc[i] == {} )
                    (s_preemptive[i, nb_preemptive] + d_preemptive[i, nb_preemptive] <= objective)
            endif);

ann: durThenStartThenMode = seq_search([
        int_search(col(s_preemptive, 1), smallest, indomain_min),
        % bool_search(is_paused, input_order, indomain_min),
        % int_search([s_preemptive[i, 2]| i in Tasks], first_fail, indomain_min),
        % int_search([nb_paused], input_order, indomain_min),
        % bool_search(mrun, input_order, indomain_max),
        int_search([d_preemptive[i, 1]| i in Tasks], input_order, indomain_max),
        int_search([sec_objective], smallest, indomain_min),
        int_search([objective], smallest, indomain_min),
        %int_search(s ++ [objective], smallest, indomain_min, complete)
    ]);

%ann: priority_smallest =
%    seq_search([
%        int_search([objective], smallest, indomain_min),
%        int_search(col(s_preemptive, 1), smallest, indomain_min),
%        priority_search(
%            [s_preemptive[i, j] | i in Tasks, j in PREEMPTIVE],
%            [seq_search([int_search([s_preemptive[a, j]],
%                                     input_order, indomain_min),
%                         %int_search([d_preemptive[a, j]],
%                         %            input_order, indomain_max),
%                         priority_search([sum(i in Tasks, k in PREEMPTIVE)(unit_used_preemptive[w, i, k]*d_preemptive[i, k])|
%                                          w in Units],
%                                         [bool_search([unit_used_preemptive[w,a,j] |
%                                                       w in Units],
%                                                    input_order, indomain_max)], smallest,
%                                                    complete)
%                                                    ]
%                      )
%                   | a in Tasks, j in PREEMPTIVE],
%                  smallest, complete)]
%             );

solve
   :: durThenStartThenMode
   % :: restart_constant(1000)
   minimize objective+sec_objective;

%-----------------------------------------------------------------------------%
output ["Start times=", show(s), "\n",
        "nb_paused=", show(nb_paused), "\n",
        "sec objective = ", show(sec_objective), "\n",
        "makespan=", show(objective), "\n"];

%-----------------------------------------------------------------------------%
%-----------------------------------------------------------------------------%
