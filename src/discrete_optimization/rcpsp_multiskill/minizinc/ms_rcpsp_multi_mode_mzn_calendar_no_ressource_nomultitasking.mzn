%---------------------------------------------------------------------------%
% Inspired by :
%
% Multi-mode Resource-constrained Project Scheduling (MRCPSP)
%---------------------------------------------------------------------------%
% Authors: Ria Szeredi
%          Andreas Schutt
%
% Copyright: Data61, CSIRO.
%---------------------------------------------------------------------------%
%
% This MRCPSP model was used in the publication of
%---------------------------------------------------------------------------%
% Includes

include "cumulative.mzn";
include "element.mzn";
include "my_cumulative.mzn";
include "chuffed.mzn";

%---------------------------------------------------------------------------%
% Mapping of parameters from the input data to the used parameter names in the
% publication


% Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

bool: one_ressource_per_task;
bool: ressource_always_here = true;



%----- Fake Tasks to handle calendars ----%
bool: include_fake_tasks_unit;
int: max_duration_fake_task_unit;
int: n_fake_task_unit;
set of int: FakeActUnit=1..n_fake_task_unit;
array[FakeActUnit] of 0..max_time: fakestart_unit;
array[FakeActUnit] of 1..max_duration_fake_task_unit: fakedur_unit;
array[Units, FakeActUnit] of 0..1: fakereq_unit;



% Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

    % Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;

array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters


int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills

int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;



array[Act,Units,Skill] of var bool: contrib; % skill contribution assignment
constraint forall(a in Act, s in Skill)(
  sum(w in Units)(contrib[a,w,s]) == array_skills_required[s, a]
);

% Skill constraint: Resources only use skills they have mastered
constraint forall(a in Act, w in Units, s in Skill)(
  contrib[a,w,s] <= skillunits[w,s]
);

% Non-Multi-Tasking constraint: Maximum of one contribution by each activity
constraint forall(a in Act, w in Units)(
  sum(s in Skill where skillunits[w,s]>0)
     (contrib[a,w,s]) <= 1
);

constraint forall(a in Act, w in Units, s in Skill)(
     (contrib[a,w,s]) <= unit_used[w, a]
);

constraint redundant_constraint(forall(a in Act, w in Units)(
     unit_used[w, a]->sum(s in Skill)(contrib[a, w, s])==1))
;


constraint forall(a in Act, s in Skill)(
        if sum(m in mode[a])(skillreq[s, m])==0 then
         forall(w in Units)(contrib[a,w,s]==0)
        endif);

constraint forall(a in Act, s in Skill)(
        if sum(m in mode[a])(skillreq[s, m])==0 then
         forall(w in Units)(contrib[a,w,s]==0)
        endif);
constraint forall(a in Act, w in Units)(
        if sum(m in mode[a], s in Skill)(skillreq[s, m]*skillunits[w,s])==0
            then unit_used[w,a] == 0
        endif
);


array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Act] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Act,
                                                                                     [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                      k in Skill, i in Act]);
array[Units, Act] of var bool: unit_used;

constraint(
if one_ressource_per_task then forall(i in Act)((sum(u in Units)(unit_used[u, i]))<=1)
endif);

%constraint(
%if one_ressource_per_task then forall(i in Act)((sum(u in Units)(unit_used[u, i]))==1)
%endif);

constraint(
if one_ressource_per_task then
forall(skill in Skill, m in Mod, u in Units)(
if skillunits[u, skill]<skillreq[skill, m] then unit_used[u, mact[m]]=0
endif
)
endif
);


constraint forall(s in Skill, i in Act)(
    sum(w in Units)(skillunits[w, s]*unit_used[w, i])>=array_skills_required[s, i]); % skills used overachieve
                                                                                     % the need of the skill.
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  forall(k in Skill)( array_skills_required[k,i] = skillreq[k, mi])
        )
    );


constraint forall(w in Units)(
    if include_fake_tasks_unit then
        let{set of FakeActUnit: FTasks = {j | j in FakeActUnit where fakereq_unit[w, j]>0}}
              in(
                 cumulative(start++[fakestart_unit[p] | p in FTasks],
                            adur++[fakedur_unit[p] | p in FTasks],
                            [unit_used[w, i] | i in Act]++[fakereq_unit[w, p] | p in FTasks],
                            1)
              )
    else
         cumulative(start, adur,
                    [unit_used[w, i] | i in Act],
                    1)
    endif);


set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes
array[Act] of set of Act: succ; % Set of successors


%---------------------------------------------------------------------------%
% Variables
int: max_time;
%int: UB = sum(i in Act)(max([mdur[m] | m in mode[i]]));
int: UB = max_time;

set of int: TIME_ARR=1..max_time+1;
set of int: TIME_ARR0=0..max_time;


array[Mod] of var bool: mrun;
array[Act] of var 0..UB: start;
array[Act] of var 0..UB: s=start;

output["\(UB)"];

array[Act] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Act];
var 0..UB: objective;
array[Units] of var TIME_ARR: res_load = [
    sum(a in Act)( adur[a] * unit_used[w,a] )
| w in Units ];


%---------------------------------------------------------------------------%
% Constraints

    % Activities and mode constraints
    %
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  sum(m in mode[i])(mrun[m]) = 1
        /\  adur[i] = mdur[mi]
        )
    );

    % Precedence constraints
    %
constraint forall(i in Act, j in succ[i])(start[i] + adur[i] <= start[j]);

    % Objective constraints
    %
%constraint redundant_constraint(
%    forall(i in Act where succ[i]={})(start[i] + adur[i] <= objective)
%);
%constraint objective = max(i in Act where succ[i]={})(start[i] + adur[i]);



%-----------------------------------------------------------------------------%
% Objective constraints
%
bool: add_objective_makespan;
% by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)

constraint
 if add_objective_makespan then
     redundant_constraint(
        forall(i in Act where succ[i]={})(start[i] + adur[i] <= objective)
    )
 else
    true
 endif;


constraint
    if add_objective_makespan then
        objective = max(i in Act where succ[i]={})(start[i] + adur[i])
    else
        true
    endif;
%-----------------------------------------------------------------------------%
% Second Objective
bool: ignore_sec_objective;
var -1000..1000: sec_objective;
% constraint sec_objective>=0; % trick to add a second part in the objective.
constraint (if ignore_sec_objective then
                sec_objective = 0
            else
                true
            endif);


%include "member.mzn";
% experiment some constraint, but not working.
%constraint forall(a in Act)(member([start[i]+adur[i]| i in Act], start[a]));
%constraint sec_objective>=sum(a in Act)(min([max([0, start[a]-start[i]-adur[i]])|i in Act]));

%---------------------------------------------------------------------------%
% Solve item
solve
    :: priority_smallest
    minimize objective+sec_objective;


constraint start[1]==0;
ann: durThenStartThenMode = seq_search([
        int_search(start++[objective], input_order, indomain_min),
        bool_search(unit_used, input_order, indomain_max),
        %int_search(adur, smallest, indomain_min, complete),
        %int_search(start, smallest, indomain_min, complete),
        bool_search(mrun, input_order, indomain_max, complete),
    ]);



ann: priority_smallest_load = priority_search(start,
        [
            seq_search([
                int_search([start[a]], input_order, indomain_min, complete),
                priority_search(res_load,
                    [
                        seq_search([
                            bool_search([unit_used[w,a]], input_order, indomain_max, complete),
                            bool_search([contrib[a,w,s] | s in Skill],  input_order, indomain_max, complete)
                        ])
                    | w in Units],
                    smallest, complete)
            ])
        | a in Act ],
        smallest, complete);

ann: priority_smallest_largest =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,w,s] | w in Units, s in Skill],
                                  input_order, indomain_max, complete)
                      ])
                   | a in Act ],
                   smallest_largest, complete);

ann: priority_first_fail =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,w,s] | w in Units, s in Skill],
                                  input_order, indomain_max, complete)
                      ])
                   | a in Act ],
                   first_fail, complete);

ann: priority_smallest =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,w,s] | w in Units, s in Skill],
                                  input_order, indomain_max, complete)])
                   | a in Act ],
                   smallest, complete);

%---------------------------------------------------------------------------%
% Output

output [
    "start = \(start);\n",
    "mrun = \(mrun);\n",
    "objective = \(objective);\n",
    "secobjective=\(sec_objective);\n",
    "array_skills_required = \(array_skills_required);\n",
    "contrib =\(contrib)",
];
