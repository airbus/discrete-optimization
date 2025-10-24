% Includes
include "cumulative.mzn";
include "member.mzn";

%---------------------------------------------------------------------------%
% Mapping of parameters from the input data to the used parameter names in the
% publication

% Number of resources
int: n_res;
int: nb_res = n_res;



bool: one_ressource_per_task;
bool: exact_skills_need;


% Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

% Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

% Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;
Act: source;
constraint(start[source]==0);

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;

array[Res, Mod] of int: rreq;
array[Res, Mod] of int: mrreq = rreq;


array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters

set of int: Res = 1..nb_res; % Set of resources
array[Res] of int: rcap;     % Resource capacity
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)


int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills

int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;


array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Act] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Act,
                                                                                     [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                      k in Skill, i in Act]);
array[Units, Act] of var bool: unit_used;

bool: include_cumulative_resource;
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
array[Units, FakeActUnit] of 0..1: fakereq_unit;



constraint(
    if one_ressource_per_task then
        forall(i in Act)(sum(u in Units)(unit_used[u, i])<=1)
    endif);

constraint(
    if one_ressource_per_task then
        forall(skill in Skill, m in Mod, u in Units)(
             if length(modes[mact[m]])==1 then
                if skillunits[u, skill] < skillreq[skill, m] then unit_used[u, mact[m]]=0
             endif
        endif
    )
endif
);


constraint forall(sk in Skill, i in Act)(
    if array_skills_required[sk, i]>0 then
        sum(w in Units)(skillunits[w, sk]*unit_used[w, i])>=array_skills_required[sk, i]
    endif); % skills used overachieve
            % the need of the skill.


% Prune.
constraint forall(w in Units, i in Act)(
    let{var set of int: sk = {skl | skl in Skill where skillunits[w, skl]>0 /\
       sum(m in modes[i])(skillreq[skl, m])>0}}
        in(
            if sk=={} then
                unit_used[w, i]=false
            endif
        )
);

constraint
    if exact_skills_need then
        forall(sk in Skill, i in Act)(
            if array_skills_required[sk, i]>0 then
            sum(w in Units)(skillunits[w, sk]*unit_used[w, i])==array_skills_required[sk, i]
            endif)
    endif;


% var int: overskill=0;
% constraint overskill=sum(s in Skill, i in Act)(
%                          if array_skills_required[s,i]>0
%                             then sum(w in Units)(skillunits[w, s]*unit_used[w, i])-array_skills_required[s,i]
%                          else 0
%                          endif);

%array[Units] of var int: res_loaded = [sum(a in Act)( adur[a] * bool2int(unit_used[w,a])) | w in Units ];
%constraint 4*(max(res_loaded)-min(res_loaded))<=max(res_loaded);


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
                            [bool2int(unit_used[w, i]) | i in Act]++[fakereq_unit[w, p] | p in FTasks],
                            1)
              )
    else
          cumulative(start, adur,
                     [unit_used[w, i] | i in Act],
                     1)
    endif);


set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};
set of int: Act = 1..nb_act;    % Set of activities
set of int: Tasks = Act;
set of int: Mod = 1..nb_mod;    % Set of modes
array[Act] of set of Act: succ; % Set of successors

int: min_mrreq = if nb_res>0 then min(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;
int: max_mrreq = if nb_res>0 then max(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;

%---------------------------------------------------------------------------%
% Variables
int: max_time;
set of int: TIME_ARR=1..max_time+1;
set of int: TIME_ARR0=0..max_time;

array[Mod] of var bool: mrun;
array[Act] of var TIME_ARR0: start;
array[Act] of var TIME_ARR0: s=start;
array[Act] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Act];
array[Res,Act] of var min_mrreq..max_mrreq: arreq = array2d(Res, Act,
                                                            [let {var {mrreq[k,m] | m in mode[i]}: x} in x |
                                                             k in Res, i in Act]);
array[Res, Act] of bool: mask_res_task = array2d(Res, Act,
                                                 [sum(m in modes[i])(mrreq[res,m])==0
                                                  | res in Res, i in Act]);
var TIME_ARR: objective;

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
        /\  forall(k in Res)( arreq[k,i] = mrreq[k, mi] )
        )
    );

    % Precedence constraints
    %
constraint forall(i in Act, j in succ[i])(start[i] + adur[i] <= start[j]);

% Renewable resource constraints
%

constraint
     if include_cumulative_resource then
         forall(k in RRes)(
             if include_fake_tasks_resource then
                let{set of Tasks: TasksR = {i| i in Act where mask_res_task[k, i]=false},
                    set of FakeActRes: FTasks = {j | j in FakeActRes where fakereq_resource[k, j]>0}}
                      in(
                         cumulative([start[i]| i in TasksR]++[fakestart_resource[p] | p in FTasks],
                                    [adur[i]| i in TasksR]++[fakedur_resource[p] | p in FTasks],
                                    [arreq[k,i] | i in TasksR]++[fakereq_resource[k, p] | p in FTasks],
                                    rcap[k])
                      )
             else
                cumulative(start, adur, [arreq[k,i] | i in Act], rcap[k])
             endif)
     endif;


% Non-renewable resource constraints
%
constraint forall(k in NRes)(sum(i in Act)(arreq[k,i]) <= rcap[k]);




%---------------------------------------------------------------------------%
% Non-overlapping constraints (NOC)

% Options for NOC
%
bool: opt_noc_half = false; % Use half-reification?

    % Check whether two activities are disjunct in all mode combinations
    %
function bool: are_act_disjunct(int: i, int: j) =
    forall(mi in mode[i], mj in mode[j])(are_mod_disjunct(mi, mj));

% Check whether two modes are in disjunction
%
function bool: are_mod_disjunct(int: mi, int: mj) =
    exists(k in RRes)(mrreq[k,mi] + mrreq[k,mj] > rcap[k]);

% Check whether two activities can be disjunct in any mode combination
%
function bool: can_act_be_disjunct(int: i, int: j) =
    exists(k in RRes)(
        can_act_be_disjunct_on_rres(i, j, k)
    );

% Check whether two activities can be disjunct on a renewable resource
% in any mode combination
%
function bool: can_act_be_disjunct_on_rres(int: i, int: j, int: k) =
    exists(mi in mode[i], mj in mode[j])(
        mrreq[k,mi] + mrreq[k,mj] > rcap[k]
    );


% Non-overlapping constraints for activities that are disjunct
% in any mode combination
%
predicate post_noc_disj(int: i, int: j) =
    if opt_noc_half then
        let {var bool: b} in (
            (    b  -> start[i] + adur[i] <= start[j])
        \/  (not(b) -> start[j] + adur[j] <= start[i])
        )
    else
        (start[i] + adur[i] <= start[j] \/ start[j] + adur[j] <= start[i])
    endif;

% Non-overlapping constraints for different mode combinations
% of potential disjunct activities
%
predicate post_noc_mode(int: i, int: j) =
    forall(mi in mode[i], mj in mode[j] where are_mod_disjunct(mi, mj))(
        if opt_noc_half then
            let {var bool: b} in (
                ((mrun[mi] /\ mrun[mj]) -> (
                        (    b  -> start[i] + mdur[mi] <= start[j])
                    \/  (not(b) -> start[j] + mdur[mj] <= start[i])
                    )
                )
                % For fixing the Boolean variable in the case that not both modes are run
            /\  (not(mrun[mi] /\ mrun[mj]) -> b)
            )
        else
            (mrun[mi] /\ mrun[mj]) ->
                (start[i] + mdur[mi] <= start[j] \/ start[j] + mdur[mj] <= start[i])
        endif
    );

% Non-overlapping constraints for different renewable
% of potential disjunct activities
%
predicate post_noc_rres(int: i, int: j) =
    forall(k in RRes where can_act_be_disjunct_on_rres(i, j, k))(
        if opt_noc_half then
            let {var bool: b} in (
                ((arreq[k,i] + arreq[k,j] > rcap[k]) -> (
                        (   (b) -> start[i] + adur[i] <= start[j])
                    \/  (not(b) -> start[j] + adur[j] <= start[i])
                    )
                )
                % For fixing the Boolean variable in the case that not both modes are run
            /\  (not(arreq[k,i] + arreq[k,j] <= rcap[k]) -> b)
            )
        else
            (arreq[k,i] + arreq[k,j] > rcap[k]) ->
                (start[i] + adur[i] <= start[j] \/ start[j] + adur[j] <= start[i])
        endif
    );

% Non-overlapping constraints for different renewable
% of potential disjunct activities via the cumulative constraint
%
predicate post_noc_cumu(int: i, int: j) =
    forall(k in RRes where can_act_be_disjunct_on_rres(i, j, k))(
        cumulative([start[i], start[j]], [adur[i], adur[j]], [arreq[k,i], arreq[k,j]], rcap[k])
    );

% Posting of the redundant NOCs
%
constraint redundant_constraint(
    forall(i,j in Act where i < j /\ can_act_be_disjunct(i, j))(
        if are_act_disjunct(i, j) then
            % Activities are disjunct all mode combinations
            post_noc_disj(i, j)
        else
            % Activities might be disjunct
            post_noc_rres(i, j)
        endif
    )
);




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
var int: sec_objective;
constraint sec_objective>=0; % trick to add a second part in the objective.
constraint (if ignore_sec_objective then
                sec_objective = 0
            else
                true
            endif);
%---------------------------------------------------------------------------%
% Solve item
ann: my_search;
solve
    :: my_search
    minimize objective+sec_objective;
bool: include_constraint_on_start_value;
constraint if include_constraint_on_start_value then
                forall(i in Act)(member([start[j]+adur[j]|j in Act]++[fakestart_resource[j]+fakedur_resource[j]| j in FakeActRes]++
                                 [fakestart_unit[j]+fakedur_unit[j]|j in FakeActUnit], start[i]))
           endif;
ann: priority_search;
ann: durThenStartThenMode = seq_search([
        int_search(start, smallest, indomain_min, complete),
        bool_search([unit_used[w, i]| i in Act, w in Units],
                    input_order,
                    indomain_max),
        %int_search(adur, smallest, indomain_min, complete),
        bool_search(mrun, input_order, indomain_max, complete),
    ]);

%constraint forall(w in Units, i in Act)(
%if sum(sk in Skill)(array_skills_required[sk, i])>0 then
%    let{var int: nb_skill=sum(sk in Skill where array_skills_required[sk, i]>0)(skillunits[w, sk])}
%    in(
%        unit_used[w, i]->forall(j in Units)(
%                                unit_used[j, i]==0 \/
%                                (sum(sk in Skill where array_skills_required[sk, i]>0)(skillunits[j, sk])<=nb_skill))
%     )
%endif);


%---------------------------------------------------------------------------%
% Output

output [
    "start = \(start);\n",
    "mrun = \(mrun);\n",
    "objective = \(objective);\n",
    "sec obj =\(sec_objective);\n",
    "array_skills_required = \(array_skills_required);\n"
];
