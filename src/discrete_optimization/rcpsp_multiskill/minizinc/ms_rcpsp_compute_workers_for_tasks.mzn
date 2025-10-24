% Includes

include "cumulative.mzn";
include "element.mzn";

bool: one_ressource_per_task;
bool: one_worker_type_per_task;
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

array[Mod] of int: dur;
array[Mod] of int: mdur = dur;
array[Mod] of var bool: mrun;

array[Res, Mod] of int: rreq;
array[Res, Mod] of int: mrreq = rreq;


array[Mod] of Act: mact = [sum(i in Act where m in mode[i])(i) | m in Mod];

%---------------------------------------------------------------------------%
% Parameters
int: nb_res;
set of int: Res = 1..nb_res; % Set of resources
array[Res] of int: rcap;     % Resource capacity
array[Res] of int: rtype;    % Resource type (1: renewable; 2: non-renewable)


int: nb_skill;
set of int: Skill = 1..nb_skill; % set of skills

% Worker Units
int: nb_units;
set of int: Units = 1..nb_units; % set of units
array[Units, Skill] of int: skillunits;
array[Units, Act] of var 0..1: unit_used;
bool: consider_units;


array[Skill, Mod] of int: skillreq;
int: min_skillreq = min(k in Skill, m in Mod)(skillreq[k,m]);
int: max_skillreq = max(k in Skill, m in Mod)(skillreq[k,m])+10;
array[Skill, Act] of var min_skillreq..max_skillreq: array_skills_required = array2d(Skill, Act,
                                                                                     [let {var {skillreq[k, m] | m in mode[i]}: x} in x |
                                                                                      k in Skill, i in Act]);
int: nb_worker_type;
set of int: WORKER_TYPE = 1..nb_worker_type;
array[WORKER_TYPE, Skill] of int: skills_worker_type;
array[WORKER_TYPE] of int: capacity_worker_type;
int: max_capacity = max(capacity_worker_type);
array[WORKER_TYPE, Act] of var 0..max_capacity: worker_type_used;
bool: consider_worker_type;

constraint(
if consider_worker_type then
    (forall(i in Act, s in Skill)(
        if array_skills_required[s, i]>0 then
            sum(wt in WORKER_TYPE)(worker_type_used[wt, i]*skills_worker_type[wt, s])>=array_skills_required[s, i]
        endif
        )
    )
else
    forall(wt in WORKER_TYPE, i in Act)(worker_type_used[wt, i]=0)
endif);
constraint forall(wt in WORKER_TYPE, i in Act)(worker_type_used[wt, i]<=capacity_worker_type[wt]);

% Prune.
constraint forall(wt in WORKER_TYPE, i in Act)(
    let{var set of Skill: sk = {skl | skl in Skill
        where skills_worker_type[wt, skl]>0 /\ array_skills_required[skl, i]>0}}
        in(
            if sk=={} then
                worker_type_used[wt, i]=0
            endif
        )
);


constraint(
    if one_ressource_per_task then
        forall(i in Act)(sum(u in Units)(unit_used[u, i])<=1)
    endif);

constraint(
    if one_worker_type_per_task then
        forall(i in Act)(sum(wt in WORKER_TYPE)(bool2int(worker_type_used[wt, i]>0))<=1)
    endif);


constraint
    if consider_units then
        forall(s in Skill, i in Act)(
            if array_skills_required[s, i]>0 then
                sum(w in Units)(skillunits[w, s]*unit_used[w, i])>=array_skills_required[s, i]
            endif) % skills used overachieve
                    % the need of the skill.
    endif;

constraint
    if consider_units = false then
        forall(w in Units, i in Act)(
            unit_used[w, i] = 0
        )
    endif;

% Prune.
constraint forall(w in Units, i in Act)(
    let{var set of Skill: sk = {skl | skl in Skill where skillunits[w, skl]>0 /\ array_skills_required[skl, i]>0}}
        in(
            if sk=={} then
                unit_used[w, i]=0
            endif
        )
);

constraint
    if exact_skills_need /\ consider_units then
        forall(s in Skill, i in Act)(
            if array_skills_required[s, i]>0 then
                sum(w in Units)(skillunits[w, s]*unit_used[w, i])==array_skills_required[s, i]
            endif)
    endif;


var int: overskill_unit;
constraint overskill_unit=sum(s in Skill, i in Act)(
                         if array_skills_required[s,i]>0
                            then sum(w in Units)(skillunits[w, s]*unit_used[w, i])-array_skills_required[s,i]
                         else 0
                         endif);

var int: overskill_type;
constraint overskill_type=sum(s in Skill, i in Act)(
                         if array_skills_required[s, i]>0
                            then sum(wt in WORKER_TYPE)(worker_type_used[wt, i]*skills_worker_type[wt, s])-array_skills_required[s,i]
                         else 0
                         endif);

constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  forall(k in Skill)( array_skills_required[k,i] = skillreq[k, mi])
        )
    );


set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes
int: min_mrreq = if nb_res>0 then min(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;
int: max_mrreq = if nb_res>0 then max(k in Res, m in Mod)(mrreq[k,m]) else 0 endif;

%---------------------------------------------------------------------------%
% Variables
array[Res,Act] of var min_mrreq..max_mrreq: arreq = array2d(Res, Act,
                                                            [let {var {mrreq[k,m] | m in mode[i]}: x} in x |
                                                             k in Res, i in Act]);


%---------------------------------------------------------------------------%
% Constraints

    % Activities and mode constraints
    %
constraint
    forall(i in Act)(
        let { var mode[i]: mi } in (
            mrun[mi] = 1
        /\  sum(m in mode[i])(mrun[m]) = 1
        /\  forall(k in Res)( arreq[k,i] = mrreq[k, mi] )
        )
    );


%---------------------------------------------------------------------------%
% Solve item



solve
    :: modeThenWorker
    satisfy;

ann: modeThenWorker = seq_search([
        int_search([overskill_unit, overskill_type],
                    input_order, indomain_min),
        bool_search(mrun, input_order, indomain_max, complete),
        int_search(unit_used, input_order, indomain_min, complete),
        int_search(worker_type_used, input_order, indomain_min, complete)
    ]);

%---------------------------------------------------------------------------%
% Output

output [
    "unitused = \(unit_used);\n",
    "worker_type_used = \(worker_type_used);\n",
    "array_skills_required = \(array_skills_required);\n"
];
