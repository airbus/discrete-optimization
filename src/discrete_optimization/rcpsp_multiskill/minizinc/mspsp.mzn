% Author: Kenneth Young 2017
% Multi-Skill Project Scheduling Problem
% Constraint Programming model
% Adapted by Airbus to fit the library needs.

include "cumulative.mzn";
include "disjunctive.mzn";
include "redefinitions.mzn";
include "chuffed.mzn";

0..1: full_output;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% INSTANCE INITIALISATION
int: nActs;
int: nResources;
int: nSkills;
int: nPrecs;
int: nUnrels; % number of unrelated activites wrt to the precedence graph
int: mint;  % minumum makespan
int: maxt;  % maximum makespan

set of int: ACT = 1..nActs;
set of int: RESOURCE = 1..nResources;
set of int: SKILL = 1..nSkills;
set of int: PREC = 1..nPrecs;
set of int: UNREL = 1..nUnrels;
set of int: TIME = 0..maxt;

array[ACT] of set of RESOURCE: USEFUL_RES; % useful resources for each act
array[RESOURCE] of set of ACT: POTENTIAL_ACT; % potential activities for each res

array[ACT] of int: dur; % duration
array[ACT,SKILL] of int: sreq; % task's skill requirement
array[RESOURCE,SKILL] of bool: mastery; % resource's skill mastery
array[PREC] of ACT: pred; % predecessor activity
array[PREC] of ACT: succ; % successor activity
array[UNREL] of ACT: unpred; % unrelated predecessor activity
array[UNREL] of ACT: unsucc; % unrelated successor activity
array[SKILL] of 0..nResources: scap =
    [sum(r in RESOURCE)(bool2int(mastery[r,s])) | s in SKILL]; % capacity of each skill


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% DECISION VARIABLES
array[ACT] of var TIME: start; % start time
array[ACT,RESOURCE] of var bool: assign; % assignment of resources to activities
array[ACT,RESOURCE,SKILL] of var bool: contrib; % skill contribution assignment
array[UNREL] of var bool: overlap; % overlapping of unrelated activities

var TIME: makespan= start[nActs];  % makespan
var int: objective;
bool: add_objective_makespan;
constraint objective>=0;
constraint sec_objective>=0;
constraint if add_objective_makespan then objective == makespan
           endif;
var int: sec_objective;
bool: ignore_sec_objective;
constraint if ignore_sec_objective then sec_objective=0 else true endif;
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% CONSTRAINTS

% Precedence constraint
constraint forall(p in PREC)(
    start[pred[p]] + dur[pred[p]] <= start[succ[p]]
);

% Unary resource constraint
constraint
  forall (
    u in UNREL
  )(
  let {int: i = unpred[u],
       int: j = unsucc[u]} in
        if exists( s in SKILL )( sreq[i,s] + sreq[j,s] > scap[s] ) then
             (     overlap[u]  -> start[i] + dur[i] <= start[j] ) /\
             ( not(overlap[u]) -> start[j] + dur[j] <= start[i] )
        else
            (
              not(overlap[u]) <-> (start[i] + dur[i] <= start[j] \/
                                   start[j] + dur[j] <= start[i])
            ) /\ (
              forall(
                r in USEFUL_RES[i] intersect USEFUL_RES[j]
              )(
                (assign[i,r] /\ assign[j,r]) -> not(overlap[u])
              )
            )
        endif
  );

% Skill constraint: Skill requirements are satisfied
constraint forall(a in ACT, s in SKILL where sreq[a,s]>0)(
  sum(r in USEFUL_RES[a])(contrib[a,r,s]) == sreq[a,s]
);

% Non-Multi-Tasking constraint: Maximum of one contribution by each activity
constraint forall(a in ACT, r in USEFUL_RES[a])(
  sum(s in SKILL where mastery[r,s]==true /\ sreq[a,s]>0)
     (contrib[a,r,s]) <= 1
);

% Skill constraint: Resources only use skills they have mastered
constraint forall(a in ACT, r in USEFUL_RES[a], s in SKILL)(
  contrib[a,r,s] <= bool2int(mastery[r,s])
);

% Linking constraint: Resources only contribute to activities they are assigned
constraint forall(a in ACT, r in USEFUL_RES[a], s in SKILL where sreq[a,s]>0) (
  contrib[a,r,s] <= bool2int(assign[a,r])
);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% REDUNDANT CONSTRAINTS

% Skill's used do not exceed their capacity
constraint forall(s in SKILL)(
  cumulative(start, dur, [ sreq[a,s] | a in ACT ], scap[s] )
%     :: ttef_filt(true)% :: name("skill\(s)")
);

% Resources required by each activity is capped by total # of resources
constraint cumulative(start, dur, [ sum(s in SKILL)(sreq[a,s]) | a in ACT ],
  nResources ) :: ttef_filt(true);% :: name("resources");


% Force assign variables which cannot be satisfied to false
constraint forall(a in ACT, r in RESOURCE diff USEFUL_RES[a])(
  assign[a,r] == false
);

% Force contrib variables which cannot be satisfied to false
constraint forall(a in ACT) (
  % if resource isn't useful for activity then it can't contribute
  forall(r in RESOURCE diff USEFUL_RES[a], s in SKILL) (
      contrib[a,r,s] == 0) /\

  % if activity does not require this skill then contribution is false
  forall(r in RESOURCE, s in SKILL where sreq[a,s]==0)(
      contrib[a,r,s] == 0)
);
constraint forall(r in RESOURCE, s in SKILL where mastery[r,s]==false)(
  % if resource doesn't have this skill then it can't contribute
  forall(a in ACT)(
      contrib[a,r,s] == 0)
);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% OBJECTIVE
% solve minimize objective;

ann: default_s;

% Basic Searches
ann: start_s = int_search(start, first_fail, indomain_min, complete);
ann: overlap_s = int_search(overlap, largest, indomain_min, complete);
ann: assign_s = bool_search([ assign[a,r] | a in ACT, r in RESOURCE ],
                           input_order, indomain_max, complete);

ann: contrib_s = bool_search([ contrib[a,r,s]
                           | a in ACT, r in RESOURCE, s in SKILL ],
                           input_order, indomain_max, complete);

% Sequencial Searches
ann: assign_Then_start = seq_search([ assign_s, start_s ]);
ann: contrib_Then_start = seq_search([ contrib_s, start_s ]);
ann: start_Then_assign = seq_search([ start_s, assign_s ]);
ann: start_Then_contrib = seq_search([ start_s, contrib_s ]);
ann: overlap_Then_assign_Then_start =
              seq_search([ overlap_s, assign_s, start_s ]);
ann: overlap_Then_contrib_Then_start =
              seq_search([ overlap_s, contrib_s, start_s ]);

% Priority Searches
ann: priority_search;

ann: priority_input_order =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                  input_order, indomain_max, complete)])
                   | a in ACT ],
                   input_order, complete);
ann: priority_smallest =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                  input_order, indomain_max, complete)])
                   | a in ACT ],
                   smallest, complete);

array[RESOURCE] of var TIME: res_load = [
    sum(a in ACT)( dur[a] * assign[a,r] )
| r in RESOURCE ];

ann: priority_smallest_load =
    priority_search(start,
        [
            seq_search([
                int_search([start[a]], input_order, indomain_min, complete),
                priority_search(res_load,
                    [
                        seq_search([
                            bool_search([assign[a,r]], input_order, indomain_max, complete),
                            bool_search([contrib[a,r,s] | s in SKILL],  input_order, indomain_max, complete)
                        ])
                    | r in RESOURCE],
                    smallest, complete)
            ])
        | a in ACT ],
        smallest, complete);

ann: priority_smallest_largest =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                  input_order, indomain_max, complete)])
                   | a in ACT ],
                   smallest_largest, complete);

ann: priority_first_fail =
  priority_search(start,
                  [seq_search([
                      int_search([start[a]], input_order, indomain_min, complete),
                      bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                  input_order, indomain_max, complete)])
                   | a in ACT ],
                   first_fail, complete);


ann: my_search;

% Solve
solve :: my_search
minimize objective+sec_objective;
% satisfy;


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% OUTPUT
output
if full_output == 0 then
  ["makespan = " ++ show(makespan) ++ "\n"]
elseif full_output == 1 then
  ["makespan = " ++ show(makespan) ++ "\n"] ++
  ["sec obj ="++show(sec_objective)++"\n"] ++
  ["start times = " ++ show(start) ++ "\n"] ++
  ["assignment = " ++ show(assign) ++ "\n"] ++
  ["contribution = " ++ show(contrib) ++ "\n"]
else
  [""]
endif
;
