include "cumulative.mzn";
include "element.mzn";
include "my_cumulative.mzn";

%---------------------------------------------------------------------------%
% Mapping of parameters from the input data to the used parameter names in the
% publication

% Number of resources
int: n_res;
int: nb_res = n_res;

% Number of activities (tasks)
int: n_tasks;
int: nb_act = n_tasks;

% Fake Tasks (to handle notably the varying resource availability)
bool: include_fake_tasks;
int: max_duration_fake_task;
int: n_fake_tasks;
set of int: FakeAct=1..n_fake_tasks;
array[FakeAct] of 0..max_time: fakestart;
array[FakeAct] of 1..max_duration_fake_task: fakedur;
array[Res,FakeAct] of 0..max(rcap): fakereq;


% Number of modes (optional activities)
int: n_opt;
int: nb_mod = n_opt;

% Set of modes for each activity
array[Act] of set of Mod: modes;
array[Act] of set of Mod: mode = modes;

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
set of int: RRes = {k | k in Res where rtype[k] = 1};
set of int: NRes = {k | k in Res where rtype[k] = 2};
set of int: Act = 1..nb_act;    % Set of activities
set of int: Mod = 1..nb_mod;    % Set of modes
array[Act] of set of Act: succ; % Set of successors

int: min_mrreq = min(k in Res, m in Mod)(mrreq[k,m]);
int: max_mrreq = max(k in Res, m in Mod)(mrreq[k,m]);

%---------------------------------------------------------------------------%
% Variables
int: UB = sum(i in Act)(max([mdur[m] | m in mode[i]]));
int: max_time;


array[Mod] of var bool: mrun;
array[Act] of var 0..max_time: start;
output["\(UB)"];


array[Act] of var min(mdur)..max(mdur): adur = [let {var {mdur[m] | m in mode[i]}: x} in x | i in Act];
array[Res,Act] of var min_mrreq..max_mrreq: arreq = array2d(Res,Act,[let {var {mrreq[k,m] | m in mode[i]}: x} in x | k in Res, i in Act]);
var 0..max_time: objective;

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


constraint forall(k in RRes)(
    if include_fake_tasks then
        cumulative(start++fakestart,
                   adur++fakedur,
                   [arreq[k,i] | i in Act]++[fakereq[k, i] | i in FakeAct],
                   rcap[k])
    else
        cumulative(start,
                   adur,
                   [arreq[k,i] | i in Act],
                   rcap[k])
    endif
);


constraint forall(k in RRes)(
    cumulative(start, adur, [arreq[k,i] | i in Act], rcap[k])
);


% Non-renewable resource constraints
constraint forall(k in NRes)(sum(i in Act)(arreq[k,i]) <= rcap[k]);

%---------------------------------------------------------------------------%
% Non-overlapping constraints (NOC)

% Options for NOC
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



% Objective constraints
%
bool: add_objective_makespan; % by default True. but by setting it false you can modify your objective function outside (i.e minizinc pyyhon)
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

solve
    :: durThenStartThenMode
    minimize objective+sec_objective;

ann: durThenStartThenMode = seq_search([
        int_search(start, smallest, indomain_min, complete),
        int_search(adur, smallest, indomain_min, complete),
        bool_search(mrun, input_order, indomain_max, complete)
    ]);

%---------------------------------------------------------------------------%
% Output

output [
    "start = \(start);\n",
    "mrun = \(mrun);\n",
    "objective = \(objective);\n"
];
