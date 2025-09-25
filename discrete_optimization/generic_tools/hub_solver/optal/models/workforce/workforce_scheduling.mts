import * as CP from '@scheduleopt/optalcp';
import * as fs from 'fs';
import { strict as assert } from 'assert';
import { getBoolOption } from '../../utils/utils.mts';

// Helper to parse a string option from command line arguments
function getStringOption(name: string, defaultValue: string, args: string[]): string {
  const index = args.indexOf(name);
  if (index !== -1 && index + 1 < args.length) {
    const value = args[index + 1];
    args.splice(index, 2); // Remove the option and its value
    return value;
  }
  return defaultValue;
}


/**
 * Reads and parses the instance data from a JSON file.
 */
function readInstance(filename: string) {
    const rawData = fs.readFileSync(filename, 'utf-8');
    return JSON.parse(rawData);
}

/**
 * Defines the CP model and returns it along with the variables needed for solution extraction.
 */
function defineModelAndModes(filename: string,
    model_dispersion: boolean
) {
    let instance = readInstance(filename);
    let model = new CP.Model('alloc-scheduling');

    const tasks = instance.tasks;
    const teams = instance.teams;
    const tasksData = instance.tasks_data;
    const compatibleTeams = instance.compatible_teams;
    const successors = instance.successors;
    const sameAllocation = instance.same_allocation;
    const horizon = 10000;

    let taskVars: { [key: string]: any } = {};

    // 1. Create task and mode variables
    for (const task of tasks) {
        let duration = tasksData[task].duration;
        let startWindow = instance.start_window[task] || [0, horizon];
        let endWindow = instance.end_window[task] || [duration, horizon];

        taskVars[task] = {
            main: model.intervalVar({ start: startWindow, end: endWindow, length: duration, name: `task_${task}` }),
            modes: [],
            teamVars: {}
        };

        const compatible = compatibleTeams[task] || teams;
        for (const team of compatible) {
            let mode = model.intervalVar({ length: duration, optional: true, name: `task_${task}_team_${team}` });
            taskVars[task].modes.push(mode);
            taskVars[task].teamVars[team] = mode;
        }
    }

    // 2. Add constraints
    for (const task of tasks) {
        model.alternative(taskVars[task].main, taskVars[task].modes);
    }

    for (const task in successors) {
        for (const succ of successors[task]) {
            model.endBeforeStart(taskVars[task].main, taskVars[succ].main);
        }
    }

    for (const group of sameAllocation) {
         for (let i = 0; i < group.length - 1; i++) {
             const task1 = group[i];
             const task2 = group[i+1];
             const commonTeams = (compatibleTeams[task1] || teams).filter((t: any) => (compatibleTeams[task2] || teams).includes(t));
             for (const team of commonTeams) {
                 model.constraint(model.eq(model.presenceOf(taskVars[task1].teamVars[team]),
                                  model.presenceOf(taskVars[task2].teamVars[team])));
             }
         }
    }

    const teamOptionalTasks: {[key: string]: CP.IntervalVar[]} = {};
    for (const team of teams) {
        teamOptionalTasks[team] = [];
    }
    for (const task of tasks) {
        for(const team in taskVars[task].teamVars) {
            teamOptionalTasks[team].push(taskVars[task].teamVars[team]);
        }
    }
    for (const team of teams) {
        model.noOverlap(teamOptionalTasks[team]);
    }

    // Now with calendar :
    for (const team of teams) {
        const intervalsForTeam = [...teamOptionalTasks[team]];
        const calendar = instance.calendar[team];

        if (calendar) {
            // Calendar specifies AVAILABLE slots. We create fixed intervals for UNAVAILABLE slots.
            let lastAvailableEnd = 0;
            //calendar.sort((a: number[], b: number[]) => a[0] - b[0]);

            for (const availableSlot of calendar) {
                const availableStart = availableSlot[0];
                const availableEnd = availableSlot[1];

                if (availableStart > lastAvailableEnd) {
                    const unavailableDuration = availableStart - lastAvailableEnd;
                    const unavailability = model.intervalVar({
                        start: lastAvailableEnd,
                        length: unavailableDuration,
                        name: `unavail_${team}_${lastAvailableEnd}`
                    });
                    intervalsForTeam.push(unavailability);
                }
                lastAvailableEnd = availableEnd;
            }

            // Add final unavailability block from the last slot to the horizon
            if (lastAvailableEnd < horizon) {
                const unavailableDuration = horizon - lastAvailableEnd;
                const unavailability = model.intervalVar({
                    start: lastAvailableEnd,
                    length: unavailableDuration,
                    name: `unavail_${team}_${lastAvailableEnd}_final`
                });
                intervalsForTeam.push(unavailability);
            }
        }
        model.noOverlap(intervalsForTeam);
    }

    // 3. Define the objective: Minimize the number of teams used
    let teamsUsed: CP.IntVar[] = [];
    for (let i = 0; i < teams.length; ++i){
        teamsUsed.push(model.intVar({range: [0, 1], name: `used_${teams[i]}`}));
        const teamTasks = teamOptionalTasks[teams[i]];
        for (let j=0; j<teamTasks.length; ++j){
            model.constraint(teamsUsed[i].ge(teamTasks[j].presence()));
        }
        model.max(teamTasks.map(task => model.presenceOf(task))).eq(teamsUsed[i]);
    }
    let nbTeamsUsed = model.sum(teamsUsed);
    let detailed_steps: CP.CumulExpr[] = []

    for (let i = 0; i < tasks.length; ++i) {
        let task = tasks[i];
        detailed_steps.push(model.stepAtStart(taskVars[task].main, -1)); // Use one resource
        detailed_steps.push(model.stepAtEnd(taskVars[task].main, 1)); // Put back re
    }
    detailed_steps.push(model.stepAt(0, nbTeamsUsed));
    //console.log(`${detailed_steps}`);
    model.cumulGe(model.cumulSum(detailed_steps), 0); // Cumulative with variable resource.
    let dispersion;
    let workload_per_team: CP.IntVar[] = [];
    if (model_dispersion){
        for (let i=0; i<teams.length;i++){
            const teamTasks = teamOptionalTasks[teams[i]];
            workload_per_team.push(model.intVar({range:[0, 10000]}))
            let workloadExpr = model.sum(teamTasks.map(
            (taskMode: CP.IntervalVar) => taskMode.lengthOr(0)));
            model.constraint(model.implies(teamsUsed[i].eq(1), model.eq(workloadExpr, workload_per_team[i])));
        }
        dispersion = model.max(workload_per_team).minus(model.min(workload_per_team));
        model.constraint(dispersion.ge(0));
    }
    else{
        dispersion = 0;
    }

    // 5. Set the final, prioritized objective
    if (model_dispersion){
            model.minimize(nbTeamsUsed.times(10000).plus(dispersion));
            console.log(`Dispersion ${dispersion}`);
    }
    else{
            model.minimize(nbTeamsUsed);
    }
    // Return all necessary components for different execution modes
    return { model, taskVars, teams, tasks, teamsUsed, workload_per_team, dispersion};
}

function defineModelRelaxed(filename: string) {
    let instance = readInstance(filename);
    let model = new CP.Model('alloc-scheduling');

    const tasks = instance.tasks;
    const teams = instance.teams;
    const tasksData = instance.tasks_data;
    const compatibleTeams = instance.compatible_teams;
    const successors = instance.successors;
    const sameAllocation = instance.same_allocation;
    const horizon = 10000;

    let taskVars: { [key: string]: any } = {};

    // 1. Create task and mode variables
    for (const task of tasks) {
        let duration = tasksData[task].duration;
        let startWindow = instance.start_window[task] || [0, horizon];
        let endWindow = instance.end_window[task] || [duration, horizon];

        taskVars[task] = {
            main: model.intervalVar({ start: startWindow, end: endWindow, length: duration, name: `task_${task}` }),
            modes: [],
        };
    }


    for (const task in successors) {
        for (const succ of successors[task]) {
            model.endBeforeStart(taskVars[task].main, taskVars[succ].main);
        }
    }

    let nbTeamsUsed = model.intVar({range: [0, teams.length], name: `used_teams`});
    let detailed_steps: CP.CumulExpr[] = []

    for (let i = 0; i < tasks.length; ++i) {
        let task = tasks[i];
        detailed_steps.push(model.stepAtStart(taskVars[task].main, -1)); // Use one resource
        detailed_steps.push(model.stepAtEnd(taskVars[task].main, 1)); // Put back re
    }
    detailed_steps.push(model.stepAt(0, nbTeamsUsed));
    //console.log(`${detailed_steps}`);
    model.cumulGe(model.cumulSum(detailed_steps), 0); // Cumulative with variable resource.
    model.minimize(nbTeamsUsed);
    // Return all necessary components for different execution modes
    return { model, taskVars, teams, tasks, nbTeamsUsed};
}

/**
 * A function usable for CP.benchmark(), which only needs the model.
 */
function defineModelForBenchmark(filename: string): CP.Model {
  return defineModelAndModes(filename, true).model;
}

/**
 * Runs the solver and writes the detailed solution to a JSON file.
 * This function now correctly extracts solution data from the active modes.
 */
async function runAndExportJson(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters, modelDispersion: boolean) {
  params.searchType
  const { model, taskVars, teams, tasks, teamsUsed, workload_per_team} = defineModelAndModes(inputFilename, modelDispersion);
  const result = await CP.solve(model, params);
  const solution = result.bestSolution;

  // Use arrays to store results, ensuring order is preserved.
  const startTimes: number[] = [];
  const endTimes: number[] = [];
  const teamAssignments: string[] = [];
  if (solution) {
    // Iterate over the canonical list of tasks to maintain order.
    const workload_nz: (number|null)[] = [];
    if (modelDispersion){
        for (let i=0; i<workload_per_team.length; i++){
            if (solution.getValue(teamsUsed[i]) == 1){
                workload_nz.push(solution.getValue(workload_per_team[i]));
            }
        }
        const dispersion = Math.max(...(workload_nz as number[]))-Math.min(...(workload_nz as number[]));
        console.log(`Dispersion is ${dispersion}`);
        console.log(`Workload is ${workload_nz}`);
    }

    for (const task of tasks) {
      let assignedTeam = 'N/A';
      let startTime = -1;
      let endTime = -1;

      // Find which team mode was selected by the solver for this task.
      for (const team of teams) {
          const modeVar = taskVars[task].teamVars[team];
          // If the mode variable exists and is present in the solution...
          if (modeVar && !solution.isAbsent(modeVar)) {
              assignedTeam = team;
              // Get the start time directly from the present mode variable.
              startTime = solution.getStart(modeVar) as number;
              endTime = solution.getEnd(modeVar) as number;
              break; // Found the active mode, so we can stop searching for this task.
          }
      }
      startTimes.push(startTime);
      endTimes.push(endTime);
      teamAssignments.push(assignedTeam);
    }
  }

  const output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes,
    endTimes,
    teamAssignments
  };

  fs.writeFileSync(outputJSON, JSON.stringify(output, null, 2));
  console.log(`✅ Solution exported to ${outputJSON}`);
}

async function runLexicoExportJson(inputFilename: string, outputJSON: string,
                                   params: CP.BenchmarkParameters) {

  //const d_relaxed = defineModelRelaxed(inputFilename);
  //const result__ = await CP.solve(d_relaxed.model, params);
  //const solution__ = result__.bestSolution;
  //let nbt = solution__?.getObjective() as number;


  const { model, taskVars, teams, tasks, teamsUsed, workload_per_team, dispersion} = defineModelAndModes(inputFilename,
    false
  );
  //model.constraint(model.sum(teamsUsed).ge(nbt));
  model.sum(teamsUsed).minimize();
  const result_ = await CP.solve(model, params);
  const solution_ = result_.bestSolution;
  let primary_obj = solution_?.getObjective() as number;
  solution_?.setObjective(10000);
  const d = defineModelAndModes(inputFilename, true);
  console.log(`${primary_obj}`);
  d.model.constraint(d.model.sum(d.teamsUsed).le(primary_obj));
  d.model.minimize(d.dispersion);
  const result = await CP.solve(d.model, params); //, solution);
  const solution = result.bestSolution;
  // Use arrays to store results, ensuring order is preserved.
  const startTimes: number[] = [];
  const endTimes: number[] = [];
  const teamAssignments: string[] = [];
  const recomputedWorkload: { [key: string]: any } = {};
  if (solution) {
    // Iterate over the canonical list of tasks to maintain order.
    const workload_nz: number[] = [];

    for (let i=0; i<d.workload_per_team.length; i++){
        if (solution.getValue(d.teamsUsed[i]) == 1){
            workload_nz.push(solution.getValue(d.workload_per_team[i]) as number);
        }
        else{
            workload_nz.push(solution.getValue(d.workload_per_team[i]) as number);
        }
    }
    const dispersion = Math.max(...workload_nz)-Math.min(...workload_nz);
    console.log(`Dispersion is ${dispersion}`);
    console.log(`Workload is ${workload_nz}`);

    for (const team of teams) {
        recomputedWorkload[team] = 0
    };
    for (const task of tasks) {
      let assignedTeam = 'N/A';
      let startTime = -1;
      let endTime = -1;

      // Find which team mode was selected by the solver for this task.
      for (const team of teams) {
          const modeVar = d.taskVars[task].teamVars[team];
          // If the mode variable exists and is present in the solution...
          if (modeVar && !solution.isAbsent(modeVar)) {
              assignedTeam = team;
              // Get the start time directly from the present mode variable.
              startTime = solution.getStart(modeVar) as number;
              endTime = solution.getEnd(modeVar) as number;
              recomputedWorkload[assignedTeam]+=endTime-startTime;
              break; // Found the active mode, so we can stop searching for this task.
          }
      }
      startTimes.push(startTime);
      endTimes.push(endTime);
      teamAssignments.push(assignedTeam);
    }
  }

  const output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes,
    endTimes,
    teamAssignments,
    recomputedWorkload
  };

  fs.writeFileSync(outputJSON, JSON.stringify(output, null, 2));
  console.log(`✅ Solution exported to ${outputJSON}`);
}

const params: CP.BenchmarkParameters = {
    usage: "Usage: node workforce.mjs [OPTIONS] INPUT_FILE1 [INPUT_FILE2] ..\n\n" +
           "Output options:\n" +
           "  --output-json <filename>     Write the solution to a JSON file.\n" +
           "                                 (Only single input file is supported with this option).\n\n" +
           "Modeling options:\n" +
           "  --run-lexico                 Use a two-stage lexicographical optimization (teams, then workload dispersion).\n" +
           "  --model-dispersion        In single-objective mode, minimize a weighted sum of teams and workload dispersion."
};
const commandLineArgs = process.argv.slice(2);
console.log(`Argv : ${process.argv}`);
console.log(`Cml : ${commandLineArgs}`);

const workforceJsonFilename = getStringOption("--output-json", "", commandLineArgs);

// Parse the new boolean flags
const runLexico = getBoolOption("--run-lexico", commandLineArgs);
const modelDispersion = getBoolOption("--model-dispersion", commandLineArgs);

// The model can be run in two modes, similar to the flexible-jobshop example.
if (workforceJsonFilename === "") {
    // Mode 1: Run benchmark and print summary table to console.
    const remainingArgs = CP.parseSomeBenchmarkParameters(params, commandLineArgs);
    CP.benchmark(defineModelForBenchmark, remainingArgs, params);
} else {
    // Mode 2: Solve a single instance and export the full solution to a JSON file.
    const remainingArgs = CP.parseSomeParameters(params, commandLineArgs);
    console.log(`Run lexico ${runLexico}, Model Disr ${modelDispersion}`);
    console.log(`${commandLineArgs}`);
    if (runLexico) {
        console.log("Running with lexicographical optimization...");
        runLexicoExportJson(remainingArgs[0], workforceJsonFilename, params);
    } else {
        console.log(`Running with single-objective optimization (model dispersion: ${modelDispersion})...`);
        runAndExportJson(remainingArgs[0], workforceJsonFilename, params, modelDispersion);
    }
}
