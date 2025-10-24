import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts';
import { strict as assert } from 'assert';
import * as fs from 'fs';

let redundantCumul = false;

type ModelWithVariables = {
  model: CP.Model;
  // For each operation (order by job and then by operation), all its possible modes:
  allModes: CP.IntervalVar[][];
}

function defineModelAndModes(filename: string): ModelWithVariables {
  // Read the input file into a string, possibly unzip it if it ends with .gz:
  let inputText = utils.readFile(filename);
  // The first line may contain 2 or 3 numbers. The third number should be ignored.
  // Therefore find end of the first line:
  let firstEOL = inputText.indexOf('\n');
  // Convert first line into an array of numbers:
  let firstLine = inputText.slice(0, firstEOL).trim().split(/\s+/).map(Number);
  // Similarly convert the rest of the file into an array of numbers:
  let input = inputText.slice(firstEOL+1).trim().split(/\s+/).map(Number);

  let model = new CP.Model(utils.makeModelName('flexible-jobshop', filename));
  const nbJobs = firstLine[0] as number;
  const nbMachines = firstLine[1] as number;
  // console.log("Flexible JobShop with " + nbMachines + " machines and " + nbJobs + " jobs.");

  // For each machine create an array of operations executed on it.
  // Initialize all machines by empty arrays:
  let machines: CP.IntervalVar[][] = [];
  for (let j = 0; j < nbMachines; j++)
    machines[j] = [];

  // End times of each job:
  let ends: CP.IntExpr[] = [];

  // Redundant cumul resource.
  // For example, with this redundant constraint instance data/Dauzere/02a.fjs,
  // has trivial lower bound 2228 (just by propagation). According to Quintiq
  // there is a solution with that objective. Without the redundant cumul it
  // takes ages to prove that there is no solution with makespan 2227.
  // It also seems that in this particular instance duration is the same in all
  // modes (what makes this redundant constraint stronger).
  let allMachines: CP.CumulExpr[] = [];
  // TODO:2 We may need more fine-grained redundant cumul(s) depending on what
  // resources are most often combined together.

  let allModes: CP.IntervalVar[][] = [];

  for (let i = 0; i < nbJobs; i++) {
    let nbOperations = input.shift() as number;
    // Previous task in the job:
    let prev: CP.IntervalVar | undefined = undefined;
    for (let j = 0; j < nbOperations; j++) {
      // Create a new operation (master of alternative constraint):
      let operation = model.intervalVar({ name: `J${i + 1}O${j + 1}` });
      let nbModes = input.shift() as number;
      let modes: CP.IntervalVar[] = [];
      for (let k = 0; k < nbModes; k++) {
        const machineId = input.shift() as number;
        const duration = input.shift() as number;
        let mode = model.intervalVar({ length: duration, optional: true, name: "J" + (i + 1) + "O" + (j + 1) + "_M" + machineId });
        // In the input file machines are counted from 1, we count from 0:
        machines[machineId - 1].push(mode);
        modes.push(mode);
      }
      model.alternative(operation, modes);
      allModes.push(modes);
      // Operation has a predecessor:
      if (prev !== undefined)
        prev.endBeforeStart(operation);
      prev = operation;
      if (redundantCumul)
        allMachines.push(operation.pulse(1));
    }
    // End time of the job is end time of the last operation:
    ends.push((prev as CP.IntervalVar).end());
  }

  // Tasks on each machine cannot overlap:
  for (let j = 0; j < nbMachines; j++)
    model.noOverlap(machines[j]);

  // TODO:1 The following constraint should be marked as redundant and shouldn't
  // be used with LNS:
  if (redundantCumul)
    model.cumulSum(allMachines).cumulLe(nbMachines);

  // Minimize the makespan:
  let makespan = model.max(ends);
  makespan.minimize();

  // There shouldn't be anything more in the input:
  assert(input.length == 0);

  return { model, allModes };
}

// Run FJSSP model and write the solution to a JSON file.
// The solution consists of 3 vectors:
//   * the first one containing the start times of each operation,
//   * the second the assigned machine for each operation,
// The order of the operations is fixed across all of these vectors.
async function runFJSSPJson(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  let { model, allModes } = defineModelAndModes(inputFilename);
  let result = await CP.solve(model, params);
  let solution = result.bestSolution;
  let startTimes = [];
  let machineAssignments = [];
  if (solution) {
    for (const modes of allModes) {
      for (const modeVar of modes) {
        if (solution.isAbsent(modeVar))
          continue;
        const start = solution.getStart(modeVar);
        const machineId = modeVar.getName()!.match(/M(\d+)/)?.[1];
        assert(machineId !== undefined);
        startTimes.push(start);
        machineAssignments.push(parseInt(machineId));
        break; // Only one mode can be assigned to the operation.
      }
    }
  }
  let output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes,
    machineAssignments
  };
  fs.writeFileSync(outputJSON, JSON.stringify(output));
}


// A function usable for CP.benchmark():
function defineModel(filename: string): CP.Model {
  return defineModelAndModes(filename).model;
}

// Default parameter settings that can be overridden on command line:
let params: CP.BenchmarkParameters = {
  usage: "Usage: node flexible-jobshop.mjs [OPTIONS] INPUT_FILE1 [INPUT_FILE2] ..\n\n" +
    "Flexible JobShop options:\n" +
    "  --redundantCumul    Add a redundant cumul constraint\n\n" +
    "Output options:\n" +
    "  --fjssp-json <filename>  Write the solution, LB and UB history to a JSON file.\n" +
    "                           Only single input file is supported."
};

let commandLineArgs = process.argv.slice(2);
let fjsspJsonFilename = utils.getStringOption("--fjssp-json", "", commandLineArgs);
redundantCumul = utils.getBoolOption("--redundantCumul", commandLineArgs);

// The model can be run in two modes:
// * Using CP.benchmark when --fjssp-json option is not specified.
// * Using CP.solve when --fjssp-json option is specified. In this case the solution is written to a JSON file.
// So, depending on --fjssp-json option, the command line may contain benchmark parameters.

if (fjsspJsonFilename === "") {
  let restArgs = CP.parseSomeBenchmarkParameters(params, commandLineArgs);
  CP.benchmark(defineModel, restArgs, params);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  if (restArgs.length !== 1) {
    console.error("Error: --fjssp-json option requires exactly one input file.");
    process.exit(1);
  }
  runFJSSPJson(restArgs[0], fjsspJsonFilename, params);
}
