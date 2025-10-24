// Adapted from original model https://github.com/ScheduleOpt/optalcp-benchmarks/blob/main/benchmarks/rcpsp/rcpsp.mts
// To get output solution
import { strict as assert } from 'assert';
import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts'
import * as fs from 'fs';



type ModelWithVariables = {
  model: CP.Model;
  jobsVarMap: Map<string, CP.IntervalVar>;
}


type CumulDemand = {
  demand: number,
  interval: CP.IntervalVar
};

const useCap2Relaxation = false;



function defineModelAndVarsJson(filename: string): ModelWithVariables {
  const model = new CP.Model(utils.makeModelName('rcpsp-json', filename));

  // 1. Read and parse the JSON file
  const fileContent = fs.readFileSync(filename, 'utf8');
  const data = JSON.parse(fileContent);

  const nbRealJobs = data.nbJobs - 2;
  const resourceNames = Object.keys(data.resources);

  // 2. Create interval variables for each real job
  const jobs: CP.IntervalVar[] = [];
  const jobsVarMap = new Map<string, CP.IntervalVar>();
  for (const taskId of data.tasksList) {
    if (taskId !== data.sourceTask && taskId !== data.sinkTask) {
      const itv = model.intervalVar({ name: `T${taskId}` });
      jobs.push(itv);
      jobsVarMap.set(taskId, itv);
    }
  }

  // 3. Prepare data structures for constraints
  const ends: CP.IntExpr[] = [];
  const cumuls: CP.CumulExpr[][] = Array(resourceNames.length).fill(0).map(() => []);

  // 4. Loop through jobs to set durations, resource demands, and precedences
  for (const taskId of data.tasksList) {
    if (taskId === data.sourceTask || taskId === data.sinkTask) continue;

    const predecessor = jobsVarMap.get(taskId);
    if (!predecessor) continue;

    // For now, we assume single-mode problems (mode "1")
    const modeDetails = data.modeDetails[taskId]['1'];

    // Set duration
    predecessor.setLength(modeDetails.duration);

    // Set resource requirements (pulses)
    resourceNames.forEach((resName, rIndex) => {
      const requirement = modeDetails[resName] || 0;
      if (requirement > 0) {
        cumuls[rIndex].push(predecessor.pulse(requirement));
      }
    });

    // Set precedence relations
    let isLast = true;
    const successors = data.successors[taskId] || [];
    for (const successorId of successors) {
      if (successorId !== data.sinkTask) {
        const successor = jobsVarMap.get(successorId);
        if (successor) {
          predecessor.endBeforeStart(successor);
          isLast = false;
        }
      }
    }
    // Identify tasks at the end of the project to define the makespan
    if (isLast) {
      ends.push(predecessor.end());
    }
  }

  // 5. Add cumulative resource constraints
  resourceNames.forEach((resName, rIndex) => {
    const capacity = data.resources[resName];
    model.cumulSum(cumuls[rIndex]).cumulLe(capacity);
  });

  // 6. Set the objective to minimize the makespan
  model.max(ends).minimize();

  return { model, jobsVarMap };
}

function defineModel(filename: string): CP.Model {
  return defineModelAndVars(filename).model
}

function defineModelAndVars(filename: string): ModelWithVariables{
  return defineModelAndVarsJson(filename);
}

async function runRcpspAndExport(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  // We no longer need orderedTaskIds
  let { model, jobsVarMap } = defineModelAndVars(inputFilename);
  let result = await CP.solve(model, params);
  let solution = result.bestSolution;

  // Initialize as objects (dictionaries) instead of arrays
  let startTimes = {};
  let endTimes = {};

  if (solution) {
    // Iterate directly on the map. This gives us both the task ID and the variable.
    for (const [taskId, jobVar] of jobsVarMap.entries()) {
      startTimes[taskId] = solution.getStart(jobVar);
      endTimes[taskId] = solution.getEnd(jobVar);
    }
  }

  let output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes,
    endTimes,
  };

  //console.log("Exporting solution...");
  fs.writeFileSync(outputJSON, JSON.stringify(output));
}

// Default parameter settings that can be overridden on the command line:
let params: CP.BenchmarkParameters = {
  usage: "Usage: node rcpsp.mts [OPTIONS] INPUT_FILE [INPUT_FILE2] .."
};
let commandLineArgs = process.argv.slice(2);
let rcpspJsonFilename = utils.getStringOption("--output-json", "", commandLineArgs);

if (rcpspJsonFilename === "") {
  let restArgs = CP.parseSomeBenchmarkParameters(params, commandLineArgs);
  CP.benchmark(defineModel, restArgs, params);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  if (restArgs.length !== 1) {
    console.error("Error: --output-json option requires exactly one input file.");
    process.exit(1);
  }
  console
  runRcpspAndExport(restArgs[0], rcpspJsonFilename, params);
}
