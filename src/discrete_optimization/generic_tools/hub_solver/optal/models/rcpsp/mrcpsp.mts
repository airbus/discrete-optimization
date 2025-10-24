// Adapted from original model https://github.com/ScheduleOpt/optalcp-benchmarks/blob/main/benchmarks/rcpsp/rcpsp.mts
// To get output solution
import { strict as assert } from 'assert';
import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts'
import * as fs from 'fs';



type ModelWithVariables = {
  model: CP.Model;
  jobsVarMap: Map<string, CP.IntervalVar>;
  modesVarMap: Map<string, Map<string, CP.IntervalVar>>;
}


const useCap2Relaxation = false;



function defineModelAndVarsJson(filename: string): ModelWithVariables {
  const model = new CP.Model(utils.makeModelName('mrcpsp-json', filename));
  // 1. Read and parse the JSON file
  const fileContent = fs.readFileSync(filename, 'utf8');
  const data = JSON.parse(fileContent);
  const nbRealJobs = data.nbJobs - 2;
  const resourceNames = Object.keys(data.resources);
  const nonRenewableSet = new Set(data.nonRenewableResources || []);
  const jobsVarMap = new Map<string, CP.IntervalVar>();
  const modesVarMap = new Map<string, Map<string, CP.IntervalVar>>();
  const ends: CP.IntExpr[] = [];
  const cumuls: CP.CumulExpr[][] = Array(resourceNames.length).fill(0).map(() => []);
  const totalConsumptions: CP.IntExpr[][] = Array(resourceNames.length).fill(0).map(() => []); // For non-renewable
  for (const taskId of data.tasksList) {
    if (taskId !== data.sourceTask && taskId !== data.sinkTask) {
      const itv = model.intervalVar({ name: `T${taskId}`});
      // Store the main interval
      jobsVarMap.set(taskId, itv);
      // Store the modes interval
      modesVarMap.set(taskId, new Map<string, CP.IntervalVar>());
      const modeIntervals: CP.IntervalVar[] = [];
      for (const [mode,  modeData] of Object.entries(data.modeDetails[taskId])){
        const duration = (modeData as any).duration as number;
        const itv_mode = model.intervalVar({name: `T${taskId}${mode}`, length: duration, optional: true});

        // --- Differentiate resource constraint based on type ---
        resourceNames.forEach((resName, rIndex) => {
            const requirement = (modeData as any)[resName] || 0;
            if (requirement > 0) {
            if (nonRenewableSet.has(resName)) {
                // we cumul on non renewable resource
                totalConsumptions[rIndex].push(model.presenceOf(itv_mode).times(requirement));
            } else {
                // Renewable: track concurrent usage (pulse).
                cumuls[rIndex].push(itv_mode.pulse(requirement));
            }
            }
        });

        modeIntervals.push(itv_mode);
        modesVarMap.get(taskId)?.set(mode, itv_mode);
      }
      model.alternative(itv, modeIntervals); // 1 mode per task
    }
  }

  // Precedence constraint

  for (const taskId of data.tasksList){
    let isLast = true;
    if (taskId !== data.sourceTask && taskId !== data.sinkTask) {
        const predecessorJob = jobsVarMap.get(taskId);
        const successors = data.successors[taskId] || [];
        for (const successorId of successors) {
        if (successorId !== data.sinkTask) {
            const successorJob = jobsVarMap.get(successorId);
            if (successorJob && predecessorJob) {
                predecessorJob.endBeforeStart(successorJob);
                isLast = false;
            }
        }
        }
        if (isLast && predecessorJob){
             ends.push(predecessorJob.end());
        }
    }
  }

  resourceNames.forEach((resName, rIndex) => {
    const capacity = data.resources[resName];
    if (nonRenewableSet.has(resName)) {
      model.constraint(model.sum(totalConsumptions[rIndex]).le(capacity));
    } else {
      model.cumulSum(cumuls[rIndex]).cumulLe(capacity);
    }
  });
  // 6. Set the objective to minimize the makespan
  model.max(ends).minimize();
  return { model, jobsVarMap, modesVarMap };
}

function defineModel(filename: string): CP.Model {
  return defineModelAndVars(filename).model
}

function defineModelAndVars(filename: string): ModelWithVariables{
  return defineModelAndVarsJson(filename);
}

async function runRcpspAndExport(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  // We no longer need orderedTaskIds
  let { model, jobsVarMap, modesVarMap} = defineModelAndVars(inputFilename);
  let result = await CP.solve(model, params);
  let solution = result.bestSolution;

  // Initialize as objects (dictionaries) instead of arrays
  let startTimes = new Map<string, number>();
  let endTimes = new Map<string, number>();
  let modes = new Map<string, string>();

  if (solution) {
    // Iterate directly on the map. This gives us both the task ID and the variable.
    for (const [taskId, jobVar] of jobsVarMap.entries()) {
      startTimes.set(taskId, solution.getStart(jobVar) as number);
      endTimes.set(taskId, solution.getEnd(jobVar) as number);
    }
    for (const [taskId, modeDict] of modesVarMap.entries()){
        for (const [mode, modeItv] of modeDict.entries()){
            if (solution.isPresent(modeItv)){
                modes.set(taskId, mode);
                break;
            }
        }
    }
  }

  let output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes: Object.fromEntries(startTimes),
    endTimes: Object.fromEntries(endTimes),
    modes: Object.fromEntries(modes),
  };

  //console.log("Exporting solution...");
  //console.log(`${startTimes}, ${modes}`);

  fs.writeFileSync(outputJSON, JSON.stringify(output));
}

// Default parameter settings that can be overridden on the command line:
let params: CP.BenchmarkParameters = {
  usage: "Usage: node rcpsp.mts [OPTIONS] INPUT_FILE [INPUT_FILE2] .."
};
let commandLineArgs = process.argv.slice(2);
//. console.log(commandLineArgs);
let rcpspJsonFilename = utils.getStringOption("--output-json", "", commandLineArgs);
// console.log(rcpspJsonFilename);

if (rcpspJsonFilename === "") {
  let restArgs = CP.parseSomeBenchmarkParameters(params, commandLineArgs);
  CP.benchmark(defineModel, restArgs, params);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  if (restArgs.length !== 1) {
    console.error("Error: --output-json option requires exactly one input file.");
    process.exit(1);
  }
  runRcpspAndExport(restArgs[0], rcpspJsonFilename, params);
}
