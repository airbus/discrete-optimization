import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts';
import * as fs from 'fs';

// Jobshop file format is the same as flowshop. The function defineModel is in a
// separate file so it could be shared:
import * as jobshopModeler from './modeler.mts';

let params = {
  usage: "Usage: node jobshop.mjs [OPTIONS] INPUT_FILE1 [INPUT_FILE2] .."
};
async function runJSPJson(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  let {model, intervals_per_job} = jobshopModeler.defineModelAndTask(inputFilename);
  let result = await CP.solve(model, params);
  let solution = result.bestSolution;
  let startTimes: number[][] = [];
  let endTimes: number[][] = []
  if (solution) {
    for (let i_job=0; i_job < intervals_per_job.length; i_job++) {
      startTimes[i_job] = []
      endTimes[i_job] = []
      for(let j_subjob=0; j_subjob < intervals_per_job[i_job].length;
          j_subjob++){
            const start = solution.getStart(intervals_per_job[i_job][j_subjob]);
            const end = solution.getEnd(intervals_per_job[i_job][j_subjob]);
            startTimes[i_job].push(start);
            endTimes[i_job].push(end);
          }
      }
    }

  let output = {
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
    duration: result.duration,
    startTimes,
    endTimes
  };
  fs.writeFileSync(outputJSON, JSON.stringify(output));
}

let commandLineArgs = process.argv.slice(2);
let outputJsonFilename = utils.getStringOption("--outputjsp", "", commandLineArgs);

if (outputJsonFilename === "") {
  let restArgs = CP.parseSomeBenchmarkParameters(params, commandLineArgs);
  CP.benchmark(jobshopModeler.defineModel, restArgs, params);
} else {
  console.log("Running with json output")
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  runJSPJson(restArgs[0], outputJsonFilename, params);
}
