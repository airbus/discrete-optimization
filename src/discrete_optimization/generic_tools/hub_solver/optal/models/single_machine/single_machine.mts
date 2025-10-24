// CP Model for the Single Machine Weighted Tardiness Problem with Release Dates

import { strict as assert } from 'assert';
import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts'; // Assuming a utils file
import * as fs from 'fs';

// Type definition to hold the model and key variables
type ModelWithVariables = {
  model: CP.Model;
  jobVars: Map<string, CP.IntervalVar>;
};

function defineModelAndVarsJson(filename: string): ModelWithVariables {
  const model = new CP.Model(utils.makeModelName('wt-json', filename));

  // 1. Read and parse the JSON problem file
  const fileContent = fs.readFileSync(filename, 'utf8');
  const data = JSON.parse(fileContent);

  const jobVars = new Map<string, CP.IntervalVar>();
  const weightedTardinessExprs: CP.IntExpr[] = [];
  const max_time = 50000;

  // 2. Create an interval variable for each job
  for (let i = 0; i < data.num_jobs; i++) {
    const p_i = data.processing_times[i] as number;
    const w_i = data.weights[i] as number;
    const d_i = data.due_dates[i] as number;
    const r_i = data.release_dates[i] as number;
    // Create the interval variable for the job.
    // The release date is handled by setting the lower bound of the start time.
    const jobVar = model.intervalVar({
      name: `Job_${i}`,
      length: p_i,
      start: [r_i, max_time] // Constraint: start_i >= release_date_i
    });
    jobVars.set(i.toString(), jobVar);

    // 3. Define tardiness for each job
    // Tardiness_i = max(0, end_i - due_date_i)
    const tardiness = model.max([0, jobVar.end().minus(d_i)]);

    // Add the weighted tardiness to our list of objective terms
    weightedTardinessExprs.push(tardiness.times(w_i));
  }

  // 4. Add the single-machine constraint (no overlap)
  // All jobs must run on the same machine and cannot overlap in time.
  model.noOverlap(Array.from(jobVars.values()));

  // 5. Set the objective to minimize the sum of weighted tardiness
  model.sum(weightedTardinessExprs).minimize();

  return { model, jobVars };
}

async function runWtAndExport(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  let { model, jobVars } = defineModelAndVarsJson(inputFilename);

  let result = await CP.solve(model, params);
  let solution = result.bestSolution;

  let schedule: { [key: string]: [number, number] } = {};

  if (solution) {
    for (const [jobId, jobVar] of jobVars.entries()) {
      const start = solution.getStart(jobVar) as number;
      const end = solution.getEnd(jobVar) as number;
      schedule[jobId] = [start, end];
    }
  }

  let output = {
    objective: solution ? solution.getObjective() : null,
    duration: result.duration,
    schedule: schedule,
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
  };

  fs.writeFileSync(outputJSON, JSON.stringify(output, null, 2));
  console.log(`Solution exported to ${outputJSON}`);
}

// --- Main execution logic ---
let params: CP.BenchmarkParameters = {
  usage: "Usage: node weighted_tardiness.mts [OPTIONS] INPUT_FILE"
};
let commandLineArgs = process.argv.slice(2);
let outputJsonFilename = utils.getStringOption("--output-json", "", commandLineArgs);

if (outputJsonFilename === "") {
    console.error("Error: --output-json option is required.");
    process.exit(1);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  if (restArgs.length !== 1) {
    console.error("Error: --output-json option requires exactly one input file.");
    process.exit(1);
  }
  runWtAndExport(restArgs[0], outputJsonFilename, params);
}
