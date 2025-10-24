import * as fs from 'fs';
import * as CP from '@scheduleopt/optalcp';
import * as path from 'path'

function readModel(filename: string): CP.ProblemDefinition {
  let problem = CP.json2problem(fs.readFileSync(filename, 'utf8'));
  let model = problem.model;
  if (!model.getName()) {
    // Use filename without .json extension as model name
    let name = path.basename(filename, '.json');
    if (name.endsWith('.JSON'))
      name.slice(0, -5);
    model.setName(name);
  }
  return problem;
}

let params: CP.BenchmarkParameters = {
  usage: "Usage: node solveJSONs.mjs [OPTIONS] INPUT_FILE1.json [INPUT_FILE2.json] .."
};
let restArgs = CP.parseSomeBenchmarkParameters(params);
CP.benchmark(readModel, restArgs, params);
