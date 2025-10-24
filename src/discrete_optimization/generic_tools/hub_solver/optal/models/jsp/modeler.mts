import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts';

/**
 * Creates jobshop model from data in the given file.
 * Jobshop and flowshop may share the same input file format.
 * Therefore the function is exported used for both problems.
 */
export function defineModelAndTask(filename: string){
  let input = utils.readFileAsNumberArray(filename);
  let model = new CP.Model(utils.makeModelName("jobshop", filename));
  const nbJobs = input.shift() as number;
  const nbMachines = input.shift() as number;

  // For each machine create an array of operations executed on it.
  // Initialize all machines by empty arrays:
  let machines: CP.IntervalVar[][] = [];
  let intervals_per_job: CP.IntervalVar[][] = [];
  for (let i = 0; i < nbJobs; i++){
    intervals_per_job[i] = [];
  }


  for (let j = 0; j < nbMachines; j++)
    machines[j] = [];

  // End times of each job:
  let ends: CP.IntExpr[] = [];

  for (let i = 0; i < nbJobs; i++) {
    // Previous task in the job:
    let prev: CP.IntervalVar | undefined = undefined;
    for (let j = 0; j < nbMachines; j++) {
      // Create a new operation:
      const machineId = input.shift() as number;
      const duration = input.shift() as number;
      let operation = model.intervalVar({
        length: duration,
        name: "J" + (i + 1) + "O" + (j + 1) + "M" + (machineId+1)
      });
      // Operation requires some machine:
      machines[machineId].push(operation);
      intervals_per_job[i].push(operation);
      // Operation has a predecessor:
      if (prev !== undefined)
        prev.endBeforeStart(operation);
      prev = operation;
    }
    // End time of the job is end time of the last operation:
    ends.push((prev as CP.IntervalVar).end());
  }

  // Tasks on each machine cannot overlap:
  for (let j = 0; j < nbMachines; j++)
    model.noOverlap(machines[j]);

  // Minimize the makespan:
  let makespan = model.max(ends);
  makespan.minimize();

  return {model, intervals_per_job};
}


export function defineModel(filename: string): CP.Model {
  return defineModelAndTask(filename).model;
}
