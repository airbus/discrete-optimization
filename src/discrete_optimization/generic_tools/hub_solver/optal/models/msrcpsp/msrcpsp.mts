// msrcpsp.mts
// CP Model for the Multi-Skill Resource-Constrained Project Scheduling Problem (MSRCPSP)

import { strict as assert } from 'assert';
import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts'; // Assuming you have a utils file
import * as fs from 'fs';

// Type definition to hold the model and key variable maps for solution retrieval
type ModelWithVariables = {
  model: CP.Model;
  taskVars: Map<string, CP.IntervalVar>;
  modeVars: Map<string, Map<string, CP.IntervalVar>>;
  employeeAssignmentVars: Map<string, Map<string, CP.IntervalVar>>;
  employeeSkillsIntervals: Map<string, Map<string, Map<string, CP.IntervalVar>>>;
};


function defineModelAndVarsJsonMine(filename: string): ModelWithVariables {
  const model = new CP.Model(utils.makeModelName('msrcpsp-json', filename));

  // 1. Read and parse the JSON problem file
  const fileContent = fs.readFileSync(filename, 'utf8');
  const data = JSON.parse(fileContent);

  const resourceNames: string[] = data.resources_set;
  const nonRenewableSet = new Set<string>(data.non_renewable_resources);
  const skillNames: string[] = data.skills_set;
  const employeeNames: string[] = data.employees_list;
  const oneSkillUsedPerWorker: Boolean = data.one_skill_used_per_worker;
  const oneWorkerPerTask: Boolean = data.one_worker_per_task;
  console.log(`One skill per worker : ${oneSkillUsedPerWorker}`);
  const taskVars = new Map<string, CP.IntervalVar>();
  const modeVars = new Map<string, Map<string, CP.IntervalVar>>();
  const employeeAssignmentVars = new Map<string, Map<string, CP.IntervalVar>>();
  const ends: CP.IntExpr[] = [];

  const renewableCumuls: Map<string, CP.CumulExpr[]> = new Map(resourceNames.map(r => [r, []]));
  const nonRenewableConsumptions: Map<string, CP.IntExpr[]> = new Map(resourceNames.map(r => [r, []]));
  const employeeCumuls: Map<string, CP.CumulExpr[]> = new Map<string, CP.CumulExpr[]>();
  const employeeIntervals: Map<string, CP.IntervalVar[]> = new Map<string, CP.IntervalVar[]>();
  const employeeSkillsIntervals: Map<string, Map<string, Map<string, CP.IntervalVar>>> = new Map<string,
                                                                                                 Map<string,
                                                                                                     Map<string, CP.IntervalVar>>>();
  for (const emp of employeeNames){
    employeeCumuls.set(emp, []);
    employeeIntervals.set(emp, []);
  }
  // 2. Create variables for tasks, modes, and employee assignments
  for (const taskId of data.tasks_list) {
    // Main interval variable for the task
    const taskVar = model.intervalVar({ name: `Task_${taskId}` });
    taskVars.set(taskId, taskVar);
    // Optional interval variables for each mode
    const taskModeIntervals: CP.IntervalVar[] = [];
    modeVars.set(taskId, new Map<string, CP.IntervalVar>());
    employeeAssignmentVars.set(taskId, new Map<string, CP.IntervalVar>());
    employeeSkillsIntervals.set(taskId, new Map<string, Map<string, CP.IntervalVar>>());

    for (const [modeId, modeData] of Object.entries(data.mode_details[taskId])) {
      const duration = (modeData as any).duration as number;
      const modeVar = model.intervalVar({ name: `Task_${taskId}_Mode_${modeId}`, length: duration, optional: true });
      modeVars.get(taskId)!.set(modeId, modeVar);
      taskModeIntervals.push(modeVar); // Optional interval for mode "modeId" of the task
      for (const skill of skillNames){
        // Required skill
        const requiredSkillLevel = (modeData as any)[skill] || 0;
        if (requiredSkillLevel > 0) {
            // Consider skilled Employees for this mode/task
            const skilledEmployees = employeeNames.filter(emp => (data.employees[emp].dict_skill[skill]?.skill_value || 0) > 0);
            const skillContributions: CP.IntExpr[]  = [];
            for (let employee of skilledEmployees){
                if (!employeeAssignmentVars.get(taskId)?.has(employee)){
                  // Check if the interval for the task is not already present in the employee task.
                  const empTaskVar = model.intervalVar({name: `Task_${taskId}_employee_${employee}`,
                                                        optional:true});
                  employeeAssignmentVars.get(taskId)?.set(employee, empTaskVar);
                  employeeSkillsIntervals.get(taskId)?.set(employee, new Map<string, CP.IntervalVar>());

                  employeeCumuls.get(employee)?.push(empTaskVar.pulse(1));
                  employeeIntervals.get(employee)?.push(empTaskVar);
                  // Manually synchronize the intervals, when the employee is used, its interval is sync with the real interval.
                  model.constraint(model.implies(model.presenceOf(empTaskVar),
                                                 empTaskVar.start().eq(taskVar.start())));
                  model.constraint(model.implies(model.presenceOf(empTaskVar),
                                                 empTaskVar.end().eq(taskVar.end())));
                  //console.log(`${employee} added to ${taskId}`);
                }
                if (!employeeSkillsIntervals.get(taskId)?.get(employee)?.has(skill)){
                  if(oneSkillUsedPerWorker){
                    const skillUserPerWorkerForTask = model.intervalVar({name: `Task_${taskId}_skill_${skill}_emp${employee}`,
                                                                         optional: true});
                    employeeSkillsIntervals.get(taskId)?.get(employee)?.set(skill, skillUserPerWorkerForTask);
                  }
                }
                if(oneSkillUsedPerWorker){
                  console.log(`${employeeSkillsIntervals.get(taskId)?.get(employee)?.get(skill)?.presence()}`);
                  skillContributions.push((employeeSkillsIntervals.get(taskId)?.get(employee)?.get(skill)?.presence() as any));
                }
                else{
                  skillContributions.push((employeeAssignmentVars.get(taskId)?.get(employee)?.presence() as any)
                                          .times(data.employees[employee].dict_skill[skill].skill_value));
                }
            }
            model.constraint(model.implies(model.presenceOf(modeVar),
                                           model.sum(skillContributions).ge(requiredSkillLevel)));
        }

      }

      // Update the cumuls of resource (renewable and not)
      resourceNames.forEach(resName => {
        const requirement = (modeData as any)[resName] || 0;
        if (requirement > 0) {
          if (nonRenewableSet.has(resName)) {
            nonRenewableConsumptions.get(resName)!.push(model.presenceOf(modeVar).times(requirement));
          } else {
            renewableCumuls.get(resName)!.push(modeVar.pulse(requirement));
          }
        }
      });
      // The real task interval is constrained to the alternatives.
      model.alternative(taskVar, taskModeIntervals);
    }


    if (oneSkillUsedPerWorker){
      //const keysList = [...employeeSkillsIntervals.get(taskId)?.keys()];
      //console.log(`Employee skills interval : ${keysList}`);
      for(const [employee, skillsDict] of employeeSkillsIntervals.get(taskId)!.entries()){
        const skillsIntervalList = [...(skillsDict as Map<string, CP.IntervalVar>)?.values()];
        console.log(`Skills interval list : ${skillsIntervalList}`);
        const skillsPresenceList = skillsIntervalList.map(interval => interval.presence());
        model.constraint(model.le(model.sum(skillsPresenceList), 1));
        model.alternative(employeeAssignmentVars.get(taskId)?.get(employee)!,
                          skillsIntervalList);
        const pres = employeeAssignmentVars.get(taskId)?.get(employee)?.presence() as CP.BoolExpr;
        model.constraint(model.implies(pres, model.sum(skillsPresenceList).ge(0)));
        for(const itv of skillsIntervalList){
          model.constraint(pres.ge(model.presenceOf(itv)));
        }
      }
    }

    if (oneWorkerPerTask){
      // We use the alternative constraint, it's a straight forward way.
      const employeeIntervalMap = [...(employeeAssignmentVars.get(taskId) as Map<string, CP.IntervalVar>)?.values()];

      if (employeeIntervalMap.length>0){
        model.alternative(taskVar, employeeIntervalMap);
      }
    }
    // Collect end variables for makespan objective
    if (taskId == data.sink_task) {
        ends.push(taskVar.end());
    }
  }

  // 3. Add constraints
  // Precedence constraints
  for (const taskId of data.tasks_list) {
    const predecessorVar = taskVars.get(taskId);
    if (data.successors[taskId]) {
      for (const successorId of data.successors[taskId]) {
        const successorVar = taskVars.get(successorId);
        if (predecessorVar && successorVar) {
          predecessorVar.endBeforeStart(successorVar);
        }
      }
    }
  }

  // Resource capacity constraints
  resourceNames.forEach(resName => {
    const capacity = data.resources_availability[resName][0]; // Assuming constant capacity for now
    if (nonRenewableSet.has(resName)) {
      model.constraint(model.sum(nonRenewableConsumptions.get(resName)!).le(capacity));
    } else {
      model.cumulSum(renewableCumuls.get(resName)!).cumulLe(capacity);
    }
  });

  // Employee disjunctive and calendar constraints
  employeeNames.forEach(empId => {
      // Each employee can only do one task at a time (capacity of 1)
      model.cumulSum(employeeCumuls.get(empId)!).cumulLe(1);
      model.noOverlap(employeeIntervals.get(empId)!);
      // TODO: Add calendar constraints if necessary by adding pulses for unavailable times
  });
  // 4. Set the objective to minimize the makespan
  model.max(ends).minimize();
  return { model, taskVars, modeVars, employeeAssignmentVars, employeeSkillsIntervals};
}



async function runMsRcpspAndExport(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  let { model, taskVars, modeVars, employeeAssignmentVars, employeeSkillsIntervals} = defineModelAndVarsJsonMine(inputFilename);

  let result = await CP.solve(model, params);
  let solution = result.bestSolution;

  let startTimes = new Map<string, number>();
  let endTimes = new Map<string, number>();
  let modes = new Map<string, string>();
  let employeeUsage = new Map<string, string[]>();
  let employeeUsageSkill = new Map<string, Map<string, string[]>>();

  if (solution) {
    // Get schedule and modes
    for (const [taskId, taskVar] of taskVars.entries()) {
      if (taskId !== "source" && taskId !== "sink") { // Adjust source/sink names if needed
          startTimes.set(taskId, solution.getStart(taskVar) as number);
          endTimes.set(taskId, solution.getEnd(taskVar) as number);
      }
    }
    for (const [taskId, modeDict] of modeVars.entries()) {
      for (const [modeId, modeVar] of modeDict.entries()) {
        if (solution.isPresent(modeVar)) {
          modes.set(taskId, modeId);
          break;
        }
      }
    }
    // Get employee assignments
    for (const [taskId, assignmentDict] of employeeAssignmentVars.entries()) {
        const assignedEmployees: string[] = [];
        employeeUsageSkill.set(taskId, new Map<string, string[]>());


        for (const [empId, assignmentVar] of assignmentDict.entries()) {
            if (solution.isPresent(assignmentVar)) {
                assignedEmployees.push(empId);
                employeeUsageSkill.get(taskId)?.set(empId, []);
                if (employeeSkillsIntervals.get(taskId)?.has(empId)){
                  for(const [skill, skillDict] of employeeSkillsIntervals.get(taskId)?.get(empId)!.entries())
                  {
                    console.log(`skill = ${skill}`);
                    console.log(`${skillDict}`);
                    if (solution.isPresent(skillDict)){
                      console.log(`Is present.. ${empId}_${skill}`);
                      employeeUsageSkill.get(taskId)?.get(empId)?.push(skill);


                    }
                  }
                }

            }
        }
        if (assignedEmployees.length > 0) {
            employeeUsage.set(taskId, assignedEmployees);
        }

    }
  }
  for(const [taskId, content] of employeeUsage.entries()){
    for(const [emp, skills] of content.entries()){
      console.log(`Employee ${emp} used on task ${taskId} with skills ${skills}`);
    }
  }
  console.log(`${JSON.stringify(employeeUsageSkill)}`);

    // 1. Create a helper function to convert the inner Map
  function convertInnerMap(innerMap: Map<string, string[]>) {
      // This converts the Map<empId, skill[]> to an object { empId: skill[] }
      return Object.fromEntries(innerMap);
  }

  // 2. Convert the entire employeeUsageSkill map structure
  const finalEmployeeUsageSkill = Object.fromEntries(
      // Map over the entries of the top-level Map (taskId)
      [...employeeUsageSkill.entries()].map(([taskId, employeeSkillsMap]) => [
          taskId,
          // Convert the nested Map<empId, skill[]> to a plain object
          convertInnerMap(employeeSkillsMap)
      ])
  );




  let output = {
    objective: solution ? solution.getObjective() : null,
    duration: result.duration,
    startTimes: Object.fromEntries(startTimes),
    endTimes: Object.fromEntries(endTimes),
    modes: Object.fromEntries(modes),
    employeeUsage: Object.fromEntries(employeeUsage),
    employeeUsageSkill: finalEmployeeUsageSkill,
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
  };

  fs.writeFileSync(outputJSON, JSON.stringify(output, null, 2));
  console.log(`Solution exported to ${outputJSON}`);
}


// --- Main execution logic ---
let params: CP.BenchmarkParameters = {
  usage: "Usage: node msrcpsp.mts [OPTIONS] INPUT_FILE"
};
let commandLineArgs = process.argv.slice(2);
let outputJsonFilename = utils.getStringOption("--output-json", "", commandLineArgs);

if (outputJsonFilename === "") {
    console.error("Error: --output-json option is required.");
    process.exit(1);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  runMsRcpspAndExport(restArgs[0], outputJsonFilename, params);
}
