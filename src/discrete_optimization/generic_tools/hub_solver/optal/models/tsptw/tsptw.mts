// tsp.mts
// An improved CP Model for the Traveling Salesperson Problem (TSP)
// using a sequence variable formulation.

import { strict as assert } from 'assert';
import * as CP from '@scheduleopt/optalcp';
import * as utils from '../../utils/utils.mts'; // Assuming a utils file
import * as fs from 'fs';

// Type definition to hold the model and key variables
type ModelWithVariables = {
  model: CP.Model;
  visitIntervals: CP.IntervalVar[];
  nodesInSequence: number[];
};

function defineModelAndVarsJson(filename:string): ModelWithVariables {
  const model = new CP.Model(utils.makeModelName('tsp-sequence', filename));

  // 1. Read and parse the JSON problem file
  const fileContent = fs.readFileSync(filename, 'utf8');
  const data = JSON.parse(fileContent);
  const depot = data.depot as number;
  const nodeCount = data.node_count;
  const timeWindows = data.time_windows;
  const distanceMatrix = data.distance_matrix;
  // 2. Create an interval variable for each node visit
  const visitIntervals = Array.from({ length: nodeCount }, (_, i) =>
    model.intervalVar({ length: 0, name: `Visit_${i}`,
    start:[timeWindows[i][0], timeWindows[i][1]]}) // Standard TSP has zero visit duration
  );

  // Create a dummy interval to mark the end of the entire tour
  const tourEnd = model.intervalVar({ length: 0, name: 'TourEnd',
    start:[timeWindows[depot][0], timeWindows[depot][1]]});

  // 3. Set up the sequence
  // The sequence variable will order all nodes that are not fixed at the start or end.
  const nodesInSequence = Array.from({ length: nodeCount }, (_, i) => i)
    .filter(i => i !== depot);

  const sequenceIntervals = nodesInSequence.map(i => visitIntervals[i]);
  const sequence = model.sequenceVar(sequenceIntervals, nodesInSequence);

  // 4. Add constraints
  // The tour starts at time 0 at the start_index node.
  visitIntervals[depot].setStart(0);

  // The noOverlap constraint on the sequence ensures two things:
  //  - The visits in the sequence don't overlap.
  //  - The travel time between two consecutive visits is respected, based on the distance matrix.
  model.noOverlap(sequence, distanceMatrix);

  // Link the fixed start/end nodes to the sequence.
  // The start node must transition to the first node of the sequence.
  for (let i = 0; i < nodesInSequence.length; i++) {
    model.endBeforeStart(visitIntervals[depot], visitIntervals[nodesInSequence[i]],
                         distanceMatrix[depot][nodesInSequence[i]]);
    model.endBeforeStart(visitIntervals[nodesInSequence[i]], tourEnd, distanceMatrix[nodesInSequence[i]][depot]);
  }


  // 5. Set the objective
  // The total tour duration is the start time of the final dummy node.
  model.minimize(tourEnd.start());
  return { model, visitIntervals, nodesInSequence };
}

async function runTspAndExport(inputFilename: string, outputJSON: string, params: CP.BenchmarkParameters) {
  const { model, visitIntervals, nodesInSequence } = defineModelAndVarsJson(inputFilename);
  const data = JSON.parse(fs.readFileSync(inputFilename, 'utf8'));

  const result = await CP.solve(model, params);
  const solution = result.bestSolution;

  let finalPermutation: number[] = [];

  if (solution) {
    // To reconstruct the tour, get the start times of all sequenced visits and sort them.
    const sequencedNodesWithTimes = nodesInSequence.map(nodeIndex => ({
      index: nodeIndex,
      startTime: solution.getStart(visitIntervals[nodeIndex]) as number
    }));

    sequencedNodesWithTimes.sort((a, b) => a.startTime - b.startTime);

    const permutation = sequencedNodesWithTimes.map(item => item.index);
    finalPermutation = permutation;
    console.log(`${permutation}`);
  }

  const output = {
    objective: solution ? solution.getObjective() : null,
    duration: result.duration,
    // We export the permutation without the start_index, as expected by the TspSolution class
    permutation: finalPermutation,
    objectiveHistory: result.objectiveHistory,
    lowerBoundHistory: result.lowerBoundHistory,
  };

  fs.writeFileSync(outputJSON, JSON.stringify(output, null, 2));
  console.log(`Solution exported to ${outputJSON}`);
}


// --- Main execution logic ---
let params: CP.BenchmarkParameters = {
  usage: "Usage: node tsp.mts [OPTIONS] INPUT_FILE"
};
let commandLineArgs = process.argv.slice(2);
let outputJsonFilename = utils.getStringOption("--output-json", "", commandLineArgs);

if (outputJsonFilename === "") {
    console.error("Error: --output-json option is required.");
    process.exit(1);
} else {
  let restArgs = CP.parseSomeParameters(params, commandLineArgs);
  console.log(`${restArgs}`);
  //if (restArgs.length !== 1) {
  //  console.error("Error: --output-json option requires exactly one input file.");
  //  process.exit(1);
  //}
  runTspAndExport(restArgs[0], outputJsonFilename, params);
}
