import { strict as assert } from 'assert';
import * as utils from './utils.mjs';

// An auxiliary function for the GEO distance function.
function toRadians(x: number) {
  let degrees = Math.trunc(x);
  let minutes = x - degrees;
  // Don't use Math.PI here to make sure we get the same result as other implementations.
  const pi = 3.141592;
  return pi * (degrees + 5.0 * minutes / 3.0) / 180.0;
}

type ParseResult = {
  type: string,
  nbNodes: number,
  transitionMatrix: number[][],
  hasDirectionSymmetry: boolean,
  // For CVRP:
  demands?: number[],
  capacity?: number,
  depots?: number[],
}

export type ParseParameters = {
  checkDirectionSymmetry?: boolean,
  checkTriangularInequality?: boolean,
  visitDuration?: number,
  forceCeil?: boolean,
}

export function parse(filename: string, params: ParseParameters): ParseResult {
  let lines = utils.readFile(filename).trim().split('\n');
  /*
  Input file looks like this:
      NAME: <name>
      TYPE: TSP
      COMMENT: <comment>
      DIMENSION: <number of nodes>
      EDGE_WEIGHT_TYPE: GEO
      EDGE_WEIGHT_FORMAT: FUNCTION
      DISPLAY_DATA_TYPE: COORD_DISPLAY
      NODE_COORD_SECTION
      1  x1  y1
      2  x2  y2
      ...
      EOF
  */
  let pos = 0;
  let nbNodes = -1;
  let edgeWeightType = "";
  let type = "UNKNOWN";
  let capacity = undefined;
  let edgeWeightFormat = "FUNCTION"
  let transitionMatrix: number[][] = [];
  let demands = undefined;
  let depots = undefined;

  let checkDirectionSymmetry = params.checkDirectionSymmetry ?? false;
  let checkTriangularInequality = params.checkTriangularInequality ?? false;
  let visitDuration = params.visitDuration ?? 0;
  let forceCeil = params.forceCeil ?? false;

  while (pos < lines.length) {

    if (lines[pos].match(/^NAME *: /)) {
      pos++;
      continue;
    }
    if (lines[pos].match(/^TYPE *:/)) {
      type = lines[pos].split(':')[1].trim();
      pos++;
      continue;
    }
    if (lines[pos].match(/^COMMENT *: /)) {
      pos++;
      continue;
    }
    if (lines[pos].match(/^DIMENSION *: /)) {
      nbNodes = parseInt(lines[pos].split(':')[1]);
      pos++;
      continue;
    }
    if (lines[pos].match(/^DISPLAY_DATA_TYPE *:/)) {
      pos++;
      continue;
    }
    if (lines[pos].match(/^DISTANCE *:/)) {
      pos++;
      continue;
    }
    if (lines[pos].match(/^EDGE_WEIGHT_TYPE *:/)) {
      edgeWeightType = lines[pos].split(':')[1].trim();
      if (["GEO", "EUC_2D", "CEIL_2D", "ATT", "EXPLICIT"].includes(edgeWeightType) == false) {
        console.error(`Unsupported edge weight type (not implemented): "${edgeWeightType}"`);
        process.exit();
      }
      if (forceCeil) {
        if (edgeWeightType == "EUC_2D")
          edgeWeightType = "CEIL_2D";
      }
      pos++;
      continue;
    }
    if (lines[pos].match(/^EDGE_WEIGHT_FORMAT *:/)) {
      edgeWeightFormat = lines[pos].split(':')[1].trim();
      pos++;
      continue;
    }
    if (lines[pos].match(/^DISPLAY_DATA_TYPE *: COORD_DISPLAY *$/)) {
      pos++;
      continue;
    }
    if (lines[pos].match(/^CAPACITY *:/)) {
      capacity = Number(lines[pos].split(':')[1].trim());
      pos++;
      continue;
    }

    if (lines[pos].trim() == "NODE_COORD_SECTION") {
      pos++;
      assert(["GEO", "EUC_2D", "CEIL_2D", "ATT"].includes(edgeWeightType), "Unsupported edge weight type");
      assert(edgeWeightFormat == "FUNCTION", "Unsupported combination of edge weight type and format");
      let nodes: { x: number, y: number }[] = [];
      for (let i = 0; i < nbNodes; i++) {
        let nodeData = lines[pos++].trim().split(/\s+/).map(Number);
        assert(nodeData.length == 3, "Invalid input file format (node data)");
        assert(nodeData[0] == i + 1, "Invalid input file format (node number)");
        nodes.push({ x: nodeData[1], y: nodeData[2] });
      }

      // Read the node coordinates and compute the transition matrix
      for (let i = 0; i < nbNodes; i++) {
        let row = [];
        if (edgeWeightType == "EUC_2D") {
          for (let j = 0; j < nbNodes; j++) {
            let distX = nodes[i].x - nodes[j].x;
            let distY = nodes[i].y - nodes[j].y;
            row[j] = Math.round(Math.sqrt(distX * distX + distY * distY));
          }
        }
        else if (edgeWeightType == "CEIL_2D") {
          for (let j = 0; j < nbNodes; j++)
            row[j] = Math.ceil(Math.sqrt(Math.pow(nodes[i].x - nodes[j].x, 2) + Math.pow(nodes[i].y - nodes[j].y, 2)));
        }
        else if (edgeWeightType == "ATT") {
          for (let j = 0; j < nbNodes; j++) {
            let distanceX = nodes[i].x - nodes[j].x;
            let distanceY = nodes[i].y - nodes[j].y;
            let dist = Math.sqrt((distanceX * distanceX + distanceY * distanceY) / 10.0);
            row[j] = Math.ceil(dist);
          }
        }
        else {
          assert(edgeWeightType == "GEO");
          // Compute geographical distance of points i and j. I.e. the distance on
          // idealized sphere with diameter the earth.
          for (let j = 0; j < nbNodes; j++) {
            /*
            TSP format doc gives the following algorithm (nint means round to nearest integer):
            PI = 3.141592;
            deg = nint( x[i] );
            min = x[i] - deg;
            latitude[i] = PI * (deg + 5.0 * min / 3.0 ) / 180.0;
            deg = nint( y[i] );
            min = y[i] - deg;
            longitude[i] = PI * (deg + 5.0 * min / 3.0 ) / 180.0;
            RRR = 6378.388;
            q1 = cos( longitude[i] - longitude[j] );
            q2 = cos( latitude[i] - latitude[j] );
            q3 = cos( latitude[i] + latitude[j] );
            dij = (int) ( RRR * acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0);
            */
            let latitudeI = toRadians(nodes[i].x);
            let longitudeI = toRadians(nodes[i].y);
            let latitudeJ = toRadians(nodes[j].x);
            let longitudeJ = toRadians(nodes[j].y);
            let q1 = Math.cos(longitudeI - longitudeJ);
            let q2 = Math.cos(latitudeI - latitudeJ);
            let q3 = Math.cos(latitudeI + latitudeJ);
            let dist = 6378.388 * Math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0;
            row[j] = forceCeil ? Math.ceil(dist) : Math.floor(dist);
          }
        }
        transitionMatrix.push(row);
      }
      continue;
    }

    if (lines[pos].match(/^EDGE_WEIGHT_SECTION *$/)) {
      pos++;
      if (edgeWeightType == "EXPLICIT") {
        if (edgeWeightFormat == "FULL_MATRIX") {
          for (let i = 0; i < nbNodes; i++) {
            let nodeData = lines[pos++].trim().split(/\s+/).map(Number);
            assert(nodeData.length == nbNodes, "Invalid input file matrix dimension format (edge data)");
            transitionMatrix.push(nodeData);
          }
        }
        else if (edgeWeightFormat == "UPPER_ROW") {
          let rows = [];
          for (let i = 0; i < nbNodes - 1; i++) {
            let nodeData = lines[pos++].trim().split(/\s+/).map(Number);
            assert(nodeData.length == nbNodes - i - 1, `Invalid UPPER_ROW matrix. Expected ${nbNodes - i - 1} values, got ${nodeData.length}`);
            rows.push(nodeData);
          }
          rows.push([]);
          for (let i = 0; i < nbNodes; i++) {
            let row = [];
            for (let j = 0; j < i; j++)
              row.push(transitionMatrix[j][i]);
            row.push(0, ...rows[i]);
            transitionMatrix.push(row);
          }
        }
        else {
          console.error(`Unsupported edge weight format "${edgeWeightFormat}".`);
          process.exit();
        }
      }
    }

    if (lines[pos].trim() == "DEMAND_SECTION") {
      pos++;
      demands = [];
      for (let i = 0; i < nbNodes; i++) {
        let nodeData = lines[pos++].trim().split(/\s+/).map(Number);
        assert(nodeData.length == 2, "Invalid input file format (node data)");
        assert(nodeData[0] == i + 1, "Invalid input file format (node number)");
        demands[i] = nodeData[1];
      }
      continue;
    }

    if (lines[pos].trim() == "DEPOT_SECTION") {
      pos++;
      depots = [];
      while (pos < lines.length) {
        let depot = Number(lines[pos++]);
        if (depot == -1)
          break;
        depots.push(depot - 1);
      }
      continue;
    }

    if (lines[pos].trim() == "DISPLAY_DATA_SECTION") {
      pos++;
      for (let i = 0; i < nbNodes; i++) {
        let nodeData = lines[pos++].trim().split(/\s+/);
        assert(Number(nodeData[0]) == i + 1, "Invalid input file format (node number)");
      }
      continue;
    }

    if (lines[pos].match(/^ *EOF *$/))
      break;

    console.error(`Unrecognized line: "${lines[pos].trim()}"`);
    process.exit();
  }

  if (demands !== undefined && depots !== undefined) {
    for (const d of depots)
      assert(demands[d] == 0, "Depot with non-zero demand");
  }

  let hasDirectionSymmetry = true;
  outer:
  for (let i = 0; i < nbNodes; i++) {
    for (let j = 0; j < nbNodes; j++) {
      if (transitionMatrix[i][j] != transitionMatrix[j][i]) {
        if (checkDirectionSymmetry) {
          console.warn(`${filename}: Direction symmetry violated: ${i} -> ${j}: ${transitionMatrix[i][j]}, ${j} -> ${i}: ${transitionMatrix[j][i]} (EDGE_WEIGHT_TYPE: ${edgeWeightType})`);
          hasDirectionSymmetry = false;
          break outer;
        }
      }
    }
  }

  if (checkTriangularInequality) {
    outermost:
    for (let i = 0; i < nbNodes; i++) {
      for (let k = 0; k < nbNodes; k++) {
        let ttIK = transitionMatrix[i][k];
        for (let j = 0; j < nbNodes; j++)
          if (transitionMatrix[i][j] > ttIK + visitDuration + transitionMatrix[k][j]) {
            console.warn(`${filename}: Triangular inequality violated: ${i} -> ${k} -> ${j}: ${transitionMatrix[i][j]} > ${transitionMatrix[i][k]} + ${visitDuration} + ${transitionMatrix[k][j]} (EDGE_WEIGHT_TYPE: ${edgeWeightType})`);
            break outermost;
          }
      }
    }
  }

  return { type, nbNodes, transitionMatrix, demands, capacity, depots, hasDirectionSymmetry };
}
