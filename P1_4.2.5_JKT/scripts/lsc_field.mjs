import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

// Least-squares collocation (ordinary kriging with heteroscedastic formal
// noise) for the optional Proto1 horizontal signal field. This module is
// intentionally data-only: it writes the same texture payload contract that
// the existing GPU particle layer already consumes.

const LSC_SCHEMA = 'deckgl_proto1_horizontal_lsc_signal_field_v1';
const LSC_ALGORITHM_VERSION = 'ok_local_fixed_parent_k32_moore8_v1';
const RAW_FIELD_JSON = 'horizontal_particle_field.json';
const RAW_FIELD_F32 = 'horizontal_particle_field_rgba_f32.bin';
const RAW_COVARIANCE_F32 = 'horizontal_particle_covariance_rgba_f32.bin';
const RAW_SPAWNS_F32 = 'horizontal_particle_spawns_rg_f32.bin';
const OUTPUT_JSON = 'horizontal_particle_field_lsc.json';
const OUTPUT_FIELD_F32 = 'horizontal_particle_lsc_field_rgba_f32.bin';
const OUTPUT_COVARIANCE_F32 = 'horizontal_particle_lsc_covariance_rgba_f32.bin';
const OUTPUT_SPAWNS_F32 = 'horizontal_particle_lsc_spawns_rg_f32.bin';
const EPS = 1e-12;

export class LscBakeAbort extends Error {
  constructor(message) {
    super(message);
    this.name = 'LscBakeAbort';
    this.code = 'LSC_BAKE_ABORTED';
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function key(i, j) {
  return `${i}:${j}`;
}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((name) => `${JSON.stringify(name)}:${stableJson(value[name])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function readFloat32(filePath) {
  const buffer = await fs.readFile(filePath);
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`Float32 asset ${path.basename(filePath)} has a non-float32 byte length.`);
  }
  return new Float32Array(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
}

async function writeJson(filePath, value) {
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeFloat32(filePath, values) {
  await fs.writeFile(filePath, Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values, p) {
  const sorted = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const position = clamp(Number(p), 0, 100) * 0.01 * (sorted.length - 1);
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sorted[low];
  const t = position - low;
  return sorted[low] * (1 - t) + sorted[high] * t;
}

function vectorDistance(ax, ay, bx, by) {
  return Math.hypot(ax - bx, ay - by);
}

function gridToLocal(grid, i, j) {
  return [
    grid.gridOriginLocalM[0] + grid.gridAxisIM[0] * i + grid.gridAxisJM[0] * j,
    grid.gridOriginLocalM[1] + grid.gridAxisIM[1] * i + grid.gridAxisJM[1] * j,
  ];
}

function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = Array.from({length: n}, (_, row) => {
    const values = new Float64Array(n + 1);
    for (let column = 0; column < n; column += 1) values[column] = matrix[row][column];
    values[n] = rhs[row];
    return values;
  });

  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    let pivotAbs = Math.abs(a[pivot][column]);
    for (let row = column + 1; row < n; row += 1) {
      const candidate = Math.abs(a[row][column]);
      if (candidate > pivotAbs) {
        pivot = row;
        pivotAbs = candidate;
      }
    }
    if (!(pivotAbs > EPS)) throw new Error('Singular linear system.');
    if (pivot !== column) [a[pivot], a[column]] = [a[column], a[pivot]];
    const pivotValue = a[column][column];
    for (let j = column; j <= n; j += 1) a[column][j] /= pivotValue;
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = a[row][column];
      if (Math.abs(factor) <= EPS) continue;
      for (let j = column; j <= n; j += 1) a[row][j] -= factor * a[column][j];
    }
  }

  return Float64Array.from(a.map((row) => row[n]));
}

function fitSecondOrderTrend(cells, component) {
  const xs = cells.map((cell) => cell.x);
  const ys = cells.map((cell) => cell.y);
  const xMean = mean(xs);
  const yMean = mean(ys);
  const xScale = Math.max(1, Math.sqrt(mean(xs.map((x) => (x - xMean) ** 2))));
  const yScale = Math.max(1, Math.sqrt(mean(ys.map((y) => (y - yMean) ** 2))));
  const normal = Array.from({length: 6}, () => new Float64Array(6));
  const rhs = new Float64Array(6);

  for (const cell of cells) {
    const x = (cell.x - xMean) / xScale;
    const y = (cell.y - yMean) / yScale;
    const design = [1, x, y, x * y, x * x, y * y];
    const value = component === 'east' ? cell.east : cell.north;
    for (let row = 0; row < 6; row += 1) {
      rhs[row] += design[row] * value;
      for (let column = 0; column < 6; column += 1) normal[row][column] += design[row] * design[column];
    }
  }

  const coefficients = solveLinearSystem(normal, rhs);
  const residuals = new Float64Array(cells.length);
  for (let index = 0; index < cells.length; index += 1) {
    const cell = cells[index];
    const x = (cell.x - xMean) / xScale;
    const y = (cell.y - yMean) / yScale;
    const predicted = coefficients[0] + coefficients[1] * x + coefficients[2] * y +
      coefficients[3] * x * y + coefficients[4] * x * x + coefficients[5] * y * y;
    const value = component === 'east' ? cell.east : cell.north;
    residuals[index] = value - predicted;
  }

  return {
    residuals,
    metadata: {
      order: 2,
      basis: '[1, x, y, xy, x², y²]',
      normalizedCoordinates: true,
      coordinateCenterM: [xMean, yMean],
      coordinateScaleM: [xScale, yScale],
      coefficients: Array.from(coefficients),
      meaning: 'Used only to estimate the semivariogram; not added back during prediction.',
    },
  };
}

function buildEmpiricalVariogram(cells, residuals, grid, {maxLagM, binWidthM, minPairsPerBin}) {
  const binCount = Math.ceil(maxLagM / binWidthM);
  const sums = new Float64Array(binCount);
  const counts = new Uint32Array(binCount);
  const byGrid = new Map(cells.map((cell, index) => [key(cell.i, cell.j), {cell, index}]));
  const axisLengths = [
    Math.hypot(grid.gridAxisIM[0], grid.gridAxisIM[1]),
    Math.hypot(grid.gridAxisJM[0], grid.gridAxisJM[1]),
  ].filter((value) => value > 0);
  const maxOffset = Math.max(1, Math.ceil(maxLagM / Math.max(1, Math.min(...axisLengths))) + 1);

  for (let sourceIndex = 0; sourceIndex < cells.length; sourceIndex += 1) {
    const source = cells[sourceIndex];
    for (let dj = -maxOffset; dj <= maxOffset; dj += 1) {
      for (let di = -maxOffset; di <= maxOffset; di += 1) {
        if (dj < 0 || (dj === 0 && di <= 0)) continue;
        const targetEntry = byGrid.get(key(source.i + di, source.j + dj));
        if (!targetEntry) continue;
        const target = targetEntry.cell;
        const distance = vectorDistance(source.x, source.y, target.x, target.y);
        if (!(distance > 0) || distance > maxLagM) continue;
        const bin = Math.floor((distance + 1e-7) / binWidthM);
        if (bin < 0 || bin >= binCount) continue;
        const difference = residuals[sourceIndex] - residuals[targetEntry.index];
        sums[bin] += 0.5 * difference * difference;
        counts[bin] += 1;
      }
    }
  }

  const bins = [];
  for (let index = 0; index < binCount; index += 1) {
    if (counts[index] < minPairsPerBin) continue;
    bins.push({
      bin: index,
      lagMinM: index * binWidthM,
      lagMaxM: (index + 1) * binWidthM,
      lagCenterM: (index + 0.5) * binWidthM,
      pairCount: counts[index],
      semivariance: sums[index] / counts[index],
    });
  }
  if (bins.length < 3) {
    throw new LscBakeAbort(`LSC variogram has only ${bins.length} populated bins; need at least three.`);
  }
  return {bins, maxOffset};
}

function fitExponentialVariogram(bins, {rangeMinM, rangeMaxM, fitWeighting = 'equal_binned_variogram_reference_locked', coarseSamples = 81, refineSamples = 81}) {
  function evaluate(rangeM) {
    let s00 = 0;
    let s01 = 0;
    let s11 = 0;
    let t0 = 0;
    let t1 = 0;
    for (const bin of bins) {
      const weight = fitWeighting === 'pair_count' ? bin.pairCount : 1;
      const q = 1 - Math.exp(-bin.lagCenterM / rangeM);
      s00 += weight;
      s01 += weight * q;
      s11 += weight * q * q;
      t0 += weight * bin.semivariance;
      t1 += weight * q * bin.semivariance;
    }
    const determinant = s00 * s11 - s01 * s01;
    let c0 = 0;
    let c1 = 0;
    if (Math.abs(determinant) > EPS) {
      c0 = (t0 * s11 - s01 * t1) / determinant;
      c1 = (s00 * t1 - s01 * t0) / determinant;
    }
    c0 = Math.max(0, c0);
    c1 = Math.max(0, c1);
    let objective = 0;
    for (const bin of bins) {
      const fitted = c0 + c1 * (1 - Math.exp(-bin.lagCenterM / rangeM));
      const residual = bin.semivariance - fitted;
      objective += (fitWeighting === 'pair_count' ? bin.pairCount : 1) * residual * residual;
    }
    return {rangeM, c0, c1, objective};
  }

  let best = null;
  const logMin = Math.log(rangeMinM);
  const logMax = Math.log(rangeMaxM);
  for (let index = 0; index < coarseSamples; index += 1) {
    const t = index / Math.max(1, coarseSamples - 1);
    const candidate = evaluate(Math.exp(logMin + (logMax - logMin) * t));
    if (!best || candidate.objective < best.objective) best = candidate;
  }

  const refineMin = Math.max(rangeMinM, best.rangeM / 1.75);
  const refineMax = Math.min(rangeMaxM, best.rangeM * 1.75);
  const refineLogMin = Math.log(refineMin);
  const refineLogMax = Math.log(refineMax);
  for (let index = 0; index < refineSamples; index += 1) {
    const t = index / Math.max(1, refineSamples - 1);
    const candidate = evaluate(Math.exp(refineLogMin + (refineLogMax - refineLogMin) * t));
    if (candidate.objective < best.objective) best = candidate;
  }
  return best;
}

function luDecompose(source) {
  const n = source.length;
  const lu = Array.from({length: n}, (_, row) => Float64Array.from(source[row]));
  const pivot = Int32Array.from({length: n}, (_, index) => index);

  for (let column = 0; column < n; column += 1) {
    let pivotRow = column;
    let pivotAbs = Math.abs(lu[pivotRow][column]);
    for (let row = column + 1; row < n; row += 1) {
      const candidate = Math.abs(lu[row][column]);
      if (candidate > pivotAbs) {
        pivotRow = row;
        pivotAbs = candidate;
      }
    }
    if (!(pivotAbs > EPS)) throw new Error('Kriging matrix is singular.');
    if (pivotRow !== column) {
      [lu[pivotRow], lu[column]] = [lu[column], lu[pivotRow]];
      [pivot[pivotRow], pivot[column]] = [pivot[column], pivot[pivotRow]];
    }
    const diagonal = lu[column][column];
    for (let row = column + 1; row < n; row += 1) {
      lu[row][column] /= diagonal;
      const factor = lu[row][column];
      for (let k = column + 1; k < n; k += 1) lu[row][k] -= factor * lu[column][k];
    }
  }
  return {lu, pivot};
}

function luSolve(factorization, rhs) {
  const {lu, pivot} = factorization;
  const n = rhs.length;
  const value = new Float64Array(n);
  for (let row = 0; row < n; row += 1) value[row] = rhs[pivot[row]];
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < row; column += 1) value[row] -= lu[row][column] * value[column];
  }
  for (let row = n - 1; row >= 0; row -= 1) {
    for (let column = row + 1; column < n; column += 1) value[row] -= lu[row][column] * value[column];
    value[row] /= lu[row][row];
  }
  return value;
}

function exponentialCovariance(signalSill, rangeM, distanceM) {
  return signalSill * Math.exp(-distanceM / rangeM);
}

function makeKrigingFactor(neighbours, component, model) {
  const n = neighbours.length;
  const matrix = Array.from({length: n + 1}, () => new Float64Array(n + 1));
  const varianceKey = component === 'east' ? 'varEast' : 'varNorth';
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      matrix[row][column] = exponentialCovariance(
        model.c1,
        model.a,
        vectorDistance(neighbours[row].x, neighbours[row].y, neighbours[column].x, neighbours[column].y),
      );
    }
    matrix[row][row] += Math.max(0, neighbours[row][varianceKey]);
    matrix[row][n] = 1;
    matrix[n][row] = 1;
  }
  return luDecompose(matrix);
}

function krigeAt({neighbours, factor, model, component, x, y, varianceFloor}) {
  const n = neighbours.length;
  const rhs = new Float64Array(n + 1);
  for (let index = 0; index < n; index += 1) {
    rhs[index] = exponentialCovariance(model.c1, model.a, vectorDistance(neighbours[index].x, neighbours[index].y, x, y));
  }
  rhs[n] = 1;
  const solution = luSolve(factor, rhs);
  const valueKey = component === 'east' ? 'east' : 'north';
  let prediction = 0;
  let weightedCovariance = 0;
  for (let index = 0; index < n; index += 1) {
    prediction += solution[index] * neighbours[index][valueKey];
    weightedCovariance += solution[index] * rhs[index];
  }
  const variance = Math.max(varianceFloor, model.c1 - weightedCovariance - solution[n]);
  return {prediction, variance, weights: solution};
}

function nearestCandidateSet(parent, liveCells, liveByKey, {neighborCount, maxRadiusM}) {
  const forced = [];
  const forcedKeys = new Set();
  const include = (cell) => {
    if (!cell || forcedKeys.has(key(cell.i, cell.j))) return;
    forced.push(cell);
    forcedKeys.add(key(cell.i, cell.j));
  };
  include(parent);
  for (let dj = -1; dj <= 1; dj += 1) {
    for (let di = -1; di <= 1; di += 1) {
      if (di === 0 && dj === 0) continue;
      include(liveByKey.get(key(parent.i + di, parent.j + dj)));
    }
  }

  const ranked = [];
  for (const candidate of liveCells) {
    if (forcedKeys.has(key(candidate.i, candidate.j))) continue;
    const distance = vectorDistance(parent.x, parent.y, candidate.x, candidate.y);
    if (distance <= maxRadiusM) ranked.push({candidate, distance});
  }
  ranked.sort((a, b) => a.distance - b.distance || a.candidate.j - b.candidate.j || a.candidate.i - b.candidate.i);
  for (const item of ranked) {
    if (forced.length >= neighborCount) break;
    forced.push(item.candidate);
  }
  return forced;
}

function outputAssetPaths(outputDir) {
  return {
    json: path.join(outputDir, OUTPUT_JSON),
    field: path.join(outputDir, OUTPUT_FIELD_F32),
    covariance: path.join(outputDir, OUTPUT_COVARIANCE_F32),
    spawns: path.join(outputDir, OUTPUT_SPAWNS_F32),
  };
}

export async function pruneLscAssets(outputDir) {
  const assets = outputAssetPaths(outputDir);
  await Promise.all(Object.values(assets).map(async (asset) => {
    try { await fs.unlink(asset); } catch (error) { if (error?.code !== 'ENOENT') throw error; }
  }));
}

function normalizeLscConfig(config = {}) {
  const source = config.horizontal_particles?.lsc ?? {};
  const subdivision = Math.max(1, Math.round(finiteNumber(source.subdivision, 4)));
  const neighborCount = Math.max(4, Math.round(finiteNumber(source.neighbor_count, 32)));
  return {
    enabled: source.enabled !== false,
    subdivision,
    neighborCount,
    minNeighbours: Math.max(4, Math.min(neighborCount, Math.round(finiteNumber(source.minimum_neighbors, 4)))),
    maxLagM: Math.max(450, finiteNumber(source.max_lag_m, 9000)),
    binWidthM: Math.max(25, finiteNumber(source.bin_width_m, 450)),
    minPairsPerBin: Math.max(1, Math.round(finiteNumber(source.minimum_pairs_per_bin, 50))),
    fitWeighting: String(source.variogram_fit_weighting ?? 'equal_binned_variogram_reference_locked') === 'pair_count'
      ? 'pair_count'
      : 'equal_binned_variogram_reference_locked',
    rangeMinM: Math.max(1, finiteNumber(source.range_min_m, 100)),
    rangeMaxM: Math.max(2, finiteNumber(source.range_max_m, 10000)),
    minEffectiveRangeM: Math.max(1, finiteNumber(source.min_effective_range_m, 1000)),
    minSignalSill: Math.max(0, finiteNumber(source.min_signal_sill_mm2_yr2, 0.02)),
    weakSignalNoiseRatio: Math.max(0, finiteNumber(source.weak_signal_noise_ratio, 0.25)),
    maxNeighbourRadiusM: Math.max(1, finiteNumber(source.max_neighbor_radius_m, 4000)),
    maxNeighbourRangeMultiplier: Math.max(0.1, finiteNumber(source.max_neighbor_range_multiplier, 3)),
    predictionVarianceFloor: Math.max(0, finiteNumber(source.prediction_variance_floor_mm2_yr2, 1e-4)),
  };
}

function lscFingerprintPayload({config, sourceHash, rawFieldHash, rawCovarianceHash, rawSpawnsHash, rawGrid}) {
  return {
    schema: LSC_SCHEMA,
    algorithmVersion: LSC_ALGORITHM_VERSION,
    config,
    sourceCsvSha256: sourceHash,
    rawFieldSha256: rawFieldHash,
    rawCovarianceSha256: rawCovarianceHash,
    rawSpawnsSha256: rawSpawnsHash,
    rawGrid,
  };
}

function lscMetadataMessage({east, north}) {
  return {
    variogramTrendPolicy: 'Variogram estimated on second-order residuals to avoid trend contamination of the covariance model; prediction uses ordinary kriging in local moving neighborhoods, whose locally-constant-mean assumption absorbs the regional trend. In pure interpolation with moving neighborhoods this is practically equivalent to trend-model kriging, while avoiding imprinting a global polynomial misfit into the predictions.',
    noiseModel: 'Formal per-cell E/N variance is placed on the kriging diagonal. Fitted nugget is recorded only as a cross-check and is not substituted for the formal variances.',
    crossComponentCovariance: 'Not modeled for the LSC prediction. Covariance texture stores 0.0 in the E/N cross-term; Monte Carlo axes are E/N aligned in LSC mode.',
    estimatedModels: {east, north},
  };
}

function validateRawPayload(rawMetadata, fieldValues, covarianceValues, spawnValues) {
  const grid = rawMetadata.grid ?? {};
  const width = Math.max(1, Math.round(finiteNumber(grid.width, 0)));
  const height = Math.max(1, Math.round(finiteNumber(grid.height, 0)));
  const expected = width * height * 4;
  if (fieldValues.length !== expected || covarianceValues.length !== expected) {
    throw new Error('Raw particle field dimensions do not match the raw field metadata.');
  }
  if (spawnValues.length % 2 !== 0) throw new Error('Raw particle spawn asset has invalid RG layout.');
  if (!Array.isArray(grid.gridOriginLocalM) || !Array.isArray(grid.gridAxisIM) || !Array.isArray(grid.gridAxisJM)) {
    throw new Error('Raw particle field lacks the metric grid required for LSC.');
  }
  return {width, height, grid};
}

function parseSourceLocations(sourceCsvBuffer) {
  const lines = sourceCsvBuffer.toString('utf8').trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('LSC source CSV has no data rows.');
  const headers = lines[0].split(',').map((value) => value.trim());
  const xIndex = headers.indexOf('x_rum');
  const yIndex = headers.indexOf('y_rum');
  if (xIndex < 0 || yIndex < 0) throw new Error('LSC source CSV requires x_rum and y_rum columns.');
  const byRumIndex = new Map();
  for (let row = 1; row < lines.length; row += 1) {
    const values = lines[row].split(',');
    const x = Number(values[xIndex]);
    const y = Number(values[yIndex]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`Invalid LSC source coordinate at CSV row ${row + 1}.`);
    byRumIndex.set(row - 1, {x, y});
  }
  return byRumIndex;
}

function buildLiveCells(rawMetadata, fieldValues, covarianceValues, sourceLocations) {
  const {width, height, grid} = validateRawPayload(rawMetadata, fieldValues, covarianceValues, new Float32Array(0));
  const cells = [];
  for (let j = 0; j < height; j += 1) {
    for (let i = 0; i < width; i += 1) {
      const offset = (j * width + i) * 4;
      if (fieldValues[offset + 3] < 0.5) continue;
      const heightRow = Math.round(fieldValues[offset + 2]);
      const sourceLocation = sourceLocations.get(heightRow);
      if (!sourceLocation) {
        throw new Error(`LSC source coordinate lookup failed for raw RUM row ${heightRow}.`);
      }
      cells.push({
        i,
        j,
        x: sourceLocation.x,
        y: sourceLocation.y,
        east: fieldValues[offset],
        north: fieldValues[offset + 1],
        heightRow,
        varEast: Math.max(0, covarianceValues[offset]),
        varNorth: Math.max(0, covarianceValues[offset + 1]),
        covarEN: covarianceValues[offset + 2],
      });
    }
  }
  if (cells.length < 4) throw new LscBakeAbort('LSC requires at least four live RUMs.');
  return cells;
}

function modelFromVariogram(cells, grid, component, settings) {
  const trend = fitSecondOrderTrend(cells, component);
  const variogram = buildEmpiricalVariogram(cells, trend.residuals, grid, settings);
  const fitted = fitExponentialVariogram(variogram.bins, settings);
  const formalMean = mean(cells.map((cell) => component === 'east' ? cell.varEast : cell.varNorth));
  const model = {
    component,
    c0: fitted.c0,
    c1: fitted.c1,
    a: fitted.rangeM,
    effectiveRangeM: 3 * fitted.rangeM,
    objective: fitted.objective,
    meanFormalVariance: formalMean,
    signalToFormalNoise: formalMean > 0 ? fitted.c1 / formalMean : null,
    fitWeighting: settings.fitWeighting,
  };
  if (model.effectiveRangeM < settings.minEffectiveRangeM) {
    throw new LscBakeAbort(`${component} LSC effective range ${model.effectiveRangeM.toFixed(1)} m is below the ${settings.minEffectiveRangeM} m guard.`);
  }
  if (model.c1 < settings.minSignalSill) {
    throw new LscBakeAbort(`${component} LSC signal sill ${model.c1.toFixed(5)} is below the ${settings.minSignalSill} guard.`);
  }
  return {model, trend: trend.metadata, variogram};
}

function makeFineGrid(rawGrid, subdivision) {
  const fineAxisIM = rawGrid.gridAxisIM.map((value) => value / subdivision);
  const fineAxisJM = rawGrid.gridAxisJM.map((value) => value / subdivision);
  const fineOriginLocalM = [
    rawGrid.gridOriginLocalM[0] - 0.375 * rawGrid.gridAxisIM[0] - 0.375 * rawGrid.gridAxisJM[0],
    rawGrid.gridOriginLocalM[1] - 0.375 * rawGrid.gridAxisIM[1] - 0.375 * rawGrid.gridAxisJM[1],
  ];
  return {
    width: rawGrid.width * subdivision,
    height: rawGrid.height * subdivision,
    maxI: rawGrid.width * subdivision - 1,
    maxJ: rawGrid.height * subdivision - 1,
    rumSizeM: Number(rawGrid.rumSizeM ?? 450) / subdivision,
    coarseRumSizeM: Number(rawGrid.rumSizeM ?? 450),
    coordinateOriginLonLat: rawGrid.coordinateOriginLonLat.slice(0, 2).map(Number),
    gridOriginLocalM: fineOriginLocalM,
    gridAxisIM: fineAxisIM,
    gridAxisJM: fineAxisJM,
    subdivision,
  };
}

function fineIndexToCoarseCoordinate(index, subdivision) {
  return index / subdivision - 0.375;
}

function fineGridToLocal(fineGrid, i, j) {
  return [
    fineGrid.gridOriginLocalM[0] + fineGrid.gridAxisIM[0] * i + fineGrid.gridAxisJM[0] * j,
    fineGrid.gridOriginLocalM[1] + fineGrid.gridAxisIM[1] * i + fineGrid.gridAxisJM[1] * j,
  ];
}

function finePrediction({cells, rawGrid, fineGrid, rawSpawns, eastModel, northModel, settings, sourceGrid}) {
  const liveByKey = new Map(cells.map((cell) => [key(cell.i, cell.j), cell]));
  const parentSystems = new Map();
  const warnings = [];
  const validation = {
    coarseCenterAbsErrorEast: [],
    coarseCenterAbsErrorNorth: [],
    coarseCenterSigmaEast: [],
    coarseCenterSigmaNorth: [],
    interiorSigma: [],
    boundarySigma: [],
    fallbackCount: 0,
    factorizationFallbackCount: 0,
  };
  const getParentSystem = (parent) => {
    const parentKey = key(parent.i, parent.j);
    if (parentSystems.has(parentKey)) return parentSystems.get(parentKey);
    const maxRadiusM = Math.min(settings.maxNeighbourRadiusM, Math.min(3 * eastModel.a, 3 * northModel.a));
    const neighbours = nearestCandidateSet(parent, cells, liveByKey, {
      neighborCount: settings.neighborCount,
      maxRadiusM,
    });
    let system = {parent, neighbours, fallback: neighbours.length < settings.minNeighbours, eastFactor: null, northFactor: null};
    if (!system.fallback) {
      try {
        system.eastFactor = makeKrigingFactor(neighbours, 'east', eastModel);
        system.northFactor = makeKrigingFactor(neighbours, 'north', northModel);
      } catch (error) {
        system.fallback = true;
        validation.factorizationFallbackCount += 1;
        warnings.push(`Kriging factorization fallback at coarse parent ${parent.i},${parent.j}: ${error.message}`);
      }
    }
    parentSystems.set(parentKey, system);
    return system;
  };

  const field = new Float32Array(fineGrid.width * fineGrid.height * 4);
  const covariance = new Float32Array(fineGrid.width * fineGrid.height * 4);
  const signalSpeeds = [];

  const predict = (system, x, y) => {
    const {parent} = system;
    if (system.fallback) {
      validation.fallbackCount += 1;
      return {
        east: parent.east,
        north: parent.north,
        varEast: parent.varEast,
        varNorth: parent.varNorth,
      };
    }
    const east = krigeAt({
      neighbours: system.neighbours,
      factor: system.eastFactor,
      model: eastModel,
      component: 'east',
      x,
      y,
      varianceFloor: settings.predictionVarianceFloor,
    });
    const north = krigeAt({
      neighbours: system.neighbours,
      factor: system.northFactor,
      model: northModel,
      component: 'north',
      x,
      y,
      varianceFloor: settings.predictionVarianceFloor,
    });
    return {east: east.prediction, north: north.prediction, varEast: east.variance, varNorth: north.variance};
  };

  // Coarse-centre validation uses exactly the same fixed-parent matrix and
  // prediction algebra as the fine texels. No special leave-one-out shortcut.
  for (const parent of cells) {
    const system = getParentSystem(parent);
    const predicted = predict(system, parent.x, parent.y);
    validation.coarseCenterAbsErrorEast.push(Math.abs(predicted.east - parent.east));
    validation.coarseCenterAbsErrorNorth.push(Math.abs(predicted.north - parent.north));
    validation.coarseCenterSigmaEast.push(Math.sqrt(Math.max(0, predicted.varEast)));
    validation.coarseCenterSigmaNorth.push(Math.sqrt(Math.max(0, predicted.varNorth)));
    let completeRing = true;
    for (let dj = -1; dj <= 1; dj += 1) {
      for (let di = -1; di <= 1; di += 1) {
        if (!liveByKey.has(key(parent.i + di, parent.j + dj))) completeRing = false;
      }
    }
    const sigmaMean = 0.5 * (Math.sqrt(Math.max(0, predicted.varEast)) + Math.sqrt(Math.max(0, predicted.varNorth)));
    (completeRing ? validation.interiorSigma : validation.boundarySigma).push(sigmaMean);
  }

  for (let fj = 0; fj < fineGrid.height; fj += 1) {
    const coarseJ = fineIndexToCoarseCoordinate(fj, fineGrid.subdivision);
    const parentJ = Math.round(coarseJ);
    for (let fi = 0; fi < fineGrid.width; fi += 1) {
      const coarseI = fineIndexToCoarseCoordinate(fi, fineGrid.subdivision);
      const parentI = Math.round(coarseI);
      const parent = liveByKey.get(key(parentI, parentJ));
      if (!parent) continue; // strict parity: no prediction into blank/no-data.
      const x = sourceGrid.originX + coarseI * sourceGrid.axisIM[0] + coarseJ * sourceGrid.axisJM[0];
      const y = sourceGrid.originY + coarseI * sourceGrid.axisIM[1] + coarseJ * sourceGrid.axisJM[1];
      const predicted = predict(getParentSystem(parent), x, y);
      const offset = (fj * fineGrid.width + fi) * 4;
      field[offset] = predicted.east;
      field[offset + 1] = predicted.north;
      field[offset + 2] = parent.heightRow;
      field[offset + 3] = 1;
      covariance[offset] = predicted.varEast;
      covariance[offset + 1] = predicted.varNorth;
      covariance[offset + 2] = 0;
      covariance[offset + 3] = Math.hypot(predicted.east, predicted.north);
      signalSpeeds.push(covariance[offset + 3]);
    }
  }

  const spawns = new Float32Array(rawSpawns.length);
  for (let index = 0; index < rawSpawns.length; index += 1) {
    spawns[index] = rawSpawns[index] * fineGrid.subdivision + 1.5;
  }

  return {
    field,
    covariance,
    spawns,
    summary: {
      validFineTexelCount: signalSpeeds.length,
      sourceLiveRumCount: cells.length,
      spawnCellCount: rawSpawns.length / 2,
      speedP95MmYr: Math.max(1e-9, percentile(signalSpeeds.filter((value) => value > 0), 95)),
      meanAbsPredMinusRawAtCoarseCentersMmYr: {
        east: mean(validation.coarseCenterAbsErrorEast),
        north: mean(validation.coarseCenterAbsErrorNorth),
      },
      meanPredSigmaAtCoarseCentersMmYr: {
        east: mean(validation.coarseCenterSigmaEast),
        north: mean(validation.coarseCenterSigmaNorth),
      },
      meanPredSigmaMmYrBySupport: {
        interior: mean(validation.interiorSigma),
        boundary: mean(validation.boundarySigma),
      },
      fallbackCount: validation.fallbackCount,
      factorizationFallbackCount: validation.factorizationFallbackCount,
      fixedParentSystemCount: parentSystems.size,
    },
    warnings,
  };
}

export async function buildLscField({root, outputDir, config, force = false, logger = console}) {
  const settings = normalizeLscConfig(config);
  if (!settings.enabled) {
    await pruneLscAssets(outputDir);
    return {available: false, reason: 'disabled'};
  }
  if (settings.rangeMaxM <= settings.rangeMinM) {
    throw new LscBakeAbort('LSC range_max_m must be greater than range_min_m.');
  }

  const sourceCsvPath = path.join(root, config.input?.source_csv ?? 'data/jakarta_enu_estimates.csv');
  const rawJsonPath = path.join(outputDir, RAW_FIELD_JSON);
  const rawFieldPath = path.join(outputDir, RAW_FIELD_F32);
  const rawCovariancePath = path.join(outputDir, RAW_COVARIANCE_F32);
  const rawSpawnsPath = path.join(outputDir, RAW_SPAWNS_F32);
  const [sourceCsvBuffer, rawJson, rawFieldBuffer, rawCovarianceBuffer, rawSpawnsBuffer] = await Promise.all([
    fs.readFile(sourceCsvPath),
    readJson(rawJsonPath),
    fs.readFile(rawFieldPath),
    fs.readFile(rawCovariancePath),
    fs.readFile(rawSpawnsPath),
  ]);
  const rawFieldValues = new Float32Array(rawFieldBuffer.buffer.slice(rawFieldBuffer.byteOffset, rawFieldBuffer.byteOffset + rawFieldBuffer.byteLength));
  const rawCovarianceValues = new Float32Array(rawCovarianceBuffer.buffer.slice(rawCovarianceBuffer.byteOffset, rawCovarianceBuffer.byteOffset + rawCovarianceBuffer.byteLength));
  const rawSpawns = new Float32Array(rawSpawnsBuffer.buffer.slice(rawSpawnsBuffer.byteOffset, rawSpawnsBuffer.byteOffset + rawSpawnsBuffer.byteLength));
  const {width: rawWidth, height: rawHeight, grid: rawGrid} = validateRawPayload(rawJson, rawFieldValues, rawCovarianceValues, rawSpawns);

  const cachePayload = lscFingerprintPayload({
    config: settings,
    sourceHash: sha256(sourceCsvBuffer),
    rawFieldHash: sha256(rawFieldBuffer),
    rawCovarianceHash: sha256(rawCovarianceBuffer),
    rawSpawnsHash: sha256(rawSpawnsBuffer),
    rawGrid: {
      width: rawWidth,
      height: rawHeight,
      gridOriginLocalM: rawGrid.gridOriginLocalM,
      gridAxisIM: rawGrid.gridAxisIM,
      gridAxisJM: rawGrid.gridAxisJM,
    },
  });
  const fingerprint = sha256(stableJson(cachePayload));
  const paths = outputAssetPaths(outputDir);

  if (!force) {
    try {
      const existing = await readJson(paths.json);
      await Promise.all([fs.access(paths.field), fs.access(paths.covariance), fs.access(paths.spawns)]);
      if (existing?.cache?.fingerprint === fingerprint && existing?.schema === LSC_SCHEMA) {
        logger.log(`[LSC] cache hit (${fingerprint.slice(0, 12)}); retained optional signal field.`);
        return {available: true, cached: true, metadata: existing, assetName: OUTPUT_JSON};
      }
    } catch {
      // Missing or incompatible cache: rebuild below.
    }
  }

  const sourceLocations = parseSourceLocations(sourceCsvBuffer);
  const cells = buildLiveCells(rawJson, rawFieldValues, rawCovarianceValues, sourceLocations);
  const sourceAxisIM = [Number(rawJson.grid?.rumSizeM ?? 450), 0];
  const sourceAxisJM = [0, Number(rawJson.grid?.rumSizeM ?? 450)];
  const sourceGrid = {
    originX: mean(cells.map((cell) => cell.x - cell.i * sourceAxisIM[0] - cell.j * sourceAxisJM[0])),
    originY: mean(cells.map((cell) => cell.y - cell.i * sourceAxisIM[1] - cell.j * sourceAxisJM[1])),
    axisIM: sourceAxisIM,
    axisJM: sourceAxisJM,
  };
  logger.log(`[LSC] fitting local signal field from ${cells.length.toLocaleString()} observed RUMs…`);
  const eastFit = modelFromVariogram(cells, rawGrid, 'east', settings);
  const northFit = modelFromVariogram(cells, rawGrid, 'north', settings);
  const warnings = [];
  for (const model of [eastFit.model, northFit.model]) {
    if (model.signalToFormalNoise !== null && model.signalToFormalNoise < settings.weakSignalNoiseRatio) {
      warnings.push(`${model.component}: weak signal-to-formal-noise ratio ${model.signalToFormalNoise.toFixed(3)} < ${settings.weakSignalNoiseRatio}.`);
    }
  }

  const fineGrid = makeFineGrid({...rawGrid, width: rawWidth, height: rawHeight}, settings.subdivision);
  const prediction = finePrediction({
    cells,
    rawGrid,
    fineGrid,
    rawSpawns,
    eastModel: eastFit.model,
    northModel: northFit.model,
    settings,
    sourceGrid,
  });
  warnings.push(...prediction.warnings);

  const rawRender = rawJson.render ?? {};
  const metadata = {
    schema: LSC_SCHEMA,
    purpose: 'Optional least-squares-collocation horizontal deformation signal field. Raw estimates remain available unchanged as a separate particle flow-field mode.',
    units: rawJson.units,
    sampling: {
      mode: 'lsc_signal_field',
      rule: 'Fine LSC texture sampled through the existing conservative-v1 shader rule. Fine texels are valid only where their nearest coarse RUM is observed; no blank/no-data prediction is emitted.',
      rawComparisonMode: 'raw estimates · conservative bilinear v1',
    },
    grid: fineGrid,
    spawnDomain: {
      ...rawJson.spawnDomain,
      rule: 'same raw eight-neighbour-supported coarse emitter set, re-expressed in fine-grid coordinates',
      coarseSpawnCellCount: rawSpawns.length / 2,
    },
    render: {
      ...rawRender,
      samplerMode: 'lsc_signal_field',
      spawnJitterCells: 0.90 * settings.subdivision,
      fieldMode: 'lsc',
    },
    history: rawJson.history,
    summary: prediction.summary,
    lscModel: {
      algorithm: 'ordinary_kriging_with_heteroscedastic_formal_noise',
      algorithmVersion: LSC_ALGORITHM_VERSION,
      subdivision: settings.subdivision,
      variogramFitWeighting: settings.fitWeighting,
      variogramFitWeightingMeaning: settings.fitWeighting === 'pair_count'
        ? 'Weighted least squares using empirical-bin pair counts.'
        : 'Equal weight per populated empirical semivariogram bin. This is reference-locked because it reproduces the Jakarta validation parameters provided with the LSC handoff; pair counts remain recorded for audit.',
      neighborhood: `fixed_per_parent_cell_k${settings.neighborCount}`,
      neighborCount: settings.neighborCount,
      forcedImmediateNeighbourPolicy: 'parent plus all available live Moore-8 neighbours are retained before distance ranking.',
      fixedNeighbourTradeoff: 'Fixed parent-cell neighbour membership reuses two component LU factorizations across 16 fine texels. Weights remain exact for that fixed set; microscopic coarse-boundary membership seams are an explicit approximation.',
      domainRule: 'fine texel valid iff the nearest coarse grid cell is a live observed RUM; no LSC value is generated into blankies or unfilled no-data.',
      sourceCoordinateGrid: sourceGrid,
      predictionRule: `nearest ${settings.neighborCount} live RUMs within min(3a, ${settings.maxNeighbourRadiusM} m), with fallback to nearest raw RUM when fewer than ${settings.minNeighbours} neighbours are available.`,
      predictionVarianceFloorMm2Yr2: settings.predictionVarianceFloor,
      east: {
        ...eastFit.model,
        variogram: eastFit.variogram,
        detrend: eastFit.trend,
      },
      north: {
        ...northFit.model,
        variogram: northFit.variogram,
        detrend: northFit.trend,
      },
      meaning: lscMetadataMessage({east: eastFit.model, north: northFit.model}),
      validation: prediction.summary,
    },
    cache: {
      fingerprint,
      algorithmVersion: LSC_ALGORITHM_VERSION,
      sourceCsvSha256: cachePayload.sourceCsvSha256,
      rawFieldSha256: cachePayload.rawFieldSha256,
      rawCovarianceSha256: cachePayload.rawCovarianceSha256,
      rawSpawnsSha256: cachePayload.rawSpawnsSha256,
      config: settings,
    },
    warnings,
    assets: {
      fieldF32: OUTPUT_FIELD_F32,
      covarianceF32: OUTPUT_COVARIANCE_F32,
      spawnGridF32: OUTPUT_SPAWNS_F32,
    },
  };

  await Promise.all([
    writeFloat32(paths.field, prediction.field),
    writeFloat32(paths.covariance, prediction.covariance),
    writeFloat32(paths.spawns, prediction.spawns),
  ]);
  await writeJson(paths.json, metadata);

  logger.log(`[LSC] east: c0=${eastFit.model.c0.toFixed(3)}, c1=${eastFit.model.c1.toFixed(3)}, a=${eastFit.model.a.toFixed(0)} m; north: c0=${northFit.model.c0.toFixed(3)}, c1=${northFit.model.c1.toFixed(3)}, a=${northFit.model.a.toFixed(0)} m.`);
  logger.log(`[LSC] fine grid ${fineGrid.width}×${fineGrid.height}; ${prediction.summary.validFineTexelCount.toLocaleString()} valid texels; mean |pred−raw| east/north ${prediction.summary.meanAbsPredMinusRawAtCoarseCentersMmYr.east.toFixed(3)}/${prediction.summary.meanAbsPredMinusRawAtCoarseCentersMmYr.north.toFixed(3)} mm/yr.`);
  return {available: true, cached: false, metadata, assetName: OUTPUT_JSON};
}

const isDirectRun = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (isDirectRun) {
  const root = process.cwd();
  const config = await readJson(path.join(root, 'config', 'project_config.json'));
  const force = process.argv.includes('--force-lsc');
  try {
    const result = await buildLscField({
      root,
      outputDir: path.join(root, 'public', 'data', 'jakarta'),
      config,
      force,
    });
    if (!result.available) console.log('[LSC] optional LSC field disabled.');
  } catch (error) {
    await pruneLscAssets(path.join(root, 'public', 'data', 'jakarta'));
    console.warn(`[LSC] optional bake skipped: ${error.message ?? error}`);
    process.exitCode = 0;
  }
}
