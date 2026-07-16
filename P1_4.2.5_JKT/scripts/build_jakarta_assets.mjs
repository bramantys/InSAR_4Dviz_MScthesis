import fs from 'node:fs/promises';
import path from 'node:path';
import proj4 from 'proj4';
import {buildLscField, pruneLscAssets} from './lsc_field.mjs';

const ROOT = process.cwd();
const CONFIG_PATH = path.join(ROOT, 'config', 'project_config.json');
const OUTPUT_DIR = path.join(ROOT, 'public', 'data', 'jakarta');
const WGS84 = 'WGS84';

function requireFinite(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number; got ${value}.`);
  }
  return value;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV is empty or contains no data rows.');

  const headers = lines[0].split(',').map((value) => value.trim());
  return lines.slice(1).map((line, rowIndex) => {
    const values = line.split(',');
    if (values.length !== headers.length) {
      throw new Error(
        `CSV row ${rowIndex + 2} has ${values.length} values; expected ${headers.length}.`,
      );
    }
    return Object.fromEntries(headers.map((header, index) => [header, values[index].trim()]));
  });
}

function requireColumns(rows, requiredColumns) {
  const available = new Set(Object.keys(rows[0] ?? {}));
  for (const column of requiredColumns) {
    if (!available.has(column)) {
      throw new Error(
        `Required CSV column "${column}" is missing. Available columns: ${[...available].join(', ')}`,
      );
    }
  }
}

function makeUtcDate(isoDate) {
  const [year, month, day] = isoDate.split('-').map(Number);
  return new Date(Date.UTC(year, month - 1, day));
}

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function buildEpochs({start_date: startDate, end_date: endDate, interval_days: intervalDays}) {
  const start = makeUtcDate(startDate);
  const end = makeUtcDate(endDate);
  const stepMs = intervalDays * 24 * 60 * 60 * 1000;

  const epochs = [];
  const yearsSinceStart = [];

  for (
    let cursor = new Date(start.getTime());
    cursor <= end;
    cursor = new Date(cursor.getTime() + stepMs)
  ) {
    epochs.push(isoDate(cursor));
    yearsSinceStart.push(
      (cursor.getTime() - start.getTime()) / (365.25 * 24 * 60 * 60 * 1000),
    );
  }

  if (epochs.at(-1) !== endDate) {
    epochs.push(endDate);
    yearsSinceStart.push(
      (end.getTime() - start.getTime()) / (365.25 * 24 * 60 * 60 * 1000),
    );
  }

  return {epochs, yearsSinceStart};
}

function percentile(values, q) {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;

  const index = (sorted.length - 1) * q;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return sorted[low];
  const fraction = index - low;
  return sorted[low] * (1 - fraction) + sorted[high] * fraction;
}

function sortedUnique(values) {
  return [...new Set(values)].sort((a, b) => a - b);
}


function ceilToStep(value, step) {
  if (!Number.isFinite(value) || value <= 0 || !Number.isFinite(step) || step <= 0) return 0;
  return Math.ceil(value / step) * step;
}

function roundAdaptiveP98Limit(value, rounding = {}) {
  if (!Number.isFinite(value) || value <= 0) return 0;
  const under10 = Number(rounding.under_10_step_mm_yr ?? 1);
  const from10To20 = Number(rounding.from_10_to_20_step_mm_yr ?? 2);
  const from20To100 = Number(rounding.from_20_to_100_step_mm_yr ?? 5);
  const over100 = Number(rounding.over_100_step_mm_yr ?? 10);
  const step = value < 10
    ? under10
    : value < 20
      ? from10To20
      : value <= 100
        ? from20To100
        : over100;
  return ceilToStep(value, step);
}

function normaliseHexColor(value, fallback) {
  const text = String(value ?? fallback ?? '').trim();
  const clean = text.startsWith('#') ? text.slice(1) : text;
  const expanded = clean.length === 3 ? clean.split('').map((character) => character + character).join('') : clean;
  return /^[0-9a-fA-F]{6}$/.test(expanded) ? `#${expanded.toLowerCase()}` : fallback;
}

function formatRateLegendValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || Math.abs(number) < 1e-10) return '0';
  const absolute = Math.abs(number);
  const digits = absolute >= 10 ? 0 : absolute >= 1 ? 1 : 2;
  const trimmed = number.toFixed(digits).replace(/\.0$/, '').replace(/(\.[0-9]*?)0+$/, '$1').replace(/\.$/, '');
  return number > 0 ? `+${trimmed}` : `−${Math.abs(Number(trimmed)).toString()}`;
}

function normalizedPalette(source = {}) {
  const fallback = {
    subsidence: ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7'],
    neutral: '#f7f7f7',
    uplift: ['#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
  };
  const asFive = (candidate, fallbackColors) => {
    const raw = Array.isArray(candidate) ? candidate : fallbackColors;
    return fallbackColors.map((fallbackColor, index) => normaliseHexColor(raw[index], fallbackColor));
  };
  return {
    subsidence: asFive(source.subsidence, fallback.subsidence),
    neutral: normaliseHexColor(source.neutral, fallback.neutral),
    uplift: asFive(source.uplift, fallback.uplift),
  };
}

function dedupeAscendingStops(stops) {
  const output = [];
  for (const stop of stops.sort((a, b) => a.valueMmYr - b.valueMmYr)) {
    const prior = output.at(-1);
    if (prior && Math.abs(prior.valueMmYr - stop.valueMmYr) < 1e-9) {
      // Retain the later semantic stop at an exact shared value. This only
      // matters for one-sided/stable fallback scales where several palette
      // knots collapse together; it avoids zero-length interpolation spans.
      output[output.length - 1] = stop;
    } else {
      output.push(stop);
    }
  }
  return output;
}

function buildManualVerticalVelocityColorScale(source = {}) {
  const manualStops = Array.isArray(source.manual_stops) ? source.manual_stops : [];
  const parsedStops = dedupeAscendingStops(manualStops.map((stop, index) => ({
    valueMmYr: requireFinite(Number(stop.value_mm_yr ?? stop.value), `manual vertical colour stop ${index + 1}`),
    color: normaliseHexColor(stop.color ?? stop.color_hex, '#999999'),
    role: String(stop.role ?? ''),
    positionPct: Number.isFinite(Number(stop.position_pct ?? stop.positionPct))
      ? Number(stop.position_pct ?? stop.positionPct)
      : undefined,
  })));
  if (parsedStops.length < 2) {
    throw new Error('style.vertical_velocity_color_scale.mode=manual requires at least two manual_stops.');
  }
  const tau = Math.max(0, Number(source.near_zero_threshold_mm_yr ?? 0));
  return {
    schema: 'adaptive_diverging_vertical_velocity_scale_v1',
    mode: 'manual',
    field: String(source.velocity_field ?? 'up'),
    varianceField: String(source.variance_field ?? 'var_up'),
    unit: String(source.unit ?? 'mm/yr'),
    zeroReferenceMmYr: Number(source.zero_reference_mm_yr ?? 0),
    nearZeroThresholdMmYr: tau,
    nearZeroThresholdRawMmYr: tau,
    subsidenceLimitMmYr: Math.abs(parsedStops[0].valueMmYr),
    upliftLimitMmYr: Math.abs(parsedStops.at(-1).valueMmYr),
    clipped: true,
    stops: parsedStops,
    legend: {
      title: 'Vertical velocity · mm/yr',
      labels: parsedStops.map((stop, index) => ({
        valueMmYr: stop.valueMmYr,
        positionPct: Number.isFinite(stop.positionPct)
          ? stop.positionPct
          : 100 * index / Math.max(1, parsedStops.length - 1),
        label: index === 0
          ? `≤ ${formatRateLegendValue(stop.valueMmYr)}`
          : index === parsedStops.length - 1
            ? `≥ ${formatRateLegendValue(stop.valueMmYr)}`
            : formatRateLegendValue(stop.valueMmYr),
      })),
      note: 'Manual synthetic display limits. Actual velocity remains available through inspection.',
    },
    meaning: String(source.meaning ?? 'Manual synthetic diverging vertical-velocity display scale.'),
  };
}

function buildAdaptiveVerticalVelocityColorScale(rums, source = {}) {
  if (String(source.mode ?? 'adaptive').toLowerCase() === 'manual') {
    return buildManualVerticalVelocityColorScale(source);
  }

  const unit = String(source.unit ?? 'mm/yr');
  const uncertaintyPercentile = clamp(Number(source.uncertainty_percentile ?? 75), 0, 100);
  const colourPercentile = clamp(Number(source.colour_percentile ?? 98), 0, 100);
  const sigmaMultiplier = Math.max(0, Number(source.near_zero_sigma_multiplier ?? 2));
  const nearZeroRoundStep = Math.max(0.01, Number(source.near_zero_round_up_step_mm_yr ?? 0.5));
  const minActiveFraction = clamp(Number(source.min_active_fraction ?? 0.01), 0, 1);
  const palette = normalizedPalette(source.palette ?? {});
  const rounding = source.p98_rounding ?? {};
  const values = rums.map((rum) => rum.upMmYr).filter(Number.isFinite);
  const twoSigma = rums
    .map((rum) => sigmaMultiplier * Math.sqrt(Math.max(0, Number(rum.varUp))))
    .filter(Number.isFinite);
  if (!values.length || !twoSigma.length) {
    throw new Error('Cannot build vertical velocity colour scale: no finite up/var_up values.');
  }

  const tauRaw = percentile(twoSigma, uncertaintyPercentile / 100);
  const tau = Math.max(nearZeroRoundStep, ceilToStep(tauRaw, nearZeroRoundStep));
  const subsidenceMagnitudes = values.filter((value) => value < -tau).map((value) => Math.abs(value));
  const upliftMagnitudes = values.filter((value) => value > tau);
  const negActive = subsidenceMagnitudes.length / values.length >= minActiveFraction;
  const posActive = upliftMagnitudes.length / values.length >= minActiveFraction;

  let subsidenceLimitRaw = negActive ? percentile(subsidenceMagnitudes, colourPercentile / 100) : 0;
  let upliftLimitRaw = posActive ? percentile(upliftMagnitudes, colourPercentile / 100) : 0;
  let subsidenceLimit = negActive ? roundAdaptiveP98Limit(subsidenceLimitRaw, rounding) : 0;
  let upliftLimit = posActive ? roundAdaptiveP98Limit(upliftLimitRaw, rounding) : 0;

  if (negActive && !posActive) upliftLimit = Math.max(subsidenceLimit, tau);
  else if (posActive && !negActive) subsidenceLimit = Math.max(upliftLimit, tau);
  else if (!negActive && !posActive) {
    subsidenceLimit = Math.max(5 * tau, tau);
    upliftLimit = Math.max(5 * tau, tau);
  }
  subsidenceLimit = Math.max(subsidenceLimit, tau);
  upliftLimit = Math.max(upliftLimit, tau);

  const negativeStops = palette.subsidence.map((color, index) => {
    const t = index / Math.max(1, palette.subsidence.length - 1);
    const magnitude = subsidenceLimit - (subsidenceLimit - tau) * t;
    return {
      valueMmYr: -magnitude,
      color,
      role: index === 0 ? 'clipped_subsidence_limit' : index === palette.subsidence.length - 1 ? 'near_zero_subsidence_edge' : 'subsidence_gradient',
    };
  });
  const positiveStops = palette.uplift.map((color, index) => {
    const t = index / Math.max(1, palette.uplift.length - 1);
    const value = tau + (upliftLimit - tau) * t;
    return {
      valueMmYr: value,
      color,
      role: index === 0 ? 'near_zero_uplift_edge' : index === palette.uplift.length - 1 ? 'clipped_uplift_limit' : 'uplift_gradient',
    };
  });
  const stops = dedupeAscendingStops([
    ...negativeStops,
    {valueMmYr: 0, color: palette.neutral, role: 'zero_reference'},
    ...positiveStops,
  ]);

  const legendPositions = source.legend_near_zero_positions_pct ?? {};
  const negEdgePosition = clamp(Number(legendPositions.negative_edge ?? 42), 1, 49.9);
  const zeroPosition = clamp(Number(legendPositions.zero ?? 50), negEdgePosition + 0.1, 99.8);
  const posEdgePosition = clamp(Number(legendPositions.positive_edge ?? 58), zeroPosition + 0.1, 99.9);
  const legendLabels = [
    {valueMmYr: -subsidenceLimit, positionPct: 0, label: `≤ ${formatRateLegendValue(-subsidenceLimit)}`},
    {valueMmYr: -tau, positionPct: negEdgePosition, label: formatRateLegendValue(-tau)},
    {valueMmYr: 0, positionPct: zeroPosition, label: '0'},
    {valueMmYr: tau, positionPct: posEdgePosition, label: formatRateLegendValue(tau)},
    {valueMmYr: upliftLimit, positionPct: 100, label: `≥ ${formatRateLegendValue(upliftLimit)}`},
  ];

  return {
    schema: 'adaptive_diverging_vertical_velocity_scale_v1',
    mode: 'adaptive_uncertainty_p75_sign_specific_p98',
    field: String(source.velocity_field ?? 'up'),
    varianceField: String(source.variance_field ?? 'var_up'),
    unit,
    zeroReferenceMmYr: 0,
    uncertaintyPercentile,
    colourPercentile,
    nearZeroSigmaMultiplier: sigmaMultiplier,
    nearZeroThresholdRawMmYr: tauRaw,
    nearZeroThresholdMmYr: tau,
    nearZeroRoundUpStepMmYr: nearZeroRoundStep,
    minActiveFraction,
    subsidenceLimitRawMmYr: subsidenceLimitRaw,
    upliftLimitRawMmYr: upliftLimitRaw,
    subsidenceLimitMmYr: subsidenceLimit,
    upliftLimitMmYr: upliftLimit,
    subsidenceActive: negActive,
    upliftActive: posActive,
    observationCount: values.length,
    meaningfulSubsidenceCount: subsidenceMagnitudes.length,
    meaningfulUpliftCount: upliftMagnitudes.length,
    clipped: true,
    p98Rounding: rounding,
    stops,
    legend: {
      title: `Vertical velocity · ${unit}`,
      labels: legendLabels,
      note: `Adaptive display limits · near-zero ±${formatRateLegendValue(tau).replace('+', '')} ${unit} from P${uncertaintyPercentile.toFixed(0)} of ${sigmaMultiplier.toFixed(0)}σ · sign-specific P${colourPercentile.toFixed(0)} clipping`,
    },
    meaning: String(source.meaning ?? 'Adaptive diverging vertical velocity display scale.'),
  };
}

function cellKey(i, j) {
  return `${i}:${j}`;
}

function buildFootprintLonLat({x, y, rumSizeM, sourceProj4}) {
  const half = rumSizeM * 0.5;
  const sourceCornersXY = [
    [x - half, y - half],
    [x + half, y - half],
    [x + half, y + half],
    [x - half, y + half],
  ];

  const footprintLonLat = sourceCornersXY.map(([cornerX, cornerY]) =>
    proj4(sourceProj4, WGS84, [cornerX, cornerY]),
  );

  return {
    sourceCornersXY,
    footprintLonLat: [...footprintLonLat, footprintLonLat[0]],
  };
}

// -----------------------------------------------------------------------------
// Completed support-envelope selection, v2.
//
// The source RUM mask is never traced literally. We build a conservative,
// grid-native working envelope in four explicit stages:
//   1) local diamond closing for pinholes / tiny notches;
//   2) bounded row/column span fill for nearby, clearly bracketed gaps;
//   3) optional shortest-corridor joining of genuinely detached components;
//   4) enclosed-pocket fill.
//
// Every added cell remains a blankie: no direct measurement, IDW model from
// live RUMs only. Candidate cells that cannot be supported by the configured
// live-neighbour rule are rejected before runtime generation.
//
// Connectivity is strictly 4-neighbour. Corner touching is never connectivity.
// -----------------------------------------------------------------------------

const FOUR_NEIGHBOURS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

function inGrid(i, j, grid) {
  return i >= 0 && i <= grid.maxI && j >= 0 && j <= grid.maxJ;
}

function diamondOffsets(radius) {
  const offsets = [];
  for (let dj = -radius; dj <= radius; dj += 1) {
    for (let di = -radius; di <= radius; di += 1) {
      if (Math.abs(di) + Math.abs(dj) <= radius) offsets.push([di, dj]);
    }
  }
  return offsets;
}

function dilateMask(mask, grid, offsets) {
  const result = new Set();

  for (const key of mask) {
    const [i, j] = key.split(':').map(Number);
    for (const [di, dj] of offsets) {
      const ni = i + di;
      const nj = j + dj;
      if (inGrid(ni, nj, grid)) result.add(cellKey(ni, nj));
    }
  }

  return result;
}

function erodeMask(mask, grid, offsets) {
  const result = new Set();

  for (let j = 0; j <= grid.maxJ; j += 1) {
    for (let i = 0; i <= grid.maxI; i += 1) {
      let survives = true;
      for (const [di, dj] of offsets) {
        const ni = i + di;
        const nj = j + dj;
        if (!inGrid(ni, nj, grid) || !mask.has(cellKey(ni, nj))) {
          survives = false;
          break;
        }
      }
      if (survives) result.add(cellKey(i, j));
    }
  }

  return result;
}

function fillEnclosedHoles(mask, grid) {
  const outsideNoData = new Set();
  const queue = [];

  function enqueueIfOpen(i, j) {
    const key = cellKey(i, j);
    if (!inGrid(i, j, grid) || mask.has(key) || outsideNoData.has(key)) return;
    outsideNoData.add(key);
    queue.push([i, j]);
  }

  for (let i = 0; i <= grid.maxI; i += 1) {
    enqueueIfOpen(i, 0);
    enqueueIfOpen(i, grid.maxJ);
  }
  for (let j = 0; j <= grid.maxJ; j += 1) {
    enqueueIfOpen(0, j);
    enqueueIfOpen(grid.maxI, j);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const [i, j] = queue[cursor];
    for (const [di, dj] of FOUR_NEIGHBOURS) enqueueIfOpen(i + di, j + dj);
  }

  const filledMask = new Set(mask);
  const added = new Set();

  for (let j = 0; j <= grid.maxJ; j += 1) {
    for (let i = 0; i <= grid.maxI; i += 1) {
      const key = cellKey(i, j);
      if (mask.has(key) || outsideNoData.has(key)) continue;
      filledMask.add(key);
      added.add(key);
    }
  }

  return {filledMask, added};
}

function addReason(reasonsByKey, key, reason) {
  if (!reasonsByKey.has(key)) reasonsByKey.set(key, new Set());
  reasonsByKey.get(key).add(reason);
}

function addReasons(reasonsByKey, keys, reason) {
  for (const key of keys) addReason(reasonsByKey, key, reason);
}

function listFourNeighbourComponents(mask, grid) {
  const unseen = new Set(mask);
  const components = [];

  while (unseen.size > 0) {
    const start = unseen.values().next().value;
    unseen.delete(start);

    const keys = [start];
    const members = new Set([start]);

    for (let cursor = 0; cursor < keys.length; cursor += 1) {
      const [i, j] = keys[cursor].split(':').map(Number);
      for (const [di, dj] of FOUR_NEIGHBOURS) {
        const neighbourKey = cellKey(i + di, j + dj);
        if (!inGrid(i + di, j + dj, grid) || !unseen.has(neighbourKey)) continue;
        unseen.delete(neighbourKey);
        members.add(neighbourKey);
        keys.push(neighbourKey);
      }
    }

    components.push({members, size: keys.length});
  }

  return components.sort((a, b) => b.size - a.size);
}

function summarizeComponents(mask, grid) {
  const components = listFourNeighbourComponents(mask, grid);
  return {
    componentCount: components.length,
    largestComponentCellCount: components[0]?.size ?? 0,
  };
}

function fillBoundedOrthogonalSpans(mask, grid, config) {
  const enabled = config.enabled !== false;
  const maximumGapCells = Math.max(0, Math.floor(Number(config.max_gap_cells ?? 0)));
  const passes = Math.max(1, Math.floor(Number(config.passes ?? 1)));
  const axes = new Set(config.axes ?? ['row', 'column']);

  const current = new Set(mask);
  const added = new Set();
  const horizontalAdded = new Set();
  const verticalAdded = new Set();
  const passCounts = [];

  if (!enabled || maximumGapCells <= 0) {
    return {
      mask: current,
      added,
      horizontalAdded,
      verticalAdded,
      passCounts,
      maximumGapCells,
      passes: 0,
    };
  }

  for (let pass = 0; pass < passes; pass += 1) {
    const snapshot = new Set(current);
    const passAdded = new Set();

    if (axes.has('row')) {
      for (let j = 0; j <= grid.maxJ; j += 1) {
        let previousI = null;
        for (let i = 0; i <= grid.maxI; i += 1) {
          const key = cellKey(i, j);
          if (!snapshot.has(key)) continue;

          if (previousI !== null) {
            const gap = i - previousI - 1;
            if (gap >= 1 && gap <= maximumGapCells) {
              for (let fillI = previousI + 1; fillI < i; fillI += 1) {
                const fillKey = cellKey(fillI, j);
                if (snapshot.has(fillKey)) continue;
                passAdded.add(fillKey);
                horizontalAdded.add(fillKey);
              }
            }
          }

          previousI = i;
        }
      }
    }

    if (axes.has('column')) {
      for (let i = 0; i <= grid.maxI; i += 1) {
        let previousJ = null;
        for (let j = 0; j <= grid.maxJ; j += 1) {
          const key = cellKey(i, j);
          if (!snapshot.has(key)) continue;

          if (previousJ !== null) {
            const gap = j - previousJ - 1;
            if (gap >= 1 && gap <= maximumGapCells) {
              for (let fillJ = previousJ + 1; fillJ < j; fillJ += 1) {
                const fillKey = cellKey(i, fillJ);
                if (snapshot.has(fillKey)) continue;
                passAdded.add(fillKey);
                verticalAdded.add(fillKey);
              }
            }
          }

          previousJ = j;
        }
      }
    }

    if (passAdded.size === 0) break;
    passCounts.push(passAdded.size);
    for (const key of passAdded) {
      current.add(key);
      added.add(key);
    }
  }

  return {
    mask: current,
    added,
    horizontalAdded,
    verticalAdded,
    passCounts,
    maximumGapCells,
    passes: passCounts.length,
  };
}

function hasEnoughLiveNeighbours(key, liveMask, grid, interpolationConfig) {
  const [i, j] = key.split(':').map(Number);
  const minimumNeighbours = Math.max(1, Math.floor(Number(interpolationConfig.min_neighbours ?? 2)));
  const maximumRadiusCells = Math.max(1, Math.floor(Number(interpolationConfig.max_radius_cells ?? 6)));

  let count = 0;
  for (let dj = -maximumRadiusCells; dj <= maximumRadiusCells; dj += 1) {
    for (let di = -maximumRadiusCells; di <= maximumRadiusCells; di += 1) {
      if (Math.max(Math.abs(di), Math.abs(dj)) > maximumRadiusCells) continue;
      const ni = i + di;
      const nj = j + dj;
      if (!inGrid(ni, nj, grid) || !liveMask.has(cellKey(ni, nj))) continue;
      count += 1;
      if (count >= minimumNeighbours) return true;
    }
  }

  return false;
}

function shortestComponentConnector({mainComponent, targetComponent, fullMask, grid}) {
  const sourceKeys = mainComponent.members;
  const targetKeys = targetComponent.members;
  const queue = [];
  const parents = new Map();

  for (const key of sourceKeys) {
    queue.push(key);
    parents.set(key, null);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const key = queue[cursor];
    const [i, j] = key.split(':').map(Number);

    for (const [di, dj] of FOUR_NEIGHBOURS) {
      const ni = i + di;
      const nj = j + dj;
      const neighbourKey = cellKey(ni, nj);

      if (!inGrid(ni, nj, grid) || parents.has(neighbourKey)) continue;

      // Other support components are barriers while finding a direct corridor
      // to this particular target. That prevents accidental teleport-style
      // merges through unrelated islands.
      if (
        fullMask.has(neighbourKey) &&
        !sourceKeys.has(neighbourKey) &&
        !targetKeys.has(neighbourKey)
      ) {
        continue;
      }

      parents.set(neighbourKey, key);

      if (targetKeys.has(neighbourKey)) {
        const path = [];
        let cursorKey = neighbourKey;
        while (cursorKey !== null) {
          path.push(cursorKey);
          cursorKey = parents.get(cursorKey);
        }
        path.reverse();

        return path.filter((pathKey) => !sourceKeys.has(pathKey) && !targetKeys.has(pathKey));
      }

      queue.push(neighbourKey);
    }
  }

  return null;
}

function bridgeDetachedComponents({mask, grid, bridgeConfig, liveMask, interpolationConfig}) {
  const enabled = bridgeConfig.enabled !== false;
  const maximumBridgeGapCells = Math.max(0, Math.floor(Number(bridgeConfig.max_bridge_gap_cells ?? 0)));
  const minimumBridgeGapCells = Math.max(0, Math.floor(Number(bridgeConfig.min_bridge_gap_cells ?? 0)));
  const maximumComponentsToAbsorb = Math.max(0, Math.floor(Number(bridgeConfig.max_components_to_absorb ?? 0)));

  const current = new Set(mask);
  const bridges = [];

  if (!enabled || maximumBridgeGapCells <= 0 || maximumComponentsToAbsorb <= 0) {
    return {mask: current, bridges};
  }

  for (let bridgeIndex = 0; bridgeIndex < maximumComponentsToAbsorb; bridgeIndex += 1) {
    const components = listFourNeighbourComponents(current, grid);
    if (components.length <= 1) break;

    const mainComponent = components[0];
    const candidates = [];

    for (let componentIndex = 1; componentIndex < components.length; componentIndex += 1) {
      const targetComponent = components[componentIndex];
      const corridor = shortestComponentConnector({
        mainComponent,
        targetComponent,
        fullMask: current,
        grid,
      });

      if (!corridor) continue;
      if (corridor.length < minimumBridgeGapCells || corridor.length > maximumBridgeGapCells) continue;
      if (!corridor.every((key) => hasEnoughLiveNeighbours(key, liveMask, grid, interpolationConfig))) continue;

      candidates.push({
        targetComponent,
        corridor,
        gapCells: corridor.length,
      });
    }

    if (candidates.length === 0) break;

    candidates.sort((a, b) => a.gapCells - b.gapCells || b.targetComponent.size - a.targetComponent.size);
    const chosen = candidates[0];

    for (const key of chosen.corridor) current.add(key);
    bridges.push({
      gapCells: chosen.gapCells,
      absorbedComponentCellCount: chosen.targetComponent.size,
      corridor: chosen.corridor,
    });
  }

  return {mask: current, bridges};
}

function selectSupportEnvelopeBlankies({liveRums, grid, selectionConfig, interpolationConfig}) {
  const closingRadiusCells = Math.max(0, Math.floor(Number(selectionConfig.closing_radius_cells ?? 3)));
  const fillHoles = selectionConfig.fill_enclosed_holes !== false;
  const liveMask = new Set(liveRums.map((rum) => cellKey(rum.gridI, rum.gridJ)));
  const reasonsByKey = new Map();

  const offsets = diamondOffsets(closingRadiusCells);
  const closedMask = erodeMask(dilateMask(liveMask, grid, offsets), grid, offsets);
  let workingMask = new Set([...liveMask, ...closedMask]);
  addReasons(
    reasonsByKey,
    [...workingMask].filter((key) => !liveMask.has(key)),
    'support_envelope_diamond_closing',
  );

  if (fillHoles) {
    const result = fillEnclosedHoles(workingMask, grid);
    workingMask = result.filledMask;
    addReasons(reasonsByKey, result.added, 'support_envelope_enclosed_hole_fill');
  }

  const spanResult = fillBoundedOrthogonalSpans(
    workingMask,
    grid,
    selectionConfig.bounded_span_fill ?? {},
  );
  workingMask = spanResult.mask;
  addReasons(reasonsByKey, spanResult.horizontalAdded, 'support_envelope_bounded_row_span_fill');
  addReasons(reasonsByKey, spanResult.verticalAdded, 'support_envelope_bounded_column_span_fill');

  const bridgeResult = bridgeDetachedComponents({
    mask: workingMask,
    grid,
    bridgeConfig: selectionConfig.component_bridge ?? {},
    liveMask,
    interpolationConfig,
  });
  workingMask = bridgeResult.mask;
  for (const bridge of bridgeResult.bridges) {
    addReasons(reasonsByKey, bridge.corridor, 'support_envelope_component_bridge_corridor');
  }

  if (fillHoles) {
    const result = fillEnclosedHoles(workingMask, grid);
    workingMask = result.filledMask;
    addReasons(reasonsByKey, result.added, 'support_envelope_post_bridge_enclosed_hole_fill');
  }

  // Keep only blankies that have enough observed live neighbours under the
  // configured IDW rule. A support envelope may be forgiving, but it must not
  // invent values in a region that is too far from measurements.
  const supportMask = new Set(liveMask);
  const rejectedCandidateKeys = [];

  for (const key of workingMask) {
    if (liveMask.has(key)) continue;
    if (hasEnoughLiveNeighbours(key, liveMask, grid, interpolationConfig)) {
      supportMask.add(key);
    } else {
      rejectedCandidateKeys.push(key);
    }
  }

  const candidates = [];
  for (let j = 0; j <= grid.maxJ; j += 1) {
    for (let i = 0; i <= grid.maxI; i += 1) {
      const key = cellKey(i, j);
      if (!supportMask.has(key) || liveMask.has(key)) continue;
      candidates.push({
        gridI: i,
        gridJ: j,
        selectionReasons: [...(reasonsByKey.get(key) ?? new Set(['support_envelope_accepted']))].sort(),
      });
    }
  }

  const liveComponents = summarizeComponents(liveMask, grid);
  const supportComponents = summarizeComponents(supportMask, grid);

  return {
    candidates,
    supportMask,
    summary: {
      mode: 'completed_support_envelope_v2_bounded_spans_component_bridging_v1',
      selectionBasis: 'live_measurements_only',
      rule: 'live_union_diamond_closing_then_bounded_orthogonal_span_fill_then_bounded_component_bridging_then_4_neighbour_hole_fill',
      closingKernel: 'manhattan_diamond_4_neighbour',
      closingRadiusCells,
      fillEnclosedHoles: fillHoles,
      selectedCellCount: candidates.length,
      closingSelectedCellCount: candidates.filter((candidate) => candidate.selectionReasons.includes('support_envelope_diamond_closing')).length,
      initialHoleFillSelectedCellCount: candidates.filter((candidate) => candidate.selectionReasons.includes('support_envelope_enclosed_hole_fill')).length,
      boundedSpanFill: {
        enabled: (selectionConfig.bounded_span_fill ?? {}).enabled !== false,
        maximumGapCells: spanResult.maximumGapCells,
        passesExecuted: spanResult.passes,
        perPassAddedCellCounts: spanResult.passCounts,
        addedCellCount: spanResult.added.size,
        rowAddedCellCount: spanResult.horizontalAdded.size,
        columnAddedCellCount: spanResult.verticalAdded.size,
      },
      componentBridging: {
        enabled: (selectionConfig.component_bridge ?? {}).enabled !== false,
        bridgeCount: bridgeResult.bridges.length,
        totalBridgeCellCount: bridgeResult.bridges.reduce((sum, bridge) => sum + bridge.corridor.length, 0),
        bridges: bridgeResult.bridges.map((bridge) => ({
          gapCells: bridge.gapCells,
          absorbedComponentCellCount: bridge.absorbedComponentCellCount,
        })),
      },
      postBridgeHoleFillSelectedCellCount: candidates.filter((candidate) => candidate.selectionReasons.includes('support_envelope_post_bridge_enclosed_hole_fill')).length,
      rejectedNoObservationSupportCellCount: rejectedCandidateKeys.length,
      diagonalRule: 'corner_touch_does_not_connect',
      liveComponents,
      supportComponents,
      outsideFill: false,
    },
  };
}


function computeIdwNeighbours({blankie, liveByKey, config}) {
  const {
    min_neighbours: minimumNeighbours = 2,
    max_radius_cells: maximumRadiusCells = 6,
    max_neighbours_used: maximumNeighboursUsed = 12,
    idw_power: idwPower = 2,
  } = config;

  let found = [];
  let selectedRadius = maximumRadiusCells;

  for (let radius = 1; radius <= maximumRadiusCells; radius += 1) {
    found = [];

    for (const live of liveByKey.values()) {
      const dx = live.gridI - blankie.gridI;
      const dy = live.gridJ - blankie.gridJ;
      if (Math.max(Math.abs(dx), Math.abs(dy)) > radius) continue;

      const distanceCells = Math.hypot(dx, dy);
      if (distanceCells <= 0) continue;
      found.push({live, distanceCells});
    }

    if (found.length >= minimumNeighbours) {
      selectedRadius = radius;
      break;
    }
  }

  found.sort((a, b) => a.distanceCells - b.distanceCells || a.live.rumIndex - b.live.rumIndex);
  const kept = found.slice(0, maximumNeighboursUsed);

  if (kept.length < minimumNeighbours) {
    throw new Error(
      `Blankie ${blankie.gridI}:${blankie.gridJ} has only ${kept.length} live neighbours ` +
      `inside ${maximumRadiusCells} cells; minimum is ${minimumNeighbours}.`,
    );
  }

  const rawWeights = kept.map((item) => 1 / (item.distanceCells ** idwPower));
  const weightSum = rawWeights.reduce((sum, value) => sum + value, 0);

  return {
    selectedRadius,
    idwPower,
    neighbours: kept.map((item, index) => ({
      rumId: item.live.rumId,
      rumIndex: item.live.rumIndex,
      gridI: item.live.gridI,
      gridJ: item.live.gridJ,
      distanceCells: item.distanceCells,
      normalizedWeight: rawWeights[index] / weightSum,
    })),
  };
}

function materializeInterpolatedBlankies({candidates, grid, sourceProj4, liveByKey, interpolationConfig}) {
  return candidates.map((candidate, blankIndex) => {
    const xCenter = grid.minX + candidate.gridI * grid.rumSizeM;
    const yCenter = grid.minY + candidate.gridJ * grid.rumSizeM;
    const {sourceCornersXY, footprintLonLat} = buildFootprintLonLat({
      x: xCenter,
      y: yCenter,
      rumSizeM: grid.rumSizeM,
      sourceProj4,
    });
    const [lon, lat] = proj4(sourceProj4, WGS84, [xCenter, yCenter]);
    const interpolation = computeIdwNeighbours({
      blankie: candidate,
      liveByKey,
      config: interpolationConfig,
    });

    const upMmYr = interpolation.neighbours.reduce(
      (sum, neighbour) => sum + liveByKey.get(cellKey(neighbour.gridI, neighbour.gridJ)).upMmYr * neighbour.normalizedWeight,
      0,
    );

    return {
      blankieId: `BLANK_i${String(candidate.gridI).padStart(4, '0')}_j${String(candidate.gridJ).padStart(4, '0')}`,
      blankIndex,
      runtimeRowIndex: null,
      gridI: candidate.gridI,
      gridJ: candidate.gridJ,
      lon,
      lat,
      xCenter,
      yCenter,
      sourceCornersXY,
      footprintLonLat,
      supportType: 'interpolated_support_no_measurement',
      valueStatus: 'idw_interpolated_model_with_inflated_sigma',
      selectionReasons: candidate.selectionReasons,
      upMmYrInterpolated: upMmYr,
      interpolation,
    };
  });
}

function deriveNorthwestAnchor(rums, grid) {
  const candidates = rums.filter(
    (rum) =>
      rum.gridI <= Math.floor(grid.maxI * 0.55) &&
      rum.gridJ >= Math.floor(grid.maxJ * 0.45),
  );
  const pool = candidates.length ? candidates : rums;
  return [...pool].sort((a, b) => a.upMmYr - b.upMmYr)[0];
}

function boundsFromCells(cells) {
  return {
    iMin: Math.min(...cells.map((cell) => cell.gridI)),
    iMax: Math.max(...cells.map((cell) => cell.gridI)),
    jMin: Math.min(...cells.map((cell) => cell.gridJ)),
    jMax: Math.max(...cells.map((cell) => cell.gridJ)),
  };
}

async function writeJson(filename, value) {
  await fs.writeFile(filename, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeFloat32(filename, values) {
  const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
  await fs.writeFile(filename, bytes);
}

// Deterministic synthetic-series helper. The V7.2 contract requires stable
// replayable measurement noise and rare sigma events; this generator keeps
// the DeckGL runtime reproducible from project_config.json.
function createSeededRandom(seed) {
  let state = (Number(seed) >>> 0) || 0x6d2b79f5;
  let spareGaussian = null;

  function uniform() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function gaussian() {
    if (spareGaussian !== null) {
      const value = spareGaussian;
      spareGaussian = null;
      return value;
    }

    let u = 0;
    let v = 0;
    while (u <= Number.EPSILON) u = uniform();
    while (v <= Number.EPSILON) v = uniform();

    const magnitude = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spareGaussian = magnitude * Math.sin(theta);
    return magnitude * Math.cos(theta);
  }

  return {uniform, gaussian};
}

function resolveSyntheticEpochSettings(config) {
  const settings = config.synthetic_epochs ?? {};
  const models = config.synthetic_epoch_models ?? {};
  const qualityPresets = config.uncertainty_quality_presets ?? {};
  const uncertaintyRelief = config.uncertainty_relief ?? {};

  const noiseMap = {low: 2.0, medium: 5.0, high: 8.0};
  const measurementNoiseName = String(settings.vertical_measurement_noise ?? 'medium');
  const uncertaintyQuality = String(settings.uncertainty_quality ?? 'high');
  const qcfg = qualityPresets[uncertaintyQuality];
  if (!qcfg) {
    throw new Error(`Unknown synthetic_epochs.uncertainty_quality: ${uncertaintyQuality}.`);
  }

  const sinusoidal = models.sinusoidal ?? {};
  const blankieSigma = uncertaintyRelief.blankie_sigma ?? {};
  const displayRange = uncertaintyRelief.display_range ?? {unit: 'sigma', value: 1};
  const visualFade = uncertaintyRelief.visual_fade ?? {};
  const displayUnit = String(displayRange.unit ?? 'sigma').toLowerCase();
  if (!['mm', 'sigma'].includes(displayUnit)) {
    throw new Error(`uncertainty_relief.display_range.unit must be "mm" or "sigma", got ${displayRange.unit}.`);
  }

  return {
    verticalModel: String(settings.vertical_model ?? 'linear'),
    measurementBehavior: String(settings.vertical_measurement_behavior ?? 'sinusoidal'),
    measurementNoiseName,
    measurementNoiseSigmaMm: Number(noiseMap[measurementNoiseName] ?? noiseMap.medium),
    uncertaintyQuality,
    randomSeed: Number(settings.random_seed ?? 6188575),
    sinusoidal: {
      amplitudeMm: Number(sinusoidal.amplitude_mm ?? 5.0),
      periodDays: Number(sinusoidal.period_days ?? 365.25),
      phaseDays: Number(sinusoidal.phase_days ?? 45.0),
    },
    quality: {
      baseSigmaMm: Number(qcfg.base_sigma_mm ?? 0.1),
      temporalGrowthMmPerYear: Number(qcfg.temporal_growth_mm_per_year ?? 0.2),
      seasonalSigmaMm: Number(qcfg.seasonal_sigma_mm ?? 0.5),
      spikeProbability: Number(qcfg.spike_probability ?? 0.005),
      spikeSigmaMm: Number(qcfg.spike_sigma_mm ?? 5.0),
    },
    sigmaFloorMm: Number(uncertaintyRelief.sigma_floor_mm ?? 0.05),
    displayRange: {unit: displayUnit, value: Number(displayRange.value ?? 1)},
    visualFade: {
      startEffectiveReliefM: Math.max(0, Number(visualFade.start_effective_relief_m ?? 0.5)),
      fullEffectiveReliefM: Math.max(0.500001, Number(visualFade.full_effective_relief_m ?? 10.0)),
      minimumRenderWeight: Math.max(0, Math.min(0.5, Number(visualFade.minimum_render_weight ?? 0.01))),
      bucketCount: Math.max(1, Math.min(16, Math.round(Number(visualFade.buckets ?? 8)))),
    },
    blankieSigma: {
      minimumMultiplierOverMaxNeighbour: Number(blankieSigma.minimum_multiplier_over_max_neighbour ?? 1.15),
      radiusPenaltyMmPerCell: Number(blankieSigma.radius_penalty_mm_per_cell ?? 0.15),
    },
  };
}

function seasonalValueMm(days, amplitudeMm, periodDays, phaseDays) {
  return amplitudeMm * Math.sin(2 * Math.PI * ((days - phaseDays) / periodDays));
}


// -----------------------------------------------------------------------------
// V7.2-compatible horizontal static glyph preparation.
//
// This remains data-only: values are derived once from direct live RUM inputs.
// The viewer renders three shared DeckGL meshes using these per-instance
// transforms. Blankies deliberately never receive a horizontal glyph.
// -----------------------------------------------------------------------------

function degrees(radiansValue) {
  return radiansValue * 180 / Math.PI;
}

function percentileSorted(values, percentileValue) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return 0;
  const fraction = clamp(Number(percentileValue) / 100, 0, 1);
  const index = (finite.length - 1) * fraction;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return finite[lo];
  const t = index - lo;
  return finite[lo] * (1 - t) + finite[hi] * t;
}

function computeGlobalPercentileRowMajor(values, rowCount, epochCount, percentileValue) {
  const safeRowCount = Math.max(0, Math.round(Number(rowCount) || 0));
  const safeEpochCount = Math.max(0, Math.round(Number(epochCount) || 0));
  const countExpected = safeRowCount * safeEpochCount;
  if (!countExpected) return 0;
  const finite = [];
  for (let index = 0; index < countExpected; index += 1) {
    const value = Number(values[index]);
    if (Number.isFinite(value)) finite.push(value);
  }
  return percentileSorted(finite, percentileValue);
}

function computePerEpochPercentileRowMajor(values, rowCount, epochCount, percentileValue) {
  const safeRowCount = Math.max(0, Math.round(Number(rowCount) || 0));
  const safeEpochCount = Math.max(0, Math.round(Number(epochCount) || 0));
  const output = new Array(safeEpochCount).fill(0);
  if (!safeRowCount || !safeEpochCount) return output;
  const scratch = new Array(safeRowCount);
  for (let epochIndex = 0; epochIndex < safeEpochCount; epochIndex += 1) {
    let count = 0;
    for (let rowIndex = 0; rowIndex < safeRowCount; rowIndex += 1) {
      const value = Number(values[rowIndex * safeEpochCount + epochIndex]);
      if (Number.isFinite(value)) {
        scratch[count] = value;
        count += 1;
      }
    }
    if (!count) {
      output[epochIndex] = 0;
      continue;
    }
    const sorted = scratch.slice(0, count).sort((a, b) => a - b);
    const fraction = clamp(Number(percentileValue) / 100, 0, 1);
    const index = (sorted.length - 1) * fraction;
    const lo = Math.floor(index);
    const hi = Math.ceil(index);
    if (lo === hi) {
      output[epochIndex] = sorted[lo];
      continue;
    }
    const t = index - lo;
    output[epochIndex] = sorted[lo] * (1 - t) + sorted[hi] * t;
  }
  return output;
}

function chi2Scale2d(probability) {
  const p = clamp(Number(probability), 1e-12, 0.999999999);
  return Math.sqrt(-2 * Math.log(1 - p));
}

function covarianceEigen2x2(varEast, varNorth, covarEN) {
  const a = Number(varEast);
  const c = Number(varNorth);
  const b = Number(covarEN);
  const traceHalf = 0.5 * (a + c);
  const diffHalf = 0.5 * (a - c);
  const root = Math.sqrt(Math.max(0, diffHalf * diffHalf + b * b));
  const lambdaMajor = Math.max(0, traceHalf + root);
  const lambdaMinor = Math.max(0, traceHalf - root);

  let vx;
  let vy;
  if (Math.abs(b) > 1e-18 || Math.abs(lambdaMajor - a) > 1e-18) {
    vx = b;
    vy = lambdaMajor - a;
  } else if (a >= c) {
    vx = 1;
    vy = 0;
  } else {
    vx = 0;
    vy = 1;
  }
  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm;
  vy /= norm;
  const angleMajorDegCcwFromEast = ((degrees(Math.atan2(vy, vx)) % 180) + 180) % 180;

  return {
    stdMajorMmYr: Math.sqrt(lambdaMajor),
    stdMinorMmYr: Math.sqrt(lambdaMinor),
    angleMajorDegCcwFromEast,
  };
}

function directionalSigmaThetaDeg({eastMmYr, northMmYr, varEast, varNorth, covarEN}, capDeg = 90) {
  const east = Number(eastMmYr);
  const north = Number(northMmYr);
  const speedMmYr = Math.hypot(east, north);
  if (!Number.isFinite(speedMmYr) || speedMmYr <= 1e-12) return NaN;

  // Small-angle directional uncertainty: sigma_theta ≈ sigma_perp / |v|.
  // The perpendicular unit vector keeps this tied to the direction actually
  // represented by the particle/glyph rather than an arbitrary E/N axis.
  const normalEast = -north / speedMmYr;
  const normalNorth = east / speedMmYr;
  const perpendicularVariance =
    normalEast * normalEast * Math.max(0, Number(varEast)) +
    2 * normalEast * normalNorth * Number(covarEN) +
    normalNorth * normalNorth * Math.max(0, Number(varNorth));
  const sigmaThetaDeg = degrees(Math.sqrt(Math.max(0, perpendicularVariance)) / speedMmYr);
  if (!Number.isFinite(sigmaThetaDeg)) return NaN;
  return Math.min(Math.max(0, Number(capDeg) || 90), sigmaThetaDeg);
}

function projectLngLat(sourceProj4, x, y) {
  return proj4(sourceProj4, WGS84, [x, y]);
}

function normalizeGlyphConfig(config, rumSizeM) {
  const source = config.horizontal_static_glyphs ?? {};
  const maxArrowLengthM = Number(source.arrow_max_length_m ?? rumSizeM * Number(source.arrow_max_length_rum_fraction ?? 0.8));
  const maxEllipseDiameterM = Number(source.ellipse_max_diameter_m ?? rumSizeM * Number(source.ellipse_max_diameter_rum_fraction ?? 0.75));
  return {
    enabled: source.enabled !== false,
    minimumSpeedMmYr: Number(source.minimum_speed_mm_yr ?? 0.02),
    visibilitySigmaMultiplier: Number(source.visibility_sigma_multiplier ?? 1.0),
    arrowSignificanceFilter: source.arrow_significance_filter !== false,
    ellipseMatchArrowFilter: source.ellipse_match_arrow_filter !== false,
    // Batch 1.21: Alex-compatible ellipse size. The original notebook sets
    // Matplotlib width/height to 2σ, which means the displayed semi-axes are
    // 1σ. Keep the old probability key out of the default path so regenerated
    // assets do not silently return to 95% joint ellipses.
    ellipseSigmaMultiplier: Math.max(0, Number(source.ellipse_sigma_multiplier ?? 1)),
    ellipseConfidenceProbability: 1 - Math.exp(-0.5 * Math.max(0, Number(source.ellipse_sigma_multiplier ?? 1)) ** 2),
    autoScale: source.auto_scale !== false,
    arrowReferencePercentile: Number(source.arrow_reference_percentile ?? 99.5),
    ellipseReferencePercentile: Number(source.ellipse_reference_percentile ?? 99.5),
    arrowMaxLengthM: maxArrowLengthM,
    ellipseMaxDiameterM: maxEllipseDiameterM,
    // Batch 1.20: follow the original 3D-Jakarta notebook contract.
    // The RUM coordinate is the arrow tail/vector origin; the confidence
    // ellipse is centred at the arrow tip. The old config key
    // arrow_anchor_fraction_at_rum_center is intentionally ignored here.
    arrowAnchorFraction: 0,
    ellipseScaleMode: String(source.ellipse_scale_mode ?? 'same_as_arrow').toLowerCase(),
    ellipseClipMode: String(source.ellipse_clip_mode ?? 'none').toLowerCase(),
    arrowShaftWidthFraction: Number(source.arrow_shaft_width_fraction ?? 0.045),
    arrowShaftWidthMinM: Number(source.arrow_shaft_width_min_m ?? rumSizeM * Number(source.arrow_shaft_width_min_rum_fraction ?? 0.010)),
    arrowShaftWidthMaxM: Number(source.arrow_shaft_width_max_m ?? rumSizeM * Number(source.arrow_shaft_width_max_rum_fraction ?? 0.040)),
    arrowheadFraction: Number(source.arrowhead_fraction ?? 0.22),
    arrowheadMinM: Number(source.arrowhead_min_m ?? rumSizeM * Number(source.arrowhead_min_rum_fraction ?? 0.060)),
    arrowheadMaxM: Number(source.arrowhead_max_m ?? rumSizeM * Number(source.arrowhead_max_rum_fraction ?? 0.250)),
    ellipseAxisMinM: Number(source.ellipse_axis_min_m ?? Math.max(1, rumSizeM * 0.004)),
    ellipseLineWidthM: Number(source.ellipse_line_width_m ?? rumSizeM * Number(source.ellipse_line_width_rum_fraction ?? 0.010)),
    ellipseSegments: Math.max(16, Math.round(Number(source.ellipse_segments ?? 64))),
    ellipseRingInnerRadius: clamp(Number(source.ellipse_ring_inner_radius ?? 0.94), 0.65, 0.995),
    clearanceAboveCapM: Math.max(0, Number(source.clearance_above_cap_m ?? 6)),
    showArrowsByDefault: source.show_arrows_by_default !== false,
    showEllipsesByDefault: source.show_ellipses_by_default !== false,
    defaultOpacity: clamp(Number(source.default_opacity ?? 0.92), 0, 1),
    arrowColorRgba: Array.isArray(source.arrow_color_rgba) ? source.arrow_color_rgba.slice(0, 4) : [34, 34, 34, 240],
    ellipseColorRgba: Array.isArray(source.ellipse_color_rgba) ? source.ellipse_color_rgba.slice(0, 4) : [0, 240, 216, 210],
  };
}

function buildHorizontalGlyphPayload({rums, sourceProj4, rumSizeM, config}) {
  const settings = normalizeGlyphConfig(config, rumSizeM);
  const confidenceScale = settings.ellipseSigmaMultiplier;
  const speeds = rums.map((rum) => Math.hypot(rum.eastMmYr, rum.northMmYr));
  const uncertaintyByRum = new Map();

  for (const rum of rums) {
    const eig = covarianceEigen2x2(rum.varEast, rum.varNorth, rum.covarEN);
    uncertaintyByRum.set(rum.rumId, eig);
  }

  const ellipseMajorValues = rums.map((rum) => {
    const eig = uncertaintyByRum.get(rum.rumId);
    return eig.stdMajorMmYr * confidenceScale;
  });
  const arrowSpeedRefMmYr = Math.max(0.05, percentileSorted(speeds.filter((value) => value > 0), settings.arrowReferencePercentile));
  const ellipseMajorRefMmYr = Math.max(0.05, percentileSorted(ellipseMajorValues.filter((value) => value > 0), settings.ellipseReferencePercentile));
  const arrowScaleMPerMmYr = settings.autoScale
    ? settings.arrowMaxLengthM / arrowSpeedRefMmYr
    : Number(config.horizontal_static_glyphs?.arrow_scale_m_per_mm_yr ?? 22.5);
  const ellipseScaleMPerMmYr = settings.autoScale && settings.ellipseScaleMode === 'same_as_arrow'
    ? arrowScaleMPerMmYr
    : settings.ellipseMaxDiameterM * 0.5 / ellipseMajorRefMmYr;

  const records = [];
  const visibleDirectionalSigmaThetaDeg = [];
  let skippedLowSpeed = 0;
  let skippedInsignificant = 0;
  let skippedMissingUncertainty = 0;

  for (const rum of rums) {
    const east = Number(rum.eastMmYr);
    const north = Number(rum.northMmYr);
    const speedMmYr = Math.hypot(east, north);
    const uncertainty = uncertaintyByRum.get(rum.rumId);
    if (speedMmYr < settings.minimumSpeedMmYr) {
      skippedLowSpeed += 1;
      continue;
    }
    if (!uncertainty || !Number.isFinite(uncertainty.stdMajorMmYr)) {
      skippedMissingUncertainty += 1;
      continue;
    }
    const significanceThreshold = settings.visibilitySigmaMultiplier * uncertainty.stdMajorMmYr;
    if (settings.arrowSignificanceFilter && speedMmYr < significanceThreshold) {
      skippedInsignificant += 1;
      continue;
    }

    const sigmaThetaDeg1Sigma = directionalSigmaThetaDeg({
      eastMmYr: east,
      northMmYr: north,
      varEast: rum.varEast,
      varNorth: rum.varNorth,
      covarEN: rum.covarEN,
    });
    if (Number.isFinite(sigmaThetaDeg1Sigma)) visibleDirectionalSigmaThetaDeg.push(sigmaThetaDeg1Sigma);

    const unitEast = speedMmYr > 0 ? east / speedMmYr : 0;
    const unitNorth = speedMmYr > 0 ? north / speedMmYr : 0;
    const azimuthDegCcwFromEast = degrees(Math.atan2(unitNorth, unitEast));
    const arrowLengthM = Math.min(speedMmYr * arrowScaleMPerMmYr, settings.arrowMaxLengthM);
    const headLengthM = Math.min(
      Math.max(arrowLengthM * settings.arrowheadFraction, settings.arrowheadMinM),
      settings.arrowheadMaxM,
      arrowLengthM * 0.65,
    );
    const shaftLengthM = Math.max(0, arrowLengthM - headLengthM);
    const shaftHalfWidthM = clamp(
      arrowLengthM * settings.arrowShaftWidthFraction,
      settings.arrowShaftWidthMinM,
      settings.arrowShaftWidthMaxM,
    );
    const headHalfWidthM = Math.max(shaftHalfWidthM * 2.4, headLengthM * 0.45);
    const tailDistanceM = 0;
    const headBaseDistanceM = shaftLengthM;
    const tipDistanceM = arrowLengthM;

    const arrowTailLonLat = projectLngLat(sourceProj4, rum.xCenter + unitEast * tailDistanceM, rum.yCenter + unitNorth * tailDistanceM);
    const arrowHeadBaseLonLat = projectLngLat(sourceProj4, rum.xCenter + unitEast * headBaseDistanceM, rum.yCenter + unitNorth * headBaseDistanceM);
    const arrowTipLonLat = projectLngLat(sourceProj4, rum.xCenter + unitEast * tipDistanceM, rum.yCenter + unitNorth * tipDistanceM);

    const ellipseMajorMmYr = uncertainty.stdMajorMmYr * confidenceScale;
    const ellipseMinorMmYr = uncertainty.stdMinorMmYr * confidenceScale;
    let ellipseMajorAxisM = Math.max(0, ellipseMajorMmYr * ellipseScaleMPerMmYr);
    let ellipseMinorAxisM = Math.max(0, ellipseMinorMmYr * ellipseScaleMPerMmYr);
    if (ellipseMajorAxisM > 0) ellipseMajorAxisM = Math.max(ellipseMajorAxisM, settings.ellipseAxisMinM);
    if (ellipseMinorAxisM > 0) ellipseMinorAxisM = Math.max(ellipseMinorAxisM, settings.ellipseAxisMinM);
    const ellipseAxisMaxM = settings.ellipseMaxDiameterM * 0.5;
    if (settings.ellipseClipMode === 'uniform' && ellipseAxisMaxM > 0 && ellipseMajorAxisM > ellipseAxisMaxM) {
      const factor = ellipseAxisMaxM / ellipseMajorAxisM;
      ellipseMajorAxisM *= factor;
      ellipseMinorAxisM *= factor;
    } else if (settings.ellipseClipMode === 'legacy_independent' && ellipseAxisMaxM > 0) {
      ellipseMajorAxisM = Math.min(ellipseMajorAxisM, ellipseAxisMaxM);
      ellipseMinorAxisM = Math.min(ellipseMinorAxisM, ellipseAxisMaxM);
    }

    records.push({
      rumId: rum.rumId,
      runtimeRowIndex: rum.rumIndex,
      gridI: rum.gridI,
      gridJ: rum.gridJ,
      speedMmYr,
      eastMmYr: east,
      northMmYr: north,
      unitEast,
      unitNorth,
      azimuthDegCcwFromEast,
      stdMajor1SigmaMmYr: uncertainty.stdMajorMmYr,
      stdMinor1SigmaMmYr: uncertainty.stdMinorMmYr,
      directionalSigmaThetaDeg1Sigma: sigmaThetaDeg1Sigma,
      speedOverStdMajor: speedMmYr / Math.max(uncertainty.stdMajorMmYr, 1e-12),
      arrow: {
        tailLonLat: arrowTailLonLat,
        headBaseLonLat: arrowHeadBaseLonLat,
        tipLonLat: arrowTipLonLat,
        yawDeg: azimuthDegCcwFromEast,
        lengthM: arrowLengthM,
        shaftLengthM,
        shaftHalfWidthM,
        headLengthM,
        headHalfWidthM,
        anchorFractionAtRumCenter: settings.arrowAnchorFraction,
        originPlacement: 'rum_coordinate_tail',
      },
      ellipse: {
        centerLonLat: arrowTipLonLat,
        yawDeg: uncertainty.angleMajorDegCcwFromEast,
        majorAxisM: ellipseMajorAxisM,
        minorAxisM: ellipseMinorAxisM,
        majorMmYr: ellipseMajorMmYr,
        minorMmYr: ellipseMinorMmYr,
        confidenceProbability: settings.ellipseConfidenceProbability,
        sigmaMultiplier: settings.ellipseSigmaMultiplier,
      },
    });
  }

  const glyphLegend = {
    statistic: 'P75',
    speedP75MmYr: percentileSorted(records.map((record) => record.speedMmYr).filter((value) => value > 0), 75),
    ellipseMajorP75MmYr: percentileSorted(records.map((record) => record.ellipse?.majorMmYr).filter((value) => value > 0), 75),
    ellipseMinorP75MmYr: percentileSorted(records.map((record) => record.ellipse?.minorMmYr).filter((value) => value > 0), 75),
    arrowReferenceMmYr: arrowSpeedRefMmYr,
    ellipseMajorReferenceMmYr: ellipseMajorRefMmYr,
    confidenceProbability: settings.ellipseConfidenceProbability,
    sigmaMultiplier: settings.ellipseSigmaMultiplier,
    label: '1σ major',
    visibleGlyphPairCount: records.length,
    directionalUncertainty: {
      sigmaThetaP75Deg: percentileSorted(visibleDirectionalSigmaThetaDeg, 75),
      sigmaMultiplier: 1,
      capDeg: 90,
      visibleCellCount: visibleDirectionalSigmaThetaDeg.length,
      rule: 'sigma_theta = sigma_perp / |v|; values capped at 90 degrees; computed only for glyph-visible RUMs.',
    },
  };

  return {
    schema: 'deckgl_proto1_horizontal_static_glyphs_v1',
    purpose: 'Notebook-faithful horizontal arrow / covariance-ellipse math prepared for native DeckGL instancing. Arrow tail is the RUM coordinate and the confidence ellipse is centred at the vector tip. Live RUMs only; no blankie horizontal interpolation.',
    units: {
      vector: 'mm/year',
      covariance: '(mm/year)^2',
      displayedGeometry: 'visual metres',
    },
    visibility: {
      minimumSpeedMmYr: settings.minimumSpeedMmYr,
      significanceSigmaMultiplier: settings.visibilitySigmaMultiplier,
      rule: 'speed >= minimumSpeed AND speed >= significanceSigmaMultiplier * stdMajor1Sigma',
    },
    ellipse: {
      confidenceProbability: settings.ellipseConfidenceProbability,
      confidenceScale: confidenceScale,
      sigmaMultiplier: settings.ellipseSigmaMultiplier,
      label: '1σ East-North uncertainty ellipse',
      centerPlacement: 'arrow_tip',
    },
    scaling: {
      rumSizeM,
      mode: settings.autoScale ? 'auto_percentile_by_rum_size_same_as_arrow' : 'manual_config',
      arrowReferencePercentile: settings.arrowReferencePercentile,
      arrowSpeedReferenceMmYr: arrowSpeedRefMmYr,
      arrowMaxLengthM: settings.arrowMaxLengthM,
      arrowScaleMPerMmYr,
      ellipseReferencePercentile: settings.ellipseReferencePercentile,
      ellipseMajorReferenceMmYr: ellipseMajorRefMmYr,
      ellipseScaleMPerMmYr,
      ellipseMaxDiameterM: settings.ellipseMaxDiameterM,
      ellipseClipMode: settings.ellipseClipMode,
      clearanceAboveCapM: settings.clearanceAboveCapM,
    },
    render: {
      enabled: settings.enabled,
      showArrowsByDefault: settings.showArrowsByDefault,
      showEllipsesByDefault: settings.showEllipsesByDefault,
      defaultOpacity: settings.defaultOpacity,
      arrowColorRgba: settings.arrowColorRgba,
      ellipseColorRgba: settings.ellipseColorRgba,
      ellipseSegments: settings.ellipseSegments,
      ellipseRingInnerRadius: settings.ellipseRingInnerRadius,
    },
    legend: glyphLegend,
    summary: {
      liveRumCount: rums.length,
      visibleGlyphPairCount: records.length,
      skippedLowSpeed,
      skippedInsignificantVsUncertainty: skippedInsignificant,
      skippedMissingUncertainty,
    },
    records,
  };
}


function buildRawParticleLegendMetadata({rums, horizontalGlyphPayload}) {
  const visibleRumIndices = new Set(
    (horizontalGlyphPayload?.records ?? []).map((record) => Number(record.runtimeRowIndex)),
  );
  const allPositiveSpeeds = rums
    .map((rum) => Math.hypot(rum.eastMmYr, rum.northMmYr))
    .filter((value) => Number.isFinite(value) && value > 0);
  const visibleDirectionalSigmaThetaDeg = rums
    .filter((rum) => visibleRumIndices.has(Number(rum.rumIndex)))
    .map((rum) => directionalSigmaThetaDeg(rum))
    .filter(Number.isFinite);

  return {
    statistic: 'P75',
    speedP75MmYr: percentileSorted(allPositiveSpeeds, 75),
    directionalUncertainty: {
      sigmaThetaP75Deg: percentileSorted(visibleDirectionalSigmaThetaDeg, 75),
      sigmaMultiplier: 1,
      capDeg: 90,
      visibleCellCount: visibleDirectionalSigmaThetaDeg.length,
      visibilityFilter: horizontalGlyphPayload?.visibility ?? null,
      rule: 'P75 sigma_theta across cells passing the horizontal glyph visibility filter; sigma_theta = sigma_perp / |v|; values capped at 90 degrees.',
    },
  };
}

async function buildLscParticleLegendMetadata({metadata, horizontalGlyphPayload}) {
  const width = Math.max(1, Math.round(Number(metadata?.grid?.width ?? 1)));
  const height = Math.max(1, Math.round(Number(metadata?.grid?.height ?? 1)));
  const expectedValues = width * height * 4;
  const fieldPath = path.join(OUTPUT_DIR, metadata.assets?.fieldF32 ?? 'horizontal_particle_lsc_field_rgba_f32.bin');
  const covariancePath = path.join(OUTPUT_DIR, metadata.assets?.covarianceF32 ?? 'horizontal_particle_lsc_covariance_rgba_f32.bin');
  const [fieldBuffer, covarianceBuffer] = await Promise.all([
    fs.readFile(fieldPath),
    fs.readFile(covariancePath),
  ]);
  const field = new Float32Array(fieldBuffer.buffer.slice(fieldBuffer.byteOffset, fieldBuffer.byteOffset + fieldBuffer.byteLength));
  const covariance = new Float32Array(covarianceBuffer.buffer.slice(covarianceBuffer.byteOffset, covarianceBuffer.byteOffset + covarianceBuffer.byteLength));
  if (field.length !== expectedValues || covariance.length !== expectedValues) {
    throw new Error('LSC field/covariance arrays do not match the LSC grid while building horizontal legend metadata.');
  }

  const visibleRumIndices = new Set(
    (horizontalGlyphPayload?.records ?? []).map((record) => Number(record.runtimeRowIndex)),
  );
  const speeds = [];
  const visibleDirectionalSigmaThetaDeg = [];
  let validFineTexelCount = 0;
  let visibleFineTexelCount = 0;
  for (let offset = 0; offset < field.length; offset += 4) {
    if (!(field[offset + 3] > 0.5)) continue;
    validFineTexelCount += 1;
    const eastMmYr = Number(field[offset]);
    const northMmYr = Number(field[offset + 1]);
    const speedMmYr = Math.hypot(eastMmYr, northMmYr);
    if (Number.isFinite(speedMmYr) && speedMmYr > 0) speeds.push(speedMmYr);

    const parentRumIndex = Math.round(Number(field[offset + 2]));
    if (!visibleRumIndices.has(parentRumIndex)) continue;
    const sigmaThetaDeg = directionalSigmaThetaDeg({
      eastMmYr,
      northMmYr,
      varEast: covariance[offset],
      varNorth: covariance[offset + 1],
      covarEN: covariance[offset + 2],
    });
    if (Number.isFinite(sigmaThetaDeg)) {
      visibleFineTexelCount += 1;
      visibleDirectionalSigmaThetaDeg.push(sigmaThetaDeg);
    }
  }

  return {
    statistic: 'P75',
    speedP75MmYr: percentileSorted(speeds, 75),
    directionalUncertainty: {
      sigmaThetaP75Deg: percentileSorted(visibleDirectionalSigmaThetaDeg, 75),
      sigmaMultiplier: 1,
      capDeg: 90,
      visibleCellCount: visibleFineTexelCount,
      validFineTexelCount,
      parentVisibilityFilter: horizontalGlyphPayload?.visibility ?? null,
      rule: 'P75 sigma_theta across valid fine LSC texels whose parent observed RUM passes the horizontal glyph visibility filter; sigma_theta = sigma_perp / |v|; E/N covariance is zero because the LSC runtime models component prediction variances only; values capped at 90 degrees.',
    },
  };
}

// -----------------------------------------------------------------------------
// GPU particle runtime preparation.
//
// This is intentionally separate from the static glyph payload. Glyphs are
// sparse, directly inspected expert geometry. Particles are a dense animated
// field: a compact regular-grid texture plus a live-RUM-only spawn domain.
// Blankies never enter either source.
// -----------------------------------------------------------------------------

function normalizeParticleConfig(config, rumSizeM) {
  const source = config.horizontal_particles ?? {};
  const capacity = Math.max(256, Math.round(Number(source.particle_capacity ?? 8000)));
  const defaultCount = clamp(
    Math.round(Number(source.default_particle_count ?? 5000)),
    0,
    capacity,
  );
  const defaultMode = String(source.default_mode ?? 'mean').toLowerCase();
  if (!['mean', 'montecarlo', 'shimmer'].includes(defaultMode)) {
    throw new Error(`horizontal_particles.default_mode must be mean, montecarlo, or shimmer; got ${defaultMode}.`);
  }
  const spawnSupportRule = String(source.spawn_support_rule ?? 'eight_live_neighbours').toLowerCase();
  if (spawnSupportRule !== 'eight_live_neighbours') {
    throw new Error(`horizontal_particles.spawn_support_rule must be eight_live_neighbours; got ${spawnSupportRule}.`);
  }

  const historySampleIntervalS = clamp(Number(source.history_sample_interval_s ?? 0.05), 1 / 120, 0.25);
  const historySamplesMin = clamp(Math.round(Number(source.history_samples_min ?? 9)), 2, 65);
  const historySamplesMax = clamp(Math.round(Number(source.history_samples_max ?? 65)), historySamplesMin, 65);
  const historySamples = clamp(Math.round(Number(source.history_samples ?? 32)), historySamplesMin, historySamplesMax);

  return {
    enabled: source.enabled !== false,
    showByDefault: source.show_by_default !== false,
    defaultMode,
    particleCapacity: capacity,
    defaultParticleCount: defaultCount,
    baseMps: Math.max(0, Number(source.base_mps ?? 1800)),
    speedMultiplier: clamp(Number(source.speed_multiplier ?? 1.5), 0, 10),
    surfaceOffsetM: Math.max(0, Number(source.surface_offset_m ?? 20)),
    stallSpeedMmYr: Math.max(0, Number(source.stall_speed_mm_yr ?? 0.05)),
    samplerMode: String(source.sampler_mode ?? 'conservative_v1').toLowerCase(),
    particleSizePixels: clamp(Number(source.particle_size_pixels ?? 2.2), 0.5, 16),
    particleSizeMultiplier: clamp(Number(source.particle_size_multiplier ?? 1.0), 0.1, 8),
    particleSizeMultiplierMin: clamp(Number(source.particle_size_multiplier_min ?? 0.5), 0.1, 8),
    particleSizeMultiplierMax: clamp(Number(source.particle_size_multiplier_max ?? 3.0), 0.1, 8),
    particleSizeMultiplierStep: clamp(Number(source.particle_size_multiplier_step ?? 0.1), 0.01, 1),
    particleOpacity: clamp(Number(source.particle_opacity ?? 1.0), 0, 1),
    trailWidthPixels: clamp(Number(source.trail_width_pixels ?? 1.15), 0.25, 12),
    trailOpacity: clamp(Number(source.trail_opacity ?? 1.0), 0, 1),
    historySampleIntervalS,
    historySamplesMin,
    historySamplesMax,
    historySamples,
    trailDurationStepS: clamp(Number(source.trail_duration_step_s ?? 0.05), 1 / 120, 0.25),
    trailPersistence: clamp(Number(source.trail_persistence ?? 0.98), 0.80, 0.999),
    trailPersistenceMin: clamp(Number(source.trail_persistence_min ?? 0.80), 0.50, 0.999),
    trailPersistenceMax: clamp(Number(source.trail_persistence_max ?? 0.999), 0.50, 0.999),
    trailPersistenceStep: clamp(Number(source.trail_persistence_step ?? 0.001), 0.001, 0.05),
    maxTrailScreenJumpPx: clamp(Number(source.max_trail_screen_jump_px ?? 120), 8, 1000),
    integrationMaxCellFraction: clamp(Number(source.integration_max_cell_fraction ?? 0.25), 0.05, 1.0),
    integrationMaxSubsteps: 24,
    tailFadeMode: 'canvas_persistence',
    birthFadeSeconds: clamp(Number(source.birth_fade_seconds ?? 0.3), 0.05, 2.0),
    spawnSupportRule,
    uncertaintyStrength: clamp(Number(source.uncertainty_strength ?? 0.5), 0, 2),
    shimmerStrength: clamp(Number(source.shimmer_strength ?? source.uncertainty_strength ?? 0.5), 0, 2),
    monteCarloStrength: clamp(Number(source.monte_carlo_strength ?? source.uncertainty_strength ?? 0.5), 0, 2),
    mcModel: String(source.mc_model ?? 'directional').toLowerCase() === 'full' ? 'full' : 'directional',
    mcMaxSigma: Math.max(0.1, Number(source.mc_max_sigma ?? 1.5)),
    mcOffsetCapMmYr: Math.max(0, Number(source.mc_offset_cap_mm_yr ?? 1.0)),
    mcOffsetCapRatioToSpeed: Math.max(0, Number(source.mc_offset_cap_ratio_to_speed ?? 1.0)),
    shimmerPixelAmplitude: Math.max(0, Number(source.shimmer_pixel_amplitude ?? 5.0)),
    spawnJitterCells: clamp(Number(source.spawn_jitter_cells ?? 0.90), 0, 16),
    colorRgba: Array.isArray(source.color_rgba) ? source.color_rgba.slice(0, 4) : [100, 100, 100, 235],
    rumSizeM,
  };
}

function meanPoint(points) {
  if (!points.length) return [0, 0];
  return [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
}

function localMetersFromLonLat(lon, lat, originLon, originLat) {
  const earthRadiusM = 6378137;
  const degToRad = Math.PI / 180;
  return [
    earthRadiusM * (lon - originLon) * degToRad * Math.cos(originLat * degToRad),
    earthRadiusM * (lat - originLat) * degToRad,
  ];
}

function deriveParticleMetricGrid(rums) {
  const originLon = rums.reduce((sum, rum) => sum + rum.lon, 0) / rums.length;
  const originLat = rums.reduce((sum, rum) => sum + rum.lat, 0) / rums.length;
  const localByKey = new Map();

  for (const rum of rums) {
    localByKey.set(cellKey(rum.gridI, rum.gridJ), localMetersFromLonLat(rum.lon, rum.lat, originLon, originLat));
  }

  const iSteps = [];
  const jSteps = [];
  for (const rum of rums) {
    const here = localByKey.get(cellKey(rum.gridI, rum.gridJ));
    const iNext = localByKey.get(cellKey(rum.gridI + 1, rum.gridJ));
    const jNext = localByKey.get(cellKey(rum.gridI, rum.gridJ + 1));
    if (iNext) iSteps.push([iNext[0] - here[0], iNext[1] - here[1]]);
    if (jNext) jSteps.push([jNext[0] - here[0], jNext[1] - here[1]]);
  }

  const axisIM = meanPoint(iSteps);
  const axisJM = meanPoint(jSteps);
  const origins = rums.map((rum) => {
    const local = localByKey.get(cellKey(rum.gridI, rum.gridJ));
    return [
      local[0] - rum.gridI * axisIM[0] - rum.gridJ * axisJM[0],
      local[1] - rum.gridI * axisIM[1] - rum.gridJ * axisJM[1],
    ];
  });

  return {
    coordinateOriginLonLat: [originLon, originLat],
    gridOriginLocalM: meanPoint(origins),
    gridAxisIM: axisIM,
    gridAxisJM: axisJM,
  };
}

function buildHorizontalParticlePayload({rums, grid, rumSizeM, config, legend}) {
  const settings = normalizeParticleConfig(config, rumSizeM);
  const width = grid.maxI + 1;
  const height = grid.maxJ + 1;
  const field = new Float32Array(width * height * 4);
  const covariance = new Float32Array(width * height * 4);
  const speeds = [];
  const spawnable = [];
  const liveByKey = new Map(rums.map((rum) => [cellKey(rum.gridI, rum.gridJ), rum]));

  for (const rum of rums) {
    const offset = (rum.gridJ * width + rum.gridI) * 4;
    const speedMmYr = Math.hypot(rum.eastMmYr, rum.northMmYr);
    field[offset] = rum.eastMmYr;
    field[offset + 1] = rum.northMmYr;
    field[offset + 2] = rum.rumIndex;
    field[offset + 3] = 1;
    covariance[offset] = Math.max(0, rum.varEast);
    covariance[offset + 1] = Math.max(0, rum.varNorth);
    covariance[offset + 2] = rum.covarEN;
    covariance[offset + 3] = speedMmYr;
    speeds.push(speedMmYr);
    if (speedMmYr > settings.stallSpeedMmYr) spawnable.push(rum);
  }

  const supportedSpawnable = spawnable.filter((rum) => {
    // Restore the V7 supported-emitter contract: seed only where every direct
    // neighbour is an observed RUM. This does not create/smooth data; it keeps
    // births away from fragmented support where trajectories would immediately die.
    for (let dj = -1; dj <= 1; dj += 1) {
      for (let di = -1; di <= 1; di += 1) {
        if (di === 0 && dj === 0) continue;
        if (!liveByKey.has(cellKey(rum.gridI + di, rum.gridJ + dj))) return false;
      }
    }
    return true;
  });
  const spawnFallbackUsed = supportedSpawnable.length === 0;
  const spawnRecords = spawnFallbackUsed ? spawnable : supportedSpawnable;
  const safeSpawnRecords = spawnRecords.length ? spawnRecords : rums;
  const spawns = new Float32Array(safeSpawnRecords.length * 2);
  for (let index = 0; index < safeSpawnRecords.length; index += 1) {
    spawns[index * 2] = safeSpawnRecords[index].gridI;
    spawns[index * 2 + 1] = safeSpawnRecords[index].gridJ;
  }

  const metricGrid = deriveParticleMetricGrid(rums);
  const speedP95MmYr = Math.max(1e-9, percentileSorted(speeds.filter((value) => value > 0), 95));

  return {
    metadata: {
      schema: 'deckgl_proto1_horizontal_gpu_particle_field_v6_long_history_safety',
      purpose: 'GPU transform-feedback particle field for directly observed Jakarta RUM horizontal velocity and E/N covariance. Blankies are excluded from the field and spawn domain; emitter cells require observed 8-neighbour support. The runtime renders fixed-cadence RGBA32F world-space history ribbons using V7-style exponential brush persistence and speed-modulated rounded caps. History supports user-selected 0.40–3.20 s durations without reseeding particles; high-speed motion uses support-checked metric substeps and a V7-style screen-jump guard.',
      units: {
        horizontalVelocity: 'mm/year',
        covariance: '(mm/year)^2',
        localParticleCoordinates: 'metres east/north relative to coordinateOriginLonLat',
      },
      sampling: {
        mode: settings.samplerMode,
        rule: 'conservative_v1: bilinear only when all four adjacent observed RUMs are available; otherwise nearest observed grid cell. Motion never interpolates blankies. Lifecycle stall checks use the nearest observed RUM speed, not a bilinear blend.',
      },
      grid: {
        width,
        height,
        maxI: grid.maxI,
        maxJ: grid.maxJ,
        rumSizeM,
        ...metricGrid,
      },
      spawnDomain: {
        rule: 'live RUM + speed above stall threshold + all eight direct neighbours observed',
        supportNeighbourCount: 8,
        fallback: 'all above-stall live RUMs only when no supported emitter cells exist',
        fallbackUsed: spawnFallbackUsed,
      },
      render: settings,
      legend,
      history: {
        storage: 'RGBA32F texture: local_x_m, local_y_m, particle_age_s, speed_ratio_to_p95 (uncapped; visual alpha/width outputs clamp)',
        samples: settings.historySamples,
        samplesMin: settings.historySamplesMin,
        samplesMax: settings.historySamplesMax,
        sampleIntervalS: settings.historySampleIntervalS,
        trailDurationMinS: (settings.historySamplesMin - 1) * settings.historySampleIntervalS,
        trailDurationMaxS: (settings.historySamplesMax - 1) * settings.historySampleIntervalS,
        trailDurationStepS: settings.trailDurationStepS,
        tailFadeMode: settings.tailFadeMode,
        trailPersistence: settings.trailPersistence,
        durationS: (settings.historySamples - 1) * settings.historySampleIntervalS,
        maxTrailScreenJumpPx: settings.maxTrailScreenJumpPx,
        integrationMaxCellFraction: settings.integrationMaxCellFraction,
        integrationMaxSubsteps: settings.integrationMaxSubsteps,
        rationale: 'Fixed simulation-time cadence makes visible trail duration independent of display frame rate. Trail alpha uses V7-style exponential canvas persistence; age resets mark respawn discontinuities; Z is recomputed from current vertical model at draw time. Requested duration reallocates only the sentinel-filled history target and never reseeds particle simulation state. At high visual speed, metric substeps validate support across the path and the 120 px screen-jump guard suppresses any remaining misleading cross-map connector.',
        atlas: {
          layout: 'runtime_tiled_particle_row_atlas',
          preferredWidth: 4096,
          rationale: 'Runtime tiles particle columns into multiple rows before multiplying by history samples, avoiding a 12K-wide RGBA32F render target while keeping texelFetch history reads exact.',
        },
      },
      summary: {
        liveRumCount: rums.length,
        spawnCellCount: safeSpawnRecords.length,
        speedP95MmYr,
        excludedStalledRums: rums.length - spawnable.length,
        excludedForMissingEightNeighbourSupport: spawnable.length - supportedSpawnable.length,
        spawnFallbackUsed,
      },
      assets: {
        fieldF32: 'horizontal_particle_field_rgba_f32.bin',
        covarianceF32: 'horizontal_particle_covariance_rgba_f32.bin',
        spawnGridF32: 'horizontal_particle_spawns_rg_f32.bin',
      },
    },
    field,
    covariance,
    spawns,
  };
}

async function main() {
  const config = JSON.parse(await fs.readFile(CONFIG_PATH, 'utf8'));
  const sourceCsvRelative = config.input.source_csv;
  const inputCsv = path.join(ROOT, sourceCsvRelative);

  try {
    await fs.access(inputCsv);
  } catch {
    throw new Error(
      `Missing source CSV: ${sourceCsvRelative}\n` +
      'Copy jakarta_enu_estimates.csv into the data folder, then run again.',
    );
  }

  const rows = parseCsv(await fs.readFile(inputCsv, 'utf8'));
  requireColumns(rows, [
    'x_rum', 'y_rum', 'east', 'north', 'up',
    'var_east', 'var_north', 'var_up',
    'covar_en', 'covar_eu', 'covar_nu',
  ]);

  const rumSizeM = requireFinite(Number(config.input.rum_size_m), 'input.rum_size_m');
  const sourceProj4 = config.input.source_proj4;
  const xValues = sortedUnique(rows.map((row) => requireFinite(Number(row.x_rum), 'x_rum')));
  const yValues = sortedUnique(rows.map((row) => requireFinite(Number(row.y_rum), 'y_rum')));
  const minX = xValues[0];
  const minY = yValues[0];

  const rums = rows.map((row, rumIndex) => {
    const x = requireFinite(Number(row.x_rum), `x_rum row ${rumIndex + 2}`);
    const y = requireFinite(Number(row.y_rum), `y_rum row ${rumIndex + 2}`);
    const {sourceCornersXY, footprintLonLat} = buildFootprintLonLat({
      x,
      y,
      rumSizeM,
      sourceProj4,
    });
    const [lon, lat] = proj4(sourceProj4, WGS84, [x, y]);

    return {
      rumId: `RUM_${String(rumIndex).padStart(5, '0')}`,
      rumIndex,
      lon,
      lat,
      xCenter: x,
      yCenter: y,
      gridI: Math.round((x - minX) / rumSizeM),
      gridJ: Math.round((y - minY) / rumSizeM),
      sourceCornersXY,
      footprintLonLat,
      eastMmYr: requireFinite(Number(row.east), `east row ${rumIndex + 2}`),
      northMmYr: requireFinite(Number(row.north), `north row ${rumIndex + 2}`),
      upMmYr: requireFinite(Number(row.up), `up row ${rumIndex + 2}`),
      varEast: requireFinite(Number(row.var_east), `var_east row ${rumIndex + 2}`),
      varNorth: requireFinite(Number(row.var_north), `var_north row ${rumIndex + 2}`),
      varUp: requireFinite(Number(row.var_up), `var_up row ${rumIndex + 2}`),
      covarEN: requireFinite(Number(row.covar_en), `covar_en row ${rumIndex + 2}`),
      covarEU: requireFinite(Number(row.covar_eu), `covar_eu row ${rumIndex + 2}`),
      covarNU: requireFinite(Number(row.covar_nu), `covar_nu row ${rumIndex + 2}`),
    };
  });

  const grid = {
    minX,
    minY,
    maxI: Math.max(...rums.map((rum) => rum.gridI)),
    maxJ: Math.max(...rums.map((rum) => rum.gridJ)),
    rumSizeM,
    sourceProj4,
  };

  const horizontalGlyphPayload = buildHorizontalGlyphPayload({
    rums,
    sourceProj4,
    rumSizeM,
    config,
  });

  const rawParticleLegend = buildRawParticleLegendMetadata({
    rums,
    horizontalGlyphPayload,
  });
  const horizontalParticleBuild = buildHorizontalParticlePayload({
    rums,
    grid,
    rumSizeM,
    config,
    legend: rawParticleLegend,
  });

  const liveByKey = new Map(rums.map((rum) => [cellKey(rum.gridI, rum.gridJ), rum]));
  const selection = selectSupportEnvelopeBlankies({
    liveRums: rums,
    grid,
    selectionConfig: config.viewer.blankies?.selection ?? {},
    interpolationConfig: config.viewer.blankies?.interpolation ?? {},
  });
  const interpolationConfig = config.viewer.blankies?.interpolation ?? {};
  const blankies = materializeInterpolatedBlankies({
    candidates: selection.candidates,
    grid,
    sourceProj4,
    liveByKey,
    interpolationConfig,
  });

  for (const blankie of blankies) {
    blankie.runtimeRowIndex = rums.length + blankie.blankIndex;
  }

  const {epochs, yearsSinceStart} = buildEpochs(config.time_settings);
  const rumCount = rums.length;
  const blankCount = blankies.length;
  const runtimeRowCount = rumCount + blankCount;
  const epochCount = epochs.length;
  const synthetic = resolveSyntheticEpochSettings(config);
  const runtimeModelMm = new Float32Array(runtimeRowCount * epochCount);
  const runtimeMeasurementMm = new Float32Array(runtimeRowCount * epochCount);
  const runtimeSigmaMm = new Float32Array(runtimeRowCount * epochCount);
  const rng = createSeededRandom(synthetic.randomSeed);

  for (const rum of rums) {
    const offset = rum.rumIndex * epochCount;
    const sigmaRateMmPerYear = Math.sqrt(Math.max(0, rum.varUp));

    for (let epochIndex = 0; epochIndex < epochCount; epochIndex += 1) {
      const years = yearsSinceStart[epochIndex];
      const days = years * 365.25;
      const linearMm = rum.upMmYr * years;
      const seasonalMm = seasonalValueMm(
        days,
        synthetic.sinusoidal.amplitudeMm,
        synthetic.sinusoidal.periodDays,
        synthetic.sinusoidal.phaseDays,
      );
      const modelMm = synthetic.verticalModel === 'sinusoidal'
        ? linearMm + seasonalMm
        : linearMm;
      const measurementMm = linearMm +
        (synthetic.measurementBehavior === 'sinusoidal' ? seasonalMm : 0) +
        rng.gaussian() * synthetic.measurementNoiseSigmaMm;

      let sigmaMm =
        synthetic.quality.baseSigmaMm +
        sigmaRateMmPerYear * years +
        synthetic.quality.temporalGrowthMmPerYear * years +
        Math.abs(seasonalValueMm(
          days,
          synthetic.quality.seasonalSigmaMm,
          synthetic.sinusoidal.periodDays,
          synthetic.sinusoidal.phaseDays,
        ));
      if (rng.uniform() < synthetic.quality.spikeProbability) {
        sigmaMm += synthetic.quality.spikeSigmaMm;
      }

      runtimeModelMm[offset + epochIndex] = modelMm;
      runtimeMeasurementMm[offset + epochIndex] = measurementMm;
      runtimeSigmaMm[offset + epochIndex] = Math.max(synthetic.sigmaFloorMm, sigmaMm);
    }
  }

  for (const blankie of blankies) {
    const offset = blankie.runtimeRowIndex * epochCount;
    const selectedRadius = Number(blankie.interpolation.selectedRadius ?? 1);

    for (let epochIndex = 0; epochIndex < epochCount; epochIndex += 1) {
      let modelMm = 0;
      let weightedSigmaMm = 0;
      let maximumNeighbourSigmaMm = 0;

      for (const neighbour of blankie.interpolation.neighbours) {
        const neighbourIndex = neighbour.rumIndex * epochCount + epochIndex;
        modelMm += runtimeModelMm[neighbourIndex] * neighbour.normalizedWeight;
        const neighbourSigmaMm = runtimeSigmaMm[neighbourIndex];
        weightedSigmaMm += neighbourSigmaMm * neighbour.normalizedWeight;
        maximumNeighbourSigmaMm = Math.max(maximumNeighbourSigmaMm, neighbourSigmaMm);
      }

      const inflatedSigmaMm = Math.max(
        weightedSigmaMm,
        maximumNeighbourSigmaMm * synthetic.blankieSigma.minimumMultiplierOverMaxNeighbour,
      ) + Math.max(0, selectedRadius - 1) * synthetic.blankieSigma.radiusPenaltyMmPerCell;

      runtimeModelMm[offset + epochIndex] = modelMm;
      // There is no direct observation for a support blankie. Keep the output
      // usable for a shared array contract, but label it as model-only in the UI.
      runtimeMeasurementMm[offset + epochIndex] = modelMm;
      runtimeSigmaMm[offset + epochIndex] = Math.max(synthetic.sigmaFloorMm, inflatedSigmaMm);
    }
  }

  const verticalVelocityColorScale = buildAdaptiveVerticalVelocityColorScale(
    rums,
    config.style?.vertical_velocity_color_scale ?? {},
  );

  const gridCellCount = (grid.maxI + 1) * (grid.maxJ + 1);
  const structuralCellCount = rumCount + blankCount;
  const unfilledNoDataCellCount = gridCellCount - structuralCellCount;
  const supportBounds = boundsFromCells([...rums, ...blankies]);
  const anchor = deriveNorthwestAnchor(rums, grid);

  await fs.mkdir(OUTPUT_DIR, {recursive: true});

  await writeJson(path.join(OUTPUT_DIR, 'rum_static.json'), {
    schema: 'deckgl_proto1_rum_static_v5',
    source: sourceCsvRelative,
    sourceCrs: config.input.source_crs,
    sourceProj4,
    targetCrs: 'EPSG:4326',
    rumSizeM,
    grid,
    rumCount,
    rums,
  });

  await writeJson(path.join(OUTPUT_DIR, 'horizontal_glyphs.json'), horizontalGlyphPayload);

  await writeJson(
    path.join(OUTPUT_DIR, 'horizontal_particle_field.json'),
    horizontalParticleBuild.metadata,
  );
  await writeFloat32(
    path.join(OUTPUT_DIR, horizontalParticleBuild.metadata.assets.fieldF32),
    horizontalParticleBuild.field,
  );
  await writeFloat32(
    path.join(OUTPUT_DIR, horizontalParticleBuild.metadata.assets.covarianceF32),
    horizontalParticleBuild.covariance,
  );
  await writeFloat32(
    path.join(OUTPUT_DIR, horizontalParticleBuild.metadata.assets.spawnGridF32),
    horizontalParticleBuild.spawns,
  );

  // LSC is an optional second signal-field asset. It never participates in the
  // raw build and a failed/degenerate LSC fit only removes its own optional
  // artifacts; the raw payload remains the runnable baseline.
  let horizontalParticleLscBuild = {available: false, reason: 'not-run'};
  try {
    horizontalParticleLscBuild = await buildLscField({
      root: ROOT,
      outputDir: OUTPUT_DIR,
      config,
      force: process.argv.includes('--force-lsc'),
      logger: console,
    });
  } catch (error) {
    await pruneLscAssets(OUTPUT_DIR);
    console.warn(`[LSC] optional signal-field bake skipped; raw particle assets remain intact. ${error.message ?? error}`);
    horizontalParticleLscBuild = {available: false, reason: error.message ?? String(error)};
  }

  if (horizontalParticleLscBuild.available && horizontalParticleLscBuild.metadata) {
    const lscLegend = await buildLscParticleLegendMetadata({
      metadata: horizontalParticleLscBuild.metadata,
      horizontalGlyphPayload,
    });
    horizontalParticleLscBuild.metadata.legend = lscLegend;
    await writeJson(
      path.join(OUTPUT_DIR, horizontalParticleLscBuild.assetName ?? 'horizontal_particle_field_lsc.json'),
      horizontalParticleLscBuild.metadata,
    );
  }

  await writeJson(path.join(OUTPUT_DIR, 'interpolated_blankies.json'), {
    schema: 'deckgl_proto1_support_envelope_interpolated_blankies_v2',
    meaning:
      'No direct InSAR measurement. Support-envelope cell selected by bounded 4-neighbour diamond closing and enclosed-hole fill; motion is IDW-interpolated from neighbouring live RUM model series.',
    selection: selection.summary,
    interpolation: {
      method: 'idw_grid_neighbours',
      ...interpolationConfig,
      sourceRows: 'live RUMs only',
      weightsComputedOnce: true,
    },
    blankCount,
    blankies,
  });

  await writeJson(path.join(OUTPUT_DIR, 'epoch_axis.json'), {
    schema: 'deckgl_proto1_epoch_axis_v1',
    ...config.time_settings,
    epochCount,
    epochs,
    yearsSinceStart,
  });

  await writeFloat32(path.join(OUTPUT_DIR, 'vertical_model_mm_f32.bin'), runtimeModelMm);
  await writeFloat32(path.join(OUTPUT_DIR, 'vertical_measurement_mm_f32.bin'), runtimeMeasurementMm);
  await writeFloat32(path.join(OUTPUT_DIR, 'vertical_sigma_mm_f32.bin'), runtimeSigmaMm);

  const verticalUncertaintyLegend = {
    statistic: 'P75',
    unit: 'mm',
    anchorScope: 'all_live_rums_all_epochs',
    liveCellCount: rumCount,
    sampleCount: rumCount * epochCount,
    globalP75Mm: computeGlobalPercentileRowMajor(runtimeSigmaMm, rumCount, epochCount, 75),
    sigma_provenance: 'synthetic_demo',
    note: 'Static reference anchor for the vertical uncertainty relief legend. It is the P75 of per-epoch sigma_z values over measured RUMs and all epochs; blank support cells are excluded.',
  };

  await writeJson(path.join(OUTPUT_DIR, 'manifest.json'), {
    schema: 'deckgl_proto1_runtime_manifest_v6',
    projectName: config.project_name,
    rumCount,
    blankCount,
    runtimeRowCount,
    structuralCellCount,
    unfilledNoDataCellCount,
    gridCellCount,
    epochCount,
    rumSizeM,
    arrayOrder: 'runtime-row-major, epoch-minor',
    indexFormula: 'valueIndex = runtimeRowIndex * epochCount + epochIndex',
    verticalModel: {
      name: `${synthetic.verticalModel}_from_up_velocity_with_support_envelope_v2_idw_blankies`,
      unit: 'mm',
      liveSourceField: 'up',
      blankieSource: 'IDW of live model series',
    },
    verticalMeasurement: {
      behavior: synthetic.measurementBehavior,
      noiseLevel: synthetic.measurementNoiseName,
      noiseSigmaMm: synthetic.measurementNoiseSigmaMm,
      meaning: 'Synthetic measurement series for charts/inspection. It is not used to animate the cap surface.',
    },
    verticalSigma: {
      qualityPreset: synthetic.uncertaintyQuality,
      qualityPresetMeaning: 'Synthetic generator profile only; it is not a measured uncertainty-quality class.',
      sigmaFloorMm: synthetic.sigmaFloorMm,
      qualityParameters: synthetic.quality,
      sigma_provenance: 'synthetic_demo',
      sourceRateUncertainty: {
        field: 'var_up',
        derivedAs: 'sqrt(var_up)',
        unit: 'mm/yr',
        role: 'Input rate-uncertainty term used by the synthetic sigma_z display generator.',
      },
      blankiePolicy: {
        source: 'IDW live sigma with maximum-neighbour inflation and radius penalty',
        ...synthetic.blankieSigma,
      },
      meaning: 'Per-row, per-epoch synthetic 1-sigma uncertainty in mm.',
    },
    verticalUncertaintyLegend,
    uncertaintyRelief: {
      displayRange: synthetic.displayRange,
      visualFade: synthetic.visualFade,
      geometry: config.uncertainty_relief?.geometry ?? {},
      status: 'deckgl_instanced_checkerboard_relief_ready',
      note: 'unit=sigma means a multiple of each cell\'s current sigma; unit=mm means a fixed absolute future relief range. The renderer instances one static checkerboard mesh per cell; no per-epoch cap geometry is rebuilt.',
    },
    horizontalStaticGlyphs: {
      ...horizontalGlyphPayload,
      records: undefined,
    },
    horizontalParticles: horizontalParticleBuild.metadata,
    horizontalParticleLsc: horizontalParticleLscBuild.available
      ? horizontalParticleLscBuild.metadata
      : undefined,
    verticalVelocityColorScale,
    // Kept as a compatibility seed for older viewers. The active DeckGL
    // runtime reads verticalVelocityColorScale, which is asymmetric and
    // uncertainty-aware by design.
    rateColorSeed: {
      field: verticalVelocityColorScale.field,
      unit: verticalVelocityColorScale.unit,
      legacy: 'superseded_by_verticalVelocityColorScale',
      symmetricAbsMax: Math.max(
        verticalVelocityColorScale.subsidenceLimitMmYr,
        verticalVelocityColorScale.upliftLimitMmYr,
      ),
    },
    viewer: config.viewer,
    blankieSelection: selection.summary,
    pitDomain: {
      scope: config.viewer.pit_mode.scope,
      iMin: 0,
      iMax: grid.maxI,
      jMin: 0,
      jMax: grid.maxJ,
      supportBounds,
      northwestMaxSubsidenceAnchor: {
        rumId: anchor.rumId,
        gridI: anchor.gridI,
        gridJ: anchor.gridJ,
        upMmYr: anchor.upMmYr,
      },
    },
    assets: {
      staticRums: 'rum_static.json',
      interpolatedBlankies: 'interpolated_blankies.json',
      epochAxis: 'epoch_axis.json',
      verticalModelF32: 'vertical_model_mm_f32.bin',
      verticalMeasurementF32: 'vertical_measurement_mm_f32.bin',
      verticalSigmaF32: 'vertical_sigma_mm_f32.bin',
      horizontalGlyphs: 'horizontal_glyphs.json',
      horizontalParticleField: 'horizontal_particle_field.json',
      horizontalParticleFieldLsc: horizontalParticleLscBuild.available
        ? horizontalParticleLscBuild.assetName
        : undefined,
    },
  });

  console.log('');
  console.log('DeckGL Jakarta completed support-envelope assets built.');
  console.log(`Live RUMs: ${rumCount}`);
  console.log(`Interpolated support-envelope blankies: ${blankCount}`);
  console.log(`Unfilled no-data cells: ${unfilledNoDataCellCount}`);
  console.log(`Grid: ${grid.maxI + 1} columns × ${grid.maxJ + 1} rows`);
  console.log(`Runtime rows: ${runtimeRowCount}`);
  console.log(`Epochs: ${epochCount}`);
  console.log(`Float32 values per series: ${runtimeModelMm.length}`);
  console.log(`Synthetic measurement: ${synthetic.measurementBehavior} + ${synthetic.measurementNoiseName} noise (${synthetic.measurementNoiseSigmaMm.toFixed(1)} mm)`);
  console.log(`Synthetic uncertainty: ${synthetic.uncertaintyQuality} confidence · floor ${synthetic.sigmaFloorMm.toFixed(2)} mm · future relief ±${synthetic.displayRange.value}${synthetic.displayRange.unit === 'sigma' ? 'σ' : ' mm'}`);
  console.log(`Vertical velocity colour: ${verticalVelocityColorScale.mode} · zero ±${verticalVelocityColorScale.nearZeroThresholdMmYr.toFixed(2)} ${verticalVelocityColorScale.unit} · subsidence ≤ −${verticalVelocityColorScale.subsidenceLimitMmYr.toFixed(2)} · uplift ≥ +${verticalVelocityColorScale.upliftLimitMmYr.toFixed(2)}`);
  console.log(`Horizontal static glyph pairs: ${horizontalGlyphPayload.summary.visibleGlyphPairCount} (low-speed skipped ${horizontalGlyphPayload.summary.skippedLowSpeed}, insignificant skipped ${horizontalGlyphPayload.summary.skippedInsignificantVsUncertainty})`);
  console.log(`Horizontal GPU particle field: ${horizontalParticleBuild.metadata.summary.liveRumCount} live cells · ${horizontalParticleBuild.metadata.summary.spawnCellCount} spawn cells · p95 ${horizontalParticleBuild.metadata.summary.speedP95MmYr.toFixed(3)} mm/year`);
  if (horizontalParticleLscBuild.available) {
    const lscSummary = horizontalParticleLscBuild.metadata.summary;
    console.log(`[LSC] optional signal field: ${lscSummary.validFineTexelCount.toLocaleString()} fine texels · ${lscSummary.spawnCellCount.toLocaleString()} parity spawns${horizontalParticleLscBuild.cached ? ' · cache hit' : ''}`);
  } else {
    console.log(`[LSC] optional signal field unavailable: ${horizontalParticleLscBuild.reason ?? 'not built'}`);
  }
  console.log(`NW max-subsidence anchor: ${anchor.rumId} (${anchor.upMmYr.toFixed(3)} mm/year)`);
  console.log(`Output: ${OUTPUT_DIR}`);
}

main().catch((error) => {
  console.error('');
  console.error('DeckGL Jakarta interpolated blankie asset build failed.');
  console.error(error.stack ?? error.message ?? error);
  process.exit(1);
});
