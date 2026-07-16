// -----------------------------------------------------------------------------
// DeckGL-native uncertainty relief mesh — visual-polish pass.
//
// One shared, low-poly 4×4 checkerboard mesh is created once. DeckGL instances
// it once per RUM/support cell using SimpleMeshLayer. The mesh carries its own
// per-facet tint, while the instance colour still carries the scientific RUM
// red/blue (or blankie grey). The material is unlit: map/camera lighting can
// no longer steal or darken the mean cap colour.
// -----------------------------------------------------------------------------

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function radians(degrees) {
  return (degrees * Math.PI) / 180;
}

function degrees(radiansValue) {
  return (radiansValue * 180) / Math.PI;
}

function normalizeDegrees(value) {
  let result = value;
  while (result > 180) result -= 360;
  while (result <= -180) result += 360;
  return result;
}

function normalizeVector(vector, fallback = [0, 0, 1]) {
  const x = Number(vector?.[0]);
  const y = Number(vector?.[1]);
  const z = Number(vector?.[2]);
  const length = Math.hypot(x, y, z);
  if (!Number.isFinite(length) || length < 1e-8) return fallback;
  return [x / length, y / length, z / length];
}

function haversineMeters(a, b) {
  const [lon1, lat1] = a;
  const [lon2, lat2] = b;
  const earthRadiusM = 6371008.8;
  const dLat = radians(lat2 - lat1);
  const dLon = radians(lon2 - lon1);
  const phi1 = radians(lat1);
  const phi2 = radians(lat2);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLon / 2) ** 2;
  return 2 * earthRadiusM * Math.asin(Math.sqrt(h));
}

function bearingFromNorthDegrees(a, b) {
  const [lon1, lat1] = a;
  const [lon2, lat2] = b;
  const phi1 = radians(lat1);
  const phi2 = radians(lat2);
  const deltaLon = radians(lon2 - lon1);
  const y = Math.sin(deltaLon) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLon);
  return degrees(Math.atan2(y, x));
}

/**
 * Derive a stable local transform for the shared relief mesh from a real,
 * already-transformed RUM footprint. This keeps the deformation runtime
 * instanced; no cap polygons are rebuilt per epoch.
 */
export function deriveFootprintTransform(footprintLonLat, fallbackCellSizeM = 450) {
  const corners = Array.isArray(footprintLonLat)
    ? footprintLonLat.slice(0, 4)
    : [];

  const valid = corners.length === 4 && corners.every(
    (point) => Array.isArray(point) && Number.isFinite(point[0]) && Number.isFinite(point[1]),
  );

  if (!valid) {
    return {
      position: [0, 0, 0],
      widthM: fallbackCellSizeM,
      heightM: fallbackCellSizeM,
      yawDeg: 0,
    };
  }

  const [sw, se, ne, nw] = corners;
  const centerLon = 0.25 * (sw[0] + se[0] + ne[0] + nw[0]);
  const centerLat = 0.25 * (sw[1] + se[1] + ne[1] + nw[1]);

  const widthM = 0.5 * (
    haversineMeters(sw, se) +
    haversineMeters(nw, ne)
  );

  const heightM = 0.5 * (
    haversineMeters(sw, nw) +
    haversineMeters(se, ne)
  );

  // At yaw = 0, a SimpleMeshLayer mesh X axis is locally east-aligned.
  const sourceEastBearing = bearingFromNorthDegrees(sw, se);
  const yawDeg = normalizeDegrees(90 - sourceEastBearing);

  return {
    position: [centerLon, centerLat, 0],
    widthM: Number.isFinite(widthM) && widthM > 0 ? widthM : fallbackCellSizeM,
    heightM: Number.isFinite(heightM) && heightM > 0 ? heightM : fallbackCellSizeM,
    yawDeg: Number.isFinite(yawDeg) ? yawDeg : 0,
  };
}

function getFacetShade(normal, kind, tint) {
  if (kind === 'flat') return tint.flat;

  const lambert = Math.max(
    0,
    normal[0] * tint.lightDirection[0] +
      normal[1] * tint.lightDirection[1] +
      normal[2] * tint.lightDirection[2],
  );

  const [minShade, maxShade] = kind === 'up' ? tint.up : tint.down;
  return minShade + (maxShade - minShade) * lambert;
}

function pushTriangle(positions, normals, colors, a, b, c, kind, tint) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const abz = b[2] - a[2];
  const acx = c[0] - a[0];
  const acy = c[1] - a[1];
  const acz = c[2] - a[2];

  let nx = aby * acz - abz * acy;
  let ny = abz * acx - abx * acz;
  let nz = abx * acy - aby * acx;
  const length = Math.hypot(nx, ny, nz) || 1;
  nx /= length;
  ny /= length;
  nz /= length;

  // SimpleMeshLayer receives explicit hard facet normals. Keep dimple facets
  // top-facing as well; their darker tint, not upside-down lighting, conveys
  // their negative relief.
  if (nz < 0) {
    nx *= -1;
    ny *= -1;
    nz *= -1;
  }

  const shade = getFacetShade([nx, ny, nz], kind, tint);

  for (const point of [a, b, c]) {
    positions.push(point[0], point[1], point[2]);
    normals.push(nx, ny, nz);
    colors.push(shade, shade, shade);
  }
}

/**
 * V7.2-inspired 4×4 checkerboard pyramids/dimples as one static shared
 * instanced-mesh template. The local X/Y extent is [-0.5, +0.5]. The local Z
 * is a unit relief: runtime instance scaling supplies the displayed sigma range
 * and vertical exaggeration.
 */
export function createCheckerboardReliefMesh(options = {}) {
  const gridN = clamp(Math.round(Number(options.grid_n_per_rum ?? 4)), 1, 8);
  const border = clamp(Number(options.flat_border_fraction ?? 0.10), 0, 0.35);
  const upGain = clamp(Number(options.up_relief_gain ?? 0.90), 0.05, 3.0);
  const downGain = clamp(Number(options.down_relief_gain ?? 1.10), 0.05, 3.0);
  const upFootprint = clamp(Number(options.up_relief_footprint_fraction ?? 0.36), 0.05, 0.95);
  // Downward and upward relief intentionally share the same footprint by default.
  // A smaller far-LOD fraction is allowed so 2×2 stays sparse without making
  // its individual pyramids physically larger than the 4×4 inspection mesh.
  const downFootprint = clamp(Number(options.down_relief_footprint_fraction ?? 0.36), 0.05, 0.95);

  const facetTint = options.facet_tint ?? {};
  const tint = {
    flat: clamp(Number(facetTint.flat ?? 1.0), 0.0, 1.5),
    up: [
      clamp(Number(facetTint.up_min ?? 0.78), 0.0, 1.5),
      clamp(Number(facetTint.up_max ?? 0.92), 0.0, 1.5),
    ],
    down: [
      clamp(Number(facetTint.down_min ?? 0.60), 0.0, 1.5),
      clamp(Number(facetTint.down_max ?? 0.78), 0.0, 1.5),
    ],
    lightDirection: normalizeVector(facetTint.light_direction_local ?? [-0.42, -0.36, 0.83]),
  };
  tint.up.sort((a, b) => a - b);
  tint.down.sort((a, b) => a - b);

  const positions = [];
  const normals = [];
  const colors = [];

  const innerMin = border;
  const innerMax = 1 - border;
  const innerSpan = Math.max(1e-6, innerMax - innerMin);
  const pitch = innerSpan / gridN;

  const coords = [0, innerMin];
  for (let k = 1; k < gridN; k++) coords.push(innerMin + pitch * k);
  coords.push(innerMax, 1);

  const innerIndexStart = 1;
  const innerIndexEnd = innerIndexStart + gridN - 1;

  const toLocal = (u, v, z = 0) => [u - 0.5, v - 0.5, z];

  for (let j = 0; j < coords.length - 1; j++) {
    for (let i = 0; i < coords.length - 1; i++) {
      const u0 = coords[i];
      const u1 = coords[i + 1];
      const v0 = coords[j];
      const v1 = coords[j + 1];

      const a = toLocal(u0, v0);
      const b = toLocal(u1, v0);
      const c = toLocal(u1, v1);
      const d = toLocal(u0, v1);

      const isReliefCell =
        i >= innerIndexStart && i <= innerIndexEnd &&
        j >= innerIndexStart && j <= innerIndexEnd;

      if (!isReliefCell) {
        pushTriangle(positions, normals, colors, a, b, c, 'flat', tint);
        pushTriangle(positions, normals, colors, a, c, d, 'flat', tint);
        continue;
      }

      const localI = i - innerIndexStart;
      const localJ = j - innerIndexStart;
      const upward = ((localI + localJ) % 2) === 0;
      const reliefGain = upward ? upGain : -downGain;
      const footprintFraction = upward ? upFootprint : downFootprint;
      const kind = upward ? 'up' : 'down';

      const insetU = 0.5 * (1 - footprintFraction) * (u1 - u0);
      const insetV = 0.5 * (1 - footprintFraction) * (v1 - v0);
      const su0 = u0 + insetU;
      const su1 = u1 - insetU;
      const sv0 = v0 + insetV;
      const sv1 = v1 - insetV;
      const uc = 0.5 * (u0 + u1);
      const vc = 0.5 * (v0 + v1);

      const p00 = toLocal(su0, sv0);
      const p10 = toLocal(su1, sv0);
      const p11 = toLocal(su1, sv1);
      const p01 = toLocal(su0, sv1);
      const tip = toLocal(uc, vc, reliefGain);

      // Flat ring preserves the exact mean-cap colour. It dominates the RUM
      // surface; the small centre relief is a local uncertainty cue only.
      pushTriangle(positions, normals, colors, a, b, p10, 'flat', tint);
      pushTriangle(positions, normals, colors, a, p10, p00, 'flat', tint);
      pushTriangle(positions, normals, colors, b, c, p11, 'flat', tint);
      pushTriangle(positions, normals, colors, b, p11, p10, 'flat', tint);
      pushTriangle(positions, normals, colors, c, d, p01, 'flat', tint);
      pushTriangle(positions, normals, colors, c, p01, p11, 'flat', tint);
      pushTriangle(positions, normals, colors, d, a, p00, 'flat', tint);
      pushTriangle(positions, normals, colors, d, p00, p01, 'flat', tint);

      // Four-sided low-poly peak/dimple. Per-facet tint is fixed in local
      // space, so it remains legible without reacting to camera or map lights.
      pushTriangle(positions, normals, colors, p00, p10, tip, kind, tint);
      pushTriangle(positions, normals, colors, p10, p11, tip, kind, tint);
      pushTriangle(positions, normals, colors, p11, p01, tip, kind, tint);
      pushTriangle(positions, normals, colors, p01, p00, tip, kind, tint);
    }
  }

  // SimpleMeshLayer v9 expects MeshAttributes, not bare typed arrays. A
  // de-indexed triangle list keeps every relief facet crisp and independent.
  const positionArray = new Float32Array(positions);
  const normalArray = new Float32Array(normals);
  const colorArray = new Float32Array(colors);

  return {
    attributes: {
      positions: {size: 3, value: positionArray},
      normals: {size: 3, value: normalArray},
      colors: {size: 3, value: colorArray},
    },
    vertexCount: positionArray.length / 3,
    triangleCount: positionArray.length / 9,
    gridN,
  };
}
