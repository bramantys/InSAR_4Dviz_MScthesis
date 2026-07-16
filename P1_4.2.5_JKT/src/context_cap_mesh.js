// DeckGL-native contextual cap texture support.
//
// A single soft-B/W study atlas is baked from raster XYZ tiles. Every cap
// receives the atlas UV of its four fixed geographic corners, projected once
// into Web Mercator. The viewer only changes cap Z, never those corner UVs.

const EARTH_RADIUS_M = 6378137;
const TILE_SIZE = 256;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lonLatToWorldPixel(lon, lat, zoom) {
  const scale = TILE_SIZE * (2 ** zoom);
  const x = ((lon + 180) / 360) * scale;
  const clampedLat = clamp(lat, -85.05112878, 85.05112878);
  const latRad = (clampedLat * Math.PI) / 180;
  const y = (0.5 - Math.log((1 + Math.sin(latRad)) / (1 - Math.sin(latRad))) / (4 * Math.PI)) * scale;
  return [x, y];
}

// EPSG:3857 axes: x increases east, y increases north. Keep this separate
// from XYZ world pixels, whose Y axis increases south, so the one required
// raster-to-WebGL V flip is explicit and never hidden in per-cell math.
function lonLatToMercatorMeters(lon, lat) {
  const clampedLat = clamp(lat, -85.05112878, 85.05112878);
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (clampedLat * Math.PI) / 180;
  return [
    EARTH_RADIUS_M * lonRad,
    EARTH_RADIUS_M * Math.log(Math.tan(Math.PI * 0.25 + latRad * 0.5)),
  ];
}

function buildTileUrl(template, z, x, y) {
  return template
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y));
}

async function loadTileImage(url) {
  // `fetch + createImageBitmap` gives us a CORS failure immediately rather
  // than silently producing a tainted canvas later.
  const response = await fetch(url, {mode: 'cors', cache: 'force-cache'});
  if (!response.ok) throw new Error(`Tile ${response.status}: ${url}`);
  const blob = await response.blob();
  if (typeof createImageBitmap === 'function') return createImageBitmap(blob);

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Image decode failed: ${url}`));
    image.src = URL.createObjectURL(blob);
  });
}

export function computeStudyBounds(cells, {paddingFraction = 0.025} = {}) {
  const lons = [];
  const lats = [];
  for (const cell of cells ?? []) {
    for (const point of cell.footprintLonLat?.slice(0, 4) ?? []) {
      if (Number.isFinite(point?.[0]) && Number.isFinite(point?.[1])) {
        lons.push(point[0]);
        lats.push(point[1]);
      }
    }
  }
  if (!lons.length || !lats.length) throw new Error('Cannot derive context-atlas bounds: no valid cap footprints.');

  const minLon0 = Math.min(...lons);
  const maxLon0 = Math.max(...lons);
  const minLat0 = Math.min(...lats);
  const maxLat0 = Math.max(...lats);
  const lonPad = Math.max(0.002, (maxLon0 - minLon0) * paddingFraction);
  const latPad = Math.max(0.002, (maxLat0 - minLat0) * paddingFraction);

  return {
    minLon: minLon0 - lonPad,
    maxLon: maxLon0 + lonPad,
    minLat: minLat0 - latPad,
    maxLat: maxLat0 + latPad,
  };
}

/**
 * Build a fixed, mipmapped study-area atlas from an XYZ raster source.
 *
 * The browser/deck.gl image prop creates the GPU texture and its full mip
 * chain. The custom sampler below selects linear filtering between mip levels;
 * no home-grown zoom/LOD tile manager is needed for this atlas.
 */
export async function buildRasterAtlas({
  bounds,
  tileTemplate,
  zoom = 13,
  maxDimension = 4096,
  maxTileCount = Infinity,
  onProgress = null,
}) {
  if (!bounds || !tileTemplate) throw new Error('Context atlas requires bounds and an XYZ tile template.');

  const [minX, maxY] = lonLatToWorldPixel(bounds.minLon, bounds.minLat, zoom);
  const [maxX, minY] = lonLatToWorldPixel(bounds.maxLon, bounds.maxLat, zoom);
  const pixelWidth = Math.max(1, maxX - minX);
  const pixelHeight = Math.max(1, maxY - minY);
  const scale = Math.min(1, maxDimension / Math.max(pixelWidth, pixelHeight));
  const width = Math.max(2, Math.round(pixelWidth * scale));
  const height = Math.max(2, Math.round(pixelHeight * scale));

  const tileX0 = Math.floor(minX / TILE_SIZE);
  const tileX1 = Math.floor((maxX - 1e-6) / TILE_SIZE);
  const tileY0 = Math.floor(minY / TILE_SIZE);
  const tileY1 = Math.floor((maxY - 1e-6) / TILE_SIZE);
  const tileCount = (tileX1 - tileX0 + 1) * (tileY1 - tileY0 + 1);
  if (Number.isFinite(maxTileCount) && tileCount > maxTileCount) {
    throw new Error(
      `Context atlas needs ${tileCount} tiles at z${zoom}, above configured limit ${maxTileCount}. ` +
      'Keep the overview atlas active or reduce the focus atlas zoom/extent.',
    );
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', {alpha: false});
  context.fillStyle = '#e7e7e3';
  context.fillRect(0, 0, width, height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';

  const jobs = [];
  const worldTileCount = 2 ** zoom;
  for (let tileY = tileY0; tileY <= tileY1; tileY++) {
    for (let tileX = tileX0; tileX <= tileX1; tileX++) {
      const wrappedX = ((tileX % worldTileCount) + worldTileCount) % worldTileCount;
      const url = buildTileUrl(tileTemplate, zoom, wrappedX, tileY);
      jobs.push({tileX, tileY, url});
    }
  }

  let complete = 0;
  let failures = 0;
  await Promise.all(jobs.map(async ({tileX, tileY, url}) => {
    try {
      const image = await loadTileImage(url);
      const drawX = (tileX * TILE_SIZE - minX) * scale;
      const drawY = (tileY * TILE_SIZE - minY) * scale;
      context.drawImage(image, drawX, drawY, TILE_SIZE * scale, TILE_SIZE * scale);
      image.close?.();
    } catch (error) {
      failures += 1;
      console.warn('[Proto1 DeckGL] Context atlas tile failed', error);
    } finally {
      complete += 1;
      onProgress?.({complete, total: tileCount, failures});
    }
  }));

  // Do not pretend a blank canvas is a valid contextual map. A few missing
  // tiles are harmless; a systemic CORS/provider failure is not.
  if (failures > Math.max(3, Math.ceil(tileCount * 0.25))) {
    throw new Error(`Context atlas failed: ${failures}/${tileCount} tiles unavailable.`);
  }

  const [mercatorMinX, mercatorMinY] = lonLatToMercatorMeters(bounds.minLon, bounds.minLat);
  const [mercatorMaxX, mercatorMaxY] = lonLatToMercatorMeters(bounds.maxLon, bounds.maxLat);
  const centerLon = 0.5 * (bounds.minLon + bounds.maxLon);
  const centerLat = 0.5 * (bounds.minLat + bounds.maxLat);
  const metersPerLon = (Math.PI * EARTH_RADIUS_M / 180) * Math.cos((centerLat * Math.PI) / 180);
  const metersPerLat = Math.PI * EARTH_RADIUS_M / 180;

  return {
    canvas,
    bounds,
    zoom,
    width,
    height,
    scale,
    tileCount,
    requestedPixelWidth: pixelWidth,
    requestedPixelHeight: pixelHeight,
    minX,
    minY,
    maxX,
    maxY,
    mercatorBounds: {
      minX: mercatorMinX,
      minY: mercatorMinY,
      maxX: mercatorMaxX,
      maxY: mercatorMaxY,
    },
    center: [centerLon, centerLat, 0],
    metersPerLon,
    metersPerLat,
    uvForLonLat(lon, lat) {
      const [mercatorX, mercatorY] = lonLatToMercatorMeters(lon, lat);
      const u = clamp(
        (mercatorX - mercatorMinX) / Math.max(1e-12, mercatorMaxX - mercatorMinX),
        0,
        1,
      );
      const vGeographic = clamp(
        (mercatorY - mercatorMinY) / Math.max(1e-12, mercatorMaxY - mercatorMinY),
        0,
        1,
      );
      // Canvas/XYZ rows start at the geographic north. SimpleMeshLayer samples
      // WebGL texture UVs from the opposite vertical convention, so flip ONCE
      // here at atlas level. Never apply a per-cell flip.
      return [u, 1 - vGeographic];
    },
    localMetersForLonLat(lon, lat) {
      return [
        (lon - centerLon) * metersPerLon,
        (lat - centerLat) * metersPerLat,
      ];
    },
  };
}

/**
 * One shared local quad used by all live/blankie cap instances. The context
 * layer supplies four geographic atlas UVs for every instance.
 */
export function createContextCapQuadMesh() {
  return {
    attributes: {
      positions: {
        size: 3,
        value: new Float32Array([
          -0.5, -0.5, 0,  0.5, -0.5, 0,  0.5,  0.5, 0,
          -0.5, -0.5, 0,  0.5,  0.5, 0, -0.5,  0.5, 0,
        ]),
      },
      normals: {
        size: 3,
        value: new Float32Array([
          0, 0, 1,  0, 0, 1,  0, 0, 1,
          0, 0, 1,  0, 0, 1,  0, 0, 1,
        ]),
      },
      texCoords: {
        size: 2,
        value: new Float32Array([
          0, 0,  1, 0,  1, 1,
          0, 0,  1, 1,  0, 1,
        ]),
      },
    },
    vertexCount: 6,
    triangleCount: 2,
  };
}

/**
 * Derive four atlas UVs directly from each footprint's true geographic
 * corners. Input footprint convention is the project-wide [SW, SE, NE, NW]
 * convention already used by deriveFootprintTransform() and wall generation.
 */
export function assignContextCornerUvs(cells, atlas, atlasKey = 'overview') {
  if (!atlas) return;
  for (const cell of cells ?? []) {
    const corners = cell.footprintLonLat?.slice(0, 4) ?? [];
    if (corners.length !== 4) continue;

    const [sw, se, ne, nw] = corners;
    const uvSw = atlas.uvForLonLat(sw[0], sw[1]);
    const uvSe = atlas.uvForLonLat(se[0], se[1]);
    const uvNe = atlas.uvForLonLat(ne[0], ne[1]);
    const uvNw = atlas.uvForLonLat(nw[0], nw[1]);

    // No per-RUM affine / rotation transform. These are the actual atlas UVs
    // at the four fixed cap corners; the GPU does the within-quad interpolation.
    const contextUvs = cell.contextUvsByAtlas ?? {};
    contextUvs[atlasKey] = {
      south: [uvSw[0], uvSw[1], uvSe[0], uvSe[1]],
      north: [uvNw[0], uvNw[1], uvNe[0], uvNe[1]],
    };
    cell.contextUvsByAtlas = contextUvs;

    // Keep the legacy aliases pointing at the most recently prepared atlas.
    // The viewer reads contextUvsByAtlas[activeKey] so atlas LOD swaps do not
    // mutate geometry or require a per-cell affine transform.
    cell.contextUvSouth = contextUvs[atlasKey].south;
    cell.contextUvNorth = contextUvs[atlasKey].north;
  }
}
