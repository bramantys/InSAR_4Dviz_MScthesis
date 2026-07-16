// Grid-native support topology for the DeckGL Jakarta pit.
//
// Corner order is SW, SE, NE, NW (indices 0..3).
// All connectivity is 4-neighbour only. Corner-touching cells remain
// separate boundary components by design.

const EDGE_E = {di: 1, dj: 0, corners: [1, 2]};
const EDGE_N = {di: 0, dj: 1, corners: [2, 3]};
const SIDES = [
  {name: 'S', di: 0, dj: -1, corners: [0, 1]},
  {name: 'E', di: 1, dj: 0, corners: [1, 2]},
  {name: 'N', di: 0, dj: 1, corners: [2, 3]},
  {name: 'W', di: -1, dj: 0, corners: [3, 0]},
];

export const cellKey = (i, j) => `${i}:${j}`;

function cellFootprintAtDatum({gridI, gridJ, grid, projectToWgs84}) {
  const x = grid.minX + gridI * grid.rumSizeM;
  const y = grid.minY + gridJ * grid.rumSizeM;
  const half = grid.rumSizeM * 0.5;

  const cornersXY = [
    [x - half, y - half],
    [x + half, y - half],
    [x + half, y + half],
    [x - half, y + half],
  ];

  return cornersXY.map(([cornerX, cornerY]) => {
    const [lon, lat] = projectToWgs84(cornerX, cornerY);
    return [lon, lat, 0];
  });
}

function buildOuterApronBands({domainBounds, grid, marginCells, projectToWgs84}) {
  const {minX, minY, rumSizeM} = grid;

  const hx0 = minX + (domainBounds.iMin - 0.5) * rumSizeM;
  const hx1 = minX + (domainBounds.iMax + 0.5) * rumSizeM;
  const hy0 = minY + (domainBounds.jMin - 0.5) * rumSizeM;
  const hy1 = minY + (domainBounds.jMax + 0.5) * rumSizeM;

  const marginM = marginCells * rumSizeM;
  const ax0 = hx0 - marginM;
  const ax1 = hx1 + marginM;
  const ay0 = hy0 - marginM;
  const ay1 = hy1 + marginM;

  const bandsXY = [
    [[ax0, ay0], [ax1, ay0], [ax1, hy0], [ax0, hy0]],
    [[ax0, hy1], [ax1, hy1], [ax1, ay1], [ax0, ay1]],
    [[ax0, hy0], [hx0, hy0], [hx0, hy1], [ax0, hy1]],
    [[hx1, hy0], [ax1, hy0], [ax1, hy1], [hx1, hy1]],
  ];

  const toLonLatZ0 = ([x, y]) => {
    const [lon, lat] = projectToWgs84(x, y);
    return [lon, lat, 0];
  };

  return {
    bands: bandsXY.map((band) => band.map(toLonLatZ0)),
    outerRing: [
      [ax0, ay0], [ax1, ay0], [ax1, ay1], [ax0, ay1], [ax0, ay0],
    ].map(toLonLatZ0),
  };
}

// Builds shared support-surface walls plus every exposed support edge.
// A rim edge exists wherever a LIVE or INTERPOLATED_SUPPORT cell touches
// DATUM_NO_DATA or lies at the outer source-grid boundary.
export function buildTopology(cells) {
  const byKey = new Map(cells.map((cell) => [cellKey(cell.gridI, cell.gridJ), cell]));
  const neighbourWalls = [];
  const rimEdges = [];

  for (const owner of cells) {
    for (const edge of [EDGE_E, EDGE_N]) {
      const neighbour = byKey.get(cellKey(owner.gridI + edge.di, owner.gridJ + edge.dj));
      if (!neighbour) continue;

      const [a, b] = edge.corners;
      neighbourWalls.push({
        cellKeyA: cellKey(owner.gridI, owner.gridJ),
        cellKeyB: cellKey(neighbour.gridI, neighbour.gridJ),
        edgeLonLat: [owner.footprintLonLat[a], owner.footprintLonLat[b]],
      });
    }

    for (const side of SIDES) {
      const neighbour = byKey.get(cellKey(owner.gridI + side.di, owner.gridJ + side.dj));
      if (neighbour) continue;

      const [a, b] = side.corners;
      rimEdges.push({
        cellKey: cellKey(owner.gridI, owner.gridJ),
        side: side.name,
        edgeLonLat: [owner.footprintLonLat[a], owner.footprintLonLat[b]],
      });
    }
  }

  return {neighbourWalls, rimEdges};
}

// Builds a depth-writing datum field from all non-support cells inside the
// original source grid plus a wide outer margin. This is a grid-native
// irregular apron: its opening is exactly the completed support envelope.
export function buildDatumGround({domainBounds, grid, supportKeys, marginCells, projectToWgs84}) {
  const datumCells = [];

  for (let gridJ = domainBounds.jMin; gridJ <= domainBounds.jMax; gridJ += 1) {
    for (let gridI = domainBounds.iMin; gridI <= domainBounds.iMax; gridI += 1) {
      if (supportKeys.has(cellKey(gridI, gridJ))) continue;

      datumCells.push({
        gridI,
        gridJ,
        polygon: cellFootprintAtDatum({gridI, gridJ, grid, projectToWgs84}),
      });
    }
  }

  const outer = buildOuterApronBands({
    domainBounds,
    grid,
    marginCells,
    projectToWgs84,
  });

  return {
    datumCells,
    outerBands: outer.bands,
    outerRing: outer.outerRing,
  };
}
