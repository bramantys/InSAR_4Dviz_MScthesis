// -----------------------------------------------------------------------------
// DeckGL-native horizontal static glyph meshes.
//
// The V7.2 glyph math lives in the data builder. These are only three shared
// unit meshes: shaft, arrowhead, and a 95% confidence-ring template. Each
// accepted RUM is instanced with its own position, rotation and scale.
// -----------------------------------------------------------------------------

function meshFromTriangles(triangles) {
  const positions = [];
  const normals = [];
  for (const triangle of triangles) {
    for (const point of triangle) {
      positions.push(point[0], point[1], point[2] ?? 0);
      normals.push(0, 0, 1);
    }
  }
  return {
    attributes: {
      positions: {size: 3, value: new Float32Array(positions)},
      normals: {size: 3, value: new Float32Array(normals)},
    },
    vertexCount: positions.length / 3,
    triangleCount: positions.length / 9,
  };
}

/**
 * Unit arrow shaft. Its local origin is the arrow tail; X points toward the
 * arrowhead. Instance scale supplies [shaftLengthM, shaftHalfWidthM, 1].
 */
export function createArrowShaftMesh() {
  return meshFromTriangles([
    [[0, -1, 0], [1, -1, 0], [1, 1, 0]],
    [[0, -1, 0], [1, 1, 0], [0, 1, 0]],
  ]);
}

/**
 * Unit arrowhead. Its local origin is the head base; X points toward the tip.
 * Instance scale supplies [headLengthM, headHalfWidthM, 1].
 */
export function createArrowHeadMesh() {
  return meshFromTriangles([
    [[0, -1, 0], [1, 0, 0], [0, 1, 0]],
  ]);
}

/**
 * Unit annulus for the horizontal confidence ellipse. Instance X/Y scale
 * supplies the 95% semi-major/semi-minor axes. The ring is deliberately
 * unlit; colour and opacity remain stable above either scientific or context
 * caps. The visual stroke is a modest relative band, comparable to V7.2's
 * 4.5 m line at the Jakarta working scale.
 */
export function createConfidenceEllipseMesh({segments = 64, innerRadius = 0.94} = {}) {
  const n = Math.max(16, Math.round(Number(segments) || 64));
  const inner = Math.min(0.995, Math.max(0.65, Number(innerRadius) || 0.94));
  const triangles = [];
  for (let index = 0; index < n; index += 1) {
    const a0 = (2 * Math.PI * index) / n;
    const a1 = (2 * Math.PI * (index + 1)) / n;
    const outer0 = [Math.cos(a0), Math.sin(a0), 0];
    const outer1 = [Math.cos(a1), Math.sin(a1), 0];
    const inner0 = [inner * Math.cos(a0), inner * Math.sin(a0), 0];
    const inner1 = [inner * Math.cos(a1), inner * Math.sin(a1), 0];
    triangles.push([outer0, outer1, inner1]);
    triangles.push([outer0, inner1, inner0]);
  }
  const mesh = meshFromTriangles(triangles);
  mesh.segments = n;
  mesh.innerRadius = inner;
  return mesh;
}
