import {Layer, project32} from '@deck.gl/core';
import {Buffer, Texture} from '@luma.gl/core';
import {BufferTransform, Model} from '@luma.gl/engine';

// DeckGL-native horizontal particle renderer.
//
// - Particle state never leaves GPU memory after initialization.
// - WebGL2 transform feedback ping-pongs current particle state; trail history lives in a tiled RGBA32F atlas.
// - The field is sampled from compact Float32 textures built from live RUMs only.
// - Trails render inside the DeckGL pass, so they stay georeferenced and depth-test
//   against the animated caps instead of behaving like the old DOM canvas overlay.

const MODE = Object.freeze({mean: 0, montecarlo: 1, shimmer: 2});
const MAX_DT_SECONDS = 0.05;
const MAX_INTEGRATION_SUBSTEPS = 24;
const MAX_HISTORY_SAMPLES = 65;

const HORIZONTAL_PARTICLE_UNIFORMS = {
  name: 'horizontalParticle',
  source: '',
  vs: /* glsl */ `\
layout(std140) uniform horizontalParticleUniforms {
  float time;
  float deltaSeconds;
  float speedP95MmYr;
  float baseMps;
  float speedMultiplier;
  float stallSpeedMmYr;
  float uncertaintyStrength;
  float mcMaxSigma;
  float mcOffsetCapMmYr;
  float mcOffsetCapRatioToSpeed;
  float verticalExaggeration;
  float surfaceOffsetM;
  float particleSizePixels;
  float particleOpacity;
  float trailWidthPixels;
  float trailOpacity;
  float shimmerPixelAmplitude;
  float shimmerReferenceSigmaThetaDeg;
  float birthFadeSeconds;
  float trailPersistence;
  float historySampleIntervalS;
  float maxTrailScreenJumpPx;
  float integrationMaxCellFraction;
  float spawnJitterCells;
  int gridWidth;
  int gridHeight;
  int spawnCount;
  int activeEpoch;
  int uncertaintyMode;
  int frameIndex;
  int particleCapacity;
  int historySamples;
  int historyStorageSamples;
  int historyHeadRow;
  int historyAtlasWidth;
  int historyParticleRows;
  int flatMode;
  float flatSurfaceZM;
  vec2 gridOriginLocalM;
  vec2 gridAxisIM;
  vec2 gridAxisJM;
  vec4 particleColor;
} horizontalParticle;
`,
  // Every custom particle UBO is vertex-stage-only. Keeping it out of fragment
  // assembly avoids the Firefox/WebGL2 cross-stage UBO link failure we hit in
  // the first TF implementation.
  fs: '',
  uniformTypes: {
    time: 'f32',
    deltaSeconds: 'f32',
    speedP95MmYr: 'f32',
    baseMps: 'f32',
    speedMultiplier: 'f32',
    stallSpeedMmYr: 'f32',
    uncertaintyStrength: 'f32',
    mcMaxSigma: 'f32',
    mcOffsetCapMmYr: 'f32',
    mcOffsetCapRatioToSpeed: 'f32',
    verticalExaggeration: 'f32',
    surfaceOffsetM: 'f32',
    particleSizePixels: 'f32',
    particleOpacity: 'f32',
    trailWidthPixels: 'f32',
    trailOpacity: 'f32',
    shimmerPixelAmplitude: 'f32',
    shimmerReferenceSigmaThetaDeg: 'f32',
    birthFadeSeconds: 'f32',
    trailPersistence: 'f32',
    historySampleIntervalS: 'f32',
    maxTrailScreenJumpPx: 'f32',
    integrationMaxCellFraction: 'f32',
    spawnJitterCells: 'f32',
    gridWidth: 'i32',
    gridHeight: 'i32',
    spawnCount: 'i32',
    activeEpoch: 'i32',
    uncertaintyMode: 'i32',
    frameIndex: 'i32',
    particleCapacity: 'i32',
    historySamples: 'i32',
    historyStorageSamples: 'i32',
    historyHeadRow: 'i32',
    historyAtlasWidth: 'i32',
    historyParticleRows: 'i32',
    flatMode: 'i32',
    flatSurfaceZM: 'f32',
    gridOriginLocalM: 'vec2<f32>',
    gridAxisIM: 'vec2<f32>',
    gridAxisJM: 'vec2<f32>',
    particleColor: 'vec4<f32>',
  },
};

const FIELD_SAMPLERS = /* glsl */ `\
uniform highp sampler2D uParticleField;
uniform highp sampler2D uParticleCovariance;

bool gridInBounds(ivec2 ij) {
  return ij.x >= 0 && ij.x < horizontalParticle.gridWidth && ij.y >= 0 && ij.y < horizontalParticle.gridHeight;
}

vec2 gridToLocal(vec2 gridPosition) {
  return horizontalParticle.gridOriginLocalM +
    horizontalParticle.gridAxisIM * gridPosition.x +
    horizontalParticle.gridAxisJM * gridPosition.y;
}

vec2 localToGrid(vec2 localPosition) {
  vec2 delta = localPosition - horizontalParticle.gridOriginLocalM;
  float det = horizontalParticle.gridAxisIM.x * horizontalParticle.gridAxisJM.y -
    horizontalParticle.gridAxisJM.x * horizontalParticle.gridAxisIM.y;
  if (abs(det) < 1e-9) return vec2(-1e9);
  return vec2(
    (delta.x * horizontalParticle.gridAxisJM.y - horizontalParticle.gridAxisJM.x * delta.y) / det,
    (horizontalParticle.gridAxisIM.x * delta.y - delta.x * horizontalParticle.gridAxisIM.y) / det
  );
}

bool cellValid(ivec2 ij) {
  return gridInBounds(ij) && texelFetch(uParticleField, ij, 0).w > 0.5;
}

bool nearestField(vec2 gridPosition, out vec4 field, out vec4 covariance) {
  ivec2 nearest = ivec2(round(gridPosition));
  if (!cellValid(nearest)) return false;
  field = texelFetch(uParticleField, nearest, 0);
  covariance = texelFetch(uParticleCovariance, nearest, 0);
  return true;
}

bool sampleField(vec2 localPosition, out vec4 field, out vec4 covariance) {
  vec2 gridPosition = localToGrid(localPosition);
  if (gridPosition.x < -0.5 || gridPosition.y < -0.5 ||
      gridPosition.x > float(horizontalParticle.gridWidth) - 0.5 ||
      gridPosition.y > float(horizontalParticle.gridHeight) - 0.5) {
    return false;
  }

  ivec2 base = ivec2(floor(gridPosition));
  vec2 fraction = fract(gridPosition);
  ivec2 c00 = base;
  ivec2 c10 = base + ivec2(1, 0);
  ivec2 c01 = base + ivec2(0, 1);
  ivec2 c11 = base + ivec2(1, 1);

  // V7.2 conservative_v1 rule: only bilinear when every corner is observed;
  // otherwise a direct nearest observed RUM sample. Blankies are never field data.
  if (cellValid(c00) && cellValid(c10) && cellValid(c01) && cellValid(c11)) {
    float w00 = (1.0 - fraction.x) * (1.0 - fraction.y);
    float w10 = fraction.x * (1.0 - fraction.y);
    float w01 = (1.0 - fraction.x) * fraction.y;
    float w11 = fraction.x * fraction.y;
    field = texelFetch(uParticleField, c00, 0) * w00 +
      texelFetch(uParticleField, c10, 0) * w10 +
      texelFetch(uParticleField, c01, 0) * w01 +
      texelFetch(uParticleField, c11, 0) * w11;
    covariance = texelFetch(uParticleCovariance, c00, 0) * w00 +
      texelFetch(uParticleCovariance, c10, 0) * w10 +
      texelFetch(uParticleCovariance, c01, 0) * w01 +
      texelFetch(uParticleCovariance, c11, 0) * w11;
    field.w = 1.0;
    return true;
  }

  return nearestField(gridPosition, field, covariance);
}

// Integer-only random source. Float sin/fract hashes become correlated at the
// 10^4–10^5 seed magnitudes produced by long-running particle simulations.
uint wangHash(uint s) {
  s = (s ^ 61u) ^ (s >> 16);
  s *= 9u;
  s = s ^ (s >> 4);
  s *= 0x27d4eb2du;
  s = s ^ (s >> 15);
  return s;
}

uint hashSeed(uint base, uint salt) {
  return wangHash(base ^ wangHash(salt + 0x9e3779b9u));
}

float hashU(uint seed) {
  return float(wangHash(seed)) * (1.0 / 4294967296.0);
}

vec2 hashU2(uint seed) {
  return vec2(hashU(hashSeed(seed, 0x68bc21ebu)), hashU(hashSeed(seed, 0x02e5be93u)));
}

vec2 randomNormalPair(uint seed) {
  vec2 u = max(hashU2(seed), vec2(1e-6));
  float magnitude = sqrt(-2.0 * log(u.x));
  float angle = 6.28318530718 * u.y;
  return magnitude * vec2(cos(angle), sin(angle));
}

vec2 monteCarloVelocity(vec4 field, vec4 covariance, vec2 normalPair) {
  vec2 meanVelocity = field.xy;
  if (horizontalParticle.uncertaintyMode != 1) return meanVelocity;

  float varEast = max(0.0, covariance.x);
  float varNorth = max(0.0, covariance.y);
  float covarEN = covariance.z;
  float mid = 0.5 * (varEast + varNorth);
  float diff = 0.5 * (varEast - varNorth);
  float root = sqrt(max(0.0, diff * diff + covarEN * covarEN));
  float lambda1 = max(0.0, mid + root);
  float lambda2 = max(0.0, mid - root);
  float theta = 0.5 * atan(2.0 * covarEN, varEast - varNorth);
  vec2 z = clamp(normalPair, vec2(-horizontalParticle.mcMaxSigma), vec2(horizontalParticle.mcMaxSigma));

  vec2 perturbation = vec2(
    cos(theta) * sqrt(lambda1) * z.x - sin(theta) * sqrt(lambda2) * z.y,
    sin(theta) * sqrt(lambda1) * z.x + cos(theta) * sqrt(lambda2) * z.y
  );

  float meanSpeed = max(length(meanVelocity), 0.5);
  // Directional model: retain the mean speed signal and perturb only across it.
  vec2 normal = vec2(-meanVelocity.y, meanVelocity.x) / meanSpeed;
  float crossTrack = dot(perturbation, normal);
  perturbation = normal * crossTrack;
  perturbation *= horizontalParticle.uncertaintyStrength;

  float capMmYr = max(
    horizontalParticle.mcOffsetCapMmYr,
    meanSpeed * horizontalParticle.mcOffsetCapRatioToSpeed
  );
  float magnitude = length(perturbation);
  if (capMmYr > 0.0 && magnitude > capMmYr) {
    perturbation *= capMmYr / magnitude;
  }

  return meanVelocity + perturbation;
}
`;

const TRANSFORM_VS = /* glsl */ `\
#version 300 es
#define SHADER_NAME horizontal-particle-transform
precision highp float;
precision highp int;

const int MAX_INTEGRATION_SUBSTEPS = 24;

in vec4 stateA;
in vec4 stateB;
in vec4 stateC;

uniform highp sampler2D uParticleSpawns;

out vec4 nextStateA;
out vec4 nextStateB;
out vec4 nextStateC;

${FIELD_SAMPLERS}

bool nearestObservedSpeed(vec2 localPosition, out float speedMmYr) {
  vec2 gridPosition = localToGrid(localPosition);
  if (gridPosition.x < -0.5 || gridPosition.y < -0.5 ||
      gridPosition.x > float(horizontalParticle.gridWidth) - 0.5 ||
      gridPosition.y > float(horizontalParticle.gridHeight) - 0.5) {
    return false;
  }
  ivec2 nearest = ivec2(round(gridPosition));
  if (!cellValid(nearest)) return false;
  speedMmYr = texelFetch(uParticleCovariance, nearest, 0).w;
  return true;
}

uint respawnSeed(uint salt) {
  uint vertex = uint(gl_VertexID);
  uint frame = uint(max(horizontalParticle.frameIndex, 0));
  return hashSeed(hashSeed(vertex, frame), salt);
}

bool respawn(uint baseSeed, out vec2 position, out vec4 auxiliary) {
  vec2 lastCandidate = vec2(0.0);
  // Keep failed births invisible. Every attempt draws a fresh supported seed;
  // no fallback ever pins a particle to a RUM centre.
  for (int attempt = 0; attempt < 8; attempt += 1) {
    uint attemptSeed = hashSeed(baseSeed, uint(attempt) + 0x51633e2du);
    float count = float(max(horizontalParticle.spawnCount, 1));
    int spawnIndex = int(floor(hashU(hashSeed(attemptSeed, 0x71d8f51bu)) * count));
    spawnIndex = clamp(spawnIndex, 0, max(horizontalParticle.spawnCount - 1, 0));
    vec2 spawnGrid = texelFetch(uParticleSpawns, ivec2(spawnIndex, 0), 0).xy;
    vec2 randomOffset = (hashU2(hashSeed(attemptSeed, 0x4a7c15b9u)) - 0.5) * horizontalParticle.spawnJitterCells;
    vec2 candidate = gridToLocal(spawnGrid + randomOffset);
    lastCandidate = candidate;

    vec4 field;
    vec4 covariance;
    float lifecycleSpeed;
    if (sampleField(candidate, field, covariance) &&
        nearestObservedSpeed(candidate, lifecycleSpeed) &&
        lifecycleSpeed >= horizontalParticle.stallSpeedMmYr) {
      float life = 2.5 + hashU(hashSeed(attemptSeed, 0x2d9c5f13u)) * 4.0;
      vec2 normals = randomNormalPair(hashSeed(attemptSeed, 0x91e10da5u));
      position = candidate;
      // Births always start at age 0.0. The renderer uses this monotonically
      // increasing age for a fade-in now and a clean history break in Phase 2.
      auxiliary = vec4(0.0, life, normals.x, normals.y);
      return true;
    }
  }

  position = lastCandidate;
  // Explicit inactive sentinel: next frame retries with a new frame seed and
  // draw shaders force alpha to zero. This is never a visible centre snap.
  auxiliary = vec4(-1.0, 0.0, 0.0, 0.0);
  return false;
}

void main(void) {
  // Transform feedback ignores rasterized position, but WebGL2 still requires it.
  // Keep the no-op primitive outside clip space as a defensive fallback; the
  // transform pass itself also enables rasterizer discard below.
  gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
  vec2 current = stateA.xy;
  vec4 field;
  vec4 covariance;
  float lifecycleSpeed;
  bool valid = sampleField(current, field, covariance);
  bool lifecycleValid = nearestObservedSpeed(current, lifecycleSpeed);
  float age = stateC.x + horizontalParticle.deltaSeconds;
  bool needsRespawn = stateC.x < 0.0 || stateC.y <= 0.0 ||
    !valid || !lifecycleValid || lifecycleSpeed < horizontalParticle.stallSpeedMmYr || age > stateC.y;

  if (needsRespawn) {
    vec2 spawned;
    vec4 auxiliary;
    respawn(respawnSeed(0x19c8a443u), spawned, auxiliary);
    nextStateA = vec4(spawned, spawned);
    nextStateB = vec4(spawned, spawned);
    nextStateC = auxiliary;
    return;
  }

  // High visual-speed settings can cross multiple RUM widths in a frame.
  // Integrate in metric substeps and validate support after every substep so a
  // particle cannot silently jump over blank/no-data space. At the extreme
  // setting, a path that would need more than MAX_INTEGRATION_SUBSTEPS is
  // rejected and respawned rather than drawing a fictional crossing.
  vec2 initialVelocityMmYr = monteCarloVelocity(field, covariance, stateC.zw);
  float speedReference = max(horizontalParticle.speedP95MmYr, 1e-9);
  vec2 initialVelocityMps = initialVelocityMmYr / speedReference *
    horizontalParticle.baseMps * horizontalParticle.speedMultiplier;
  float cellScaleM = max(1.0, min(length(horizontalParticle.gridAxisIM), length(horizontalParticle.gridAxisJM)));
  float maxSubstepDistanceM = max(1.0, cellScaleM * horizontalParticle.integrationMaxCellFraction);
  int requiredSubsteps = int(ceil(length(initialVelocityMps) * horizontalParticle.deltaSeconds / maxSubstepDistanceM));
  requiredSubsteps = max(requiredSubsteps, 1);

  bool advanceValid = requiredSubsteps <= MAX_INTEGRATION_SUBSTEPS;
  vec2 advanced = current;
  if (advanceValid) {
    float substepSeconds = horizontalParticle.deltaSeconds / float(requiredSubsteps);
    for (int step = 0; step < MAX_INTEGRATION_SUBSTEPS; step += 1) {
      if (step >= requiredSubsteps) break;
      vec4 stepField;
      vec4 stepCovariance;
      float stepLifecycleSpeed;
      if (!sampleField(advanced, stepField, stepCovariance) ||
          !nearestObservedSpeed(advanced, stepLifecycleSpeed) ||
          stepLifecycleSpeed < horizontalParticle.stallSpeedMmYr) {
        advanceValid = false;
        break;
      }
      vec2 stepVelocityMmYr = monteCarloVelocity(stepField, stepCovariance, stateC.zw);
      vec2 stepVelocityMps = stepVelocityMmYr / speedReference *
        horizontalParticle.baseMps * horizontalParticle.speedMultiplier;
      vec2 candidate = advanced + stepVelocityMps * substepSeconds;
      vec4 candidateField;
      vec4 candidateCovariance;
      float candidateLifecycleSpeed;
      if (!sampleField(candidate, candidateField, candidateCovariance) ||
          !nearestObservedSpeed(candidate, candidateLifecycleSpeed) ||
          candidateLifecycleSpeed < horizontalParticle.stallSpeedMmYr) {
        advanceValid = false;
        break;
      }
      advanced = candidate;
    }
  }

  if (!advanceValid) {
    vec2 spawned;
    vec4 auxiliary;
    respawn(respawnSeed(0x41e98b77u), spawned, auxiliary);
    nextStateA = vec4(spawned, spawned);
    nextStateB = vec4(spawned, spawned);
    nextStateC = auxiliary;
    return;
  }

  nextStateA = vec4(advanced, current);
  nextStateB = vec4(stateA.zw, stateB.xy);
  nextStateC = vec4(age, stateC.yzw);
}
`;


const DRAW_HELPERS = /* glsl */ `\
${FIELD_SAMPLERS}
uniform highp sampler2D uVerticalModel;

float verticalHeightMm(float rowIndex) {
  int row = int(round(rowIndex));
  if (row < 0) return 0.0;
  return texelFetch(uVerticalModel, ivec2(horizontalParticle.activeEpoch, row), 0).r;
}

float surfaceHeightM(vec2 localPosition) {
  // Real 2D Analysis mode deliberately bypasses the animated vertical texture.
  // The horizontal particle field is still the same world-space field, but all
  // heads, ribbons and shimmer vectors live on one flat z plane.
  if (horizontalParticle.flatMode != 0) return horizontalParticle.flatSurfaceZM;

  vec2 gridPosition = localToGrid(localPosition);
  ivec2 base = ivec2(floor(gridPosition));
  vec2 fraction = fract(gridPosition);
  ivec2 c00 = base;
  ivec2 c10 = base + ivec2(1, 0);
  ivec2 c01 = base + ivec2(0, 1);
  ivec2 c11 = base + ivec2(1, 1);
  float mm = 0.0;

  if (cellValid(c00) && cellValid(c10) && cellValid(c01) && cellValid(c11)) {
    float w00 = (1.0 - fraction.x) * (1.0 - fraction.y);
    float w10 = fraction.x * (1.0 - fraction.y);
    float w01 = (1.0 - fraction.x) * fraction.y;
    float w11 = fraction.x * fraction.y;
    mm = verticalHeightMm(texelFetch(uParticleField, c00, 0).z) * w00 +
      verticalHeightMm(texelFetch(uParticleField, c10, 0).z) * w10 +
      verticalHeightMm(texelFetch(uParticleField, c01, 0).z) * w01 +
      verticalHeightMm(texelFetch(uParticleField, c11, 0).z) * w11;
  } else {
    vec4 field;
    vec4 covariance;
    if (nearestField(gridPosition, field, covariance)) mm = verticalHeightMm(field.z);
  }

  return mm * horizontalParticle.verticalExaggeration + horizontalParticle.surfaceOffsetM;
}

vec2 shimmerOffsetPixels(vec2 localPosition, vec4 state, float sampleAge) {
  if (horizontalParticle.uncertaintyMode != 2 || horizontalParticle.shimmerPixelAmplitude <= 0.0) return vec2(0.0);
  vec4 field;
  vec4 covariance;
  if (!sampleField(localPosition, field, covariance)) return vec2(0.0);
  float speed = max(length(field.xy), 0.5);
  vec2 normal = vec2(-field.y, field.x) / speed;
  float varPerp = max(0.0,
    normal.x * normal.x * max(0.0, covariance.x) +
    2.0 * normal.x * normal.y * covariance.z +
    normal.y * normal.y * max(0.0, covariance.y)
  );
  float thetaDeg = degrees(sqrt(varPerp) / speed);
  // The P75 directional 1σ reference is generated from the active field
  // metadata. At that value shimmer reaches its configured amplitude; smaller
  // and larger per-cell sigmas scale smoothly below/above the reference.
  float referenceThetaDeg = max(horizontalParticle.shimmerReferenceSigmaThetaDeg, 1e-3);
  float normalized = clamp(thetaDeg / referenceThetaDeg, 0.0, 1.0);
  float eased = normalized * normalized * (3.0 - 2.0 * normalized);
  uint shimmerSeed = hashSeed(uint(gl_InstanceID) ^ floatBitsToUint(state.z), floatBitsToUint(state.w));
  float phase = hashU(hashSeed(shimmerSeed, 0x63d83595u)) * 6.28318530718;
  float frequency = 0.35 + hashU(hashSeed(shimmerSeed, 0x8a5cd789u)) * 0.75;
  // World-space ribbons carry particle age per stored sample. Reconstruct the
  // historical phase at that sample, rather than applying one current offset
  // to the complete ribbon. This restores the V7 slithering travelling wave
  // instead of a rigid whole-snake side-to-side rock.
  float sampleTime = horizontalParticle.time - max(0.0, state.x - max(sampleAge, 0.0));
  float wobble = horizontalParticle.shimmerPixelAmplitude * horizontalParticle.uncertaintyStrength * eased *
    sin(6.28318530718 * frequency * sampleTime + phase);

  vec3 here = vec3(localPosition, surfaceHeightM(localPosition));
  vec3 stepPosition = vec3(localPosition + normal, surfaceHeightM(localPosition + normal));
  vec4 hereClip = project_position_to_clipspace(here, vec3(0.0), vec3(0.0), geometry.position);
  vec4 stepClip = project_position_to_clipspace(stepPosition, vec3(0.0), vec3(0.0), geometry.position);
  vec2 screenDirection = stepClip.xy / max(stepClip.w, 1e-6) - hereClip.xy / max(hereClip.w, 1e-6);
  float lengthDirection = length(screenDirection);
  if (lengthDirection < 1e-6) return vec2(0.0);
  return screenDirection / lengthDirection * wobble;
}

float particleBirthAlpha(vec4 state) {
  if (state.x < 0.0 || state.y <= 0.0) return 0.0;
  return smoothstep(0.0, max(horizontalParticle.birthFadeSeconds, 1e-4), state.x);
}
`;

const HEAD_VS = /* glsl */ `\
#version 300 es
#define SHADER_NAME horizontal-particle-head
precision highp float;
precision highp int;

in vec2 positions;
in vec4 stateA;
in vec4 stateB;
in vec4 stateC;
out vec4 vColor;
out vec2 vHeadUv;

${DRAW_HELPERS}

void main(void) {
  vec2 localXY = stateA.xy;
  vec3 localPosition = vec3(localXY, surfaceHeightM(localXY));
  geometry.worldPosition = localPosition;
  vec4 clip = project_position_to_clipspace(localPosition, vec3(0.0), vec3(0.0), geometry.position);

  // V7 brush contract: the head is the rounded end-cap of the same speed-
  // scaled stroke as the ribbon, never a fixed-size marker. Keep the legacy
  // 2.2px default as the neutral size-control reference.
  vec4 field;
  vec4 covariance;
  float speedQ = 0.0;
  if (sampleField(localXY, field, covariance)) {
    speedQ = max(0.0, length(field.xy) / max(horizontalParticle.speedP95MmYr, 1e-9));
  }
  float brushScale = max(0.10, horizontalParticle.particleSizePixels / 2.2);
  float brushRadius = clamp(0.70 + 0.60 * speedQ, 0.70, 1.80) *
    horizontalParticle.trailWidthPixels * brushScale;
  vec2 offset = positions * brushRadius + shimmerOffsetPixels(localXY, stateC, stateC.x);
  gl_Position = clip + vec4(project_pixel_size_to_clipspace(offset), 0.0, 0.0);

  float alphaSpeed = clamp(0.20 + 0.55 * speedQ, 0.15, 0.85);
  float birthAlpha = particleBirthAlpha(stateC);
  vHeadUv = positions;
  vColor = vec4(
    horizontalParticle.particleColor.rgb,
    horizontalParticle.particleColor.a * horizontalParticle.particleOpacity * alphaSpeed * birthAlpha
  );
}
`;

const HEAD_FS = /* glsl */ `\
#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vHeadUv;
out vec4 fragColor;

void main(void) {
  // Soft disk = rounded line cap. This removes square, oversized heads while
  // retaining a clean antialiased end to every speed-scaled brush ribbon.
  float radial = length(vHeadUv);
  float capAlpha = 1.0 - smoothstep(0.82, 1.0, radial);
  if (capAlpha <= 0.001) discard;
  fragColor = vec4(vColor.rgb, vColor.a * capAlpha);
}
`;

const APPEND_HISTORY_VS = /* glsl */ `\
#version 300 es
#define SHADER_NAME horizontal-particle-history-append
precision highp float;
precision highp int;

in vec4 stateA;
in vec4 stateC;
out vec4 vHistory;

${FIELD_SAMPLERS}

void main(void) {
  // The history texture is a tiled particle atlas, not a single ultra-wide row:
  // x = particle column; y = particle-atlas-row × ring-sample + history head.
  // The +0.5 lands rasterization at texel centres; omitting it causes silent
  // row/column drift on some WebGL implementations.
  int atlasWidth = max(horizontalParticle.historyAtlasWidth, 1);
  int particleRow = gl_VertexID / atlasWidth;
  int atlasColumn = gl_VertexID - particleRow * atlasWidth;
  int atlasRow = particleRow * max(horizontalParticle.historyStorageSamples, 1) + horizontalParticle.historyHeadRow;
  float x = ((float(atlasColumn) + 0.5) / float(atlasWidth)) * 2.0 - 1.0;
  float y = ((float(atlasRow) + 0.5) / float(max(horizontalParticle.historyParticleRows * horizontalParticle.historyStorageSamples, 1))) * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = 1.0;

  float age = stateC.x;
  float speedQ = 0.0;
  vec4 field;
  vec4 covariance;
  if (age < 0.0 || !sampleField(stateA.xy, field, covariance)) {
    // Initial texture rows and inactive particles use age=-1 sentinel. Ribbon
    // rendering rejects such samples without any CPU-side history reset.
    vHistory = vec4(0.0, 0.0, -1.0, 0.0);
    return;
  }

  speedQ = max(0.0, length(field.xy) / max(horizontalParticle.speedP95MmYr, 1e-9));
  vHistory = vec4(stateA.xy, age, speedQ);
}
`;

const APPEND_HISTORY_FS = /* glsl */ `\
#version 300 es
precision highp float;
in vec4 vHistory;
out vec4 fragColor;
void main(void) { fragColor = vHistory; }
`;

const RIBBON_VS = /* glsl */ `\
#version 300 es
#define SHADER_NAME horizontal-particle-history-ribbon
precision highp float;
precision highp int;

in vec2 ribbonVertex;
in vec4 stateA;
in vec4 stateC;
out vec4 vColor;

uniform highp sampler2D uParticleHistory;

${DRAW_HELPERS}

int historyRowFromOldest(int index) {
  // headRow is the newest written ring row. The oldest logical sample is the
  // row immediately after headRow, wrapping around the logical ring height.
  return (horizontalParticle.historyHeadRow + 1 + index) % horizontalParticle.historySamples;
}

ivec2 historyTexelForParticle(int particleIndex, int ringRow) {
  int atlasWidth = max(horizontalParticle.historyAtlasWidth, 1);
  int particleRow = particleIndex / atlasWidth;
  int atlasColumn = particleIndex - particleRow * atlasWidth;
  int atlasRow = particleRow * max(horizontalParticle.historyStorageSamples, 1) + ringRow;
  return ivec2(atlasColumn, atlasRow);
}

void hideRibbonVertex(void) {
  gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
  vColor = vec4(0.0);
}

void main(void) {
  if (horizontalParticle.historyHeadRow < 0 || horizontalParticle.historySamples < 2 || stateC.x < 0.0) {
    hideRibbonVertex();
    return;
  }

  int segmentIndex = gl_VertexID / 6;
  int lastSegment = horizontalParticle.historySamples - 2;
  if (segmentIndex < 0 || segmentIndex > lastSegment) {
    hideRibbonVertex();
    return;
  }

  int sourceRow = historyRowFromOldest(segmentIndex);
  int targetRow = historyRowFromOldest(segmentIndex + 1);
  vec4 sourceSample = texelFetch(uParticleHistory, historyTexelForParticle(gl_InstanceID, sourceRow), 0);
  vec4 targetSample = texelFetch(uParticleHistory, historyTexelForParticle(gl_InstanceID, targetRow), 0);

  // The newest stored sample is at most one append interval behind live state.
  // Replace only that endpoint with current position so the bright head stays
  // attached between fixed cadence writes.
  if (segmentIndex == lastSegment) {
    targetSample = vec4(stateA.xy, stateC.x, targetSample.w);
  }

  // Age resets are a zero-cost respawn-discontinuity marker. Hide any segment
  // that crosses a reset, contains sentinel initialization, or belongs to a
  // previous life that is newer than the currently active particle age.
  if (sourceSample.z < 0.0 || targetSample.z < 0.0 ||
      targetSample.z + 1e-5 < sourceSample.z ||
      sourceSample.z > stateC.x + 1e-5 || targetSample.z > stateC.x + 1e-5) {
    hideRibbonVertex();
    return;
  }

  vec2 source = sourceSample.xy;
  vec2 target = targetSample.xy;
  vec3 sourcePosition = vec3(source, surfaceHeightM(source));
  vec3 targetPosition = vec3(target, surfaceHeightM(target));
  geometry.worldPosition = targetPosition;

  vec4 sourceClip = project_position_to_clipspace(sourcePosition, vec3(0.0), vec3(0.0), geometry.position);
  vec4 targetClip = project_position_to_clipspace(targetPosition, vec3(0.0), vec3(0.0), geometry.position);
  float along = ribbonVertex.x;
  vec4 clip = mix(sourceClip, targetClip, along);

  vec2 direction = targetClip.xy / max(targetClip.w, 1e-6) - sourceClip.xy / max(sourceClip.w, 1e-6);
  vec2 pixelClip = abs(project_pixel_size_to_clipspace(vec2(1.0, 1.0)));
  float screenJumpPx = length(direction / max(pixelClip, vec2(1e-6)));
  if (screenJumpPx > horizontalParticle.maxTrailScreenJumpPx) {
    // V7 parity guard: do not connect a history segment that projects as a
    // cross-map lightning bolt. Adaptive sim substeps above should prevent
    // this normally; this final render guard makes the failure non-deceptive.
    hideRibbonVertex();
    return;
  }
  float directionLength = length(direction);
  vec2 normal = directionLength > 1e-6 ? vec2(-direction.y, direction.x) / directionLength : vec2(0.0, 1.0);

  // Each stored endpoint receives its own historical shimmer phase. The blend
  // produces a travelling slither along the ribbon rather than rigid rocking.
  vec2 sourceShimmer = shimmerOffsetPixels(source, stateC, sourceSample.z);
  vec2 targetShimmer = shimmerOffsetPixels(target, stateC, targetSample.z);
  vec2 shimmer = mix(sourceShimmer, targetShimmer, along);
  // Preserve V7's uncapped speed ratio. The established alpha/width formulas
  // below clamp their outputs, so high-speed RUMs read stronger without
  // unbounded geometry.
  float speedQ = max(0.0, mix(sourceSample.w, targetSample.w, along));
  float brushScale = max(0.10, horizontalParticle.particleSizePixels / 2.2);
  float strokeWidth = clamp(0.70 + 0.60 * speedQ, 0.70, 1.80) *
    horizontalParticle.trailWidthPixels * brushScale;
  vec2 offset = normal * ribbonVertex.y * strokeWidth + shimmer;
  gl_Position = clip + vec4(project_pixel_size_to_clipspace(offset), 0.0, 0.0);

  // V7 canvas trails used exponential frame persistence, not a hard cutoff.
  // Convert stored trail age to equivalent 60Hz fade steps. At p=0.98, the
  // 1.55-second tail remains visible (~15%) rather than vanishing quadratically.
  float chronological = (float(segmentIndex) + along) / float(max(horizontalParticle.historySamples - 1, 1));
  float tailAgeSeconds = (1.0 - chronological) *
    float(max(horizontalParticle.historySamples - 1, 0)) * horizontalParticle.historySampleIntervalS;
  float tailFade = pow(clamp(horizontalParticle.trailPersistence, 0.80, 0.999), tailAgeSeconds * 60.0);
  float alphaSpeed = clamp(0.20 + 0.55 * speedQ, 0.15, 0.85);
  float birthAlpha = particleBirthAlpha(stateC);
  float alpha = horizontalParticle.particleColor.a * horizontalParticle.particleOpacity *
    horizontalParticle.trailOpacity * alphaSpeed * tailFade * birthAlpha;
  vColor = vec4(horizontalParticle.particleColor.rgb, alpha);
}
`;

const DRAW_FS = /* glsl */ `\
#version 300 es
precision highp float;
in vec4 vColor;
out vec4 fragColor;
void main(void) { fragColor = vColor; }
`;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function modeCode(mode) {
  return MODE[String(mode).toLowerCase()] ?? MODE.mean;
}

function makeRandom(seed) {
  let state = (seed >>> 0) || 0x6d2b79f5;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(random) {
  const a = Math.max(random(), 1e-6);
  const b = Math.max(random(), 1e-6);
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

function createTexture(device, id, format, width, height, data, usage = Texture.SAMPLE | Texture.COPY_DST) {
  return device.createTexture({
    id,
    format,
    width,
    height,
    data,
    usage,
    sampler: {
      minFilter: 'nearest',
      magFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    },
  });
}

function makeHistoryAtlasLayout(particleCapacity, historySamples, maxTextureSize) {
  const capacity = Math.max(1, Math.round(particleCapacity));
  const samples = Math.max(1, Math.round(historySamples));
  const deviceLimit = Math.max(0, Math.floor(Number(maxTextureSize)));
  if (!Number.isFinite(deviceLimit) || deviceLimit < samples) {
    throw new Error(`Horizontal GPU ribbons require a usable MAX_TEXTURE_SIZE (at least ${samples}); this device reports ${maxTextureSize || 'unknown'}.`);
  }

  // Prefer 4096 columns so 12K particles become 3 rows × historySamples rather
  // than asking WebGL2 for a fragile 12K-wide render target. Smaller devices
  // automatically use a narrower atlas as long as the tiled height still fits.
  const atlasWidth = Math.min(4096, capacity, deviceLimit);
  const particleRows = Math.ceil(capacity / atlasWidth);
  const atlasHeight = particleRows * samples;
  if (atlasHeight > deviceLimit) {
    throw new Error(
      `Horizontal GPU ribbons need a ${atlasWidth}×${atlasHeight} RGBA32F history atlas for ${capacity.toLocaleString()} particles × ${samples} samples, but MAX_TEXTURE_SIZE is ${deviceLimit}.`,
    );
  }
  return {
    atlasWidth,
    particleRows,
    atlasHeight,
    texelCapacity: atlasWidth * particleRows,
    unusedParticleSlots: atlasWidth * particleRows - capacity,
    bytes: atlasWidth * atlasHeight * 4 * Float32Array.BYTES_PER_ELEMENT,
  };
}

function createHistoryTexture(device, id, layout) {
  const sentinelData = new Float32Array(layout.atlasWidth * layout.atlasHeight * 4);
  for (let offset = 2; offset < sentinelData.length; offset += 4) sentinelData[offset] = -1.0;
  return createTexture(
    device,
    id,
    'rgba32float',
    layout.atlasWidth,
    layout.atlasHeight,
    sentinelData,
    Texture.SAMPLE | Texture.RENDER_ATTACHMENT | Texture.COPY_DST,
  );
}

function historyConfig(render = {}, requestedSamples = null) {
  const storageSamples = clamp(Math.round(Number(render.historySamplesMax ?? MAX_HISTORY_SAMPLES)), 2, MAX_HISTORY_SAMPLES);
  const defaultSamples = clamp(Math.round(Number(render.historySamples ?? 32)), 2, storageSamples);
  const samples = clamp(Math.round(Number(requestedSamples ?? defaultSamples)), 2, storageSamples);
  const sampleIntervalS = clamp(Number(render.historySampleIntervalS ?? 0.05), 1 / 120, 0.25);
  const trailPersistence = clamp(Number(render.trailPersistence ?? 0.98), 0.80, 0.999);
  return {
    samples,
    storageSamples,
    sampleIntervalS,
    trailPersistence,
    tailFadeMode: 'canvas_persistence',
  };
}

export class HorizontalParticleLayer extends Layer {
  static layerName = 'HorizontalParticleLayer';

  static defaultProps = {
    field: {type: 'object', value: null, compare: false},
    verticalModelMm: {type: 'object', value: null, compare: false},
    epochCount: {type: 'number', value: 1},
    activeEpoch: {type: 'number', value: 0},
    verticalExaggeration: {type: 'number', value: 10},
    flatMode: {type: 'boolean', value: false},
    flatSurfaceZM: {type: 'number', value: 0},
    visible: true,
    particleCount: {type: 'number', value: 5000},
    particleCapacity: {type: 'number', value: 12000},
    mode: {type: 'string', value: 'mean'},
    speedMultiplier: {type: 'number', value: 1.5},
    particleSizeMultiplier: {type: 'number', value: 1.0},
    particleOpacity: {type: 'number', value: 1.0},
    trailPersistence: {type: 'number', value: 0.98},
    historySamples: {type: 'number', value: 32},
    uncertaintyStrength: {type: 'number', value: 0.5},
    onStatus: {type: 'function', value: null, optional: true},
  };

  getModels() {
    // DeckGL owns viewport/project uniforms for render models only. The
    // transform-feedback model is driven separately and intentionally stays
    // outside DeckGL's render-pass parameter propagation.
    const gpu = this.state?.gpu;
    if (!gpu) return [];
    return [gpu.ribbonModel, gpu.headModel].filter(Boolean);
  }

  initializeState() {
    if (!BufferTransform.isSupported(this.context.device)) {
      throw new Error('HorizontalParticleLayer requires the WebGL2 transform-feedback renderer.');
    }
    this._createGpuResources();
  }

  updateState({props, oldProps}) {
    if (props.field !== oldProps.field || props.verticalModelMm !== oldProps.verticalModelMm || props.epochCount !== oldProps.epochCount) {
      this._destroyGpuResources();
      this._createGpuResources();
      return;
    }
    if (props.historySamples !== oldProps.historySamples) {
      this._resetHistoryWindow(props.historySamples);
    }
  }

  finalizeState() {
    this._destroyGpuResources();
  }

  draw() {
    const gpu = this.state.gpu;
    if (!gpu || !this.props.visible || this.props.particleCount <= 0) return;

    const now = performance.now() * 0.001;
    const previous = gpu.lastTime ?? now;
    const deltaSeconds = clamp(now - previous, 0, MAX_DT_SECONDS);
    gpu.lastTime = now;
    const activeCount = clamp(Math.round(this.props.particleCount), 0, gpu.capacity);
    if (activeCount <= 0) return;
    gpu.frameIndex = (gpu.frameIndex + 1) & 0x7fffffff;

    const read = gpu.readIndex;
    const write = 1 - read;
    this._setUniforms({time: now, deltaSeconds, frameIndex: gpu.frameIndex});
    gpu.transform.model.setVertexCount(activeCount);
    gpu.transform.run({
      // BufferTransform still begins a normal WebGL render pass. Chrome rejects
      // a draw whose fragment shader intentionally discards while a color
      // buffer is active (GL_INVALID_OPERATION: missing fragment outputs).
      // This is a pure transform-feedback computation: disable rasterization.
      discard: true,
      inputBuffers: {
        stateA: gpu.stateA[read],
        stateB: gpu.stateB[read],
        stateC: gpu.stateC[read],
      },
      outputBuffers: {
        nextStateA: gpu.stateA[write],
        nextStateB: gpu.stateB[write],
        nextStateC: gpu.stateC[write],
      },
      // BufferTransform opens a separate luma pass. It must never clear the
      // shared DeckGL frame buffer already holding caps, basemap, or glyphs.
      clearColor: false,
      clearDepth: false,
      clearStencil: false,
      parameters: {
        blend: false,
        depthWriteEnabled: false,
        depthCompare: 'always',
        cullMode: 'none',
      },
    });
    gpu.readIndex = write;

    // Fixed simulation-time history cadence keeps trail duration invariant to
    // display FPS. MAX_DT_SECONDS protects integration and bounds this loop.
    gpu.historyAccumulator += deltaSeconds;
    if (gpu.historyAccumulator + 1e-9 >= gpu.historySampleIntervalS) {
      gpu.historyAccumulator = Math.max(0, gpu.historyAccumulator - gpu.historySampleIntervalS);
      gpu.historyHeadRow = (gpu.historyHeadRow + 1) % gpu.historySamples;
      this._setUniforms({time: now, deltaSeconds: 0, frameIndex: gpu.frameIndex});
      this._appendHistory(gpu.readIndex, activeCount);
    }

    this._bindRenderState(gpu.readIndex, activeCount);
    this._setUniforms({time: now, deltaSeconds: 0, frameIndex: gpu.frameIndex});
    if (gpu.historyHeadRow >= 0) gpu.ribbonModel.draw(this.context.renderPass);
    gpu.headModel.draw(this.context.renderPass);
    this.setNeedsRedraw();
  }

  _appendHistory(readIndex, activeCount) {
    const gpu = this.state.gpu;
    gpu.appendModel.setAttributes({
      stateA: gpu.stateA[readIndex],
      stateC: gpu.stateC[readIndex],
    });
    gpu.appendModel.setVertexCount(activeCount);

    // Dedicated color-only FBO. Every clear is explicit false; blending must
    // stay disabled or position samples silently blend with old history data.
    const pass = this.context.device.beginRenderPass({
      framebuffer: gpu.historyFramebuffer,
      clearColor: false,
      clearDepth: false,
      clearStencil: false,
      parameters: {
        blend: false,
        depthWriteEnabled: false,
        depthCompare: 'always',
        cullMode: 'none',
      },
    });
    gpu.appendModel.draw(pass);
    pass.end();
  }

  _resetHistoryWindow(requestedSamples) {
    const gpu = this.state?.gpu;
    if (!gpu || !this.props.field) return;
    const history = historyConfig(this.props.field.render ?? {}, requestedSamples);
    if (history.samples === gpu.historySamples) return;

    // Changing requested trail duration must not re-seed or restart simulation.
    // Rebuild only the sentinel-filled history target and ribbon model; state
    // buffers, field textures, and live particle positions remain intact.
    gpu.historyFramebuffer?.destroy();
    gpu.historyTexture?.destroy();
    gpu.ribbonModel?.destroy();

    const device = this.context.device;
    const historyTexture = createHistoryTexture(device, `${this.props.id}-history-rgba32f`, {
      atlasWidth: gpu.historyAtlasWidth,
      atlasHeight: gpu.historyAtlasHeight,
    });
    const historyFramebuffer = device.createFramebuffer({
      id: `${this.props.id}-history-framebuffer`,
      width: gpu.historyAtlasWidth,
      height: gpu.historyAtlasHeight,
      colorAttachments: [historyTexture],
      depthStencilAttachment: null,
    });
    const ribbonModel = this._createRibbonModel({
      id: `${this.props.id}-ribbon`,
      vertexBuffer: gpu.ribbonBuffer,
      stateA: gpu.stateA[gpu.readIndex],
      stateC: gpu.stateC[gpu.readIndex],
      bindings: {
        uParticleField: gpu.fieldTexture,
        uParticleCovariance: gpu.covarianceTexture,
        uVerticalModel: gpu.verticalTexture,
        uParticleHistory: historyTexture,
      },
      capacity: gpu.capacity,
      vertexCount: (history.samples - 1) * 6,
    });

    gpu.historyTexture = historyTexture;
    gpu.historyFramebuffer = historyFramebuffer;
    gpu.ribbonModel = ribbonModel;
    gpu.historySamples = history.samples;
    gpu.historyHeadRow = -1;
    gpu.historyAccumulator = 0;
    gpu.historyTrailPersistence = history.trailPersistence;

    this.props.onStatus?.({
      renderer: 'webgl2_transform_feedback_history_ribbon',
      capacity: gpu.capacity,
      fieldCells: this.props.field.liveRumCount,
      spawnCells: this.props.field.spawnCount,
      speedP95MmYr: this.props.field.speedP95MmYr,
      historySamples: gpu.historySamples,
      historyStorageSamples: gpu.historyStorageSamples,
      historySampleIntervalS: gpu.historySampleIntervalS,
      historyFormat: 'rgba32float',
      historyDurationS: (gpu.historySamples - 1) * gpu.historySampleIntervalS,
      trailPersistence: Number(this.props.trailPersistence ?? gpu.historyTrailPersistence),
      activeParticleCount: clamp(Math.round(this.props.particleCount ?? 0), 0, gpu.capacity),
      historyAtlasWidth: gpu.historyAtlasWidth,
      historyParticleRows: gpu.historyParticleRows,
      historyAtlasHeight: gpu.historyAtlasHeight,
      historyTextureMiB: gpu.historyTextureBytes / (1024 * 1024),
      historyAtlasUnusedParticleSlots: gpu.historyAtlasUnusedParticleSlots,
      maxTextureSize: gpu.maxTextureSize,
      float32Renderable: gpu.float32Renderable,
      maxIntegrationSubsteps: MAX_INTEGRATION_SUBSTEPS,
      maxTrailScreenJumpPx: Number(this.props.field.render?.maxTrailScreenJumpPx ?? 120),
    });
  }

  _createGpuResources() {
    const {field, verticalModelMm, epochCount, particleCapacity} = this.props;
    if (!field || !verticalModelMm) return;
    const device = this.context.device;
    const grid = field.grid;
    const fieldValues = field.fieldValues;
    const covarianceValues = field.covarianceValues;
    const spawnValues = field.spawnValues;
    if (!grid || !(fieldValues instanceof Float32Array) || !(covarianceValues instanceof Float32Array) || !(spawnValues instanceof Float32Array)) {
      throw new Error('HorizontalParticleLayer received an incomplete horizontal particle runtime payload.');
    }

    const capacity = Math.max(1, Math.round(particleCapacity));
    const history = historyConfig(field.render ?? {}, this.props.historySamples);
    const float32Renderable = Boolean(device.features?.has('float32-renderable-webgl'));
    if (!float32Renderable) {
      throw new Error('Horizontal GPU ribbons require EXT_color_buffer_float (luma feature float32-renderable-webgl). This browser/GPU cannot render the RGBA32F history texture.');
    }
    const maxTextureSize = Number(device.limits?.maxTextureDimension2D ?? 0);
    const historyAtlas = makeHistoryAtlasLayout(capacity, history.storageSamples, maxTextureSize);

    const state = this._createInitialState(field, capacity);
    const stateA = [
      device.createBuffer({id: `${this.props.id}-state-a-0`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateA}),
      device.createBuffer({id: `${this.props.id}-state-a-1`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateA}),
    ];
    const stateB = [
      device.createBuffer({id: `${this.props.id}-state-b-0`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateB}),
      device.createBuffer({id: `${this.props.id}-state-b-1`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateB}),
    ];
    const stateC = [
      device.createBuffer({id: `${this.props.id}-state-c-0`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateC}),
      device.createBuffer({id: `${this.props.id}-state-c-1`, usage: Buffer.VERTEX | Buffer.COPY_DST, data: state.stateC}),
    ];

    const fieldTexture = createTexture(device, `${this.props.id}-field`, 'rgba32float', grid.width, grid.height, fieldValues);
    const covarianceTexture = createTexture(device, `${this.props.id}-covariance`, 'rgba32float', grid.width, grid.height, covarianceValues);
    const spawnTexture = createTexture(device, `${this.props.id}-spawns`, 'rg32float', field.spawnCount, 1, spawnValues);
    const verticalTexture = createTexture(device, `${this.props.id}-vertical-model`, 'r32float', epochCount, field.runtimeRowCount, verticalModelMm);
    const historyTexture = createHistoryTexture(device, `${this.props.id}-history-rgba32f`, historyAtlas);
    const historyFramebuffer = device.createFramebuffer({
      id: `${this.props.id}-history-framebuffer`,
      width: historyAtlas.atlasWidth,
      height: historyAtlas.atlasHeight,
      colorAttachments: [historyTexture],
      depthStencilAttachment: null,
    });

    const transform = new BufferTransform(device, {
      id: `${this.props.id}-transform`,
      vs: TRANSFORM_VS,
      // No custom fragment shader: BufferTransform supplies luma's active
      // passthrough output. The actual transform pass enables rasterizer
      // discard, so this shader is never rasterized into the Deck canvas.
      modules: [HORIZONTAL_PARTICLE_UNIFORMS],
      topology: 'point-list',
      vertexCount: capacity,
      bufferLayout: [
        {name: 'stateA', stepMode: 'vertex', format: 'float32x4'},
        {name: 'stateB', stepMode: 'vertex', format: 'float32x4'},
        {name: 'stateC', stepMode: 'vertex', format: 'float32x4'},
      ],
      outputs: ['nextStateA', 'nextStateB', 'nextStateC'],
      bindings: {
        uParticleField: fieldTexture,
        uParticleCovariance: covarianceTexture,
        uParticleSpawns: spawnTexture,
      },
      parameters: {
        blend: false,
        depthWriteEnabled: false,
        depthCompare: 'always',
        cullMode: 'none',
      },
      debugShaders: 'errors',
    });

    const quadBuffer = device.createBuffer({
      id: `${this.props.id}-head-quad`,
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: new Float32Array([
        -1, -1, 1, -1, 1, 1,
        -1, -1, 1, 1, -1, 1,
      ]),
    });
    const ribbonVertices = [];
    const ribbonTemplate = [
      [0, -1], [0, 1], [1, 1],
      [0, -1], [1, 1], [1, -1],
    ];
    for (let segment = 0; segment < history.storageSamples - 1; segment += 1) {
      for (const [along, side] of ribbonTemplate) ribbonVertices.push(along, side);
    }
    const ribbonBuffer = device.createBuffer({
      id: `${this.props.id}-ribbon-geometry`,
      usage: Buffer.VERTEX | Buffer.COPY_DST,
      data: new Float32Array(ribbonVertices),
    });

    const appendModel = new Model(device, {
      id: `${this.props.id}-history-append`,
      vs: APPEND_HISTORY_VS,
      fs: APPEND_HISTORY_FS,
      modules: [HORIZONTAL_PARTICLE_UNIFORMS],
      topology: 'point-list',
      vertexCount: capacity,
      bufferLayout: [
        {name: 'stateA', stepMode: 'vertex', format: 'float32x4'},
        {name: 'stateC', stepMode: 'vertex', format: 'float32x4'},
      ],
      attributes: {stateA: stateA[0], stateC: stateC[0]},
      bindings: {uParticleField: fieldTexture, uParticleCovariance: covarianceTexture},
      parameters: {
        blend: false,
        depthWriteEnabled: false,
        depthCompare: 'always',
        cullMode: 'none',
      },
      debugShaders: 'errors',
    });

    const headModel = this._createHeadModel({
      id: `${this.props.id}-head`,
      vertexBuffer: quadBuffer,
      stateA: stateA[0],
      stateB: stateB[0],
      stateC: stateC[0],
      bindings: {uParticleField: fieldTexture, uParticleCovariance: covarianceTexture, uVerticalModel: verticalTexture},
      capacity,
    });
    const ribbonModel = this._createRibbonModel({
      id: `${this.props.id}-ribbon`,
      vertexBuffer: ribbonBuffer,
      stateA: stateA[0],
      stateC: stateC[0],
      bindings: {
        uParticleField: fieldTexture,
        uParticleCovariance: covarianceTexture,
        uVerticalModel: verticalTexture,
        uParticleHistory: historyTexture,
      },
      capacity,
      vertexCount: (history.storageSamples - 1) * 6,
    });

    this.setState({
      gpu: {
        capacity,
        readIndex: 0,
        lastTime: null,
        frameIndex: 0,
        historySamples: history.samples,
        historyStorageSamples: history.storageSamples,
        historySampleIntervalS: history.sampleIntervalS,
        historyHeadRow: -1,
        historyAccumulator: 0,
        historyTailFadeMode: history.tailFadeMode,
        historyTrailPersistence: history.trailPersistence,
        historyAtlasWidth: historyAtlas.atlasWidth,
        historyParticleRows: historyAtlas.particleRows,
        historyAtlasHeight: historyAtlas.atlasHeight,
        historyTextureBytes: historyAtlas.bytes,
        historyAtlasUnusedParticleSlots: historyAtlas.unusedParticleSlots,
        maxTextureSize,
        float32Renderable,
        stateA,
        stateB,
        stateC,
        fieldTexture,
        covarianceTexture,
        spawnTexture,
        verticalTexture,
        historyTexture,
        historyFramebuffer,
        transform,
        appendModel,
        headModel,
        ribbonModel,
        quadBuffer,
        ribbonBuffer,
      },
    });
    this.props.onStatus?.({
      renderer: 'webgl2_transform_feedback_history_ribbon',
      capacity,
      fieldCells: field.liveRumCount,
      spawnCells: field.spawnCount,
      speedP95MmYr: field.speedP95MmYr,
      historySamples: history.samples,
      historyStorageSamples: history.storageSamples,
      historySampleIntervalS: history.sampleIntervalS,
      historyFormat: 'rgba32float',
      historyDurationS: (history.samples - 1) * history.sampleIntervalS,
      trailPersistence: history.trailPersistence,
      activeParticleCount: clamp(Math.round(this.props.particleCount ?? 0), 0, capacity),
      historyAtlasWidth: historyAtlas.atlasWidth,
      historyParticleRows: historyAtlas.particleRows,
      historyAtlasHeight: historyAtlas.atlasHeight,
      historyTextureMiB: historyAtlas.bytes / (1024 * 1024),
      historyAtlasUnusedParticleSlots: historyAtlas.unusedParticleSlots,
      maxTextureSize,
      float32Renderable,
      maxIntegrationSubsteps: MAX_INTEGRATION_SUBSTEPS,
      maxTrailScreenJumpPx: Number(field.render?.maxTrailScreenJumpPx ?? 120),
    });
  }

  _createHeadModel({id, vertexBuffer, stateA, stateB, stateC, bindings, capacity}) {
    const shaders = this.getShaders({modules: [project32, HORIZONTAL_PARTICLE_UNIFORMS]});
    return new Model(this.context.device, {
      ...shaders,
      id,
      vs: HEAD_VS,
      fs: HEAD_FS,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: 6,
      instanceCount: capacity,
      bufferLayout: [
        {name: 'positions', stepMode: 'vertex', format: 'float32x2'},
        {name: 'stateA', stepMode: 'instance', format: 'float32x4'},
        {name: 'stateB', stepMode: 'instance', format: 'float32x4'},
        {name: 'stateC', stepMode: 'instance', format: 'float32x4'},
      ],
      attributes: {positions: vertexBuffer, stateA, stateB, stateC},
      bindings,
      parameters: this._transparentRenderParameters(),
      debugShaders: 'errors',
    });
  }

  _createRibbonModel({id, vertexBuffer, stateA, stateC, bindings, capacity, vertexCount}) {
    const shaders = this.getShaders({modules: [project32, HORIZONTAL_PARTICLE_UNIFORMS]});
    return new Model(this.context.device, {
      ...shaders,
      id,
      vs: RIBBON_VS,
      fs: DRAW_FS,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount,
      instanceCount: capacity,
      bufferLayout: [
        {name: 'ribbonVertex', stepMode: 'vertex', format: 'float32x2'},
        {name: 'stateA', stepMode: 'instance', format: 'float32x4'},
        {name: 'stateC', stepMode: 'instance', format: 'float32x4'},
      ],
      attributes: {ribbonVertex: vertexBuffer, stateA, stateC},
      bindings,
      parameters: this._transparentRenderParameters(),
      debugShaders: 'errors',
    });
  }

  _transparentRenderParameters() {
    return {
      depthWriteEnabled: false,
      depthCompare: 'less-equal',
      cullMode: 'none',
      blend: true,
      blendColorOperation: 'add',
      blendColorSrcFactor: 'src-alpha',
      blendColorDstFactor: 'one-minus-src-alpha',
      blendAlphaOperation: 'add',
      blendAlphaSrcFactor: 'one',
      blendAlphaDstFactor: 'one-minus-src-alpha',
    };
  }

  _createInitialState(field, capacity) {
    const random = makeRandom(6188575);
    const stateA = new Float32Array(capacity * 4);
    const stateB = new Float32Array(capacity * 4);
    const stateC = new Float32Array(capacity * 4);
    const grid = field.grid;
    const spawnCount = Math.max(1, field.spawnCount);
    const spawnJitterCells = Math.max(0, Number(field.render?.spawnJitterCells ?? 0.90));

    const gridToLocal = (i, j) => [
      grid.gridOriginLocalM[0] + grid.gridAxisIM[0] * i + grid.gridAxisJM[0] * j,
      grid.gridOriginLocalM[1] + grid.gridAxisIM[1] * i + grid.gridAxisJM[1] * j,
    ];

    for (let index = 0; index < capacity; index += 1) {
      const spawnIndex = Math.min(spawnCount - 1, Math.floor(random() * spawnCount));
      const gridI = field.spawnValues[spawnIndex * 2] + (random() - 0.5) * spawnJitterCells;
      const gridJ = field.spawnValues[spawnIndex * 2 + 1] + (random() - 0.5) * spawnJitterCells;
      const [x, y] = gridToLocal(gridI, gridJ);
      const offset = index * 4;
      stateA.set([x, y, x, y], offset);
      stateB.set([x, y, x, y], offset);
      const life = 2.5 + random() * 4.0;
      // Match runtime respawn semantics: age begins at exactly zero so births
      // fade in cleanly and later history ribbons can break at respawns.
      stateC.set([0.0, life, randomNormal(random), randomNormal(random)], offset);
    }
    return {stateA, stateB, stateC};
  }

  _bindRenderState(readIndex, activeCount) {
    const gpu = this.state.gpu;
    const buffers = {stateA: gpu.stateA[readIndex], stateB: gpu.stateB[readIndex], stateC: gpu.stateC[readIndex]};
    gpu.headModel.setAttributes(buffers);
    gpu.headModel.setInstanceCount(activeCount);
    gpu.ribbonModel.setAttributes({stateA: buffers.stateA, stateC: buffers.stateC});
    gpu.ribbonModel.setVertexCount(Math.max(0, gpu.historySamples - 1) * 6);
    gpu.ribbonModel.setInstanceCount(activeCount);
  }

  _setUniforms({time, deltaSeconds, frameIndex = 0}) {
    const gpu = this.state.gpu;
    if (!gpu) return;
    const field = this.props.field;
    const render = field.render ?? {};
    const uniformProps = {
      time,
      deltaSeconds,
      speedP95MmYr: Math.max(1e-9, Number(field.speedP95MmYr ?? 1)),
      baseMps: Number(render.baseMps ?? 1800),
      speedMultiplier: Number(this.props.speedMultiplier ?? render.speedMultiplier ?? 1.5),
      stallSpeedMmYr: Number(render.stallSpeedMmYr ?? 0.05),
      uncertaintyStrength: Number(this.props.uncertaintyStrength ?? render.uncertaintyStrength ?? 0.5),
      mcMaxSigma: Number(render.mcMaxSigma ?? 1.5),
      mcOffsetCapMmYr: Number(render.mcOffsetCapMmYr ?? 1),
      mcOffsetCapRatioToSpeed: Number(render.mcOffsetCapRatioToSpeed ?? 1),
      verticalExaggeration: Number(this.props.verticalExaggeration ?? 10),
      surfaceOffsetM: Number(render.surfaceOffsetM ?? 20),
      particleSizePixels: Number(render.particleSizePixels ?? 2.2) * clamp(Number(this.props.particleSizeMultiplier ?? render.particleSizeMultiplier ?? 1.0), 0.1, 8),
      particleOpacity: clamp(Number(this.props.particleOpacity ?? render.particleOpacity ?? 1.0), 0, 1),
      trailWidthPixels: Number(render.trailWidthPixels ?? 1.15),
      trailOpacity: Number(render.trailOpacity ?? 1.0),
      shimmerPixelAmplitude: Number(render.shimmerPixelAmplitude ?? 5),
      shimmerReferenceSigmaThetaDeg: Math.max(1e-3, Number(field.legend?.directionalUncertainty?.sigmaThetaP75Deg ?? 25)),
      birthFadeSeconds: Number(render.birthFadeSeconds ?? 0.3),
      trailPersistence: clamp(Number(this.props.trailPersistence ?? gpu.historyTrailPersistence), 0.50, 0.999),
      historySampleIntervalS: gpu.historySampleIntervalS,
      maxTrailScreenJumpPx: Math.max(1, Number(render.maxTrailScreenJumpPx ?? 120)),
      integrationMaxCellFraction: clamp(Number(render.integrationMaxCellFraction ?? 0.25), 0.05, 1.0),
      spawnJitterCells: Math.max(0, Number(render.spawnJitterCells ?? 0.90)),
      gridWidth: field.grid.width,
      gridHeight: field.grid.height,
      spawnCount: field.spawnCount,
      activeEpoch: clamp(Math.round(this.props.activeEpoch ?? 0), 0, Math.max(0, this.props.epochCount - 1)),
      uncertaintyMode: modeCode(this.props.mode),
      frameIndex,
      particleCapacity: gpu.capacity,
      historySamples: gpu.historySamples,
      historyStorageSamples: gpu.historyStorageSamples,
      historyHeadRow: gpu.historyHeadRow,
      historyAtlasWidth: gpu.historyAtlasWidth,
      historyParticleRows: gpu.historyParticleRows,
      flatMode: this.props.flatMode ? 1 : 0,
      flatSurfaceZM: Number(this.props.flatSurfaceZM ?? 0),
      gridOriginLocalM: field.grid.gridOriginLocalM,
      gridAxisIM: field.grid.gridAxisIM,
      gridAxisJM: field.grid.gridAxisJM,
      particleColor: (render.colorRgba ?? [100, 100, 100, 235]).map((value) => Number(value) / 255),
    };

    gpu.transform.model.shaderInputs.setProps({horizontalParticle: uniformProps});
    gpu.appendModel.shaderInputs.setProps({horizontalParticle: uniformProps});
    gpu.headModel.shaderInputs.setProps({horizontalParticle: uniformProps});
    gpu.ribbonModel.shaderInputs.setProps({horizontalParticle: uniformProps});
  }

  _destroyGpuResources() {
    const gpu = this.state?.gpu;
    if (!gpu) return;
    gpu.transform?.destroy();
    gpu.appendModel?.destroy();
    gpu.headModel?.destroy();
    gpu.ribbonModel?.destroy();
    for (const buffer of [...(gpu.stateA ?? []), ...(gpu.stateB ?? []), ...(gpu.stateC ?? [])]) buffer.destroy();
    gpu.quadBuffer?.destroy();
    gpu.ribbonBuffer?.destroy();
    gpu.fieldTexture?.destroy();
    gpu.covarianceTexture?.destroy();
    gpu.spawnTexture?.destroy();
    gpu.verticalTexture?.destroy();
    gpu.historyFramebuffer?.destroy();
    gpu.historyTexture?.destroy();
    this.setState({gpu: null});
  }
}
