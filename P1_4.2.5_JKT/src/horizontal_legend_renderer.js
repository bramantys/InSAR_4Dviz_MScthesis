// V7.2-style permanent horizontal legend renderers.
//
// The legend samples are deliberately small, but they are not decorative:
// - static glyphs use the same P75/reference scaling as the real arrows and
//   1σ uncertainty ellipses;
// - dynamic particles are one persistent mini simulation. Individual particles
//   travel through the complete capsule, then acquire the active uncertainty
//   treatment only after crossing the divider.
//
// This keeps the HUD legend a visual mirror of the live DeckGL controls without
// adding a second data path or a second field calculation.

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function smoothstep(edge0, edge1, value) {
  const span = Math.max(1e-6, edge1 - edge0);
  const t = clamp((value - edge0) / span, 0, 1);
  return t * t * (3 - 2 * t);
}

function rgba(color, fallback) {
  const values = Array.isArray(color) ? color : fallback;
  const [r, g, b, a = 255] = values;
  return `rgba(${Math.round(finite(r, fallback[0]))}, ${Math.round(finite(g, fallback[1]))}, ${Math.round(finite(b, fallback[2]))}, ${clamp(finite(a, fallback[3]) / 255, 0, 1).toFixed(3)})`;
}

function configureCanvas(canvas, {clear = false} = {}) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const ratio = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  const targetWidth = Math.round(width * ratio);
  const targetHeight = Math.round(height * ratio);
  const resized = canvas.width !== targetWidth || canvas.height !== targetHeight;
  if (resized) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const context = canvas.getContext('2d');
  if (!context) return null;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (clear || resized) context.clearRect(0, 0, width, height);
  return {context, width, height, resized};
}

function hash32(seed) {
  let value = seed >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}

function hashUnit(seed) {
  return hash32(seed) / 4294967296;
}

function gaussianFromSeed(seed) {
  const u = Math.max(hashUnit(seed ^ 0x68bc21eb), 1e-6);
  const v = hashUnit(seed ^ 0x02e5be93);
  return clamp(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v), -3, 3);
}

function drawReferenceSupport(context, x0, x1, y, height, alpha = 1) {
  const halfHeight = height * 0.38;
  context.save();
  context.globalAlpha = alpha;
  context.strokeStyle = 'rgba(245, 247, 249, 0.76)';
  context.lineWidth = 1.3;
  context.lineCap = 'round';
  context.beginPath();
  context.moveTo(x0, y - halfHeight);
  context.lineTo(x0, y + halfHeight);
  context.moveTo(x1, y - halfHeight);
  context.lineTo(x1, y + halfHeight);
  context.stroke();
  context.globalAlpha = alpha * 0.42;
  context.beginPath();
  context.moveTo(x0, y + halfHeight);
  context.lineTo(x1, y + halfHeight);
  context.stroke();
  context.restore();
}

function drawArrow(context, x0, x1, y, color, alpha = 1) {
  const head = clamp((x1 - x0) * 0.24, 4, 7);
  context.save();
  context.globalAlpha = alpha;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  // The thin pale under-stroke keeps the real dark Deck arrow readable on the
  // dark HUD surface without changing its actual data colour.
  context.strokeStyle = 'rgba(246, 246, 246, 0.66)';
  context.lineWidth = 3.25;
  context.beginPath();
  context.moveTo(x0, y);
  context.lineTo(x1, y);
  context.stroke();
  context.strokeStyle = color;
  context.lineWidth = 1.65;
  context.beginPath();
  context.moveTo(x0, y);
  context.lineTo(x1, y);
  context.stroke();
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(x1 + 1, y);
  context.lineTo(x1 - head, y - head * 0.58);
  context.lineTo(x1 - head, y + head * 0.58);
  context.closePath();
  context.fill();
  context.restore();
}

function drawEllipse(context, x, y, rx, ry, color, alpha = 1) {
  context.save();
  context.globalAlpha = alpha;
  context.strokeStyle = 'rgba(247, 247, 247, 0.42)';
  context.lineWidth = 3.2;
  context.beginPath();
  context.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  context.stroke();
  context.strokeStyle = color;
  context.lineWidth = 1.45;
  context.beginPath();
  context.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  context.stroke();
  context.fillStyle = color;
  context.globalAlpha = alpha * 0.85;
  context.beginPath();
  context.arc(x, y, 1.2, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawStaticLegend(canvas, state) {
  const surface = configureCanvas(canvas, {clear: true});
  if (!surface) return;
  const {context, width, height} = surface;
  const midpoint = width * 0.5;
  const y = height * 0.5;
  const active = Boolean(state?.active);
  const glyphOpacity = clamp(finite(state?.glyphOpacity, 1), 0, 1);
  const glyphScale = clamp(finite(state?.glyphScale, 1), 0.5, 2.5);
  const arrowActive = active && Boolean(state?.arrowActive ?? true);
  const ellipseActive = active && Boolean(state?.ellipseActive ?? true);
  const arrowAlpha = glyphOpacity * (arrowActive ? 1 : 0.22);
  const ellipseAlpha = glyphOpacity * (ellipseActive ? 1 : 0.22);
  const arrowColor = rgba(state?.arrowColorRgba, [34, 34, 34, 240]);
  const ellipseColor = rgba(state?.ellipseColorRgba, [0, 240, 216, 210]);
  const speedRatio = clamp(
    finite(state?.speedP75MmYr) / Math.max(1e-6, finite(state?.arrowReferenceMmYr, 1)),
    0.18,
    1,
  );
  const majorRatio = clamp(
    finite(state?.ellipseMajorP75MmYr) / Math.max(1e-6, finite(state?.ellipseMajorReferenceMmYr, 1)),
    0.18,
    1,
  );
  const minorRatio = clamp(
    finite(state?.ellipseMinorP75MmYr) / Math.max(1e-6, finite(state?.ellipseMajorP75MmYr, 1)),
    0.25,
    1,
  );
  const ellipseConfidenceVisualScale = clamp(
    finite(state?.ellipseConfidenceVisualScale ?? state?.ellipseVisualScale, 1),
    0.15,
    2.5,
  );
  const visualSpeedRatio = clamp(speedRatio * glyphScale, 0.08, 1.45);
  const visualMajorRatio = clamp(majorRatio * glyphScale * ellipseConfidenceVisualScale, 0.08, 1.45);

  // V7.2-style mini RUM support brackets. The text occupies the outer part of
  // each half, so the reference geometry remains visible rather than becoming
  // an icon detached from its value.
  const left0 = 12;
  const left1 = Math.min(midpoint - 84, left0 + Math.max(48, midpoint * 0.39));
  const right0 = midpoint + 12;
  const right1 = Math.min(width - 84, right0 + Math.max(48, midpoint * 0.39));
  const boxHeight = height * 0.82;

  drawReferenceSupport(context, left0, left1, y, boxHeight, arrowAlpha);
  const arrowAnchor = 0.5 * (left0 + left1);
  const arrowLength = Math.max(7, Math.min((left1 - arrowAnchor) - 4, (left1 - left0) * 0.46 * visualSpeedRatio));
  drawArrow(
    context,
    arrowAnchor,
    Math.min(left1 - 4, arrowAnchor + arrowLength),
    y,
    arrowColor,
    arrowAlpha,
  );

  drawReferenceSupport(context, right0, right1, y, boxHeight, ellipseAlpha);
  const radiusX = Math.max(3.8, Math.min((right1 - right0) * 0.50, (right1 - right0) * 0.34 * visualMajorRatio));
  const radiusY = Math.max(2.4, Math.min(boxHeight * 0.36, radiusX * minorRatio));
  drawEllipse(context, 0.5 * (right0 + right1), y, radiusX, radiusY, ellipseColor, ellipseAlpha);
}

function targetParticleCount(particleCount) {
  // The HUD does not need thousands of instances, but density must visibly
  // track the live particle-count control. 5,000 real particles maps to ~20
  // legend particles; the range stays legible in a 31 px capsule.
  return clamp(Math.round(finite(particleCount, 5000) / 250), 6, 32);
}

function makeLegendParticle(width, height, index, generation = 0, randomX = true) {
  const seed = hash32((index + 1) * 0x9e3779b9 ^ (generation + 1) * 0x85ebca6b);
  const verticalPadding = Math.max(3, Math.min(5, height * 0.14));
  const x = randomX
    ? hashUnit(seed ^ 0x4a7c15b9) * Math.max(1, width)
    : -10 - hashUnit(seed ^ 0x71d8f51b) * Math.max(12, width * 0.18);
  return {
    x,
    yBase: verticalPadding + hashUnit(seed ^ 0x51633e2d) * Math.max(1, height - verticalPadding * 2),
    phase: hashUnit(seed ^ 0x63d83595) * Math.PI * 2,
    frequency: 0.35 + hashUnit(seed ^ 0x8a5cd789) * 0.75,
    mcSample: gaussianFromSeed(seed ^ 0x91e10da5),
    generation,
    previousX: null,
    previousY: null,
  };
}

function resetParticleRuntime(runtimeState, width, height) {
  runtimeState.particles = [];
  runtimeState.lastTimestamp = 0;
  runtimeState.width = width;
  runtimeState.height = height;
  runtimeState.pendingReset = false;
  runtimeState.needsClear = true;
}

function ensureParticlePopulation(runtimeState, width, height, count) {
  const target = targetParticleCount(count);
  while (runtimeState.particles.length < target) {
    const index = runtimeState.particles.length;
    runtimeState.particles.push(makeLegendParticle(width, height, index, 0, true));
  }
  while (runtimeState.particles.length > target) runtimeState.particles.pop();
}

function respawnLegendParticle(runtimeState, index, width, height) {
  const current = runtimeState.particles[index];
  runtimeState.particles[index] = makeLegendParticle(
    width,
    height,
    index,
    Math.max(0, Number(current?.generation ?? 0)) + 1,
    false,
  );
  return runtimeState.particles[index];
}

function particleYAt(particle, x, state, timeMs, width, height) {
  const midpoint = width * 0.5;
  if (x <= midpoint) return particle.yBase;

  const progress = smoothstep(midpoint, midpoint + Math.max(12, width * 0.16), x);
  const mode = String(state?.mode ?? 'mean').toLowerCase();
  const strength = clamp(finite(state?.uncertaintyStrength, 0), 0, 2);

  if (mode === 'shimmer' && strength > 0) {
    // Mirrors the live shader’s P75 reference case: sigma(theta) / reference
    // is 1 here, so the configured pixel amplitude is reached after the soft
    // divider ramp. Phase and frequency are stable per particle life.
    const amplitude = clamp(
      finite(state?.shimmerPixelAmplitude, 5) * strength,
      0,
      Math.max(0, height * 0.36),
    );
    const wave = Math.sin(Math.PI * 2 * particle.frequency * (timeMs / 1000) + particle.phase);
    return particle.yBase + wave * amplitude * progress;
  }

  if (mode === 'montecarlo' && strength > 0) {
    // Directional Monte Carlo parity: the real shader samples covariance,
    // projects the perturbation cross-track, and retains the mean-speed signal.
    // At P75 we can show that as one deterministic sampled angle per particle
    // life. The clamp matches the runtime’s default mcMaxSigma guard.
    const sigmaThetaRad = Math.max(0, finite(state?.sigmaThetaP75Deg, 0)) * Math.PI / 180;
    const maxSigma = Math.max(0.1, finite(state?.mcMaxSigma, 1.5));
    const angle = clamp(particle.mcSample, -maxSigma, maxSigma) * sigmaThetaRad * strength;
    const safeAngle = clamp(angle, -0.95, 0.95);
    // The HUD is far wider than it is tall, so a modest screen-aspect
    // compression keeps the sampled fan inside the capsule. The sign,
    // ordering, P75 sigma(theta), and deterministic per-life samples remain
    // the same as the directional Monte Carlo model.
    return particle.yBase + Math.tan(safeAngle) * (x - midpoint) * 0.32;
  }

  return particle.yBase;
}

function fadeParticleCanvas(context, width, height, deltaSeconds, persistence, trailDurationSeconds) {
  const retainedPerFrame = clamp(finite(persistence, 0.98), 0.50, 0.999);
  const duration = clamp(finite(trailDurationSeconds, 1.55), 0.40, 3.20);
  // The real ribbon uses the same exponential persistence over its requested
  // stored duration. Re-scale frame fading so the separate duration slider has
  // a visible effect in this persistent 2D HUD canvas as well.
  const equivalentFrames = Math.max(0.2, deltaSeconds * 60 * (1.55 / duration));
  const eraseAlpha = clamp(1 - Math.pow(retainedPerFrame, equivalentFrames), 0.001, 0.92);
  context.save();
  context.globalCompositeOperation = 'destination-out';
  context.fillStyle = `rgba(0, 0, 0, ${eraseAlpha.toFixed(4)})`;
  context.fillRect(0, 0, width, height);
  context.restore();
}

function drawParticleSegment(context, x0, y0, x1, y1, color, lineWidth, alpha) {
  if (!Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(x1) || !Number.isFinite(y1)) return;
  context.save();
  context.lineCap = 'round';
  context.lineJoin = 'round';
  // A light HUD halo preserves readability against the dark capsule while the
  // narrower inner stroke retains the live renderer’s configured particle tint.
  context.globalAlpha = alpha * 0.36;
  context.strokeStyle = 'rgba(238, 243, 246, 1)';
  context.lineWidth = lineWidth + 1.05;
  context.beginPath();
  context.moveTo(x0, y0);
  context.lineTo(x1, y1);
  context.stroke();
  context.globalAlpha = alpha;
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.beginPath();
  context.moveTo(x0, y0);
  context.lineTo(x1, y1);
  context.stroke();
  context.restore();
}

function drawParticleHead(context, x, y, color, radius, alpha) {
  context.save();
  context.globalAlpha = alpha;
  context.fillStyle = color;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function drawParticleReference(canvas, state, timeMs, runtimeState) {
  const surface = configureCanvas(canvas, {clear: false});
  if (!surface) return;
  const {context, width, height, resized} = surface;
  if (resized || runtimeState.width !== width || runtimeState.height !== height || runtimeState.pendingReset) {
    resetParticleRuntime(runtimeState, width, height);
  }

  const active = Boolean(state?.active);
  if (!active) {
    context.clearRect(0, 0, width, height);
    // Re-enabling particles starts a fresh miniature population rather than
    // reviving a stale half-faded trail from the previous enabled state.
    runtimeState.particles = [];
    runtimeState.pendingReset = true;
    runtimeState.needsClear = false;
    runtimeState.lastTimestamp = timeMs;
    return;
  }

  if (runtimeState.needsClear) {
    context.clearRect(0, 0, width, height);
    runtimeState.needsClear = false;
  }

  if (!runtimeState.lastTimestamp) runtimeState.lastTimestamp = timeMs;
  const deltaSeconds = clamp((timeMs - runtimeState.lastTimestamp) / 1000, 0, 0.05);
  runtimeState.lastTimestamp = timeMs;

  ensureParticlePopulation(runtimeState, width, height, state?.particleCount);
  fadeParticleCanvas(
    context,
    width,
    height,
    deltaSeconds,
    state?.trailPersistence,
    state?.trailDurationSeconds,
  );

  const speed = Math.max(0, finite(state?.speedP75MmYr));
  const p95 = Math.max(1e-6, finite(state?.speedP95MmYr, speed || 1));
  const speedRatio = clamp(speed / p95, 0.15, 1.45);
  const pxPerSecond = (42 + 58 * speedRatio) * clamp(finite(state?.speedMultiplier, 1), 0.10, 6);
  const particleOpacity = clamp(finite(state?.particleOpacity, 1), 0, 1);
  const lineWidth = clamp(0.72 + 0.58 * speedRatio, 0.72, 1.80) * clamp(finite(state?.particleSizeMultiplier, 1), 0.5, 3);
  const strokeAlpha = (0.30 + 0.42 * speedRatio) * particleOpacity;
  const color = rgba(state?.particleColorRgba, [155, 207, 241, 255]);

  for (let index = 0; index < runtimeState.particles.length; index += 1) {
    let particle = runtimeState.particles[index];
    const previousX = particle.x;
    const previousY = particle.previousX === null
      ? particleYAt(particle, previousX, state, timeMs - deltaSeconds * 1000, width, height)
      : particle.previousY;

    particle.x += pxPerSecond * deltaSeconds;
    if (particle.x > width + 14) {
      particle = respawnLegendParticle(runtimeState, index, width, height);
      continue;
    }

    const currentY = particleYAt(particle, particle.x, state, timeMs, width, height);
    if (particle.previousX !== null && Math.hypot(particle.x - previousX, currentY - previousY) < width * 0.45) {
      drawParticleSegment(context, previousX, previousY, particle.x, currentY, color, lineWidth, strokeAlpha);
    }
    drawParticleHead(context, particle.x, currentY, color, Math.max(0.8, lineWidth * 0.72), strokeAlpha * 0.95);
    particle.previousX = particle.x;
    particle.previousY = currentY;
  }
}

export function createHorizontalLegendRenderer({glyphCanvas, particleCanvas} = {}) {
  let state = {
    glyph: {},
    particle: {},
  };
  let animationFrame = null;
  let started = false;
  const particleRuntime = {
    particles: [],
    lastTimestamp: 0,
    width: 0,
    height: 0,
    pendingReset: true,
    needsClear: true,
    resetKey: null,
  };

  const render = (timeMs) => {
    drawStaticLegend(glyphCanvas, state.glyph);
    drawParticleReference(particleCanvas, state.particle, timeMs, particleRuntime);
    animationFrame = window.requestAnimationFrame(render);
  };

  return {
    start() {
      if (started) return;
      started = true;
      animationFrame = window.requestAnimationFrame(render);
    },
    setState(next = {}) {
      const nextParticle = {...state.particle, ...(next.particle ?? {})};
      const resetKey = String(nextParticle.resetKey ?? 'default');
      if (resetKey !== particleRuntime.resetKey) {
        particleRuntime.resetKey = resetKey;
        particleRuntime.pendingReset = true;
      }
      state = {
        glyph: {...state.glyph, ...(next.glyph ?? {})},
        particle: nextParticle,
      };
    },
    redraw() {
      drawStaticLegend(glyphCanvas, state.glyph);
      drawParticleReference(particleCanvas, state.particle, performance.now(), particleRuntime);
    },
    destroy() {
      if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
      animationFrame = null;
      started = false;
      particleRuntime.particles = [];
      particleRuntime.pendingReset = true;
    },
  };
}
