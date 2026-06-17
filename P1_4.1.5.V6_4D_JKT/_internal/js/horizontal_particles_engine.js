// ============================================================
// HORIZONTAL PARTICLES ENGINE — V4.4 Monte Carlo selected-RUM validation lab
// ============================================================
// Extracted from the main viewer HTML to reduce UI/code coupling.
// This is intentionally a classic browser script, not an ES module,
// so it shares the viewer's global lexical scope and keeps the existing
// viewer behaviour unchanged.
//
// Render modes:
//   canvas           = old screen-space canvas trails, no true depth test
//   primitive_points = dormant depth-tested point renderer kept as fallback/checkpoint
//
// V4.2 focus:
//   uncertainty_mode = off | shimmer | montecarlo
//   shimmer    = old screen-space shalalala wobble, render-only
//   montecarlo = path-level velocity realization from east/north covariance
//
// Primitive trails are intentionally not included in this V4 engine.
// V4.3 adds Monte Carlo tuning: raw/full vs directional/capped realizations, scale/cap console controls, and clearer diagnostics.
// V4.4 adds selected-RUM Monte Carlo sampling diagnostics so visual spaghetti can be checked against the input covariance.

// ============================================================
// HORIZONTAL PARTICLE PROTOTYPE
// ============================================================

let hCtx = hParticleCanvas.getContext("2d");
let hField = null;
let hCells = [];
let hValidSpawnCells = [];
let hLookup = new Map();
let hCellByRumId = new Map();
let hAffine = null;
let hParticles = [];
let hParticlesVisible = H_PARTICLES_ENABLED_INITIAL;
let hAnimationId = null;
let hLastTimestamp = 0;
let hSpeedP95 = 2.0;
let hCameraMoving = false;
let hCameraStableTimer = null;
let hPointerDown = false;
let hCesiumCameraMoving = false;

// CPU-side vertical surface cache for particle projection height.
// The GPU shaders still drive the real cap/wall geometry.
// This cache only lets the canvas particles sit above the current RUM surface.
let hHeightMeta = null;
let hHeightImageCanvas = null;
let hHeightImageCtx = null;
let hCurrentDispByRow = null;
let hCurrentSigmaByRow = null;
let hCurrentDispEpoch = -1;
let hHeightTextureReady = false;
let hDispEpochCache = null;
let hSigmaEpochCache = null;
let hSurfaceCacheMode = "none";
let hSurfaceCacheBuildMs = 0;

// Packed vertical series from Step 06.
// Clean contract:
//   measurement_mm = trendline / popup measurement series
//   model_mm       = RUM height / texture/model series
//   sigma_mm       = uncertainty series
let packedSeriesData = null;
let packedSeriesArrays = null;
let packedSeriesEpochCount = 0;
let packedSeriesRumCount = 0;
let packedMeasurementAvailable = false;

// V4.1 uncertainty mode contract. The renderer is canvas-first; uncertainty is
// a separate simulation/render choice so the two viewer files can share this
// same engine.
//   off        : mean field only
//   shimmer    : old render-only screen-space wobble
//   montecarlo : path-level sampled velocity realization from covariance
// Do not redeclare this with let here: the viewer HTML owns the global so
// each viewer file can choose its default mode before loading this engine.
if (typeof hParticleUncertaintyMode === "undefined") {
    hParticleUncertaintyMode = (typeof H_PARTICLE_DEFAULT_UNCERTAINTY_MODE !== "undefined")
        ? normalizeHParticleUncertaintyMode(H_PARTICLE_DEFAULT_UNCERTAINTY_MODE)
        : "shimmer";
} else {
    hParticleUncertaintyMode = normalizeHParticleUncertaintyMode(hParticleUncertaintyMode);
}

// V4.2 Monte Carlo diagnostics/reproducibility state.
// The default remains random so the viewer feels alive. Calling
// setHParticleMonteCarloSeed(seed) switches the particle system into a
// deterministic debug/screenshot mode and reinitializes the particle population.
let hMonteCarloSeedMode = "random";
let hMonteCarloSeed = null;
let hMonteCarloRngState = 0;
let hMonteCarloSeedApplyCount = 0;

// V4.3 MC tuning.
// "full" is the raw 2D covariance realization. It is scientifically direct, but can
// look chaotic when covariance is larger than the mean velocity.
// "directional" keeps the mean along-flow motion and applies only the covariance
// component perpendicular to the mean path. This is the readable/default thesis view.
// "capped_full" keeps the full covariance direction but caps extreme offsets.
let hMonteCarloModel = normalizeHParticleMonteCarloModel(
    (typeof H_PARTICLE_MONTE_CARLO_MODEL !== "undefined") ? H_PARTICLE_MONTE_CARLO_MODEL : "directional"
);
let hMonteCarloMaxSigma = Number.isFinite(Number(typeof H_PARTICLE_MONTE_CARLO_MAX_SIGMA !== "undefined" ? H_PARTICLE_MONTE_CARLO_MAX_SIGMA : NaN))
    ? Math.max(0.1, Number(H_PARTICLE_MONTE_CARLO_MAX_SIGMA))
    : 1.5;
let hMonteCarloOffsetCapMmYr = Number.isFinite(Number(typeof H_PARTICLE_MONTE_CARLO_OFFSET_CAP_MM_YR !== "undefined" ? H_PARTICLE_MONTE_CARLO_OFFSET_CAP_MM_YR : NaN))
    ? Math.max(0.0, Number(H_PARTICLE_MONTE_CARLO_OFFSET_CAP_MM_YR))
    : 1.0;
let hMonteCarloOffsetCapRatioToSpeed = Number.isFinite(Number(typeof H_PARTICLE_MONTE_CARLO_OFFSET_CAP_RATIO_TO_SPEED !== "undefined" ? H_PARTICLE_MONTE_CARLO_OFFSET_CAP_RATIO_TO_SPEED : NaN))
    ? Math.max(0.0, Number(H_PARTICLE_MONTE_CARLO_OFFSET_CAP_RATIO_TO_SPEED))
    : 1.0;
let hMcFrameStats = null;
let hMcLastFrameSummary = null;
let hMcLifetimeStats = null;

function resetHMonteCarloStats() {
    hMcLastFrameSummary = null;
    hMcLifetimeStats = {
        frames: 0,
        samples: 0,
        respawns: 0,
        invalidCovariance: 0,
        psdClamps: 0,
        cappedOffsets: 0,
        maxOffsetSpeedMmYr: 0.0,
        sumOffsetSpeedMmYr: 0.0,
        maxRealizedSpeedMmYr: 0.0,
        sumRealizedSpeedMmYr: 0.0
    };
}
resetHMonteCarloStats();

function beginHMonteCarloFrameStats() {
    hMcFrameStats = {
        samples: 0,
        respawns: 0,
        invalidCovariance: 0,
        psdClamps: 0,
        cappedOffsets: 0,
        maxOffsetSpeedMmYr: 0.0,
        sumOffsetSpeedMmYr: 0.0,
        maxRealizedSpeedMmYr: 0.0,
        sumRealizedSpeedMmYr: 0.0
    };
}

function finishHMonteCarloFrameStats() {
    if (!hMcFrameStats) return;
    const s = hMcFrameStats;
    const meanOffset = s.samples > 0 ? s.sumOffsetSpeedMmYr / s.samples : 0.0;
    const meanRealized = s.samples > 0 ? s.sumRealizedSpeedMmYr / s.samples : 0.0;
    hMcLastFrameSummary = {
        samples: s.samples,
        respawns: s.respawns,
        invalidCovariance: s.invalidCovariance,
        psdClamps: s.psdClamps,
        cappedOffsets: s.cappedOffsets,
        meanOffsetSpeedMmYr: Number(meanOffset.toFixed(4)),
        maxOffsetSpeedMmYr: Number(s.maxOffsetSpeedMmYr.toFixed(4)),
        meanRealizedSpeedMmYr: Number(meanRealized.toFixed(4)),
        maxRealizedSpeedMmYr: Number(s.maxRealizedSpeedMmYr.toFixed(4))
    };

    if (hMcLifetimeStats) {
        hMcLifetimeStats.frames += 1;
        hMcLifetimeStats.samples += s.samples;
        hMcLifetimeStats.respawns += s.respawns;
        hMcLifetimeStats.invalidCovariance += s.invalidCovariance;
        hMcLifetimeStats.psdClamps += s.psdClamps;
        hMcLifetimeStats.cappedOffsets += s.cappedOffsets;
        hMcLifetimeStats.maxOffsetSpeedMmYr = Math.max(hMcLifetimeStats.maxOffsetSpeedMmYr, s.maxOffsetSpeedMmYr);
        hMcLifetimeStats.sumOffsetSpeedMmYr += s.sumOffsetSpeedMmYr;
        hMcLifetimeStats.maxRealizedSpeedMmYr = Math.max(hMcLifetimeStats.maxRealizedSpeedMmYr, s.maxRealizedSpeedMmYr);
        hMcLifetimeStats.sumRealizedSpeedMmYr += s.sumRealizedSpeedMmYr;
    }
    hMcFrameStats = null;
}

function normalizeHParticleMonteCarloModel(model) {
    const m = String(model || "").trim().toLowerCase();
    if (m === "full" || m === "raw" || m === "raw_full" || m === "2d" || m === "covariance") return "full";
    if (m === "capped" || m === "capped_full" || m === "full_capped") return "capped_full";
    if (m === "direction" || m === "directional" || m === "perp" || m === "perpendicular" || m === "readable") return "directional";
    return "directional";
}

function clampHMcZ(value) {
    const z = Number(value);
    if (!Number.isFinite(z)) return 0.0;
    const lim = Math.max(0.1, Number(hMonteCarloMaxSigma || 1.5));
    return Math.max(-lim, Math.min(lim, z));
}

function currentHMonteCarloTuning() {
    return {
        model: hMonteCarloModel,
        strength: hUncertaintyStrength,
        maxSigma: hMonteCarloMaxSigma,
        offsetCapMmYr: hMonteCarloOffsetCapMmYr,
        offsetCapRatioToSpeed: hMonteCarloOffsetCapRatioToSpeed,
        interpretation: hMonteCarloModel === "full"
            ? "raw 2D covariance realization; can look messy when uncertainty exceeds mean flow"
            : hMonteCarloModel === "capped_full"
                ? "full covariance direction with capped extreme offsets"
                : "directional/perpendicular realization around the mean flow; readable default"
    };
}

function setHParticleMonteCarloScale(value) {
    const n = Number(value);
    if (Number.isFinite(n)) {
        hUncertaintyStrength = Math.max(0.0, Math.min(2.0, n));
        if (hUncertaintyStrengthSlider) hUncertaintyStrengthSlider.value = String(Math.max(0.0, Math.min(1.0, hUncertaintyStrength)));
        if (hUncertaintyStrengthValue) hUncertaintyStrengthValue.textContent = hUncertaintyStrength.toFixed(2);
    }
    resetHMonteCarloStats();
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    updateHorizontalLegendLabels();
    return getHParticleDiagnostics();
}
window.__setHParticleMonteCarloScale = setHParticleMonteCarloScale;

function setHParticleMonteCarloTuning(options = {}) {
    if (typeof options === "string") {
        hMonteCarloModel = normalizeHParticleMonteCarloModel(options);
    } else if (options && typeof options === "object") {
        if (options.model !== undefined) hMonteCarloModel = normalizeHParticleMonteCarloModel(options.model);
        if (options.strength !== undefined || options.scale !== undefined) {
            const n = Number(options.strength !== undefined ? options.strength : options.scale);
            if (Number.isFinite(n)) hUncertaintyStrength = Math.max(0.0, Math.min(2.0, n));
        }
        if (options.maxSigma !== undefined || options.max_sigma !== undefined) {
            const n = Number(options.maxSigma !== undefined ? options.maxSigma : options.max_sigma);
            if (Number.isFinite(n)) hMonteCarloMaxSigma = Math.max(0.1, n);
        }
        if (options.offsetCapMmYr !== undefined || options.offset_cap_mm_yr !== undefined) {
            const n = Number(options.offsetCapMmYr !== undefined ? options.offsetCapMmYr : options.offset_cap_mm_yr);
            if (Number.isFinite(n)) hMonteCarloOffsetCapMmYr = Math.max(0.0, n);
        }
        if (options.offsetCapRatioToSpeed !== undefined || options.offset_cap_ratio_to_speed !== undefined) {
            const n = Number(options.offsetCapRatioToSpeed !== undefined ? options.offsetCapRatioToSpeed : options.offset_cap_ratio_to_speed);
            if (Number.isFinite(n)) hMonteCarloOffsetCapRatioToSpeed = Math.max(0.0, n);
        }
    }

    if (hUncertaintyStrengthSlider) hUncertaintyStrengthSlider.value = String(Math.max(0.0, Math.min(1.0, hUncertaintyStrength)));
    if (hUncertaintyStrengthValue) hUncertaintyStrengthValue.textContent = hUncertaintyStrength.toFixed(2);

    resetHMonteCarloStats();
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    for (const p of hParticles) resetHParticleMonteCarloRealization(p);
    updateHorizontalLegendLabels();
    setH4Status();
    console.log("[H particles MC] tuning", currentHMonteCarloTuning());
    return getHParticleDiagnostics();
}
window.__setHParticleMonteCarloTuning = setHParticleMonteCarloTuning;

function hashHParticleSeed(value) {
    const str = String(value ?? "");
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
    }
    return h || 0x6d2b79f5;
}

function hSeededRandom01() {
    // Mulberry32: compact deterministic PRNG for visualization reproducibility.
    hMonteCarloRngState = (hMonteCarloRngState + 0x6D2B79F5) >>> 0;
    let t = hMonteCarloRngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function hParticleRandom() {
    return hMonteCarloSeedMode === "seeded" ? hSeededRandom01() : Math.random();
}

function reinitializeHParticlesAfterSeedChange() {
    resetHMonteCarloStats();
    if (hCells && hCells.length && hValidSpawnCells && hValidSpawnCells.length) {
        initHParticles(hParticleCount);
    } else {
        for (const p of hParticles) resetHParticleMonteCarloRealization(p);
    }
    resetHMonteCarloStats();
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    hLastTimestamp = performance.now();
    if (hParticlesVisible && !hAnimationId) {
        hAnimationId = requestAnimationFrame(drawHorizontalParticles);
    }
}

function setHParticleMonteCarloSeed(seed) {
    const normalizedSeed = seed === undefined || seed === null || seed === "" ? "jakarta-v4" : String(seed);
    hMonteCarloSeedMode = "seeded";
    hMonteCarloSeed = normalizedSeed;
    hMonteCarloRngState = hashHParticleSeed(normalizedSeed);
    hMonteCarloSeedApplyCount += 1;
    reinitializeHParticlesAfterSeedChange();
    updateHorizontalLegendLabels();
    console.log("[H particles MC] deterministic seed", {
        seed: hMonteCarloSeed,
        seedHash: hMonteCarloRngState,
        applyCount: hMonteCarloSeedApplyCount,
        note: "Particle spawn/lifetime/MC realization are now deterministic until random mode is restored."
    });
    return getHParticleDiagnostics();
}
window.__setHParticleMonteCarloSeed = setHParticleMonteCarloSeed;

function setHParticleMonteCarloRandomized() {
    hMonteCarloSeedMode = "random";
    hMonteCarloSeed = null;
    hMonteCarloRngState = 0;
    reinitializeHParticlesAfterSeedChange();
    updateHorizontalLegendLabels();
    console.log("[H particles MC] random realization mode restored");
    return getHParticleDiagnostics();
}
window.__setHParticleMonteCarloRandomized = setHParticleMonteCarloRandomized;

function getHParticleDiagnostics() {
    const lifetime = hMcLifetimeStats || {};
    const lifetimeMeanOffset = lifetime.samples > 0 ? lifetime.sumOffsetSpeedMmYr / lifetime.samples : 0.0;
    const lifetimeMeanRealized = lifetime.samples > 0 ? lifetime.sumRealizedSpeedMmYr / lifetime.samples : 0.0;
    return {
        engineMode: hParticleEngineMode,
        uncertaintyMode: hParticleUncertaintyMode,
        uncertaintyEnabled: hUncertaintyEnabled,
        uncertaintyStrength: hUncertaintyStrength,
        rendererContract: hParticleRendererContractLabel(),
        samplerMode: hParticleSamplerMode,
        particles: hParticles.length,
        visible: hParticlesVisible,
        seedMode: hMonteCarloSeedMode,
        seed: hMonteCarloSeed,
        seedApplyCount: hMonteCarloSeedApplyCount,
        monteCarloTuning: currentHMonteCarloTuning(),
        speedP95MmYr: hSpeedP95,
        surfaceOffsetM: H_PARTICLE_SURFACE_OFFSET_M,
        lastFrame: hMcLastFrameSummary,
        lifetime: {
            frames: lifetime.frames || 0,
            samples: lifetime.samples || 0,
            respawns: lifetime.respawns || 0,
            invalidCovariance: lifetime.invalidCovariance || 0,
            psdClamps: lifetime.psdClamps || 0,
            cappedOffsets: lifetime.cappedOffsets || 0,
            meanOffsetSpeedMmYr: Number(lifetimeMeanOffset.toFixed(4)),
            maxOffsetSpeedMmYr: Number((lifetime.maxOffsetSpeedMmYr || 0.0).toFixed(4)),
            meanRealizedSpeedMmYr: Number(lifetimeMeanRealized.toFixed(4)),
            maxRealizedSpeedMmYr: Number((lifetime.maxRealizedSpeedMmYr || 0.0).toFixed(4))
        }
    };
}
window.__getHParticleDiagnostics = getHParticleDiagnostics;

// ------------------------------------------------------------
// V4.4 selected-RUM Monte Carlo validation lab
// ------------------------------------------------------------
// These functions are intentionally diagnostic-only. They do not mutate the
// active particle population or canvas. They answer the thesis/debug question:
// "Does the MC sampler reproduce the covariance of the selected RUM, or is the
// spaghetti a rendering bug?"

function hMcRound(value, digits = 4) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    const f = Math.pow(10, digits);
    return Math.round(n * f) / f;
}

function hMcHeadingFromNorthDeg(east, north) {
    const e = Number(east);
    const n = Number(north);
    if (!Number.isFinite(e) || !Number.isFinite(n) || Math.hypot(e, n) < 1e-12) return null;
    const deg = (Math.atan2(e, n) * 180.0 / Math.PI) % 360.0;
    return (deg + 360.0) % 360.0;
}

function hMcCovarianceEllipseStats(varEast, varNorth, covarEn) {
    const ve0 = Number(varEast);
    const vn0 = Number(varNorth);
    const c0 = Number(covarEn);
    const ve = Number.isFinite(ve0) ? Math.max(0.0, ve0) : 0.0;
    const vn = Number.isFinite(vn0) ? Math.max(0.0, vn0) : 0.0;
    const c = Number.isFinite(c0) ? c0 : 0.0;

    const mid = 0.5 * (ve + vn);
    const diff = 0.5 * (ve - vn);
    const root = Math.sqrt(Math.max(0.0, diff * diff + c * c));
    const lambda1Raw = mid + root;
    const lambda2Raw = mid - root;
    const lambda1 = Math.max(0.0, lambda1Raw);
    const lambda2 = Math.max(0.0, lambda2Raw);

    // Major-axis eigenvector in EN coordinates. theta is measured from +East
    // toward +North. Convert to conventional heading from north for display.
    const theta = 0.5 * Math.atan2(2.0 * c, ve - vn);
    const evEast = Math.cos(theta);
    const evNorth = Math.sin(theta);
    const heading = hMcHeadingFromNorthDeg(evEast, evNorth);

    const major1 = Math.sqrt(lambda1);
    const minor1 = Math.sqrt(lambda2);
    return {
        varEast: hMcRound(ve, 6),
        varNorth: hMcRound(vn, 6),
        covarEn: hMcRound(c, 6),
        eigenvalues: { major: hMcRound(lambda1, 6), minor: hMcRound(lambda2, 6) },
        sigma1: { major: hMcRound(major1, 4), minor: hMcRound(minor1, 4) },
        sigma2: { major: hMcRound(2.0 * major1, 4), minor: hMcRound(2.0 * minor1, 4) },
        aspectRatio: minor1 > 0 ? hMcRound(major1 / minor1, 4) : null,
        majorAxisHeadingDegFromNorth: heading === null ? null : hMcRound(heading, 2),
        psdClamped: lambda1Raw < -1e-12 || lambda2Raw < -1e-12
    };
}

function hMcNormal01WithRng(rng) {
    const u1 = Math.max(1e-12, rng());
    const u2 = rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function hMcMakeLocalRng(seed) {
    let state = hashHParticleSeed(seed || 'rum-mc-lab');
    return function rng() {
        state = (state + 0x6D2B79F5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function hMcFeatureProperty(feature, names) {
    if (!feature) return undefined;
    if (typeof pickedFeatureProperty === 'function') {
        try {
            const v = pickedFeatureProperty(feature, names);
            if (v !== undefined && v !== null && v !== '') return v;
        } catch (e) {}
    }
    if (typeof feature.getProperty === 'function') {
        for (const name of names) {
            try {
                const v = feature.getProperty(name);
                if (v !== undefined && v !== null && v !== '') return v;
            } catch (e) {}
        }
    }
    return undefined;
}

function hMcCellByRow(row) {
    const r = Math.floor(Number(row));
    if (!Number.isFinite(r) || r < 0) return null;
    return hCells.find(c => Math.floor(Number(c.height_row)) === r) || null;
}

function resolveSelectedHParticleCellForMcLab(options = {}) {
    if (options && options.cell) return { cell: options.cell, source: 'options.cell' };

    const requestedRumId = options && options.rumId !== undefined && options.rumId !== null
        ? String(options.rumId).trim()
        : '';
    if (requestedRumId && hCellByRumId && hCellByRumId.has(requestedRumId)) {
        return { cell: hCellByRumId.get(requestedRumId), source: 'options.rumId' };
    }

    const requestedRow = options && options.row !== undefined ? Number(options.row) : NaN;
    if (Number.isFinite(requestedRow)) {
        const rowCell = hMcCellByRow(requestedRow);
        if (rowCell) return { cell: rowCell, source: 'options.row' };
    }

    let feature = null;
    try { feature = (typeof selectedRumFeature !== 'undefined') ? selectedRumFeature : null; } catch (e) {}
    if (feature) {
        const rumIdRaw = hMcFeatureProperty(feature, [
            'rum_id', 'RUM_ID', 'rumId', 'id', 'name', 'display_id', 'cell_id'
        ]);
        const rumId = rumIdRaw !== undefined && rumIdRaw !== null ? String(rumIdRaw).trim() : '';
        if (rumId && hCellByRumId && hCellByRumId.has(rumId)) {
            return { cell: hCellByRumId.get(rumId), source: 'selectedFeature.rum_id' };
        }

        let row = NaN;
        try {
            if (typeof rowIndexFromPickedFeature === 'function') row = Number(rowIndexFromPickedFeature(feature, rumId));
        } catch (e) {}
        if (!Number.isFinite(row)) {
            row = Number(hMcFeatureProperty(feature, ['row_index', 'height_row', 'texture_row', 'rum_index']));
        }
        const rowCell = hMcCellByRow(row);
        if (rowCell) return { cell: rowCell, source: 'selectedFeature.row' };

        // Last-resort direct feature metadata. This still lets the diagnostic run
        // even if the horizontal particle field omitted the selected cell.
        const east = Number(hMcFeatureProperty(feature, ['east_mm_yr', 'east', 've_mm_yr', 'v_east_mm_yr']));
        const north = Number(hMcFeatureProperty(feature, ['north_mm_yr', 'north', 'vn_mm_yr', 'v_north_mm_yr']));
        if (Number.isFinite(east) && Number.isFinite(north)) {
            const fallback = {
                rum_id: rumId,
                height_row: Number.isFinite(row) ? row : -1,
                east_mm_yr: east,
                north_mm_yr: north,
                speed_mm_yr: Math.hypot(east, north),
                var_east: Number(hMcFeatureProperty(feature, ['var_east', 'variance_east', 'cov_ee'])) || 0.0,
                var_north: Number(hMcFeatureProperty(feature, ['var_north', 'variance_north', 'cov_nn'])) || 0.0,
                covar_en: Number(hMcFeatureProperty(feature, ['covar_en', 'cov_en', 'cov_en_mm_yr2'])) || 0.0,
                lon: Number(hMcFeatureProperty(feature, ['lon_center', 'center_lon', 'lon', 'longitude'])) || NaN,
                lat: Number(hMcFeatureProperty(feature, ['lat_center', 'center_lat', 'lat', 'latitude'])) || NaN
            };
            return { cell: fallback, source: 'selectedFeature.directProperties' };
        }
    }

    // Human fallback: current popup title is normally the RUM id.
    try {
        const title = document.getElementById('rumInfoTitle');
        const titleRumId = title ? String(title.textContent || '').trim() : '';
        if (titleRumId && hCellByRumId && hCellByRumId.has(titleRumId)) {
            return { cell: hCellByRumId.get(titleRumId), source: 'rumInfoTitle' };
        }
    } catch (e) {}

    return { cell: null, source: 'none' };
}

function hMcApplyModelToOffset(meanEast, meanNorth, rawOffset, model, strength, capMmYr, capRatioToSpeed) {
    const meanSpeed = Math.hypot(meanEast, meanNorth);
    let offset = { east: rawOffset.east, north: rawOffset.north };

    if (model === 'directional') {
        const denom = Math.max(meanSpeed, H_UNCERTAINTY_SPEED_FLOOR_MM_YR, 1e-9);
        const nx = -meanNorth / denom;
        const ny =  meanEast / denom;
        const perp = offset.east * nx + offset.north * ny;
        offset = { east: nx * perp, north: ny * perp };
    }

    let eastOffset = offset.east * strength;
    let northOffset = offset.north * strength;
    let offsetSpeed = Math.hypot(eastOffset, northOffset);
    let capped = false;

    if (model === 'directional' || model === 'capped_full') {
        const cap = Math.max(Number(capMmYr || 0.0), meanSpeed * Number(capRatioToSpeed || 0.0));
        if (cap > 0.0 && offsetSpeed > cap) {
            const f = cap / Math.max(offsetSpeed, 1e-12);
            eastOffset *= f;
            northOffset *= f;
            offsetSpeed = cap;
            capped = true;
        }
    }

    return {
        east: meanEast + eastOffset,
        north: meanNorth + northOffset,
        eastOffset,
        northOffset,
        offsetSpeed,
        capped
    };
}

function sampleHParticleMonteCarloForCell(cell, sampleCount = 10000, options = {}) {
    const n = Math.max(10, Math.min(200000, Math.floor(Number(sampleCount) || 10000)));
    const model = normalizeHParticleMonteCarloModel(options.model !== undefined ? options.model : hMonteCarloModel);
    const strength = Number.isFinite(Number(options.strength ?? options.scale))
        ? Math.max(0.0, Number(options.strength ?? options.scale))
        : Math.max(0.0, Number(hUncertaintyStrength || 0.0));
    const maxSigma = Number.isFinite(Number(options.maxSigma ?? options.max_sigma))
        ? Math.max(0.1, Number(options.maxSigma ?? options.max_sigma))
        : Math.max(0.1, Number(hMonteCarloMaxSigma || 1.5));
    const capMmYr = Number.isFinite(Number(options.offsetCapMmYr ?? options.offset_cap_mm_yr))
        ? Math.max(0.0, Number(options.offsetCapMmYr ?? options.offset_cap_mm_yr))
        : Math.max(0.0, Number(hMonteCarloOffsetCapMmYr || 0.0));
    const capRatio = Number.isFinite(Number(options.offsetCapRatioToSpeed ?? options.offset_cap_ratio_to_speed))
        ? Math.max(0.0, Number(options.offsetCapRatioToSpeed ?? options.offset_cap_ratio_to_speed))
        : Math.max(0.0, Number(hMonteCarloOffsetCapRatioToSpeed || 0.0));
    const seed = String(options.seed || hMonteCarloSeed || `mc-lab-${cell?.rum_id || cell?.height_row || 'cell'}-${model}-${strength}`);
    const rng = hMcMakeLocalRng(seed);

    const meanEast = Number(cell.east_mm_yr ?? cell.east ?? 0.0);
    const meanNorth = Number(cell.north_mm_yr ?? cell.north ?? 0.0);
    const meanSpeed = Math.hypot(meanEast, meanNorth);
    const varEast = Math.max(0.0, Number(cell.var_east || 0.0));
    const varNorth = Math.max(0.0, Number(cell.var_north || 0.0));
    const covarEn = Number(cell.covar_en || 0.0);

    let sumE = 0.0, sumN = 0.0, sumEE = 0.0, sumNN = 0.0, sumEN = 0.0;
    let sumOff = 0.0, maxOff = 0.0, maxSpeed = 0.0;
    let reversals = 0, perturbGtMean = 0, cappedOffsets = 0, belowStall = 0;

    const oldStats = hMcFrameStats;
    hMcFrameStats = null;
    try {
        for (let k = 0; k < n; k++) {
            let z1 = hMcNormal01WithRng(rng);
            let z2 = hMcNormal01WithRng(rng);
            z1 = Math.max(-maxSigma, Math.min(maxSigma, z1));
            z2 = Math.max(-maxSigma, Math.min(maxSigma, z2));
            const rawOffset = covarianceRealizationOffset({ var_east: varEast, var_north: varNorth, covar_en: covarEn }, z1, z2);
            const realized = hMcApplyModelToOffset(meanEast, meanNorth, rawOffset, model, strength, capMmYr, capRatio);
            const e = realized.east;
            const no = realized.north;
            const sp = Math.hypot(e, no);

            sumE += e; sumN += no;
            sumEE += e * e; sumNN += no * no; sumEN += e * no;
            sumOff += realized.offsetSpeed;
            maxOff = Math.max(maxOff, realized.offsetSpeed);
            maxSpeed = Math.max(maxSpeed, sp);
            if (realized.capped) cappedOffsets += 1;
            if (meanSpeed > 1e-9 && (meanEast * e + meanNorth * no) < 0.0) reversals += 1;
            if (realized.offsetSpeed > meanSpeed) perturbGtMean += 1;
            if (sp < H_PARTICLE_STALL_SPEED_MM_YR) belowStall += 1;
        }
    } finally {
        hMcFrameStats = oldStats;
    }

    const mE = sumE / n;
    const mN = sumN / n;
    const covEE = Math.max(0.0, sumEE / n - mE * mE);
    const covNN = Math.max(0.0, sumNN / n - mN * mN);
    const covEN = sumEN / n - mE * mN;

    return {
        rum: {
            rumId: String(cell.rum_id || ''),
            row: Number.isFinite(Number(cell.height_row)) ? Math.floor(Number(cell.height_row)) : null,
            lon: hMcRound(cell.lon, 6),
            lat: hMcRound(cell.lat, 6)
        },
        tuningUsed: { model, strength, maxSigma, offsetCapMmYr: capMmYr, offsetCapRatioToSpeed: capRatio, seed, samples: n },
        input: {
            mean: { east: hMcRound(meanEast, 6), north: hMcRound(meanNorth, 6), speed: hMcRound(meanSpeed, 6), headingDegFromNorth: hMcRound(hMcHeadingFromNorthDeg(meanEast, meanNorth), 3) },
            covariance: { varEast: hMcRound(varEast, 6), varNorth: hMcRound(varNorth, 6), covarEn: hMcRound(covarEn, 6) },
            ellipse: hMcCovarianceEllipseStats(varEast, varNorth, covarEn)
        },
        sample: {
            mean: { east: hMcRound(mE, 6), north: hMcRound(mN, 6), speed: hMcRound(Math.hypot(mE, mN), 6), headingDegFromNorth: hMcRound(hMcHeadingFromNorthDeg(mE, mN), 3) },
            covariance: { varEast: hMcRound(covEE, 6), varNorth: hMcRound(covNN, 6), covarEn: hMcRound(covEN, 6) },
            ellipse: hMcCovarianceEllipseStats(covEE, covNN, covEN),
            meanOffsetSpeedMmYr: hMcRound(sumOff / n, 6),
            maxOffsetSpeedMmYr: hMcRound(maxOff, 6),
            maxRealizedSpeedMmYr: hMcRound(maxSpeed, 6),
            directionReversalFraction: hMcRound(reversals / n, 6),
            perturbationGreaterThanMeanFraction: hMcRound(perturbGtMean / n, 6),
            cappedOffsetFraction: hMcRound(cappedOffsets / n, 6),
            belowStallFraction: hMcRound(belowStall / n, 6)
        },
        interpretation: model === 'full' && strength === 1.0 && maxSigma >= 5.0
            ? 'Raw Gaussian check: sample covariance should be close to input covariance.'
            : 'Active viewer-behaviour check: sample covariance includes current strength, z-clamping, model projection, and caps.'
    };
}

function sampleSelectedRumMonteCarlo(sampleCount = 10000, options = {}) {
    const resolved = resolveSelectedHParticleCellForMcLab(options || {});
    if (!resolved.cell) {
        const msg = {
            error: 'No selected real RUM cell found. Click a real RUM cap first, or pass {rumId:"..."} / {row:123}.',
            availableCells: hCells.length,
            selectedFeaturePresent: (() => { try { return typeof selectedRumFeature !== 'undefined' && !!selectedRumFeature; } catch (e) { return false; } })()
        };
        console.warn('[H particles MC lab]', msg);
        return msg;
    }
    const result = sampleHParticleMonteCarloForCell(resolved.cell, sampleCount, options || {});
    result.selectionSource = resolved.source;
    console.log('[H particles MC lab] selected RUM sample', result);
    return result;
}
window.__sampleSelectedRumMonteCarlo = sampleSelectedRumMonteCarlo;

function getSelectedRumMonteCarloLab(sampleCount = 10000, options = {}) {
    return sampleSelectedRumMonteCarlo(sampleCount, options);
}
window.__getSelectedRumMonteCarloLab = getSelectedRumMonteCarloLab;

function clearHParticleTrails() {
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    return { cleared: true, particles: hParticles.length, mode: hParticleUncertaintyMode, renderer: hParticleEngineMode };
}
window.__clearHParticleTrails = clearHParticleTrails;


function resizeHParticleCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;

    const targetW = Math.max(1, Math.floor(w * dpr));
    const targetH = Math.max(1, Math.floor(h * dpr));

    if (hParticleCanvas.width !== targetW || hParticleCanvas.height !== targetH) {
        hParticleCanvas.width = targetW;
        hParticleCanvas.height = targetH;
        hParticleCanvas.style.width = `${w}px`;
        hParticleCanvas.style.height = `${h}px`;
        hCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        clearHParticlesCanvas();
        resetHParticleScreenHistory();
    }
}

function clearHParticlesCanvas() {
    hCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
}

function resetHParticleScreenHistory() {
    for (const p of hParticles) {
        p.prevX = null;
        p.prevY = null;
        p.prevTrueX = null;
        p.prevTrueY = null;
    }
}

function solve3x3(A, b) {
    // Small Gaussian elimination solver for affine fit.
    const m = [
        [A[0][0], A[0][1], A[0][2], b[0]],
        [A[1][0], A[1][1], A[1][2], b[1]],
        [A[2][0], A[2][1], A[2][2], b[2]],
    ];

    for (let col = 0; col < 3; col++) {
        let pivot = col;
        for (let r = col + 1; r < 3; r++) {
            if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
        }
        if (Math.abs(m[pivot][col]) < 1e-20) {
            throw new Error("Singular affine fit matrix");
        }
        if (pivot !== col) {
            const tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
        }

        const div = m[col][col];
        for (let c = col; c < 4; c++) m[col][c] /= div;

        for (let r = 0; r < 3; r++) {
            if (r === col) continue;
            const factor = m[r][col];
            for (let c = col; c < 4; c++) {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    return [m[0][3], m[1][3], m[2][3]];
}

function fitGridAffine(cells) {
    // Fits:
    //   lon = a0 + ai*grid_i + aj*grid_j
    //   lat = b0 + bi*grid_i + bj*grid_j
    let A = [[0,0,0], [0,0,0], [0,0,0]];
    let bLon = [0,0,0];
    let bLat = [0,0,0];

    for (const c of cells) {
        const x = [1, c.grid_i, c.grid_j];

        for (let r = 0; r < 3; r++) {
            for (let col = 0; col < 3; col++) {
                A[r][col] += x[r] * x[col];
            }
            bLon[r] += x[r] * c.lon;
            bLat[r] += x[r] * c.lat;
        }
    }

    const lonCoef = solve3x3(A, bLon);
    const latCoef = solve3x3(A, bLat);

    const a0 = lonCoef[0], ai = lonCoef[1], aj = lonCoef[2];
    const b0 = latCoef[0], bi = latCoef[1], bj = latCoef[2];
    const det = ai * bj - aj * bi;

    if (Math.abs(det) < 1e-20) {
        throw new Error("Invalid grid affine inverse");
    }

    return { a0, ai, aj, b0, bi, bj, det };
}

function gridToLonLat(i, j) {
    return {
        lon: hAffine.a0 + hAffine.ai * i + hAffine.aj * j,
        lat: hAffine.b0 + hAffine.bi * i + hAffine.bj * j,
    };
}

function lonLatToGrid(lon, lat) {
    const dl = lon - hAffine.a0;
    const dp = lat - hAffine.b0;

    return {
        i: (dl * hAffine.bj - hAffine.aj * dp) / hAffine.det,
        j: (hAffine.ai * dp - dl * hAffine.bi) / hAffine.det,
    };
}

function hCellAt(i, j) {
    return hLookup.get(`${i},${j}`) || null;
}

function decodeDispMmFromBytes(r, g) {
    const encoded16 = r * 256 + g;
    const normalized = encoded16 / 65535.0;
    return normalized * (V_MAX - V_MIN) + V_MIN;
}

function decodeSigmaMmFromByte(b) {
    const bb = Number(b);
    if (!Number.isFinite(bb)) return NaN;
    return (bb / 255.0) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN;
}

function readDispSigmaForRowEpoch(row, epochIdx) {
    if (!hHeightTextureReady || !hHeightImageCtx) return { dispMm: null, sigmaMm: null };
    const r = Math.max(0, Math.min(NUM_RUMS - 1, Math.floor(Number(row))));
    const c = Math.max(0, Math.min(NUM_EPOCHS - 1, Math.floor(Number(epochIdx))));
    if (!Number.isFinite(r) || !Number.isFinite(c)) return { dispMm: null, sigmaMm: null };

    try {
        const px = hHeightImageCtx.getImageData(c, r, 1, 1).data;
        return {
            dispMm: decodeDispMmFromBytes(px[0], px[1]),
            sigmaMm: decodeSigmaMmFromByte(px[2])
        };
    } catch (e) {
        return { dispMm: null, sigmaMm: null };
    }
}

function currentEpochIndexFromUIFallback() {
    const slider = document.getElementById("epochSlider");
    const sliderValue = slider ? Number(slider.value) : NaN;
    if (Number.isFinite(sliderValue)) {
        return Math.max(0, Math.min(Math.max(0, NUM_EPOCHS - 1), Math.round(sliderValue)));
    }
    if (Number.isFinite(hCurrentDispEpoch) && hCurrentDispEpoch >= 0) {
        return Math.max(0, Math.min(Math.max(0, NUM_EPOCHS - 1), Math.round(hCurrentDispEpoch)));
    }
    return 0;
}

function installPackedSeriesData(packed) {
    packedSeriesData = null;
    packedSeriesArrays = null;
    packedSeriesEpochCount = 0;
    packedSeriesRumCount = 0;
    packedMeasurementAvailable = false;

    const arrays = packed && packed.arrays ? packed.arrays : null;
    const meta = packed && packed.metadata ? packed.metadata : {};
    const epochs = Array.isArray(packed?.epochs) ? packed.epochs : [];

    const measurement = arrays && Array.isArray(arrays.measurement_mm) ? arrays.measurement_mm : null;
    const model = arrays && Array.isArray(arrays.model_mm) ? arrays.model_mm : null;
    const sigma = arrays && Array.isArray(arrays.sigma_mm) ? arrays.sigma_mm : null;

    const epochCount = Number(meta.epoch_count ?? epochs.length ?? NUM_EPOCHS);
    const rumCount = Number(meta.rum_count ?? (Array.isArray(packed?.rum_order) ? packed.rum_order.length : 0));

    if (!arrays || !Number.isFinite(epochCount) || epochCount <= 0 || !Number.isFinite(rumCount) || rumCount <= 0) {
        console.warn("[packed_series] unsupported schema; measurement trendline will fall back to height texture", packed);
        return false;
    }

    packedSeriesData = packed;
    packedSeriesArrays = arrays;
    packedSeriesEpochCount = Math.floor(epochCount);
    packedSeriesRumCount = Math.floor(rumCount);
    packedMeasurementAvailable = Boolean(measurement && measurement.length >= packedSeriesEpochCount * packedSeriesRumCount);

    console.log("[packed_series] loaded", {
        schema: meta.schema,
        rumCount: packedSeriesRumCount,
        epochCount: packedSeriesEpochCount,
        measurementAvailable: packedMeasurementAvailable,
        hasModel: Boolean(model),
        hasSigma: Boolean(sigma),
        roleContract: viewerTuning?.visual_defaults?.vertical_series_contract || viewerTuning?.vertical_series_contract || null
    });

    return true;
}

async function loadPackedSeriesOptional() {
    try {
        const response = await fetch(PACKED_SERIES_URL);
        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }
        const packed = await response.json();
        installPackedSeriesData(packed);
        return packedSeriesData;
    } catch (error) {
        console.warn("[packed_series] not loaded; trendline will use model texture fallback", error);
        return null;
    }
}

function packedFlatArray(name) {
    if (!packedSeriesArrays) return null;
    const arr = packedSeriesArrays[name];
    return Array.isArray(arr) ? arr : null;
}

function packedValueByRowEpoch(name, row, epochIdx) {
    const arr = packedFlatArray(name);
    const r = Math.floor(Number(row));
    const e = Math.floor(Number(epochIdx));
    if (!arr || !Number.isFinite(r) || !Number.isFinite(e)) return null;
    if (r < 0 || e < 0 || r >= packedSeriesRumCount || e >= packedSeriesEpochCount) return null;
    const value = Number(arr[r * packedSeriesEpochCount + e]);
    return Number.isFinite(value) ? value : null;
}

function packedSeriesForRow(name, row) {
    const arr = packedFlatArray(name);
    const r = Math.floor(Number(row));
    if (!arr || !Number.isFinite(r) || r < 0 || r >= packedSeriesRumCount) return [];
    const start = r * packedSeriesEpochCount;
    const out = [];
    for (let i = 0; i < packedSeriesEpochCount; i++) {
        const value = Number(arr[start + i]);
        out.push(Number.isFinite(value) ? value : NaN);
    }
    return out;
}

function measurementSeriesForRow(row) {
    return packedSeriesForRow("measurement_mm", row);
}

function measurementValueForRowEpoch(row, epochIdx) {
    return packedValueByRowEpoch("measurement_mm", row, epochIdx);
}

function modelValueForRowEpoch(row, epochIdx) {
    const packedModel = packedValueByRowEpoch("model_mm", row, epochIdx);
    if (packedModel !== null) return packedModel;

    const textureSample = readDispSigmaForRowEpoch(row, epochIdx);
    return textureSample && textureSample.dispMm !== null ? textureSample.dispMm : null;
}

async function loadHParticleHeightTexture(meta) {
    hHeightMeta = meta;
    hHeightImageCanvas = document.createElement("canvas");
    hHeightImageCanvas.width = NUM_EPOCHS;
    hHeightImageCanvas.height = NUM_RUMS;
    hHeightImageCtx = hHeightImageCanvas.getContext("2d", { willReadFrequently: true });

    const img = new Image();
    img.decoding = "async";
    img.src = HEIGHT_TEXTURE_URL;

    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Could not load height_texture.png for H-particle surface heights"));
    });

    hHeightImageCtx.drawImage(img, 0, 0);

    const t0 = performance.now();
    hDispEpochCache = new Float32Array(NUM_EPOCHS * NUM_RUMS);
    hSigmaEpochCache = new Float32Array(NUM_EPOCHS * NUM_RUMS);

    // D2A: decode the full height texture once at startup.
    // Playback then selects a cached epoch row via subarray(), avoiding synchronous
    // Canvas2D readback (`getImageData`) on every epoch change.
    const pixels = hHeightImageCtx.getImageData(0, 0, NUM_EPOCHS, NUM_RUMS).data;
    for (let row = 0; row < NUM_RUMS; row++) {
        for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            const src = (row * NUM_EPOCHS + epoch) * 4;
            const dst = epoch * NUM_RUMS + row;
            hDispEpochCache[dst] = decodeDispMmFromBytes(pixels[src], pixels[src + 1]);
            hSigmaEpochCache[dst] = decodeSigmaMmFromByte(pixels[src + 2]);
        }
    }

    hSurfaceCacheMode = "predecoded_epoch_major_texture";
    hSurfaceCacheBuildMs = performance.now() - t0;
    hHeightTextureReady = true;
    updateHParticleDisplacementForEpoch(0);

    window.__hParticleSurfaceCacheStats = function() {
        return {
            mode: hSurfaceCacheMode,
            rows: NUM_RUMS,
            epochs: NUM_EPOCHS,
            dispCacheBytes: hDispEpochCache ? hDispEpochCache.byteLength : 0,
            sigmaCacheBytes: hSigmaEpochCache ? hSigmaEpochCache.byteLength : 0,
            buildMs: Number(hSurfaceCacheBuildMs.toFixed(2)),
            currentEpoch: hCurrentDispEpoch
        };
    };

    console.log("[H particles] Dynamic surface height enabled", {
        rows: NUM_RUMS,
        epochs: NUM_EPOCHS,
        surfaceOffsetM: H_PARTICLE_SURFACE_OFFSET_M,
        cacheMode: hSurfaceCacheMode,
        cacheBuildMs: Number(hSurfaceCacheBuildMs.toFixed(2)),
        note: "Canvas particles project at datum + MODEL displacement_mm * vertical scale_m_per_mm + offset; surface heights are predecoded once for smooth playback"
    });
}

function updateHParticleDisplacementForEpoch(epochIdx) {
    if (!hHeightTextureReady || !hDispEpochCache) return;

    const idx = Math.max(0, Math.min(NUM_EPOCHS - 1, Math.floor(epochIdx)));
    if (idx === hCurrentDispEpoch) return;

    const start = idx * NUM_RUMS;
    const end = start + NUM_RUMS;
    hCurrentDispByRow = hDispEpochCache.subarray(start, end);
    hCurrentSigmaByRow = hSigmaEpochCache ? hSigmaEpochCache.subarray(start, end) : null;
    hCurrentDispEpoch = idx;
}

function rowDispMm(cell) {
    if (!cell || !hCurrentDispByRow) return 0.0;
    const row = Number(cell.height_row);
    if (!Number.isFinite(row) || row < 0 || row >= hCurrentDispByRow.length) return 0.0;
    return hCurrentDispByRow[row];
}

function sampleVerticalDispMm(lon, lat) {
    if (!hAffine || hCells.length === 0 || !hCurrentDispByRow) return 0.0;

    const g = lonLatToGrid(lon, lat);
    const i0 = Math.floor(g.i);
    const j0 = Math.floor(g.j);
    const fi = g.i - i0;
    const fj = g.j - j0;

    const c00 = hCellAt(i0,     j0);
    const c10 = hCellAt(i0 + 1, j0);
    const c01 = hCellAt(i0,     j0 + 1);
    const c11 = hCellAt(i0 + 1, j0 + 1);

    if (c00 && c10 && c01 && c11) {
        const w00 = (1.0 - fi) * (1.0 - fj);
        const w10 = fi * (1.0 - fj);
        const w01 = (1.0 - fi) * fj;
        const w11 = fi * fj;

        return (
            rowDispMm(c00) * w00 +
            rowDispMm(c10) * w10 +
            rowDispMm(c01) * w01 +
            rowDispMm(c11) * w11
        );
    }

    const nearest = hCellAt(Math.round(g.i), Math.round(g.j));
    return rowDispMm(nearest);
}

function hParticleHeightM(lon, lat) {
    // Shared particle height contract for canvas and primitive modes.
    // XY follows horizontal advection. Z follows the current-epoch MODEL surface
    // sampled at that XY, plus a small visual offset above the cap surface.
    const dispMm = sampleVerticalDispMm(lon, lat);
    return (
        DISPLAY_DATUM_HEIGHT_M +
        dispMm * currentVerticalExaggeration +
        H_PARTICLE_SURFACE_OFFSET_M
    );
}

function refreshHParticleSurfaceState(p) {
    if (!p) return DISPLAY_DATUM_HEIGHT_M + H_PARTICLE_SURFACE_OFFSET_M;
    p.heightM = hParticleHeightM(p.lon, p.lat);
    return p.heightM;
}

function hParticleCartesian(p) {
    const heightM = refreshHParticleSurfaceState(p);
    return Cesium.Cartesian3.fromDegrees(p.lon, p.lat, heightM);
}

function ensureHPrimitiveParticleCollection() {
    if (!hPrimitiveParticleCollection) {
        hPrimitiveParticleCollection = new Cesium.PointPrimitiveCollection({
            show: hParticlesVisible && isHParticlePrimitiveMode()
        });
        viewer.scene.primitives.add(hPrimitiveParticleCollection);
        console.log("[H particles primitive points] PointPrimitiveCollection installed");
    }
    return hPrimitiveParticleCollection;
}

function syncHPrimitiveParticleCollection() {
    if (!isHParticlePrimitiveMode()) return null;
    const collection = ensureHPrimitiveParticleCollection();

    while (collection.length < hParticles.length) {
        collection.add({
            position: Cesium.Cartesian3.fromDegrees(0, 0, DISPLAY_DATUM_HEIGHT_M),
            pixelSize: H_PRIMITIVE_POINTS_PIXEL_SIZE,
            color: new Cesium.Color(
                H_PRIMITIVE_POINTS_COLOR_RGB[0],
                H_PRIMITIVE_POINTS_COLOR_RGB[1],
                H_PRIMITIVE_POINTS_COLOR_RGB[2],
                Math.max(0.0, Math.min(1.0, hParticleOpacity))
            ),
            outlineColor: new Cesium.Color(
                H_PRIMITIVE_POINTS_OUTLINE_RGB[0],
                H_PRIMITIVE_POINTS_OUTLINE_RGB[1],
                H_PRIMITIVE_POINTS_OUTLINE_RGB[2],
                0.70
            ),
            outlineWidth: H_PRIMITIVE_POINTS_OUTLINE_WIDTH,
            disableDepthTestDistance: 0.0,
            show: false
        });
    }

    while (collection.length > hParticles.length) {
        collection.remove(collection.get(collection.length - 1));
    }

    collection.show = hParticlesVisible && isHParticlePrimitiveMode();
    return collection;
}

function hideHPrimitiveParticles() {
    if (!hPrimitiveParticleCollection) return;
    hPrimitiveParticleCollection.show = false;
    for (let i = 0; i < hPrimitiveParticleCollection.length; i++) {
        hPrimitiveParticleCollection.get(i).show = false;
    }
}

function updateHPrimitivePoint(point, particle, field) {
    if (!point || !particle || !field) return;
    point.position = hParticleCartesian(particle);

    const speedRatio = Math.max(0.0, Math.min(1.6, field.speed / Math.max(hSpeedP95, 1e-9)));
    const alpha = Math.max(0.25, Math.min(0.95, 0.35 + 0.50 * speedRatio)) * Math.max(0.0, Math.min(1.0, hParticleOpacity));
    const size = (H_PRIMITIVE_POINTS_PIXEL_SIZE + 2.0 * speedRatio) * Math.max(0.1, hParticleSizeMultiplier);

    point.pixelSize = size;
    point.color = new Cesium.Color(
        H_PRIMITIVE_POINTS_COLOR_RGB[0],
        H_PRIMITIVE_POINTS_COLOR_RGB[1],
        H_PRIMITIVE_POINTS_COLOR_RGB[2],
        alpha
    );
    point.outlineColor = new Cesium.Color(
        H_PRIMITIVE_POINTS_OUTLINE_RGB[0],
        H_PRIMITIVE_POINTS_OUTLINE_RGB[1],
        H_PRIMITIVE_POINTS_OUTLINE_RGB[2],
        Math.max(0.35, Math.min(0.85, alpha + 0.10))
    );
    point.outlineWidth = H_PRIMITIVE_POINTS_OUTLINE_WIDTH;
    point.disableDepthTestDistance = 0.0;
    point.show = true;
}

function normalizeHParticleUncertaintyMode(mode) {
    const m = String(mode || "").trim().toLowerCase();
    if (m === "mc" || m === "monte_carlo" || m === "monte-carlo" || m === "montecarlo" || m === "realization" || m === "realizations") return "montecarlo";
    if (m === "shimmer" || m === "wobble" || m === "shalalala" || m === "legacy") return "shimmer";
    if (m === "off" || m === "none" || m === "clean" || m === "mean") return "off";
    return "shimmer";
}

function isHParticleShimmerMode() {
    return hParticleUncertaintyMode === "shimmer" && hUncertaintyEnabled && hUncertaintyStrength > 0.0;
}

function isHParticleMonteCarloMode() {
    return hParticleUncertaintyMode === "montecarlo" && hUncertaintyEnabled && hUncertaintyStrength > 0.0;
}

function hParticleUncertaintyLabel() {
    if (hParticleUncertaintyMode === "montecarlo") return "Monte Carlo";
    if (hParticleUncertaintyMode === "off") return "clean mean";
    return "shimmer";
}

function setHParticleUncertaintyMode(mode) {
    const normalized = normalizeHParticleUncertaintyMode(mode);
    hParticleUncertaintyMode = normalized;
    hUncertaintyEnabled = normalized !== "off";

    // New mode = new interpretation of the same particle population.
    // Resetting avoids mixing old screen-space trails with MC path traces.
    resetHMonteCarloStats();
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    for (const p of hParticles) resetHParticleMonteCarloRealization(p);

    updateHorizontalLegendLabels();
    setH4Status();
    console.log("[H particles] uncertainty mode", hParticleUncertaintyMode, {
        enabled: hUncertaintyEnabled,
        strength: hUncertaintyStrength,
        seedMode: hMonteCarloSeedMode,
        seed: hMonteCarloSeed,
        monteCarloTuning: currentHMonteCarloTuning(),
        note: hParticleUncertaintyMode === "montecarlo"
            ? "path-level velocity realization from covariance"
            : hParticleUncertaintyMode === "shimmer"
                ? "render-only screen-space wobble"
                : "mean field only"
    });
    return hParticleUncertaintyMode;
}
window.__setHParticleUncertaintyMode = setHParticleUncertaintyMode;

function hParticleEngineLabel() {
    if (hParticleEngineMode === "primitive_points") return "primitive points";
    return "canvas overlay";
}

function hParticleRendererContractLabel() {
    if (isHParticlePrimitiveMode()) return "scene-space primitive dots, depth-tested, no trails";
    if (hParticleUncertaintyMode === "montecarlo") return `screen-space canvas trails + Monte Carlo ${hMonteCarloModel} realization`;
    if (hParticleUncertaintyMode === "shimmer") return "screen-space canvas trails + render-only shimmer";
    return "screen-space canvas trails, mean field only";
}

function setHPrimitivePointsDebug(enabled) {
    hPrimitivePointsDebugEnabled = !!enabled;
    hPrimitiveLastDebugLogMs = 0;
    console.log("[H particles primitive points] debug", hPrimitivePointsDebugEnabled ? "on" : "off");
    return hPrimitivePointsDebugEnabled;
}
window.__setHPrimitivePointsDebug = setHPrimitivePointsDebug;

function setHParticleEngineMode(mode) {
    const normalized = normalizeHParticleEngineMode(mode);
    if (normalized === hParticleEngineMode) return hParticleEngineMode;

    hParticleEngineMode = normalized;
    clearHParticlesCanvas();
    resetHParticleScreenHistory();

    if (isHParticlePrimitiveMode()) {
        hParticleCanvas.style.display = "none";
        syncHPrimitiveParticleCollection();
    } else {
        hideHPrimitiveParticles();
        hParticleCanvas.style.display = hParticlesVisible ? "block" : "none";
    }

    hLastTimestamp = performance.now();
    if (hParticlesVisible && !hAnimationId) {
        hAnimationId = requestAnimationFrame(drawHorizontalParticles);
    }
    updateHorizontalLegendLabels();
    setH4Status();
    viewer.scene.requestRender();
    console.log("[H particles] engine mode", hParticleEngineMode, hParticleRendererContractLabel());
    return hParticleEngineMode;
}
window.__setHParticleEngineMode = setHParticleEngineMode;

function weightedHorizontalSample(items) {
    let wSum = 0.0;
    let east = 0.0;
    let north = 0.0;
    let varEast = 0.0;
    let varNorth = 0.0;
    let covarEn = 0.0;

    for (const item of items) {
        if (!item.cell || item.weight <= 0.0) continue;
        wSum += item.weight;
        east += item.cell.east_mm_yr * item.weight;
        north += item.cell.north_mm_yr * item.weight;
        varEast += Number(item.cell.var_east || 0.0) * item.weight;
        varNorth += Number(item.cell.var_north || 0.0) * item.weight;
        covarEn += Number(item.cell.covar_en || 0.0) * item.weight;
    }

    if (wSum <= 0.0) return null;

    east /= wSum;
    north /= wSum;
    varEast /= wSum;
    varNorth /= wSum;
    covarEn /= wSum;

    const speed = Math.sqrt(east * east + north * north);
    return {
        east,
        north,
        speed,
        var_east: Math.max(0.0, varEast),
        var_north: Math.max(0.0, varNorth),
        covar_en: covarEn,
        valid: true
    };
}

function nearestHorizontalSample(g) {
    const nearest = hCellAt(Math.round(g.i), Math.round(g.j));
    if (!nearest) return null;

    return {
        east: nearest.east_mm_yr,
        north: nearest.north_mm_yr,
        speed: nearest.speed_mm_yr,
        var_east: Math.max(0.0, Number(nearest.var_east || 0.0)),
        var_north: Math.max(0.0, Number(nearest.var_north || 0.0)),
        covar_en: Number(nearest.covar_en || 0.0),
        valid: true
    };
}

function hasFullBilinearSupportAtGrid(g) {
    const i0 = Math.floor(g.i);
    const j0 = Math.floor(g.j);

    return (
        hCellAt(i0,     j0) &&
        hCellAt(i0 + 1, j0) &&
        hCellAt(i0,     j0 + 1) &&
        hCellAt(i0 + 1, j0 + 1)
    );
}

function hasEightNeighborSupport(cell) {
    if (!cell) return false;
    const i = cell.grid_i;
    const j = cell.grid_j;

    for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
            if (!hCellAt(i + di, j + dj)) return false;
        }
    }
    return true;
}

function sampleHorizontalField(lon, lat) {
    if (!hAffine || hCells.length === 0) return null;

    const g = lonLatToGrid(lon, lat);

    if (hParticleSamplerMode === "nearest") {
        return nearestHorizontalSample(g);
    }

    const i0 = Math.floor(g.i);
    const j0 = Math.floor(g.j);
    const fi = g.i - i0;
    const fj = g.j - j0;

    const c00 = hCellAt(i0,     j0);
    const c10 = hCellAt(i0 + 1, j0);
    const c01 = hCellAt(i0,     j0 + 1);
    const c11 = hCellAt(i0 + 1, j0 + 1);

    const samples = [
        { cell: c00, weight: (1.0 - fi) * (1.0 - fj) },
        { cell: c10, weight: fi * (1.0 - fj) },
        { cell: c01, weight: (1.0 - fi) * fj },
        { cell: c11, weight: fi * fj },
    ];

    if (hParticleSamplerMode === "bilinear") {
        // Loose bilinear:
        // blend all available neighbours and renormalize weights.
        // Smoothest visually, but can imply continuity over small gaps.
        const loose = weightedHorizontalSample(samples);
        if (loose) return loose;
        return nearestHorizontalSample(g);
    }

    if (hParticleSamplerMode === "conservative_v1") {
        // Conservative bilinear v1:
        // blend only with full 4-cell support, then fall back to nearest.
        // Useful diagnostic, but not final because it can carry nearest-RUM
        // velocity into unsupported blank/no-data zones.
        if (c00 && c10 && c01 && c11) {
            return weightedHorizontalSample(samples);
        }
        return nearestHorizontalSample(g);
    }

    // Conservative bilinear v2:
    // final/scientific mode. Blend only with full 4-cell support.
    // If support is incomplete, return null so the particle dies/respawns.
    // This prevents velocity extrapolation into blankies, holes, and domain edges.
    if (c00 && c10 && c01 && c11) {
        return weightedHorizontalSample(samples);
    }

    return null;
}

function smoothstep(edge0, edge1, x) {
    const t = Math.max(0.0, Math.min(1.0, (x - edge0) / Math.max(edge1 - edge0, 1e-9)));
    return t * t * (3.0 - 2.0 * t);
}

function fieldSigmaThetaDeg(field) {
    if (!field) return 0.0;

    const east = Number(field.east || 0.0);
    const north = Number(field.north || 0.0);
    const speed = Math.sqrt(east * east + north * north);
    const denom = Math.max(speed, H_UNCERTAINTY_SPEED_FLOOR_MM_YR);

    // Unit vector perpendicular to mean horizontal velocity.
    const nx = -north / denom;
    const ny =  east / denom;

    const varE = Math.max(0.0, Number(field.var_east || 0.0));
    const varN = Math.max(0.0, Number(field.var_north || 0.0));
    const covEN = Number(field.covar_en || 0.0);

    const varPerp = Math.max(0.0, nx * nx * varE + 2.0 * nx * ny * covEN + ny * ny * varN);
    const sigmaPerp = Math.sqrt(varPerp);

    // Small-angle directional uncertainty proxy, radians → degrees.
    const sigmaThetaRad = sigmaPerp / denom;
    return Cesium.Math.toDegrees(sigmaThetaRad);
}

// ------------------------------------------------------------
// Animated horizontal velocity / uncertainty legend
// ------------------------------------------------------------
let hGlyphLegendCtx = hGlyphLegendCanvas ? hGlyphLegendCanvas.getContext("2d") : null;
let hLegendCtx = hLegendCanvas ? hLegendCanvas.getContext("2d") : null;
let hLegendParticles = [];
let hLegendAnimationId = null;
let hLegendLastTimestamp = 0;
let hLegendSpeedP75 = NaN;
let hLegendSpeedP995 = NaN;
let hLegendSigmaThetaP75 = NaN;
let hGlyphEllipseMajor2SigmaP75 = NaN;
let hGlyphEllipseMinor2SigmaP75 = NaN;
let hGlyphEllipseMajor2SigmaP995 = NaN;

function percentileFromSorted(sortedValues, pct) {
    if (!sortedValues || sortedValues.length === 0) return NaN;
    const p = Math.max(0.0, Math.min(1.0, Number(pct)));
    const idx = p * (sortedValues.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const f = idx - lo;
    if (lo === hi) return sortedValues[lo];
    return sortedValues[lo] * (1.0 - f) + sortedValues[hi] * f;
}

function formatHLegendSpeed(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (Math.abs(n) >= 10.0) return n.toFixed(1);
    return n.toFixed(2).replace(/\.00$/, ".0");
}

function formatHLegendAngle(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (n >= 10.0) return n.toFixed(0);
    return n.toFixed(1).replace(/\.0$/, "");
}

function covarianceStdMajorMinor(varEast, varNorth, covarEn) {
    const ve = Math.max(0.0, Number(varEast || 0.0));
    const vn = Math.max(0.0, Number(varNorth || 0.0));
    const c = Number(covarEn || 0.0);
    const mid = 0.5 * (ve + vn);
    const diff = 0.5 * (ve - vn);
    const root = Math.sqrt(Math.max(0.0, diff * diff + c * c));
    const lambdaMajor = Math.max(0.0, mid + root);
    const lambdaMinor = Math.max(0.0, mid - root);
    return {
        major: Math.sqrt(lambdaMajor),
        minor: Math.sqrt(lambdaMinor)
    };
}

function cssVar(name, fallback) {
    try {
        const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return value || fallback;
    } catch (e) {
        return fallback;
    }
}

function updateHorizontalGlyphLegendLabels() {
    if (hGlyphArrowText) {
        hGlyphArrowText.innerHTML = `<span class="hGlyphLegendMain">P75</span><span class="hGlyphLegendSub">${formatHLegendSpeed(hLegendSpeedP75)} mm/yr</span>`;
    }
    if (hGlyphEllipseText) {
        hGlyphEllipseText.innerHTML = `<span class="hGlyphLegendMain">${H_ELLIPSE_SIGMA_LABEL} major</span><span class="hGlyphLegendSub">${formatHLegendSpeed(hGlyphEllipseMajor2SigmaP75)} mm/yr</span>`;
    }
}

function resizeHGlyphLegendCanvas() {
    if (!hGlyphLegendCanvas || !hGlyphLegendCtx || !hGlyphLegendBar) return false;

    const rect = hGlyphLegendBar.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1.0;
    const targetW = Math.max(1, Math.floor(rect.width * dpr));
    const targetH = Math.max(1, Math.floor(rect.height * dpr));

    if (hGlyphLegendCanvas.width !== targetW || hGlyphLegendCanvas.height !== targetH) {
        hGlyphLegendCanvas.width = targetW;
        hGlyphLegendCanvas.height = targetH;
        hGlyphLegendCanvas.style.width = `${rect.width}px`;
        hGlyphLegendCanvas.style.height = `${rect.height}px`;
        hGlyphLegendCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return true;
    }
    return false;
}

function drawRumReferenceBox(ctx, x0, x1, y, h) {
    const color = cssVar("--h-glyph-reference-color", "rgba(255,255,255,0.76)");
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.lineCap = "round";

    // RUM side boundaries: |       |
    ctx.beginPath();
    ctx.moveTo(x0, y - h * 0.38);
    ctx.lineTo(x0, y + h * 0.38);
    ctx.moveTo(x1, y - h * 0.38);
    ctx.lineTo(x1, y + h * 0.38);
    ctx.stroke();

    // Very subtle bottom reference, so users read the two bars as one RUM width.
    ctx.globalAlpha = 0.38;
    ctx.beginPath();
    ctx.moveTo(x0, y + h * 0.38);
    ctx.lineTo(x1, y + h * 0.38);
    ctx.stroke();
    ctx.restore();
}

function drawStaticArrow(ctx, x0, x1, y, boxH) {
    const color = cssVar("--h-arrow-color", "rgba(255,184,31,1)");
    const boxW = Math.max(1, x1 - x0);
    const ref = Number.isFinite(hLegendSpeedP995) && hLegendSpeedP995 > 0.0 ? hLegendSpeedP995 : hLegendSpeedP75;
    const rawLen = Number.isFinite(hLegendSpeedP75) && Number.isFinite(ref) && ref > 0.0
        ? boxW * 0.90 * Math.max(0.0, Math.min(1.0, hLegendSpeedP75 / ref))
        : boxW * 0.42;
    const len = Math.max(13.0, Math.min(boxW * 0.90, rawLen));
    const cx = (x0 + x1) * 0.5;
    const start = Math.max(x0 + 5, cx - len * 0.5);
    const end = Math.min(x1 - 5, cx + len * 0.5);
    const head = Math.max(4.0, Math.min(8.0, len * 0.25));

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.1;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(start, y);
    ctx.lineTo(end, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(end, y);
    ctx.lineTo(end - head, y - head * 0.55);
    ctx.lineTo(end - head, y + head * 0.55);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
}

function drawStaticEllipse(ctx, x0, x1, y, boxH) {
    const color = cssVar("--h-ellipse-color", "rgba(0,242,217,1)");
    const boxW = Math.max(1, x1 - x0);
    const ref = Number.isFinite(hGlyphEllipseMajor2SigmaP995) && hGlyphEllipseMajor2SigmaP995 > 0.0
        ? hGlyphEllipseMajor2SigmaP995
        : hGlyphEllipseMajor2SigmaP75;

    const majorRatio = Number.isFinite(hGlyphEllipseMajor2SigmaP75) && Number.isFinite(ref) && ref > 0.0
        ? Math.max(0.0, Math.min(1.0, hGlyphEllipseMajor2SigmaP75 / ref))
        : 0.55;
    const minorRatioRaw = Number.isFinite(hGlyphEllipseMinor2SigmaP75) && Number.isFinite(hGlyphEllipseMajor2SigmaP75) && hGlyphEllipseMajor2SigmaP75 > 0.0
        ? hGlyphEllipseMinor2SigmaP75 / hGlyphEllipseMajor2SigmaP75
        : 0.55;
    const minorRatio = Math.max(0.28, Math.min(1.0, minorRatioRaw));

    const rx = Math.max(5.5, Math.min(boxW * 0.45, boxW * 0.45 * majorRatio));
    const ry = Math.max(3.8, Math.min(boxH * 0.33, rx * minorRatio));
    const cx = (x0 + x1) * 0.5;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = 0.98;
    ctx.beginPath();
    ctx.ellipse(cx, y, rx, ry, 0.0, 0.0, Math.PI * 2.0);
    ctx.stroke();

    // Soft center point, matching the RUM-centred glyph semantics.
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.72;
    ctx.beginPath();
    ctx.arc(cx, y, 1.5, 0.0, Math.PI * 2.0);
    ctx.fill();
    ctx.restore();
}

function drawHorizontalGlyphLegend() {
    if (!hGlyphLegendCanvas || !hGlyphLegendCtx || !hGlyphLegendBar) return;

    resizeHGlyphLegendCanvas();

    const rect = hGlyphLegendBar.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const y = height * 0.50;
    const half = width * 0.5;
    const boxH = height * 0.82;

    hGlyphLegendCtx.clearRect(0, 0, width, height);

    // Left and right glyph boxes are the same screen width: one symbolic RUM width.
    const leftBox0 = 15;
    const leftBox1 = Math.min(half - 82, leftBox0 + Math.max(48, half * 0.39));
    const rightBox0 = half + 15;
    const rightBox1 = Math.min(width - 82, rightBox0 + Math.max(48, half * 0.39));

    drawRumReferenceBox(hGlyphLegendCtx, leftBox0, leftBox1, y, boxH);
    drawStaticArrow(hGlyphLegendCtx, leftBox0, leftBox1, y, boxH);

    drawRumReferenceBox(hGlyphLegendCtx, rightBox0, rightBox1, y, boxH);
    drawStaticEllipse(hGlyphLegendCtx, rightBox0, rightBox1, y, boxH);

    updateHorizontalGlyphLegendLabels();
}

function updateHorizontalLegendLabels() {
    const speedTxt = formatHLegendSpeed(hLegendSpeedP75);
    const thetaTxt = formatHLegendAngle(hLegendSigmaThetaP75);

    if (isHParticlePrimitiveMode()) {
        if (hLegendLeftText) {
            hLegendLeftText.textContent = `Primitive dots | P75 ${speedTxt} mm/yr`;
        }
        if (hLegendRightText) {
            hLegendRightText.textContent = `Depth-tested | Z +${H_PARTICLE_SURFACE_OFFSET_M.toFixed(0)} m`;
        }
        return;
    }

    if (hLegendLeftText) {
        const modeTxt = hParticleUncertaintyMode === "montecarlo"
            ? "Canvas MC"
            : hParticleUncertaintyMode === "shimmer"
                ? "Canvas shimmer"
                : "Canvas mean";
        hLegendLeftText.textContent = `${modeTxt} | P75 ${speedTxt} mm/yr`;
    }
    if (hLegendRightText) {
        if (hParticleUncertaintyMode === "montecarlo") {
            const seedTxt = hMonteCarloSeedMode === "seeded" ? `seed ${hMonteCarloSeed}` : "random";
            hLegendRightText.textContent = `${hMonteCarloModel} MC | scale ${hUncertaintyStrength.toFixed(2)} | ${seedTxt}`;
        } else if (hParticleUncertaintyMode === "shimmer") {
            hLegendRightText.textContent = `Screen shimmer | σθ≈${thetaTxt}°`;
        } else {
            hLegendRightText.textContent = "Mean path | uncertainty off";
        }
    }
}


function updateHorizontalLegendStatsFromCells() {
    if (!hCells || hCells.length === 0) {
        hLegendSpeedP75 = NaN;
        hLegendSpeedP995 = NaN;
        hLegendSigmaThetaP75 = NaN;
        hGlyphEllipseMajor2SigmaP75 = NaN;
        hGlyphEllipseMinor2SigmaP75 = NaN;
        hGlyphEllipseMajor2SigmaP995 = NaN;
        updateHorizontalGlyphLegendLabels();
        updateHorizontalLegendLabels();
        drawHorizontalGlyphLegend();
        return;
    }

    const speeds = hCells
        .map(c => Number(c.speed_mm_yr))
        .filter(v => Number.isFinite(v) && v > 0.0)
        .sort((a, b) => a - b);

    const thetaValues = hCells
        .map(c => fieldSigmaThetaDeg({
            east: Number(c.east_mm_yr || 0.0),
            north: Number(c.north_mm_yr || 0.0),
            var_east: Number(c.var_east || 0.0),
            var_north: Number(c.var_north || 0.0),
            covar_en: Number(c.covar_en || 0.0),
        }))
        .filter(v => Number.isFinite(v) && v >= 0.0)
        .sort((a, b) => a - b);

    const ellipseMajor2Sigma = hCells
        .map(c => covarianceStdMajorMinor(c.var_east, c.var_north, c.covar_en).major * H_ELLIPSE_SIGMA_MULTIPLIER)
        .filter(v => Number.isFinite(v) && v > 0.0)
        .sort((a, b) => a - b);

    const ellipseMinor2Sigma = hCells
        .map(c => covarianceStdMajorMinor(c.var_east, c.var_north, c.covar_en).minor * H_ELLIPSE_SIGMA_MULTIPLIER)
        .filter(v => Number.isFinite(v) && v > 0.0)
        .sort((a, b) => a - b);

    hLegendSpeedP75 = percentileFromSorted(speeds, 0.75);
    hLegendSpeedP995 = percentileFromSorted(speeds, 0.995);
    hLegendSigmaThetaP75 = percentileFromSorted(thetaValues, 0.75);
    hGlyphEllipseMajor2SigmaP75 = percentileFromSorted(ellipseMajor2Sigma, 0.75);
    hGlyphEllipseMinor2SigmaP75 = percentileFromSorted(ellipseMinor2Sigma, 0.75);
    hGlyphEllipseMajor2SigmaP995 = percentileFromSorted(ellipseMajor2Sigma, 0.995);

    updateHorizontalGlyphLegendLabels();
    updateHorizontalLegendLabels();
    drawHorizontalGlyphLegend();

    console.log("[H legend] P75 stats", {
        speedP75MmYr: hLegendSpeedP75,
        speedP995MmYr: hLegendSpeedP995,
        sigmaThetaP75Deg: hLegendSigmaThetaP75,
        ellipseMajor2SigmaP75MmYr: hGlyphEllipseMajor2SigmaP75,
        ellipseMajor2SigmaP995MmYr: hGlyphEllipseMajor2SigmaP995,
        source: "horizontal particle field cells"
    });
}

function targetHLegendParticleCount() {
    const count = Number(hParticleCount || 0);
    // Scale the real particle-count slider down to a tiny legend canvas.
    return Math.max(8, Math.min(48, Math.round(count / 180.0)));
}

function resizeHLegendCanvas() {
    if (!hLegendCanvas || !hLegendCtx || !hLegendBar) return false;

    const rect = hLegendBar.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1.0;
    const targetW = Math.max(1, Math.floor(rect.width * dpr));
    const targetH = Math.max(1, Math.floor(rect.height * dpr));

    if (hLegendCanvas.width !== targetW || hLegendCanvas.height !== targetH) {
        hLegendCanvas.width = targetW;
        hLegendCanvas.height = targetH;
        hLegendCanvas.style.width = `${rect.width}px`;
        hLegendCanvas.style.height = `${rect.height}px`;
        hLegendCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        hLegendCtx.clearRect(0, 0, rect.width, rect.height);
        hLegendParticles = [];
        return true;
    }

    return false;
}

function seedHLegendParticle(width, height, randomX = true) {
    return {
        x: randomX ? hParticleRandom() * width : -8 - hParticleRandom() * width * 0.15,
        yBase: 5 + hParticleRandom() * Math.max(6, height - 10),
        y: null,
        prevX: null,
        prevY: null,
        phase: hParticleRandom() * Math.PI * 2.0,
        freq: 0.8 + hParticleRandom() * 1.4,
        ampScale: 0.55 + hParticleRandom() * 0.75,
    };
}

function ensureHLegendParticles(width, height) {
    const target = targetHLegendParticleCount();
    while (hLegendParticles.length < target) {
        hLegendParticles.push(seedHLegendParticle(width, height, true));
    }
    while (hLegendParticles.length > target) {
        hLegendParticles.pop();
    }
}

function drawHorizontalLegend(timestamp) {
    if (!hLegendCanvas || !hLegendCtx || !hLegendBar) {
        hLegendAnimationId = null;
        return;
    }

    resizeHLegendCanvas();

    const rect = hLegendBar.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const midX = width * 0.5;

    ensureHLegendParticles(width, height);

    if (!hLegendLastTimestamp) hLegendLastTimestamp = timestamp;
    let dt = (timestamp - hLegendLastTimestamp) / 1000.0;
    hLegendLastTimestamp = timestamp;
    dt = Math.min(Math.max(dt, 0.0), 0.05);

    // Match the real particle trail persistence slider.
    hLegendCtx.save();
    hLegendCtx.globalCompositeOperation = "destination-out";
    hLegendCtx.fillStyle = `rgba(255,255,255,${(1.0 - hParticleTrailPersistence).toFixed(3)})`;
    hLegendCtx.fillRect(0, 0, width, height);
    hLegendCtx.restore();

    const speedRatio = Number.isFinite(hLegendSpeedP75) && Number.isFinite(hSpeedP95) && hSpeedP95 > 0
        ? Math.max(0.15, Math.min(1.45, hLegendSpeedP75 / hSpeedP95))
        : 0.55;

    // Visual legend speed: uses P75 horizontal velocity and the same speed slider.
    const pxPerSecond = (42.0 + 58.0 * speedRatio) * Math.max(0.1, Number(hParticleSpeedMultiplier || 1.0));

    const thetaNorm = Number.isFinite(hLegendSigmaThetaP75)
        ? Math.max(0.0, Math.min(1.0, hLegendSigmaThetaP75 / Math.max(H_UNCERTAINTY_THETA_HIGH_DEG, 1e-6)))
        : 0.55;
    const maxWobblePx = Math.max(0.0, hUncertaintyStrength) * (1.2 + 6.8 * thetaNorm);

    hLegendCtx.save();
    hLegendCtx.globalCompositeOperation = "source-over";
    hLegendCtx.lineCap = "round";
    hLegendCtx.lineJoin = "round";

    for (const p of hLegendParticles) {
        p.x += pxPerSecond * dt;
        if (p.x > width + 12) {
            Object.assign(p, seedHLegendParticle(width, height, false));
        }

        let drawY = p.yBase;
        if (p.x >= midX) {
            const t = timestamp * 0.001;
            const uncertaintyRamp = Math.min(1.0, Math.max(0.0, (p.x - midX) / Math.max(12.0, width * 0.18)));
            drawY += Math.sin(t * Math.PI * 2.0 * p.freq + p.phase) * maxWobblePx * p.ampScale * uncertaintyRamp;
        }

        const alpha = 0.38 + 0.35 * speedRatio;
        const lw = (0.75 + 0.55 * speedRatio) * Math.max(0.1, hParticleSizeMultiplier);

        if (p.prevX !== null && p.prevY !== null) {
            const jump = Math.hypot(p.x - p.prevX, drawY - p.prevY);
            if (jump < width * 0.45) {
                hLegendCtx.strokeStyle = `rgba(238,238,238,${alpha.toFixed(3)})`;
                hLegendCtx.lineWidth = lw;
                hLegendCtx.beginPath();
                hLegendCtx.moveTo(p.prevX, p.prevY);
                hLegendCtx.lineTo(p.x, drawY);
                hLegendCtx.stroke();
            }
        }

        p.prevX = p.x;
        p.prevY = drawY;
    }

    hLegendCtx.restore();
    hLegendAnimationId = requestAnimationFrame(drawHorizontalLegend);
}

function ensureHorizontalLegendAnimation() {
    if (!hLegendCanvas || !hLegendCtx) return;
    resizeHLegendCanvas();
    updateHorizontalLegendLabels();
    if (!hLegendAnimationId) {
        hLegendLastTimestamp = performance.now();
        hLegendAnimationId = requestAnimationFrame(drawHorizontalLegend);
    }
}

function hUncertaintyWobblePx(field, particle, timestampMs) {
    if (!isHParticleShimmerMode() || !field || !particle) return 0.0;

    const thetaDeg = fieldSigmaThetaDeg(field);

    // H7_0 distribution with speed floor 0.50:
    // P75 ≈ 21.4°, P90 ≈ 28.5°, P95 ≈ 32.6°.
    // Start subtle shimmer below P75, approach max near P95.
    const amount = smoothstep(H_UNCERTAINTY_THETA_LOW_DEG, H_UNCERTAINTY_THETA_HIGH_DEG, thetaDeg);
    if (amount <= 0.0) return 0.0;

    const timeSec = timestampMs * 0.001;
    const phase = particle.uncPhase || 0.0;
    const freq = particle.uncFreq || 1.0;

    // Zero-mean, non-accumulating render-only sinusoidal offset.
    return (
        H_UNCERTAINTY_MAX_WOBBLE_PX *
        hUncertaintyStrength *
        amount *
        Math.sin(2.0 * Math.PI * freq * timeSec + phase)
    );
}

function offsetLonLatMeters(lon, lat, eastM, northM) {
    const latRad = Cesium.Math.toRadians(lat);
    const metersPerDegLat = 111320.0;
    const metersPerDegLon = Math.max(1e-9, 111320.0 * Math.cos(latRad));

    return {
        lon: lon + eastM / metersPerDegLon,
        lat: lat + northM / metersPerDegLat
    };
}

// Batch 9.3.3: renderer-neutral particle simulation helpers.
// These functions are the seam we will later move into a standalone module.
function hParticleFrameDeltaSeconds(timestamp) {
    if (!hLastTimestamp) hLastTimestamp = timestamp;
    let dt = (timestamp - hLastTimestamp) / 1000.0;
    hLastTimestamp = timestamp;
    return Math.min(Math.max(dt, 0.0), 0.05);
}

function randomNormal01() {
    // Box-Muller with safety against log(0).
    const u1 = Math.max(1e-12, hParticleRandom());
    const u2 = hParticleRandom();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function resetHParticleMonteCarloRealization(p) {
    if (!p) return;
    p.mcZ1 = clampHMcZ(randomNormal01());
    p.mcZ2 = clampHMcZ(randomNormal01());
}

function covarianceRealizationOffset(field, z1, z2) {
    const veRaw = Number(field?.var_east || 0.0);
    const vnRaw = Number(field?.var_north || 0.0);
    const cRaw = Number(field?.covar_en || 0.0);

    if (!Number.isFinite(veRaw) || !Number.isFinite(vnRaw) || !Number.isFinite(cRaw)) {
        if (hMcFrameStats) hMcFrameStats.invalidCovariance += 1;
        return { east: 0.0, north: 0.0 };
    }

    const ve = Math.max(0.0, veRaw);
    const vn = Math.max(0.0, vnRaw);
    const c = cRaw;

    // Symmetric 2x2 eigen decomposition, with PSD clamping.
    const mid = 0.5 * (ve + vn);
    const diff = 0.5 * (ve - vn);
    const root = Math.sqrt(Math.max(0.0, diff * diff + c * c));
    const lambda1Raw = mid + root;
    const lambda2Raw = mid - root;
    if (lambda1Raw < -1e-12 || lambda2Raw < -1e-12) {
        if (hMcFrameStats) hMcFrameStats.psdClamps += 1;
    }
    const lambda1 = Math.max(0.0, lambda1Raw);
    const lambda2 = Math.max(0.0, lambda2Raw);

    if (lambda1 <= 0.0 && lambda2 <= 0.0) return { east: 0.0, north: 0.0 };

    // Eigenvector angle for the major axis.
    const theta = 0.5 * Math.atan2(2.0 * c, ve - vn);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    const a = Math.sqrt(lambda1) * Number(z1 || 0.0);
    const b = Math.sqrt(lambda2) * Number(z2 || 0.0);

    // R * [a, b], where minor eigenvector is perpendicular to major.
    return {
        east: cosT * a - sinT * b,
        north: sinT * a + cosT * b
    };
}

function hParticleMotionField(field, particle) {
    if (!field || !isHParticleMonteCarloMode() || !particle) return field;

    if (!Number.isFinite(particle.mcZ1) || !Number.isFinite(particle.mcZ2)) {
        resetHParticleMonteCarloRealization(particle);
    }

    const meanEast = Number(field.east || 0.0);
    const meanNorth = Number(field.north || 0.0);
    const meanSpeed = Math.sqrt(meanEast * meanEast + meanNorth * meanNorth);
    let offset = covarianceRealizationOffset(field, particle.mcZ1, particle.mcZ2);

    if (hMonteCarloModel === "directional") {
        // Keep along-flow motion readable. Only use the uncertainty component
        // perpendicular to the mean path. This shows divergence around the flow
        // without turning low-speed particles into visual spaghetti.
        const denom = Math.max(meanSpeed, H_UNCERTAINTY_SPEED_FLOOR_MM_YR, 1e-9);
        const nx = -meanNorth / denom;
        const ny =  meanEast / denom;
        const perp = offset.east * nx + offset.north * ny;
        offset = { east: nx * perp, north: ny * perp };
    }

    const scale = Math.max(0.0, Number(hUncertaintyStrength || 0.0));
    let scaledEastOffset = offset.east * scale;
    let scaledNorthOffset = offset.north * scale;
    let offsetSpeed = Math.sqrt(scaledEastOffset * scaledEastOffset + scaledNorthOffset * scaledNorthOffset);

    if (hMonteCarloModel === "directional" || hMonteCarloModel === "capped_full") {
        const cap = Math.max(
            Number(hMonteCarloOffsetCapMmYr || 0.0),
            meanSpeed * Number(hMonteCarloOffsetCapRatioToSpeed || 0.0)
        );
        if (cap > 0.0 && offsetSpeed > cap) {
            const f = cap / Math.max(offsetSpeed, 1e-12);
            scaledEastOffset *= f;
            scaledNorthOffset *= f;
            offsetSpeed = cap;
            if (hMcFrameStats) hMcFrameStats.cappedOffsets += 1;
        }
    }

    const east = meanEast + scaledEastOffset;
    const north = meanNorth + scaledNorthOffset;
    const realizedSpeed = Math.sqrt(east * east + north * north);

    if (hMcFrameStats) {
        hMcFrameStats.samples += 1;
        hMcFrameStats.sumOffsetSpeedMmYr += offsetSpeed;
        hMcFrameStats.maxOffsetSpeedMmYr = Math.max(hMcFrameStats.maxOffsetSpeedMmYr, offsetSpeed);
        hMcFrameStats.sumRealizedSpeedMmYr += realizedSpeed;
        hMcFrameStats.maxRealizedSpeedMmYr = Math.max(hMcFrameStats.maxRealizedSpeedMmYr, realizedSpeed);
    }

    return {
        ...field,
        east,
        north,
        speed: realizedSpeed,
        monteCarlo: true,
        monteCarloModel: hMonteCarloModel
    };
}

function hParticleVelocityMetersPerSecond(field) {
    const speedRef = Math.max(hSpeedP95, 1e-9);
    return {
        eastMps: (field.east / speedRef) * H_PARTICLE_BASE_MPS * hParticleSpeedMultiplier,
        northMps: (field.north / speedRef) * H_PARTICLE_BASE_MPS * hParticleSpeedMultiplier
    };
}

function hParticleIsUsableField(field) {
    return Boolean(field && Number.isFinite(field.speed) && field.speed >= H_PARTICLE_STALL_SPEED_MM_YR);
}

function hParticleAdvanceLonLat(p, field, dt) {
    const velocity = hParticleVelocityMetersPerSecond(field);
    const next = offsetLonLatMeters(p.lon, p.lat, velocity.eastMps * dt, velocity.northMps * dt);
    p.lon = next.lon;
    p.lat = next.lat;
}

function randomLife() {
    return 2.5 + hParticleRandom() * 4.0;
}

function respawnHParticle(p) {
    if (hValidSpawnCells.length === 0 || !hAffine) return;

    for (let attempt = 0; attempt < 12; attempt++) {
        const cell = hValidSpawnCells[Math.floor(hParticleRandom() * hValidSpawnCells.length)];

        // Spawn within approximately one RUM cell in grid coordinates.
        const gi = cell.grid_i + (hParticleRandom() - 0.5) * 0.9;
        const gj = cell.grid_j + (hParticleRandom() - 0.5) * 0.9;
        const ll = gridToLonLat(gi, gj);

        const sample = sampleHorizontalField(ll.lon, ll.lat);
        if (!sample || sample.speed < H_PARTICLE_STALL_SPEED_MM_YR) continue;

        p.lon = ll.lon;
        p.lat = ll.lat;
        p.age = hParticleRandom() * randomLife();
        p.life = randomLife();
        p.prevX = null;
        p.prevY = null;
        p.prevTrueX = null;
        p.prevTrueY = null;
        resetHParticleMonteCarloRealization(p);
        refreshHParticleSurfaceState(p);
        return;
    }

    // Last-resort fallback to a valid cell centre.
    const cell = hValidSpawnCells[Math.floor(hParticleRandom() * hValidSpawnCells.length)];
    p.lon = cell.lon;
    p.lat = cell.lat;
    p.age = 0;
    p.life = randomLife();
    p.prevX = null;
    p.prevY = null;
    p.prevTrueX = null;
    p.prevTrueY = null;
    resetHParticleMonteCarloRealization(p);
    refreshHParticleSurfaceState(p);
}

function initHParticles(count = hParticleCount) {
    hParticles = [];
    resetHMonteCarloStats();
    for (let i = 0; i < count; i++) {
        const p = {
            lon: 0,
            lat: 0,
            heightM: DISPLAY_DATUM_HEIGHT_M + H_PARTICLE_SURFACE_OFFSET_M,
            age: 0,
            life: randomLife(),
            prevX: null,
            prevY: null,
            prevTrueX: null,
            prevTrueY: null,
            mcZ1: randomNormal01(),
            mcZ2: randomNormal01(),
            uncPhase: hParticleRandom() * Math.PI * 2.0,
            uncFreq: H_UNCERTAINTY_FREQ_MIN_HZ +
                hParticleRandom() * (H_UNCERTAINTY_FREQ_MAX_HZ - H_UNCERTAINTY_FREQ_MIN_HZ)
        };
        respawnHParticle(p);
        hParticles.push(p);
    }
    clearHParticlesCanvas();
    syncHPrimitiveParticleCollection();
}

function hSamplerLabel() {
    if (hParticleSamplerMode === "nearest") return "nearest";
    if (hParticleSamplerMode === "conservative_v1") return "conservative bilinear v1";
    if (hParticleSamplerMode === "conservative_v2") return "conservative bilinear v2";
    if (hParticleSamplerMode === "bilinear") return "loose bilinear";
    return hParticleSamplerMode;
}

function setH4Status() {
    setStatus(`loaded ✓  |  H particles: ${hParticleEngineLabel()} / ${hSamplerLabel()} / ${hParticleUncertaintyLabel()} + Step 17 arrows/ellipses`);
}

function rebuildHValidSpawnCells() {
    if (!hCells || hCells.length === 0) {
        hValidSpawnCells = [];
        return;
    }

    if (hParticleSamplerMode === "conservative_v2") {
        // Strict/scientific test:
        // spawn only where the full local support exists.
        hValidSpawnCells = hCells.filter(c =>
            Number(c.speed_mm_yr) > H_PARTICLE_STALL_SPEED_MM_YR &&
            hasEightNeighborSupport(c)
        );
    } else {
        // Original behaviour for nearest, conservative v1, and loose bilinear:
        // spawn throughout measured RUM support, excluding only near-zero cells.
        hValidSpawnCells = hCells.filter(c =>
            Number(c.speed_mm_yr) > H_PARTICLE_STALL_SPEED_MM_YR
        );
    }
}

function firstFiniteNumber(obj, keys, fallback = NaN) {
    if (!obj) return fallback;
    for (const key of keys) {
        let value;
        if (key.includes(".")) {
            value = key.split(".").reduce((acc, part) => acc && acc[part], obj);
        } else {
            value = obj[key];
        }
        const n = Number(value);
        if (Number.isFinite(n)) return n;
    }
    return fallback;
}

function firstDefinedValue(obj, keys, fallback = undefined) {
    if (!obj) return fallback;
    for (const key of keys) {
        let value;
        if (key.includes(".")) {
            value = key.split(".").reduce((acc, part) => acc && acc[part], obj);
        } else {
            value = obj[key];
        }
        if (value !== undefined && value !== null && value !== "") return value;
    }
    return fallback;
}

function normalizeHorizontalParticleCells(field, heightMeta) {
    const raw = Array.isArray(field)
        ? field
        : Array.isArray(field?.cells)
            ? field.cells
            : Array.isArray(field?.records)
                ? field.records
                : Array.isArray(field?.features)
                    ? field.features.map(f => ({ ...(f.properties || {}), ...(f.geometry || {}) }))
                    : [];

    const out = [];

    for (const rec0 of raw) {
        const rec = rec0 && rec0.properties ? { ...rec0.properties, ...(rec0.geometry || {}) } : rec0;
        if (!rec) continue;

        const rumIdRaw = firstDefinedValue(rec, ["rum_id", "RUM_ID", "rumId", "id", "name"]);
        const rumId = rumIdRaw !== undefined && rumIdRaw !== null ? String(rumIdRaw) : "";

        const gridI = firstFiniteNumber(rec, [
            "grid_i", "gridI", "i", "col", "grid_col", "grid_column", "column", "rum_i"
        ]);
        const gridJ = firstFiniteNumber(rec, [
            "grid_j", "gridJ", "j", "row", "grid_row", "rum_j"
        ]);

        let lon = firstFiniteNumber(rec, [
            "lon", "latlon.lon", "lon_center", "center_lon", "longitude", "x_lon"
        ]);
        let lat = firstFiniteNumber(rec, [
            "lat", "latlon.lat", "lat_center", "center_lat", "latitude", "y_lat"
        ]);

        // GeoJSON Point fallback: coordinates = [lon, lat, ...]
        if ((!Number.isFinite(lon) || !Number.isFinite(lat)) && Array.isArray(rec.coordinates)) {
            lon = Number(rec.coordinates[0]);
            lat = Number(rec.coordinates[1]);
        }

        const east = firstFiniteNumber(rec, [
            "east_mm_yr", "east", "ve_mm_yr", "v_east_mm_yr", "velocity_east", "E_mm_yr"
        ]);
        const north = firstFiniteNumber(rec, [
            "north_mm_yr", "north", "vn_mm_yr", "v_north_mm_yr", "velocity_north", "N_mm_yr"
        ]);
        const speed = firstFiniteNumber(rec, ["speed_mm_yr", "horizontal_speed_mm_yr", "speed", "magnitude_mm_yr"]);

        const varEast = firstFiniteNumber(rec, [
            "var_east", "variance_east", "covariance.var_east", "covariance.var_e", "cov_ee", "varE"
        ], 0.0);
        const varNorth = firstFiniteNumber(rec, [
            "var_north", "variance_north", "covariance.var_north", "covariance.var_n", "cov_nn", "varN"
        ], 0.0);
        const covarEn = firstFiniteNumber(rec, [
            "covar_en", "cov_en", "covariance.covar_en", "covariance.cov_en", "covariance.cov_east_north", "cov_en_mm_yr2"
        ], 0.0);

        let heightRow = firstFiniteNumber(rec, ["height_row", "row_v", "row_index", "texture_row", "rum_index"], NaN);
        if ((!Number.isFinite(heightRow) || heightRow < 0) && heightMeta && heightMeta.rum_index && rumId) {
            const maybeRow = Number(heightMeta.rum_index[rumId]);
            if (Number.isFinite(maybeRow)) heightRow = maybeRow;
        }

        if (!Number.isFinite(gridI) || !Number.isFinite(gridJ)) continue;
        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        if (!Number.isFinite(east) || !Number.isFinite(north)) continue;

        const speedOut = Number.isFinite(speed) ? speed : Math.sqrt(east * east + north * north);

        out.push({
            ...rec,
            rum_id: rumId,
            grid_i: Math.round(gridI),
            grid_j: Math.round(gridJ),
            lon,
            lat,
            east_mm_yr: east,
            north_mm_yr: north,
            speed_mm_yr: speedOut,
            var_east: Math.max(0.0, varEast),
            var_north: Math.max(0.0, varNorth),
            covar_en: covarEn,
            height_row: Number.isFinite(heightRow) ? heightRow : -1
        });
    }

    return out;
}

async function loadHorizontalParticleField() {
    try {
        const particleFieldUrl = HORIZONTAL_PARTICLE_FIELD_URL || HORIZONTAL_FIELD_URL;
        hField = await loadJson(particleFieldUrl, "load horizontal particle field");
        hCells = normalizeHorizontalParticleCells(hField, hHeightMeta);

        if (!hCells.length) {
            throw new Error("horizontal particle field loaded, but no particle-compatible cells were found");
        }

        hLookup = new Map();
        hCellByRumId = new Map();
        for (const c of hCells) {
            hLookup.set(`${c.grid_i},${c.grid_j}`, c);
            if (c.rum_id !== undefined && c.rum_id !== null && c.rum_id !== "") {
                hCellByRumId.set(String(c.rum_id), c);
            }
        }

        const speeds = hCells
            .map(c => Number(c.speed_mm_yr))
            .filter(v => Number.isFinite(v) && v > 0.0)
            .sort((a, b) => a - b);
        const p95Idx = speeds.length ? Math.min(speeds.length - 1, Math.floor(0.95 * (speeds.length - 1))) : -1;

        hSpeedP95 =
            Number(hField.stats?.speed_p95_mm_yr) ||
            Number(hField.stats?.speed_mm_yr?.p95) ||
            (p95Idx >= 0 ? speeds[p95Idx] : 2.0) ||
            2.0;

        updateHorizontalLegendStatsFromCells();
        ensureHorizontalLegendAnimation();

        hAffine = fitGridAffine(hCells);
        if (rumGridVisible) buildRumGridCollection();

        // Build spawn cells according to the selected interpolation mode.
        // Previous three modes keep the original broad spawn behaviour.
        // Conservative v2 alone uses strict supported-domain spawning.
        rebuildHValidSpawnCells();

        console.log("[H particles] Loaded horizontal particle field", {
            sourceUrl: HORIZONTAL_PARTICLE_FIELD_URL || HORIZONTAL_FIELD_URL,
            schema: hField?.metadata?.schema || null,
            cells: hCells.length,
            spawnCells: hValidSpawnCells.length,
            speedP95: hSpeedP95,
            affine: hAffine,
            dynamicSurfaceHeight: hHeightTextureReady,
            surfaceOffsetM: H_PARTICLE_SURFACE_OFFSET_M,
            samplerMode: hParticleSamplerMode,
            stallSpeedMmYr: H_PARTICLE_STALL_SPEED_MM_YR,
            spawnRule: hParticleSamplerMode === "conservative_v2"
                ? "speed > stall threshold and full 8-neighbor support"
                : "speed > stall threshold only",
            horizontalUncertainty: {
                enabled: hUncertaintyEnabled,
                strength: hUncertaintyStrength,
                speedFloorMmYr: H_UNCERTAINTY_SPEED_FLOOR_MM_YR,
                thetaLowDeg: H_UNCERTAINTY_THETA_LOW_DEG,
                thetaHighDeg: H_UNCERTAINTY_THETA_HIGH_DEG,
                maxWobblePx: H_UNCERTAINTY_MAX_WOBBLE_PX,
                freqHz: [H_UNCERTAINTY_FREQ_MIN_HZ, H_UNCERTAINTY_FREQ_MAX_HZ]
            },
            note: "var/cov fields are used for H7 render-only particle shimmer"
        });

        initHParticles(hParticleCount);
        setHorizontalParticlesVisible(H_PARTICLES_ENABLED_INITIAL);

    } catch (error) {
        console.warn("[H particles] load failed/skipped:", error);
        if (hParticleToggleBtn) {
            setLayerToggleButtonState(hParticleToggleBtn, false, "Horizontal particles missing");
            hParticleToggleBtn.disabled = true;
        }
        hParticleCanvas.style.display = "none";
    }
}

function pauseHParticlesForCameraInteraction() {
    if (!hParticlesVisible) return;

    // Primitive point particles are scene-space objects. Keep them flowing during
    // camera motion so Phase 0 can test spatial anchoring and occlusion honestly.
    if (isHParticlePrimitiveMode()) return;

    hCameraMoving = true;

    if (hCameraStableTimer) {
        clearTimeout(hCameraStableTimer);
        hCameraStableTimer = null;
    }

    clearHParticlesCanvas();
    resetHParticleScreenHistory();
}

function resumeHParticlesAfterCameraStable() {
    if (!hParticlesVisible) return;
    if (isHParticlePrimitiveMode()) return;

    if (hCameraStableTimer) {
        clearTimeout(hCameraStableTimer);
        hCameraStableTimer = null;
    }

    hCameraStableTimer = setTimeout(function() {
        // Do not resume while the mouse/finger is still held down
        // or while Cesium reports the camera is still moving.
        if (hPointerDown || hCesiumCameraMoving) {
            return;
        }

        hCameraMoving = false;

        // Start fresh in the new view.
        rebuildHValidSpawnCells();
        initHParticles(hParticleCount);
        resetHParticleScreenHistory();
        clearHParticlesCanvas();

        hLastTimestamp = performance.now();

        if (hParticlesVisible && !hAnimationId) {
            hAnimationId = requestAnimationFrame(drawHorizontalParticles);
        }
    }, H_PARTICLE_CAMERA_STABLE_DELAY_MS);
}

function beginHParticleCameraInteraction() {
    if (cinematic3DActive) {
        stopCinematic3D(false);
    }
    hPointerDown = true;
    pauseHParticlesForCameraInteraction();
}

function endHParticleCameraInteraction() {
    hPointerDown = false;
    resumeHParticlesAfterCameraStable();
}

function markHParticleCameraMoving() {
    // Use this for wheel/inertial/changed events where there may be no mouse-down state.
    pauseHParticlesForCameraInteraction();

    if (!hPointerDown && !hCesiumCameraMoving) {
        resumeHParticlesAfterCameraStable();
    }
}

function drawHorizontalParticles(timestamp) {
    if (!hParticlesVisible) {
        hAnimationId = null;
        return;
    }

    if (isHParticlePrimitiveMode()) {
        drawHorizontalParticlesPrimitivePoints(timestamp);
        return;
    }

    resizeHParticleCanvas();

    if (hCameraMoving) {
        clearHParticlesCanvas();
        resetHParticleScreenHistory();
        hLastTimestamp = timestamp;
        hAnimationId = requestAnimationFrame(drawHorizontalParticles);
        return;
    }

    // Renderer-neutral sim timing. Avoid huge particle jumps after tab switching / debugger pauses.
    const dt = hParticleFrameDeltaSeconds(timestamp);
    if (isHParticleMonteCarloMode()) beginHMonteCarloFrameStats();

    // Fade old trails on a transparent canvas.
    hCtx.save();
    hCtx.globalCompositeOperation = "destination-out";
    hCtx.fillStyle = `rgba(255,255,255,${(1.0 - hParticleTrailPersistence).toFixed(3)})`;
    hCtx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    hCtx.restore();

    hCtx.save();
    hCtx.globalCompositeOperation = "source-over";
    hCtx.lineCap = "round";
    hCtx.lineJoin = "round";

    const scratch = new Cesium.Cartesian2();

    for (const p of hParticles) {
        p.age += dt;

        const field = sampleHorizontalField(p.lon, p.lat);

        if (!field || field.speed < H_PARTICLE_STALL_SPEED_MM_YR || p.age > p.life) {
            respawnHParticle(p);
            if (hMcFrameStats) hMcFrameStats.respawns += 1;
            continue;
        }

        // Visual advection. Speed is normalized by p95 so outliers do not dominate.
        // V4.1 Monte Carlo mode perturbs the path-level velocity here; shimmer mode
        // leaves this mean path untouched and only offsets the rendered screen point.
        const motionField = hParticleMotionField(field, p);
        hParticleAdvanceLonLat(p, motionField, dt);

        const after = sampleHorizontalField(p.lon, p.lat);
        if (!after) {
            respawnHParticle(p);
            if (hMcFrameStats) hMcFrameStats.respawns += 1;
            continue;
        }

        const particleHeight = refreshHParticleSurfaceState(p);
        const cart = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, particleHeight);
        const screen = Cesium.SceneTransforms.worldToWindowCoordinates(viewer.scene, cart, scratch);

        if (!screen || !Number.isFinite(screen.x) || !Number.isFinite(screen.y)) {
            p.prevX = null;
            p.prevY = null;
            p.prevTrueX = null;
            p.prevTrueY = null;
            continue;
        }

        const alpha = Math.max(0.15, Math.min(0.85, 0.20 + 0.55 * (field.speed / Math.max(hSpeedP95, 1e-9))));
        const lw = Math.max(0.7, Math.min(1.8, 0.7 + 0.6 * (field.speed / Math.max(hSpeedP95, 1e-9)))) * Math.max(0.1, hParticleSizeMultiplier);

        let drawX = screen.x;
        let drawY = screen.y;

        // H7 uncertainty shimmer:
        // previous TRUE screen position defines the local path normal.
        // Wobble changes only the rendered point, never p.lon/p.lat.
        if (p.prevTrueX !== null && p.prevTrueY !== null) {
            const tdx = screen.x - p.prevTrueX;
            const tdy = screen.y - p.prevTrueY;
            const tlen = Math.sqrt(tdx * tdx + tdy * tdy);

            if (tlen > 1e-3) {
                const nx = -tdy / tlen;
                const ny =  tdx / tlen;
                const wobble = hUncertaintyWobblePx(field, p, timestamp);

                drawX += nx * wobble;
                drawY += ny * wobble;
            }
        }

        if (p.prevX !== null && p.prevY !== null) {
            const dx = drawX - p.prevX;
            const dy = drawY - p.prevY;
            const jump = Math.sqrt(dx * dx + dy * dy);

            // v8 smooth mode: restore the old permissive trail rule.
            // Do not break trails just because the vertical MODEL height changed
            // across a cap boundary; that was the main source of visual stutter.
            if (jump < H_PARTICLE_MAX_TRAIL_SCREEN_JUMP_PX) {
                hCtx.strokeStyle = `rgba(100,100,100,${alpha.toFixed(3)})`;
                hCtx.lineWidth = lw;
                hCtx.beginPath();
                hCtx.moveTo(p.prevX, p.prevY);
                hCtx.lineTo(drawX, drawY);
                hCtx.stroke();
            }
        }

        p.prevTrueX = screen.x;
        p.prevTrueY = screen.y;
        p.prevX = drawX;
        p.prevY = drawY;
    }

    hCtx.restore();
    if (isHParticleMonteCarloMode()) finishHMonteCarloFrameStats();

    hAnimationId = requestAnimationFrame(drawHorizontalParticles);
}

function logHPrimitivePointsDebug(timestamp, shownCount, respawnCount) {
    if (!isHParticlePrimitiveMode()) return;
    if (!hPrimitivePointsDebugEnabled) return;
    if (!Number.isFinite(timestamp)) return;
    if (timestamp - hPrimitiveLastDebugLogMs < H_PRIMITIVE_POINTS_DEBUG_LOG_INTERVAL_MS) return;
    hPrimitiveLastDebugLogMs = timestamp;

    // Keep this intentionally sparse: one line every few seconds while testing.
    const sample = hParticles.length ? hParticles[Math.floor(hParticles.length / 2)] : null;
    console.log("[H particles primitive points] runtime", {
        particles: hParticles.length,
        collectionLength: hPrimitiveParticleCollection ? hPrimitiveParticleCollection.length : 0,
        shown: shownCount,
        respawnedThisFrame: respawnCount,
        heightContract: "MODEL surface at particle XY/current epoch + offset",
        sampleHeightM: sample && Number.isFinite(sample.heightM) ? Number(sample.heightM.toFixed(2)) : null,
        surfaceOffsetM: H_PARTICLE_SURFACE_OFFSET_M,
        depthTest: "enabled; disableDepthTestDistance=0"
    });
}

function drawHorizontalParticlesPrimitivePoints(timestamp) {
    const collection = syncHPrimitiveParticleCollection();
    if (!collection) {
        hAnimationId = requestAnimationFrame(drawHorizontalParticles);
        return;
    }

    hParticleCanvas.style.display = "none";
    collection.show = true;

    const dt = hParticleFrameDeltaSeconds(timestamp);

    let shownCount = 0;
    let respawnCount = 0;

    for (let idx = 0; idx < hParticles.length; idx++) {
        const p = hParticles[idx];
        const point = collection.get(idx);
        p.age += dt;

        let field = sampleHorizontalField(p.lon, p.lat);
        if (!field || field.speed < H_PARTICLE_STALL_SPEED_MM_YR || p.age > p.life) {
            respawnHParticle(p);
            respawnCount++;
            field = sampleHorizontalField(p.lon, p.lat);
        }

        if (!field || field.speed < H_PARTICLE_STALL_SPEED_MM_YR) {
            if (point) point.show = false;
            continue;
        }

        const motionField = hParticleMotionField(field, p);
        hParticleAdvanceLonLat(p, motionField, dt);

        const after = sampleHorizontalField(p.lon, p.lat);
        if (!after) {
            respawnHParticle(p);
            respawnCount++;
            field = sampleHorizontalField(p.lon, p.lat);
            if (!field) {
                if (point) point.show = false;
                continue;
            }
        } else {
            field = after;
        }

        updateHPrimitivePoint(point, p, field);
        if (point && point.show) shownCount++;
    }

    logHPrimitivePointsDebug(timestamp, shownCount, respawnCount);
    viewer.scene.requestRender();
    hAnimationId = requestAnimationFrame(drawHorizontalParticles);
}

function setHorizontalParticlesVisible(visible) {
    hParticlesVisible = visible;
    setLayerToggleButtonState(hParticleToggleBtn, visible, "Horizontal particles");
    hParticleCanvas.style.display = visible && !isHParticlePrimitiveMode() ? "block" : "none";
    hParticleCanvas.style.opacity = String(hParticleOpacity);

    if (hPrimitiveParticleCollection) {
        hPrimitiveParticleCollection.show = visible && isHParticlePrimitiveMode();
    }

    if (visible) {
        if (isHParticlePrimitiveMode()) {
            syncHPrimitiveParticleCollection();
        } else {
            resizeHParticleCanvas();
        }
        resetHParticleScreenHistory();
        hLastTimestamp = performance.now();
        if (!hAnimationId) {
            hAnimationId = requestAnimationFrame(drawHorizontalParticles);
        }
    } else {
        if (hAnimationId) {
            cancelAnimationFrame(hAnimationId);
            hAnimationId = null;
        }
        clearHParticlesCanvas();
        hideHPrimitiveParticles();
    }
    syncLegendActiveStates();
}

function setHorizontalUncertaintyVisible(visible) {
    hUncertaintyEnabled = visible;
    if (!visible) {
        hParticleUncertaintyMode = "off";
    } else if (hParticleUncertaintyMode === "off") {
        hParticleUncertaintyMode = "shimmer";
    }
    resetHParticleScreenHistory();
    clearHParticlesCanvas();
    updateHorizontalLegendLabels();
    setH4Status();
    syncLegendActiveStates();
}

function applyHParticleControls() {
    hParticleSamplerMode = hParticleSamplerSelect ? hParticleSamplerSelect.value : hParticleSamplerMode;
    hParticleCount = hParticleCountSlider ? Number(hParticleCountSlider.value) : hParticleCount;
    hParticleSizeMultiplier = hParticleSizeSlider ? Number(hParticleSizeSlider.value) : hParticleSizeMultiplier;
    hParticleSpeedMultiplier = hParticleSpeedSlider ? Number(hParticleSpeedSlider.value) : hParticleSpeedMultiplier;
    hParticleTrailPersistence = hParticleTrailSlider ? Number(hParticleTrailSlider.value) : hParticleTrailPersistence;
    hUncertaintyStrength = hUncertaintyStrengthSlider ? Number(hUncertaintyStrengthSlider.value) : hUncertaintyStrength;
    hParticleOpacity = hParticleOpacitySlider ? Number(hParticleOpacitySlider.value) : hParticleOpacity;

    if (hParticleSamplerValue) hParticleSamplerValue.textContent = hSamplerLabel();
    if (hParticleCountValue) hParticleCountValue.textContent = String(hParticleCount);
    if (hParticleSizeValue) hParticleSizeValue.textContent = hParticleSizeMultiplier.toFixed(2);
    if (hParticleSpeedValue) hParticleSpeedValue.textContent = hParticleSpeedMultiplier.toFixed(1);
    if (hParticleTrailValue) hParticleTrailValue.textContent = hParticleTrailPersistence.toFixed(2);
    if (hUncertaintyStrengthValue) {
        hUncertaintyStrengthValue.textContent = hUncertaintyStrength.toFixed(2);
        const label = hUncertaintyStrengthValue.closest("label");
        const nameSpan = label ? label.querySelector("span:first-child") : null;
        if (nameSpan) {
            nameSpan.textContent = hParticleUncertaintyMode === "montecarlo"
                ? "MC strength"
                : hParticleUncertaintyMode === "shimmer"
                    ? "Shimmer strength"
                    : "Uncertainty strength";
        }
    }
    updateHorizontalLegendLabels();
    if (hParticleOpacityValue) hParticleOpacityValue.textContent = hParticleOpacity.toFixed(2);
    if (hParticleCanvas) hParticleCanvas.style.opacity = String(hParticleOpacity);
    if (hPrimitiveParticleCollection) {
        hPrimitiveParticleCollection.show = hParticlesVisible && isHParticlePrimitiveMode();
        for (let i = 0; i < hPrimitiveParticleCollection.length; i++) {
            const point = hPrimitiveParticleCollection.get(i);
            point.pixelSize = H_PRIMITIVE_POINTS_PIXEL_SIZE * Math.max(0.1, hParticleSizeMultiplier);
            point.color = new Cesium.Color(
                H_PRIMITIVE_POINTS_COLOR_RGB[0],
                H_PRIMITIVE_POINTS_COLOR_RGB[1],
                H_PRIMITIVE_POINTS_COLOR_RGB[2],
                Math.max(0.0, Math.min(1.0, hParticleOpacity))
            );
        }
    }
}
