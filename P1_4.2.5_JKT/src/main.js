import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

import proj4 from 'proj4';
import {COORDINATE_SYSTEM, MapView} from '@deck.gl/core';
import {MapboxOverlay} from '@deck.gl/mapbox';
import {PathLayer, SolidPolygonLayer} from '@deck.gl/layers';
import {SimpleMeshLayer} from '@deck.gl/mesh-layers';
import {ContextCapLayer} from './context_cap_layer.js';
import {HorizontalParticleLayer} from './horizontal_particle_layer.js';
import {createHorizontalLegendRenderer} from './horizontal_legend_renderer.js';
import {createStudioPolygonAnnotations} from '../_internal/studio_mode/studio_polygon_annotations.js';
import {createStudioCaptureMode} from '../_internal/studio_mode/studio_capture_mode.js';
import tuDelftLogoUrl from '../_internal/assets/tu_delft_logo_color.png';
import compassNeedleUrl from '../_internal/assets/compass_arrow2.png';

import {buildDatumGround, buildTopology, cellKey} from './geometry.js';
import {createCheckerboardReliefMesh, deriveFootprintTransform} from './uncertainty_relief_mesh.js';
import {
  createArrowHeadMesh,
  createArrowShaftMesh,
  createConfidenceEllipseMesh,
} from './horizontal_glyph_mesh.js';
import {
  assignContextCornerUvs,
  buildRasterAtlas,
  computeStudyBounds,
  createContextCapQuadMesh,
} from './context_cap_mesh.js';
import './style.css';

// -----------------------------------------------------------------------------
// Proto1 DeckGL — real Jakarta RUM surface.
//
// Completed support-envelope + uncertainty relief pass:
// - preserves the accepted irregular pit / datum-ground geometry;
// - keeps blankies as moving IDW support, distinct from observed RUMs;
// - renders V7.2-inspired 4×4 vertical uncertainty as one DeckGL-native
//   instanced mesh, with amplitude-aware visual fading rather than a Cesium/GLB shader port.
// -----------------------------------------------------------------------------

const DATA_ROOT = new URL(
  `${import.meta.env.BASE_URL}data/jakarta/`,
  window.location.href,
);

function runtimeAssetUrl(relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(`Invalid runtime asset path: ${String(relativePath)}`);
  }
  return new URL(relativePath, DATA_ROOT).href;
}
const WALL_EPSILON_M = 0.0001;
const RIM_EPSILON_M = 0.01;
const DATUM_LINE_Z = 0.12;
const DATUM_LINE_COLOR = [248, 215, 103, 255];
const APRON_COLOR = [203, 198, 188, 255];
const APRON_INNER_LIP = [150, 144, 132, 255];
const BLANKIE_CAP_COLOR = [130, 132, 130, 142];
const BLANKIE_WALL_COLOR = [96, 98, 96, 168];
const COLOR_NEUTRAL = [247, 247, 247];
const COLOR_SUBSIDENCE = [103, 0, 31];
const COLOR_UPLIFT = [5, 48, 97];
const DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE = Object.freeze({
  field: 'up',
  unit: 'mm/yr',
  mode: 'fallback_manual',
  nearZeroThresholdMmYr: 1,
  subsidenceLimitMmYr: 50,
  upliftLimitMmYr: 10,
  stops: Object.freeze([
    {valueMmYr: -50, color: COLOR_SUBSIDENCE, role: 'clipped_subsidence_limit', positionPct: 0},
    {valueMmYr: -40, color: [178, 24, 43], role: 'subsidence_gradient'},
    {valueMmYr: -20, color: [214, 96, 77], role: 'subsidence_gradient'},
    {valueMmYr: -1, color: COLOR_NEUTRAL, role: 'stable_boundary_low'},
    {valueMmYr: 0, color: COLOR_NEUTRAL, role: 'zero_reference'},
    {valueMmYr: 1, color: COLOR_NEUTRAL, role: 'stable_boundary_high'},
    {valueMmYr: 4, color: [67, 147, 195], role: 'uplift_gradient'},
    {valueMmYr: 8, color: [33, 102, 172], role: 'uplift_gradient'},
    {valueMmYr: 10, color: COLOR_UPLIFT, role: 'clipped_uplift_limit', positionPct: 100},
  ]),
  legend: {
    title: 'Vertical velocity · mm/yr',
    labels: [
      {valueMmYr: -50, positionPct: 0, label: '≤ −50'},
      {valueMmYr: -1, positionPct: 42, label: '−1'},
      {valueMmYr: 0, positionPct: 50, label: '0'},
      {valueMmYr: 1, positionPct: 58, label: '+1'},
      {valueMmYr: 10, positionPct: 100, label: '≥ +10'},
    ],
    note: 'Fallback vertical-velocity display scale.',
  },
});
const FAR_Z_MULTIPLIER_FALLBACK = 8;
const NEAR_Z_MULTIPLIER_FALLBACK = 0.1;
const GRAZING_FAR_Z_MULTIPLIER_FALLBACK = 16;
const GRAZING_PITCH_START_FALLBACK = 60;
// Relief carries controlled per-facet tint in its shared mesh. Keeping this
// material unlit preserves the exact scientific cap colour on all flat faces;
// camera/map lighting cannot turn the mean surface into a darker material.
const RELIEF_MATERIAL = {unlit: true};
const CONTEXT_CAP_MATERIAL = {unlit: true};
const CONTEXT_CAP_TEXTURE_PARAMETERS = {
  // deck.gl creates the atlas texture with a full mip chain. This sampler
  // uses trilinear minification instead of a home-grown zoom tile system.
  minFilter: 'linear',
  magFilter: 'linear',
  mipmapFilter: 'linear',
  maxAnisotropy: 4,
  addressModeU: 'clamp-to-edge',
  addressModeV: 'clamp-to-edge',
};
const CONTEXT_LIVE_TINT_ALPHA = 116;
const CONTEXT_BLANKIE_TINT_COLOR = [119, 123, 119, 154];
const HORIZONTAL_GLYPH_MATERIAL = {unlit: true};
const DEFAULT_TWO_D_ANALYSIS = Object.freeze({
  rumFillOpacity: 0.66,
  rumFillOpacityMin: 0.30,
  rumFillOpacityMax: 0.90,
  rumFillOpacityStep: 0.02,
  preferredBasemapMode: 'bw',
  rumOutlineRgba: [20, 29, 38, 72],
  rumOutlineWidthPixels: 0.65,
  flatParticleZM: 0,
});

let runtime = null;
let liveCells = [];
let blankieCells = [];
let structuralCells = [];
let cellsByKey = new Map();
let topology = null;
let datumGround = null;
let referenceGridMode = 'off';
let referenceGridPlanPaths = [];
let referenceGridFrameCorners = [];
let activeLiveWalls = [];
let activeBlankieWalls = [];
let activeLiveRimWalls = [];
let activeBlankieRimWalls = [];
let activeEpoch = 0;
let verticalExaggeration = 10;
let apronMode = 'see-through';
let datumLineEnabled = false;
let depthOccludersEnabled = true;
// Camera-depth contract: base clip range stays tight for ordinary inspection;
// only grazing views receive a modest, controlled far-plane extension.
let farZMultiplier = FAR_Z_MULTIPLIER_FALLBACK;
let nearZMultiplier = NEAR_Z_MULTIPLIER_FALLBACK;
let grazingFarZMultiplier = GRAZING_FAR_Z_MULTIPLIER_FALLBACK;
let grazingPitchStartDeg = GRAZING_PITCH_START_FALLBACK;
let activeFarZMultiplier = FAR_Z_MULTIPLIER_FALLBACK;
let uncertaintyReliefEnabled = true;
let reliefMesh = null;
let reliefMeshSets = {far: [], near: []};
let activeReliefLodKey = 'far';
let activeReliefCellPixels = 0;
let activeLiveReliefBuckets = [];
let reliefVisualFade = {
  startEffectiveReliefM: 0.5,
  fullEffectiveReliefM: 10.0,
  minimumRenderWeight: 0.01,
  bucketCount: 8,
};
let reliefLod = {
  enabled: true,
  farGridN: 2,
  nearGridN: 4,
  nearMinCellPixels: 48,
  farMaxCellPixels: 36,
  grazingPitchForceFar: 66,
};
let playbackSpeedMultiplier = 1;
let basemapMode = 'map';
// Basemap choices are remembered per scene. 2D begins with the quiet Soft
// B/W reference map, while 3D restores whatever context the user left there.
let threeDBasemapMode = 'map';
let twoDBasemapMode = null;
let capAppearance = 'scientific';
let showPistonWalls = true;
let showBlankieCaps = true;
let reliefBeforeContextMode = true;
let contextCapConfig = {
  atlasZoom: 13,
  atlasMaxDimension: 4096,
  atlasPaddingFraction: 0.025,
  atlasMipmaps: true,
  atlasMaxAnisotropy: 4,
  liveTintAlpha: CONTEXT_LIVE_TINT_ALPHA,
  blankieTintColor: CONTEXT_BLANKIE_TINT_COLOR,
  atlasLod: {
    enabled: true,
    focusAtlasZoom: 14,
    focusAtlasMaxDimension: 6144,
    focusAtlasMaxTiles: 384,
    focusEnterMapZoom: 13.0,
    overviewReturnMapZoom: 12.6,
  },
};
// Context imagery is deliberately a two-level fixed-atlas system, not a
// reimplementation of MapLibre's live tile lifecycle. Atlas A covers the
// full study extent at overview detail. Atlas B uses the same fixed extent at
// higher source zoom and swaps only at one hysteretic focus boundary.
let contextAtlases = {
  overview: {state: 'idle', atlas: null, error: null, progress: null},
  focus: {state: 'idle', atlas: null, error: null, progress: null},
};
let activeContextAtlasKey = 'overview';
let contextAtlas = null;
let contextAtlasState = 'idle';
let contextStudyBounds = null;
let contextCapMesh = null;
let horizontalGlyphRecords = [];
let horizontalGlyphConfig = {
  enabled: true,
  showArrowsByDefault: true,
  showEllipsesByDefault: true,
  defaultOpacity: 0.92,
  arrowColorRgba: [34, 34, 34, 240],
  ellipseColorRgba: [0, 240, 216, 210],
  clearanceAboveCapM: 6,
  ellipseSegments: 64,
  ellipseRingInnerRadius: 0.94,
  legend: {
    speedP75MmYr: NaN,
    ellipseMajorP75MmYr: NaN,
    ellipseMinorP75MmYr: NaN,
    arrowReferenceMmYr: NaN,
    ellipseMajorReferenceMmYr: NaN,
    confidenceProbability: 1 - Math.exp(-0.5),
    sigmaMultiplier: 1,
  },
  ellipseSourceConfidenceScale: 1,
  ellipseDisplayFactor: 1,
  ellipseLegendVisualScale: 1 / Math.sqrt(-2 * Math.log(1 - 0.95)),
};
let horizontalGlyphMeshes = {shaft: null, head: null, ellipse: null};
let showHorizontalArrows = true;
let showHorizontalEllipses = true;
let horizontalGlyphOpacity = 0.92;
let horizontalGlyphScale = 1.0;
let horizontalParticleRuntime = null;
let horizontalParticleLscRuntime = null;
let horizontalParticleFieldMode = 'raw';
let horizontalParticleConfig = {
  enabled: true,
  showByDefault: true,
  defaultMode: 'mean',
  particleCapacity: 12000,
  defaultParticleCount: 5000,
  speedMultiplier: 1.5,
  particleSizeMultiplier: 1.0,
  particleOpacity: 1.0,
  trailPersistence: 0.98,
  historySampleIntervalS: 0.05,
  historySamplesMin: 9,
  historySamplesMax: 65,
  historySamples: 32,
  uncertaintyStrength: 0.5,
  shimmerStrength: 0.5,
  monteCarloStrength: 0.5,
  samplerMode: 'conservative_v1',
};
let showHorizontalParticles = true;
let horizontalParticleMode = 'mean';
let horizontalParticleCount = 5000;
let horizontalParticleSpeedMultiplier = 1.5;
let horizontalParticleSizeMultiplier = 1.0;
let horizontalParticleOpacity = 1.0;
let horizontalParticleTrailPersistence = 0.98;
let horizontalParticleHistorySamples = 32;
let horizontalParticleTrailDurationSeconds = 1.55;
let horizontalParticleUncertaintyStrengths = {shimmer: 0.5, montecarlo: 0.5};
let horizontalParticleUncertaintyStrength = 0.0;
let horizontalParticleGpuStatus = null;
// Scene modes are two different layer contracts, not a camera preset.
// 2D Analysis keeps only static observed vertical-rate cells plus horizontal
// motion encodings on z=0; 3D Time Scene keeps the full animated surface.
let sceneMode = '3d';
let twoDAnalysisConfig = {...DEFAULT_TWO_D_ANALYSIS};
let savedThreeDCamera = null;
let savedTwoDCamera = null;
let isPlaying = false;
let playbackFrame = null;
let lastPlaybackTime = null;

const epochSlider = document.querySelector('#epochSlider');
const epochLabel = document.querySelector('#epochLabel');
const verticalExagSlider = document.querySelector('#verticalExagSlider');
const verticalExagValue = document.querySelector('#verticalExagValue');
const playButton = document.querySelector('#playButton');
const playbackSpeedSlider = document.querySelector('#playbackSpeedSlider');
const playbackSpeedValue = document.querySelector('#playbackSpeedValue');
const apronModeControl = document.querySelector('#apronModeControl');
const datumLineToggle = document.querySelector('#datumLineToggle');
const depthOccluderToggle = document.querySelector('#depthOccluderToggle');
const cameraDiagnostic = document.querySelector('#cameraDiagnostic');
const readingNote = document.querySelector('#readingNote');
const focusLabel = document.querySelector('#focusLabel');
const tooltip = document.querySelector('#tooltip');
let selectedCell = null;
let selectedCellIsBlankie = false;
let selectedTooltipExpanded = false;
let selectedTooltipPosition = {x: 0, y: 0};
let trendlineOpen = false;
let trendlineCell = null;
let trendlineAxisMode = 'auto';
let trendlineCustomMin = NaN;
let trendlineCustomMax = NaN;
let trendlineHeightPx = 170;
const TRENDLINE_MIN_HEIGHT_PX = 120;
const TRENDLINE_MAX_HEIGHT_PX = 320;
let trendlineProjectRangeCache = null;
let trendlineDrawFrame = null;
let trendlineResizeDrag = null;
let suppressNextMapClickClear = false;
const epochPanel = document.querySelector('#epochPanel');
const rumTrendlinePanel = document.querySelector('#rumTrendlinePanel');
const rumTrendlineCanvas = document.querySelector('#rumTrendlineCanvas');
const rumTrendlineTitle = document.querySelector('#rumTrendlineTitle');
const rumTrendlineSubtitle = document.querySelector('#rumTrendlineSubtitle');
const rumTrendlineAxisModeSelect = document.querySelector('#rumTrendlineAxisModeSelect');
const rumTrendlineMinInput = document.querySelector('#rumTrendlineMinInput');
const rumTrendlineMaxInput = document.querySelector('#rumTrendlineMaxInput');
const rumTrendlineResizeHandle = document.querySelector('#rumTrendlineResizeHandle');
const rumTrendlinePngButton = document.querySelector('#rumTrendlinePngButton');
const rumTrendlineCloseButton = document.querySelector('#rumTrendlineCloseButton');
const fpsCounter = document.querySelector('#fpsCounter');
const frameMsCounter = document.querySelector('#frameMsCounter');
const tuDelftLogo = document.querySelector('#tuDelftLogo');
const bottomDistanceScaleBar = document.querySelector('#bottomDistanceScaleBar');
const bottomDistanceScaleLabel = document.querySelector('#bottomDistanceScaleLabel');
const bottomCameraLabel = document.querySelector('#bottomCameraLabel');
const bottomCoordLabel = document.querySelector('#bottomCoordLabel');
const compassBubbleButton = document.querySelector('#compassBubbleButton');
const compassNeedle = document.querySelector('#compassNeedle');
const horizontalGlyphLegendBar = document.querySelector('#horizontalGlyphLegendBar');
const horizontalGlyphLegendCanvas = document.querySelector('#horizontalGlyphLegendCanvas');
const horizontalGlyphLegendBarArrowText = document.querySelector('#horizontalGlyphLegendBarArrowText');
const horizontalGlyphLegendBarEllipseText = document.querySelector('#horizontalGlyphLegendBarEllipseText');
const horizontalParticleLegendBar = document.querySelector('#horizontalParticleLegendBar');
const horizontalParticleLegendCanvas = document.querySelector('#horizontalParticleLegendCanvas');
const horizontalParticleLegendBarSpeedText = document.querySelector('#horizontalParticleLegendBarSpeedText');
const horizontalParticleLegendBarUncertaintyText = document.querySelector('#horizontalParticleLegendBarUncertaintyText');
const horizontalLegendRenderer = createHorizontalLegendRenderer({
  glyphCanvas: horizontalGlyphLegendCanvas,
  particleCanvas: horizontalParticleLegendCanvas,
});
const verticalUncertaintyLegendBar = document.querySelector('#verticalUncertaintyLegendBar');
const verticalUncertaintyLegendSvg = document.querySelector('#verticalUncertaintyLegendSvg');
const verticalUncertaintyLegendMidline = document.querySelector('#verticalUncertaintyLegendMidline');
const verticalUncertaintyLegendProfile = document.querySelector('#verticalUncertaintyLegendProfile');
const verticalUncertaintyLegendZeroLabel = document.querySelector('#verticalUncertaintyLegendZeroLabel');
const verticalUncertaintyLegendRightLabel = document.querySelector('#verticalUncertaintyLegendRightLabel');
const verticalUncertaintyLegendTitle = document.querySelector('#verticalUncertaintyLegendTitle');
const verticalUncertaintyLegendProvenanceTag = document.querySelector('#verticalUncertaintyLegendProvenanceTag');
const verticalUncertaintyLegendBarText = document.querySelector('#verticalUncertaintyLegendBarText');
const sceneModeControl = document.querySelector('#sceneModeControl');
const sceneModeNote = document.querySelector('#sceneModeNote');
const helpText = document.querySelector('#helpText');
const uncertaintyReliefToggle = document.querySelector('#uncertaintyReliefToggle');
const reliefDiagnostic = document.querySelector('#reliefDiagnostic');
const basemapControl = document.querySelector('#basemapControl');
const sceneGridModeSelect = document.querySelector('#sceneGridModeSelect');
const twoDRumOpacitySlider = document.querySelector('#twoDRumOpacitySlider');
const twoDRumOpacityValue = document.querySelector('#twoDRumOpacityValue');
const twoDAnalysisLegend = document.querySelector('#twoDAnalysisLegend');
const verticalVelocityLegendTitle = document.querySelector('#verticalVelocityLegendTitle');
const verticalVelocityLegendScale = document.querySelector('#verticalVelocityLegendScale');
const verticalVelocityLegendLabels = document.querySelector('#verticalVelocityLegendLabels');
const verticalVelocityLegendNote = document.querySelector('#verticalVelocityLegendNote');
const capAppearanceControl = document.querySelector('#capAppearanceControl');
const capAppearanceSelect = document.querySelector('#capAppearanceSelect');
const pistonWallsToggle = document.querySelector('#pistonWallsToggle');
const blankieCapsToggle = document.querySelector('#blankieCapsToggle');
const contextCapDiagnostic = document.querySelector('#contextCapDiagnostic');
const horizontalArrowsToggle = document.querySelector('#horizontalArrowsToggle');
const horizontalEllipsesToggle = document.querySelector('#horizontalEllipsesToggle');
const horizontalGlyphOpacitySlider = document.querySelector('#horizontalGlyphOpacitySlider');
const horizontalGlyphOpacityValue = document.querySelector('#horizontalGlyphOpacityValue');
const horizontalGlyphScaleSlider = document.querySelector('#horizontalGlyphScaleSlider');
const horizontalGlyphScaleValue = document.querySelector('#horizontalGlyphScaleValue');
const horizontalGlyphDiagnostic = document.querySelector('#horizontalGlyphDiagnostic');
const horizontalGlyphLegend = document.querySelector('#horizontalGlyphLegend');
const horizontalParticlesToggle = document.querySelector('#horizontalParticlesToggle');
const horizontalParticleModeControl = document.querySelector('#horizontalParticleModeControl');
const horizontalParticleFieldModeBlock = document.querySelector('#horizontalParticleFieldModeBlock');
const horizontalParticleFieldModeControl = document.querySelector('#horizontalParticleFieldModeControl');
const horizontalParticleFieldNote = document.querySelector('#horizontalParticleFieldNote');
const horizontalParticleMcCaveat = document.querySelector('#horizontalParticleMcCaveat');
const horizontalParticleCountSlider = document.querySelector('#horizontalParticleCountSlider');
const horizontalParticleCountValue = document.querySelector('#horizontalParticleCountValue');
const horizontalParticleSizeSlider = document.querySelector('#horizontalParticleSizeSlider');
const horizontalParticleSizeValue = document.querySelector('#horizontalParticleSizeValue');
const horizontalParticleSpeedSlider = document.querySelector('#horizontalParticleSpeedSlider');
const horizontalParticleSpeedValue = document.querySelector('#horizontalParticleSpeedValue');
const horizontalParticleTrailDurationSlider = document.querySelector('#horizontalParticleTrailDurationSlider');
const horizontalParticleTrailDurationValue = document.querySelector('#horizontalParticleTrailDurationValue');
const horizontalParticleTrailPersistenceSlider = document.querySelector('#horizontalParticleTrailPersistenceSlider');
const horizontalParticleTrailPersistenceValue = document.querySelector('#horizontalParticleTrailPersistenceValue');
const horizontalParticleOpacitySlider = document.querySelector('#horizontalParticleOpacitySlider');
const horizontalParticleOpacityValue = document.querySelector('#horizontalParticleOpacityValue');
const horizontalParticleUncertaintyLabel = document.querySelector('#horizontalParticleUncertaintyLabel');
const horizontalParticleUncertaintySlider = document.querySelector('#horizontalParticleUncertaintySlider');
const horizontalParticleUncertaintyValue = document.querySelector('#horizontalParticleUncertaintyValue');
const horizontalParticleSamplerValue = document.querySelector('#horizontalParticleSamplerValue');
const horizontalParticleDiagnostic = document.querySelector('#horizontalParticleDiagnostic');
const horizontalParticleLegend = document.querySelector('#horizontalParticleLegend');

// UI-B2 shell: compact right-side control drawer, timeline, and native-canvas utilities.
const viewerShell = document.querySelector('#viewerShell');
const rightControlRoot = document.querySelector('#rightControlRoot');
const rightDrawerBurger = document.querySelector('#rightDrawerBurger');
const drawerDefaultsButton = document.querySelector('#drawerDefaultsButton');
const rightDrawerPanel = document.querySelector('#rightDrawerPanel');
const rightDrawerScroll = document.querySelector('#rightDrawerScroll');
const drawerSectionTitles = Array.from(document.querySelectorAll('#rightDrawerPanel .drawerSectionTitle'));
const drawerSections = Array.from(document.querySelectorAll('#rightDrawerPanel .drawerSection'));
const drawerDisplay = document.querySelector('#drawerDisplay');
const miniViewerCanvas = document.querySelector('#miniViewerCanvas');
const miniViewerWell = document.querySelector('#studioMiniViewerWell');
const miniViewerStatus = document.querySelector('#miniViewerStatus');
const drawerMiniViewerSection = document.querySelector('#drawerMiniViewer');
const scenePolygonToggle = document.querySelector('#scenePolygonToggle');
const studioPolygonAddButton = document.querySelector('#studioPolygonAddButton');
const studioPolygonStatus = document.querySelector('#studioPolygonStatus');
const studioPolygonDrawBar = document.querySelector('#studioPolygonDrawBar');
const studioPolygonDrawStatus = document.querySelector('#studioPolygonDrawStatus');
const studioPolygonUndoButton = document.querySelector('#studioPolygonUndoButton');
const studioPolygonFinishButton = document.querySelector('#studioPolygonFinishButton');
const studioPolygonCancelButton = document.querySelector('#studioPolygonCancelButton');
const studioPolygonSaveForm = document.querySelector('#studioPolygonSaveForm');
const studioPolygonNameInput = document.querySelector('#studioPolygonNameInput');
const studioPolygonInfoInput = document.querySelector('#studioPolygonInfoInput');
const studioPolygonSaveButton = document.querySelector('#studioPolygonSaveButton');
const studioPolygonFormCancelButton = document.querySelector('#studioPolygonFormCancelButton');
const studioPolygonList = document.querySelector('#studioPolygonList');
const studioCaptureAccordion = document.querySelector('#studioCaptureAccordion');
const studioCaptureStatus = document.querySelector('#studioCaptureStatus');
const studioCapturePanel = document.querySelector('#studioCapturePanel');
const studioCaptureViewfinderOverlay = document.querySelector('#studioViewfinderOverlay');
const studioCaptureViewfinderToggle = document.querySelector('#studioCaptureViewfinderToggle');
const studioCaptureCurrentViewButton = document.querySelector('#studioCaptureCurrentViewButton');
const studioCaptureIntroButton = document.querySelector('#studioCaptureIntroButton');
const studioCapturePreviewButton = document.querySelector('#studioCapturePreviewButton');
const studioCaptureClearButton = document.querySelector('#studioCaptureClearButton');
const studioCaptureList = document.querySelector('#studioCaptureList');
let miniViewerBounds = null;
let miniViewerTransform = null;
let miniViewerDrawFrame = null;
let miniViewerCameraFootprintCache = null;
let miniViewerCameraDirty = true;
let miniViewerCameraFrozen = false;
let miniViewerCameraIdleTimer = null;
const MINI_VIEWER_CAMERA_IDLE_DELAY_MS = 520;

// UI-B2D Batch 1.12 — accordion master switch model.
// The CSS owns the lamp/binder shape. Main.js owns the click contract and
// the actual layer visibility bridge, so the title still opens/collapses while
// the binder lamp toggles the whole section on/off.
const DRAWER_MASTER_OFF_CLASS = 'sectionOff';
const DRAWER_MASTER_FEATURE_IDS = Object.freeze({
  scene: ['drawerScene', 'drawerLayers'],
  vertical: ['drawerVertical', 'drawerVerticalPistons', 'drawerPistons'],
  horizontalGlyphs: ['drawerHStatic', 'drawerHorizontalGlyphs', 'drawerHorizontalGlyph', 'drawerGlyphs'],
  horizontalParticles: ['drawerHDynamic', 'drawerHorizontalParticles', 'drawerParticles'],
  miniViewer: ['drawerMiniViewer'],
});

const zoomInButton = document.querySelector('#zoomInButton');
const zoomOutButton = document.querySelector('#zoomOutButton');
const resetViewButton = document.querySelector('#resetViewButton');
const flyToRumsButton = document.querySelector('#flyToRumsButton');
const viewModeToggleButton = document.querySelector('#viewModeToggleButton');
const screenshotButton = document.querySelector('#screenshotButton');
const fullscreenButton = document.querySelector('#fullscreenButton');
const navInfoButton = document.querySelector('#navInfoButton');
const viewerInfoPanel = document.querySelector('#viewerInfoPanel');
const viewerInfoCloseButton = document.querySelector('#viewerInfoCloseButton');
const screenshotStatus = document.querySelector('#screenshotStatus');
const epochFirstButton = document.querySelector('#epochFirstButton');
const epochPrevButton = document.querySelector('#epochPrevButton');
const epochNextButton = document.querySelector('#epochNextButton');
const epochLastButton = document.querySelector('#epochLastButton');
let screenshotStatusTimer = null;
let activeDrawerId = 'drawerScene';
let studioPolygons = null;
let studioCaptureMode = null;
let bottomStatusPointerLngLat = null;
let bottomStatusFrame = null;

epochSlider.disabled = true;
verticalExagSlider.disabled = true;
playButton.disabled = true;
if (playbackSpeedSlider) playbackSpeedSlider.disabled = true;
function setTrendlineHeight(heightPx) {
  trendlineHeightPx = clamp(Math.round(Number(heightPx)), TRENDLINE_MIN_HEIGHT_PX, TRENDLINE_MAX_HEIGHT_PX);
  const value = `${trendlineHeightPx}px`;
  if (epochPanel) epochPanel.style.setProperty('--trendline-chart-height', value);
  if (rumTrendlinePanel) rumTrendlinePanel.style.setProperty('--trendline-chart-height', value);
  scheduleTrendlineDraw();
}

setTrendlineHeight(trendlineHeightPx);
if (uncertaintyReliefToggle) uncertaintyReliefToggle.disabled = true;
if (horizontalArrowsToggle) horizontalArrowsToggle.disabled = true;
if (horizontalEllipsesToggle) horizontalEllipsesToggle.disabled = true;
if (horizontalGlyphOpacitySlider) horizontalGlyphOpacitySlider.disabled = true;
if (horizontalGlyphScaleSlider) horizontalGlyphScaleSlider.disabled = true;
if (horizontalParticlesToggle) horizontalParticlesToggle.disabled = true;
if (horizontalParticleModeControl) horizontalParticleModeControl.disabled = true;
if (horizontalParticleFieldModeControl) horizontalParticleFieldModeControl.disabled = true;
if (horizontalParticleCountSlider) horizontalParticleCountSlider.disabled = true;
if (horizontalParticleSizeSlider) horizontalParticleSizeSlider.disabled = true;
if (horizontalParticleSpeedSlider) horizontalParticleSpeedSlider.disabled = true;
if (horizontalParticleTrailDurationSlider) horizontalParticleTrailDurationSlider.disabled = true;
if (horizontalParticleTrailPersistenceSlider) horizontalParticleTrailPersistenceSlider.disabled = true;
if (horizontalParticleOpacitySlider) horizontalParticleOpacitySlider.disabled = true;
if (horizontalParticleUncertaintySlider) horizontalParticleUncertaintySlider.disabled = true;
epochLabel.textContent = 'Loading Jakarta runtime…';

function startFpsCounter() {
  if (!fpsCounter && !frameMsCounter) return;

  let frameCount = 0;
  let frameMsSum = 0;
  let measuredFrameCount = 0;
  let windowStart = performance.now();
  let previousFrameAt = windowStart;

  function tick(now) {
    const frameMs = now - previousFrameAt;
    previousFrameAt = now;
    frameCount += 1;
    if (Number.isFinite(frameMs) && frameMs > 0 && frameMs < 250) {
      frameMsSum += frameMs;
      measuredFrameCount += 1;
    }

    const elapsed = now - windowStart;
    if (elapsed >= 500) {
      const fps = Math.round((frameCount * 1000) / elapsed);
      const averageFrameMs = measuredFrameCount > 0 ? frameMsSum / measuredFrameCount : 0;
      const band = fps >= 45 ? 'good' : fps >= 25 ? 'warn' : 'bad';
      if (fpsCounter) {
        fpsCounter.textContent = averageFrameMs > 0
          ? `${averageFrameMs.toFixed(1)} ms / ${fps} FPS`
          : `— ms / ${fps} FPS`;
        fpsCounter.dataset.band = band;
      }
      if (frameMsCounter) {
        frameMsCounter.textContent = averageFrameMs > 0 ? `${averageFrameMs.toFixed(1)} ms` : '— ms';
        frameMsCounter.dataset.band = band;
      }
      frameCount = 0;
      frameMsSum = 0;
      measuredFrameCount = 0;
      windowStart = now;
    }
    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

function installShellAssets() {
  if (tuDelftLogo) tuDelftLogo.src = tuDelftLogoUrl;
  if (compassNeedle) compassNeedle.style.backgroundImage = `url("${compassNeedleUrl}")`;
}

function mapAttributionText(mode = basemapMode) {
  const style = BASEMAPS[mode]?.style;
  const source = style?.sources ? Object.values(style.sources)[0] : null;
  return String(source?.attribution ?? 'Basemap attribution unavailable').replace(/\s+/g, ' ').trim();
}

function formatDistanceMeters(valueMeters) {
  if (!Number.isFinite(valueMeters) || valueMeters <= 0) return '—';
  if (valueMeters >= 1000) {
    const km = valueMeters / 1000;
    return `${km >= 10 ? Math.round(km) : km.toFixed(km >= 2 ? 1 : 2)} km`;
  }
  return `${Math.round(valueMeters)} m`;
}

function niceScaleDistance(targetMeters) {
  if (!Number.isFinite(targetMeters) || targetMeters <= 0) return 0;
  const exponent = Math.floor(Math.log10(targetMeters));
  const base = 10 ** exponent;
  const fraction = targetMeters / base;
  const niceFraction = fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1;
  return niceFraction * base;
}

function mapMetersPerPixel(latitudeDeg, zoom) {
  const latitudeRadians = clamp(Number(latitudeDeg), -85, 85) * Math.PI / 180;
  return 156543.03392804097 * Math.cos(latitudeRadians) / (2 ** Number(zoom));
}

function estimateCameraZMeters(center, zoom, pitchDegrees) {
  const latitude = Number(center?.lat);
  if (!Number.isFinite(latitude) || !Number.isFinite(Number(zoom))) return NaN;
  const canvasHeight = Math.max(1, Number(map?.getContainer?.().clientHeight ?? 0));
  // MapLibre's default vertical FOV is 36.87° (0.6435011088 rad). Use the
  // public getter when available; otherwise keep that stable default.
  const fovRadians = typeof map?.getFov === 'function'
    ? Number(map.getFov())
    : 0.6435011087932844;
  const safeFov = Number.isFinite(fovRadians) && fovRadians > 0 ? fovRadians : 0.6435011087932844;
  const cameraToCenterPixels = (canvasHeight * 0.5) / Math.tan(safeFov * 0.5);
  const distanceAlongViewMeters = cameraToCenterPixels * mapMetersPerPixel(latitude, Number(zoom));
  const pitchRadians = clamp(Number(pitchDegrees), 0, 85) * Math.PI / 180;
  return Math.max(0, distanceAlongViewMeters * Math.cos(pitchRadians));
}

function formatCameraZKilometers(heightMeters) {
  if (!Number.isFinite(heightMeters) || heightMeters < 0) return '—';
  const kilometers = heightMeters / 1000;
  if (kilometers >= 10) return `${Math.round(kilometers)} km`;
  if (kilometers >= 1) return `${kilometers.toFixed(1)} km`;
  return `${kilometers.toFixed(2)} km`;
}

function scheduleBottomStatusUpdate() {
  if (bottomStatusFrame) return;
  bottomStatusFrame = requestAnimationFrame(() => {
    bottomStatusFrame = null;
    updateBottomStatus();
  });
}

function updateBottomStatus() {
  if (!map) return;
  const center = map.getCenter?.();
  const pointer = bottomStatusPointerLngLat ?? center;
  const zoom = map.getZoom?.();
  const bearing = map.getBearing?.() ?? 0;
  const pitch = map.getPitch?.() ?? 0;
  const latitude = Number(pointer?.lat ?? center?.lat);

  if (bottomCoordLabel) {
    bottomCoordLabel.textContent = Number.isFinite(latitude) && Number.isFinite(Number(pointer?.lng))
      ? `Lat: ${latitude.toFixed(5)}, Lon: ${Number(pointer.lng).toFixed(5)}`
      : 'Lat: —, Lon: —';
  }
  if (bottomCameraLabel) {
    const mode = sceneMode === '2d' ? '2D' : '3D';
    const cameraZ = estimateCameraZMeters(center, zoom, pitch);
    bottomCameraLabel.textContent = `${mode} Camera z: ${formatCameraZKilometers(cameraZ)}`;
  }
  if (bottomDistanceScaleBar && bottomDistanceScaleLabel && Number.isFinite(latitude) && Number.isFinite(zoom)) {
    const metersPerPixel = mapMetersPerPixel(latitude, zoom);
    const maxBarPixels = 132;
    const targetMeters = metersPerPixel * maxBarPixels;
    const scaleMeters = niceScaleDistance(targetMeters);
    const barPixels = clamp(scaleMeters / metersPerPixel, 24, maxBarPixels);
    bottomDistanceScaleBar.style.width = `${barPixels.toFixed(1)}px`;
    bottomDistanceScaleLabel.textContent = formatDistanceMeters(scaleMeters);
  }
  if (compassNeedle) compassNeedle.style.transform = `rotate(${-bearing.toFixed(2)}deg)`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep01(value) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function reliefRangeMm(cell) {
  const range = runtime?.uncertaintyRelief?.displayRange ?? {unit: 'sigma', value: 1};
  return range.unit === 'sigma'
    ? cell.sigmaMm * range.value
    : range.value;
}

function formatReliefRange(cell) {
  const range = runtime?.uncertaintyRelief?.displayRange ?? {unit: 'sigma', value: 1};
  const unitLabel = range.unit === 'sigma' ? `${range.value}σ` : `${range.value} mm`;
  return `${unitLabel} = ±${reliefRangeMm(cell).toFixed(2)} mm`;
}

function normalizeReliefVisualFade(source = {}) {
  const startEffectiveReliefM = Math.max(0, Number(source.start_effective_relief_m ?? 0.5));
  const fullEffectiveReliefM = Math.max(
    startEffectiveReliefM + 1e-6,
    Number(source.full_effective_relief_m ?? 10.0),
  );
  return {
    startEffectiveReliefM,
    fullEffectiveReliefM,
    minimumRenderWeight: clamp(Number(source.minimum_render_weight ?? 0.01), 0, 0.5),
    bucketCount: clamp(Math.round(Number(source.buckets ?? 8)), 1, 16),
  };
}

function reliefVisualWeightForAmplitude(amplitudeM) {
  if (!Number.isFinite(amplitudeM) || amplitudeM <= reliefVisualFade.startEffectiveReliefM) {
    return 0;
  }
  const normalized =
    (amplitudeM - reliefVisualFade.startEffectiveReliefM) /
    (reliefVisualFade.fullEffectiveReliefM - reliefVisualFade.startEffectiveReliefM);
  return smoothstep01(normalized);
}

function reliefBucketForWeight(weight) {
  if (!Number.isFinite(weight) || weight <= reliefVisualFade.minimumRenderWeight) return -1;
  return clamp(
    Math.ceil(weight * reliefVisualFade.bucketCount) - 1,
    0,
    reliefVisualFade.bucketCount - 1,
  );
}

function mixTowardMean(value, weight) {
  return 1 + (Number(value ?? 1) - 1) * weight;
}

function createReliefMeshes(geometry, visualFade) {
  const facetTint = geometry?.facet_tint ?? {};
  return Array.from({length: visualFade.bucketCount}, (_, index) => {
    const weight = (index + 1) / visualFade.bucketCount;
    return createCheckerboardReliefMesh({
      ...geometry,
      facet_tint: {
        ...facetTint,
        // Exact scientific colour on mean-surface triangles. Relief tint is
        // blended toward that mean colour for low-amplitude instances.
        flat: 1,
        up_min: mixTowardMean(facetTint.up_min ?? 0.90, weight),
        up_max: mixTowardMean(facetTint.up_max ?? 0.99, weight),
        down_min: mixTowardMean(facetTint.down_min ?? 0.86, weight),
        down_max: mixTowardMean(facetTint.down_max ?? 0.95, weight),
      },
    });
  });
}

function normalizeReliefLod(source = {}, geometry = {}) {
  const nearGridN = clamp(
    Math.round(Number(source.near_grid_n_per_rum ?? geometry.grid_n_per_rum ?? 4)),
    1,
    8,
  );
  const farGridN = clamp(
    Math.round(Number(source.far_grid_n_per_rum ?? 2)),
    1,
    nearGridN,
  );
  const nearMinCellPixels = Math.max(8, Number(source.near_min_cell_pixels ?? 48));
  const farMaxCellPixels = clamp(
    Number(source.far_max_cell_pixels ?? 36),
    4,
    nearMinCellPixels - 0.001,
  );
  const rawGrazingPitchForceFar = Number(source.grazing_pitch_force_far_deg ?? 66);
  const grazingPitchForceFar = Number.isFinite(rawGrazingPitchForceFar)
    ? clamp(rawGrazingPitchForceFar, 0, 89)
    : 66;
  return {
    enabled: source.enabled ?? true,
    farGridN,
    nearGridN,
    nearMinCellPixels,
    farMaxCellPixels,
    grazingPitchForceFar,
    preservePyramidFootprintAcrossLods: source.preserve_pyramid_footprint_across_lods !== false,
    footprintReferenceGridN: clamp(
      Math.round(Number(source.footprint_reference_grid_n_per_rum ?? nearGridN)),
      1,
      8,
    ),
  };
}

function geometryForReliefLod(geometry, targetGridN, lod) {
  const baseGeometry = {...geometry, grid_n_per_rum: targetGridN};
  const preserveFootprint = lod.preservePyramidFootprintAcrossLods !== false;
  if (!preserveFootprint) return baseGeometry;

  const referenceGridN = clamp(
    Math.round(Number(lod.footprintReferenceGridN ?? lod.nearGridN ?? targetGridN)),
    1,
    8,
  );
  const scale = targetGridN / referenceGridN;

  // Each local checkerboard cell is innerSpan / gridN wide. Scaling the
  // fractional footprint by target/reference grid size preserves the physical
  // pyramid base width across LODs: 2×2 becomes sparser, not chunkier.
  return {
    ...baseGeometry,
    up_relief_footprint_fraction: clamp(
      Number(geometry.up_relief_footprint_fraction ?? 0.36) * scale,
      0.05,
      0.95,
    ),
    down_relief_footprint_fraction: clamp(
      Number(geometry.down_relief_footprint_fraction ?? geometry.up_relief_footprint_fraction ?? 0.36) * scale,
      0.05,
      0.95,
    ),
  };
}

function createReliefMeshSets(geometry, visualFade, lod) {
  const farGeometry = geometryForReliefLod(geometry, lod.farGridN, lod);
  const nearGeometry = geometryForReliefLod(geometry, lod.nearGridN, lod);
  return {
    far: createReliefMeshes(farGeometry, visualFade),
    near: createReliefMeshes(nearGeometry, visualFade),
  };
}

function activeReliefMeshes() {
  return reliefMeshSets[activeReliefLodKey] ?? reliefMeshSets.near ?? [];
}

function activeReliefMesh() {
  return activeReliefMeshes().at(-1) ?? null;
}

function estimatedReliefCellPixels() {
  if (!runtime) return 0;
  const rumSizeM = Math.max(1, Number(runtime.grid?.rumSizeM ?? 450));
  const latitude = map.getCenter().lat;
  const zoom = map.getZoom();
  const metresPerPixel =
    (156543.03392804097 * Math.cos((latitude * Math.PI) / 180)) /
    Math.pow(2, zoom);
  return rumSizeM / Math.max(metresPerPixel, 1e-9);
}

function resolveReliefLodKey() {
  if (!reliefLod.enabled || reliefLod.farGridN === reliefLod.nearGridN) return 'near';
  // Horizon/grazing views concentrate many relief facets into a small screen
  // area. Force the sparse mesh before that becomes an overdraw bottleneck.
  if (map.getPitch() >= reliefLod.grazingPitchForceFar) return 'far';
  const pixels = estimatedReliefCellPixels();
  if (activeReliefLodKey === 'near') {
    return pixels <= reliefLod.farMaxCellPixels ? 'far' : 'near';
  }
  return pixels >= reliefLod.nearMinCellPixels ? 'near' : 'far';
}

function reliefLodLabel() {
  const mesh = activeReliefMesh();
  const mode = activeReliefLodKey === 'near' ? 'inspection' : 'overview';
  return mesh ? `${mesh.gridN}×${mesh.gridN} ${mode}` : 'relief unavailable';
}

function updateReliefDiagnostic() {
  if (!reliefDiagnostic) return;
  const mesh = activeReliefMesh();
  if (!mesh) {
    reliefDiagnostic.textContent = 'Relief mesh unavailable.';
    return;
  }
  const pixels = Number.isFinite(activeReliefCellPixels) ? activeReliefCellPixels.toFixed(0) : '—';
  const grazingForced = map.getPitch() >= reliefLod.grazingPitchForceFar;
  reliefDiagnostic.textContent =
    `Measured RUMs only · ${reliefLodLabel()} · ` +
    `${mesh.triangleCount.toLocaleString()} triangles per RUM · ` +
    `${reliefVisualFade.bucketCount} fade bands · ` +
    `${activeReliefLodKey === 'far' && reliefLod.preservePyramidFootprintAcrossLods ? 'same pyramid size, sparse spacing · ' : ''}` +
    `${grazingForced ? 'grazing-angle safeguard · ' : ''}` +
    `~${pixels}px/RUM.`;
}

function syncReliefLod({redraw = true} = {}) {
  if (!runtime) return;
  activeReliefCellPixels = estimatedReliefCellPixels();
  const nextKey = resolveReliefLodKey();
  const changed = nextKey !== activeReliefLodKey;
  activeReliefLodKey = nextKey;
  reliefMesh = activeReliefMesh();
  updateReliefDiagnostic();

  if (changed && redraw) {
    updateReadingNote();
    deckOverlay.setProps({layers: makeLayers()});
  }
}

function resolveFarZMultiplierForPitch(pitchDeg = map.getPitch()) {
  const maxPitch = Math.max(grazingPitchStartDeg + 0.001, runtime?.maxCameraPitch ?? 70);
  if (pitchDeg <= grazingPitchStartDeg || grazingFarZMultiplier <= farZMultiplier) {
    return farZMultiplier;
  }
  const t = smoothstep01((pitchDeg - grazingPitchStartDeg) / (maxPitch - grazingPitchStartDeg));
  return farZMultiplier + (grazingFarZMultiplier - farZMultiplier) * t;
}

function updateCameraDiagnostic() {
  if (!cameraDiagnostic || !runtime) return;
  const pitch = map.getPitch();
  const grazing = pitch >= grazingPitchStartDeg;
  const sparse = pitch >= reliefLod.grazingPitchForceFar;
  cameraDiagnostic.textContent =
    `Camera guard: max ${runtime.maxCameraPitch.toFixed(0)}° · now ${pitch.toFixed(0)}° · ` +
    `near ${nearZMultiplier.toFixed(2)}× / far ${activeFarZMultiplier.toFixed(1)}×` +
    `${grazing ? ' (grazing)' : ''}${sparse ? ' · sparse relief' : ''}. ` +
    'A/B toggle disables alpha-zero depth prepasses.';
}

function syncCameraDepthContract({force = false} = {}) {
  if (!runtime) return;
  const target = resolveFarZMultiplierForPitch();
  // Quantise to 0.25×. This avoids recreating MapView on every pointer-move
  // while still extending the far clip smoothly across the guarded pitch range.
  const quantized = Math.max(1.01, Math.round(target * 4) / 4);
  if (force || Math.abs(quantized - activeFarZMultiplier) >= 0.25) {
    activeFarZMultiplier = quantized;
    deckOverlay.setProps({
      views: new MapView({
        id: 'mapbox',
        nearZMultiplier,
        farZMultiplier: activeFarZMultiplier,
      }),
    });
  }
  updateCameraDiagnostic();
}

function rebuildLiveReliefBuckets() {
  const buckets = Array.from({length: reliefVisualFade.bucketCount}, () => []);
  for (const cell of liveCells) {
    const bucket = reliefBucketForWeight(cell.reliefVisualWeight);
    cell.reliefVisualBucket = bucket;
    if (bucket >= 0) buckets[bucket].push(cell);
  }
  activeLiveReliefBuckets = buckets;
}

function mixColor(a, b, amount) {
  return a.map((value, index) => Math.round(value + (b[index] - value) * amount));
}

function parseHexRgb(value, fallback = [153, 153, 153]) {
  if (Array.isArray(value) && value.length >= 3) {
    return value.slice(0, 3).map((channel, index) => clamp(Math.round(Number(channel ?? fallback[index])), 0, 255));
  }
  const raw = String(value ?? '').trim().replace(/^#/, '');
  const hex = raw.length === 3 ? raw.split('').map((character) => character + character).join('') : raw;
  if (!/^[0-9a-f]{6}$/i.test(hex)) return [...fallback];
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
  ];
}

function normalizeVerticalVelocityColorScale(source = {}) {
  const rawStops = Array.isArray(source.stops) ? source.stops : DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.stops;
  const parsedStops = rawStops
    .map((stop, index) => ({
      valueMmYr: Number(stop.valueMmYr ?? stop.value_mm_yr ?? stop.value),
      color: parseHexRgb(stop.color, DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.stops[
        Math.min(index, DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.stops.length - 1)
      ].color),
      role: String(stop.role ?? ''),
      positionPct: Number.isFinite(Number(stop.positionPct ?? stop.position_pct))
        ? Number(stop.positionPct ?? stop.position_pct)
        : undefined,
    }))
    .filter((stop) => Number.isFinite(stop.valueMmYr))
    .sort((a, b) => a.valueMmYr - b.valueMmYr)
    .filter((stop, index, ordered) => index === 0 || Math.abs(stop.valueMmYr - ordered[index - 1].valueMmYr) > 1e-9);
  const sourceStops = parsedStops.length >= 2
    ? parsedStops
    : DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.stops.map((stop) => ({...stop, color: [...stop.color]}));
  const tau = Math.max(0, Number(
    source.nearZeroThresholdMmYr
      ?? source.near_zero_threshold_mm_yr
      ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.nearZeroThresholdMmYr,
  ));

  // The stable reference band is a real display contract, not legend-only
  // decoration: every velocity in [−tau, +tau] must render white.  We insert
  // the two boundaries when a runtime asset has no explicit stop there, and
  // overwrite any pale shoulder colours exactly at / inside that band.
  const whitePlateauStops = [
    ...sourceStops,
    ...(tau > 0 ? [
      {valueMmYr: -tau, color: [...COLOR_NEUTRAL], role: 'stable_boundary_low'},
      {valueMmYr: 0, color: [...COLOR_NEUTRAL], role: 'zero_reference'},
      {valueMmYr: tau, color: [...COLOR_NEUTRAL], role: 'stable_boundary_high'},
    ] : [{valueMmYr: 0, color: [...COLOR_NEUTRAL], role: 'zero_reference'}]),
  ]
    .sort((a, b) => a.valueMmYr - b.valueMmYr)
    .reduce((accumulator, stop) => {
      const previous = accumulator.at(-1);
      const insideStableBand = tau > 0
        ? stop.valueMmYr >= -tau - 1e-9 && stop.valueMmYr <= tau + 1e-9
        : Math.abs(stop.valueMmYr) <= 1e-9;
      const normalized = {
        ...stop,
        color: insideStableBand ? [...COLOR_NEUTRAL] : [...stop.color],
        role: insideStableBand
          ? (Math.abs(stop.valueMmYr) <= 1e-9 ? 'zero_reference' : stop.valueMmYr < 0 ? 'stable_boundary_low' : 'stable_boundary_high')
          : stop.role,
        positionPct: undefined,
      };
      if (previous && Math.abs(previous.valueMmYr - normalized.valueMmYr) <= 1e-9) {
        accumulator[accumulator.length - 1] = normalized;
      } else {
        accumulator.push(normalized);
      }
      return accumulator;
    }, []);

  const rawLegendLabels = Array.isArray(source.legend?.labels)
    ? source.legend.labels
    : DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.legend.labels;
  const labels = rawLegendLabels
    .map((label) => ({
      valueMmYr: Number(label.valueMmYr ?? label.value_mm_yr ?? label.value),
      positionPct: Number(label.positionPct ?? label.position_pct),
      label: String(label.label ?? ''),
    }))
    .filter((label) => Number.isFinite(label.valueMmYr) && Number.isFinite(label.positionPct) && label.label);
  return {
    ...DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE,
    ...source,
    unit: String(source.unit ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.unit),
    nearZeroThresholdMmYr: tau,
    subsidenceLimitMmYr: Math.max(0, Number(source.subsidenceLimitMmYr ?? source.subsidence_limit_mm_yr ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.subsidenceLimitMmYr)),
    upliftLimitMmYr: Math.max(0, Number(source.upliftLimitMmYr ?? source.uplift_limit_mm_yr ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.upliftLimitMmYr)),
    stops: whitePlateauStops,
    legend: {
      title: String(source.legend?.title ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.legend.title),
      labels: labels.length ? labels : DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.legend.labels,
      note: String(source.legend?.note ?? DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.legend.note),
    },
  };
}

function velocityColor(upMmYr, colorScale) {
  const stops = colorScale?.stops?.length >= 2
    ? colorScale.stops
    : DEFAULT_VERTICAL_VELOCITY_COLOR_SCALE.stops;
  const value = Number(upMmYr);
  if (!Number.isFinite(value)) return [...COLOR_NEUTRAL, 255];
  if (value <= stops[0].valueMmYr) return [...stops[0].color, 255];
  for (let index = 0; index < stops.length - 1; index += 1) {
    const start = stops[index];
    const end = stops[index + 1];
    if (value <= end.valueMmYr) {
      const span = Math.max(end.valueMmYr - start.valueMmYr, 1e-9);
      const amount = clamp((value - start.valueMmYr) / span, 0, 1);
      return [...mixColor(start.color, end.color, amount), 255];
    }
  }
  return [...stops.at(-1).color, 255];
}

function formatVelocityLegendNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || Math.abs(number) < 1e-9) return '0';
  const magnitude = Math.abs(number);
  const digits = magnitude >= 10 ? 0 : magnitude >= 1 ? 1 : 2;
  const compact = magnitude.toFixed(digits).replace(/\.0$/, '').replace(/(\.[0-9]*?)0+$/, '$1').replace(/\.$/, '');
  return number > 0 ? `+${compact}` : `−${compact}`;
}

function darken(color, amount = 0.72) {
  return [
    Math.round(color[0] * amount),
    Math.round(color[1] * amount),
    Math.round(color[2] * amount),
    255,
  ];
}

function formatExaggeration(value) {
  return `${Number(value.toFixed(2))}×`;
}

function formatPlaybackSpeed(value) {
  return `${Number(Number(value).toFixed(2))}×`;
}

async function fetchOrThrow(url, label = 'Runtime asset') {
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) {
    throw new Error(`${label} failed: ${response.status} ${response.statusText} (${url})`);
  }
  return response;
}

function htmlResponseHint(text) {
  return /^\s*<!doctype html|^\s*<html[\s>]/i.test(text);
}

async function fetchJsonOrThrow(url, label) {
  const response = await fetchOrThrow(url, label);
  const text = await response.text();
  if (htmlResponseHint(text)) {
    throw new Error(
      `${label} expected JSON but received HTML from ${url}. ` +
      'The requested runtime asset is missing or the runtime bundle is stale.',
    );
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(
      `${label} is not valid JSON (${url}): ${error.message}. ` +
      `Response starts: ${text.slice(0, 100).replace(/\s+/g, ' ')}`,
    );
  }
}

async function fetchArrayBufferOrThrow(url, label) {
  const response = await fetchOrThrow(url, label);
  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('text/html')) {
    throw new Error(
      `${label} expected binary data but received HTML from ${url}. ` +
      'The requested runtime asset is missing or the runtime bundle is stale.',
    );
  }
  return response.arrayBuffer();
}

function requireManifestAsset(manifest, assetKey, fallbackFileName) {
  const relativePath = manifest?.assets?.[assetKey] ?? fallbackFileName;
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(`Runtime manifest is missing assets.${assetKey}.`);
  }
  return runtimeAssetUrl(relativePath);
}

function requirePayloadAsset(payload, assetKey, label) {
  const relativePath = payload?.assets?.[assetKey];
  if (typeof relativePath !== 'string' || !relativePath.trim()) {
    throw new Error(`${label} is missing assets.${assetKey}. Regenerate runtime assets.`);
  }
  return runtimeAssetUrl(relativePath);
}

function createRasterStyle({id, tiles, attribution, maxzoom = 19}) {
  return {
    version: 8,
    sources: {
      [id]: {
        type: 'raster',
        tiles,
        tileSize: 256,
        maxzoom,
        attribution,
      },
    },
    layers: [{
      id,
      type: 'raster',
      source: id,
      paint: {'raster-fade-duration': 0},
    }],
  };
}

const BASEMAPS = {
  off: {
    label: 'Off',
    style: {
      version: 8,
      sources: {},
      layers: [{
        id: 'blank-background',
        type: 'background',
        paint: {'background-color': '#13212a'},
      }],
    },
  },
  map: {
    label: 'Map',
    style: createRasterStyle({
      id: 'osm-street',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      attribution: '© OpenStreetMap contributors',
      maxzoom: 19,
    }),
  },
  satellite: {
    label: 'Satellite',
    style: createRasterStyle({
      id: 'esri-world-imagery',
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
      maxzoom: 19,
    }),
  },
  bw: {
    label: 'Soft B/W',
    // CARTO Positron is deliberately soft, low-contrast context — not the
    // high-ink Toner look. This same raster source is also used by the
    // fixed study-area cap atlas.
    tileTemplate: 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
    style: createRasterStyle({
      id: 'carto-soft-bw',
      tiles: ['https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'],
      attribution: '© OpenStreetMap contributors © CARTO',
      maxzoom: 20,
    }),
  },
};

function updateBasemapControls() {
  if (basemapControl) {
    const contextLocked = sceneMode === '3d' && capAppearance === 'context-map';
    const select = basemapControl.querySelector('[data-basemap-select]');
    if (select) {
      select.value = basemapMode;
      select.disabled = contextLocked;
      select.title = contextLocked ? 'Context-map caps use the Soft B/W basemap.' : '';
    }
    for (const button of basemapControl.querySelectorAll('.seg')) {
      const mode = button.dataset.mode;
      button.classList.toggle('active', mode === basemapMode);
      const unavailable = contextLocked && mode !== 'bw';
      button.disabled = unavailable;
      button.title = unavailable ? 'Context-map caps use the Soft B/W basemap.' : '';
    }
  }
  updateTwoDAnalysisControls();
  scheduleBottomStatusUpdate();
}

function updateCapAppearanceControls() {
  if (capAppearanceSelect) {
    capAppearanceSelect.value = capAppearance;
  }
  if (capAppearanceControl) {
    for (const button of capAppearanceControl.querySelectorAll('.seg')) {
      button.classList.toggle('active', button.dataset.mode === capAppearance);
    }
  }

  updatePistonComponentControls();

  if (uncertaintyReliefToggle) {
    uncertaintyReliefToggle.disabled = !runtime || sceneMode !== '3d' || capAppearance === 'context-map';
    uncertaintyReliefToggle.checked = uncertaintyReliefEnabled;
  }
  updateFloatingLegendBars();
}

function setToggleButtonState(button, enabled) {
  if (!button) return;
  button.classList.toggle('active', enabled);
  button.setAttribute('aria-pressed', enabled ? 'true' : 'false');
}

function updatePistonComponentControls() {
  setToggleButtonState(pistonWallsToggle, showPistonWalls);
  setToggleButtonState(blankieCapsToggle, showBlankieCaps);
}

function getContextCapTextureParameters() {
  return {
    ...CONTEXT_CAP_TEXTURE_PARAMETERS,
    mipmapFilter: contextCapConfig.atlasMipmaps ? 'linear' : 'none',
    maxAnisotropy: contextCapConfig.atlasMaxAnisotropy,
  };
}

function contextAtlasRecord(key) {
  return contextAtlases[key] ?? null;
}

function activeContextAtlasRecord() {
  return contextAtlasRecord(activeContextAtlasKey);
}

function contextAtlasLodEnabled() {
  return Boolean(contextCapConfig.atlasLod?.enabled);
}

function contextAtlasTextureLimit() {
  // MapLibre already owns this canvas/context. Asking for the existing context
  // returns the current WebGL object; it lets us avoid requesting a texture
  // that is larger than this browser/GPU can accept. Fall back conservatively
  // on browsers that do not expose it.
  try {
    const canvas = map?.getCanvas?.();
    const gl = canvas?.getContext?.('webgl2') ?? canvas?.getContext?.('webgl');
    const max = gl?.getParameter?.(gl.MAX_TEXTURE_SIZE);
    return Number.isFinite(max) ? Math.max(512, Math.floor(max)) : 4096;
  } catch {
    return 4096;
  }
}

function requestedContextAtlasKey(zoom = map.getZoom()) {
  const lod = contextCapConfig.atlasLod;
  if (!lod?.enabled) return 'overview';

  // Hysteresis: once focus is active, stay there until the user actually
  // exits the close-inspection range. While its atlas is still loading, use
  // the same enter threshold but keep Atlas A visibly active.
  if (activeContextAtlasKey === 'focus') {
    return zoom <= lod.overviewReturnMapZoom ? 'overview' : 'focus';
  }
  return zoom >= lod.focusEnterMapZoom ? 'focus' : 'overview';
}

function setActiveContextAtlas(key, {redraw = true} = {}) {
  const record = contextAtlasRecord(key);
  if (!record?.atlas) return false;

  const changed = key !== activeContextAtlasKey || contextAtlas !== record.atlas;
  activeContextAtlasKey = key;
  contextAtlas = record.atlas;
  contextAtlasState = record.state;

  if (changed) {
    updateContextCapDiagnostic();
    updateReadingNote();
    if (redraw && runtime) deckOverlay.setProps({layers: makeLayers()});
  }
  return changed;
}

function contextUvsForCell(cell) {
  const uvs = cell.contextUvsByAtlas?.[activeContextAtlasKey];
  return uvs ?? {
    south: cell.contextUvSouth ?? [0, 1, 1, 1],
    north: cell.contextUvNorth ?? [0, 0, 1, 0],
  };
}

function contextUvSouthForCell(cell) {
  return contextUvsForCell(cell).south;
}

function contextUvNorthForCell(cell) {
  return contextUvsForCell(cell).north;
}

function focusAtlasStatusText() {
  const lod = contextCapConfig.atlasLod;
  if (!lod?.enabled) return 'single overview atlas';

  const focus = contextAtlasRecord('focus');
  if (focus?.state === 'loading') {
    const progress = focus.progress;
    return progress
      ? `focus atlas loading ${progress.complete}/${progress.total} tiles · overview stays active`
      : 'focus atlas loading · overview stays active';
  }
  if (focus?.state === 'ready' && focus.atlas) {
    const state = activeContextAtlasKey === 'focus' ? 'focus atlas active' : 'focus atlas cached';
    return `${state} · enter ≥${lod.focusEnterMapZoom.toFixed(1)} / return ≤${lod.overviewReturnMapZoom.toFixed(1)} map zoom`;
  }
  if (focus?.state === 'failed') {
    return 'focus atlas unavailable · overview atlas retained';
  }
  return `focus atlas prepares at map zoom ≥${lod.focusEnterMapZoom.toFixed(1)}`;
}

function updateContextCapDiagnostic() {
  if (!contextCapDiagnostic) return;
  const overview = contextAtlasRecord('overview');
  const active = activeContextAtlasRecord();

  if (overview?.state === 'loading' && !overview.atlas) {
    const progress = overview.progress;
    contextCapDiagnostic.textContent = progress
      ? `Context atlas: building overview ${progress.complete}/${progress.total} soft B/W tiles…`
      : 'Context atlas: building soft B/W overview texture…';
    return;
  }
  if (overview?.state === 'failed' && !overview.atlas) {
    contextCapDiagnostic.textContent = 'Context atlas unavailable — flat-cap fallback is active. Check the browser console/network.';
    return;
  }
  if (active?.atlas) {
    const trianglesPerCap = contextCapMesh?.triangleCount ?? 0;
    const sampling = contextCapConfig.atlasMipmaps
      ? `linear mipmaps · ${contextCapConfig.atlasMaxAnisotropy}× anisotropy`
      : 'single-level sampling';
    const label = activeContextAtlasKey === 'focus' ? 'focus' : 'overview';
    contextCapDiagnostic.textContent =
      `Context atlas ${label} active · ${active.atlas.width}×${active.atlas.height}px · z${active.atlas.zoom} · ` +
      `geographic corner UVs · ${sampling} · ${focusAtlasStatusText()} · ` +
      `${(trianglesPerCap * liveCells.length).toLocaleString()} live + ` +
      `${(trianglesPerCap * blankieCells.length).toLocaleString()} blankie cap triangles.`;
    return;
  }
  contextCapDiagnostic.textContent = 'Context atlas will load with the Jakarta runtime.';
}

async function prepareContextAtlas(key = 'overview') {
  const record = contextAtlasRecord(key);
  if (!runtime || !record || record.state === 'loading' || record.state === 'ready') return;

  const lod = contextCapConfig.atlasLod;
  const isFocus = key === 'focus';
  const zoom = isFocus ? lod.focusAtlasZoom : contextCapConfig.atlasZoom;
  const configuredMaxDimension = isFocus ? lod.focusAtlasMaxDimension : contextCapConfig.atlasMaxDimension;
  const gpuLimit = contextAtlasTextureLimit();
  const maxDimension = Math.min(configuredMaxDimension, gpuLimit);
  const maxTileCount = isFocus ? lod.focusAtlasMaxTiles : Infinity;

  record.state = 'loading';
  record.error = null;
  record.progress = null;
  updateContextCapDiagnostic();

  try {
    if (!contextStudyBounds) {
      contextStudyBounds = computeStudyBounds(structuralCells, {
        paddingFraction: contextCapConfig.atlasPaddingFraction,
      });
    }
    const atlas = await buildRasterAtlas({
      bounds: contextStudyBounds,
      tileTemplate: BASEMAPS.bw.tileTemplate,
      zoom,
      maxDimension,
      maxTileCount,
      onProgress: ({complete, total, failures}) => {
        record.progress = {complete, total, failures};
        if (complete === total || complete % 12 === 0) updateContextCapDiagnostic();
      },
    });
    record.atlas = atlas;
    record.state = 'ready';
    record.progress = null;

    assignContextCornerUvs(liveCells, atlas, key);
    assignContextCornerUvs(blankieCells, atlas, key);
    contextCapMesh ??= createContextCapQuadMesh();

    if (key === 'overview' && !contextAtlas) {
      setActiveContextAtlas('overview', {redraw: false});
    }

    // The focus texture is loaded lazily. Only swap when the current camera
    // still requests it; otherwise keep it cached and retain the overview.
    syncContextAtlasLod({redraw: true});
    updateContextCapDiagnostic();
    updateReadingNote();
    if (runtime) deckOverlay.setProps({layers: makeLayers()});
  } catch (error) {
    record.state = 'failed';
    record.error = error;
    record.progress = null;
    console.warn(`[Proto1 DeckGL] ${key} context atlas unavailable; overview/flat fallback retained.`, error);
    updateContextCapDiagnostic();
    updateReadingNote();
  }
}

function syncContextAtlasLod({redraw = true} = {}) {
  if (sceneMode !== '3d' || !runtime || !contextAtlasRecord('overview')?.atlas) return;
  const desired = requestedContextAtlasKey();
  const focus = contextAtlasRecord('focus');

  if (desired === 'focus') {
    if (focus?.state === 'idle') void prepareContextAtlas('focus');
    if (focus?.state === 'ready' && focus.atlas) {
      setActiveContextAtlas('focus', {redraw});
    } else {
      setActiveContextAtlas('overview', {redraw});
      updateContextCapDiagnostic();
    }
    return;
  }

  setActiveContextAtlas('overview', {redraw});
  updateContextCapDiagnostic();
}

function setBasemapMode(mode, {recordForScene = true} = {}) {
  if (!BASEMAPS[mode]) return;
  if (sceneMode === '3d' && capAppearance === 'context-map' && mode !== 'bw') {
    setCapAppearance('scientific');
  }
  basemapMode = mode;
  if (recordForScene) {
    if (sceneMode === '2d') twoDBasemapMode = mode;
    else threeDBasemapMode = mode;
  }
  map.setStyle(BASEMAPS[mode].style);
  updateBasemapControls();
  updateReadingNote();
}

function setCapAppearance(mode) {
  if (mode !== 'scientific' && mode !== 'context-map') return;
  if (mode === capAppearance) return;

  if (mode === 'context-map') {
    reliefBeforeContextMode = uncertaintyReliefEnabled;
    uncertaintyReliefEnabled = false;
    basemapMode = 'bw';
    threeDBasemapMode = 'bw';
    map.setStyle(BASEMAPS.bw.style);
  } else {
    uncertaintyReliefEnabled = reliefBeforeContextMode;
  }

  capAppearance = mode;
  updateBasemapControls();
  updateCapAppearanceControls();
  updateReadingNote();
  if (runtime) applyEpoch();
}

const map = new maplibregl.Map({
  container: 'map',
  style: BASEMAPS.map.style,
  center: [106.84, -6.2],
  zoom: 10,
  maxZoom: 19,
  bearing: -25,
  pitch: 62,
  maxPitch: 80,
  antialias: true,
});

// UI-A supplies a V7.1-style navigation bar; do not duplicate MapLibre's default controls.

function installGoogleEarthMouseControls(mapInstance) {
  mapInstance.dragPan.enable();
  mapInstance.scrollZoom.enable();
  mapInstance.dragRotate.disable();
  mapInstance.boxZoom.disable();
  mapInstance.doubleClickZoom.disable();

  const surface = mapInstance.getContainer();
  let activeMode = null;
  let activePointerId = null;
  let lastX = 0;
  let lastY = 0;

  const isControl = (target) => target instanceof Element && target.closest('.maplibregl-ctrl');

  surface.addEventListener('contextmenu', (event) => event.preventDefault());
  surface.addEventListener('auxclick', (event) => event.preventDefault());

  surface.addEventListener('pointerdown', (event) => {
    if (isControl(event.target)) return;
    if (event.button === 1 && sceneMode === '3d') activeMode = 'orbit';
    if (event.button === 2) activeMode = 'zoom';
    if (!activeMode) return;

    activePointerId = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    surface.setPointerCapture?.(event.pointerId);
    surface.style.cursor = activeMode === 'orbit' ? 'grabbing' : 'ns-resize';
    event.preventDefault();
    event.stopPropagation();
  });

  surface.addEventListener('pointermove', (event) => {
    if (!activeMode || event.pointerId !== activePointerId) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;

    if (activeMode === 'orbit') {
      mapInstance.setBearing(mapInstance.getBearing() + dx * 0.35);
      mapInstance.setPitch(clamp(mapInstance.getPitch() - dy * 0.28, 0, runtime?.maxCameraPitch ?? 80));
    } else {
      mapInstance.zoomTo(mapInstance.getZoom() - dy * 0.018, {duration: 0});
    }

    lastX = event.clientX;
    lastY = event.clientY;
    event.preventDefault();
    event.stopPropagation();
  });

  function stop(event) {
    if (event.pointerId !== activePointerId) return;
    if (surface.hasPointerCapture?.(event.pointerId)) surface.releasePointerCapture(event.pointerId);
    activeMode = null;
    activePointerId = null;
    surface.style.cursor = '';
  }

  surface.addEventListener('pointerup', stop);
  surface.addEventListener('pointercancel', stop);
}

installGoogleEarthMouseControls(map);

function captureCameraState() {
  const center = map.getCenter();
  return {
    center: [center.lng, center.lat],
    zoom: map.getZoom(),
    bearing: map.getBearing(),
    pitch: map.getPitch(),
  };
}


function studioCaptureIntroCameraState() {
  const fallback = captureCameraState();
  const bounds = defaultSceneBounds?.({liveOnly: false});
  const center = bounds?.getCenter?.();
  const lon = Number(center?.lng ?? fallback.center?.[0]);
  const lat = Number(center?.lat ?? fallback.center?.[1]);
  return {
    center: [Number.isFinite(lon) ? lon : fallback.center[0], Number.isFinite(lat) ? lat : fallback.center[1]],
    zoom: sceneMode === '2d' ? 10.15 : 9.85,
    bearing: sceneMode === '2d' ? 0 : -25,
    pitch: sceneMode === '2d' ? 0 : Math.min(58, Number(runtime?.maxCameraPitch ?? 58)),
  };
}

function getStudioCaptureSceneState() {
  return {
    sceneMode,
    activeEpoch,
    verticalExaggeration,
    basemapMode,
    threeDBasemapMode,
    twoDBasemapMode,
    capAppearance,
    referenceGridMode,
    showPistonWalls,
    showBlankieCaps,
    uncertaintyReliefEnabled,
    datumLineEnabled,
    depthOccludersEnabled,
    apronMode,
    showHorizontalArrows,
    showHorizontalEllipses,
    horizontalGlyphOpacity,
    horizontalGlyphScale,
    showHorizontalParticles,
    horizontalParticleMode,
    horizontalParticleFieldMode,
    horizontalParticleCount,
    horizontalParticleSpeedMultiplier,
    horizontalParticleSizeMultiplier,
    horizontalParticleOpacity,
    horizontalParticleTrailDurationSeconds,
    horizontalParticleTrailPersistence,
    horizontalParticleUncertaintyStrength,
    polygonVisible: scenePolygonToggle?.classList.contains('active') ?? true,
  };
}

function applyStudioCaptureSceneState(state = {}) {
  if (!runtime || !state) return;
  const targetMode = state.sceneMode === '2d' ? '2d' : '3d';

  if (targetMode !== sceneMode) setSceneMode(targetMode);

  const nextEpoch = clamp(Math.round(Number(state.activeEpoch ?? activeEpoch)), 0, runtime.epochCount - 1);
  activeEpoch = nextEpoch;
  if (epochSlider) epochSlider.value = String(activeEpoch);

  if (Number.isFinite(Number(state.verticalExaggeration))) {
    verticalExaggeration = clamp(Number(state.verticalExaggeration), Number(verticalExagSlider?.min ?? 0), Number(verticalExagSlider?.max ?? 20));
    if (verticalExagSlider) verticalExagSlider.value = String(verticalExaggeration);
    if (verticalExagValue) verticalExagValue.textContent = formatExaggeration(verticalExaggeration);
  }

  if (typeof state.threeDBasemapMode === 'string') threeDBasemapMode = state.threeDBasemapMode;
  if (typeof state.twoDBasemapMode === 'string') twoDBasemapMode = state.twoDBasemapMode;
  if (typeof state.capAppearance === 'string' && sceneMode === '3d') setCapAppearance(state.capAppearance);
  if (typeof state.basemapMode === 'string' && BASEMAPS[state.basemapMode]) setBasemapMode(state.basemapMode, {recordForScene: true});
  if (typeof state.referenceGridMode === 'string') setReferenceGridMode(state.referenceGridMode);

  showPistonWalls = state.showPistonWalls !== false;
  showBlankieCaps = state.showBlankieCaps !== false;
  uncertaintyReliefEnabled = state.uncertaintyReliefEnabled !== false;
  datumLineEnabled = Boolean(state.datumLineEnabled);
  depthOccludersEnabled = state.depthOccludersEnabled !== false;
  if (typeof state.apronMode === 'string') apronMode = state.apronMode;

  showHorizontalArrows = state.showHorizontalArrows !== false;
  showHorizontalEllipses = state.showHorizontalEllipses !== false;
  if (Number.isFinite(Number(state.horizontalGlyphOpacity))) horizontalGlyphOpacity = clamp(Number(state.horizontalGlyphOpacity), 0, 1);
  if (Number.isFinite(Number(state.horizontalGlyphScale))) horizontalGlyphScale = clamp(Number(state.horizontalGlyphScale), 0.35, 3.0);

  showHorizontalParticles = state.showHorizontalParticles !== false;
  if (typeof state.horizontalParticleMode === 'string') horizontalParticleMode = state.horizontalParticleMode;
  if (typeof state.horizontalParticleFieldMode === 'string') horizontalParticleFieldMode = state.horizontalParticleFieldMode;
  if (Number.isFinite(Number(state.horizontalParticleCount))) horizontalParticleCount = clamp(Math.round(Number(state.horizontalParticleCount)), 0, Number(horizontalParticleConfig.particleCapacity ?? 12000));
  if (Number.isFinite(Number(state.horizontalParticleSpeedMultiplier))) horizontalParticleSpeedMultiplier = clamp(Number(state.horizontalParticleSpeedMultiplier), 0.1, 6);
  if (Number.isFinite(Number(state.horizontalParticleSizeMultiplier))) horizontalParticleSizeMultiplier = clamp(Number(state.horizontalParticleSizeMultiplier), 0.2, 3.5);
  if (Number.isFinite(Number(state.horizontalParticleOpacity))) horizontalParticleOpacity = clamp(Number(state.horizontalParticleOpacity), 0, 1);
  if (Number.isFinite(Number(state.horizontalParticleTrailPersistence))) horizontalParticleTrailPersistence = clamp(Number(state.horizontalParticleTrailPersistence), 0.5, 0.999);
  if (Number.isFinite(Number(state.horizontalParticleTrailDurationSeconds))) {
    horizontalParticleTrailDurationSeconds = clamp(Number(state.horizontalParticleTrailDurationSeconds), 0.2, 5.0);
    horizontalParticleHistorySamples = particleHistorySamplesForDuration(horizontalParticleTrailDurationSeconds, horizontalParticleConfig);
  }
  if (Number.isFinite(Number(state.horizontalParticleUncertaintyStrength))) horizontalParticleUncertaintyStrength = clamp(Number(state.horizontalParticleUncertaintyStrength), 0, 2);

  updateSceneModeUi();
  updateCapAppearanceControls();
  updatePistonComponentControls();
  updateHorizontalGlyphControls();
  updateHorizontalParticleControls();
  updateBasemapControls();
  updateModeSpecificLegends();
  updateFloatingLegendBars();
  updateVerticalVelocityLegend();
  updateReadingNote();
  syncHorizontalParticleFieldRuntime({resetGpuStatus: false});
  applyEpoch();
}

function applyStudioCaptureCameraState(camera = {}, {duration = 650} = {}) {
  if (!camera || !map) return;
  map.stop();
  map.easeTo({
    center: Array.isArray(camera.center) ? camera.center : captureCameraState().center,
    zoom: Number.isFinite(Number(camera.zoom)) ? Number(camera.zoom) : map.getZoom(),
    bearing: Number.isFinite(Number(camera.bearing)) ? Number(camera.bearing) : map.getBearing(),
    pitch: clamp(Number.isFinite(Number(camera.pitch)) ? Number(camera.pitch) : map.getPitch(), 0, Number(runtime?.maxCameraPitch ?? 80)),
    duration: Math.max(0, Number(duration) || 0),
    essential: true,
  });
}


function drawerMasterCssPx(name, fallbackPx) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallbackPx;
}

function drawerSectionLabel(section) {
  return (section?.querySelector('.drawerSectionName, .titleLead, .drawerSectionTitle')?.textContent || 'Section')
    .replace(/\s+/g, ' ')
    .trim();
}

function drawerSectionByIds(ids = []) {
  for (const id of ids) {
    const section = document.getElementById(id);
    if (section?.classList?.contains('drawerSection')) return section;
  }
  return null;
}

function drawerSectionMasterEnabled(section) {
  if (!section) return true;

  // Batch 1.13: dataset is the source of truth once the master switch has
  // been initialized. The old aria attributes are only a visual/a11y mirror.
  // Previously, after switching OFF, aria-pressed stayed false long enough to
  // override a later dataset=true update, so the section could not turn back ON.
  if (section.dataset.masterEnabled === 'false') return false;
  if (section.dataset.masterEnabled === 'true') return true;

  if (section.classList.contains(DRAWER_MASTER_OFF_CLASS) || section.classList.contains('masterOff')) return false;
  const master = section.querySelector('.drawerSectionMaster');
  if (master) {
    if (master.getAttribute('aria-pressed') === 'false') return false;
    if (master.getAttribute('aria-checked') === 'false') return false;
  }
  return true;
}

function drawerFeatureMasterEnabled(featureKey) {
  const section = drawerSectionByIds(DRAWER_MASTER_FEATURE_IDS[featureKey] ?? []);
  return section ? drawerSectionMasterEnabled(section) : true;
}

function horizontalGlyphMasterEnabled() {
  return drawerFeatureMasterEnabled('horizontalGlyphs');
}

function horizontalParticleMasterEnabled() {
  return drawerFeatureMasterEnabled('horizontalParticles');
}

function verticalMasterEnabled() {
  return drawerFeatureMasterEnabled('vertical');
}

function drawerMasterBodyControls(section) {
  if (!section) return [];
  return Array.from(section.querySelectorAll('.drawerSectionBody button, .drawerSectionBody input, .drawerSectionBody select, .drawerSectionBody textarea'));
}

function setDrawerMasterBodyControlsEnabled(section, enabled) {
  for (const control of drawerMasterBodyControls(section)) {
    if (!enabled) {
      if (!Object.prototype.hasOwnProperty.call(control.dataset, 'masterPrevDisabled')) {
        control.dataset.masterPrevDisabled = control.disabled ? '1' : '0';
      }
      control.disabled = true;
      control.setAttribute('aria-disabled', 'true');
    } else if (Object.prototype.hasOwnProperty.call(control.dataset, 'masterPrevDisabled')) {
      control.disabled = control.dataset.masterPrevDisabled === '1';
      control.removeAttribute('aria-disabled');
      delete control.dataset.masterPrevDisabled;
    }
  }
}

function syncDrawerMasterA11y(section) {
  if (!section) return;
  const enabled = drawerSectionMasterEnabled(section);
  const label = `${drawerSectionLabel(section)} ${enabled ? 'on' : 'off'}`;

  const binder = section.querySelector('.drawerBinder');
  if (binder) {
    binder.dataset.masterSwitch = 'true';
    binder.setAttribute('role', 'switch');
    binder.setAttribute('aria-checked', enabled ? 'true' : 'false');
    binder.setAttribute('aria-label', label);
  }

  const master = section.querySelector('.drawerSectionMaster');
  if (master) {
    master.type = 'button';
    master.setAttribute('aria-pressed', enabled ? 'true' : 'false');
    master.setAttribute('aria-checked', enabled ? 'true' : 'false');
    master.setAttribute('aria-label', label);
    master.title = label;
  }
}

function syncDrawerMasterControls(section) {
  if (!section) return;
  setDrawerMasterBodyControlsEnabled(section, drawerSectionMasterEnabled(section));
  syncDrawerMasterA11y(section);
}

function ensureDrawerMasterSwitches() {
  for (const section of drawerSections) {
    if (!section || section.classList.contains('drawerStudioSection')) continue;

    const title = section.querySelector('.drawerSectionTitle');
    if (!title) continue;

    let binder = section.querySelector('.drawerBinder');
    if (!binder) {
      binder = document.createElement('span');
      binder.className = 'drawerBinder';
    }
    if (binder.parentElement !== section) {
      section.insertBefore(binder, title);
    } else if (binder.nextElementSibling !== title) {
      section.insertBefore(binder, title);
    }

    let master = binder.querySelector('.drawerSectionMaster');
    if (!master) {
      master = document.createElement('button');
      master.className = 'drawerSectionMaster';
      master.type = 'button';
      binder.appendChild(master);
    }

    if (!section.dataset.masterEnabled) {
      section.dataset.masterEnabled = drawerSectionMasterEnabled(section) ? 'true' : 'false';
    }
    syncDrawerMasterControls(section);
  }
}

function eventPoint(event) {
  const source = event.touches?.[0] ?? event.changedTouches?.[0] ?? event;
  if (!Number.isFinite(source?.clientX) || !Number.isFinite(source?.clientY)) return null;
  return {x: source.clientX, y: source.clientY};
}

function drawerSectionFromMasterTarget(target) {
  if (!target?.closest) return null;
  return target.closest('#rightDrawerPanel .drawerBinder, #rightDrawerPanel .drawerSectionMaster')?.closest('.drawerSection') ?? null;
}

function drawerSectionFromMasterHitZone(event) {
  const direct = drawerSectionFromMasterTarget(event.target);
  if (direct) return direct;

  if (!rightControlRoot?.classList.contains('drawerOpen')) return null;
  const point = eventPoint(event);
  if (!point) return null;

  const gutterPx = drawerMasterCssPx('--drawer-binder-gutter', 28) + 6;
  const binderWidthPx = drawerMasterCssPx('--drawer-binder-width', 25) + 6;

  for (const section of drawerSections) {
    if (!section || section.classList.contains('drawerStudioSection')) continue;

    const binder = section.querySelector('.drawerBinder');
    if (binder) {
      const rect = binder.getBoundingClientRect();
      if (
        rect.width > 0 &&
        rect.height > 0 &&
        point.x >= rect.left - 3 &&
        point.x <= rect.right + 3 &&
        point.y >= rect.top - 3 &&
        point.y <= rect.bottom + 3
      ) {
        return section;
      }
    }

    const title = section.querySelector('.drawerSectionTitle');
    if (!title) continue;
    const titleRect = title.getBoundingClientRect();
    const inTitleBand = point.y >= titleRect.top && point.y <= titleRect.bottom;
    const inLeftSwitchGutter = point.x >= titleRect.left - gutterPx && point.x <= titleRect.left + Math.min(5, binderWidthPx);
    if (inTitleBand && inLeftSwitchGutter) return section;
  }

  return null;
}

function applyDrawerMasterSideEffects(section) {
  if (!section) return;
  const id = section.id || '';

  if (DRAWER_MASTER_FEATURE_IDS.miniViewer.includes(id)) {
    syncMiniViewerActiveState({draw: true});
    return;
  }

  if (DRAWER_MASTER_FEATURE_IDS.horizontalGlyphs.includes(id)) {
    updateHorizontalGlyphControls();
    updateFloatingLegendBars();
    if (runtime) deckOverlay.setProps({layers: makeLayers()});
    return;
  }

  if (DRAWER_MASTER_FEATURE_IDS.horizontalParticles.includes(id)) {
    updateHorizontalParticleControls();
    updateFloatingLegendBars();
    if (runtime) deckOverlay.setProps({layers: makeLayers()});
    return;
  }

  if (DRAWER_MASTER_FEATURE_IDS.vertical.includes(id)) {
    updatePistonComponentControls();
    updateFloatingLegendBars();
    if (runtime) deckOverlay.setProps({layers: makeLayers()});
  }
}

function setDrawerSectionMasterEnabled(section, enabled, {dispatch = true} = {}) {
  if (!section) return;
  const nextEnabled = Boolean(enabled);
  section.dataset.masterEnabled = nextEnabled ? 'true' : 'false';
  section.classList.toggle(DRAWER_MASTER_OFF_CLASS, !nextEnabled);
  section.classList.toggle('masterOff', !nextEnabled);

  syncDrawerMasterControls(section);
  applyDrawerMasterSideEffects(section);

  if (dispatch) {
    section.dispatchEvent(new CustomEvent('drawer-section-master-change', {
      bubbles: true,
      detail: {
        enabled: nextEnabled,
        sectionId: section.id || '',
        sectionLabel: drawerSectionLabel(section),
      },
    }));
  }
}

function toggleDrawerSectionMaster(section) {
  setDrawerSectionMasterEnabled(section, !drawerSectionMasterEnabled(section));
}

function stopDrawerMasterEvent(event) {
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation?.();
}

function handleDrawerMasterPointer(event) {
  const section = drawerSectionFromMasterHitZone(event);
  if (!section) return;
  stopDrawerMasterEvent(event);
  section.dataset.masterPointerToggleAt = String(performance.now());
  toggleDrawerSectionMaster(section);
}

function handleDrawerMasterClick(event) {
  const section = drawerSectionFromMasterHitZone(event);
  if (!section) return;
  stopDrawerMasterEvent(event);

  const lastToggleAt = Number(section.dataset.masterPointerToggleAt ?? 0);
  if (!Number.isFinite(lastToggleAt) || performance.now() - lastToggleAt > 700) {
    toggleDrawerSectionMaster(section);
  }
}

function handleDrawerMasterKeydown(event) {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  const section = drawerSectionFromMasterTarget(event.target);
  if (!section) return;
  stopDrawerMasterEvent(event);
  toggleDrawerSectionMaster(section);
}

function installDrawerMasterSwitches() {
  if (!rightDrawerPanel || !drawerSections.length || window.__drawerMasterSwitchIntegratedInstalled) return;
  window.__drawerMasterSwitchIntegratedInstalled = true;
  // Blocks the previous standalone click-bridge file if it is still referenced
  // after main.js. This integrated bridge is the source of truth now.
  window.__drawerMasterSwitchHitFixInstalled = true;

  ensureDrawerMasterSwitches();
  window.addEventListener('pointerdown', handleDrawerMasterPointer, true);
  window.addEventListener('click', handleDrawerMasterClick, true);
  window.addEventListener('keydown', handleDrawerMasterKeydown, true);

  window.drawerMasterSwitch = {
    setSectionEnabled: (sectionOrId, enabled) => {
      const section = typeof sectionOrId === 'string' ? document.getElementById(sectionOrId) : sectionOrId;
      setDrawerSectionMasterEnabled(section, enabled);
    },
    isSectionEnabled: (sectionOrId) => {
      const section = typeof sectionOrId === 'string' ? document.getElementById(sectionOrId) : sectionOrId;
      return drawerSectionMasterEnabled(section);
    },
    refresh: ensureDrawerMasterSwitches,
  };
}

function defaultVerticalExaggeration() {
  const runtimeDefault = Number(runtime?.verticalExaggeration?.defaultMPerMm);
  const sliderDefault = Number(verticalExagSlider?.defaultValue);
  const fallback = Number.isFinite(sliderDefault) ? sliderDefault : 10;
  const min = Number(verticalExagSlider?.min ?? runtime?.verticalExaggeration?.minMPerMm ?? 0);
  const max = Number(verticalExagSlider?.max ?? runtime?.verticalExaggeration?.maxMPerMm ?? 20);
  return clamp(Number.isFinite(runtimeDefault) ? runtimeDefault : fallback, min, max);
}

function resetDrawerVisualDefaults() {
  stopPlayback();

  for (const section of drawerSections) {
    if (!section || section.classList.contains('drawerStudioSection') || section.classList.contains('drawerMiniViewerSection')) continue;
    setDrawerSectionMasterEnabled(section, true, {dispatch: false});
  }

  referenceGridMode = 'off';
  updateReferenceGridControl();

  capAppearance = 'scientific';
  reliefBeforeContextMode = true;
  uncertaintyReliefEnabled = Boolean(runtime?.uncertaintyRelief?.geometry?.enabled ?? true);
  showPistonWalls = true;
  showBlankieCaps = true;
  depthOccludersEnabled = true;
  datumLineEnabled = false;
  apronMode = 'see-through';

  verticalExaggeration = defaultVerticalExaggeration();
  if (verticalExagSlider) verticalExagSlider.value = String(verticalExaggeration);
  if (verticalExagValue) verticalExagValue.textContent = formatExaggeration(verticalExaggeration);
  if (uncertaintyReliefToggle) uncertaintyReliefToggle.checked = uncertaintyReliefEnabled;
  if (datumLineToggle) datumLineToggle.checked = datumLineEnabled;
  if (depthOccluderToggle) depthOccluderToggle.checked = depthOccludersEnabled;
  if (apronModeControl) {
    for (const segment of apronModeControl.querySelectorAll('.seg')) {
      segment.classList.toggle('active', segment.dataset.mode === apronMode);
    }
  }

  twoDAnalysisConfig = {
    ...twoDAnalysisConfig,
    rumFillOpacity: DEFAULT_TWO_D_ANALYSIS.rumFillOpacity,
  };

  showHorizontalArrows = horizontalGlyphConfig.showArrowsByDefault !== false;
  showHorizontalEllipses = horizontalGlyphConfig.showEllipsesByDefault !== false;
  horizontalGlyphOpacity = Number.isFinite(Number(horizontalGlyphConfig.defaultOpacity))
    ? clamp(Number(horizontalGlyphConfig.defaultOpacity), 0, 1)
    : 0.92;
  horizontalGlyphScale = 1.0;

  horizontalParticleFieldMode = 'raw';
  syncHorizontalParticleFieldRuntime({resetGpuStatus: true});
  showHorizontalParticles = Boolean(horizontalParticleConfig.enabled && horizontalParticleConfig.showByDefault);
  horizontalParticleMode = horizontalParticleConfig.defaultMode ?? 'mean';
  horizontalParticleCount = clamp(
    Math.round(Number(horizontalParticleConfig.defaultParticleCount ?? 5000)),
    0,
    Number(horizontalParticleConfig.particleCapacity ?? 12000),
  );
  horizontalParticleSpeedMultiplier = Number(horizontalParticleConfig.speedMultiplier ?? 1.5);
  horizontalParticleSizeMultiplier = Number(horizontalParticleConfig.particleSizeMultiplier ?? 1.0);
  horizontalParticleOpacity = clamp(Number(horizontalParticleConfig.particleOpacity ?? 1.0), 0, 1);
  horizontalParticleTrailPersistence = clamp(Number(horizontalParticleConfig.trailPersistence ?? 0.98), 0.5, 0.999);
  horizontalParticleHistorySamples = Math.round(Number(horizontalParticleConfig.historySamples ?? 32));
  horizontalParticleTrailDurationSeconds = particleHistoryDurationForSamples(horizontalParticleHistorySamples, horizontalParticleConfig);
  horizontalParticleUncertaintyStrengths = {
    shimmer: Number(horizontalParticleConfig.shimmerStrength ?? horizontalParticleConfig.uncertaintyStrength ?? 0.5),
    montecarlo: Number(horizontalParticleConfig.monteCarloStrength ?? horizontalParticleConfig.uncertaintyStrength ?? 0.5),
  };
  horizontalParticleUncertaintyStrength = horizontalParticleUncertaintyForMode(horizontalParticleMode);
  horizontalParticleGpuStatus = null;

  threeDBasemapMode = 'map';
  twoDBasemapMode = twoDAnalysisConfig.preferredBasemapMode ?? DEFAULT_TWO_D_ANALYSIS.preferredBasemapMode;
  const defaultBasemap = sceneMode === '2d'
    ? (twoDAnalysisConfig.preferredBasemapMode ?? DEFAULT_TWO_D_ANALYSIS.preferredBasemapMode)
    : 'map';
  setBasemapMode(defaultBasemap, {recordForScene: true});

  updateCapAppearanceControls();
  updatePistonComponentControls();
  updateTwoDAnalysisControls();
  updateHorizontalGlyphControls();
  updateHorizontalParticleControls();
  updateModeSpecificLegends();
  updateFloatingLegendBars();
  updateVerticalVelocityLegend();
  updateReadingNote();

  studioPolygons?.resetVisualDefaults?.();

  if (runtime) applyEpoch();

  if (drawerDefaultsButton) {
    drawerDefaultsButton.classList.remove('justReset');
    void drawerDefaultsButton.offsetWidth;
    drawerDefaultsButton.classList.add('justReset');
  }
}


function setDrawerOpen(open, requestedDrawerId = null) {
  if (!rightControlRoot || !rightDrawerPanel) return;
  const shouldOpen = Boolean(open);
  if (requestedDrawerId) activeDrawerId = requestedDrawerId;

  rightControlRoot.classList.toggle('drawerOpen', shouldOpen);
  rightControlRoot.classList.toggle('drawerClosed', !shouldOpen);
  // Display remains available in both states. Only the accordion body closes.
  rightDrawerPanel.setAttribute('aria-hidden', 'false');
  if (rightDrawerScroll) {
    rightDrawerScroll.setAttribute('aria-hidden', shouldOpen ? 'false' : 'true');
    rightDrawerScroll.inert = !shouldOpen;
  }
  if (rightDrawerBurger) {
    rightDrawerBurger.textContent = shouldOpen ? '☰' : '☰';
    rightDrawerBurger.title = shouldOpen ? 'Collapse controls' : 'Open controls';
    rightDrawerBurger.setAttribute('aria-label', rightDrawerBurger.title);
    rightDrawerBurger.setAttribute('aria-expanded', shouldOpen ? 'true' : 'false');
  }

  if (shouldOpen && requestedDrawerId) {
    const requested = document.querySelector(`#${requestedDrawerId}`);
    if (requested?.classList.contains('collapsed')) {
      requested.classList.remove('collapsed');
      requested.querySelector('.drawerSectionTitle')?.setAttribute('aria-expanded', 'true');
    }
  }
  syncMiniViewerActiveState({draw: true});
}

function openDrawerSection(sectionId) {
  const section = document.querySelector(`#${sectionId}`);
  if (!section) return;
  activeDrawerId = sectionId;
  setDrawerOpen(true, sectionId);
}

function updateEpochNavigationControls() {
  const enabled = Boolean(runtime) && sceneMode === '3d';
  if (epochFirstButton) epochFirstButton.disabled = !enabled || activeEpoch <= 0;
  if (epochPrevButton) epochPrevButton.disabled = !enabled || activeEpoch <= 0;
  if (epochNextButton) epochNextButton.disabled = !enabled || activeEpoch >= (runtime?.epochCount ?? 1) - 1;
  if (epochLastButton) epochLastButton.disabled = !enabled || activeEpoch >= (runtime?.epochCount ?? 1) - 1;
}

function setEpochIndex(nextEpoch) {
  if (!runtime || sceneMode !== '3d') return;
  stopPlayback();
  activeEpoch = clamp(Math.round(Number(nextEpoch)), 0, runtime.epochCount - 1);
  epochSlider.value = String(activeEpoch);
  applyEpoch();
}


function updateMiniViewerBounds() {
  miniViewerBounds = boundsFromPolygons(structuralCells.map((cell) => cell.footprintLonLat));
  miniViewerTransform = null;
  miniViewerCameraFootprintCache = null;
  miniViewerCameraDirty = true;
}

function miniViewerSection() {
  return drawerMiniViewerSection ?? miniViewerCanvas?.closest?.('.drawerSection') ?? null;
}

function miniViewerSectionExpanded() {
  const section = miniViewerSection();
  return Boolean(
    miniViewerCanvas &&
    section &&
    !section.classList.contains('collapsed') &&
    (!rightControlRoot || rightControlRoot.classList.contains('drawerOpen'))
  );
}

function miniViewerMasterEnabled() {
  const section = miniViewerSection();
  return section ? drawerSectionMasterEnabled(section) : true;
}

function miniViewerCanRun() {
  return miniViewerSectionExpanded() && miniViewerMasterEnabled();
}

function cancelMiniViewerPendingWork() {
  if (miniViewerDrawFrame !== null) {
    window.cancelAnimationFrame(miniViewerDrawFrame);
    miniViewerDrawFrame = null;
  }
  if (miniViewerCameraIdleTimer !== null) {
    window.clearTimeout(miniViewerCameraIdleTimer);
    miniViewerCameraIdleTimer = null;
  }
}

function setMiniViewerSuspendedUi(suspended) {
  const section = miniViewerSection();
  section?.classList.toggle('miniViewerSuspended', Boolean(suspended));
  miniViewerWell?.classList.toggle('miniViewerSuspended', Boolean(suspended));
}

function drawMiniViewerInactive(message = 'Mini viewer off') {
  if (!miniViewerCanvas || !miniViewerSectionExpanded()) return;
  const rect = miniViewerCanvas.getBoundingClientRect();
  const cssWidth = Math.max(80, Math.floor(rect.width || miniViewerCanvas.clientWidth || 0));
  const cssHeight = Math.max(64, Math.floor(rect.height || miniViewerCanvas.clientHeight || 0));
  if (!cssWidth || !cssHeight) return;
  const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
  const pixelWidth = Math.round(cssWidth * dpr);
  const pixelHeight = Math.round(cssHeight * dpr);
  if (miniViewerCanvas.width !== pixelWidth || miniViewerCanvas.height !== pixelHeight) {
    miniViewerCanvas.width = pixelWidth;
    miniViewerCanvas.height = pixelHeight;
  }
  const ctx = miniViewerCanvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  const gradient = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
  gradient.addColorStop(0, 'rgba(12, 17, 22, 0.98)');
  gradient.addColorStop(1, 'rgba(4, 7, 10, 0.98)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = 'rgba(183, 199, 210, 0.55)';
  ctx.font = '800 11px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, cssWidth / 2, cssHeight / 2);
  if (miniViewerStatus) miniViewerStatus.textContent = message.includes('collapsed') ? 'Paused' : 'Off';
}

function syncMiniViewerActiveState({draw = true} = {}) {
  if (!miniViewerCanvas) return false;
  const active = miniViewerCanRun();
  setMiniViewerSuspendedUi(!active);
  if (!active) {
    cancelMiniViewerPendingWork();
    miniViewerCameraFrozen = false;
    miniViewerCameraDirty = true;
    const message = miniViewerMasterEnabled() ? 'Mini viewer paused' : 'Mini viewer off';
    if (draw) drawMiniViewerInactive(message);
    return false;
  }
  if (draw) scheduleMiniViewerDraw();
  return true;
}

function markMiniViewerCameraMoving() {
  if (!miniViewerCanRun()) {
    syncMiniViewerActiveState({draw: false});
    return;
  }
  if (miniViewerCameraIdleTimer !== null) {
    window.clearTimeout(miniViewerCameraIdleTimer);
    miniViewerCameraIdleTimer = null;
  }
  if (miniViewerCameraFrozen) return;
  miniViewerCameraFrozen = true;
  miniViewerCameraDirty = true;
}

function refreshMiniViewerCameraFootprint({redraw = true} = {}) {
  if (!miniViewerCanRun()) {
    syncMiniViewerActiveState({draw: false});
    return;
  }
  miniViewerCameraFootprintCache = miniViewerCameraFootprint();
  miniViewerCameraDirty = false;
  miniViewerCameraFrozen = false;
  if (miniViewerCameraIdleTimer !== null) {
    window.clearTimeout(miniViewerCameraIdleTimer);
    miniViewerCameraIdleTimer = null;
  }
  if (redraw) scheduleMiniViewerDraw();
}

function scheduleMiniViewerCameraIdleRefresh(delay = MINI_VIEWER_CAMERA_IDLE_DELAY_MS) {
  if (!miniViewerCanRun()) {
    syncMiniViewerActiveState({draw: false});
    return;
  }
  if (miniViewerCameraIdleTimer !== null) window.clearTimeout(miniViewerCameraIdleTimer);
  miniViewerCameraIdleTimer = window.setTimeout(() => {
    miniViewerCameraIdleTimer = null;
    refreshMiniViewerCameraFootprint({redraw: true});
  }, delay);
}

function scheduleMiniViewerDraw() {
  if (!miniViewerCanvas) return;
  if (!miniViewerCanRun()) {
    syncMiniViewerActiveState({draw: true});
    return;
  }
  setMiniViewerSuspendedUi(false);
  if (miniViewerDrawFrame !== null) return;
  miniViewerDrawFrame = window.requestAnimationFrame(() => {
    miniViewerDrawFrame = null;
    drawMiniViewer();
  });
}

function miniViewerCellFill(cell) {
  if (cell?.isBlankie) return 'rgba(154, 159, 156, 0.26)';
  const color = Array.isArray(cell?.fillColor) ? cell.fillColor : [142, 220, 255];
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.54)`;
}

function miniViewerCellStroke(cell) {
  if (cell?.isBlankie) return 'rgba(212, 220, 217, 0.16)';
  return 'rgba(7, 12, 18, 0.34)';
}

function miniViewerCameraFootprint() {
  if (!map?.getCanvas || !map?.unproject) return null;
  const canvas = map.getCanvas();
  const width = canvas.clientWidth || canvas.width || 0;
  const height = canvas.clientHeight || canvas.height || 0;
  const corners = [[0, 0], [width, 0], [width, height], [0, height]];
  const points = [];
  for (const corner of corners) {
    try {
      const lngLat = map.unproject(corner);
      const lon = Number(lngLat?.lng);
      const lat = Number(lngLat?.lat);
      if (Number.isFinite(lon) && Number.isFinite(lat)) points.push([lon, lat]);
    } catch {
      // Some extreme pitch/zoom combinations can fail near the horizon.
    }
  }
  if (points.length >= 3) return points;
  try {
    const bounds = map.getBounds();
    const west = bounds.getWest();
    const east = bounds.getEast();
    const south = bounds.getSouth();
    const north = bounds.getNorth();
    return [[west, north], [east, north], [east, south], [west, south]];
  } catch {
    return null;
  }
}

function miniViewerBoundsObject() {
  if (!miniViewerBounds) return null;
  const west = Number(miniViewerBounds.getWest?.());
  const east = Number(miniViewerBounds.getEast?.());
  const south = Number(miniViewerBounds.getSouth?.());
  const north = Number(miniViewerBounds.getNorth?.());
  if (![west, east, south, north].every(Number.isFinite) || east <= west || north <= south) return null;
  return {west, east, south, north};
}

function miniViewerComputeTransform(width, height) {
  const bounds = miniViewerBoundsObject();
  if (!bounds) return null;
  const dataW = Math.max(1e-9, bounds.east - bounds.west);
  const dataH = Math.max(1e-9, bounds.north - bounds.south);
  const pad = clamp(Math.round(Math.min(width, height) * 0.075), 8, 18);
  const usableW = Math.max(24, width - (pad * 2));
  const usableH = Math.max(24, height - (pad * 2));
  const scale = Math.min(usableW / dataW, usableH / dataH);
  const plotW = dataW * scale;
  const plotH = dataH * scale;
  const offsetX = (width - plotW) * 0.5;
  const offsetY = (height - plotH) * 0.5;
  return {...bounds, scale, offsetX, offsetY, plotW, plotH};
}

function miniViewerProjectLonLat(lonLat, transform = miniViewerTransform) {
  if (!transform || !Array.isArray(lonLat)) return null;
  const lon = Number(lonLat[0]);
  const lat = Number(lonLat[1]);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  return [
    transform.offsetX + ((lon - transform.west) * transform.scale),
    transform.offsetY + ((transform.north - lat) * transform.scale),
  ];
}

function miniViewerUnprojectPoint(x, y) {
  const transform = miniViewerTransform;
  if (!transform || !Number.isFinite(x) || !Number.isFinite(y)) return null;
  const lon = transform.west + ((x - transform.offsetX) / transform.scale);
  const lat = transform.north - ((y - transform.offsetY) / transform.scale);
  return [
    clamp(lon, transform.west, transform.east),
    clamp(lat, transform.south, transform.north),
  ];
}

function miniViewerDrawPolygon(ctx, polygon, transform, {fill = null, stroke = null, lineWidth = 1, close = true} = {}) {
  if (!Array.isArray(polygon) || polygon.length < 2 || !transform) return false;
  let started = false;
  ctx.beginPath();
  for (const point of polygon) {
    const projected = miniViewerProjectLonLat(point, transform);
    if (!projected) continue;
    if (!started) {
      ctx.moveTo(projected[0], projected[1]);
      started = true;
    } else {
      ctx.lineTo(projected[0], projected[1]);
    }
  }
  if (!started) return false;
  if (close) ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
  return true;
}

function drawMiniViewer() {
  if (!miniViewerCanvas) return;
  if (!miniViewerCanRun()) {
    syncMiniViewerActiveState({draw: true});
    return;
  }
  const rect = miniViewerCanvas.getBoundingClientRect();
  const cssWidth = Math.max(80, Math.floor(rect.width || miniViewerCanvas.clientWidth || 0));
  const cssHeight = Math.max(64, Math.floor(rect.height || miniViewerCanvas.clientHeight || 0));
  if (!cssWidth || !cssHeight) return;
  const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
  const pixelWidth = Math.round(cssWidth * dpr);
  const pixelHeight = Math.round(cssHeight * dpr);
  if (miniViewerCanvas.width !== pixelWidth || miniViewerCanvas.height !== pixelHeight) {
    miniViewerCanvas.width = pixelWidth;
    miniViewerCanvas.height = pixelHeight;
  }
  const ctx = miniViewerCanvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  const gradient = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
  gradient.addColorStop(0, 'rgba(16, 23, 31, 0.98)');
  gradient.addColorStop(1, 'rgba(5, 9, 14, 0.98)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  if (!runtime || !structuralCells.length || !miniViewerBounds) {
    ctx.fillStyle = 'rgba(229, 240, 248, 0.66)';
    ctx.font = '700 11px Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Loading RUM field…', cssWidth / 2, cssHeight / 2);
    if (miniViewerStatus) miniViewerStatus.textContent = 'Loading…';
    return;
  }

  const transform = miniViewerComputeTransform(cssWidth, cssHeight);
  miniViewerTransform = transform;
  if (!transform) return;

  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, cssWidth, cssHeight);
  ctx.clip();

  ctx.fillStyle = 'rgba(255,255,255,0.025)';
  ctx.fillRect(transform.offsetX, transform.offsetY, transform.plotW, transform.plotH);

  for (const cell of structuralCells) {
    miniViewerDrawPolygon(ctx, cell.footprintLonLat, transform, {
      fill: miniViewerCellFill(cell),
      stroke: miniViewerCellStroke(cell),
      lineWidth: 0.45,
    });
  }

  if ((!miniViewerCameraFootprintCache || miniViewerCameraDirty) && !miniViewerCameraFrozen) {
    miniViewerCameraFootprintCache = miniViewerCameraFootprint();
    miniViewerCameraDirty = false;
  }
  const footprint = miniViewerCameraFootprintCache;
  if (footprint?.length >= 3) {
    miniViewerDrawPolygon(ctx, footprint, transform, {
      fill: 'rgba(126, 236, 255, 0.055)',
      stroke: 'rgba(126, 236, 255, 0.92)',
      lineWidth: 1.55,
    });
    ctx.setLineDash([3, 3]);
    miniViewerDrawPolygon(ctx, footprint, transform, {
      fill: null,
      stroke: 'rgba(255, 255, 255, 0.68)',
      lineWidth: 0.75,
    });
    ctx.setLineDash([]);
  }

  if (selectedCell) {
    miniViewerDrawPolygon(ctx, selectedCell.footprintLonLat, transform, {
      fill: selectedCellIsBlankie ? 'rgba(255, 214, 94, 0.28)' : 'rgba(255, 224, 84, 0.34)',
      stroke: 'rgba(255, 232, 92, 1.0)',
      lineWidth: 2.2,
    });
    const center = cellCenterLonLat(selectedCell);
    const projected = miniViewerProjectLonLat(center, transform);
    if (projected) {
      ctx.fillStyle = 'rgba(255, 232, 92, 0.98)';
      ctx.strokeStyle = 'rgba(20, 15, 0, 0.84)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(projected[0], projected[1], 4.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  // North marker. The mini viewer stays north-up; the wedge rotates opposite
  // the main map bearing so the user can see when the main scene is rotated.
  const bearingRad = (-map.getBearing() * Math.PI) / 180;
  const cx = cssWidth - 18;
  const cy = 20;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(bearingRad);
  ctx.fillStyle = 'rgba(226, 246, 255, 0.86)';
  ctx.beginPath();
  ctx.moveTo(0, -8);
  ctx.lineTo(4.5, 6);
  ctx.lineTo(0, 3.5);
  ctx.lineTo(-4.5, 6);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = 'rgba(226, 246, 255, 0.72)';
  ctx.font = '800 8px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('N', cx, cy - 9);

  ctx.restore();

  if (miniViewerStatus) {
    miniViewerStatus.textContent = selectedCell
      ? String(selectedCell.rumId ?? selectedCell.cellId ?? 'selected')
      : `${liveCells.length.toLocaleString()} RUMs`;
  }
}

function handleMiniViewerClick(event) {
  if (!miniViewerCanRun()) return;
  if (!miniViewerCanvas || !runtime || !miniViewerTransform) return;
  const rect = miniViewerCanvas.getBoundingClientRect();
  const point = miniViewerUnprojectPoint(event.clientX - rect.left, event.clientY - rect.top);
  if (!point) return;
  map.easeTo({
    center: point,
    duration: 460,
    essential: true,
  });
}

function boundsFromPolygons(polygons = []) {
  const bounds = new maplibregl.LngLatBounds();
  let count = 0;
  for (const polygon of polygons) {
    for (const point of polygon ?? []) {
      if (!Array.isArray(point) || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) continue;
      bounds.extend([point[0], point[1]]);
      count += 1;
    }
  }
  return count ? bounds : null;
}

function defaultSceneBounds({liveOnly = false} = {}) {
  if (liveOnly) return boundsFromPolygons(liveCells.map((cell) => cell.footprintLonLat));
  if (datumGround?.outerRing?.length) return boundsFromPolygons([datumGround.outerRing]);
  return boundsFromPolygons(structuralCells.map((cell) => cell.footprintLonLat));
}

function frameScene({liveOnly = false} = {}) {
  const bounds = defaultSceneBounds({liveOnly});
  if (!bounds || !runtime) return;
  const is2d = sceneMode === '2d';
  const padding = is2d
    ? {top: 64, right: 86, bottom: 72, left: 88}
    : {top: 100, right: 108, bottom: 116, left: 420};
  map.stop();
  map.fitBounds(bounds, {
    padding,
    maxZoom: liveOnly ? 12.6 : 11.8,
    duration: 540,
    essential: true,
  });
  map.once('moveend', () => {
    map.easeTo({
      bearing: is2d ? 0 : -25,
      pitch: is2d ? 0 : Math.min(62, runtime.maxCameraPitch),
      duration: 230,
      essential: true,
    });
  });
}

function updateFullscreenButton() {
  if (!fullscreenButton) return;
  const active = document.fullscreenElement === viewerShell;
  fullscreenButton.textContent = active ? '⛶' : '⛶';
  fullscreenButton.title = active ? 'Exit fullscreen' : 'Fullscreen';
}

function toggleFullscreen() {
  if (!viewerShell) return;
  if (document.fullscreenElement) document.exitFullscreen?.();
  else viewerShell.requestFullscreen?.();
}

function flashScreenshotStatus(message, kind = 'ok') {
  if (!screenshotStatus) return;
  if (screenshotStatusTimer) clearTimeout(screenshotStatusTimer);
  screenshotStatus.textContent = message;
  screenshotStatus.classList.toggle('error', kind === 'error');
  screenshotStatus.classList.add('visible');
  screenshotStatusTimer = window.setTimeout(() => {
    screenshotStatus.classList.remove('visible');
  }, 2800);
}

function safeScreenshotStamp() {
  return new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '');
}

async function saveCompositeScreenshot() {
  try {
    if (!runtime) throw new Error('Viewer is still loading.');
    map.triggerRepaint();
    deckOverlay.setProps({layers: makeLayers()});
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

    const baseCanvas = map.getCanvas();
    const particleCanvas = deckOverlay.getCanvas?.();
    if (!baseCanvas) throw new Error('Map canvas is unavailable.');

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = baseCanvas.width;
    exportCanvas.height = baseCanvas.height;
    const context = exportCanvas.getContext('2d');
    if (!context) throw new Error('2D export canvas is unavailable.');

    context.drawImage(baseCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
    if (particleCanvas && particleCanvas !== baseCanvas) {
      context.drawImage(particleCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
    }

    const blob = await new Promise((resolve) => exportCanvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Browser refused the canvas export.');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `jakarta_proto1_${sceneMode}_${safeScreenshotStamp()}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    flashScreenshotStatus('Screenshot saved · map + native DeckGL particles', 'ok');
  } catch (error) {
    console.error('[Proto1 DeckGL] Screenshot failed:', error);
    flashScreenshotStatus(`Screenshot failed: ${error?.message ?? String(error)}`, 'error');
  }
}

function normalizedTwoDAnalysisConfig(source = {}) {
  const rumFillOpacityMin = clamp(
    Number(source.rum_fill_opacity_min ?? DEFAULT_TWO_D_ANALYSIS.rumFillOpacityMin),
    0.10,
    0.95,
  );
  const rumFillOpacityMax = clamp(
    Number(source.rum_fill_opacity_max ?? DEFAULT_TWO_D_ANALYSIS.rumFillOpacityMax),
    rumFillOpacityMin,
    0.98,
  );
  const preferred = String(
    source.preferred_basemap_mode ?? DEFAULT_TWO_D_ANALYSIS.preferredBasemapMode,
  ).toLowerCase();
  return {
    rumFillOpacity: clamp(
      Number(source.rum_fill_opacity ?? DEFAULT_TWO_D_ANALYSIS.rumFillOpacity),
      rumFillOpacityMin,
      rumFillOpacityMax,
    ),
    rumFillOpacityMin,
    rumFillOpacityMax,
    rumFillOpacityStep: clamp(
      Number(source.rum_fill_opacity_step ?? DEFAULT_TWO_D_ANALYSIS.rumFillOpacityStep),
      0.01,
      0.10,
    ),
    preferredBasemapMode: BASEMAPS[preferred] ? preferred : DEFAULT_TWO_D_ANALYSIS.preferredBasemapMode,
    rumOutlineRgba: clampRgba(
      source.rum_outline_rgba,
      DEFAULT_TWO_D_ANALYSIS.rumOutlineRgba,
    ),
    rumOutlineWidthPixels: clamp(
      Number(source.rum_outline_width_px ?? DEFAULT_TWO_D_ANALYSIS.rumOutlineWidthPixels),
      0,
      2.0,
    ),
    flatParticleZM: Number(source.flat_particle_z_m ?? DEFAULT_TWO_D_ANALYSIS.flatParticleZM),
  };
}

function formatTwoDOpacity(value) {
  return `${Math.round(clamp(Number(value), 0, 1) * 100)}%`;
}

function updateTwoDAnalysisControls() {
  if (twoDRumOpacitySlider) {
    twoDRumOpacitySlider.min = String(twoDAnalysisConfig.rumFillOpacityMin);
    twoDRumOpacitySlider.max = String(twoDAnalysisConfig.rumFillOpacityMax);
    twoDRumOpacitySlider.step = String(twoDAnalysisConfig.rumFillOpacityStep);
    twoDRumOpacitySlider.value = String(twoDAnalysisConfig.rumFillOpacity);
    twoDRumOpacitySlider.disabled = !runtime;
  }
  if (twoDRumOpacityValue) {
    twoDRumOpacityValue.textContent = formatTwoDOpacity(twoDAnalysisConfig.rumFillOpacity);
  }
  if (twoDAnalysisLegend) {
    const preferred = twoDAnalysisConfig.preferredBasemapMode;
    const currentLabel = BASEMAPS[basemapMode]?.label ?? basemapMode;
    const preference = basemapMode === preferred
      ? `${currentLabel} is the recommended analytical background.`
      : `${BASEMAPS[preferred]?.label ?? 'Soft B/W'} is recommended for clean field reading.`;
    twoDAnalysisLegend.innerHTML =
      `<span>Fill = observed long-term vertical velocity · ${formatTwoDOpacity(twoDAnalysisConfig.rumFillOpacity)} opacity</span>` +
      `<span>Vertical colour is adaptive: white = 0, red = subsidence, blue = uplift.</span>` +
      `<span>Thin cell edges preserve RUM boundaries without hiding the map.</span>` +
      `<span>Basemap: ${preference}</span>`;
  }
}

function velocityLegendAnchors(scale, stops) {
  const tau = Math.max(0, Number(scale.nearZeroThresholdMmYr));
  const subLimit = Math.max(
    Math.abs(Number(stops[0]?.valueMmYr ?? 0)),
    Number(scale.subsidenceLimitMmYr ?? 0),
  );
  const upliftLimit = Math.max(
    Math.abs(Number(stops.at(-1)?.valueMmYr ?? 0)),
    Number(scale.upliftLimitMmYr ?? 0),
  );
  const layout = scale.legendLayout ?? scale.legend_layout ?? {};
  const damping = clamp(Number(layout.zeroPositionDamping ?? layout.zero_position_damping ?? 0.60), 0, 1);
  const minZeroPct = clamp(Number(layout.zeroPositionMinPct ?? layout.zero_position_min_pct ?? 35), 0, 100);
  const maxZeroPct = clamp(Number(layout.zeroPositionMaxPct ?? layout.zero_position_max_pct ?? 72), minZeroPct, 100);
  const stableBandWidthPct = clamp(Number(layout.stableBandWidthPct ?? layout.stable_band_width_pct ?? 16), 4, 32);
  const rawZeroPct = (subLimit > 0 || upliftLimit > 0)
    ? 100 * subLimit / Math.max(1e-9, subLimit + upliftLimit)
    : 50;
  // Preserve the true asymmetric proportions enough to communicate the
  // overwhelmingly subsiding Jakarta field, while damping the shift so the
  // uplift wing remains readable.
  const zeroPct = clamp(50 + (rawZeroPct - 50) * damping, minZeroPct, maxZeroPct);
  const negativeEdgePct = clamp(zeroPct - stableBandWidthPct * 0.5, 0, zeroPct);
  const positiveEdgePct = clamp(zeroPct + stableBandWidthPct * 0.5, zeroPct, 100);

  const negativeStops = stops.filter((stop) => stop.valueMmYr < -tau - 1e-9);
  const positiveStops = stops.filter((stop) => stop.valueMmYr > tau + 1e-9);
  const clipPadFraction = (sideStops, sideSpan, fromStart) => {
    if (!(sideSpan > 1e-9) || sideStops.length < 2) return 0.18;
    const edgeStep = fromStart
      ? Math.abs(sideStops[1].valueMmYr - sideStops[0].valueMmYr)
      : Math.abs(sideStops.at(-1).valueMmYr - sideStops.at(-2).valueMmYr);
    return clamp(edgeStep / sideSpan, 0.08, 0.32);
  };
  const negativeClipPadFraction = clipPadFraction(negativeStops, Math.max(1e-9, subLimit - tau), true);
  const positiveClipPadFraction = clipPadFraction(positiveStops, Math.max(1e-9, upliftLimit - tau), false);
  const negativeLimitPct = negativeEdgePct * negativeClipPadFraction;
  const positiveLimitPct = 100 - (100 - positiveEdgePct) * positiveClipPadFraction;

  return {
    tau,
    subLimit,
    upliftLimit,
    rawZeroPct,
    zeroPct,
    negativeEdgePct,
    positiveEdgePct,
    negativeLimitPct,
    positiveLimitPct,
  };
}

function velocityLegendPosition(valueMmYr, scale, stops, anchors) {
  const value = Number(valueMmYr);
  if (!Number.isFinite(value)) return anchors.zeroPct;
  const {
    tau,
    subLimit,
    upliftLimit,
    negativeLimitPct,
    negativeEdgePct,
    zeroPct,
    positiveEdgePct,
    positiveLimitPct,
  } = anchors;
  const interpolate = (start, end, amount) => start + (end - start) * clamp(amount, 0, 1);

  if (tau > 0 && subLimit > tau && value <= -tau) {
    return interpolate(negativeLimitPct, negativeEdgePct, (value + subLimit) / (subLimit - tau));
  }
  if (tau > 0 && value <= 0) {
    return interpolate(negativeEdgePct, zeroPct, (value + tau) / tau);
  }
  if (tau > 0 && value <= tau) {
    return interpolate(zeroPct, positiveEdgePct, value / tau);
  }
  if (tau > 0 && upliftLimit > tau) {
    return interpolate(positiveEdgePct, positiveLimitPct, (value - tau) / (upliftLimit - tau));
  }
  return zeroPct;
}

function pickVelocityLegendIntermediate(stops, minimum, maximum, target) {
  const candidates = stops.filter((stop) => (
    stop.valueMmYr > minimum + 1e-6 && stop.valueMmYr < maximum - 1e-6
  ));
  if (!candidates.length) return null;
  return candidates.reduce((best, candidate) => (
    Math.abs(candidate.valueMmYr - target) < Math.abs(best.valueMmYr - target) ? candidate : best
  ));
}

function nearestVelocityLegendStop(stops, valueMmYr) {
  return stops.reduce((best, stop) => (
    Math.abs(stop.valueMmYr - valueMmYr) < Math.abs(best.valueMmYr - valueMmYr) ? stop : best
  ));
}

function buildVelocityLegendEntries(scale, stops, anchors) {
  const entries = [];
  const epsilon = 1e-6;
  const upsert = (entry) => {
    const index = entries.findIndex((candidate) => (
      Math.abs(candidate.valueMmYr - entry.valueMmYr) <= epsilon
    ));
    if (index >= 0) entries[index] = {...entries[index], ...entry};
    else entries.push(entry);
  };
  const add = (valueMmYr, label, positionPct) => {
    const stop = nearestVelocityLegendStop(stops, valueMmYr);
    upsert({
      valueMmYr,
      label,
      color: stop.color,
      positionPct: clamp(positionPct, 0, 100),
    });
  };

  const {subLimit, upliftLimit, tau, negativeLimitPct, zeroPct, positiveLimitPct} = anchors;
  const negativeLimit = -subLimit;
  const positiveLimit = upliftLimit;
  add(negativeLimit, `≤${formatVelocityLegendNumber(negativeLimit)}`, negativeLimitPct);

  const negativeMid = pickVelocityLegendIntermediate(
    stops,
    negativeLimit,
    -tau,
    -0.5 * (subLimit + tau),
  );
  if (negativeMid) add(
    negativeMid.valueMmYr,
    formatVelocityLegendNumber(negativeMid.valueMmYr),
    velocityLegendPosition(negativeMid.valueMmYr, scale, stops, anchors),
  );

  if (tau > epsilon && negativeLimit < -tau - epsilon) {
    add(-tau, formatVelocityLegendNumber(-tau), anchors.negativeEdgePct);
  }
  add(0, '0', zeroPct);
  if (tau > epsilon && positiveLimit > tau + epsilon) {
    add(tau, formatVelocityLegendNumber(tau), anchors.positiveEdgePct);
  }

  const positiveMid = pickVelocityLegendIntermediate(
    stops,
    tau,
    positiveLimit,
    0.5 * (upliftLimit + tau),
  );
  if (positiveMid) add(
    positiveMid.valueMmYr,
    formatVelocityLegendNumber(positiveMid.valueMmYr),
    velocityLegendPosition(positiveMid.valueMmYr, scale, stops, anchors),
  );

  add(positiveLimit, `≥${formatVelocityLegendNumber(positiveLimit)}`, positiveLimitPct);
  return entries.sort((left, right) => left.positionPct - right.positionPct);
}

function updateVerticalVelocityLegend() {
  const scale = runtime?.verticalVelocityColorScale;
  if (!scale) return;
  if (verticalVelocityLegendTitle) verticalVelocityLegendTitle.textContent = scale.legend?.title ?? `Vertical velocity · ${scale.unit ?? 'mm/yr'}`;

  const stops = scale.stops ?? [];
  if (stops.length < 2) return;
  const anchors = velocityLegendAnchors(scale, stops);

  if (verticalVelocityLegendScale) {
    // The unlabelled end caps represent values clipped beyond the sign-specific
    // P98 limits. They use the same endpoint colours as the real RUM scale.
    const first = stops[0];
    const last = stops.at(-1);
    const gradient = [
      `rgb(${first.color.join(', ')}) 0%`,
      ...stops.map((stop) => {
        const position = velocityLegendPosition(stop.valueMmYr, scale, stops, anchors);
        return `rgb(${stop.color.join(', ')}) ${position.toFixed(2)}%`;
      }),
      `rgb(${last.color.join(', ')}) 100%`,
    ];
    verticalVelocityLegendScale.style.background = `linear-gradient(to right, ${gradient.join(', ')})`;
  }

  if (verticalVelocityLegendLabels) {
    const labels = buildVelocityLegendEntries(scale, stops, anchors);
    verticalVelocityLegendLabels.innerHTML = labels.map((entry, index) => {
      const edgeClass = index === 0 ? ' legendLabelStart' : index === labels.length - 1 ? ' legendLabelEnd' : '';
      return `<span class="legendLabel${edgeClass}" style="left:${entry.positionPct.toFixed(2)}%">${entry.label}</span>`;
    }).join('');
  }

  if (verticalVelocityLegendNote) {
    const tau = formatVelocityLegendNumber(scale.nearZeroThresholdMmYr);
    const sub = formatVelocityLegendNumber(-scale.subsidenceLimitMmYr);
    const up = formatVelocityLegendNumber(scale.upliftLimitMmYr);
    verticalVelocityLegendNote.textContent =
      `0 = stable · red = subsidence · blue = uplift · display clipping ${sub} / ${up} ${scale.unit}; near-zero ±${tau}`;
  }
}

function updateModeSpecificLegends() {
  const is2d = sceneMode === '2d';
  if (horizontalGlyphLegend) {
    horizontalGlyphLegend.innerHTML = is2d
      ? '<span class="glyph-legend-arrow">Arrow = observed horizontal velocity on the flat analysis plane</span>' +
        '<span class="glyph-legend-ellipse">Ellipse = 1σ E/N uncertainty at arrow tip</span>'
      : '<span class="glyph-legend-arrow">Arrow = horizontal velocity</span>' +
        '<span class="glyph-legend-ellipse">Ellipse = 1σ E/N uncertainty</span>';
  }
  if (horizontalParticleLegend) {
    horizontalParticleLegend.innerHTML = is2d
      ? '<span>Flat analysis-plane trails · no epoch height attachment</span>' +
        '<span>Live RUM field only; blankies never seed or steer particles</span>'
      : '<span>GPU state + world-space trails</span>' +
        '<span>Live RUM field only; blankies never seed or steer particles</span>';
  }
}

function setLegendBarState(element, active) {
  if (!element) return;
  element.classList.toggle('inactive', !active);
  element.setAttribute('aria-hidden', active ? 'false' : 'true');
}

function updateFloatingLegendBars() {
  const glyphMasterEnabled = horizontalGlyphMasterEnabled();
  const particleMasterEnabled = horizontalParticleMasterEnabled();
  const glyphsAvailable = Boolean(runtime && horizontalGlyphConfig.enabled && horizontalGlyphRecords.length);
  const glyphsActive = glyphMasterEnabled && glyphsAvailable && (showHorizontalArrows || showHorizontalEllipses);
  const glyphLegend = horizontalGlyphConfig.legend ?? {};
  const glyphEllipseLabel = glyphLegend.label ?? '1σ major';
  setLegendBarState(horizontalGlyphLegendBar, glyphsActive);
  if (horizontalGlyphLegendBarArrowText) {
    horizontalGlyphLegendBarArrowText.innerHTML =
      `<span class="horizontalLegendTextMain">P75</span>` +
      `<span class="horizontalLegendTextSub">${formatLegendMmYr(glyphLegend.speedP75MmYr)}</span>`;
  }
  if (horizontalGlyphLegendBarEllipseText) {
    horizontalGlyphLegendBarEllipseText.innerHTML =
      `<span class="horizontalLegendTextMain">${glyphEllipseLabel}</span>` +
      `<span class="horizontalLegendTextSub">${formatLegendMmYr(glyphLegend.ellipseMajorP75MmYr)}</span>`;
  }
  if (horizontalGlyphLegendBar) {
    horizontalGlyphLegendBar.title =
      `Horizontal glyph reference · visible RUMs only · |v| P75 ${formatLegendMmYr(glyphLegend.speedP75MmYr)} · ` +
      `1σ major P75 ${formatLegendMmYr(glyphLegend.ellipseMajorP75MmYr)} at vector tip`;
  }

  const activeField = activeHorizontalParticleFieldRuntime();
  const particleLegend = activeField?.legend ?? {directionalUncertainty: {}};
  const particleDirectional = particleLegend.directionalUncertainty ?? {};
  const particlesActive = Boolean(particleMasterEnabled && runtime && horizontalParticleConfig.enabled && showHorizontalParticles && activeField);
  const uncertaintyStrength = horizontalParticleUncertaintyForMode(horizontalParticleMode);
  setLegendBarState(horizontalParticleLegendBar, particlesActive);
  if (horizontalParticleLegendBarSpeedText) {
    horizontalParticleLegendBarSpeedText.textContent = `${formatLegendMmYr(particleLegend.speedP75MmYr)} (P75)`;
  }
  if (horizontalParticleLegendBarUncertaintyText) {
    if (horizontalParticleMode === 'mean') {
      horizontalParticleLegendBarUncertaintyText.textContent = 'σθ 0°';
    } else {
      const multiplierText = Math.abs(uncertaintyStrength - 1) > 0.005
        ? ` ×${uncertaintyStrength.toFixed(2).replace(/^0/, '')}`
        : '';
      horizontalParticleLegendBarUncertaintyText.textContent =
        `σθ ${formatLegendTheta(particleDirectional.sigmaThetaP75Deg)} (1σ)${multiplierText}`;
    }
  }
  if (horizontalParticleLegendBar) {
    const fieldLabel = horizontalParticleIsLscMode() ? 'LSC signal' : 'Raw V1';
    const modeLabel = horizontalParticleMode === 'montecarlo'
      ? 'Monte Carlo directional sampling'
      : horizontalParticleMode === 'shimmer'
        ? 'Shimmer directional reference'
        : 'Mean flow';
    horizontalParticleLegendBar.title =
      `${fieldLabel} · ${modeLabel} · |v| P75 ${formatLegendMmYr(particleLegend.speedP75MmYr)} · ` +
      `σθ P75 ${formatLegendTheta(particleDirectional.sigmaThetaP75Deg)} (1σ; glyph-visible support)`;
  }

  horizontalLegendRenderer.setState({
    glyph: {
      active: glyphsActive,
      arrowActive: glyphMasterEnabled && glyphsAvailable && showHorizontalArrows,
      ellipseActive: glyphMasterEnabled && glyphsAvailable && showHorizontalEllipses,
      glyphOpacity: horizontalGlyphOpacity,
      glyphScale: horizontalGlyphScale,
      speedP75MmYr: glyphLegend.speedP75MmYr,
      ellipseMajorP75MmYr: glyphLegend.ellipseMajorP75MmYr,
      ellipseMinorP75MmYr: glyphLegend.ellipseMinorP75MmYr,
      arrowReferenceMmYr: glyphLegend.arrowReferenceMmYr || horizontalGlyphConfig.arrowSpeedReferenceMmYr,
      ellipseMajorReferenceMmYr: glyphLegend.ellipseMajorReferenceMmYr,
      ellipseConfidenceVisualScale: horizontalGlyphConfig.ellipseLegendVisualScale,
      arrowColorRgba: horizontalGlyphConfig.arrowColorRgba,
      ellipseColorRgba: horizontalGlyphConfig.ellipseColorRgba,
    },
    particle: {
      active: particlesActive,
      // Reset only when the scientific field or uncertainty renderer changes.
      // Slider updates intentionally keep the same small particle population
      // alive, so speed/opacity/tail changes read as live control changes.
      resetKey: `${horizontalParticleIsLscMode() ? 'lsc' : 'raw'}:${horizontalParticleMode}`,
      mode: horizontalParticleMode,
      fieldMode: horizontalParticleIsLscMode() ? 'lsc' : 'raw',
      speedP75MmYr: particleLegend.speedP75MmYr,
      speedP95MmYr: activeField?.speedP95MmYr,
      sigmaThetaP75Deg: particleDirectional.sigmaThetaP75Deg,
      uncertaintyStrength,
      particleCount: horizontalParticleCount,
      speedMultiplier: horizontalParticleSpeedMultiplier,
      particleSizeMultiplier: horizontalParticleSizeMultiplier,
      particleOpacity: horizontalParticleOpacity,
      trailPersistence: horizontalParticleTrailPersistence,
      trailDurationSeconds: horizontalParticleTrailDurationSeconds,
      shimmerPixelAmplitude: activeField?.render?.shimmerPixelAmplitude,
      mcMaxSigma: activeField?.render?.mcMaxSigma,
      particleColorRgba: activeField?.render?.colorRgba,
    },
  });

  const reliefAvailable = Boolean(runtime);
  const reliefActive = Boolean(
    runtime && sceneMode === '3d' && verticalMasterEnabled() && uncertaintyReliefEnabled && capAppearance !== 'context-map',
  );
  setLegendBarState(verticalUncertaintyLegendBar, reliefActive);
  if (verticalUncertaintyLegendBar) {
    verticalUncertaintyLegendBar.classList.toggle('scene-disabled', reliefAvailable && sceneMode !== '3d');
  }
  if (verticalUncertaintyLegendBar) {
    const range = runtime?.uncertaintyRelief?.displayRange ?? {unit: 'sigma', value: 1};
    const anchor = verticalUncertaintyLegendAnchor(range);
    const titleText = verticalUncertaintyLegendSigmaLabel(range);
    const provenanceText = verticalUncertaintyLegendProvenanceLabel(runtime?.verticalUncertaintyLegend);
    if (verticalUncertaintyLegendZeroLabel) verticalUncertaintyLegendZeroLabel.textContent = '0';
    if (verticalUncertaintyLegendRightLabel) verticalUncertaintyLegendRightLabel.textContent = anchor.label;
    if (verticalUncertaintyLegendTitle) verticalUncertaintyLegendTitle.textContent = titleText;
    if (verticalUncertaintyLegendProvenanceTag) {
      verticalUncertaintyLegendProvenanceTag.hidden = !provenanceText;
      verticalUncertaintyLegendProvenanceTag.textContent = provenanceText;
    }
    if (verticalUncertaintyLegendBarText) {
      verticalUncertaintyLegendBarText.textContent = `Vertical uncertainty relief · ${titleText} · 0 to ${anchor.label}${provenanceText ? ' · synthetic demonstration uncertainty' : ''}`;
    }
    verticalUncertaintyLegendBar.title = `Vertical uncertainty relief · live RUMs only · ${titleText} · 0 to ${anchor.label} · global anchor across all epochs · ${reliefLodLabel()} relief${provenanceText ? ' · synthetic demonstration uncertainty' : ''}`;
    drawVerticalUncertaintyLegendProfile(anchor);
  }
}

function updateSceneModeUi() {
  const is2d = sceneMode === '2d';
  document.body.classList.toggle('scene-2d', is2d);
  document.body.classList.toggle('scene-3d', !is2d);

  if (sceneModeControl) {
    for (const button of sceneModeControl.querySelectorAll('.seg')) {
      button.classList.toggle('active', button.dataset.mode === sceneMode);
      button.disabled = !runtime;
    }
  }
  if (sceneModeNote) {
    sceneModeNote.textContent = is2d
      ? 'Flat analysis plane · static vertical velocity + horizontal deformation field'
      : 'Animated vertical deformation scene · epochs, relief, depth and context caps';
  }
  if (helpText) {
    helpText.innerHTML = is2d
      ? 'Observed RUM fill = static vertical velocity · white = 0, red = subsidence, blue = uplift<br>' +
        'Adaptive colour limits are display clipping, not hazard thresholds<br>' +
        'Opacity controls how much geographic context remains visible<br>' +
        'Particles, arrows, and ellipses lie on the flat analysis plane<br>' +
        'No epochs, pistons, walls, blankie support, or vertical relief<br>' +
        'Left drag = pan<br>Wheel = zoom<br>North-up · tilt and rotation locked'
      : 'Live RUM = observed InSAR product · cap colour = long-term vertical velocity<br>' +
        'Adaptive colour limits are display clipping, not hazard thresholds<br>' +
        'Grey map-textured cell = interpolated support, not measured, and stays flat<br>' +
        'Context-map mode locks Soft B/W and turns uncertainty relief off<br>' +
        'Horizontal glyphs: measured RUMs only; ellipse centred at arrow tip<br>' +
        'Amber datum line = completed support perimeter<br>' +
        'Left drag = pan<br>Middle drag = tilt / heading<br>Wheel = zoom<br>Right drag = zoom';
  }
  if (viewModeToggleButton) {
    // The raised badge is a current-view indicator, not an action label.
    // Clicking still toggles to the opposite scene mode.
    viewModeToggleButton.textContent = is2d ? '2D' : '3D';
    viewModeToggleButton.title = is2d
      ? 'Current view: 2D Analysis · click to switch to 3D Time Scene'
      : 'Current view: 3D Time Scene · click to switch to 2D Analysis';
    viewModeToggleButton.setAttribute(
      'aria-label',
      is2d
        ? 'Current view: 2D Analysis. Switch to 3D Time Scene.'
        : 'Current view: 3D Time Scene. Switch to 2D Analysis.',
    );
  }
  if (epochPanel) {
    epochPanel.classList.toggle('scene-disabled', is2d);
    epochPanel.title = is2d
      ? (trendlineOpen ? 'Trendline available; epoch playback controls are available in 3D mode' : 'Epoch playback controls available in 3D mode')
      : 'Time controls';
    epochPanel.setAttribute(
      'aria-disabled',
      is2d ? 'true' : 'false',
    );
  }
  updateTrendlinePanelState();
  updateEpochNavigationControls();
  updateReferenceGridControl();
  updateModeSpecificLegends();
  updateFloatingLegendBars();
  updateVerticalVelocityLegend();
  updateTwoDAnalysisControls();
  scheduleBottomStatusUpdate();

  if (!runtime) return;
  const disableTimeSceneControls = is2d;
  epochSlider.disabled = disableTimeSceneControls;
  verticalExagSlider.disabled = disableTimeSceneControls;
  playButton.disabled = disableTimeSceneControls;
  if (playbackSpeedSlider) playbackSpeedSlider.disabled = disableTimeSceneControls;
  updateBasemapControls();
  updateCapAppearanceControls();
}

function setSceneMode(requestedMode) {
  const nextMode = requestedMode === '2d' ? '2d' : '3d';
  if (!runtime || nextMode === sceneMode) {
    updateSceneModeUi();
    return;
  }

  stopPlayback();
  if (sceneMode === '3d') {
    savedThreeDCamera = captureCameraState();
    threeDBasemapMode = basemapMode;
  } else {
    savedTwoDCamera = captureCameraState();
    twoDBasemapMode = basemapMode;
  }
  sceneMode = nextMode;

  map.stop();
  if (sceneMode === '2d') {
    // First entry deliberately starts from a quiet B/W map; after that, retain
    // the user's own 2D basemap choice independently from the 3D scene.
    const desiredBasemap = twoDBasemapMode ?? twoDAnalysisConfig.preferredBasemapMode;
    if (BASEMAPS[desiredBasemap] && desiredBasemap !== basemapMode) {
      basemapMode = desiredBasemap;
      map.setStyle(BASEMAPS[desiredBasemap].style);
    }
    map.setMaxPitch(0);
    map.dragRotate.disable();
    map.touchZoomRotate?.disableRotation?.();
    const fallback = captureCameraState();
    const camera = savedTwoDCamera ?? {...fallback, bearing: 0, pitch: 0};
    map.easeTo({
      center: camera.center,
      zoom: camera.zoom,
      bearing: 0,
      pitch: 0,
      duration: 280,
      essential: true,
    });
  } else {
    map.setMaxPitch(runtime.maxCameraPitch);
    map.dragRotate.disable();
    map.touchZoomRotate?.enableRotation?.();
    const desiredBasemap = capAppearance === 'context-map'
      ? 'bw'
      : (threeDBasemapMode ?? basemapMode);
    if (BASEMAPS[desiredBasemap] && desiredBasemap !== basemapMode) {
      basemapMode = desiredBasemap;
      map.setStyle(BASEMAPS[desiredBasemap].style);
    }
    const fallback = {...captureCameraState(), bearing: -25, pitch: 62};
    const camera = savedThreeDCamera ?? fallback;
    map.easeTo({
      center: camera.center,
      zoom: camera.zoom,
      bearing: camera.bearing,
      pitch: clamp(camera.pitch, 0, runtime.maxCameraPitch),
      duration: 280,
      essential: true,
    });
  }

  updateSceneModeUi();
  syncCameraDepthContract({force: true});
  applyEpoch();
}

function updateReadingNote() {
  if (sceneMode === '2d') {
    readingNote.className = 'reading-note good';
    readingNote.innerHTML =
      `<strong>2D analysis plane.</strong> Observed RUMs show static vertical velocity at ${formatTwoDOpacity(twoDAnalysisConfig.rumFillOpacity)} opacity, with thin boundaries so geographic context remains readable. ` +
      'Horizontal particles, arrows, and ellipses are flattened to z = 0. ' +
      `<br><strong>${BASEMAPS[twoDAnalysisConfig.preferredBasemapMode]?.label ?? 'Soft B/W'} is the recommended analysis background.</strong> Map and satellite remain available for geographic inspection. ` +
      '<br><strong>Time-scene geometry is intentionally absent:</strong> no epochs, pistons, walls, blankie support, datum apron, or uncertainty relief.';
    return;
  }

  const blankCount = runtime?.blankCount?.toLocaleString() ?? '…';
  const atlasReady = contextAtlasState === 'ready';

  if (!depthOccludersEnabled) {
    readingNote.className = 'reading-note diagnostic';
    readingNote.innerHTML =
      '<strong>Diagnostic B — depth masks off.</strong> The see-through apron and blankie depth prepasses are disabled. ' +
      'Only use this for the top-down A/B check.';
    return;
  }

  if (capAppearance === 'context-map') {
    readingNote.className = 'reading-note good';
    readingNote.innerHTML =
      `<strong>Context-map caps.</strong> Soft B/W is locked as the basemap. Live RUM caps carry their local map texture with a transparent deformation-colour veil; ` +
      `${blankCount} blankies retain local soft B/W texture with a grey support veil. ` +
      `<br><strong>Uncertainty relief is off in this mode.</strong> ${atlasReady ? 'The cap atlas is locked to the true geographic cap corners, so it moves with each RUM rather than acting like a transparent window.' : 'Atlas is still loading; flat-cap fallback is temporary.'}`;
    return;
  }

  if (apronMode === 'see-through') {
    readingNote.className = 'reading-note good';
    readingNote.innerHTML =
      `<strong>Completed support envelope.</strong> ${blankCount} no-data cells move by IDW from nearby live RUMs. ` +
      `${atlasReady ? 'Blankies use their own local soft B/W cap texture plus a grey support veil, so they are no longer transparent windows. ' : 'Blankies remain flat grey until the local context atlas finishes loading. '}` +
      `<br><strong>Vertical uncertainty.</strong> ${uncertaintyReliefEnabled
        ? `Instanced ${reliefLodLabel()} checkerboard relief is active on measured RUMs only at ${runtime?.uncertaintyRelief?.displayRange?.unit === 'sigma' ? `±${runtime?.uncertaintyRelief?.displayRange?.value ?? 1}σ` : `±${runtime?.uncertaintyRelief?.displayRange?.value ?? 0} mm`}. Low-amplitude relief fades back to the mean cap.`
        : 'Flat-cap mode is active; sigma remains available in measured-RUM tooltips.'}`;
  } else if (apronMode === 'solid') {
    readingNote.className = 'reading-note good';
    readingNote.innerHTML =
      '<strong>Solid datum apron.</strong> Blankies retain their interpolated motion and local context texture; only the outer datum apron is shown solid.';
  } else {
    readingNote.className = 'reading-note bad';
    readingNote.innerHTML =
      '<strong>Outer apron off.</strong> The support cells still animate, but the exterior rim may be visible again.';
  }
}

function createLiveCells(staticRums, verticalVelocityColorScale) {
  const fallbackCellSizeM = runtime?.grid?.rumSizeM ?? 450;
  return staticRums.map((rum) => {
    const transform = deriveFootprintTransform(rum.footprintLonLat, fallbackCellSizeM);
    return {
      cellId: rum.rumId,
      rumId: rum.rumId,
      runtimeRowIndex: rum.rumIndex,
      gridI: rum.gridI,
      gridJ: rum.gridJ,
      isLive: true,
      isBlankie: false,
      footprintLonLat: rum.footprintLonLat,
      lon: Number(rum.lon),
      lat: Number(rum.lat),
      upMmYr: rum.upMmYr,
      sourceSigmaUpMmYr: Math.sqrt(Math.max(0, Number(rum.varUp) || 0)),
      fillColor: velocityColor(rum.upMmYr, verticalVelocityColorScale),
      reliefPosition: transform.position,
      reliefWidthM: transform.widthM,
      reliefHeightM: transform.heightM,
      reliefYawDeg: transform.yawDeg,
      reliefAmplitudeM: 0,
      reliefVisualWeight: 0,
      reliefVisualBucket: -1,
      displacementMm: 0,
      measurementMm: 0,
      sigmaMm: 0,
      displayZ: 0,
      capPolygon3d: null,
    };
  });
}

function createBlankieCells(blankies) {
  const fallbackCellSizeM = runtime?.grid?.rumSizeM ?? 450;
  return blankies.map((blankie) => {
    const transform = deriveFootprintTransform(blankie.footprintLonLat, fallbackCellSizeM);
    return {
      cellId: blankie.blankieId,
      rumId: blankie.blankieId,
      runtimeRowIndex: blankie.runtimeRowIndex,
      gridI: blankie.gridI,
      gridJ: blankie.gridJ,
      isLive: false,
      isBlankie: true,
      footprintLonLat: blankie.footprintLonLat,
      upMmYr: blankie.upMmYrInterpolated,
      selectionReasons: blankie.selectionReasons,
      supportType: blankie.supportType,
      valueStatus: blankie.valueStatus,
      interpolation: blankie.interpolation,
      reliefPosition: transform.position,
      reliefWidthM: transform.widthM,
      reliefHeightM: transform.heightM,
      reliefYawDeg: transform.yawDeg,
      reliefAmplitudeM: 0,
      reliefVisualWeight: 0,
      reliefVisualBucket: -1,
      displacementMm: 0,
      measurementMm: 0,
      sigmaMm: 0,
      displayZ: 0,
      capPolygon3d: null,
    };
  });
}

function buildWallRecord(cellA, cellB, edgeLonLat, lower, upper) {
  const isSupportWall = cellA.isBlankie || cellB.isBlankie;
  return {
    polygon3d: [
      [edgeLonLat[0][0], edgeLonLat[0][1], lower],
      [edgeLonLat[1][0], edgeLonLat[1][1], lower],
      [edgeLonLat[1][0], edgeLonLat[1][1], upper],
      [edgeLonLat[0][0], edgeLonLat[0][1], upper],
    ],
    fillColor: isSupportWall
      ? BLANKIE_WALL_COLOR
      : darken(velocityColor(0.5 * (cellA.upMmYr + cellB.upMmYr), runtime.verticalVelocityColorScale)),
    isSupportWall,
  };
}


function clampRgba(source, fallback) {
  const values = Array.isArray(source) ? source : fallback;
  return [0, 1, 2, 3].map((index) => clamp(Math.round(Number(values[index] ?? fallback[index])), 0, 255));
}

function finiteLegendValue(value, fallback = NaN) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

const HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER = 1;
const HORIZONTAL_GLYPH_ELLIPSE_BASELINE_95_SCALE = Math.sqrt(-2 * Math.log(1 - 0.95));

function chi2Scale2dProbability(probability) {
  const p = clamp(Number(probability), 1e-9, 0.999999999);
  return Math.sqrt(-2 * Math.log(1 - p));
}

function probabilityFromChi2Scale2d(scale) {
  const safeScale = Math.max(0, Number(scale) || 0);
  return clamp(1 - Math.exp(-0.5 * safeScale * safeScale), 1e-9, 0.999999999);
}

function horizontalGlyphEllipseSourceConfidenceScale(payload = {}) {
  const ellipse = payload.ellipse ?? {};
  const explicitScale = Number(ellipse.confidenceScale ?? ellipse.sigmaMultiplier);
  if (Number.isFinite(explicitScale) && explicitScale > 0) return explicitScale;
  const probability = Number(ellipse.confidenceProbability ?? payload.legend?.confidenceProbability);
  if (Number.isFinite(probability) && probability > 0 && probability < 1) {
    return chi2Scale2dProbability(probability);
  }
  return HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER;
}

function scaleLegendValue(value, factor) {
  const numeric = finiteLegendValue(value);
  return Number.isFinite(numeric) ? numeric * factor : numeric;
}

function normalizeHorizontalGlyphLegend(source = {}, ellipseDisplayFactor = 1) {
  const safeFactor = Number.isFinite(Number(ellipseDisplayFactor)) ? Math.max(0, Number(ellipseDisplayFactor)) : 1;
  return {
    statistic: String(source.statistic ?? 'P75'),
    speedP75MmYr: finiteLegendValue(source.speedP75MmYr),
    ellipseMajorP75MmYr: scaleLegendValue(source.ellipseMajorP75MmYr, safeFactor),
    ellipseMinorP75MmYr: scaleLegendValue(source.ellipseMinorP75MmYr, safeFactor),
    arrowReferenceMmYr: finiteLegendValue(source.arrowReferenceMmYr),
    ellipseMajorReferenceMmYr: scaleLegendValue(source.ellipseMajorReferenceMmYr, safeFactor),
    confidenceProbability: probabilityFromChi2Scale2d(HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER),
    sigmaMultiplier: HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER,
    label: '1σ major',
    visibleGlyphPairCount: Math.max(0, Math.round(finiteLegendValue(source.visibleGlyphPairCount, 0))),
  };
}

function normalizeHorizontalParticleLegend(source = {}) {
  const directional = source.directionalUncertainty ?? {};
  return {
    statistic: String(source.statistic ?? 'P75'),
    speedP75MmYr: finiteLegendValue(source.speedP75MmYr),
    directionalUncertainty: {
      sigmaThetaP75Deg: clamp(finiteLegendValue(directional.sigmaThetaP75Deg, 0), 0, 90),
      sigmaMultiplier: Math.max(0, finiteLegendValue(directional.sigmaMultiplier, 1)),
      capDeg: clamp(finiteLegendValue(directional.capDeg, 90), 1, 180),
      visibleCellCount: Math.max(0, Math.round(finiteLegendValue(directional.visibleCellCount, 0))),
      validFineTexelCount: Math.max(0, Math.round(finiteLegendValue(directional.validFineTexelCount, 0))),
    },
  };
}

function formatLegendMmYr(value, digits = 2) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)} mm/yr` : '— mm/yr';
}

function formatLegendTheta(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}°` : '—';
}

function formatLegendMm(value, digits = 2) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)} mm` : '— mm';
}

function percentileFromSorted(sortedValues, fraction) {
  if (!Array.isArray(sortedValues) || !sortedValues.length) return 0;
  const q = clamp(Number(fraction), 0, 1);
  const index = (sortedValues.length - 1) * q;
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sortedValues[lo];
  const t = index - lo;
  return sortedValues[lo] * (1 - t) + sortedValues[hi] * t;
}

function computeGlobalPercentile(values, percentileFraction = 0.75) {
  if (!Array.isArray(values) || !values.length) return 0;
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  return percentileFromSorted(finite, percentileFraction);
}

function computeVerticalUncertaintyLegendMetadata(staticRums, verticalSigmaMm, epochCount) {
  const liveRowIndices = Array.isArray(staticRums)
    ? staticRums
      .map((rum) => Number(rum?.rumIndex))
      .filter(Number.isFinite)
      .map((value) => Math.max(0, Math.round(value)))
    : [];
  const safeEpochCount = Math.max(0, Math.round(Number(epochCount) || 0));
  if (!liveRowIndices.length || !verticalSigmaMm?.length || !safeEpochCount) {
    return {
      statistic: 'P75',
      unit: 'mm',
      anchorScope: 'all_live_rums_all_epochs',
      liveCellCount: liveRowIndices.length,
      sampleCount: 0,
      globalP75Mm: 0,
      sigma_provenance: 'unknown',
    };
  }

  const values = new Array(liveRowIndices.length * safeEpochCount);
  let count = 0;
  for (const rowIndex of liveRowIndices) {
    const offset = rowIndex * safeEpochCount;
    for (let epochIndex = 0; epochIndex < safeEpochCount; epochIndex += 1) {
      const value = Number(verticalSigmaMm[offset + epochIndex]);
      if (Number.isFinite(value)) {
        values[count] = value;
        count += 1;
      }
    }
  }

  return {
    statistic: 'P75',
    unit: 'mm',
    anchorScope: 'all_live_rums_all_epochs',
    liveCellCount: liveRowIndices.length,
    sampleCount: count,
    globalP75Mm: computeGlobalPercentile(values.slice(0, count), 0.75),
    sigma_provenance: 'unknown',
  };
}

function normalizeVerticalUncertaintyLegend(payload = {}, staticRums, verticalSigmaMm, epochCount) {
  const globalP75Mm = Number(payload.globalP75Mm);
  if (Number.isFinite(globalP75Mm) && globalP75Mm >= 0) {
    return {
      statistic: String(payload.statistic ?? 'P75'),
      unit: String(payload.unit ?? 'mm'),
      anchorScope: String(payload.anchorScope ?? 'all_live_rums_all_epochs'),
      liveCellCount: Math.max(0, Math.round(Number(payload.liveCellCount ?? staticRums?.length ?? 0))),
      sampleCount: Math.max(0, Math.round(Number(payload.sampleCount ?? 0))),
      globalP75Mm,
      sigma_provenance: String(payload.sigma_provenance ?? payload.sigmaProvenance ?? 'unknown'),
    };
  }
  return computeVerticalUncertaintyLegendMetadata(staticRums, verticalSigmaMm, epochCount);
}

function verticalUncertaintyLegendSigmaLabel(displayRange = {unit: 'sigma', value: 1}) {
  if (displayRange?.unit === 'sigma') {
    const sigmaValue = Math.max(0, Number(displayRange.value ?? 1));
    const sigmaText = Number.isInteger(sigmaValue) ? sigmaValue.toFixed(0) : sigmaValue.toFixed(1);
    return `σz (${sigmaText}σ) · mm`;
  }
  return 'σz · mm';
}

function verticalUncertaintyLegendAnchor(displayRange = {unit: 'sigma', value: 1}) {
  const globalP75SigmaMm = Math.max(0, Number(runtime?.verticalUncertaintyLegend?.globalP75Mm) || 0);
  if (displayRange?.unit === 'sigma') {
    const sigmaMultiplier = Math.max(0, Number(displayRange.value ?? 1));
    return {
      valueMm: globalP75SigmaMm * sigmaMultiplier,
      label: `P75 ${formatLegendMm(globalP75SigmaMm * sigmaMultiplier)}`,
      globalP75SigmaMm,
      sigmaMultiplier,
      isPercentileAnchor: true,
    };
  }
  const valueMm = Math.max(0, Number(displayRange?.value ?? 0));
  return {
    valueMm,
    label: `max ${formatLegendMm(valueMm)}`,
    globalP75SigmaMm,
    sigmaMultiplier: 0,
    isPercentileAnchor: false,
  };
}

function verticalUncertaintyLegendProvenanceLabel(meta = {}) {
  const provenance = String(meta?.sigma_provenance ?? '').trim().toLowerCase();
  return provenance.startsWith('synthetic') ? 'demo σ' : '';
}

function buildVerticalUncertaintyLegendPath({
  startX = 30,
  endX = 244,
  baselineY = 11.75,
  maxAmplitudeUp = 3.5,
  maxAmplitudeDown = 4.0,
  halfCycles = 12,
} = {}) {
  const usableHalfCycles = Math.max(1, Math.round(Number(halfCycles) || 1));
  const span = Math.max(12, endX - startX);
  const commands = [`M ${startX - 8} ${baselineY.toFixed(2)}`, `L ${startX.toFixed(2)} ${baselineY.toFixed(2)}`];
  for (let cycleIndex = 0; cycleIndex < usableHalfCycles; cycleIndex += 1) {
    const tApex = (cycleIndex + 0.5) / usableHalfCycles;
    const xApex = startX + span * tApex;
    const xEnd = startX + span * ((cycleIndex + 1) / usableHalfCycles);
    const amplitudeScale = tApex;
    const signedAmplitude = (cycleIndex % 2 === 0)
      ? -maxAmplitudeUp * amplitudeScale
      : maxAmplitudeDown * amplitudeScale;
    const yApex = baselineY + signedAmplitude;
    commands.push(`L ${xApex.toFixed(2)} ${yApex.toFixed(2)}`);
    commands.push(`L ${xEnd.toFixed(2)} ${baselineY.toFixed(2)}`);
  }
  commands.push(`L ${(endX + 8).toFixed(2)} ${baselineY.toFixed(2)}`);
  return commands.join(' ');
}

function verticalUncertaintyLegendProfileState(anchor) {
  const geometry = runtime?.uncertaintyRelief?.geometry ?? {};
  const upGain = Math.max(0.05, Number(geometry.up_relief_gain ?? 0.75));
  const downGain = Math.max(0.05, Number(geometry.down_relief_gain ?? 0.85));
  const maxGain = Math.max(upGain, downGain);
  const maxExaggeration = Math.max(
    0.001,
    Number(runtime?.verticalExaggeration?.maxMPerMm ?? (verticalExaggeration || 1)),
  );
  const anchorSigmaMm = Math.max(0, Number(anchor?.globalP75SigmaMm) || 0);
  const anchorDisplayMm = Math.max(0, Number(anchor?.valueMm) || 0);
  const amplitudeM = anchorDisplayMm * Math.max(0, Number(verticalExaggeration) || 0);
  // Calibration is intentionally in the same visual currency as the scene:
  // global P75 σz at the maximum configured vertical exaggeration. This lets
  // the HUD profile react linearly to VE and to the 1σ/2σ display multiplier
  // without re-scaling the data anchor across playback.
  const calibrationM = Math.max(1e-6, anchorSigmaMm * maxExaggeration);
  const envelopeScale = clamp(amplitudeM / calibrationM, 0, 1);
  const visibleWeight = reliefVisualWeightForAmplitude(amplitudeM);
  const maxFullAmplitudePx = 4.0;
  const opacity = envelopeScale <= 0 ? 0 : clamp(0.16 + 0.84 * visibleWeight, 0.16, 1);

  return {
    maxAmplitudeUp: maxFullAmplitudePx * envelopeScale * (upGain / maxGain),
    maxAmplitudeDown: maxFullAmplitudePx * envelopeScale * (downGain / maxGain),
    opacity,
    envelopeScale,
    visibleWeight,
  };
}

function drawVerticalUncertaintyLegendProfile(anchor) {
  const state = verticalUncertaintyLegendProfileState(anchor);
  if (verticalUncertaintyLegendProfile) {
    verticalUncertaintyLegendProfile.setAttribute('d', buildVerticalUncertaintyLegendPath({
      maxAmplitudeUp: state.maxAmplitudeUp,
      maxAmplitudeDown: state.maxAmplitudeDown,
    }));
    verticalUncertaintyLegendProfile.style.opacity = state.opacity.toFixed(3);
  }
  if (verticalUncertaintyLegendMidline) {
    verticalUncertaintyLegendMidline.setAttribute('x1', '30');
    verticalUncertaintyLegendMidline.setAttribute('x2', '244');
    verticalUncertaintyLegendMidline.setAttribute('y1', '11.75');
    verticalUncertaintyLegendMidline.setAttribute('y2', '11.75');
    verticalUncertaintyLegendMidline.style.opacity = state.envelopeScale <= 0 ? '0.2' : '1';
  }
}

function normalizeHorizontalGlyphConfig(payload = {}) {
  const render = payload.render ?? {};
  const scaling = payload.scaling ?? {};
  const visibility = payload.visibility ?? {};
  const ellipse = payload.ellipse ?? {};
  const ellipseSourceConfidenceScale = horizontalGlyphEllipseSourceConfidenceScale(payload);
  const ellipseDisplayFactor = ellipseSourceConfidenceScale > 0
    ? HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER / ellipseSourceConfidenceScale
    : 1;
  const ellipseLegendVisualScale = HORIZONTAL_GLYPH_ELLIPSE_SIGMA_MULTIPLIER / HORIZONTAL_GLYPH_ELLIPSE_BASELINE_95_SCALE;
  return {
    enabled: render.enabled !== false,
    showArrowsByDefault: render.showArrowsByDefault !== false,
    showEllipsesByDefault: render.showEllipsesByDefault !== false,
    defaultOpacity: clamp(Number(render.defaultOpacity ?? 0.92), 0, 1),
    arrowColorRgba: clampRgba(render.arrowColorRgba, [34, 34, 34, 240]),
    ellipseColorRgba: clampRgba(render.ellipseColorRgba, [0, 240, 216, 210]),
    clearanceAboveCapM: Math.max(0, Number(scaling.clearanceAboveCapM ?? 6)),
    ellipseSegments: clamp(Math.round(Number(render.ellipseSegments ?? 64)), 16, 128),
    ellipseRingInnerRadius: clamp(Number(render.ellipseRingInnerRadius ?? 0.94), 0.65, 0.995),
    visibleGlyphPairCount: Number(payload.summary?.visibleGlyphPairCount ?? 0),
    skippedLowSpeed: Number(payload.summary?.skippedLowSpeed ?? 0),
    skippedInsignificant: Number(payload.summary?.skippedInsignificantVsUncertainty ?? 0),
    minimumSpeedMmYr: Number(visibility.minimumSpeedMmYr ?? 0.02),
    significanceSigmaMultiplier: Number(visibility.significanceSigmaMultiplier ?? 1),
    ellipseLabel: '1σ East-North uncertainty ellipse at vector tip',
    ellipseSourceConfidenceScale,
    ellipseDisplayFactor,
    ellipseLegendVisualScale,
    arrowScaleMPerMmYr: Number(scaling.arrowScaleMPerMmYr ?? 0),
    arrowSpeedReferenceMmYr: Number(scaling.arrowSpeedReferenceMmYr ?? 0),
    legend: normalizeHorizontalGlyphLegend(payload.legend ?? {}, ellipseDisplayFactor),
  };
}


function normalizeHorizontalParticleHistory(render = {}) {
  const historySampleIntervalS = clamp(Number(render.historySampleIntervalS ?? 0.05), 1 / 120, 0.25);
  const historySamplesMin = clamp(Math.round(Number(render.historySamplesMin ?? 9)), 2, 65);
  const historySamplesMax = clamp(Math.round(Number(render.historySamplesMax ?? 65)), historySamplesMin, 65);
  const historySamples = clamp(Math.round(Number(render.historySamples ?? 32)), historySamplesMin, historySamplesMax);
  return {
    historySampleIntervalS,
    historySamplesMin,
    historySamplesMax,
    historySamples,
    trailDurationStepS: clamp(Number(render.trailDurationStepS ?? historySampleIntervalS), 1 / 120, 0.25),
    trailDurationMinS: (historySamplesMin - 1) * historySampleIntervalS,
    trailDurationMaxS: (historySamplesMax - 1) * historySampleIntervalS,
    trailDurationS: (historySamples - 1) * historySampleIntervalS,
  };
}

function particleHistorySamplesForDuration(durationS, config = horizontalParticleConfig) {
  const interval = Math.max(1 / 120, Number(config.historySampleIntervalS ?? 0.05));
  const minSamples = Math.max(2, Math.round(Number(config.historySamplesMin ?? 9)));
  const maxSamples = Math.max(minSamples, Math.round(Number(config.historySamplesMax ?? 65)));
  return clamp(Math.round(Number(durationS) / interval) + 1, minSamples, maxSamples);
}

function particleHistoryDurationForSamples(samples, config = horizontalParticleConfig) {
  const interval = Math.max(1 / 120, Number(config.historySampleIntervalS ?? 0.05));
  return Math.max(0, (Math.round(Number(samples)) - 1) * interval);
}

function normalizeHorizontalParticleRuntime(payload = {}, fieldValues, covarianceValues, spawnValues, runtimeRowCount) {
  const grid = payload.grid ?? {};
  const render = payload.render ?? {};
  const width = Math.max(1, Math.round(Number(grid.width ?? 1)));
  const height = Math.max(1, Math.round(Number(grid.height ?? 1)));
  const expectedGridValues = width * height * 4;
  const spawnCount = Math.max(1, Math.round(Number(payload.summary?.spawnCellCount ?? spawnValues.length / 2)));
  if (fieldValues.length !== expectedGridValues) {
    throw new Error(`Horizontal particle field has ${fieldValues.length} Float32 values; expected ${expectedGridValues}.`);
  }
  if (covarianceValues.length !== expectedGridValues) {
    throw new Error(`Horizontal particle covariance has ${covarianceValues.length} Float32 values; expected ${expectedGridValues}.`);
  }
  if (spawnValues.length !== spawnCount * 2) {
    throw new Error(`Horizontal particle spawn domain has ${spawnValues.length} Float32 values; expected ${spawnCount * 2}.`);
  }
  if (!Array.isArray(grid.coordinateOriginLonLat) || !Array.isArray(grid.gridOriginLocalM) ||
      !Array.isArray(grid.gridAxisIM) || !Array.isArray(grid.gridAxisJM)) {
    throw new Error('Horizontal particle field is missing its metric grid transform. Regenerate runtime assets.');
  }
  const history = normalizeHorizontalParticleHistory(render);

  return {
    schema: payload.schema,
    purpose: payload.purpose,
    grid: {
      width,
      height,
      coordinateOriginLonLat: grid.coordinateOriginLonLat.slice(0, 2).map(Number),
      gridOriginLocalM: grid.gridOriginLocalM.slice(0, 2).map(Number),
      gridAxisIM: grid.gridAxisIM.slice(0, 2).map(Number),
      gridAxisJM: grid.gridAxisJM.slice(0, 2).map(Number),
    },
    render: {
      ...render,
      enabled: render.enabled !== false,
      showByDefault: render.showByDefault !== false,
      defaultMode: ['mean', 'montecarlo', 'shimmer'].includes(String(render.defaultMode).toLowerCase())
        ? String(render.defaultMode).toLowerCase()
        : 'mean',
      particleCapacity: clamp(Math.round(Number(render.particleCapacity ?? 12000)), 256, 20000),
      defaultParticleCount: clamp(Math.round(Number(render.defaultParticleCount ?? 5000)), 0, 20000),
      speedMultiplier: clamp(Number(render.speedMultiplier ?? 1.5), 0, 10),
      particleSizeMultiplier: clamp(Number(render.particleSizeMultiplier ?? 1.0), 0.1, 8),
      particleSizeMultiplierMin: clamp(Number(render.particleSizeMultiplierMin ?? 0.5), 0.1, 8),
      particleSizeMultiplierMax: clamp(Number(render.particleSizeMultiplierMax ?? 3.0), 0.1, 8),
      particleSizeMultiplierStep: clamp(Number(render.particleSizeMultiplierStep ?? 0.1), 0.01, 1),
      particleOpacity: clamp(Number(render.particleOpacity ?? 1.0), 0, 1),
      ...history,
      trailPersistence: clamp(Number(render.trailPersistence ?? 0.98), 0.80, 0.999),
      trailPersistenceMin: clamp(Number(render.trailPersistenceMin ?? 0.80), 0.50, 0.999),
      trailPersistenceMax: clamp(Number(render.trailPersistenceMax ?? 0.999), 0.50, 0.999),
      trailPersistenceStep: clamp(Number(render.trailPersistenceStep ?? 0.001), 0.001, 0.05),
      maxTrailScreenJumpPx: Math.max(8, Number(render.maxTrailScreenJumpPx ?? 120)),
      integrationMaxCellFraction: clamp(Number(render.integrationMaxCellFraction ?? 0.25), 0.05, 1.0),
      uncertaintyStrength: clamp(Number(render.uncertaintyStrength ?? 0.5), 0, 2),
      shimmerStrength: clamp(Number(render.shimmerStrength ?? render.uncertaintyStrength ?? 0.5), 0, 2),
      monteCarloStrength: clamp(Number(render.monteCarloStrength ?? render.uncertaintyStrength ?? 0.5), 0, 2),
      samplerMode: String(render.samplerMode ?? 'conservative_v1'),
    },
    fieldMode: String(render.fieldMode ?? 'raw'),
    lscModel: payload.lscModel ?? null,
    legend: normalizeHorizontalParticleLegend(payload.legend ?? {}),
    warnings: Array.isArray(payload.warnings) ? payload.warnings.slice() : [],
    fieldValues,
    covarianceValues,
    spawnValues,
    spawnCount,
    liveRumCount: Number(payload.summary?.liveRumCount ?? 0),
    validFineTexelCount: Number(payload.summary?.validFineTexelCount ?? 0),
    speedP95MmYr: Math.max(1e-9, Number(payload.summary?.speedP95MmYr ?? 1)),
    runtimeRowCount,
  };
}

function activeHorizontalParticleFieldRuntime() {
  return horizontalParticleFieldMode === 'lsc' && horizontalParticleLscRuntime
    ? horizontalParticleLscRuntime
    : horizontalParticleRuntime;
}

function horizontalParticleIsLscMode() {
  return horizontalParticleFieldMode === 'lsc' && Boolean(horizontalParticleLscRuntime);
}

function syncHorizontalParticleFieldRuntime({resetGpuStatus = false} = {}) {
  const active = activeHorizontalParticleFieldRuntime();
  if (!active) return;
  horizontalParticleConfig = active.render;
  horizontalParticleCount = clamp(horizontalParticleCount, 0, horizontalParticleConfig.particleCapacity);
  horizontalParticleHistorySamples = clamp(
    horizontalParticleHistorySamples,
    horizontalParticleConfig.historySamplesMin,
    horizontalParticleConfig.historySamplesMax,
  );
  horizontalParticleTrailDurationSeconds = particleHistoryDurationForSamples(
    horizontalParticleHistorySamples,
    horizontalParticleConfig,
  );
  if (resetGpuStatus) horizontalParticleGpuStatus = null;
}

function lscRangeLabel(runtimeField = horizontalParticleLscRuntime) {
  const east = Number(runtimeField?.lscModel?.east?.effectiveRangeM ?? 0);
  const north = Number(runtimeField?.lscModel?.north?.effectiveRangeM ?? 0);
  const values = [east, north].filter((value) => Number.isFinite(value) && value > 0);
  if (!values.length) return 'fitted local range';
  return `range ${((values.reduce((sum, value) => sum + value, 0) / values.length) / 1000).toFixed(1)} km`;
}

function horizontalParticleUncertaintyForMode(mode = horizontalParticleMode) {
  return mode === 'shimmer' || mode === 'montecarlo'
    ? Number(horizontalParticleUncertaintyStrengths[mode] ?? 0)
    : 0;
}

function updateHorizontalParticleControls() {
  const activeField = activeHorizontalParticleFieldRuntime();
  const masterEnabled = horizontalParticleMasterEnabled();
  const enabled = Boolean(masterEnabled && runtime && activeField && horizontalParticleConfig.enabled);
  const uncertaintyActive = enabled && (horizontalParticleMode === 'shimmer' || horizontalParticleMode === 'montecarlo');
  const sizeMin = Math.min(
    horizontalParticleConfig.particleSizeMultiplierMin ?? 0.5,
    horizontalParticleConfig.particleSizeMultiplierMax ?? 3,
  );
  const sizeMax = Math.max(
    horizontalParticleConfig.particleSizeMultiplierMin ?? 0.5,
    horizontalParticleConfig.particleSizeMultiplierMax ?? 3,
  );
  const persistenceMin = Math.min(
    horizontalParticleConfig.trailPersistenceMin ?? 0.80,
    horizontalParticleConfig.trailPersistenceMax ?? 0.995,
  );
  const persistenceMax = Math.max(
    horizontalParticleConfig.trailPersistenceMin ?? 0.80,
    horizontalParticleConfig.trailPersistenceMax ?? 0.999,
  );
  const durationMin = Number(horizontalParticleConfig.trailDurationMinS ?? particleHistoryDurationForSamples(horizontalParticleConfig.historySamplesMin));
  const durationMax = Number(horizontalParticleConfig.trailDurationMaxS ?? particleHistoryDurationForSamples(horizontalParticleConfig.historySamplesMax));
  const durationStep = Number(horizontalParticleConfig.trailDurationStepS ?? horizontalParticleConfig.historySampleIntervalS ?? 0.05);

  if (horizontalParticlesToggle) {
    horizontalParticlesToggle.disabled = !enabled;
    horizontalParticlesToggle.checked = showHorizontalParticles;
  }
  if (horizontalParticleModeControl) {
    horizontalParticleModeControl.disabled = !enabled;
    horizontalParticleModeControl.value = horizontalParticleMode;
  }
  const lscAvailable = Boolean(horizontalParticleLscRuntime);
  if (horizontalParticleFieldModeBlock) horizontalParticleFieldModeBlock.hidden = !lscAvailable;
  if (horizontalParticleFieldModeControl) {
    horizontalParticleFieldModeControl.disabled = !enabled || !lscAvailable;
    horizontalParticleFieldModeControl.value = horizontalParticleIsLscMode() ? 'lsc' : 'raw';
  }
  if (horizontalParticleFieldNote) {
    horizontalParticleFieldNote.textContent = horizontalParticleIsLscMode()
      ? `LSC signal field · ${lscRangeLabel()} · noise-weighted using formal covariance · support boundary unchanged.`
      : 'Raw estimates exactly as delivered · conservative bilinear V1 motion · strict eight-neighbour emitters.';
  }
  if (horizontalParticleMcCaveat) {
    const showCaveat = horizontalParticleIsLscMode() && horizontalParticleMode === 'montecarlo';
    horizontalParticleMcCaveat.hidden = !showCaveat;
    if (showCaveat) {
      horizontalParticleMcCaveat.textContent = 'MC uncertainty: per-component LSC prediction variance; cross-component covariance not modeled (E/N-aligned axes).';
    }
  }
  if (horizontalParticleSamplerValue) {
    horizontalParticleSamplerValue.textContent = horizontalParticleIsLscMode()
      ? 'LSC fine field · same eight-neighbour coarse emitters'
      : 'Raw motion: conservative bilinear V1 · emitters: eight-neighbour support';
  }
  if (horizontalParticleCountSlider) {
    horizontalParticleCountSlider.disabled = !enabled;
    horizontalParticleCountSlider.min = '250';
    horizontalParticleCountSlider.max = String(horizontalParticleConfig.particleCapacity);
    horizontalParticleCountSlider.step = '250';
    horizontalParticleCountSlider.value = String(horizontalParticleCount);
  }
  if (horizontalParticleCountValue) horizontalParticleCountValue.textContent = horizontalParticleCount.toLocaleString();
  if (horizontalParticleSizeSlider) {
    horizontalParticleSizeSlider.disabled = !enabled;
    horizontalParticleSizeSlider.min = String(sizeMin);
    horizontalParticleSizeSlider.max = String(sizeMax);
    horizontalParticleSizeSlider.step = String(horizontalParticleConfig.particleSizeMultiplierStep ?? 0.1);
    horizontalParticleSizeSlider.value = String(horizontalParticleSizeMultiplier);
  }
  if (horizontalParticleSizeValue) horizontalParticleSizeValue.textContent = `${horizontalParticleSizeMultiplier.toFixed(2)}×`;
  if (horizontalParticleSpeedSlider) {
    horizontalParticleSpeedSlider.disabled = !enabled;
    horizontalParticleSpeedSlider.min = '0.1';
    horizontalParticleSpeedSlider.max = '6';
    horizontalParticleSpeedSlider.step = '0.05';
    horizontalParticleSpeedSlider.value = String(horizontalParticleSpeedMultiplier);
  }
  if (horizontalParticleSpeedValue) horizontalParticleSpeedValue.textContent = `${horizontalParticleSpeedMultiplier.toFixed(2)}×`;
  if (horizontalParticleTrailDurationSlider) {
    horizontalParticleTrailDurationSlider.disabled = !enabled;
    horizontalParticleTrailDurationSlider.min = String(durationMin);
    horizontalParticleTrailDurationSlider.max = String(durationMax);
    horizontalParticleTrailDurationSlider.step = String(durationStep);
    horizontalParticleTrailDurationSlider.value = String(horizontalParticleTrailDurationSeconds);
  }
  if (horizontalParticleTrailDurationValue) {
    horizontalParticleTrailDurationValue.textContent = `${horizontalParticleTrailDurationSeconds.toFixed(1)} s`;
  }
  if (horizontalParticleTrailPersistenceSlider) {
    horizontalParticleTrailPersistenceSlider.disabled = !enabled;
    horizontalParticleTrailPersistenceSlider.min = String(persistenceMin);
    horizontalParticleTrailPersistenceSlider.max = String(persistenceMax);
    horizontalParticleTrailPersistenceSlider.step = String(horizontalParticleConfig.trailPersistenceStep ?? 0.005);
    horizontalParticleTrailPersistenceSlider.value = String(horizontalParticleTrailPersistence);
  }
  if (horizontalParticleTrailPersistenceValue) horizontalParticleTrailPersistenceValue.textContent = horizontalParticleTrailPersistence.toFixed(2);
  if (horizontalParticleOpacitySlider) {
    horizontalParticleOpacitySlider.disabled = !enabled;
    horizontalParticleOpacitySlider.value = String(horizontalParticleOpacity);
  }
  if (horizontalParticleOpacityValue) horizontalParticleOpacityValue.textContent = `${Math.round(horizontalParticleOpacity * 100)}%`;
  if (horizontalParticleUncertaintyLabel) {
    horizontalParticleUncertaintyLabel.textContent = 'Uncertainty strength';
  }
  if (horizontalParticleUncertaintySlider) {
    horizontalParticleUncertaintySlider.disabled = !uncertaintyActive;
    horizontalParticleUncertaintySlider.value = String(horizontalParticleUncertaintyForMode());
  }
  if (horizontalParticleUncertaintyValue) {
    horizontalParticleUncertaintyValue.textContent = uncertaintyActive
      ? horizontalParticleUncertaintyForMode().toFixed(2)
      : 'off';
  }
  if (horizontalParticleDiagnostic) {
    if (!masterEnabled) {
      horizontalParticleDiagnostic.textContent = 'Horizontal particles are off at the section master switch.';
    } else if (!enabled) {
      horizontalParticleDiagnostic.textContent = 'GPU horizontal particles are unavailable for this runtime.';
    } else if (horizontalParticleGpuStatus) {
      const capacity = Number(horizontalParticleGpuStatus.capacity ?? horizontalParticleConfig.particleCapacity ?? 0);
      const active = clamp(horizontalParticleCount, 0, capacity || horizontalParticleCount);
      const atlasWidth = Number(horizontalParticleGpuStatus.historyAtlasWidth ?? 0);
      const atlasHeight = Number(horizontalParticleGpuStatus.historyAtlasHeight ?? 0);
      const atlasMiB = Number(horizontalParticleGpuStatus.historyTextureMiB ?? 0);
      const maxTexture = Number(horizontalParticleGpuStatus.maxTextureSize ?? 0);
      const atlasText = atlasWidth > 0 && atlasHeight > 0
        ? ` · ${active.toLocaleString()} active / ${capacity.toLocaleString()} capacity · atlas ${atlasWidth}×${atlasHeight} RGBA32F (${atlasMiB.toFixed(1)} MiB)`
        : ` · ${active.toLocaleString()} active / ${capacity.toLocaleString()} capacity`;
      const capabilityText = horizontalParticleGpuStatus.float32Renderable
        ? ` · float FBO OK · max texture ${maxTexture.toLocaleString()}`
        : '';
      const historyDuration = Number(horizontalParticleGpuStatus.historyDurationS ?? horizontalParticleTrailDurationSeconds);
      const historySamples = Number(horizontalParticleGpuStatus.historySamples ?? horizontalParticleHistorySamples);
      const maxSubsteps = Number(horizontalParticleGpuStatus.maxIntegrationSubsteps ?? 24);
      const jumpGuard = Number(horizontalParticleGpuStatus.maxTrailScreenJumpPx ?? horizontalParticleConfig.maxTrailScreenJumpPx ?? 120);
      horizontalParticleDiagnostic.textContent =
        `${horizontalParticleGpuStatus.renderer.replaceAll('_', ' ')}${atlasText}${capabilityText} · ` +
        `${historyDuration.toFixed(2)} s / ${historySamples} samples · adaptive substeps 1–${maxSubsteps} · ${jumpGuard.toFixed(0)} px jump guard · ` +
        `${horizontalParticleGpuStatus.fieldCells.toLocaleString()} live field cells · ${horizontalParticleGpuStatus.spawnCells.toLocaleString()} supported emitters · ` +
        `p95 ${horizontalParticleGpuStatus.speedP95MmYr.toFixed(2)} mm/year`;
    } else {
      const fieldDescription = horizontalParticleIsLscMode()
        ? `${activeField.validFineTexelCount || activeField.liveRumCount} LSC fine field texels · ${lscRangeLabel(activeField)}`
        : `${activeField.liveRumCount.toLocaleString()} observed field cells · V1 motion`;
      horizontalParticleDiagnostic.textContent =
        `${fieldDescription} · strict supported emitters · initializing GPU state…`;
    }
  }
  updateFloatingLegendBars();
}

function handleHorizontalParticleStatus(status) {
  horizontalParticleGpuStatus = status;
  updateHorizontalParticleControls();
}

function attachHorizontalGlyphRecords(payload) {
  const liveByRumId = new Map(liveCells.map((cell) => [cell.rumId, cell]));
  const records = Array.isArray(payload?.records) ? payload.records : [];
  const attached = [];
  for (const record of records) {
    const cell = liveByRumId.get(record.rumId);
    if (!cell) continue;
    const glyph = {
      ...record,
      cell,
      glyphZ: cell.displayZ,
      // Batch 1.20: runtime follows the original Jakarta notebook contract:
      // RUM coordinate is the arrow tail/vector origin; confidence ellipse is
      // centred at the scaled vector tip. Older payloads that stored a
      // tail-offset arrow are corrected here without needing an immediate
      // asset rebuild.
      rumCenterLonLat: cellCenterLonLat(cell) ?? record.arrow?.tailLonLat ?? record.ellipse?.centerLonLat,
    };
    cell.horizontalGlyph = glyph;
    attached.push(glyph);
  }
  return attached;
}

function visibleUpReliefHeightM(cell) {
  if (!cell?.isLive || !verticalMasterEnabled() || !uncertaintyReliefEnabled || capAppearance === 'context-map') return 0;
  if (cell.reliefVisualWeight <= reliefVisualFade.minimumRenderWeight) return 0;
  const gain = Number(runtime?.uncertaintyRelief?.geometry?.up_relief_gain ?? 0.75);
  return Math.max(0, cell.reliefAmplitudeM * gain);
}

function updateHorizontalGlyphHeights() {
  const clearance = horizontalGlyphConfig.clearanceAboveCapM;
  for (const glyph of horizontalGlyphRecords) {
    glyph.glyphZ = sceneMode === '2d'
      ? 0
      : glyph.cell.displayZ + visibleUpReliefHeightM(glyph.cell) + clearance;
  }
}

function glyphColorWithOpacity(color) {
  return [color[0], color[1], color[2], Math.round(color[3] * horizontalGlyphOpacity)];
}

function formatGlyphScale(value) {
  return `${Number(value).toFixed(2)}×`;
}

function averageFootprintCenterLonLat(footprintLonLat) {
  if (!Array.isArray(footprintLonLat) || footprintLonLat.length === 0) return null;
  const points = footprintLonLat
    .filter((point, index) => {
      if (!Array.isArray(point) || point.length < 2) return false;
      // RUM polygons normally repeat the first corner as the closing point.
      // Dropping that duplicate keeps the centre unbiased.
      if (index === footprintLonLat.length - 1 && footprintLonLat.length > 1) {
        const first = footprintLonLat[0];
        return Math.abs(Number(point[0]) - Number(first?.[0])) > 1e-12 ||
          Math.abs(Number(point[1]) - Number(first?.[1])) > 1e-12;
      }
      return true;
    })
    .map((point) => [Number(point[0]), Number(point[1])])
    .filter(([lon, lat]) => Number.isFinite(lon) && Number.isFinite(lat));
  if (!points.length) return null;
  return [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
}

function cellCenterLonLat(cell) {
  const lon = Number(cell?.lon);
  const lat = Number(cell?.lat);
  if (Number.isFinite(lon) && Number.isFinite(lat)) return [lon, lat];
  return averageFootprintCenterLonLat(cell?.footprintLonLat);
}

function horizontalGlyphAnchorLonLat(glyph) {
  return glyph?.rumCenterLonLat ?? cellCenterLonLat(glyph?.cell) ?? glyph?.arrow?.tailLonLat ?? glyph?.ellipse?.centerLonLat ?? [0, 0];
}

function offsetLonLatByMeters(anchorLonLat, eastM, northM) {
  const lon0 = Number(anchorLonLat?.[0]);
  const lat0 = Number(anchorLonLat?.[1]);
  const east = Number(eastM);
  const north = Number(northM);
  if (![lon0, lat0, east, north].every(Number.isFinite)) return anchorLonLat;
  const earthRadiusM = 6378137;
  const degPerRad = 180 / Math.PI;
  const cosLat = Math.max(1e-9, Math.cos(lat0 * Math.PI / 180));
  return [
    lon0 + (east / (earthRadiusM * cosLat)) * degPerRad,
    lat0 + (north / earthRadiusM) * degPerRad,
  ];
}

function horizontalGlyphUnitVector(glyph) {
  const east = Number(glyph?.unitEast);
  const north = Number(glyph?.unitNorth);
  const norm = Math.hypot(east, north);
  if (norm > 1e-12) return [east / norm, north / norm];
  const yawRad = Number(glyph?.arrow?.yawDeg ?? glyph?.azimuthDegCcwFromEast ?? 0) * Math.PI / 180;
  return [Math.cos(yawRad), Math.sin(yawRad)];
}

function horizontalGlyphPointFromTail(glyph, distanceM, scale = horizontalGlyphScale) {
  const anchor = horizontalGlyphAnchorLonLat(glyph);
  const [unitEast, unitNorth] = horizontalGlyphUnitVector(glyph);
  const safeDistanceM = Number(distanceM);
  const safeScale = Math.max(0, Number(scale) || 0);
  if (!Number.isFinite(safeDistanceM)) return anchor;
  return offsetLonLatByMeters(anchor, unitEast * safeDistanceM * safeScale, unitNorth * safeDistanceM * safeScale);
}

function horizontalGlyphArrowTipDistanceM(glyph) {
  const explicit = Number(glyph?.arrow?.lengthM);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return Math.max(0, Number(glyph?.arrow?.shaftLengthM ?? 0)) + Math.max(0, Number(glyph?.arrow?.headLengthM ?? 0));
}

function updateHorizontalGlyphControls() {
  const masterEnabled = horizontalGlyphMasterEnabled();
  const enabled = Boolean(masterEnabled && runtime && horizontalGlyphConfig.enabled && horizontalGlyphRecords.length);
  if (horizontalArrowsToggle) {
    horizontalArrowsToggle.disabled = !enabled;
    horizontalArrowsToggle.checked = showHorizontalArrows;
  }
  if (horizontalEllipsesToggle) {
    horizontalEllipsesToggle.disabled = !enabled;
    horizontalEllipsesToggle.checked = showHorizontalEllipses;
  }
  if (horizontalGlyphOpacitySlider) {
    horizontalGlyphOpacitySlider.disabled = !enabled;
    horizontalGlyphOpacitySlider.value = String(horizontalGlyphOpacity);
  }
  if (horizontalGlyphOpacityValue) horizontalGlyphOpacityValue.textContent = `${Math.round(horizontalGlyphOpacity * 100)}%`;
  if (horizontalGlyphScaleSlider) {
    horizontalGlyphScaleSlider.disabled = !enabled;
    horizontalGlyphScaleSlider.value = String(horizontalGlyphScale);
  }
  if (horizontalGlyphScaleValue) horizontalGlyphScaleValue.textContent = formatGlyphScale(horizontalGlyphScale);
  if (horizontalGlyphDiagnostic) {
    horizontalGlyphDiagnostic.textContent = !masterEnabled
      ? 'Horizontal glyphs are off at the section master switch.'
      : enabled
        ? `${horizontalGlyphRecords.length.toLocaleString()} observed glyph pairs · ≥${horizontalGlyphConfig.minimumSpeedMmYr.toFixed(2)} mm/yr and ≥${horizontalGlyphConfig.significanceSigmaMultiplier.toFixed(0)}σ · ellipse = 1σ E/N uncertainty`
        : 'Horizontal glyphs unavailable — no eligible live-RUM records.';
  }
  updateFloatingLegendBars();
}

function featureCellFromInfo(info) {
  const object = info?.object;
  if (!object) return null;
  return object.cell ?? object;
}

function formatSignedMmYr(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  const text = number.toFixed(2).replace('-', '−');
  return `${number > 0 ? '+' : ''}${text} mm/yr`;
}

function formatMm(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return `${number.toFixed(2).replace('-', '−')} mm`;
}

function formatPlainNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return number.toFixed(digits).replace('-', '−');
}

function positionSelectedTooltip(x, y) {
  if (!tooltip) return;
  const margin = 12;
  const offset = 14;
  const width = Math.max(220, tooltip.offsetWidth || 260);
  const height = Math.max(80, tooltip.offsetHeight || 150);
  const left = clamp(Number(x) + offset, margin, window.innerWidth - width - margin);
  const top = clamp(Number(y) + offset, margin, window.innerHeight - height - margin);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function compactHorizontalText(cell) {
  const glyph = cell?.horizontalGlyph;
  if (!glyph) return 'Horizontal: hidden by filter';
  return `Horizontal: ${formatSignedMmYr(glyph.speedMmYr)}`;
}

function expandedHorizontalHtml(cell) {
  const glyph = cell?.horizontalGlyph;
  if (!glyph) {
    return '<div class="popupMuted">Horizontal glyph hidden by the speed/significance filter.</div>';
  }
  return [
    `<div>East: <strong>${formatSignedMmYr(glyph.eastMmYr)}</strong></div>`,
    `<div>North: <strong>${formatSignedMmYr(glyph.northMmYr)}</strong></div>`,
    `<div>Horizontal speed: <strong>${formatSignedMmYr(glyph.speedMmYr)}</strong></div>`,
    `<div>1σ major: <strong>${formatSignedMmYr(glyph.stdMajor1SigmaMmYr)}</strong></div>`,
    Number.isFinite(Number(glyph.directionalSigmaThetaDeg1Sigma))
      ? `<div>σθ: <strong>${formatPlainNumber(glyph.directionalSigmaThetaDeg1Sigma, 1)}°</strong></div>`
      : '',
  ].join('');
}


function epochDateString(index) {
  return String(runtime?.epochAxis?.epochs?.[index] ?? `Epoch ${index + 1}`);
}

function decimalYearFromEpoch(index) {
  const dateText = runtime?.epochAxis?.epochs?.[index];
  const date = new Date(`${dateText}T00:00:00Z`);
  if (!Number.isFinite(date.getTime())) return index / 12;
  const year = date.getUTCFullYear();
  const start = Date.UTC(year, 0, 1);
  const end = Date.UTC(year + 1, 0, 1);
  return year + ((date.getTime() - start) / Math.max(1, end - start));
}

function trendlineSeriesForCell(cell) {
  if (!runtime || !cell || cell.isBlankie || !Number.isFinite(Number(cell.runtimeRowIndex))) return [];
  const row = Math.round(Number(cell.runtimeRowIndex));
  const start = row * runtime.epochCount;
  const values = [];
  for (let i = 0; i < runtime.epochCount; i += 1) {
    const y = Number(runtime.verticalMeasurementMm[start + i]);
    values.push({index: i, x: decimalYearFromEpoch(i), y});
  }
  return values;
}

function finiteTrendlinePoints(series) {
  return series.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

function linearRegression(points) {
  const n = points.length;
  if (n < 2) return null;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const point of points) {
    sx += point.x;
    sy += point.y;
    sxx += point.x * point.x;
    sxy += point.x * point.y;
  }
  const denominator = (n * sxx) - (sx * sx);
  if (Math.abs(denominator) < 1e-9) return null;
  const slope = ((n * sxy) - (sx * sy)) / denominator;
  const intercept = (sy - (slope * sx)) / n;
  return {slope, intercept};
}

function niceTrendStep(span) {
  if (!Number.isFinite(span) || span <= 0) return 1;
  const rough = span / 5;
  const exponent = Math.floor(Math.log10(rough));
  const base = 10 ** exponent;
  const fraction = rough / base;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * base;
}

function niceTrendlineRange(minValue, maxValue) {
  let min = Number(minValue);
  let max = Number(maxValue);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return {min: -10, max: 10, step: 5};
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const span = Math.max(1e-6, max - min);
  const paddedMin = min - (span * 0.12);
  const paddedMax = max + (span * 0.12);
  const step = niceTrendStep(paddedMax - paddedMin);
  return {
    min: Math.floor(paddedMin / step) * step,
    max: Math.ceil(paddedMax / step) * step,
    step,
  };
}

function trendlineProjectYRange() {
  if (trendlineProjectRangeCache) return trendlineProjectRangeCache;
  if (!runtime?.verticalMeasurementMm?.length) {
    trendlineProjectRangeCache = {min: -10, max: 10, step: 5};
    return trendlineProjectRangeCache;
  }
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < runtime.verticalMeasurementMm.length; i += 1) {
    const value = Number(runtime.verticalMeasurementMm[i]);
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  trendlineProjectRangeCache = niceTrendlineRange(min, max);
  return trendlineProjectRangeCache;
}

function trendlineYRangeForSeries(series, fit) {
  if (trendlineAxisMode === 'project') return trendlineProjectYRange();
  if (trendlineAxisMode === 'custom') {
    const customMin = Number(trendlineCustomMin);
    const customMax = Number(trendlineCustomMax);
    if (Number.isFinite(customMin) && Number.isFinite(customMax) && customMax > customMin) {
      return {min: customMin, max: customMax, step: niceTrendStep(customMax - customMin)};
    }
  }

  const points = finiteTrendlinePoints(series);
  let min = Infinity;
  let max = -Infinity;
  for (const point of points) {
    if (point.y < min) min = point.y;
    if (point.y > max) max = point.y;
  }
  if (fit && points.length) {
    const x0 = points[0].x;
    const x1 = points[points.length - 1].x;
    const y0 = fit.intercept + (fit.slope * x0);
    const y1 = fit.intercept + (fit.slope * x1);
    min = Math.min(min, y0, y1);
    max = Math.max(max, y0, y1);
  }
  return niceTrendlineRange(min, max);
}

function updateTrendlineAxisControls() {
  if (rumTrendlineAxisModeSelect) rumTrendlineAxisModeSelect.value = trendlineAxisMode;
  const customEnabled = trendlineAxisMode === 'custom';
  if (rumTrendlineMinInput) {
    rumTrendlineMinInput.disabled = !customEnabled;
    if (Number.isFinite(Number(trendlineCustomMin))) rumTrendlineMinInput.value = String(trendlineCustomMin);
  }
  if (rumTrendlineMaxInput) {
    rumTrendlineMaxInput.disabled = !customEnabled;
    if (Number.isFinite(Number(trendlineCustomMax))) rumTrendlineMaxInput.value = String(trendlineCustomMax);
  }
  if (rumTrendlinePanel) rumTrendlinePanel.classList.toggle('custom-y-axis', customEnabled);
}

function updateTrendlinePanelState() {
  if (!rumTrendlinePanel) return;
  const customEnabled = trendlineAxisMode === 'custom';
  // Trendline is an analysis/readout tool, so it stays available in both 2D
  // and 3D. Only the epoch playback controls remain 3D-only.
  rumTrendlinePanel.classList.remove('scene-disabled');
  rumTrendlinePanel.title = 'Selected RUM trendline';
  rumTrendlinePanel.setAttribute('aria-disabled', 'false');
  updateTrendlineAxisControls();
  if (rumTrendlineAxisModeSelect) rumTrendlineAxisModeSelect.disabled = false;
  if (rumTrendlineMinInput) rumTrendlineMinInput.disabled = !customEnabled;
  if (rumTrendlineMaxInput) rumTrendlineMaxInput.disabled = !customEnabled;
  if (rumTrendlinePngButton) rumTrendlinePngButton.disabled = false;
}

function setTrendlineOpen(open) {
  trendlineOpen = Boolean(open && rumTrendlinePanel && rumTrendlineCanvas && trendlineCell);
  if (!rumTrendlinePanel) return;
  rumTrendlinePanel.hidden = !trendlineOpen;
  rumTrendlinePanel.classList.toggle('open', trendlineOpen);
  if (epochPanel) epochPanel.classList.toggle('trendline-open', trendlineOpen);
  updateTrendlinePanelState();
  if (trendlineOpen) scheduleTrendlineDraw();
}

function openTrendlineForCell(cell) {
  if (!cell || cell.isBlankie) return;
  trendlineCell = cell;
  if (rumTrendlineTitle) rumTrendlineTitle.textContent = `${cell.rumId ?? cell.cellId ?? 'Observed RUM'} trendline`;
  if (rumTrendlineSubtitle) {
    const date = epochDateString(activeEpoch);
    rumTrendlineSubtitle.textContent = `${date} · measurement + fitted vertical trend`;
  }
  setTrendlineOpen(true);
}

function closeTrendline() {
  trendlineOpen = false;
  trendlineCell = null;
  if (rumTrendlinePanel) {
    rumTrendlinePanel.hidden = true;
    rumTrendlinePanel.classList.remove('open');
  }
  if (epochPanel) epochPanel.classList.remove('trendline-open', 'trendline-resizing');
}

function showTrendlineBlankieNoData(cell) {
  if (!trendlineOpen || !rumTrendlinePanel || !rumTrendlineCanvas) return;
  trendlineCell = cell;
  if (rumTrendlineTitle) rumTrendlineTitle.textContent = `${cell?.cellId ?? 'Blankie'} trendline`;
  if (rumTrendlineSubtitle) rumTrendlineSubtitle.textContent = 'No measurement trendline for blankies';
  rumTrendlinePanel.hidden = false;
  rumTrendlinePanel.classList.add('open');
  if (epochPanel) epochPanel.classList.add('trendline-open');
  updateTrendlinePanelState();
  scheduleTrendlineDraw();
}

function drawTrendlineNoData(ctx, width, height, message) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = 'rgba(14, 18, 23, 0.92)';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  ctx.fillStyle = 'rgba(222, 231, 238, 0.78)';
  ctx.font = '700 12px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, width / 2, height / 2);
}

function scheduleTrendlineDraw() {
  if (!trendlineOpen || !rumTrendlineCanvas) return;
  if (trendlineDrawFrame !== null) return;
  trendlineDrawFrame = requestAnimationFrame(() => {
    trendlineDrawFrame = null;
    drawRumTrendline();
  });
}

function trendlineCssPx(name, fallback) {
  const source = epochPanel || rumTrendlinePanel || document.documentElement;
  const value = Number.parseFloat(getComputedStyle(source).getPropertyValue(name));
  return Number.isFinite(value) ? value : fallback;
}

function drawTrendlineCallout(ctx, anchorX, anchorY, text, color, plotLeft, plotRight, plotTop, plotBottom, preferredSide = 1, yOffset = 0) {
  if (!Number.isFinite(anchorX) || !Number.isFinite(anchorY)) return;
  const labelW = Math.max(74, Math.min(148, ctx.measureText(text).width + 14));
  const labelH = 18;
  const side = anchorX + labelW + 18 < plotRight ? 1 : anchorX - labelW - 18 > plotLeft ? -1 : preferredSide;
  let x = side > 0 ? anchorX + 10 : anchorX - labelW - 10;
  let y = anchorY - (labelH / 2) + yOffset;
  x = clamp(x, plotLeft + 3, plotRight - labelW - 3);
  y = clamp(y, plotTop + 3, plotBottom - labelH - 3);
  const leaderEndX = side > 0 ? x : x + labelW;
  const leaderEndY = y + (labelH / 2);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(anchorX, anchorY);
  ctx.lineTo(leaderEndX, leaderEndY);
  ctx.stroke();
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = color;
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, labelW, labelH, 7);
  else ctx.rect(x, y, labelW, labelH);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = '700 9.5px Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + 7, y + (labelH / 2));
  ctx.restore();
}


function drawTrendlineStaticReadout(ctx, plotLeft, plotRight, plotTop, plotBottom, measurementText, trendText, trendSlope) {
  const boxW = 158;
  const boxH = 36;
  const x = Math.max(plotLeft + 6, plotRight - boxW - 8);
  const y = trendSlope > 0
    ? Math.max(plotTop + 6, plotBottom - boxH - 8)
    : Math.min(plotBottom - boxH - 6, plotTop + 8);
  ctx.save();
  ctx.fillStyle = 'rgba(11, 15, 20, 0.72)';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, boxW, boxH, 7);
  else ctx.rect(x, y, boxW, boxH);
  ctx.fill();
  ctx.stroke();
  ctx.font = '800 9.5px Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(255, 130, 105, 0.98)';
  ctx.fillText(`Trend: ${trendText}`, x + 8, y + 12);
  ctx.fillStyle = 'rgba(125, 245, 255, 0.98)';
  ctx.fillText(`Measurement: ${measurementText}`, x + 8, y + 25);
  ctx.restore();
}

function drawRumTrendline() {
  if (!trendlineOpen || !rumTrendlineCanvas || !trendlineCell || !runtime) return;
  const canvas = rumTrendlineCanvas;
  const dpr = Math.max(1, Math.min(2.5, window.devicePixelRatio || 1));
  const cssWidth = Math.max(280, Math.floor(canvas.clientWidth || 640));
  const cssHeight = Math.max(120, Math.floor(canvas.clientHeight || trendlineHeightPx));
  const pixelWidth = Math.round(cssWidth * dpr);
  const pixelHeight = Math.round(cssHeight * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = 'rgba(11, 15, 20, 0.90)';
  ctx.fillRect(0, 0, cssWidth, cssHeight);

  if (trendlineCell?.isBlankie || selectedCellIsBlankie) {
    drawTrendlineNoData(ctx, cssWidth, cssHeight, 'No measurement trendline for blankies');
    if (rumTrendlineSubtitle) rumTrendlineSubtitle.textContent = 'No direct measurement data at this support cell';
    return;
  }

  const series = trendlineSeriesForCell(trendlineCell);
  const points = finiteTrendlinePoints(series);
  if (points.length < 2) {
    drawTrendlineNoData(ctx, cssWidth, cssHeight, 'No measurement trendline available');
    return;
  }
  const fit = linearRegression(points);
  const range = trendlineYRangeForSeries(series, fit);
  const xMin = points[0].x;
  const xMax = points[points.length - 1].x;
  const xSpan = Math.max(1e-9, xMax - xMin);
  const ySpan = Math.max(1e-9, range.max - range.min);

  const margin = {
    left: Math.max(36, trendlineCssPx('--trendline-plot-left', 50)),
    right: Math.max(8, trendlineCssPx('--trendline-plot-right', 14)),
    top: 10,
    bottom: 29,
  };
  // Batch 1.33 manual alignment knobs:
  // These inset the chart X-axis endpoints so the cyan epoch cursor lines up with
  // the epoch slider thumb centers. Increase both values if the chart cursor range
  // is wider than the slider; decrease if it is narrower.
  const xLeftNudge = trendlineCssPx('--trendline-x-left-nudge', 8);
  const xRightNudge = trendlineCssPx('--trendline-x-right-nudge', 8);
  const plotLeft = margin.left + xLeftNudge;
  const plotRight = cssWidth - margin.right - xRightNudge;
  const plotTop = margin.top;
  const plotBottom = cssHeight - margin.bottom;
  const plotW = Math.max(24, plotRight - plotLeft);
  const plotH = Math.max(24, plotBottom - plotTop);
  const xToPx = (x) => plotLeft + ((x - xMin) / xSpan) * plotW;
  const yToPx = (y) => plotTop + ((range.max - y) / ySpan) * plotH;

  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.035)';
  ctx.fillRect(plotLeft, plotTop, plotW, plotH);
  ctx.strokeStyle = 'rgba(255,255,255,0.16)';
  ctx.lineWidth = 1;
  ctx.strokeRect(plotLeft, plotTop, plotW, plotH);

  ctx.font = '10px Arial, sans-serif';
  ctx.fillStyle = 'rgba(219,228,236,0.78)';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  ctx.strokeStyle = 'rgba(255,255,255,0.095)';
  ctx.lineWidth = 1;
  const yTickStart = Math.ceil(range.min / range.step) * range.step;
  for (let y = yTickStart; y <= range.max + (range.step * 0.5); y += range.step) {
    const py = yToPx(y);
    ctx.beginPath();
    ctx.moveTo(plotLeft, py);
    ctx.lineTo(plotRight, py);
    ctx.stroke();
    const label = Math.abs(y) < 1e-9 ? '0' : String(Number(y.toFixed(2))).replace('-', '−');
    ctx.fillText(label, plotLeft - 6, py);
  }

  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  const firstYear = Math.ceil(xMin);
  const lastYear = Math.floor(xMax);
  for (let year = firstYear; year <= lastYear; year += 1) {
    const px = xToPx(year);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.beginPath();
    ctx.moveTo(px, plotTop);
    ctx.lineTo(px, plotBottom + 4);
    ctx.stroke();
    ctx.fillStyle = 'rgba(219,228,236,0.68)';
    ctx.fillText(String(year), px, plotBottom + 8);
  }

  ctx.strokeStyle = 'rgba(89, 169, 255, 0.70)';
  ctx.lineWidth = 1.35;
  ctx.beginPath();
  let started = false;
  for (const point of points) {
    const px = xToPx(point.x);
    const py = yToPx(point.y);
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(116, 190, 255, 0.92)';
  for (const point of points) {
    const px = xToPx(point.x);
    const py = yToPx(point.y);
    ctx.beginPath();
    ctx.arc(px, py, 2.15, 0, Math.PI * 2);
    ctx.fill();
  }

  if (fit) {
    const y0 = fit.intercept + (fit.slope * xMin);
    const y1 = fit.intercept + (fit.slope * xMax);
    ctx.strokeStyle = 'rgba(255, 111, 88, 0.94)';
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    ctx.moveTo(xToPx(xMin), yToPx(y0));
    ctx.lineTo(xToPx(xMax), yToPx(y1));
    ctx.stroke();
  }

  const activePoint = series[clamp(activeEpoch, 0, series.length - 1)];
  if (activePoint && Number.isFinite(activePoint.x)) {
    const cursorX = xToPx(activePoint.x);
    ctx.strokeStyle = 'rgba(125, 245, 255, 0.88)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cursorX, plotTop - 1);
    ctx.lineTo(cursorX, plotBottom + 1);
    ctx.stroke();

    if (Number.isFinite(activePoint.y)) {
      const py = yToPx(activePoint.y);
      ctx.fillStyle = 'rgba(125, 245, 255, 0.95)';
      ctx.beginPath();
      ctx.arc(cursorX, py, 3.4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.save();
  ctx.translate(14, plotTop + (plotH / 2));
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = 'rgba(226, 236, 242, 0.86)';
  ctx.font = '700 10.5px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Vertical displacement [mm]', 0, 0);
  ctx.restore();

  const currentValue = Number(activePoint?.y);
  const currentText = Number.isFinite(currentValue) ? formatMm(currentValue) : '—';
  const trendRateText = fit ? `${fit.slope.toFixed(2).replace('-', '−')} mm/yr` : '—';
  let trendValueText = '—';
  if (activePoint && Number.isFinite(activePoint.x) && fit) {
    const cursorX = xToPx(activePoint.x);
    const trendValue = fit.intercept + (fit.slope * activePoint.x);
    trendValueText = Number.isFinite(trendValue) ? formatMm(trendValue) : '—';
    const trendY = yToPx(trendValue);
    ctx.fillStyle = 'rgba(255, 130, 105, 0.96)';
    ctx.beginPath();
    ctx.arc(cursorX, trendY, 3.0, 0, Math.PI * 2);
    ctx.fill();
  }
  drawTrendlineStaticReadout(
    ctx,
    plotLeft,
    plotRight,
    plotTop,
    plotBottom,
    currentText,
    trendValueText,
    Number(fit?.slope),
  );

  ctx.fillStyle = 'rgba(220, 229, 237, 0.70)';
  ctx.font = '9.5px Arial, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillText(epochDateString(activeEpoch), plotRight, plotBottom - 4);
  ctx.restore();

  if (rumTrendlineSubtitle) {
    rumTrendlineSubtitle.textContent = `${epochDateString(activeEpoch)} · trend ${trendRateText} · measurement ${currentText}`;
  }
}

function exportTrendlinePng() {
  if (!rumTrendlineCanvas || !trendlineOpen) return;
  drawRumTrendline();
  const link = document.createElement('a');
  const id = String(trendlineCell?.rumId ?? trendlineCell?.cellId ?? 'rum').replace(/[^a-z0-9_-]+/gi, '_');
  link.download = `${id}_trendline.png`;
  link.href = rumTrendlineCanvas.toDataURL('image/png');
  link.click();
}

function renderSelectedPopup() {
  if (!tooltip || !selectedCell) return;
  const cell = selectedCell;
  const isBlankie = selectedCellIsBlankie || cell.isBlankie;
  const expanded = selectedTooltipExpanded;
  tooltip.style.display = 'block';
  tooltip.className = `tooltip selectedRumPopup${expanded ? ' expanded' : ''}${isBlankie ? ' blankie' : ' live'}`;

  const toggleLabel = expanded ? 'Less ▴' : 'More ▾';
  if (isBlankie) {
    const neighbourCount = cell.interpolation?.neighbours?.length ?? 0;
    const radius = cell.interpolation?.selectedRadius ?? '—';
    const title = cell.rumId ?? cell.cellId ?? 'Blankie support cell';
    tooltip.innerHTML = `
      <div class="popupTopline">
        <strong>${title}</strong>
        <button type="button" class="popupClose" data-popup-action="close" aria-label="Close selected RUM popup">×</button>
      </div>
      <div class="popupStatus blankieStatus">Blankie / interpolated support</div>
      <div class="popupCompact">
        <div>No direct InSAR measurement data here.</div>
        <div>Interpolated vertical velocity: <strong>${formatSignedMmYr(cell.upMmYr)}</strong></div>
      </div>
      ${expanded ? `
        <div class="popupExpanded">
          <div>Support model displacement: <strong>${formatMm(cell.displacementMm)}</strong></div>
          <div>IDW neighbours: <strong>${neighbourCount}</strong> within <strong>${radius}</strong> cells</div>
          <div>Grid: <strong>i${cell.gridI}, j${cell.gridJ}</strong></div>
          <div>Visual elevation: <strong>${formatPlainNumber(cell.displayZ, 2)} m</strong></div>
          <div class="popupMuted">Blankies are support/interpolation cells only; uncertainty relief is not drawn on them.</div>
        </div>
      ` : ''}
      <div class="popupActions">
        <button type="button" class="popupPolygonButton" data-popup-action="add-polygon">Add polygon</button>
        <button type="button" class="popupTrendlineDisabled" disabled title="Blankies have no direct measurement trendline">No trendline</button>
        <button type="button" data-popup-action="toggle-more">${toggleLabel}</button>
      </div>
    `;
  } else {
    tooltip.innerHTML = `
      <div class="popupTopline">
        <strong>${cell.rumId}</strong>
        <button type="button" class="popupClose" data-popup-action="close" aria-label="Close selected RUM popup">×</button>
      </div>
      <div class="popupStatus liveStatus">Observed RUM</div>
      <div class="popupCompact">
        <div>Vertical: <strong>${formatSignedMmYr(cell.upMmYr)}</strong></div>
        <div>${compactHorizontalText(cell)}</div>
      </div>
      ${expanded ? `
        <div class="popupExpanded">
          <div>Model displacement: <strong>${formatMm(cell.displacementMm)}</strong></div>
          <div>Synthetic measurement: <strong>${formatMm(cell.measurementMm)}</strong></div>
          ${Number.isFinite(cell.sourceSigmaUpMmYr) ? `<div>Rate uncertainty σv: <strong>${formatSignedMmYr(cell.sourceSigmaUpMmYr)}</strong></div>` : ''}
          <div>Position uncertainty σz: <strong>${formatMm(cell.sigmaMm)}</strong> <span class="popupMuted">synthetic demo</span></div>
          <div>Displayed relief range: <strong>${formatReliefRange(cell)}</strong></div>
          ${expandedHorizontalHtml(cell)}
          <div>Grid: <strong>i${cell.gridI}, j${cell.gridJ}</strong></div>
          <div>Visual elevation: <strong>${formatPlainNumber(cell.displayZ, 2)} m</strong></div>
        </div>
      ` : ''}
      <div class="popupActions">
        <button type="button" class="popupPolygonButton" data-popup-action="add-polygon">Add polygon</button>
        <button type="button" class="popupTrendlineButton" data-popup-action="open-trendline">Open trendline</button>
        <button type="button" data-popup-action="toggle-more">${toggleLabel}</button>
      </div>
    `;
  }

  positionSelectedTooltip(selectedTooltipPosition.x, selectedTooltipPosition.y);
}

function clearSelectedFeature() {
  selectedCell = null;
  selectedCellIsBlankie = false;
  selectedTooltipExpanded = false;
  if (tooltip) {
    tooltip.style.display = 'none';
    tooltip.classList.remove('selectedRumPopup', 'expanded', 'blankie', 'live');
  }
  map.getCanvas().style.cursor = '';
  scheduleMiniViewerDraw();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
}

function handleFeatureClick(info, isBlankieHint = false) {
  if (studioPolygons?.isDrawing?.()) {
    return true;
  }
  const cell = featureCellFromInfo(info);
  if (!cell) return false;
  selectedCell = cell;
  selectedCellIsBlankie = Boolean(isBlankieHint || cell.isBlankie);
  selectedTooltipExpanded = false;
  selectedTooltipPosition = {
    x: Number.isFinite(Number(info?.x)) ? Number(info.x) : window.innerWidth * 0.5,
    y: Number.isFinite(Number(info?.y)) ? Number(info.y) : window.innerHeight * 0.5,
  };
  suppressNextMapClickClear = true;
  renderSelectedPopup();
  if (trendlineOpen) {
    if (selectedCellIsBlankie) showTrendlineBlankieNoData(cell);
    else openTrendlineForCell(cell);
  }
  map.getCanvas().style.cursor = 'pointer';
  scheduleMiniViewerDraw();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
  return true;
}


function gridCoordinateKey(point) {
  const lon = Number(point?.[0]);
  const lat = Number(point?.[1]);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  return `${lon.toFixed(9)},${lat.toFixed(9)}`;
}

function gridEdgeKey(a, b) {
  const keyA = gridCoordinateKey(a);
  const keyB = gridCoordinateKey(b);
  if (!keyA || !keyB) return null;
  return keyA < keyB ? `${keyA}|${keyB}` : `${keyB}|${keyA}`;
}

function buildReferenceGridGeometry(cells = structuralCells) {
  const edgeMap = new Map();
  const frameCornerMap = new Map();
  for (const cell of cells) {
    const footprint = Array.isArray(cell?.footprintLonLat) ? cell.footprintLonLat.slice(0, 4) : [];
    if (footprint.length < 4) continue;
    for (let index = 0; index < 4; index += 1) {
      const a = footprint[index];
      const b = footprint[(index + 1) % 4];
      const key = gridEdgeKey(a, b);
      if (!key || edgeMap.has(key)) continue;
      edgeMap.set(key, {
        path: [
          [Number(a[0]), Number(a[1]), 0],
          [Number(b[0]), Number(b[1]), 0],
        ],
      });
    }
    for (const corner of footprint) {
      const lon = Number(corner?.[0]);
      const lat = Number(corner?.[1]);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
      const key = gridCoordinateKey(corner);
      if (!key) continue;
      if (!frameCornerMap.has(key)) frameCornerMap.set(key, {lon, lat, cells: []});
      frameCornerMap.get(key).cells.push(cell);
    }
  }
  referenceGridPlanPaths = [...edgeMap.values()];
  referenceGridFrameCorners = [...frameCornerMap.values()];
}

function setReferenceGridMode(mode) {
  const nextMode = mode === 'plan' || mode === 'frame' ? mode : 'off';
  if (nextMode === referenceGridMode) {
    updateReferenceGridControl();
    return;
  }
  referenceGridMode = nextMode;
  updateReferenceGridControl();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
}

function updateReferenceGridControl() {
  if (!sceneGridModeSelect) return;
  sceneGridModeSelect.value = referenceGridMode;
  sceneGridModeSelect.title = referenceGridMode === 'off'
    ? 'Reference grid hidden'
    : referenceGridMode === 'plan'
      ? 'Reference grid: RUM plan grid at z = 0'
      : 'Reference grid: plan grid at z = 0 plus vertical reference lines to current epoch height';
}

const REFERENCE_GRID_FRAME_RIBBON_HALF_WIDTH_M = 4.0;
const REFERENCE_GRID_FRAME_MIN_ABS_Z_M = 0.05;

function referenceGridCornerDisplayZ(corner) {
  const cells = Array.isArray(corner?.cells) ? corner.cells : [];
  let selectedZ = 0;
  for (const cell of cells) {
    const z = Number(cell?.displayZ) || 0;
    if (Math.abs(z) > Math.abs(selectedZ)) selectedZ = z;
  }
  return Math.abs(selectedZ) >= REFERENCE_GRID_FRAME_MIN_ABS_Z_M ? selectedZ : 0;
}

function referenceGridFrameRibbonData() {
  if (referenceGridMode !== 'frame' || sceneMode !== '3d') return [];
  const ribbons = [];
  for (const corner of referenceGridFrameCorners) {
    const z = referenceGridCornerDisplayZ(corner);
    if (!z) continue;
    ribbons.push({...corner, z, axis: 'east'});
    ribbons.push({...corner, z, axis: 'north'});
  }
  return ribbons;
}

function referenceGridFrameRibbonPolygon(item) {
  const halfWidthM = REFERENCE_GRID_FRAME_RIBBON_HALF_WIDTH_M;
  const anchor = [Number(item?.lon), Number(item?.lat)];
  const z = Number(item?.z) || 0;
  const horizontal = item?.axis === 'east';
  const a = horizontal
    ? offsetLonLatByMeters(anchor, -halfWidthM, 0)
    : offsetLonLatByMeters(anchor, 0, -halfWidthM);
  const b = horizontal
    ? offsetLonLatByMeters(anchor, halfWidthM, 0)
    : offsetLonLatByMeters(anchor, 0, halfWidthM);
  return [
    [a[0], a[1], 0],
    [b[0], b[1], 0],
    [b[0], b[1], z],
    [a[0], a[1], z],
  ];
}

function referenceGridLayers() {
  if (!runtime || referenceGridMode === 'off') return [];
  const frameRibbonData = referenceGridFrameRibbonData();
  const layers = [
    new PathLayer({
      id: 'reference-grid-plan-z0',
      data: referenceGridPlanPaths,
      _full3d: true,
      pickable: false,
      getPath: (item) => item.path,
      getColor: [255, 226, 166, 118],
      getWidth: 1.05,
      widthUnits: 'pixels',
      widthMinPixels: 0.75,
      widthMaxPixels: 2.0,
      capRounded: false,
      jointRounded: false,
      parameters: {depthWriteEnabled: false, depthCompare: 'always'},
    }),
  ];

  if (frameRibbonData.length) {
    layers.push(new SolidPolygonLayer({
      id: 'reference-grid-frame-z-ribbons',
      data: frameRibbonData,
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: referenceGridFrameRibbonPolygon,
      getFillColor: (item) => item.z < 0 ? [255, 198, 80, 118] : [167, 225, 255, 108],
      parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
      updateTriggers: {
        getPolygon: [activeEpoch, verticalExaggeration],
        getFillColor: [activeEpoch, verticalExaggeration],
      },
    }));
  }

  return layers;
}

function selectedCellOutlinePath(cell) {
  const footprint = Array.isArray(cell?.footprintLonLat) ? cell.footprintLonLat : [];
  if (!footprint.length) return [];
  const z = sceneMode === '2d'
    ? 0.08
    : (Number(cell.displayZ) || 0) + Math.max(1.5, Number(runtime?.grid?.rumSizeM ?? 450) * 0.004);
  const path = footprint
    .filter((point) => Array.isArray(point) && point.length >= 2)
    .map(([lon, lat]) => [Number(lon), Number(lat), z])
    .filter(([lon, lat]) => Number.isFinite(lon) && Number.isFinite(lat));
  if (path.length >= 2) {
    const first = path[0];
    const last = path[path.length - 1];
    if (Math.abs(first[0] - last[0]) > 1e-12 || Math.abs(first[1] - last[1]) > 1e-12) {
      path.push([...first]);
    }
  }
  return path;
}

function selectedFeatureOutlineLayers() {
  if (!selectedCell) return [];
  const data = [{cell: selectedCell}];
  const triggers = [selectedCell.cellId ?? selectedCell.rumId, selectedCell.displayZ, sceneMode];
  const common = {
    data,
    pickable: false,
    getPath: (item) => selectedCellOutlinePath(item.cell),
    widthUnits: 'pixels',
    capRounded: true,
    jointRounded: true,
    parameters: {depthWriteEnabled: false, depthCompare: 'always'},
    updateTriggers: {getPath: triggers},
  };
  return [
    new PathLayer({
      id: 'selected-rum-outline-glow',
      ...common,
      getColor: [255, 198, 35, 118],
      getWidth: 12,
      widthMinPixels: 8,
      widthMaxPixels: 18,
    }),
    new PathLayer({
      id: 'selected-rum-outline-core',
      ...common,
      getColor: [255, 235, 94, 255],
      getWidth: 4.2,
      widthMinPixels: 3.2,
      widthMaxPixels: 7,
    }),
  ];
}

async function loadJakartaRuntime() {
  const manifest = await fetchJsonOrThrow(
    runtimeAssetUrl('manifest.json'),
    'Proto1 runtime manifest',
  );

  const [staticPayload, blankPayload, epochAxis, modelBuffer, measurementBuffer, sigmaBuffer, horizontalGlyphPayload, horizontalParticlePayload] = await Promise.all([
    fetchJsonOrThrow(
      requireManifestAsset(manifest, 'staticRums', 'rum_static.json'),
      'Live RUM static data',
    ),
    fetchJsonOrThrow(
      requireManifestAsset(manifest, 'interpolatedBlankies', 'interpolated_blankies.json'),
      'Interpolated support-cell data',
    ),
    fetchJsonOrThrow(
      requireManifestAsset(manifest, 'epochAxis', 'epoch_axis.json'),
      'Epoch axis',
    ),
    fetchArrayBufferOrThrow(
      requireManifestAsset(manifest, 'verticalModelF32', 'vertical_model_mm_f32.bin'),
      'Vertical model Float32 data',
    ),
    fetchArrayBufferOrThrow(
      requireManifestAsset(manifest, 'verticalMeasurementF32', 'vertical_measurement_mm_f32.bin'),
      'Vertical measurement Float32 data',
    ),
    fetchArrayBufferOrThrow(
      requireManifestAsset(manifest, 'verticalSigmaF32', 'vertical_sigma_mm_f32.bin'),
      'Vertical uncertainty Float32 data',
    ),
    fetchJsonOrThrow(
      requireManifestAsset(manifest, 'horizontalGlyphs', 'horizontal_glyphs.json'),
      'Horizontal glyph data',
    ),
    fetchJsonOrThrow(
      requireManifestAsset(manifest, 'horizontalParticleField', 'horizontal_particle_field.json'),
      'Horizontal GPU particle metadata',
    ),
  ]);

  const [horizontalParticleFieldBuffer, horizontalParticleCovarianceBuffer, horizontalParticleSpawnBuffer] = await Promise.all([
    fetchArrayBufferOrThrow(
      requirePayloadAsset(horizontalParticlePayload, 'fieldF32', 'Horizontal GPU particle metadata'),
      'Horizontal GPU particle field Float32 data',
    ),
    fetchArrayBufferOrThrow(
      requirePayloadAsset(horizontalParticlePayload, 'covarianceF32', 'Horizontal GPU particle metadata'),
      'Horizontal GPU particle covariance Float32 data',
    ),
    fetchArrayBufferOrThrow(
      requirePayloadAsset(horizontalParticlePayload, 'spawnGridF32', 'Horizontal GPU particle metadata'),
      'Horizontal GPU particle spawn-domain Float32 data',
    ),
  ]);

  const verticalModelMm = new Float32Array(modelBuffer);
  const verticalMeasurementMm = new Float32Array(measurementBuffer);
  const verticalSigmaMm = new Float32Array(sigmaBuffer);
  trendlineProjectRangeCache = null;
  const horizontalParticleFieldValues = new Float32Array(horizontalParticleFieldBuffer);
  const horizontalParticleCovarianceValues = new Float32Array(horizontalParticleCovarianceBuffer);
  const horizontalParticleSpawnValues = new Float32Array(horizontalParticleSpawnBuffer);
  const expectedCount = manifest.runtimeRowCount * manifest.epochCount;
  for (const [label, values] of [
    ['Vertical model', verticalModelMm],
    ['Vertical measurement', verticalMeasurementMm],
    ['Vertical sigma', verticalSigmaMm],
  ]) {
    if (values.length !== expectedCount) {
      throw new Error(`${label} Float32 has ${values.length} values; expected ${expectedCount}.`);
    }
  }
  if (staticPayload.rums.length !== manifest.rumCount) {
    throw new Error('Live RUM static count does not match manifest.');
  }
  if (blankPayload.blankies.length !== manifest.blankCount) {
    throw new Error('Interpolated blankie count does not match manifest.');
  }
  if (!Array.isArray(horizontalGlyphPayload.records)) {
    throw new Error('Horizontal glyph payload is missing records.');
  }

  horizontalParticleRuntime = normalizeHorizontalParticleRuntime(
    horizontalParticlePayload,
    horizontalParticleFieldValues,
    horizontalParticleCovarianceValues,
    horizontalParticleSpawnValues,
    manifest.runtimeRowCount,
  );

  // Optional LSC assets are deliberately soft-fail: raw estimates are the
  // baseline product and remain fully usable when an older package does not
  // carry LSC or an optional LSC fetch fails.
  horizontalParticleLscRuntime = null;
  const lscAsset = manifest.assets?.horizontalParticleFieldLsc;
  if (typeof lscAsset === 'string' && lscAsset.trim()) {
    try {
      const lscPayload = await fetchJsonOrThrow(runtimeAssetUrl(lscAsset), 'Optional LSC particle metadata');
      const [lscFieldBuffer, lscCovarianceBuffer, lscSpawnBuffer] = await Promise.all([
        fetchArrayBufferOrThrow(
          requirePayloadAsset(lscPayload, 'fieldF32', 'Optional LSC particle metadata'),
          'Optional LSC particle field Float32 data',
        ),
        fetchArrayBufferOrThrow(
          requirePayloadAsset(lscPayload, 'covarianceF32', 'Optional LSC particle metadata'),
          'Optional LSC particle covariance Float32 data',
        ),
        fetchArrayBufferOrThrow(
          requirePayloadAsset(lscPayload, 'spawnGridF32', 'Optional LSC particle metadata'),
          'Optional LSC particle spawn-domain Float32 data',
        ),
      ]);
      horizontalParticleLscRuntime = normalizeHorizontalParticleRuntime(
        lscPayload,
        new Float32Array(lscFieldBuffer),
        new Float32Array(lscCovarianceBuffer),
        new Float32Array(lscSpawnBuffer),
        manifest.runtimeRowCount,
      );
      console.info('[Proto1 DeckGL] optional LSC particle field ready', {
        grid: horizontalParticleLscRuntime.grid,
        lsc: horizontalParticleLscRuntime.lscModel,
      });
    } catch (error) {
      console.warn('[Proto1 DeckGL] Optional LSC field unavailable; staying raw-only.', error);
      horizontalParticleLscRuntime = null;
    }
  }

  const viewer = manifest.viewer;
  twoDAnalysisConfig = normalizedTwoDAnalysisConfig(viewer.two_d_analysis ?? {});
  const pitMode = viewer.pit_mode;
  const exag = viewer.vertical_exaggeration;

  runtime = {
    manifest,
    epochAxis,
    verticalModelMm,
    verticalMeasurementMm,
    verticalSigmaMm,
    verticalMeasurement: manifest.verticalMeasurement ?? {},
    verticalSigma: manifest.verticalSigma ?? {},
    verticalUncertaintyLegend: normalizeVerticalUncertaintyLegend(
      manifest.verticalUncertaintyLegend ?? {},
      staticPayload.rums,
      verticalSigmaMm,
      manifest.epochCount,
    ),
    uncertaintyRelief: manifest.uncertaintyRelief ?? {
      displayRange: {unit: 'sigma', value: 1},
      geometry: {enabled: true, grid_n_per_rum: 4},
    },
    epochCount: manifest.epochCount,
    rumCount: manifest.rumCount,
    blankCount: manifest.blankCount,
    runtimeRowCount: manifest.runtimeRowCount,
    structuralCellCount: manifest.structuralCellCount,
    unfilledNoDataCellCount: manifest.unfilledNoDataCellCount ?? 0,
    blankieSelection: manifest.blankieSelection ?? blankPayload.selection ?? {},
    grid: staticPayload.grid,
    domainBounds: manifest.pitDomain,
    verticalVelocityColorScale: normalizeVerticalVelocityColorScale(
      manifest.verticalVelocityColorScale ?? manifest.rateColorSeed ?? {},
    ),
    playbackEpochsPerSecond: viewer.playback_epochs_per_second ?? 8,
    verticalExaggeration: {
      defaultMPerMm: Number(exag.default_m_per_mm ?? 10),
      minMPerMm: Number(exag.min_m_per_mm ?? 0),
      maxMPerMm: Number(exag.max_m_per_mm ?? 20),
    },
    playbackSpeed: {
      defaultMultiplier: Number(viewer.playback_speed?.default_multiplier ?? 1),
      minMultiplier: Number(viewer.playback_speed?.min_multiplier ?? 0.25),
      maxMultiplier: Number(viewer.playback_speed?.max_multiplier ?? 4),
      stepMultiplier: Number(viewer.playback_speed?.step_multiplier ?? 0.25),
    },
    maxCameraPitch: pitMode.max_camera_pitch ?? 80,
    horizontalGlyphPayload,
    horizontalParticlePayload,
    horizontalParticle: horizontalParticleRuntime,
    horizontalParticleLsc: horizontalParticleLscRuntime,
  };

  const cameraDepthTest = viewer.camera_depth_test ?? {};
  farZMultiplier = Math.max(1.01, Number(cameraDepthTest.far_z_multiplier ?? FAR_Z_MULTIPLIER_FALLBACK));
  nearZMultiplier = clamp(
    Number(cameraDepthTest.near_z_multiplier ?? NEAR_Z_MULTIPLIER_FALLBACK),
    0.01,
    1.0,
  );
  grazingFarZMultiplier = Math.max(
    farZMultiplier,
    Number(cameraDepthTest.grazing_far_z_multiplier ?? GRAZING_FAR_Z_MULTIPLIER_FALLBACK),
  );
  grazingPitchStartDeg = clamp(
    Number(cameraDepthTest.grazing_pitch_start_deg ?? GRAZING_PITCH_START_FALLBACK),
    0,
    Math.max(0, runtime.maxCameraPitch - 1),
  );
  activeFarZMultiplier = farZMultiplier;
  depthOccludersEnabled = cameraDepthTest.depth_occluders_enabled ?? true;
  const contextCap = viewer.context_cap ?? {};
  const contextAtlasLod = contextCap.atlas_lod ?? {};
  const overviewAtlasZoom = clamp(Math.round(Number(contextCap.atlas_zoom ?? 13)), 10, 15);
  const focusEnterMapZoom = clamp(Number(contextAtlasLod.focus_enter_map_zoom ?? 13.0), 0, 22);
  const overviewReturnMapZoom = clamp(
    Number(contextAtlasLod.overview_return_map_zoom ?? 12.6),
    0,
    focusEnterMapZoom,
  );
  contextCapConfig = {
    atlasZoom: overviewAtlasZoom,
    atlasMaxDimension: clamp(Math.round(Number(contextCap.atlas_max_dimension ?? 4096)), 512, 8192),
    atlasPaddingFraction: clamp(Number(contextCap.atlas_padding_fraction ?? 0.025), 0, 0.2),
    atlasMipmaps: contextCap.atlas_mipmaps !== false,
    atlasMaxAnisotropy: clamp(Math.round(Number(contextCap.atlas_max_anisotropy ?? 4)), 1, 16),
    liveTintAlpha: clamp(Math.round(Number(contextCap.live_tint_alpha ?? CONTEXT_LIVE_TINT_ALPHA)), 0, 220),
    blankieTintColor: Array.isArray(contextCap.blankie_tint_rgba) && contextCap.blankie_tint_rgba.length >= 4
      ? contextCap.blankie_tint_rgba.slice(0, 4).map((value) => clamp(Math.round(Number(value)), 0, 255))
      : CONTEXT_BLANKIE_TINT_COLOR,
    atlasLod: {
      enabled: contextAtlasLod.enabled !== false,
      focusAtlasZoom: clamp(
        Math.round(Number(contextAtlasLod.focus_atlas_zoom ?? overviewAtlasZoom + 1)),
        overviewAtlasZoom + 1,
        16,
      ),
      focusAtlasMaxDimension: clamp(
        Math.round(Number(contextAtlasLod.focus_atlas_max_dimension ?? 6144)),
        512,
        8192,
      ),
      focusAtlasMaxTiles: clamp(
        Math.round(Number(contextAtlasLod.focus_atlas_max_tiles ?? 384)),
        16,
        2048,
      ),
      focusEnterMapZoom,
      overviewReturnMapZoom,
    },
  };
  contextAtlases = {
    overview: {state: 'idle', atlas: null, error: null, progress: null},
    focus: {state: 'idle', atlas: null, error: null, progress: null},
  };
  activeContextAtlasKey = 'overview';
  contextAtlas = null;
  contextAtlasState = 'idle';
  contextStudyBounds = null;
  contextCapMesh = null;

  // The invisible datum apron is a real z=0 ground plane. Looking along it at
  // 80° is physically unreadable and causes intense overdraw, so operational
  // pitch is deliberately capped by config instead of hiding the problem.
  map.setMaxPitch(runtime.maxCameraPitch);
  if (map.getPitch() > runtime.maxCameraPitch) map.setPitch(runtime.maxCameraPitch);
  syncCameraDepthContract({force: true});

  const reliefGeometry = runtime.uncertaintyRelief?.geometry ?? {};
  reliefVisualFade = normalizeReliefVisualFade(runtime.uncertaintyRelief?.visualFade ?? {});
  reliefLod = normalizeReliefLod(reliefGeometry.lod ?? {}, reliefGeometry);
  reliefMeshSets = createReliefMeshSets(reliefGeometry, reliefVisualFade, reliefLod);
  activeReliefLodKey = reliefLod.enabled ? 'far' : 'near';
  reliefMesh = activeReliefMesh() ?? createCheckerboardReliefMesh(reliefGeometry);
  uncertaintyReliefEnabled = reliefGeometry.enabled ?? true;
  updateCapAppearanceControls();

  verticalExaggeration = exag.default_m_per_mm;
  apronMode = pitMode.default_apron_mode ?? 'see-through';
  datumLineEnabled = Boolean(pitMode.show_datum_outline);

  verticalExagSlider.min = String(exag.min_m_per_mm);
  verticalExagSlider.max = String(exag.max_m_per_mm);
  verticalExagSlider.step = String(exag.step_m_per_mm);
  verticalExagSlider.value = String(verticalExaggeration);
  verticalExagValue.textContent = formatExaggeration(verticalExaggeration);
  datumLineToggle.checked = datumLineEnabled;
  depthOccluderToggle.checked = depthOccludersEnabled;

  liveCells = createLiveCells(staticPayload.rums, runtime.verticalVelocityColorScale);
  blankieCells = createBlankieCells(blankPayload.blankies);
  horizontalGlyphConfig = normalizeHorizontalGlyphConfig(horizontalGlyphPayload);
  horizontalGlyphMeshes = {
    shaft: createArrowShaftMesh(),
    head: createArrowHeadMesh(),
    ellipse: createConfidenceEllipseMesh({
      segments: horizontalGlyphConfig.ellipseSegments,
      innerRadius: horizontalGlyphConfig.ellipseRingInnerRadius,
    }),
  };
  showHorizontalArrows = horizontalGlyphConfig.showArrowsByDefault;
  showHorizontalEllipses = horizontalGlyphConfig.showEllipsesByDefault;
  horizontalGlyphOpacity = horizontalGlyphConfig.defaultOpacity;
  horizontalGlyphScale = 1.0;
  horizontalGlyphRecords = attachHorizontalGlyphRecords(horizontalGlyphPayload);
  horizontalParticleFieldMode = 'raw';
  horizontalParticleConfig = horizontalParticleRuntime.render;
  showHorizontalParticles = horizontalParticleConfig.enabled && horizontalParticleConfig.showByDefault;
  horizontalParticleMode = horizontalParticleConfig.defaultMode;
  horizontalParticleCount = clamp(
    horizontalParticleConfig.defaultParticleCount,
    0,
    horizontalParticleConfig.particleCapacity,
  );
  horizontalParticleSpeedMultiplier = horizontalParticleConfig.speedMultiplier;
  horizontalParticleSizeMultiplier = horizontalParticleConfig.particleSizeMultiplier;
  horizontalParticleOpacity = horizontalParticleConfig.particleOpacity;
  horizontalParticleTrailPersistence = horizontalParticleConfig.trailPersistence;
  horizontalParticleHistorySamples = horizontalParticleConfig.historySamples;
  horizontalParticleTrailDurationSeconds = particleHistoryDurationForSamples(horizontalParticleHistorySamples, horizontalParticleConfig);
  horizontalParticleUncertaintyStrengths = {
    shimmer: horizontalParticleConfig.shimmerStrength,
    montecarlo: horizontalParticleConfig.monteCarloStrength,
  };
  horizontalParticleUncertaintyStrength = horizontalParticleUncertaintyForMode(horizontalParticleMode);
  horizontalParticleGpuStatus = null;
  structuralCells = [...liveCells, ...blankieCells];
  cellsByKey = new Map(structuralCells.map((cell) => [cellKey(cell.gridI, cell.gridJ), cell]));
  buildReferenceGridGeometry(structuralCells);
  updateMiniViewerBounds();
  scheduleMiniViewerDraw();

  topology = buildTopology(structuralCells);
  const supportKeys = new Set(
    structuralCells.map((cell) => cellKey(cell.gridI, cell.gridJ)),
  );
  const projectToWgs84 = (x, y) => proj4(staticPayload.sourceProj4, 'WGS84', [x, y]);
  datumGround = buildDatumGround({
    domainBounds: runtime.domainBounds,
    grid: runtime.grid,
    supportKeys,
    marginCells: pitMode.apron_margin_cells,
    projectToWgs84,
  });

  epochSlider.min = '0';
  epochSlider.max = String(runtime.epochCount - 1);
  activeEpoch = runtime.epochCount - 1;
  epochSlider.value = String(activeEpoch);
  epochSlider.disabled = false;
  verticalExagSlider.disabled = false;
  playButton.disabled = false;
  const playbackSpeed = runtime.playbackSpeed;
  playbackSpeedMultiplier = clamp(
    playbackSpeed.defaultMultiplier,
    playbackSpeed.minMultiplier,
    playbackSpeed.maxMultiplier,
  );
  if (playbackSpeedSlider) {
    playbackSpeedSlider.min = String(playbackSpeed.minMultiplier);
    playbackSpeedSlider.max = String(playbackSpeed.maxMultiplier);
    playbackSpeedSlider.step = String(playbackSpeed.stepMultiplier);
    playbackSpeedSlider.value = String(playbackSpeedMultiplier);
    playbackSpeedSlider.disabled = false;
  }
  if (playbackSpeedValue) playbackSpeedValue.textContent = formatPlaybackSpeed(playbackSpeedMultiplier);
  updateHorizontalGlyphControls();
  updateHorizontalParticleControls();
  sceneMode = '3d';
  updateSceneModeUi();

  for (const button of apronModeControl.querySelectorAll('.seg')) {
    button.classList.toggle('active', button.dataset.mode === apronMode);
  }

  updateBasemapControls();
  updateCapAppearanceControls();
  updateContextCapDiagnostic();

  focusLabel.textContent =
    `${runtime.rumCount.toLocaleString()} live RUMs + ${runtime.blankCount.toLocaleString()} envelope blankies · ` +
    `${runtime.unfilledNoDataCellCount.toLocaleString()} datum/no-data cells`;

  const bounds = new maplibregl.LngLatBounds();
  for (const corner of datumGround.outerRing.slice(0, 4)) bounds.extend(corner);
  map.fitBounds(bounds, {
    padding: {top: 90, right: 90, bottom: 90, left: 380},
    maxZoom: 11.8,
    duration: 0,
  });
  map.setBearing(-25);
  map.setPitch(62);
  savedThreeDCamera = captureCameraState();
  // Choose the relief mesh once the camera has settled. Camera movement itself
  // keeps using DeckGL's normal transform path; only a LOD boundary rebuilds
  // layers, never the mesh geometry or per-epoch arrays.
  syncCameraDepthContract({force: true});
  syncReliefLod({redraw: false});

  applyEpoch();
  void prepareContextAtlas();

  console.log('[Proto1 DeckGL] synthetic sigma data pass ready', {
    liveRums: runtime.rumCount,
    interpolatedBlankies: runtime.blankCount,
    openNoDataCells: runtime.unfilledNoDataCellCount,
    runtimeRows: runtime.runtimeRowCount,
    epochs: runtime.epochCount,
    farZMultiplier,
    activeFarZMultiplier,
    maxCameraPitch: runtime.maxCameraPitch,
    grazingPitchStartDeg,
    grazingFarZMultiplier,
    measurementNoise: runtime.verticalMeasurement.noiseLevel,
    uncertaintyQuality: runtime.verticalSigma.qualityPreset,
    reliefRange: runtime.uncertaintyRelief.displayRange,
    reliefMesh: reliefMesh ? {gridN: reliefMesh.gridN, trianglesPerRum: reliefMesh.triangleCount} : null,
    verticalVelocityColorScale: {
      mode: runtime.verticalVelocityColorScale.mode,
      tauMmYr: runtime.verticalVelocityColorScale.nearZeroThresholdMmYr,
      subsidenceLimitMmYr: runtime.verticalVelocityColorScale.subsidenceLimitMmYr,
      upliftLimitMmYr: runtime.verticalVelocityColorScale.upliftLimitMmYr,
    },
    horizontalGlyphs: {
      visiblePairs: horizontalGlyphRecords.length,
      arrowScaleMPerMmYr: horizontalGlyphConfig.arrowScaleMPerMmYr,
      speedReferenceMmYr: horizontalGlyphConfig.arrowSpeedReferenceMmYr,
    },
    horizontalParticles: {
      mode: horizontalParticleMode,
      defaultCount: horizontalParticleCount,
      capacity: horizontalParticleConfig.particleCapacity,
      speedP95MmYr: horizontalParticleRuntime.speedP95MmYr,
      spawnCells: horizontalParticleRuntime.spawnCount,
      lscAvailable: Boolean(horizontalParticleLscRuntime),
      lscFineTexels: horizontalParticleLscRuntime?.validFineTexelCount ?? 0,
    },
  });
}

function applyEpoch() {
  if (!runtime) return;

  if (sceneMode === '2d') {
    updateHorizontalGlyphHeights();
    updateReadingNote();
    updateEpochNavigationControls();
    scheduleTrendlineDraw();
    deckOverlay.setProps({layers: makeLayers()});
    return;
  }

  const date = runtime.epochAxis.epochs[activeEpoch];

  for (const cell of structuralCells) {
    const valueIndex = cell.runtimeRowIndex * runtime.epochCount + activeEpoch;
    cell.displacementMm = runtime.verticalModelMm[valueIndex];
    cell.measurementMm = runtime.verticalMeasurementMm[valueIndex];
    cell.sigmaMm = runtime.verticalSigmaMm[valueIndex];
    // V7.2 convention: 1 mm × 10× = 10 visual metres at the default setting.
    cell.displayZ = cell.displacementMm * verticalExaggeration;
    cell.capPolygon3d = cell.footprintLonLat
      .slice(0, 4)
      .map(([lon, lat]) => [lon, lat, cell.displayZ]);
    cell.reliefPosition = [
      cell.reliefPosition[0],
      cell.reliefPosition[1],
      cell.displayZ,
    ];
    cell.reliefAmplitudeM = Math.max(0, reliefRangeMm(cell) * verticalExaggeration);
    cell.reliefVisualWeight = cell.isLive
      ? reliefVisualWeightForAmplitude(cell.reliefAmplitudeM)
      : 0;
    cell.reliefVisualBucket = -1;
  }

  rebuildLiveReliefBuckets();
  updateHorizontalGlyphHeights();

  activeLiveWalls = [];
  activeBlankieWalls = [];
  for (const wall of topology.neighbourWalls) {
    const cellA = cellsByKey.get(wall.cellKeyA);
    const cellB = cellsByKey.get(wall.cellKeyB);
    const lower = Math.min(cellA.displayZ, cellB.displayZ);
    const upper = Math.max(cellA.displayZ, cellB.displayZ);
    if (upper - lower <= WALL_EPSILON_M) continue;

    const record = buildWallRecord(cellA, cellB, wall.edgeLonLat, lower, upper);
    if (record.isSupportWall) activeBlankieWalls.push(record);
    else activeLiveWalls.push(record);
  }

  activeLiveRimWalls = [];
  activeBlankieRimWalls = [];
  for (const edge of topology.rimEdges) {
    const cell = cellsByKey.get(edge.cellKey);
    if (Math.abs(cell.displayZ) <= RIM_EPSILON_M) continue;

    const lower = Math.min(0, cell.displayZ);
    const upper = Math.max(0, cell.displayZ);
    const record = {
      polygon3d: [
        [edge.edgeLonLat[0][0], edge.edgeLonLat[0][1], lower],
        [edge.edgeLonLat[1][0], edge.edgeLonLat[1][1], lower],
        [edge.edgeLonLat[1][0], edge.edgeLonLat[1][1], upper],
        [edge.edgeLonLat[0][0], edge.edgeLonLat[0][1], upper],
      ],
      fillColor: cell.isBlankie ? BLANKIE_WALL_COLOR : APRON_INNER_LIP,
    };

    if (cell.isBlankie) activeBlankieRimWalls.push(record);
    else activeLiveRimWalls.push(record);
  }

  epochLabel.textContent = `${date} · epoch ${activeEpoch + 1}/${runtime.epochCount}`;
  updateReadingNote();
  updateEpochNavigationControls();
  scheduleTrendlineDraw();
  deckOverlay.setProps({layers: makeLayers()});
}

function twoDVelocityFillColor(cell) {
  const opacity = clamp(Math.round(twoDAnalysisConfig.rumFillOpacity * 255), 0, 255);
  return [cell.fillColor[0], cell.fillColor[1], cell.fillColor[2], opacity];
}

function makeTwoDAnalysisLayers() {
  const horizontalGlyphsActive = horizontalGlyphMasterEnabled() && horizontalGlyphConfig.enabled && horizontalGlyphRecords.length > 0;
  const arrowGlyphData = horizontalGlyphsActive && showHorizontalArrows
    ? horizontalGlyphRecords.filter((glyph) => glyph.arrow?.shaftLengthM > 0 && glyph.arrow?.headLengthM > 0)
    : [];
  const ellipseGlyphData = horizontalGlyphsActive && showHorizontalEllipses
    ? horizontalGlyphRecords.filter((glyph) => glyph.ellipse?.majorAxisM > 0 && glyph.ellipse?.minorAxisM > 0)
    : [];
  const horizontalGlyphUpdateTriggers = {
    getPosition: ['2d', horizontalGlyphScale],
    getScale: [horizontalGlyphScale],
    getColor: [horizontalGlyphOpacity],
  };
  const field = activeHorizontalParticleFieldRuntime();

  return [
    new SolidPolygonLayer({
      id: 'two-d-static-vertical-rate-rums',
      data: liveCells,
      pickable: true,
      filled: true,
      extruded: false,
      getPolygon: (cell) => cell.footprintLonLat,
      getFillColor: twoDVelocityFillColor,
      stroked: twoDAnalysisConfig.rumOutlineWidthPixels > 0 && twoDAnalysisConfig.rumOutlineRgba[3] > 0,
      getLineColor: () => twoDAnalysisConfig.rumOutlineRgba,
      getLineWidth: () => twoDAnalysisConfig.rumOutlineWidthPixels,
      lineWidthUnits: 'pixels',
      lineWidthMinPixels: twoDAnalysisConfig.rumOutlineWidthPixels,
      parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
      updateTriggers: {
        getFillColor: [twoDAnalysisConfig.rumFillOpacity],
        getLineColor: [twoDAnalysisConfig.rumOutlineRgba],
        getLineWidth: [twoDAnalysisConfig.rumOutlineWidthPixels],
      },
      onClick: (info) => handleFeatureClick(info, false),
    }),

    ...referenceGridLayers(),

    ...(horizontalParticleMasterEnabled() && showHorizontalParticles && field ? [
      new HorizontalParticleLayer({
        id: 'horizontal-gpu-particles',
        field,
        verticalModelMm: runtime.verticalModelMm,
        epochCount: runtime.epochCount,
        activeEpoch,
        verticalExaggeration: 0,
        flatMode: true,
        flatSurfaceZM: twoDAnalysisConfig.flatParticleZM,
        coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
        coordinateOrigin: [field.grid.coordinateOriginLonLat[0], field.grid.coordinateOriginLonLat[1], 0],
        particleCapacity: horizontalParticleConfig.particleCapacity,
        particleCount: horizontalParticleCount,
        mode: horizontalParticleMode,
        speedMultiplier: horizontalParticleSpeedMultiplier,
        particleSizeMultiplier: horizontalParticleSizeMultiplier,
        particleOpacity: horizontalParticleOpacity,
        trailPersistence: horizontalParticleTrailPersistence,
        historySamples: horizontalParticleHistorySamples,
        uncertaintyStrength: horizontalParticleUncertaintyForMode(horizontalParticleMode),
        pickable: false,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
        onStatus: handleHorizontalParticleStatus,
      }),
    ] : []),

    ...(arrowGlyphData.length && horizontalGlyphMeshes.shaft ? [
      new SimpleMeshLayer({
        id: 'two-d-horizontal-arrow-shafts',
        data: arrowGlyphData,
        mesh: horizontalGlyphMeshes.shaft,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphAnchorLonLat(glyph);
          return [lon, lat, 0];
        },
        getOrientation: (glyph) => [0, glyph.arrow.yawDeg, 0],
        getScale: (glyph) => [glyph.arrow.shaftLengthM * horizontalGlyphScale, glyph.arrow.shaftHalfWidthM * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.arrowColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({...info, object: info.object?.cell}, false),
      }),
      new SimpleMeshLayer({
        id: 'two-d-horizontal-arrow-heads',
        data: arrowGlyphData,
        mesh: horizontalGlyphMeshes.head,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphPointFromTail(glyph, glyph.arrow.shaftLengthM);
          return [lon, lat, 0];
        },
        getOrientation: (glyph) => [0, glyph.arrow.yawDeg, 0],
        getScale: (glyph) => [glyph.arrow.headLengthM * horizontalGlyphScale, glyph.arrow.headHalfWidthM * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.arrowColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({...info, object: info.object?.cell}, false),
      }),
    ] : []),

    ...(ellipseGlyphData.length && horizontalGlyphMeshes.ellipse ? [
      new SimpleMeshLayer({
        id: 'two-d-horizontal-confidence-ellipses',
        data: ellipseGlyphData,
        mesh: horizontalGlyphMeshes.ellipse,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphPointFromTail(glyph, horizontalGlyphArrowTipDistanceM(glyph));
          return [lon, lat, 0];
        },
        getOrientation: (glyph) => [0, glyph.ellipse.yawDeg, 0],
        getScale: (glyph) => [glyph.ellipse.majorAxisM * horizontalGlyphConfig.ellipseDisplayFactor * horizontalGlyphScale, glyph.ellipse.minorAxisM * horizontalGlyphConfig.ellipseDisplayFactor * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.ellipseColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({...info, object: info.object?.cell}, false),
      }),
    ] : []),

    ...(studioPolygons?.getLayers?.({sceneMode, verticalExaggeration}) ?? []),

    ...selectedFeatureOutlineLayers(),
  ];
}

function makeLayers() {
  if (!runtime) return [];
  if (sceneMode === '2d') return makeTwoDAnalysisLayers();

  const showApron = apronMode !== 'off';
  const seeThrough = apronMode === 'see-through';
  const datumData = depthOccludersEnabled && showApron
    ? [
      ...datumGround.outerBands.map((polygon) => ({
        polygon,
        kind: 'outer_margin',
        fillColor: seeThrough ? [0, 0, 0, 0] : APRON_COLOR,
      })),
      ...datumGround.datumCells.map((cell) => ({
        polygon: cell.polygon,
        kind: 'datum_no_data',
        fillColor: seeThrough ? [0, 0, 0, 0] : APRON_COLOR,
      })),
    ]
    : [];

  const transparentDepthParameters = {
    cullMode: 'none',
    depthWriteEnabled: true,
    depthCompare: 'less-equal',
  };

  const transparentVisualParameters = {
    cullMode: 'none',
    depthWriteEnabled: false,
    depthCompare: 'less-equal',
  };

  const apronParameters = seeThrough
    ? transparentDepthParameters
    : {cullMode: 'none'};

  const verticalComponentsActive = verticalMasterEnabled();
  const pistonWallsVisible = verticalComponentsActive && showPistonWalls;
  const blankieCapsVisible = verticalComponentsActive && showBlankieCaps;
  const contextCapActive = capAppearance === 'context-map' && contextAtlasState === 'ready' && contextCapMesh;
  const blankieContextActive = blankieCapsVisible && contextAtlasState === 'ready' && contextCapMesh;

  // Blankies remain moving support, never uncertainty relief. When their
  // contextual texture is ready, that opaque texture writes the cap depth
  // itself; avoid an almost-coincident SolidPolygon depth pass that could
  // interfere with the atlas mesh at close inspection.
  const blankieFlatDepthData = depthOccludersEnabled && blankieCapsVisible && !blankieContextActive ? blankieCells : [];
  const blankieWallDepthData = depthOccludersEnabled && pistonWallsVisible ? activeBlankieWalls : [];
  const blankieRimDepthData = depthOccludersEnabled && pistonWallsVisible ? activeBlankieRimWalls : [];

  // In normal scientific mode, relief owns active measured caps and the flat
  // cap remains only for zero/near-zero relief. Context-map mode replaces all
  // live cap geometry with the study-area texture mesh and colour veil.
  const reliefEnabledForLayers = verticalComponentsActive && uncertaintyReliefEnabled;
  const liveInactiveReliefData = contextCapActive
    ? []
    : (reliefEnabledForLayers
      ? liveCells.filter((cell) => cell.reliefVisualWeight <= reliefVisualFade.minimumRenderWeight)
      : liveCells);
  const liveFlatCapData = liveInactiveReliefData;
  const blankieFlatCapData = blankieCapsVisible && !blankieContextActive ? blankieCells : [];
  const liveReliefBuckets = (!contextCapActive && reliefEnabledForLayers) ? activeLiveReliefBuckets : [];
  const currentReliefMeshes = activeReliefMeshes();
  const horizontalGlyphsActive = horizontalGlyphMasterEnabled() && horizontalGlyphConfig.enabled && horizontalGlyphRecords.length > 0;
  const arrowGlyphData = horizontalGlyphsActive && showHorizontalArrows
    ? horizontalGlyphRecords.filter((glyph) => glyph.arrow?.shaftLengthM > 0 && glyph.arrow?.headLengthM > 0)
    : [];
  const ellipseGlyphData = horizontalGlyphsActive && showHorizontalEllipses
    ? horizontalGlyphRecords.filter((glyph) => glyph.ellipse?.majorAxisM > 0 && glyph.ellipse?.minorAxisM > 0)
    : [];
  const horizontalGlyphUpdateTriggers = {
    getPosition: [activeEpoch, verticalExaggeration, uncertaintyReliefEnabled, capAppearance, horizontalGlyphScale],
    getScale: [horizontalGlyphScale],
    getColor: [horizontalGlyphOpacity],
  };

  const reliefUpdateTriggers = {
    getPosition: [activeEpoch, verticalExaggeration, uncertaintyReliefEnabled],
    getScale: [activeEpoch, verticalExaggeration, uncertaintyReliefEnabled],
  };

  return [
    // Datum ground MUST remain first. It is an invisible depth-writing surface
    // outside the completed support envelope, so exterior rim faces are hidden.
    new SolidPolygonLayer({
      id: 'datum-ground-depth-prepass',
      data: datumData,
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon,
      getFillColor: (item) => item.fillColor,
      parameters: apronParameters,
      updateTriggers: {
        getPolygon: [apronMode, depthOccludersEnabled],
        getFillColor: [apronMode, depthOccludersEnabled],
      },
    }),

    // Depth-only passes let blankies look translucent while still behaving as
    // actual support geometry in the depth buffer.
    new SolidPolygonLayer({
      id: 'interpolated-blankie-cap-depth-prepass',
      data: blankieFlatDepthData,
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (cell) => cell.capPolygon3d,
      getFillColor: [0, 0, 0, 0],
      parameters: transparentDepthParameters,
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration, depthOccludersEnabled]},
    }),

    new SolidPolygonLayer({
      id: 'interpolated-blankie-wall-depth-prepass',
      data: blankieWallDepthData,
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: [0, 0, 0, 0],
      parameters: transparentDepthParameters,
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration, depthOccludersEnabled]},
    }),

    new SolidPolygonLayer({
      id: 'interpolated-blankie-rim-depth-prepass',
      data: blankieRimDepthData,
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: [0, 0, 0, 0],
      parameters: transparentDepthParameters,
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration, depthOccludersEnabled]},
    }),

    new SolidPolygonLayer({
      id: 'live-rum-rim-walls',
      data: pistonWallsVisible ? activeLiveRimWalls : [],
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: (item) => item.fillColor,
      parameters: {cullMode: 'none'},
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration]},
    }),

    new SolidPolygonLayer({
      id: 'live-rum-shared-step-walls',
      data: pistonWallsVisible ? activeLiveWalls : [],
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: (item) => item.fillColor,
      parameters: {cullMode: 'none'},
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration]},
    }),

    new SolidPolygonLayer({
      id: 'live-rum-mean-caps',
      data: liveFlatCapData,
      pickable: true,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (cell) => cell.capPolygon3d,
      getFillColor: (cell) => cell.fillColor,
      parameters: {cullMode: 'none'},
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration, uncertaintyReliefEnabled, capAppearance]},
      onClick: (info) => handleFeatureClick(info, false),
    }),

    ...(contextCapActive ? [
      new ContextCapLayer({
        id: 'live-rum-context-map-caps',
        data: liveCells,
        mesh: contextCapMesh,
        texture: contextAtlas.canvas,
        textureParameters: getContextCapTextureParameters(),
        pickable: true,
        getPosition: (cell) => cell.reliefPosition,
        getOrientation: (cell) => [0, cell.reliefYawDeg, 0],
        getScale: (cell) => [cell.reliefWidthM, cell.reliefHeightM, 1],
        getContextUvSouth: (cell) => contextUvSouthForCell(cell),
        getContextUvNorth: (cell) => contextUvNorthForCell(cell),
        // Alpha is not canvas transparency. ContextCapLayer uses it only as
        // the internal blend weight for scientific deformation colour over
        // the opaque B/W atlas.
        getColor: (cell) => [...cell.fillColor.slice(0, 3), contextCapConfig.liveTintAlpha],
        material: CONTEXT_CAP_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: true, depthCompare: 'less-equal'},
        updateTriggers: {
          getPosition: [activeEpoch, verticalExaggeration],
          getContextUvSouth: [activeContextAtlasKey],
          getContextUvNorth: [activeContextAtlasKey],
          getColor: [contextCapConfig.liveTintAlpha],
        },
        onClick: (info) => handleFeatureClick(info, false),
      }),
    ] : []),

    ...liveReliefBuckets.flatMap((bucketData, bucketIndex) => {
      if (!bucketData.length || !currentReliefMeshes[bucketIndex]) return [];
      return [new SimpleMeshLayer({
        id: `live-rum-uncertainty-relief-${activeReliefLodKey}-band-${bucketIndex}`,
        data: bucketData,
        mesh: currentReliefMeshes[bucketIndex],
        pickable: true,
        wireframe: false,
        getPosition: (cell) => cell.reliefPosition,
        getOrientation: (cell) => [0, cell.reliefYawDeg, 0],
        getScale: (cell) => [cell.reliefWidthM, cell.reliefHeightM, cell.reliefAmplitudeM],
        getColor: (cell) => cell.fillColor,
        material: RELIEF_MATERIAL,
        parameters: {
          cullMode: 'none',
          depthWriteEnabled: true,
          depthCompare: 'less-equal',
        },
        updateTriggers: reliefUpdateTriggers,
        onClick: (info) => handleFeatureClick(info, false),
      })];
    }),

    new SolidPolygonLayer({
      id: 'interpolated-blankie-rim-visual',
      data: pistonWallsVisible ? activeBlankieRimWalls : [],
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: (item) => item.fillColor,
      parameters: transparentVisualParameters,
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration]},
    }),

    new SolidPolygonLayer({
      id: 'interpolated-blankie-wall-visual',
      data: pistonWallsVisible ? activeBlankieWalls : [],
      pickable: false,
      filled: true,
      extruded: false,
      _full3d: true,
      getPolygon: (item) => item.polygon3d,
      getFillColor: (item) => item.fillColor,
      parameters: transparentVisualParameters,
      updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration]},
    }),

    ...(blankieContextActive ? [
      new ContextCapLayer({
        id: 'interpolated-blankie-context-caps',
        data: blankieCells,
        mesh: contextCapMesh,
        texture: contextAtlas.canvas,
        textureParameters: getContextCapTextureParameters(),
        pickable: true,
        getPosition: (cell) => cell.reliefPosition,
        getOrientation: (cell) => [0, cell.reliefYawDeg, 0],
        getScale: (cell) => [cell.reliefWidthM, cell.reliefHeightM, 1],
        getContextUvSouth: (cell) => contextUvSouthForCell(cell),
        getContextUvNorth: (cell) => contextUvNorthForCell(cell),
        // Opaque local map texture with a grey support veil, composited in
        // the same fragment pass. This replaces the old transparent-window
        // effect without a coplanar colour layer.
        getColor: contextCapConfig.blankieTintColor,
        material: CONTEXT_CAP_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: true, depthCompare: 'less-equal'},
        updateTriggers: {
          getPosition: [activeEpoch, verticalExaggeration],
          getContextUvSouth: [activeContextAtlasKey],
          getContextUvNorth: [activeContextAtlasKey],
          getColor: [contextCapConfig.blankieTintColor],
        },
        onClick: (info) => handleFeatureClick(info, true),
      }),
    ] : [
      new SolidPolygonLayer({
        id: 'interpolated-blankie-caps',
        data: blankieFlatCapData,
        pickable: true,
        filled: true,
        extruded: false,
        _full3d: true,
        getPolygon: (cell) => cell.capPolygon3d,
        getFillColor: BLANKIE_CAP_COLOR,
        parameters: transparentVisualParameters,
        updateTriggers: {getPolygon: [activeEpoch, verticalExaggeration]},
        onClick: (info) => handleFeatureClick(info, true),
      }),
    ]),


    ...(horizontalParticleMasterEnabled() && showHorizontalParticles && activeHorizontalParticleFieldRuntime() ? [
      new HorizontalParticleLayer({
        id: 'horizontal-gpu-particles',
        field: activeHorizontalParticleFieldRuntime(),
        verticalModelMm: runtime.verticalModelMm,
        epochCount: runtime.epochCount,
        activeEpoch,
        verticalExaggeration,
        flatMode: false,
        flatSurfaceZM: 0,
        coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
        coordinateOrigin: [
          activeHorizontalParticleFieldRuntime().grid.coordinateOriginLonLat[0],
          activeHorizontalParticleFieldRuntime().grid.coordinateOriginLonLat[1],
          0,
        ],
        particleCapacity: horizontalParticleConfig.particleCapacity,
        particleCount: horizontalParticleCount,
        mode: horizontalParticleMode,
        speedMultiplier: horizontalParticleSpeedMultiplier,
        particleSizeMultiplier: horizontalParticleSizeMultiplier,
        particleOpacity: horizontalParticleOpacity,
        trailPersistence: horizontalParticleTrailPersistence,
        historySamples: horizontalParticleHistorySamples,
        uncertaintyStrength: horizontalParticleUncertaintyForMode(horizontalParticleMode),
        pickable: false,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'less-equal'},
        onStatus: handleHorizontalParticleStatus,
      }),
    ] : []),

    ...(arrowGlyphData.length && horizontalGlyphMeshes.shaft ? [
      new SimpleMeshLayer({
        id: 'horizontal-arrow-shafts',
        data: arrowGlyphData,
        mesh: horizontalGlyphMeshes.shaft,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphAnchorLonLat(glyph);
          return [lon, lat, glyph.glyphZ];
        },
        getOrientation: (glyph) => [0, glyph.arrow.yawDeg, 0],
        getScale: (glyph) => [glyph.arrow.shaftLengthM * horizontalGlyphScale, glyph.arrow.shaftHalfWidthM * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.arrowColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({ ...info, object: info.object?.cell }, false),
      }),
      new SimpleMeshLayer({
        id: 'horizontal-arrow-heads',
        data: arrowGlyphData,
        mesh: horizontalGlyphMeshes.head,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphPointFromTail(glyph, glyph.arrow.shaftLengthM);
          return [lon, lat, glyph.glyphZ];
        },
        getOrientation: (glyph) => [0, glyph.arrow.yawDeg, 0],
        getScale: (glyph) => [glyph.arrow.headLengthM * horizontalGlyphScale, glyph.arrow.headHalfWidthM * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.arrowColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({ ...info, object: info.object?.cell }, false),
      }),
    ] : []),

    ...(ellipseGlyphData.length && horizontalGlyphMeshes.ellipse ? [
      new SimpleMeshLayer({
        id: 'horizontal-confidence-ellipses',
        data: ellipseGlyphData,
        mesh: horizontalGlyphMeshes.ellipse,
        pickable: true,
        getPosition: (glyph) => {
          const [lon, lat] = horizontalGlyphPointFromTail(glyph, horizontalGlyphArrowTipDistanceM(glyph));
          return [lon, lat, glyph.glyphZ];
        },
        getOrientation: (glyph) => [0, glyph.ellipse.yawDeg, 0],
        getScale: (glyph) => [glyph.ellipse.majorAxisM * horizontalGlyphConfig.ellipseDisplayFactor * horizontalGlyphScale, glyph.ellipse.minorAxisM * horizontalGlyphConfig.ellipseDisplayFactor * horizontalGlyphScale, 1],
        getColor: glyphColorWithOpacity(horizontalGlyphConfig.ellipseColorRgba),
        material: HORIZONTAL_GLYPH_MATERIAL,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: horizontalGlyphUpdateTriggers,
        onClick: (info) => handleFeatureClick({ ...info, object: info.object?.cell }, false),
      }),
    ] : []),

    ...referenceGridLayers(),

    ...(studioPolygons?.getLayers?.({sceneMode, verticalExaggeration}) ?? []),

    ...selectedFeatureOutlineLayers(),

    new PathLayer({
      id: 'datum-outline',
      data: datumLineEnabled
        ? topology.rimEdges.map((edge) => ({
          path: edge.edgeLonLat.map(([lon, lat]) => [lon, lat, DATUM_LINE_Z]),
        }))
        : [],
      pickable: false,
      getPath: (item) => item.path,
      getColor: DATUM_LINE_COLOR,
      getWidth: 2.5,
      widthUnits: 'pixels',
      widthMinPixels: 1.5,
      capRounded: true,
      jointRounded: true,
      updateTriggers: {getPath: [datumLineEnabled]},
    }),
  ];
}

function updatePlaybackButton() {
  playButton.textContent = isPlaying ? '❚❚ Pause' : '▶ Play';
}

function stopPlayback() {
  if (!isPlaying) return;
  isPlaying = false;
  if (playbackFrame !== null) cancelAnimationFrame(playbackFrame);
  playbackFrame = null;
  lastPlaybackTime = null;
  updatePlaybackButton();
}

function playbackTick(timestamp) {
  if (!isPlaying || !runtime) return;
  if (lastPlaybackTime === null) lastPlaybackTime = timestamp;

  const msPerEpoch = 1000 / (runtime.playbackEpochsPerSecond * playbackSpeedMultiplier);
  const elapsed = timestamp - lastPlaybackTime;
  if (elapsed >= msPerEpoch) {
    const steps = Math.floor(elapsed / msPerEpoch);
    activeEpoch = (activeEpoch + steps) % runtime.epochCount;
    epochSlider.value = String(activeEpoch);
    lastPlaybackTime += steps * msPerEpoch;
    applyEpoch();
  }
  playbackFrame = requestAnimationFrame(playbackTick);
}

function startPlayback() {
  if (!runtime || isPlaying) return;
  isPlaying = true;
  lastPlaybackTime = null;
  updatePlaybackButton();
  playbackFrame = requestAnimationFrame(playbackTick);
}

sceneModeControl?.addEventListener('click', (event) => {
  const button = event.target.closest('.seg');
  if (!button || button.disabled) return;
  setSceneMode(button.dataset.mode);
});

epochSlider.addEventListener('input', (event) => {
  stopPlayback();
  activeEpoch = Number(event.target.value);
  applyEpoch();
});

verticalExagSlider.addEventListener('input', (event) => {
  verticalExaggeration = Number(event.target.value);
  verticalExagValue.textContent = formatExaggeration(verticalExaggeration);
  applyEpoch();
});

playbackSpeedSlider?.addEventListener('input', (event) => {
  const playbackSpeed = runtime?.playbackSpeed;
  playbackSpeedMultiplier = clamp(
    Number(event.target.value),
    playbackSpeed?.minMultiplier ?? 0.25,
    playbackSpeed?.maxMultiplier ?? 4,
  );
  if (playbackSpeedValue) playbackSpeedValue.textContent = formatPlaybackSpeed(playbackSpeedMultiplier);
  if (isPlaying) lastPlaybackTime = performance.now();
});

uncertaintyReliefToggle?.addEventListener('change', (event) => {
  if (capAppearance === 'context-map') {
    event.target.checked = false;
    return;
  }
  uncertaintyReliefEnabled = event.target.checked;
  updateFloatingLegendBars();
  applyEpoch();
});

basemapControl?.addEventListener('click', (event) => {
  const button = event.target.closest('.seg');
  if (!button || button.disabled) return;
  setBasemapMode(button.dataset.mode);
});

basemapControl?.addEventListener('change', (event) => {
  const select = event.target.closest('[data-basemap-select]');
  if (!select || select.disabled) return;
  setBasemapMode(select.value);
});

sceneGridModeSelect?.addEventListener('change', (event) => {
  setReferenceGridMode(event.target.value);
});

pistonWallsToggle?.addEventListener('click', () => {
  if (pistonWallsToggle.disabled) return;
  showPistonWalls = !showPistonWalls;
  updatePistonComponentControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

blankieCapsToggle?.addEventListener('click', () => {
  if (blankieCapsToggle.disabled) return;
  showBlankieCaps = !showBlankieCaps;
  updatePistonComponentControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

capAppearanceSelect?.addEventListener('change', (event) => {
  const select = event.target;
  if (!select || select.disabled) return;
  setCapAppearance(select.value);
});

capAppearanceControl?.addEventListener('click', (event) => {
  const button = event.target.closest('.seg');
  if (!button || button.disabled) return;
  setCapAppearance(button.dataset.mode);
});

playButton.addEventListener('click', () => {
  if (isPlaying) stopPlayback();
  else startPlayback();
});

apronModeControl.addEventListener('click', (event) => {
  const button = event.target.closest('.seg');
  if (!button) return;
  apronMode = button.dataset.mode;
  for (const segment of apronModeControl.querySelectorAll('.seg')) {
    segment.classList.toggle('active', segment === button);
  }
  applyEpoch();
});

datumLineToggle.addEventListener('change', (event) => {
  datumLineEnabled = event.target.checked;
  applyEpoch();
});

depthOccluderToggle.addEventListener('change', (event) => {
  depthOccludersEnabled = event.target.checked;
  applyEpoch();
});

twoDRumOpacitySlider?.addEventListener('input', (event) => {
  twoDAnalysisConfig.rumFillOpacity = clamp(
    Number(event.target.value),
    twoDAnalysisConfig.rumFillOpacityMin,
    twoDAnalysisConfig.rumFillOpacityMax,
  );
  updateTwoDAnalysisControls();
  updateReadingNote();
  if (runtime && sceneMode === '2d') deckOverlay.setProps({layers: makeLayers()});
});

horizontalArrowsToggle?.addEventListener('change', (event) => {
  showHorizontalArrows = Boolean(event.target.checked);
  updateFloatingLegendBars();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalEllipsesToggle?.addEventListener('change', (event) => {
  showHorizontalEllipses = Boolean(event.target.checked);
  updateFloatingLegendBars();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalGlyphOpacitySlider?.addEventListener('input', (event) => {
  horizontalGlyphOpacity = clamp(Number(event.target.value), 0, 1);
  updateHorizontalGlyphControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalGlyphScaleSlider?.addEventListener('input', (event) => {
  horizontalGlyphScale = clamp(Number(event.target.value), 0.5, 2.5);
  updateHorizontalGlyphControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticlesToggle?.addEventListener('change', (event) => {
  showHorizontalParticles = Boolean(event.target.checked);
  updateHorizontalParticleControls();
  updateFloatingLegendBars();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleFieldModeControl?.addEventListener('change', (event) => {
  const requested = String(event.target.value).toLowerCase();
  horizontalParticleFieldMode = requested === 'lsc' && horizontalParticleLscRuntime ? 'lsc' : 'raw';
  syncHorizontalParticleFieldRuntime({resetGpuStatus: true});
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleModeControl?.addEventListener('change', (event) => {
  const mode = String(event.target.value).toLowerCase();
  horizontalParticleMode = ['mean', 'montecarlo', 'shimmer'].includes(mode) ? mode : 'mean';
  horizontalParticleUncertaintyStrength = horizontalParticleUncertaintyForMode(horizontalParticleMode);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleCountSlider?.addEventListener('input', (event) => {
  horizontalParticleCount = clamp(
    Math.round(Number(event.target.value)),
    0,
    horizontalParticleConfig.particleCapacity,
  );
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleSizeSlider?.addEventListener('input', (event) => {
  horizontalParticleSizeMultiplier = clamp(Number(event.target.value), 0.1, 8);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleSpeedSlider?.addEventListener('input', (event) => {
  horizontalParticleSpeedMultiplier = clamp(Number(event.target.value), 0.1, 6);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleTrailDurationSlider?.addEventListener('input', (event) => {
  const min = Number(horizontalParticleConfig.trailDurationMinS ?? 0.40);
  const max = Number(horizontalParticleConfig.trailDurationMaxS ?? 3.20);
  horizontalParticleTrailDurationSeconds = clamp(Number(event.target.value), min, max);
  horizontalParticleHistorySamples = particleHistorySamplesForDuration(horizontalParticleTrailDurationSeconds);
  horizontalParticleTrailDurationSeconds = particleHistoryDurationForSamples(horizontalParticleHistorySamples);
  updateHorizontalParticleControls();
});

horizontalParticleTrailDurationSlider?.addEventListener('change', () => {
  // A duration change resets only the sentinel history window. The particle
  // simulation state remains alive; release the slider to commit that reset.
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleTrailPersistenceSlider?.addEventListener('input', (event) => {
  horizontalParticleTrailPersistence = clamp(Number(event.target.value), 0.50, 0.999);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleOpacitySlider?.addEventListener('input', (event) => {
  horizontalParticleOpacity = clamp(Number(event.target.value), 0, 1);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

horizontalParticleUncertaintySlider?.addEventListener('input', (event) => {
  if (horizontalParticleMode !== 'shimmer' && horizontalParticleMode !== 'montecarlo') return;
  horizontalParticleUncertaintyStrengths[horizontalParticleMode] = clamp(Number(event.target.value), 0, 2);
  horizontalParticleUncertaintyStrength = horizontalParticleUncertaintyForMode(horizontalParticleMode);
  updateHorizontalParticleControls();
  if (runtime) deckOverlay.setProps({layers: makeLayers()});
});

// UI-B2 compact drawer interactions -----------------------------------------
drawerDefaultsButton?.addEventListener('click', () => {
  resetDrawerVisualDefaults();
});

rightDrawerBurger?.addEventListener('click', () => {
  const open = !rightControlRoot?.classList.contains('drawerOpen');
  setDrawerOpen(open, activeDrawerId);
});

drawerSectionTitles.forEach((title) => {
  title.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.target?.closest?.('.drawerBinder, .drawerSectionMaster')) return;
    const section = title.closest('.drawerSection');
    if (!section) return;
    activeDrawerId = section.id;
    if (!rightControlRoot?.classList.contains('drawerOpen')) {
      setDrawerOpen(true, activeDrawerId);
      return;
    }
    const willOpen = section.classList.contains('collapsed');
    section.classList.toggle('collapsed', !willOpen);
    title.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    syncMiniViewerActiveState({draw: true});
  });
});

zoomInButton?.addEventListener('click', () => map.zoomTo(map.getZoom() + 0.75, {duration: 180}));
zoomOutButton?.addEventListener('click', () => map.zoomTo(map.getZoom() - 0.75, {duration: 180}));
resetViewButton?.addEventListener('click', () => frameScene({liveOnly: false}));
flyToRumsButton?.addEventListener('click', () => frameScene({liveOnly: true}));
viewModeToggleButton?.addEventListener('click', () => setSceneMode(sceneMode === '2d' ? '3d' : '2d'));
screenshotButton?.addEventListener('click', () => { void saveCompositeScreenshot(); });
fullscreenButton?.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenButton);
navInfoButton?.addEventListener('click', () => {
  const next = !viewerInfoPanel?.classList.contains('open');
  viewerInfoPanel?.classList.toggle('open', next);
  viewerInfoPanel?.setAttribute('aria-hidden', next ? 'false' : 'true');
});
viewerInfoCloseButton?.addEventListener('click', () => {
  viewerInfoPanel?.classList.remove('open');
  viewerInfoPanel?.setAttribute('aria-hidden', 'true');
});
miniViewerCanvas?.addEventListener('click', handleMiniViewerClick);

epochFirstButton?.addEventListener('click', () => setEpochIndex(0));
epochPrevButton?.addEventListener('click', () => setEpochIndex(activeEpoch - 1));
epochNextButton?.addEventListener('click', () => setEpochIndex(activeEpoch + 1));
epochLastButton?.addEventListener('click', () => setEpochIndex((runtime?.epochCount ?? 1) - 1));

setDrawerOpen(true);
updateFullscreenButton();
installShellAssets();
horizontalLegendRenderer.start();

window.addEventListener('resize', () => {
  horizontalLegendRenderer.redraw();
  scheduleTrendlineDraw();
  syncMiniViewerActiveState({draw: true});
  studioCaptureMode?.handleResize?.();
});
window.addEventListener('beforeunload', () => {
  stopPlayback();
  if (trendlineDrawFrame !== null) cancelAnimationFrame(trendlineDrawFrame);
  if (miniViewerDrawFrame !== null) cancelAnimationFrame(miniViewerDrawFrame);
  if (miniViewerCameraIdleTimer !== null) window.clearTimeout(miniViewerCameraIdleTimer);
  studioCaptureMode?.destroy?.();
  horizontalLegendRenderer.destroy();
});
startFpsCounter();

const deckOverlay = new MapboxOverlay({
  interleaved: false,
  views: new MapView({
    id: 'mapbox',
    farZMultiplier: FAR_Z_MULTIPLIER_FALLBACK,
  }),
  layers: [],
});

map.addControl(deckOverlay);
studioPolygons = createStudioPolygonAnnotations({
  map,
  force2D: () => setSceneMode('2d'),
  openStudioDrawer: () => openDrawerSection('drawerStudio'),
  requestRedraw: () => {
    if (runtime && deckOverlay) deckOverlay.setProps({layers: makeLayers()});
  },
  flashStatus: (message, tone = 'info') => flashScreenshotStatus(message, tone),
  elements: {
    sceneToggle: scenePolygonToggle,
    addButton: studioPolygonAddButton,
    status: studioPolygonStatus,
    drawBar: studioPolygonDrawBar,
    drawStatus: studioPolygonDrawStatus,
    undoButton: studioPolygonUndoButton,
    finishButton: studioPolygonFinishButton,
    cancelButton: studioPolygonCancelButton,
    saveForm: studioPolygonSaveForm,
    nameInput: studioPolygonNameInput,
    infoInput: studioPolygonInfoInput,
    saveButton: studioPolygonSaveButton,
    formCancelButton: studioPolygonFormCancelButton,
    list: studioPolygonList,
  },
});

studioCaptureMode = createStudioCaptureMode({
  elements: {
    accordion: studioCaptureAccordion,
    panel: studioCapturePanel,
    status: studioCaptureStatus,
    viewfinderOverlay: studioCaptureViewfinderOverlay,
    viewfinderToggle: studioCaptureViewfinderToggle,
    captureButton: studioCaptureCurrentViewButton,
    introButton: studioCaptureIntroButton,
    previewButton: studioCapturePreviewButton,
    clearButton: studioCaptureClearButton,
    list: studioCaptureList,
  },
  api: {
    isReady: () => Boolean(runtime),
    openStudioDrawer: () => openDrawerSection('drawerStudio'),
    getCameraState: () => captureCameraState(),
    getSceneState: () => getStudioCaptureSceneState(),
    getIntroCameraState: () => studioCaptureIntroCameraState(),
    applySceneState: (state) => applyStudioCaptureSceneState(state),
    applyCameraState: (camera, options) => applyStudioCaptureCameraState(camera, options),
    getEpochLabel: (epoch) => runtime?.epochAxis?.epochs?.[clamp(Math.round(Number(epoch ?? activeEpoch)), 0, Math.max(0, (runtime?.epochCount ?? 1) - 1))]?.label
      ?? runtime?.epochAxis?.epochs?.[clamp(Math.round(Number(epoch ?? activeEpoch)), 0, Math.max(0, (runtime?.epochCount ?? 1) - 1))]
      ?? `Epoch ${Number(epoch ?? activeEpoch) + 1}`,
    getEpochCount: () => runtime?.epochCount ?? 0,
    flashStatus: (message, tone = 'info') => flashScreenshotStatus(message, tone),
  },
});
installDrawerMasterSwitches();
syncMiniViewerActiveState({draw: true});
studioCaptureMode?.refresh?.();

map.on('movestart', () => {
  markMiniViewerCameraMoving();
});
map.on('move', () => {
  scheduleBottomStatusUpdate();
});
map.on('rotate', () => {
  scheduleBottomStatusUpdate();
});
map.on('pitch', () => {
  scheduleBottomStatusUpdate();
});
map.on('styledata', scheduleBottomStatusUpdate);
map.on('mousemove', (event) => {
  bottomStatusPointerLngLat = event.lngLat;
  scheduleBottomStatusUpdate();
});
map.getContainer().addEventListener('mouseleave', () => {
  bottomStatusPointerLngLat = null;
  scheduleBottomStatusUpdate();
});

map.on('click', (event) => {
  if (studioPolygons?.handleMapClick?.(event)) return;
  if (suppressNextMapClickClear) {
    suppressNextMapClickClear = false;
    return;
  }
  if (selectedCell) clearSelectedFeature();
});


rumTrendlineAxisModeSelect?.addEventListener('change', (event) => {
  trendlineAxisMode = String(event.target.value || 'auto').toLowerCase();
  updateTrendlinePanelState();
  scheduleTrendlineDraw();
});

rumTrendlineMinInput?.addEventListener('input', (event) => {
  trendlineCustomMin = Number(event.target.value);
  scheduleTrendlineDraw();
});

rumTrendlineMaxInput?.addEventListener('input', (event) => {
  trendlineCustomMax = Number(event.target.value);
  scheduleTrendlineDraw();
});

rumTrendlineResizeHandle?.addEventListener('pointerdown', (event) => {
  event.preventDefault();
  trendlineResizeDrag = {
    pointerId: event.pointerId,
    startY: Number(event.clientY),
    startHeight: trendlineHeightPx,
  };
  rumTrendlineResizeHandle.setPointerCapture?.(event.pointerId);
  epochPanel?.classList.add('trendline-resizing');
});

rumTrendlineResizeHandle?.addEventListener('pointermove', (event) => {
  if (!trendlineResizeDrag || event.pointerId !== trendlineResizeDrag.pointerId) return;
  event.preventDefault();
  const deltaY = trendlineResizeDrag.startY - Number(event.clientY);
  setTrendlineHeight(trendlineResizeDrag.startHeight + deltaY);
});

function endTrendlineResize(event) {
  if (!trendlineResizeDrag || event.pointerId !== trendlineResizeDrag.pointerId) return;
  rumTrendlineResizeHandle?.releasePointerCapture?.(event.pointerId);
  trendlineResizeDrag = null;
  epochPanel?.classList.remove('trendline-resizing');
}

rumTrendlineResizeHandle?.addEventListener('pointerup', endTrendlineResize);
rumTrendlineResizeHandle?.addEventListener('pointercancel', endTrendlineResize);

rumTrendlinePngButton?.addEventListener('click', exportTrendlinePng);
rumTrendlineCloseButton?.addEventListener('click', closeTrendline);

tooltip?.addEventListener('click', (event) => {
  event.stopPropagation();
  const action = event.target?.closest?.('[data-popup-action]')?.dataset?.popupAction;
  if (action === 'close') {
    clearSelectedFeature();
  } else if (action === 'toggle-more') {
    selectedTooltipExpanded = !selectedTooltipExpanded;
    renderSelectedPopup();
  } else if (action === 'open-trendline') {
    if (selectedCell && !selectedCellIsBlankie) openTrendlineForCell(selectedCell);
  } else if (action === 'add-polygon') {
    if (selectedCell) {
      studioPolygons?.startDrawing?.({
        sourceCell: selectedCell,
        name: 'Subsidence bowl',
        info: `Started from ${selectedCell.rumId ?? selectedCell.cellId ?? 'selected cell'}`,
      });
    }
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && selectedCell) clearSelectedFeature();
});

scheduleBottomStatusUpdate();

map.on('moveend', () => {
  if (runtime) {
    if (sceneMode === '2d') savedTwoDCamera = captureCameraState();
    else savedThreeDCamera = captureCameraState();
  }
  scheduleMiniViewerCameraIdleRefresh();
});

// Relief LOD normally responds to zoom, but high-pitch views also force the
// sparse mesh. Camera-depth values are quantised and updated only when needed.
map.on('zoomend', () => {
  syncReliefLod();
  syncContextAtlasLod();
});
let pendingPitchSync = null;
function schedulePitchSync() {
  if (sceneMode === '2d') return;
  if (pendingPitchSync) return;
  pendingPitchSync = requestAnimationFrame(() => {
    pendingPitchSync = null;
    syncCameraDepthContract();
    syncReliefLod();
  });
}
map.on('pitch', schedulePitchSync);
map.on('pitchend', () => {
  syncCameraDepthContract({force: true});
  syncReliefLod();
});

map.on('load', () => {
  loadJakartaRuntime().catch((error) => {
    console.error('[Proto1 DeckGL] Runtime failed to load:', error);
    epochLabel.textContent = 'Runtime failed — open F12 Console.';
    focusLabel.textContent = error.message ?? String(error);
  });
});
