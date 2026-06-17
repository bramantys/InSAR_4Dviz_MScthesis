# Prototype 1 Handoff Map

## RUM-Based Jakarta 4D InSAR Visualization Prototype

**Status:** accepted / parked technical baseline
**Recommended source package:** latest full Prototype 1 folder ZIP
**Accepted baseline:** `C4 + D2A`
**Main purpose of this document:** provide a navigation map for future chats, thesis writing, meetings, presentations, reviewers, and later prototype development.

---

# 0. Read this first

Prototype 1 is the RUM-based Jakarta 4D InSAR visualization prototype developed in CesiumJS.

It demonstrates how post-processed InSAR deformation products can be visualized in an interactive web viewer using:

* animated vertical deformation surfaces;
* horizontal motion particles inspired by Windy-style maps;
* vertical uncertainty relief;
* horizontal uncertainty glyphs and particle uncertainty modes;
* time playback;
* RUM picking and trendline inspection;
* web-based deployment.

Prototype 1 is now considered technically accepted and parked.

The current accepted version is:

```text
Prototype 1 = C4 + D2A
```

Meaning:

* **C4:** accepted semantic vertical-uncertainty LOD with 2×2 far relief and 4×4 near relief.
* **D2A:** playback-smoothing fix using predecoded particle surface-height caches.

Do not reopen Prototype 1 optimization unless there is a clear reason. It is good enough for thesis use, demonstration, and as an architectural reference for future prototypes.

---

# 1. What this handoff is and is not

## This handoff is

A map for understanding the project folder and the final design state.

It is intended for:

* a new ChatGPT coding thread;
* a thesis/report-writing thread;
* future Prototype 2 / Prototype 3 development;
* meeting preparation;
* presentation preparation;
* external reviewers or collaborators;
* the student returning later after losing context.

## This handoff is not

A replacement for the actual project files.

The source of truth is still:

```text
latest full Prototype 1 folder ZIP
latest_manifest.json
latest_run.log
pipeline scripts
viewer HTML
generated tilesets and metadata
```

This markdown explains what those files mean and which parts are accepted.

---

# 2. Recommended upload bundle for any new chat

When starting a new chat, upload or provide:

```text
1. latest full Prototype 1 ZIP
2. this markdown handoff
3. latest_manifest.json
4. latest_run.log
```

Recommended instruction to the new chat:

```text
Read the handoff first. Treat Prototype 1 as accepted and parked. Do not optimize Prototype 1 unless I explicitly ask. Use it as the architecture and methodology reference for future prototype work.
```

If the task is only thesis writing, still upload this markdown and optionally the latest ZIP.
If the task is coding, upload the latest ZIP as well.

---

# 3. Final accepted Prototype 1 identity

Prototype 1 is:

```text
RUM-based Jakarta 4D InSAR visualization prototype
```

Core design:

```text
vertical deformation       → animated 3D RUM caps
horizontal deformation     → animated particles
vertical uncertainty       → lowpoly checkerboard relief
horizontal uncertainty     → Monte Carlo particles / shimmer / confidence glyphs
time dimension             → epoch slider + playback
inspection                 → RUM picking + popup + trendline
deployment target          → browser / GitHub Pages-style static web app
```

It is a thesis prototype, not a production-grade monitoring portal.

---

# 4. Research role of Prototype 1

Prototype 1 supports the thesis topic:

```text
Spatiotemporal Visualization of InSAR Ground Deformation
```

It addresses the problem that InSAR deformation is fundamentally multidimensional:

```text
3D space + time + uncertainty
```

while conventional portals often show:

```text
2D colored maps + separate clicked time-series plots
```

Prototype 1 explores whether a browser-based 4D geospatial viewer can help communicate deformation patterns and uncertainty more intuitively.

Its role in the thesis:

* first end-to-end prototype;
* RUM-based product case study;
* Jakarta subsidence case;
* tool-performance experiment for CesiumJS;
* basis for expert/operator viewer design;
* source of exported clips or figures for non-expert interpretation testing;
* architecture reference for later parcel/seasonal or structural prototypes.

---

# 5. Dataset / product type

Prototype 1 uses a RUM-based Jakarta subsidence product.

RUM means:

```text
Region of Uniform Motion
```

Key data characteristics:

* gridded / cell-like RUM geometry;
* vertical deformation time series;
* horizontal deformation vectors;
* horizontal uncertainty information;
* vertical uncertainty information;
* multiple epochs;
* suitable for testing both spatial and temporal visualization.

The final viewer uses packed and generated data products rather than raw processing data.

Important generated data products include:

```text
_internal/data_pipeline/packed_series.json
_internal/data_pipeline/horizontal_field.json
_internal/data_pipeline/horizontal_particle_field.json
_internal/data_pipeline/tiles/height_texture.png
_internal/data_pipeline/tiles/height_meta.json
_internal/data_pipeline/tiles/epoch_axis.json
```

---

# 6. Most important runtime files

The key runtime files are:

```text
viz1_dev_v4_dualmode.html
_internal/js/horizontal_particles_engine.js
config/project_config.json
_internal/data_pipeline/
run_records/latest_manifest.json
run_records/latest_run.log
```

The key generated 3D Tiles products are:

```text
_internal/data_pipeline/tiles/tileset.json
_internal/data_pipeline/tiles_flat_real/tileset.json
_internal/data_pipeline/tiles_walls_real/tileset.json
_internal/data_pipeline/tiles_blank/tileset.json
_internal/data_pipeline/tiles_walls_blank/tileset.json
_internal/data_pipeline/tiles_arrows/tileset.json
_internal/data_pipeline/tiles_ellipses/tileset.json
```

Cesium runtime should exist at:

```text
_internal/cesium/Cesium.js
_internal/cesium/Widgets/widgets.css
_internal/cesium/Assets/
_internal/cesium/ThirdParty/
_internal/cesium/Workers/
```

If Cesium files are missing or nested incorrectly, the viewer may show MIME errors such as `text/html` for `Cesium.js` or `widgets.css`.

---

# 7. Pipeline overview

The pipeline converts processed InSAR/RUM products into a static web viewer package.

Conceptually:

```text
input data
→ coordinate preparation
→ RUM geometry preparation
→ vertical time-series packing
→ horizontal field preparation
→ B3DM / 3D Tiles generation
→ viewer tuning metadata
→ final CesiumJS viewer
```

Important pipeline responsibilities:

* generate RUM cap tiles;
* generate wall tiles;
* generate flat-cap product;
* generate blank/no-data products;
* generate horizontal arrow and ellipse glyphs;
* generate height texture and metadata;
* generate packed time-series data;
* generate horizontal particle field;
* write viewer tuning.

The viewer should be treated as a generated/assembled runtime product, not the only source of truth.

---

# 8. Viewer architecture overview

Prototype 1 is a CesiumJS single-page viewer.

Major viewer components:

```text
Cesium globe / scene
3D Tiles cap layers
3D Tiles wall layers
3D Tiles arrows / ellipses
CustomShader for epoch-based vertical animation
Canvas-based horizontal particle layer
UI toolbox / accordions
epoch slider and playback controls
popup and trendline panel
debug/performance harnesses
```

The viewer uses Cesium for:

* geospatial camera navigation;
* imagery/background map;
* 3D Tiles loading;
* feature picking;
* rendering cap/wall/glyph geometry.

The viewer uses a separate canvas overlay for:

* animated horizontal particles;
* shimmer uncertainty mode;
* Monte Carlo particle realization mode.

---

# 9. Final visual encodings

## 9.1 Vertical deformation

Vertical deformation is shown by animating RUM cap height through time.

The cap height follows the selected epoch.

Users can control vertical exaggeration.

The height information is stored in texture/packed time-series products and sampled or applied through the Cesium CustomShader path.

## 9.2 Vertical uncertainty

Accepted encoding:

```text
lowpoly checkerboard pyramid relief
```

Visual idea:

* the real RUM cap remains the base surface;
* small alternating upward/downward pyramids represent vertical uncertainty;
* the relief is visibly synthetic, not terrain;
* the base color remains readable;
* uncertainty remains visible without overwhelming the deformation color.

Accepted C4 geometry:

```text
far LOD  → 2×2 relief
near LOD → 4×4 relief
OFF      → true-flat caps
```

Important C4 decision:

```text
The 4×4 near relief keeps the old 6×6 pyramid footprint.
```

So the pyramids remain small, but are more widely spaced.

## 9.3 Horizontal deformation

Horizontal motion is shown using animated particles.

The visual inspiration is Windy-style motion maps, but the deformation field is RUM-based and patchier than wind.

The particle system supports:

```text
mean direction mode
shimmer / jitter uncertainty mode
Monte Carlo realization uncertainty mode
```

The Monte Carlo mode is more scientifically defensible because particle paths diverge according to sampled uncertainty realizations.

## 9.4 Horizontal static glyphs

Static horizontal arrows and confidence ellipses are available as B3DM tilesets.

A 1σ significance filter is applied to reduce clutter.

Final result:

```text
static glyphs are useful for expert inspection
static glyphs are not currently a major performance bottleneck
```

## 9.5 Time and trendline

Time is represented with:

* epoch slider;
* playback;
* date labels;
* popup/trendline inspection.

The popup/trendline distinguishes model and measurement logic where available.

---

# 10. Accepted performance batches and decisions

## C3 — semantic LOD introduced

C3 introduced semantic LOD for vertical uncertainty relief:

```text
2×2 far relief → 6×6 near relief
```

It also kept a separate true-flat product for uncertainty OFF.

This solved the problem that uncertainty should not simply disappear or become visually flat because of distance.

## C4 — 6×6 near relief reduced to 4×4

C4 changed:

```text
near relief: 6×6 → 4×4
```

while preserving the old 6×6 pyramid footprint.

This reduced near geometry by roughly 55.6% in vertices/triangles and was visually accepted.

Accepted hierarchy:

```text
2×2 far relief → 4×4 near relief
```

## D2A — playback hiccup fixed

D2A fixed a playback frame-pacing issue.

Old behavior:

```text
halt → catch up → halt
```

Cause:

```text
per-epoch Canvas2D getImageData readback
+ decoding 5,077 rows during playback
```

Fix:

```text
predecode particle surface-height cache once at startup
then swap cached Float32Array views per epoch
```

Validation example:

```text
cacheMode: predecoded_epoch_major_texture
cacheBuildMs: about 21 ms
epoch update mean: about 0 ms
epoch update max: about 1 ms
```

Result:

```text
playback hiccup removed
remaining issue is low but stable FPS
```

---

# 11. Current accepted performance interpretation

Prototype 1 is usable, but not high-FPS.

Accepted performance summary:

```text
2D overview:
acceptable

3D static:
borderline but usable

3D playback:
stable but low FPS

3D roaming + playback:
limited

epoch playback pacing:
fixed by D2A
```

Important measured/observed conclusions:

```text
2×2 far relief and true-flat caps had similar city-scale FPS
→ keep 2×2 far relief for semantic correctness

near AUTO and forced 4×4 behaved similarly at close view
→ LOD wiring is correct

vertical exaggeration 10× vs 20× no longer changed FPS much
→ C4 reduced the old geometry sensitivity

horizontal particles cost some FPS
→ real but secondary

static glyphs had little effect after filtering
→ not a priority bottleneck

remaining low FPS is steady rendering cost
→ not playback scheduling anymore
```

---

# 12. CesiumJS tool-performance conclusion

Prototype 1 reached a practical CesiumJS/WebGL limit.

This should be treated as a thesis finding, not a failure.

Final interpretation:

```text
CesiumJS is suitable for accessible browser-based geospatial 4D visualization,
interactive expert inspection,
and zero-install stakeholder-facing prototypes.
```

But:

```text
CesiumJS is not ideal for high-FPS game-like 4D animation,
dense animated uncertainty relief,
fully occluded 3D particles,
or cinematic simultaneous camera motion and time playback.
```

A Unity/Unreal-style engine would likely be more suitable for high-FPS immersive/cinematic rendering, but that would conflict with the web-accessible, low-installation-friction goal.

---

# 13. What is accepted and should not be reopened casually

Accepted:

```text
C4 + D2A baseline
2×2 far relief
4×4 near relief
true-flat OFF product
1σ horizontal glyph filter
opaque 3D pass at opacity = 1.0
predecoded playback cache
CesiumJS as Prototype 1 platform
```

Do not reopen unless explicitly requested:

```text
more 3×3 / 4×4 / 5×5 relief variants
more shader surgery
flat-far default mode
Cesium internal LOD hacks
particle-engine rewrite
Unreal/Unity migration for Prototype 1
further FPS chasing for Prototype 1
```

Prototype 1 is parked.

---

# 14. Known limitations to document

Important limitations:

1. **Performance limitation**
   3D playback and roaming are stable but low FPS.

2. **Cesium suitability boundary**
   CesiumJS provides excellent geospatial context and web deployment, but not game-engine-level animation smoothness.

3. **3D particle occlusion**
   Horizontal particles are canvas-based and do not fully behave like native occluded 3D objects.

4. **Visual complexity**
   Combining vertical deformation, vertical uncertainty, horizontal motion, horizontal uncertainty, walls, particles, and time playback can overload both renderer and user.

5. **Dataset-specific tuning**
   Color scales, exaggeration, uncertainty visibility, camera defaults, and particle settings may need dataset-specific adjustment.

6. **Not a production portal**
   This is a research prototype for visualization design and evaluation.

---

# 15. Deployment recommendations

A clean deployment/export step is recommended.

Do not delete pipeline intermediates from the working directory.

Instead:

```text
full pipeline workspace
→ clean runtime export folder
→ GitHub Pages / static web deployment
```

The runtime export should include only files required by the viewer.

Exclude from deployment if not fetched by the viewer:

```text
large intermediate JSON files
pipeline QA files
temporary validation files
__pycache__
duplicate particle engine copies
old experimental viewers
```

Keep these in the reproducible working directory, but not in the public runtime bundle.

---

# 16. Reusable architecture for future prototypes

Future prototypes should reuse the architecture, not restart from zero.

Reusable from Prototype 1:

```text
CesiumJS viewer shell
UI layout
epoch slider and playback controls
vertical exaggeration control
camera initialization from dataset bounds
viewer_tuning metadata pattern
config/project_config pattern
height texture / height metadata pattern
packed time-series pattern
3D Tiles generation pattern
CustomShader epoch animation pattern
feature picking
popup and trendline logic
debug harnesses
deployment/export structure
```

Reusable conceptually:

```text
separate data preparation pipeline
generated static web viewer
tileset-based geometry products
metadata-driven viewer tuning
runtime debug/performance tools
clear distinction between observed data, missing data, and visual encoding
```

---

# 17. Prototype 2 direction

Prototype 2 should likely focus on:

```text
parcel / field-based seasonal deformation
```

Main idea:

```text
irregular polygons acting as piston-like parcels
moving up and down through time
possibly sinusoidal or reversible seasonal motion
```

Prototype 2 should not blindly copy all Prototype 1 encodings.

Prototype 1 was:

```text
regular square RUMs
Jakarta subsidence
monotonic deformation
horizontal motion important
vertical uncertainty relief important
```

Prototype 2 may be:

```text
irregular parcel polygons
seasonal / cyclic vertical motion
horizontal motion optional or absent
time behavior more important than horizontal particles
visual clarity of parcel motion more important than uncertainty spikes
```

Recommended Prototype 2 first milestone:

```text
1. load parcel polygons
2. assign parcel IDs
3. attach synthetic sinusoidal time series
4. generate animated parcel caps as 3D Tiles
5. reuse Cesium viewer and epoch slider
6. support parcel picking
7. show parcel trendline
```

Do not start Prototype 2 with uncertainty, particles, or complex styling.

First prove:

```text
parcel polygons move correctly through time
```

---

# 18. Prototype 3 / later possibilities

Possible future directions:

## Parcel / seasonal products

* irregular polygon piston visualization;
* seasonal/cyclic motion;
* phase and amplitude encoding;
* parcel-level time-series inspection.

## Structural / mesh products

* dams, buildings, bridges, embankments;
* exaggerated deformation mesh;
* point scatterers linked to structure geometry;
* uncertainty shown as bands, halos, or confidence envelopes.

## Point-cloud / PSI products

* scatterer time-series inspection;
* uncertainty and quality filtering;
* cluster/neighbor comparison;
* 3D context integration.

## Presentation/export products

* 10–20 second flyover clips;
* simplified non-expert visual explanation;
* still figures with linked live viewer;
* interactive thesis appendix.

---

# 19. How to use Prototype 1 in the thesis report

Prototype 1 can support several thesis sections.

## Methodology

Use it to explain:

* data preparation;
* RUM-based product handling;
* CesiumJS pipeline;
* 3D Tiles generation;
* shader-based epoch animation;
* horizontal particle visualization;
* uncertainty encoding design;
* viewer interaction design.

## Results

Use it to show:

* final viewer screenshots;
* 2D and 3D modes;
* vertical deformation animation;
* horizontal particle visualization;
* vertical uncertainty relief;
* horizontal uncertainty modes;
* trendline interaction;
* performance findings.

## Discussion

Use it to discuss:

* effectiveness of 4D visualization;
* uncertainty communication tradeoffs;
* CesiumJS strengths and limits;
* visual clarity versus scientific correctness;
* expert/operator versus non-expert presentation needs;
* why Prototype 2 is needed.

## Limitations

Use it to state:

* low 3D FPS;
* no true native 3D particle occlusion;
* dataset-specific tuning;
* prototype not production portal;
* browser/WebGL limitation.

---

# 20. Meeting / presentation summary

Short version for meetings:

```text
Prototype 1 is the accepted RUM-based Jakarta 4D InSAR viewer. It visualizes vertical deformation as animated 3D RUM surfaces, horizontal deformation as Windy-style particles, vertical uncertainty as lowpoly checkerboard relief, and horizontal uncertainty through Monte Carlo particles and confidence glyphs. The final architecture uses semantic LOD for uncertainty relief, with 2×2 far relief and 4×4 near relief, plus true-flat caps when uncertainty is off. Playback hiccups were fixed by predecoding particle surface-height caches. The viewer is usable for expert inspection and web-based demonstration, but 3D playback remains low-FPS, showing the practical limit of CesiumJS for dense game-like 4D animation. Prototype 1 is now parked and will serve as the foundation for Prototype 2, likely focused on parcel or seasonal deformation.
```

Very short version:

```text
Prototype 1 proves the RUM-based 4D visualization concept in CesiumJS, including deformation, uncertainty, particles, time playback, and interaction. It is accepted and parked. The main lesson is that CesiumJS is strong for accessible geospatial prototypes but limited for high-FPS animated simulation. The architecture will be reused for later prototypes.
```

---

# 21. Final status statement

Prototype 1 is parked as:

```text
accepted RUM-based Jakarta 4D InSAR visualization prototype
```

The accepted baseline is:

```text
C4 + D2A
```

Future work should use Prototype 1 as:

```text
architecture reference
methodology evidence
tool-performance case study
presentation/demo material
foundation for Prototype 2 and beyond
```

Do not spend more thesis time chasing small FPS gains unless performance becomes a hard blocker for evaluation or presentation.
