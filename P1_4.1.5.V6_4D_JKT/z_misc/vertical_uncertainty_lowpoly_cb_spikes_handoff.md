# Vertical Uncertainty Encoding Handoff

## Selected design: Lowpoly Checkerboard Spikes

**Source sandbox:** `dimple_sandbox_proto1_clean_v16.html`  
**Decision status:** selected visual encoding for the next Prototype 1 integration pass  
**Primary targets:** square RUM caps first, parcel polygons next

---

## 1. Final design decision

The selected vertical-uncertainty encoding is:

> **Lowpoly CB spikes** — a checkerboard lattice of alternating upward and downward four-sided pyramids embedded in the RUM or parcel surface.

`CB` means **checkerboard**.

The selected development settings are:

| Setting | Selected value | Status |
|---|---:|---|
| Encoding alternative | Lowpoly CB spikes | Decided |
| Pyramid shape | Four-sided square pyramid | Decided |
| Checkerboard frequency | 6 × 6 per square RUM cap | Decided for current RUM test |
| Relief cue mode | v16 **Neutral slope shade** | Decided |
| Relief cue strength | 100% | Decided |
| Amplitude mapping | Not applicable | Decided |
| Smooth-surface tessellation | Not applicable | Decided |
| Vertical exaggeration linkage | Follow the real viewer control | Open integration decision |
| Sigma display range | 1σ or 2σ | Open user-interface/scientific decision |

The lowpoly geometry must remain lowpoly regardless of the sandbox tessellation selector. It is generated through a separate geometry builder and must not pass through the smooth tessellated-cap path.

---

## 2. Why this alternative was selected

The lowpoly checkerboard spikes performed better than the other tested alternatives for the intended use case.

### 2.1 It remains visible at low uncertainty

Smooth alternatives such as egg-carton and radial relief require enough slope before their shading becomes visible. The lowpoly pyramids create discrete faces with stable face normals, so they remain perceptible at lower uncertainty amplitudes and lower exaggeration.

### 2.2 It remains visible from farther zoom levels

At distance, users can still perceive the alternating light/dark peak pattern. The checkerboard spacing preserves enough of the original flat cap color between pyramids, so the velocity color remains visible.

### 2.3 It protects the quantitative base color better than egg-carton relief

Egg-carton relief can become dominated by dark slopes when uncertainty and exaggeration are high. The lowpoly pattern deliberately leaves neutral flat space around every pyramid footprint.

The intended visual grammar is:

- **base color:** vertical velocity or displacement attribute;
- **main cap height:** vertical deformation;
- **alternating lowpoly relief:** vertical uncertainty;
- **bounded shading:** readability aid only.

### 2.4 It is clearly synthetic

The pyramids are not intended to resemble natural terrain. Their faceted, checkerboard appearance helps communicate that the roughness is a visualization encoding rather than a literal topographic surface.

### 2.5 It is cheaper than a dense curved surface

At frequency 6, the current lowpoly prototype generates approximately:

- 36 checkerboard cells per RUM;
- 12 top triangles per cell in the current simple builder;
- about 432 top triangles plus roughly 48 outer-wall triangles per RUM.

This is around **480 triangles per RUM**, before future mesh optimization.

For comparison, a `96 × 96` smooth grid uses approximately **18,432 top triangles per RUM** before walls. The lowpoly path is therefore roughly one to two orders of magnitude lighter.

The current builder is deliberately simple and duplicates some flat vertices. It can be optimized later by sharing the flat grid vertices, but optimization is not required for the first pipeline integration.

---

## 3. Geometry semantics

Each square cap is divided into a checkerboard lattice.

For a frequency of 6:

```text
u d u d u d
 d u d u d u
u d u d u d
 d u d u d u
u d u d u d
 d u d u d u
```

Where:

- `u` = upward pyramid;
- `d` = downward pyramid/cavity;
- both use the same uncertainty magnitude;
- the real RUM surface remains the zero/reference plane between pyramids.

The alternating sign is assigned with:

```js
const signValue = ((i + j) % 2 === 0) ? 1.0 : -1.0;
```

The apex height is:

```js
const apexZ = baseZ + signValue * uncertaintyAmplitude;
```

This preserves symmetric `+σ` and `−σ` relief around the real cap height.

---

## 4. Geometry sizing for square RUM caps

The current sandbox uses a square RUM cap and a frequency parameter.

```js
const cells = 6;
const cellSize = rumSize / cells;
const halfBase = cellSize * 0.28;
const pyramidBaseWidth = 2.0 * halfBase; // 0.56 × cellSize
const clearGap = cellSize - pyramidBaseWidth; // 0.44 × cellSize
```

Therefore, for any square RUM cap:

```text
cell size            = RUM width / 6
pyramid base width   = 0.56 × cell size
clear gap per cell   = 0.44 × cell size
```

The base width and spacing automatically scale with the RUM size.

### Recommended implementation constants

```js
const CHECKERBOARD_FREQUENCY = 6;
const PYRAMID_HALF_BASE_RATIO = 0.28;
```

Do not hard-code Jakarta-specific pyramid dimensions in metres. Derive them from each cap or parcel-local footprint.

---

## 5. Amplitude and uncertainty mapping

The lowpoly alternative does **not** use the sandbox `linear / sqrt / log` amplitude-remapping modes.

The geometry should use the chosen sigma range directly:

```js
const displayedSigma = sigma * sigmaMultiplier; // 1σ or 2σ
const uncertaintyAmplitude = displayedSigma * verticalDisplayScale;
```

In the sandbox, this was represented by:

```js
function reliefAmpM(row, sigmaMm) {
  const displaySigma = sigmaMm * STATE.sigmaMultiplier;
  return (
    displaySigma *
    CONFIG.MODEL_DISPLAY_M_PER_MM *
    STATE.gain *
    row.gain
  );
}
```

For the real viewer, this formula must be replaced or aligned with the existing vertical-deformation/exaggeration contract. Avoid creating an unrelated second exaggeration system unless the project deliberately decides to expose one.

### Open decision: vertical exaggeration linkage

The intended direction is:

```text
one real-viewer vertical exaggeration control
→ affects the main vertical deformation
→ also scales the uncertainty relief
→ optional internal uncertainty gain may remain as a fixed calibration constant
```

Possible integration formula:

```js
const modelHeight = displacement * verticalExaggeration;
const uncertaintyHeight = sigma * sigmaMultiplier
                        * verticalExaggeration
                        * uncertaintyReliefGain;
```

`uncertaintyReliefGain` should be treated as an internal calibration parameter unless user testing shows a need to expose it.

### Open decision: 1σ versus 2σ

Keep support for both until the scientific/user-interface decision is made:

```js
const sigmaMultiplier = sigmaMode === "2sigma" ? 2.0 : 1.0;
```

Do not bake one choice irreversibly into the pipeline data.

---

## 6. Lowpoly geometry builder

The selected row uses a dedicated geometry path:

```js
function buildCapGeometry(row, sigmaMm) {
  if (row.type === "lowpoly_cb_spikes") {
    return buildLowpolyCbSpikesGeometry(row, sigmaMm);
  }

  // Other smooth alternatives use the tessellated height-field path.
  return buildSmoothCapGeometry(row, sigmaMm);
}
```

This separation is essential. The lowpoly pyramid shape must not change when the smooth-cap tessellation setting changes.

### Core pyramid construction

Simplified from the v16 sandbox:

```js
function buildLowpolyCbSpikesGeometry({ width, height, baseZ, amplitude }) {
  const cells = 6;
  const cellW = width / cells;
  const cellH = height / cells;

  // Square RUM case. For rectangles, use the smaller cell dimension
  // for a square pyramid footprint unless a rectangular footprint is desired.
  const halfBase = Math.min(cellW, cellH) * 0.28;

  for (let j = 0; j < cells; j++) {
    for (let i = 0; i < cells; i++) {
      const cx = -width / 2 + (i + 0.5) * cellW;
      const cy = -height / 2 + (j + 0.5) * cellH;

      const signValue = ((i + j) % 2 === 0) ? 1.0 : -1.0;
      const apex = [cx, cy, baseZ + signValue * amplitude];

      const sw = [cx - halfBase, cy - halfBase, baseZ];
      const se = [cx + halfBase, cy - halfBase, baseZ];
      const ne = [cx + halfBase, cy + halfBase, baseZ];
      const nw = [cx - halfBase, cy + halfBase, baseZ];

      addTriangle(sw, se, apex);
      addTriangle(se, ne, apex);
      addTriangle(ne, nw, apex);
      addTriangle(nw, sw, apex);

      // Keep the remaining cell area flat at baseZ.
      addFlatRingAroundPyramidFootprint(...);
    }
  }
}
```

The selected shape is a **four-sided square pyramid**, not a smooth cone and not a triangular pyramid.

---

## 7. Required per-vertex attributes

The successful v13–v16 architecture uses the actual lowpoly geometry as the only source of truth for shading.

Do not reimplement the checkerboard shape analytically in the fragment shader.

Recommended attributes:

```text
position       actual pyramid/cap vertex position
normal         real face normal
slope          1 − normal.z
reliefValue    signed normalized relief: −1 at downward apex, +1 at upward apex, 0 at base
surfaceMask    1 for cap top, 0 for walls
```

Curvature can remain available for experiments, but it is not required for the selected neutral-slope mode.

### Face normals

Each pyramid face has one flat normal:

```js
function triangleNormal(a, b, c) {
  const u = subtract(b, a);
  const v = subtract(c, a);
  let n = normalize(cross(u, v));

  if (n.z < 0) n = negate(n);
  return n;
}
```

Use the same normal for the three vertices of one face to preserve the faceted lowpoly appearance.

---

## 8. Selected rendering cue: v16 Neutral slope shade

The selected shader is the **v16 neutral slope shade**, at **100% cue strength**.

Despite the name, the lowpoly branch is not a simple uniform steepness darkening. It combines:

1. the real face normal;
2. a fixed light direction in the cap/model frame;
3. a relief-presence gradient toward the apex;
4. a slope gate that guarantees `no spike = no shade`.

### Selected GLSL logic

```glsl
vec3 neutralLightMC = normalize(vec3(-0.36, 0.28, 0.89));

float presence = smoothstep(
    0.035,
    0.90,
    abs(v_reliefValue)
);

float shapeGate = smoothstep(
    0.003,
    0.065,
    clamp(v_slope, 0.0, 1.0)
);

float ndl = clamp(
    dot(normalMC, neutralLightMC) * 0.5 + 0.5,
    0.0,
    1.0
);

float facetedFactor = mix(0.87, 1.06, ndl);

factor = mix(
    1.0,
    facetedFactor,
    cueStrength * presence * shapeGate
);
```

Selected value:

```js
cueStrength = 1.0; // 100%
```

### Why this shader was selected

- Flat/no-uncertainty cells remain unshaded.
- Different pyramid faces receive different brightness, preserving the 3D faceted appearance.
- Shading increases toward the apex because `v_reliefValue` is interpolated from zero at the footprint to `±1` at the apex.
- Low-amplitude pyramids remain visible.
- The neutral base cap remains readable between pyramid footprints.
- It avoids the permanently painted checkerboard problem found in the earlier curvature mode.

### Important rendering rule

The cue may modify brightness/value, but must not alter the hue mapping used for velocity.

```glsl
finalRgb = clamp(baseRgb * factor, vec3(0.0), vec3(1.0));
finalAlpha = globalOpacity;
```

Do not use the relief cue to modify alpha.

---

## 9. Appearance/render-state requirements

The successful sandbox architecture uses a custom `Cesium.Appearance` for raw `Primitive + Geometry` rendering.

Required behavior:

```text
- use real geometry normals;
- do not enable Cesium Phong/PBR lighting on top;
- depth test enabled;
- opaque depth writing at full opacity;
- blending only when global opacity is below 1;
- back-face culling disabled for the experimental relief surface;
- cap shading must not leak onto the outer RUM walls.
```

The `surfaceMask` distinguishes top geometry from walls:

```glsl
float topMask = step(0.5, v_surfaceMask);
```

The cap relief cue only runs when `topMask > 0.5`.

### Do not regress to the old duplicated shader architecture

Rejected architecture:

```text
CPU builds one geometry shape
+
fragment shader independently reconstructs the same shape from UV coordinates
```

This caused shading/geometry mismatch and made every new alternative require two implementations.

Selected architecture:

```text
CPU/pipeline builds geometry and normals once
→ shader consumes the real normal/relief attributes
→ silhouette and shading always match
```

---

## 10. Prototype 1 RUM integration strategy

### Pipeline responsibility

The Prototype 1 data/geometry pipeline should generate:

1. the RUM cap footprint;
2. a 6 × 6 checkerboard lattice in the cap-local plane;
3. four-sided up/down pyramids;
4. flat regions between pyramid footprints;
5. stable outer walls/skirt;
6. real face normals;
7. signed relief and top/wall mask attributes;
8. sufficiently large tile/model bounding volumes for the maximum displayed relief.

### Viewer responsibility

The viewer should control:

```text
- current epoch/model displacement;
- vertical exaggeration;
- 1σ/2σ selection;
- uncertainty visibility toggle;
- global opacity;
- velocity color;
- selected neutral faceted shading.
```

### Geometry update options

Two implementation routes are possible.

#### Route A — pipeline-generated final pyramid geometry

Generate the pyramid vertices in the pipeline at a normalized or reference amplitude, then displace the apex vertices in the viewer shader.

Advantages:

- low runtime geometry-generation cost;
- static topology;
- efficient uniform/texture-driven amplitude updates.

Requirements:

- tag apex/base vertices or store a signed relief attribute;
- expand bounding volumes for maximum possible exaggeration;
- recalculate/derive normals if vertex displacement changes face slopes substantially.

#### Route B — rebuild geometry when settings change

Generate the final pyramid geometry in JavaScript whenever sigma range or exaggeration changes.

Advantages:

- straightforward;
- real normals always match final geometry.

Disadvantages:

- rebuilding thousands of RUM meshes may be expensive;
- less suitable for animated/interactive updates.

**Recommended direction:** static lowpoly topology with signed relief attributes, plus GPU displacement and a normal strategy designed for the final displaced pyramid faces. However, the pipeline/viewer team should benchmark this against precomputed geometry because correct normals are central to the selected cue.

---

## 11. Parcel integration strategy

The visual concept is intended to transfer to parcel-based products.

The parcel surface remains one polygonal cap, but the checkerboard pattern is generated in a local 2D parcel coordinate system.

### Recommended parcel-local workflow

1. Triangulate the parcel polygon for its flat base.
2. Establish a local 2D coordinate frame:
   - centroid as origin;
   - principal axes/PCA or oriented bounding box for orientation;
   - local east/north axes if consistent geographic orientation is preferred.
3. Generate a checkerboard lattice using a **target physical spacing**, not necessarily exactly six cells across every parcel.
4. Place a pyramid only when its centre and complete footprint are inside the parcel polygon.
5. Leave boundary fragments flat rather than clipping malformed partial pyramids.
6. Alternate pyramid sign using lattice indices, not polygon triangle indices.

### Recommended spacing rule for parcels

For RUMs, `frequency = 6` is the chosen layout. For differently sized parcels, convert this to a physical target spacing derived from the reference RUM:

```js
const targetCellSize = referenceRumWidth / 6;

const cols = Math.max(1, Math.floor(parcelLocalWidth  / targetCellSize));
const rows = Math.max(1, Math.floor(parcelLocalHeight / targetCellSize));
```

Then:

```js
const cellW = parcelLocalWidth / cols;
const cellH = parcelLocalHeight / rows;
const halfBase = Math.min(cellW, cellH) * 0.28;
```

This keeps pyramid sizes visually comparable across datasets while adapting the count to parcel dimensions.

### Boundary policy

Preferred initial rule:

```text
full pyramid footprint inside parcel → include pyramid
partial footprint at parcel edge     → leave area flat
```

This is simpler and more legible than clipping pyramids into irregular edge fragments.

### Very small parcels

If a parcel cannot contain at least one complete pyramid plus a visible flat margin:

```text
fallback to flat cap or a single centred pyramid
```

The exact threshold should be documented and tested.

---

## 12. Data semantics and legend wording

Recommended legend language:

```text
Vertical uncertainty
Lowpoly checkerboard spikes
Alternating upward/downward pyramid height represents ±σ.
CB = checkerboard.
```

If 1σ/2σ is selectable, display it explicitly:

```text
Relief range: ±1σ
```

or

```text
Relief range: ±2σ
```

Avoid describing the pyramids as terrain, roughness measurements, or physical surface texture.

---

## 13. Required validation tests

Before integrating into the full viewer, verify the following.

### Geometry correctness

- Zero uncertainty produces a completely flat, unshaded cap.
- Upward and downward amplitudes are equal and symmetric.
- The real cap reference height remains fixed.
- The RUM outer walls remain stable.
- Pyramid shape is independent of smooth tessellation settings.

### Shader correctness

- No uncertainty produces no cue.
- Cue strength changes RGB only, not alpha.
- Velocity hue remains recognizable at 100% cue strength.
- Shading remains fixed in the local/model frame when the camera orbits.
- Different faces remain distinguishable at oblique, grazing, and top-down views.

### Scale and zoom

- Low sigma remains perceptible without becoming a false strong signal.
- High sigma does not erase the base velocity color.
- Pattern remains detectable at the intended far zoom.
- Pattern does not alias or flicker during camera movement.

### Performance

Test at minimum:

```text
100 RUMs
1,000 RUMs
~5,000 RUMs
```

Measure:

```text
initial load time
GPU memory
frame rate while orbiting
frame rate during epoch animation
cost of changing vertical exaggeration
cost of switching 1σ/2σ
```

### Parcel transfer

- checkerboard orientation is stable;
- pyramid size is consistent across differently sized parcels;
- boundary handling does not create clipped spikes;
- small-parcel fallback is predictable.

---

## 14. Known open decisions

The following items are intentionally unresolved and should be discussed during integration.

### 14.1 Vertical exaggeration contract

Decide whether uncertainty relief uses:

```text
A. exactly the same exaggeration as the main vertical displacement;
B. the same exaggeration plus a fixed internal gain;
C. a separate expert-only uncertainty gain.
```

Current preference: **one visible viewer knob**, with an internal calibration factor if required.

### 14.2 Sigma range

Decide the default and exposed options:

```text
1σ only
2σ only
1σ / 2σ toggle
```

Current sandbox supports both.

### 14.3 Runtime versus pipeline displacement

Decide whether apex heights are:

```text
precomputed in B3DM/glTF geometry
or
displaced dynamically in the vertex shader
```

This decision must account for normals, bounding volumes, epoch animation, and performance.

### 14.4 Pattern density across datasets

For the current square RUM:

```text
6 × 6 checkerboard cells
```

For parcels and future shapes, use a physical target spacing derived from the reference RUM rather than blindly forcing six cells across every object.

---

## 15. Do-not-regress checklist

Do not:

- smooth the lowpoly pyramids through the normal tessellation selector;
- reconstruct the checkerboard shape independently in GLSL;
- use alpha changes as a shading cue;
- allow zero-uncertainty cells to retain checkerboard light/shadow;
- remove the flat gaps between pyramid footprints;
- let uncertainty relief move the real cap reference plane;
- let the outer sidewall height follow individual pyramid peaks/troughs;
- apply Cesium global `shadowMap` for the micro-relief;
- let PBR/environment lighting unpredictably alter the velocity color;
- hard-code Jakarta-specific pyramid dimensions.

---

## 16. Minimum files needed by the implementation chat

Provide the next implementation chat with:

```text
dimple_sandbox_proto1_clean_v16.html
current Prototype1 viewer HTML
current cap/wall B3DM generation script(s)
current custom cap and wall shader code
height_meta.json
viewer_tuning.json
one representative generated RUM tile / glTF / B3DM
parcel geometry sample when available
```

The most important source reference is the v16 lowpoly builder plus the v16 neutral-slope branch.

---

## 17. Compact handoff summary

```text
Selected encoding:
Lowpoly CB spikes (CB = checkerboard)

Geometry:
6 × 6 alternating up/down four-sided pyramids per square RUM
Pyramid base width = 0.56 × checkerboard cell size
Flat gaps remain around each pyramid
Reference cap height stays fixed

Amplitude:
±sigma around the reference cap
No linear/sqrt/log shape remapping
1σ/2σ and vertical-exaggeration linkage still to be finalized

Rendering:
v16 neutral slope shade
100% cue strength
Real face normals + fixed local-frame light
Presence and slope gates guarantee no spike = no shade
RGB brightness only; alpha unchanged

Architecture:
Dedicated lowpoly geometry path independent of tessellation
Single source of truth: geometry and real normals
Do not duplicate shape formulas in GLSL

Transfer:
RUM first
Parcels use a local 2D lattice and physical target cell spacing
Only place complete pyramid footprints inside parcel boundaries
```
