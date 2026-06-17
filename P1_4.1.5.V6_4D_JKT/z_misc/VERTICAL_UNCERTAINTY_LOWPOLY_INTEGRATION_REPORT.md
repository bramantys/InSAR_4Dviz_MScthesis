# Vertical Uncertainty Lowpoly Integration Report

## Status

The four implementation batches are complete in this package.

- Final pipeline status: **SUCCESS_WITH_WARNINGS**
- Clean end-to-end runtime: **72.76 s**
- Python pipeline scripts compiled: **20**
- Viewer inline JavaScript syntax: **passed**
- Active horizontal particle engine syntax: **passed**

## Implemented contract

- Encoding: **6 × 6 lowpoly checkerboard spikes per square RUM**
- Shape: alternating upward/downward four-sided truncated-pyramid-capable cells
- Pyramid half-base ratio: **0.28** of the checker cell width
- Current Jakarta half-base: **21.00 m**
- Cue: **neutral slope shade, 100%**
- Vertical linkage: the same real-viewer vertical exaggeration scales MODEL height and uncertainty relief
- Sigma mode: **1σ** internal multiplier, retained as metadata for future adjustment
- Blank/no-data caps: remain flat
- Existing separate RUM walls: unchanged; no duplicate skirts were added to caps

## Batch 1 — synthetic epoch uncertainty and RGB data texture

Step 04 now generates an explicitly synthetic visualization-test uncertainty field using:

1. the source vertical-velocity sigma as the persistent spatial baseline;
2. a modest topology multiplier for edge/isolated cells;
3. smooth spatially varying periodic components;
4. three broad regional rise/recovery episodes;
5. one compact temporary quality-drop episode to exercise the above-p98 encoding.

MODEL and MEASUREMENT are verified unchanged by Step 04.

The height texture remains **RGB**:

- R + G: MODEL displacement
- B: full raw vertical sigma
- A: unused/reserved

Texture dimensions: **114 × 5077**, mode **RGB**.

- Raw sigma storage maximum: **5.3110 mm**
- Fixed real-RUM global p98 display ceiling: **2.2032 mm**
- Sigma texture clipping: **0 pixels**
- Maximum sampled 8-bit decode error: **0.01041 mm**
- Mean sampled decode error: **0.00519 mm**

## Batch 2 — p98 truncation and lowpoly B3DM geometry

The viewer computes:

```text
relief height = min(raw sigma, global p98)
plateau ratio = max(0, 1 - global p98 / raw sigma)
```

Exactly **2.00%** of real RUM/epoch sigma values exceed p98.
The strongest outlier produces a plateau-side ratio of **58.52%**.

Production geometry per RUM:

- 36 checker cells
- 12 vertices per cell
- 18 triangles per cell
- **432 vertices and 648 triangles per RUM**

Whole real-cap set:

- vertices: **2,064,528**
- triangles: **3,096,792**
- B3DM size: **107.14 MB**
- previous flat-cap size: **1.59 MB**
- size increase: **67.5×**
- average new cap geometry: **23.0 KB per RUM**

Step 11 generated the complete real-cap set in approximately **11.5 s** during the final clean pipeline run.

## Batch 3 — viewer shader and UI

The real-cap custom shader now:

- samples current-epoch raw vertical sigma from B;
- applies the fixed p98 height ceiling;
- expands top vertices into square plateaus above p98;
- alternates upward/downward checkerboard signs;
- derives face normals from the actually deformed surface using fragment derivatives;
- applies bounded neutral-slope shading without changing velocity hue semantics;
- follows the existing vertical-exaggeration control;
- replaces the former SNR hatch toggle with **Uncertainty relief**.

The legend now distinguishes pointed, p98-tip, and flat-top states.

## Batch 4 — validation and compatibility

Horizontal uncertainty was not modified. The following regenerated outputs are byte-identical to the supplied baseline:

- `horizontal_field.json`: **True**
- `horizontal_particle_field.json`: **True**
- `horizontal_uncertainty_check.json`: **True**

Expected pipeline warnings only:

- no `expected_rum_count` was supplied;
- 201 near-zero horizontal vectors were skipped in unit-vector validation.

## Modified source files

- `_internal/pipeline/pipeline_config.py`
- `_internal/pipeline/04_enhance_vertical_sigma.py`
- `_internal/pipeline/10_build_height_texture.py`
- `_internal/pipeline/11_build_real_caps_b3dm.py`
- `config/project_config.json`
- `viz1_dev_v4_dualmode.html`

## Remaining visual check

The package intentionally omits the heavy `_internal/cesium` engine. Therefore the full browser render could not be launched in this environment. The data pipeline, B3DM contract, RGB decoding, JavaScript syntax, shader wiring, output counts, and clean end-to-end build were validated. The first local browser run with Cesium restored should specifically inspect:

- pyramid/plateau orientation in 2D and 3D;
- neutral-slope shade direction;
- relief response to epoch and vertical-exaggeration changes;
- selected-RUM highlighting and opacity;
- cap/wall/particle occlusion.
