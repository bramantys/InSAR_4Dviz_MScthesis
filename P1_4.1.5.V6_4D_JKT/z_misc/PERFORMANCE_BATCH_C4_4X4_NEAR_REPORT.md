# Performance Batch C4 — 4×4 Near Relief with Fixed 6×6 Pyramid Footprint

## Purpose

Batch C4 tests whether reducing the close-range checkerboard from 6×6 to 4×4 improves steady FPS, playback, and camera-motion smoothness without enlarging individual pyramids.

The semantic LOD hierarchy is now:

```text
Uncertainty ON, far:  2×2 relief
Uncertainty ON, near: 4×4 relief
Uncertainty OFF:      separate true-flat cap tileset
```

The previous C3 near level was 6×6.

## Critical geometry decision

The 4×4 grid does **not** use ordinary 4×4-sized pyramids. Pyramid footprint is referenced to the former 6×6 design:

```text
RUM width                         = 450 m
4×4 centre spacing                = 112.5 m
old 6×6 pyramid full base width   = 42.0 m
new 4×4 pyramid full base width   = 42.0 m
neutral gap between 4×4 bases     = 70.5 m
```

Configuration:

```json
"checkerboard_frequency_near": 4,
"pyramid_footprint_reference_frequency_near": 6,
"pyramid_half_base_ratio": 0.28
```

This makes the relief sparser while retaining the selected small-pyramid visual character.

## Measured geometry reduction

| Near level | Vertices | Triangles | B3DM size |
|---|---:|---:|---:|
| C3 6×6 | 2,064,528 | 3,096,792 | 107.14 MiB |
| C4 4×4 | 917,568 | 1,376,352 | 48.08 MiB |

Reduction:

- vertices: 55.6%
- triangles: 55.6%
- near-level B3DM bytes: 55.1%

The far 2×2 product remains 12.64 MiB and is unchanged.

## Unchanged scientific/display contract

- vertical relief visibility threshold remains global p50;
- Jakarta resolved threshold remains 0.6484 mm;
- p98 height clipping and progressive square plateaus remain unchanged;
- horizontal arrow/ellipse filter remains 1σ;
- RGB texture packing remains unchanged;
- uncertainty OFF still switches to a separate true-flat cap product;
- epoch animation, picking, walls, particles, and trendline data are unchanged.

## Local comparison test

Use the same camera and startup settings for C3 and C4.

Record:

1. 2D idle FPS;
2. 3D idle FPS;
3. 3D playback FPS;
4. camera-motion smoothness while crossing LOD transitions;
5. near-view readability at 10× and 20× exaggeration;
6. whether the larger neutral gaps improve or weaken uncertainty perception.

Useful controls:

```js
__vuncLod.setMode("AUTO")
__vuncLod.setMode("LOWPOLY")  // forces the 4×4 near children
__vuncLod.setMode("FLAT")
__vuncLod.report()
```

## Success criterion

C4 is preferable if it materially improves playback/camera feel and the 4×4 relief still communicates elevated vertical uncertainty without appearing too sparse.

## Validation

- Python pipeline files compile;
- all 18 stages completed when run directly in sequence;
- 48 far, 48 near, and 48 flat B3DM files exist;
- all B3DM headers and byte lengths validate;
- tileset hierarchy is `2×2 parent → 4×4 child` with `REPLACE` refinement;
- viewer inline JavaScript passes `node --check`.
