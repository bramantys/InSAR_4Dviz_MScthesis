# Performance Batch C — Flat Parent / Lowpoly Child LOD

## Scope

This batch changes the real RUM cap tileset from one fixed lowpoly level into a two-level `REPLACE` hierarchy:

```text
flat animated cap parent
└── 6 × 6 lowpoly checkerboard child
```

No vertical-sigma values, p98 clipping, model displacement, velocity colour, horizontal field, horizontal particles, static arrows, or confidence ellipses were changed.

## Generated geometry

Across 4,779 real RUMs:

| Level | Vertices | Triangles | B3DM size |
|---|---:|---:|---:|
| Flat parents | 19,116 | 9,558 | 1.59 MB |
| Lowpoly children | 2,064,528 | 3,096,792 | 107.14 MB |

There are 48 spatial parent tiles and 48 corresponding lowpoly child tiles.

## Tileset contract

Each spatial tile has:

```json
{
  "geometricError": 100.0,
  "refine": "REPLACE",
  "content": { "uri": "flat_tile_rXX_cYY.b3dm" },
  "children": [
    {
      "geometricError": 0.0,
      "content": { "uri": "tile_rXX_cYY.b3dm" }
    }
  ]
}
```

Both levels preserve the same B3DM batch-table picking properties.

## Flat-parent shader marker

`TEXCOORD_0.x = 99` is reserved for flat LOD-parent vertices.

The viewer converts that to a zero relief role and sets `v_reliefSurface = 0`. This lets every flat-parent fragment uniformly bypass derivative-normal and relief-lighting work while retaining:

- animated model displacement;
- velocity colour;
- global opacity;
- RUM picking.

Lowpoly children retain the existing `-4..+4` role contract and `v_reliefSurface = 1`.

## Viewer LOD controls

Generated defaults:

```text
AUTO / uncertainty ON SSE  = 4.0
uncertainty OFF SSE        = 100.0
force LOWPOLY SSE          = 0.1
parent geometric error     = 100 m
```

UI behavior:

```text
Uncertainty relief ON  → AUTO: flat far, lowpoly near
Uncertainty relief OFF → hold flat parents
```

Developer controls:

```js
__vuncLod.setMode("AUTO")
__vuncLod.setMode("FLAT")
__vuncLod.setMode("LOWPOLY")
__vuncLod.report()
```

## Required FPS test

Use resolution scale 1.0, opacity 1, Auto render pass, and the same camera for every comparison.

### City view (~30–35 km)

1. `LOWPOLY` — reproduces the previous expensive behavior.
2. `FLAT` — measures the maximum expected far-view gain.
3. `AUTO` — should be close to `FLAT` at this distance.

### Close view (~1.7–2.0 km)

1. `FLAT` — confirms the lightweight parent.
2. `LOWPOLY` — confirms the detailed child.
3. `AUTO` — should visually match `LOWPOLY` and have similar FPS.

Also verify:

- zooming from city to close swaps from flat to spikes;
- zooming back swaps to flat;
- no parent and child are simultaneously visible during transition;
- RUM picking works at both levels;
- epoch playback and vertical exaggeration work at both levels;
- uncertainty OFF remains flat even close up;
- uncertainty ON restores automatic refinement.

## Validation completed

- All 18 pipeline stages regenerated successfully when executed directly in sequence.
- All Python pipeline scripts compile.
- Viewer JavaScript passes Node syntax validation.
- 48/48 flat parents and 48/48 lowpoly children exist.
- Every parent uses `REPLACE`; every child has geometric error 0.
- Flat parent marker verified as `99` in B3DM `TEXCOORD_0.x`.
- Lowpoly roles verified as `-4..+4`.
- Horizontal field, particle field, horizontal uncertainty report, arrow tileset, ellipse tileset, and height texture are byte-identical to the previous build.

## Current limitation

Cesium is intentionally absent from the supplied template, so actual LOD selection and FPS must be verified in the user's local browser. The `AUTO`, `FLAT`, and `LOWPOLY` modes are included specifically to make that diagnosis unambiguous.
