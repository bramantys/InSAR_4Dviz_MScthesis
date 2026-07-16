# Proto1 DeckGL — Jakarta Focus Pit

A DeckGL + MapLibre implementation of the Proto1 RUM viewer that makes negative vertical deformation read as a **subsidence pit below a datum** rather than as geometry floating above the city.

## The visual contract

- Real Jakarta RUM centres and exact four-corner footprints.
- One cap per RUM; no centre-square approximation.
- One canonical step wall per shared E/N RUM edge; no closed per-RUM boxes and no duplicate internal faces.
- A rectangular **focus pit** is automatically centred on the strongest negative `up` velocity in the north-west sector of the Jakarta grid (Soekarno–Hatta area).
- A wide, horizontal datum apron remains at z = 0 around that focus window.
- In default **See-through** mode the apron has alpha 0 but is forced to write depth. The genuine MapLibre OSM basemap shows through, while the outer faces of the rim walls are hidden. This is the Minecraft-crater grammar.

The datum apron is a visual reference and occlusion device. It is **not** a measured topographic surface or a geological fault/scarp.

## First run

1. Copy `jakarta_enu_estimates.csv` into `data/`.
2. Double-click `RUN_VIEWER.bat`.

Or in a terminal from this folder:

```powershell
npm install
npm run dev
```

`npm run dev` automatically rebuilds the runtime assets first. It outputs:

```text
public/data/jakarta/
├── manifest.json
├── rum_static.json
├── epoch_axis.json
└── vertical_model_mm_f32.bin
```

The package intentionally excludes `node_modules/`, raw source data, generated runtime data and `dist/`.

## Controls

- Left drag: pan
- Middle drag: heading and tilt
- Wheel: zoom
- Right drag: zoom
- Epoch slider / Play: temporal navigation
- Vertical exaggeration: V7.1 convention, **metres per millimetre**. Default `10×`; maximum `20×`.
- Ground datum apron:
  - **See-through**: default. Alpha-zero, depth-writing apron; basemap stays visible.
  - **Solid**: debug fallback; neutral apron replaces basemap but preserves depth occlusion.
  - **Off**: reproduces the old exposed-rim / fence failure mode.

## Focus-window configuration

Edit `config/project_config.json`, then run the viewer again:

```json
"pit_mode": {
  "focus_selection": "northwest_max_subsidence",
  "focus_width_cells": 34,
  "focus_height_cells": 28,
  "apron_margin_cells": 30,
  "default_apron_mode": "see-through"
}
```

`30` margin cells equals `13.5 km` around the focus window. Increase it only if rim walls become visible beneath the apron during very low camera views.

## Architecture

The data pipeline only creates the compact runtime data. It does not create terrain tiles, GLBs, B3DMs or a second imagery layer.

```text
CSV
→ scripts/build_jakarta_assets.mjs
→ static RUM metadata + exact corners
→ rum-major Float32 epochs
→ DeckGL cap / shared-wall / rim-wall / depth-apron layers
```

The active runtime uses linear displacement derived from the real `up` velocity column. It is an architecture visualisation test, not a claim that this is a measured epoch series.

## Build deployment

Run `BUILD_RELEASE.bat` or:

```powershell
npm run build
```

Vite writes the static deployment package to `dist/`.
