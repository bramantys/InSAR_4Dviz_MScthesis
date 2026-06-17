# Prototype 2 V3.1

## Krimpenerwaard Parcel-Based 4D Seasonal Deformation Viewer

**Release date:** 17 June 2026  
**Author:** R. Bramantya (Ridan Bramantya)  
**Research context:** MSc thesis work in progress at TU Delft  
**Current thesis title:** *Spatiotemporal Visualization of InSAR Ground Deformation*

Prototype 2 V3.1 is a research pipeline and browser-based viewer for parcel-level vertical ground-deformation time series. The supplied Krimpenerwaard case visualizes seasonal reversible movement together with gradual irreversible subsidence: **breathing while drowning**.

This release is a thesis prototype. It is intended for research, visualization experiments, and expert review—not for operational monitoring or safety-critical decisions.

For the shortest run instructions, read [`README.txt`](README.txt).

---

## Quick start

1. Install Python 3.12, Visual Studio Code, the VS Code Python extension, and the VS Code Live Server extension.
2. Install the Python dependencies:

   ```powershell
   python -m pip install numpy pandas geopandas pyarrow shapely mapbox-earcut
   ```

3. Place the required input files under `data/`.
4. Open `config/project_config.json` in VS Code and update the input paths.
5. Save the configuration.
6. In Windows File Explorer, double-click `config/run_baby_run.bat`.
7. Confirm that the newest receipt in `run_records/` reports `PASS`.
8. In VS Code, right-click `viz2_dev_v11.html` and select **Open with Live Server**.

CesiumJS and Three.js are already bundled. Node.js and npm are not required.

---

## 1. What Prototype 2 visualizes

The viewer separates parcel deformation into the following components:

- **Irreversible:** accumulated permanent subsidence—the slowly sinking floor.
- **Reversible:** seasonal swelling or shrinkage around a datum—the breathing motion.
- **Total:** reversible plus irreversible displacement shown as one moving surface.
- **Combined:** the irreversible parcel body together with the reversible layer riding on it; the moving upper cap represents total displacement.

The current supplied case uses irregular parcel polygons in Krimpenerwaard and daily displacement epochs for 2025.

### Interpretation concept

The central visual metaphor is **breathing while drowning**:

- the irreversible component gradually lowers the parcel baseline;
- the reversible component moves seasonally above and below that evolving baseline;
- the combined mode allows both processes to be inspected at the same time.

Vertical movement is visually exaggerated because millimetre-scale deformation would otherwise be invisible at landscape scale. Exaggeration affects only rendering; it does not change the stored displacement values.

---

## 2. Supported platform and software

### Primary supported setup

Prototype 2 V3.1 is documented for:

- Windows 10 or 11;
- Python 3.12;
- Visual Studio Code;
- a modern WebGL-capable browser, preferably Chrome or Edge.

The pipeline itself is Python-based. The provided user launcher is a Windows batch file, so other operating systems require running the Python scripts manually and are not covered by this release guide.

### VS Code extensions

Install:

- **Python**, for working with the configuration and Python environment;
- **Live Server**, for serving the generated viewer over local HTTP.

This guide assumes the project is opened as a complete folder in VS Code.

### Python dependencies

Install these packages in the Python environment used by the launcher:

```powershell
python -m pip install numpy pandas geopandas pyarrow shapely mapbox-earcut
```

The preflight script checks for:

```text
numpy
pandas
geopandas
pyarrow
shapely
mapbox_earcut
```

The installation package is named `mapbox-earcut`, while the imported Python module is `mapbox_earcut`.

### Bundled JavaScript engines

CesiumJS and Three.js are included locally under `_internal`. Users do **not** need to install:

- CesiumJS;
- Three.js;
- Node.js;
- npm.

An internet connection is still needed for the online background map layers. The parcel geometry, time-series products, CesiumJS engine, and Three.js engine are local project assets.

### Hardware guidance

No strict minimum hardware specification has been established for this research prototype. For the full Krimpenerwaard case, a practical setup is:

- 16 GB RAM recommended;
- a modern integrated or dedicated GPU with WebGL support;
- sufficient free disk space for the source CSV and generated runtime products.

---

## 3. Project folder structure

The user-facing project structure is expected to resemble:

```text
Prototype2/
├── README.txt
├── README.md
├── viz2_dev_v11.html
│
├── config/
│   ├── project_config.json
│   ├── viewer_tuning.json
│   └── run_baby_run.bat
│
├── data/
│   ├── displacement/
│   │   └── example_spams_model_2025.csv
│   ├── shapefile/
│   │   ├── krimpenerwaard_attributes_wgs84.shp
│   │   ├── krimpenerwaard_attributes_wgs84.dbf
│   │   ├── krimpenerwaard_attributes_wgs84.shx
│   │   ├── krimpenerwaard_attributes_wgs84.prj
│   │   ├── krimpenerwaard_attributes_wgs84.cpg
│   │   └── krimpenerwaard_attributes_wgs84.qmd
│   ├── model_params/
│   │   ├── nl_krimpenerwaard_spams10.parquet
│   │   └── nl_krimpenerwaard_spams10.json
│   └── pyspams/
│       ├── utils.py
│       └── spams_main.py
│
├── _internal/
│   ├── assets/
│   ├── cesium/
│   ├── three/
│   ├── pipeline/
│   ├── templates/
│   └── data_pipeline/
│
└── run_records/
```

Do not rename, move, or delete `_internal`. The viewer and pipeline use relative paths into this directory.

---

## 4. Required input data

Prototype 2 V3.1 requires five declared input components:

| Input | Role in V3.1 |
|---|---|
| Precomputed displacement CSV | **Active deformation source** used to construct all time-dependent arrays |
| Parcel shapefile and sidecars | Parcel geometry, parcel IDs, and parcel attributes |
| SPAMS parameter parquet | Model-parameter identity and cross-validation support |
| SPAMS metadata JSON | Metadata associated with the SPAMS parameter set |
| PySPAMS directory | Model-engine provenance and future integration; required by the input contract |

> **Important:** V3.1 reads deformation from the precomputed displacement CSV. It does not run PySPAMS automatically. The parquet, JSON, and PySPAMS folder are nevertheless required by the V3.1 preflight and source-contract checks.

### 4.1 Active deformation source

Keep the following setting in `config/project_config.json`:

```json
"pipeline_source": {
  "deformation_source": "displacement_csv",
  "pyspams_automated": false
}
```

Changing the deformation source is outside the supported V3.1 workflow.

### 4.2 Parcel shapefile

A shapefile is a collection of files, not only one `.shp` file. Keep the following files together with the same base name:

```text
.shp   geometry
.dbf   attributes and parcel identifiers
.shx   geometry index
.prj   coordinate reference system
```

Keep any supplied `.cpg` and `.qmd` files as well.

The current pipeline accepts parcel IDs from common geometry fields such as:

```text
parcel_id
int_id
pnt_id
```

The selected field must contain unique numeric IDs. Every moving parcel in the displacement CSV must match a parcel in the shapefile.

### 4.3 Coordinate reference system

Prototype 2 V3.1 currently supports:

```text
EPSG:4326
```

Both the declared parcel CRS and displacement CRS must be `EPSG:4326`, and the shapefile itself must open as EPSG:4326.

---

## 5. Displacement CSV contract

### 5.1 Required columns

The CSV must contain all of the following columns:

```text
pnt_id
pnt_gid
epoch
reversible
irreversible
h_spams_final
pnt_lat
pnt_lon
vI
std_vI
var_vI
```

### 5.2 Column meanings

| Column | Meaning | Expected unit/type |
|---|---|---|
| `pnt_id` | Parcel ID used to join displacement to geometry and model parameters | Numeric integer-like ID |
| `pnt_gid` | Source/model identifier retained from the SPAMS product | Numeric |
| `epoch` | Date of the displacement sample | Date; `YYYY-MM-DD` recommended |
| `reversible` | Seasonal reversible displacement | mm |
| `irreversible` | Accumulated irreversible displacement | mm |
| `h_spams_final` | Total displacement | mm |
| `pnt_lat` | Parcel/sample latitude | WGS84 decimal degrees |
| `pnt_lon` | Parcel/sample longitude | WGS84 decimal degrees |
| `vI` | Irreversible velocity | mm/year |
| `std_vI` | Standard deviation of irreversible velocity | mm/year |
| `var_vI` | Variance of irreversible velocity | corresponding squared unit |

### 5.3 Required relationships and validation rules

The pipeline checks that:

```text
h_spams_final = reversible + irreversible
```

It also requires:

- one row per `pnt_id` and `epoch`;
- no duplicate parcel-date combinations;
- numeric parcel IDs;
- every displacement parcel exists in the shapefile;
- every displacement parcel exists in the parameter parquet;
- all dates can be parsed;
- the unique epoch sequence is continuous at daily frequency;
- the source is in the declared EPSG:4326 coordinate system.

Use UTF-8 and comma-separated values where possible.

### 5.4 Minimal format example

```csv
pnt_id,pnt_gid,epoch,reversible,irreversible,h_spams_final,pnt_lat,pnt_lon,vI,std_vI,var_vI
1001,2001,2025-01-01,1.250,-0.010,1.240,51.950000,4.730000,-3.500,0.300,0.090
1001,2001,2025-01-02,1.180,-0.020,1.160,51.950000,4.730000,-3.500,0.300,0.090
```

The example IDs are illustrative. Real `pnt_id` values must exist in both the parcel geometry and model-parameter parquet.

### 5.5 Blank parcels

Parcels present in the shapefile but absent from the displacement CSV are retained as **blank parcels**. They provide spatial context and are displayed separately from moving parcels.

Blank parcels mean **no displacement source is available**. They must not be interpreted as measured zero displacement.

---

## 6. Configure the project

Open the complete project folder in VS Code, then open:

```text
config/project_config.json
```

### 6.1 Required user input paths

Update the `user_inputs` block so it points to the actual files:

```json
"user_inputs": {
  "displacement_csv": "data/displacement/example_spams_model_2025.csv",
  "displacement_crs": "EPSG:4326",
  "parcel_shapefile": "data/shapefile/krimpenerwaard_attributes_wgs84.shp",
  "parcel_crs": "EPSG:4326",
  "model_parameters_parquet": "data/model_params/nl_krimpenerwaard_spams10.parquet",
  "model_metadata_json": "data/model_params/nl_krimpenerwaard_spams10.json",
  "pyspams_directory": "data/pyspams"
}
```

Use project-relative paths whenever possible. Forward slashes work well in JSON on Windows.

### 6.2 Project identity

When adapting the project to another case, update any available project metadata such as:

```json
"project": {
  "project_id": "krimpenerwaard_2025",
  "display_name": "Krimpenerwaard Seasonal Parcel Deformation",
  "short_label": "Krimpenerwaard · Parcels",
  "page_title": "Krimpenerwaard Seasonal Parcel Deformation Viewer",
  "export_filename_prefix": "krimpenerwaard_parcel_deformation"
}
```

These values are used in the assembled viewer and exported filenames.

### 6.3 Optional viewer defaults

Where present in the configuration, supported defaults include:

```json
"viewer": {
  "default_deformation_mode": "irreversible",
  "default_view_mode": "3D",
  "default_map_layer": "map",
  "playback_step_ms": 80,
  "language": "en"
}
```

Supported map-layer values:

```text
map
satellite
bw
```

Supported deformation-mode values:

```text
irreversible
reversible
total
combined
```

Save `project_config.json` before running. Invalid JSON—such as a missing comma or extra trailing comma—will stop the preflight check.

### 6.4 Files normal users should not edit

Normal users should not edit:

- `_internal/pipeline/`;
- `_internal/templates/`;
- generated files under `_internal/data_pipeline/`;
- generated binary animation arrays;
- the generated viewer as a substitute for configuration changes.

`config/viewer_tuning.json` contains advanced display settings and should be treated as an expert/developer file unless a specific adjustment is required.

---

## 7. Run the pipeline

### 7.1 Recommended user workflow

1. Save all changes in VS Code.
2. Open the project folder in Windows File Explorer.
3. Double-click:

   ```text
   config/run_baby_run.bat
   ```

4. Keep the command window open while the pipeline runs.
5. Read the final pipeline status.

The launcher performs the user preflight and runs the ordered production stages. The pipeline stops when a required check or stage fails.

### 7.2 What the pipeline checks

The preflight and Phase 0 checks cover:

- selected Python executable;
- required Python packages;
- valid `project_config.json` syntax;
- all required input paths;
- shapefile sidecars;
- required PySPAMS files;
- supported CRS declarations;
- readable and valid parcel geometry;
- unique numeric parcel IDs;
- required displacement columns;
- date parsing;
- geometry and parquet joins;
- continuous daily epochs;
- duplicate parcel-date rows;
- displacement decomposition consistency.

### 7.3 Production stages

A successful V3.1 run executes the following sequence:

```text
00_phase0_sanity_check.py
01_adapt_parcel_displacement.py
02_ingest_parcels.py
03_prepare_parcel_footprints.py
04_triangulate_parcel_caps.py
05_package_animation_arrays.py
06_build_runtime_geometry.py
07_build_lookup_assets.py
08_build_viewer_products.py
91_publish_runtime_products.py --publish
09_assemble_viewer.py --validate-files
99_validate_release.py
```

The pipeline creates or updates:

- animation arrays;
- parcel geometry products;
- pick, search, and trendline lookup products;
- colour scales and viewer metadata;
- the assembled viewer `viz2_dev_v11.html`;
- run receipts.

---

## 8. Read the run receipt

Every run writes a matching pair in:

```text
run_records/
```

Files:

```text
run_<timestamp>.json
run_<timestamp>.txt
```

### 8.1 Successful result

Check the newest JSON receipt. A successful run should report:

```text
status: PASS
failed_stage: null
error: null
viewer.exists: true
```

The receipt also records:

- active deformation source;
- configured user inputs;
- total, moving, and blank parcel counts;
- epoch count and date range;
- generated viewer path and size;
- stages completed.

### 8.2 Failed result

For a failed run:

1. Read `failed_stage` and `error` in the JSON receipt.
2. Open the matching TXT receipt.
3. Search the full console log for `[FAIL]`, `[ERROR]`, or the failed stage name.
4. Correct the input or configuration issue.
5. Run `run_baby_run.bat` again.

Do not assume a viewer is valid after a failed run merely because an older HTML file remains in the folder.

---

## 9. Open the viewer locally

The viewer loads local JSON, GLB, and binary files using browser fetch requests. It must therefore be served through HTTP.

### 9.1 VS Code Live Server

1. Open the complete project folder in VS Code.
2. Find `viz2_dev_v11.html` in the project root.
3. Right-click the file.
4. Select **Open with Live Server**.
5. The browser should open an address similar to:

   ```text
   http://127.0.0.1:5500/viz2_dev_v11.html
   ```

Do not open the viewer directly as a `file://` URL.

### 9.2 Python HTTP server fallback

From the project root:

```powershell
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/viz2_dev_v11.html
```

Stop the server with `Ctrl+C` in the terminal.

### 9.3 After rebuilding

When the browser still displays an older version after a successful run:

- refresh the page;
- use `Ctrl+F5` for a hard refresh;
- confirm that Live Server is serving the correct project folder;
- confirm that the receipt points to the expected viewer.

---

## 10. Viewer controls

### 10.1 Background layers

Open the left control drawer and choose:

- **Map**;
- **Satellite**;
- **B/W**.

These layers use online map services. A missing background with otherwise visible parcel geometry usually indicates an internet or tile-service issue rather than a deformation-data failure.

### 10.2 Parcel deformation modes

The **Parcel deformation** selector provides:

#### Irreversible

- parcel piston height represents irreversible displacement;
- colour can represent irreversible velocity or irreversible displacement;
- use **Irreversible color** to switch between the two colour encodings.

#### Reversible

- the moving cap represents reversible displacement around the datum;
- **Datum reference** shows or hides the reference parcel surface;
- **Breathing walls** show or hide vertical guides between datum and moving cap.

#### Total

- the piston height represents total displacement;
- cap colour represents reversible displacement;
- wall colour represents irreversible velocity.

#### Combined

- the irreversible displacement forms the lower parcel body/cap;
- the reversible layer moves on top of the irreversible baseline;
- the upper moving cap represents total displacement;
- datum and breathing-wall controls remain available for interpreting the decomposition.

### 10.3 Datum reference

The **Datum reference** toggle is relevant mainly in reversible and combined modes:

- ON: shows the reference parcel surface with partial opacity;
- OFF: hides the moving reference surface immediately;
- blank/no-data parcel context remains visible.

### 10.4 Breathing walls

The **Breathing walls** toggle controls the vertical guides used to show the gap between datum/baseline and the moving cap in reversible or combined views.

### 10.5 Vertical exaggeration

The **Vertical exaggeration** slider changes the visual conversion from millimetres of displacement to metres of rendered height.

It does not modify:

- the displacement CSV;
- popup values;
- trendline values;
- scientific units.

Vertical exaggeration is a communication aid. Always read numerical values from the popup or trendline when quantitative interpretation is needed.

### 10.6 Time controls

The lower time panel contains:

- first epoch;
- previous epoch;
- play/pause;
- next epoch;
- last epoch;
- draggable epoch slider;
- active date and epoch information.

The animated time controls are available in 3D mode. The viewer displays a lock note when time controls are not active in 2D mode.

### 10.7 Parcel selection and popup

Click a parcel to open the parcel popup. For a moving parcel, the popup reports:

- parcel ID;
- current epoch;
- parcel status;
- reversible displacement;
- irreversible displacement;
- total displacement.

A blank parcel is labelled as having no displacement source.

### 10.8 Trendline chart

From a selected moving parcel, press **Open trendline**.

Available chart views include:

- Auto;
- Irreversible;
- Reversible;
- Total;
- Decomposition.

The decomposition view displays the irreversible baseline and total trajectory, with the reversible gap between them.

Axis modes include:

- Auto;
- Fixed;
- Manual.

The chart can be exported as PNG.

### 10.9 Mini parcel viewer

The mini parcel viewer provides a local 3D inspection of the selected parcel. Available controls include:

- mini-viewer vertical exaggeration;
- optional millimetre ruler;
- reset;
- PNG export;
- drag to rotate;
- mouse wheel to zoom;
- double-click to reset.

The mini viewer reports the selected parcel, epoch, coordinate, active mode, reversible displacement, irreversible displacement, total displacement, and irreversible velocity.

### 10.10 Navigation and export

The lower navigation bar includes:

- zoom in and out;
- 2D/3D switch;
- compass and camera controls;
- project information;
- screenshot export;
- fullscreen.

The compass popup allows camera tilt and heading adjustment, reset to north, and reset view.

### 10.11 Current UI limitations

- The global opacity slider is intentionally disabled in V3.1.
- The address-search field is a placeholder and is not part of the supported V3.1 workflow.
- Time animation and vertical exaggeration are intended for the 3D view.

---

## 11. Generated outputs and overwrite behaviour

The pipeline writes generated runtime products under:

```text
_internal/data_pipeline/
```

The assembled viewer is written to:

```text
viz2_dev_v11.html
```

Run receipts are written to:

```text
run_records/
```

Generated products are overwritten by later successful pipeline runs. Treat the following as sources of truth:

- input datasets;
- `config/project_config.json`;
- approved source templates and pipeline code.

Do not make important long-term changes only inside generated runtime products or the generated viewer, because a subsequent pipeline run may replace them.

---

## 12. Troubleshooting

### Missing Python packages

Example message:

```text
Missing Python packages: ...
```

Install the missing packages in the same Python environment used by the launcher:

```powershell
python -m pip install numpy pandas geopandas pyarrow shapely mapbox-earcut
```

In VS Code, confirm the selected interpreter matches the Python executable printed by the preflight.

### `project_config.json` not found

The full folder structure may be incomplete, or the pipeline may have been moved away from `_internal/pipeline/`. Restore the release folder structure and run from the project package.

### Invalid JSON configuration

Open `config/project_config.json` in VS Code and correct syntax errors. Typical causes include:

- missing commas;
- extra trailing commas;
- mismatched braces;
- unescaped backslashes.

Prefer forward slashes in paths.

### Configured input file not found

Check spelling, extension, and path relative to the project root. Verify that the file is not still inside an unextracted ZIP archive.

### Incomplete shapefile

Keep `.shp`, `.dbf`, `.shx`, and `.prj` together with identical base names. The `.dbf` is required for parcel IDs and attributes.

### CSV missing required columns

Compare the CSV header with the exact contract in Section 5. Column names are case-sensitive.

### Parcel join failure

The pipeline requires every CSV `pnt_id` to exist in:

- the shapefile parcel-ID field;
- the model-parameter parquet `pnt_id` field.

Check data types and leading/trailing spaces. IDs must parse as numeric values.

### Duplicate parcel-date rows

Each `pnt_id` may occur only once per epoch. Remove or resolve duplicate rows before running.

### Epoch sequence is not continuous

V3.1 requires a continuous daily sequence between the earliest and latest date. Add the missing dates or prepare a complete daily dataset.

### Total displacement check fails

For every row:

```text
h_spams_final = reversible + irreversible
```

Correct the source data rather than weakening the validation.

### Viewer opens but remains blank

Check that:

- the viewer is served with Live Server or another HTTP server;
- `_internal/data_pipeline/` exists and contains the generated products;
- the latest receipt reports `PASS`;
- the browser console does not report missing files or MIME errors;
- the complete `_internal/cesium/` and `_internal/three/` folders are present.

### Parcel geometry appears but background map does not

The local viewer is working, but the online imagery request may be blocked or unavailable. Check the internet connection and browser developer console.

### Old viewer remains visible

Use `Ctrl+F5`, close duplicate Live Server sessions, and verify the served folder and URL.

---

## 13. Known limitations

Prototype 2 V3.1 has the following deliberate scope and limitations:

- It is a research prototype, not an operational deformation-monitoring platform.
- The thesis was still in progress when this release was produced.
- The active deformation source is a precomputed CSV.
- PySPAMS is not automatically run by the pipeline.
- Parcel and displacement inputs are currently restricted to EPSG:4326.
- The time-series contract expects continuous daily epochs.
- The supplied viewer is optimized around parcel-level vertical deformation.
- Vertical exaggeration is visual and should not be mistaken for physical elevation.
- Blank parcels indicate missing displacement data, not stability.
- Background maps require internet access.
- Windows is the documented user platform for the supplied batch launcher.
- The address-search field and global-opacity control are not active V3.1 features.

---

## 14. GitHub and deployment notes

### Repository contents

A practical repository should contain:

- pipeline source;
- viewer template and generated viewer;
- project configuration or configuration example;
- documentation;
- bundled runtime libraries required by the viewer;
- licences and third-party notices;
- a small example dataset where redistribution is permitted.

Large source datasets may exceed normal GitHub repository limits or may have separate redistribution restrictions. Use Git LFS, a release archive, an institutional repository, or a documented external download where appropriate.

### GitHub Pages

GitHub Pages can serve the generated viewer as a static website when all runtime paths are preserved. The pipeline itself does not run in GitHub Pages; generate the viewer locally and publish the resulting static files.

For a clean landing URL, an optional `index.html` may redirect to or contain the released viewer. Keep path capitalization exact because deployed web paths can be case-sensitive.

### Data-free deployment

A deployed viewer does not necessarily need the raw CSV, shapefile, parquet, or PySPAMS files when the generated runtime products are already present. Those source inputs are required to rebuild the viewer locally, not to serve a completed static build.

---

## 15. Data, licences, and third-party components

This README does not grant redistribution rights for third-party data or software.

The project bundles or references components including:

- CesiumJS;
- Three.js;
- PySPAMS;
- parcel geometry and SPAMS-related datasets;
- online background-map services.

Retain the original licences, attribution files, notices, and provider requirements. Review the project `LICENSE`, third-party licence files, the PySPAMS licence, and dataset terms before public distribution.

Do not remove map-provider credits or software attribution shown by the viewer.

---

## 16. Citation

Until the final thesis and repository citation are available, the software may be described as:

> Bramantya, R. (2026). *Prototype 2 V3.1: Krimpenerwaard Parcel-Based 4D Seasonal Deformation Viewer* [Research software prototype]. TU Delft MSc thesis work in progress.

Update this section with the final thesis repository link, software DOI, and release URL when available.

---

## 17. Release identity

```text
Product      : Prototype 2 V3.1
Release date : 17 June 2026
Case study   : Krimpenerwaard parcel deformation
Author       : R. Bramantya (Ridan Bramantya)
Institution  : Delft University of Technology
Thesis       : Spatiotemporal Visualization of InSAR Ground Deformation
Status       : Thesis in progress at the time of release
```
