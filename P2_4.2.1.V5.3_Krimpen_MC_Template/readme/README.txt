PROTO2 — KRIMPENERWAARD PARCEL VIEWER TEMPLATE (V5.3)
=========================================================

This package is deliberately simple to run:

    1. Put the required source files in data/.
    2. Edit config/project_config.json.
    3. Copy your existing complete _internal/cesium/ folder into this project.
    4. Double-click config/run_baby_run.bat.
    5. After PASS, open viz2_parcel_viewer.html with VS Code Live Server.

That is the full normal workflow. You do not need to run Python scripts individually,
inspect intermediate files, or manually copy generated assets.

WHAT THE BAT DOES
-----------------
- finds and actually executes the selected Python;
- checks Python packages, configuration, source files, shapefile sidecars,
  PySPAMS, KNMI coverage, the Monte Carlo NPZ contract, and Cesium;
- runs the full pipeline automatically once preflight passes;
- writes only the viewer runtime bundle needed by the HTML;
- removes temporary geometry, audit, staging, and duplicate build files after PASS;
- writes a receipt to run_records/latest_run.txt.

USER-EDITED LOCATIONS
---------------------
config/project_config.json
    The single source of truth for input paths, time period, project labels,
    output name, and optional build behaviour.

data/
    Put your input data here, or point project_config.json to its actual location.

_internal/cesium/
    Not shipped in this template. Copy your existing complete Cesium folder here.

GENERATED OUTPUTS
-----------------
viz2_parcel_viewer.html
    The assembled viewer. Serve it locally with VS Code Live Server.

_internal/data_pipeline/runtime/
    The only generated internal data retained after a successful run. It contains
    exactly the GLB geometry, Float32 arrays, lookup JSON, style JSON, and manifests
    loaded by the viewer.

run_records/latest_run.txt
run_records/latest_run.json
    Human-readable and machine-readable receipt for the most recent run.

A failed run deliberately retains _internal/data_pipeline/work/ so the error can
be diagnosed from the receipt. A successful run removes that temporary work by default.
Set pipeline_behavior.keep_build_work to true only when you are debugging the pipeline.

REQUIRED INPUTS
---------------
- parcel shapefile plus .dbf, .shx and .prj sidecars
- SPAMS model parameter Parquet and metadata JSON
- data/pyspams/utils.py
- KNMI daily weather files for stations 344 and 348
- Monte Carlo NPZ containing mean_t and sigma_t

PYTHON
------
The BAT uses, in order:
1. PROTO2_PYTHON (when set)
2. currently active Conda Python
3. .venv or venv in this project
4. Python from PATH

The first screen tells you the actual Python executable and validates the required packages.
