PROTOTYPE 2 V3.1 - QUICK START
Krimpenerwaard Parcel-Based 4D Seasonal Deformation Viewer

Release date : 17 June 2026
Author       : R. Bramantya (Ridan Bramantya)
Research     : MSc thesis work in progress at TU Delft
Thesis title : Spatiotemporal Visualization of InSAR Ground Deformation

This file contains only the essential steps needed to run the pipeline and
open the viewer. For the complete input specification, viewer controls,
troubleshooting, limitations, and GitHub notes, open README.md in this same
project folder.

==========================================================================
1. WHAT YOU NEED
==========================================================================

Recommended setup:
- Windows 10 or 11
- Python 3.12
- Visual Studio Code
- VS Code Python extension
- VS Code Live Server extension
- A modern WebGL-capable browser, preferably Chrome or Edge

CesiumJS and Three.js are already included in the project package.
You do NOT need to install CesiumJS, Three.js, Node.js, or npm.

Install the required Python packages in the Python environment that will run
the pipeline:

python -m pip install numpy pandas geopandas pyarrow shapely mapbox-earcut

An internet connection is needed for the online background map tiles.
The deformation viewer code and JavaScript engines are stored locally.

==========================================================================
2. REQUIRED INPUTS
==========================================================================

Prototype 2 V3.1 requires all of the following:

1. Precomputed displacement CSV
2. Parcel shapefile, including its sidecar files
3. SPAMS model-parameter parquet
4. SPAMS metadata JSON
5. PySPAMS folder containing utils.py and spams_main.py

IMPORTANT:
V3.1 uses ONLY the precomputed displacement CSV as the active deformation
source. PySPAMS is NOT executed automatically in this release. The parquet,
JSON, and PySPAMS files are still required by the V3.1 input contract and
validation workflow.

The displacement CSV must contain these columns:

pnt_id,pnt_gid,epoch,reversible,irreversible,h_spams_final,pnt_lat,pnt_lon,vI,std_vI,var_vI

Core rules:
- displacement values are in millimetres
- vI and std_vI are in millimetres per year
- epoch values must be valid dates; YYYY-MM-DD is recommended
- one row per pnt_id and epoch
- no duplicate parcel-date rows
- h_spams_final = reversible + irreversible
- all moving parcel IDs must exist in the shapefile and parquet
- the current release supports EPSG:4326 input
- the epoch sequence must be continuous daily data

A shapefile is a file set, not only one .shp file. Keep at least these files
together with the same base name:

.shp  .dbf  .shx  .prj

Keep supplied .cpg and .qmd files as well.

==========================================================================
3. PLACE THE INPUT DATA
==========================================================================

Use this folder layout from the project root:

data/displacement/<your_displacement_file>.csv
data/shapefile/<your_parcel_file>.shp and sidecars
data/model_params/<your_parameters>.parquet
data/model_params/<your_metadata>.json
data/pyspams/utils.py
data/pyspams/spams_main.py

Do not rename, move, or delete the _internal folder.

==========================================================================
4. EDIT THE PROJECT CONFIGURATION IN VS CODE
==========================================================================

Open the complete project folder in VS Code, then open:

config/project_config.json

Set the paths under user_inputs so they point to your files, for example:

"user_inputs": {
  "displacement_csv": "data/displacement/example_spams_model_2025.csv",
  "displacement_crs": "EPSG:4326",
  "parcel_shapefile": "data/shapefile/krimpenerwaard_attributes_wgs84.shp",
  "parcel_crs": "EPSG:4326",
  "model_parameters_parquet": "data/model_params/nl_krimpenerwaard_spams10.parquet",
  "model_metadata_json": "data/model_params/nl_krimpenerwaard_spams10.json",
  "pyspams_directory": "data/pyspams"
}

Keep this setting for V3.1:

"deformation_source": "displacement_csv"

Use forward slashes in relative paths. Save the JSON file before running.

==========================================================================
5. RUN THE PIPELINE OUTSIDE VS CODE
==========================================================================

In Windows File Explorer, open the project folder and double-click:

config/run_baby_run.bat

The launcher performs the environment/input checks and then builds the
runtime products and viewer. Keep the command window open until the final
result is shown.

==========================================================================
6. CHECK THE RUN RECEIPT
==========================================================================

After every run, open the newest files in:

run_records/run_<timestamp>.json
run_records/run_<timestamp>.txt

A successful run should show:

status       : PASS
failed_stage : null
error        : null
viewer.exists: true

When a run fails, read failed_stage and error in the JSON receipt, then read
the matching TXT file for the complete console log.

==========================================================================
7. OPEN THE VIEWER LOCALLY WITH VS CODE
==========================================================================

Do not open the viewer by double-clicking the HTML file.

In VS Code:
1. Open the complete project folder.
2. Find viz2_dev_v11.html in the project root.
3. Right-click it.
4. Select "Open with Live Server".
5. The viewer should open at a local http://127.0.0.1 or localhost address.

Command-line fallback from the project root:

python -m http.server 8000

Then open:

http://localhost:8000/viz2_dev_v11.html

==========================================================================
8. IMPORTANT NOTES
==========================================================================

- This is a research prototype, not an operational monitoring system.
- Vertical exaggeration changes only the display, not the source data.
- Grey/blank parcels have no displacement time series; they do not represent
  measured zero displacement.
- Pipeline runs overwrite generated runtime products and the generated viewer.
- Edit the input data and project configuration, not generated runtime files.
- See README.md for the complete guide.
