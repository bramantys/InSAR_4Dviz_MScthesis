INTERNAL PIPELINE — V5.3
========================

Normal users do not run these files individually. Run config/run_baby_run.bat.

Sequence:
00 Validate configured inputs and model contract
01 Compute SPAMS components
02 Prepare moving/blank parcel inventory
03 Prepare footprints
04 Triangulate parcel geometry
05 Pack Float32 animation arrays
06 Build main runtime GLB geometry
07 Prepare uncertainty carrier layouts (temporary work only)
08 Build uncertainty runtime GLB LOD geometry
09 Build inspection assets (pick, trendline, search)
10 Build colour scales, tuning and runtime manifest
11 Publish runtime assets
12 Assemble viewer HTML
13 Validate release

Final retained output: _internal/data_pipeline/runtime/
Temporary output:       _internal/data_pipeline/work/ (removed after success)
