# Clean Rebuild Success — Jakarta RUM Template

## Date

2026-05-11

## Folder

`4.Web/4.1.CesiumApp/4.1.3.V3_4D_6`

## Result

Clean rebuild test passed.

The `Data/` folder was reduced to only:

- `Data/1_OG/`

Then the new `pipeline_template/` scripts were run from Step 01 to Step 18.

The pipeline successfully rebuilt:

- WGS84 point GeoJSON
- synthetic vertical epochs
- corrected RUM footprints
- enhanced vertical sigma
- packed vertical series
- tile index
- epoch axis
- blank cells
- height texture
- real RUM cap tiles
- blankie cap tiles
- real RUM wall tiles
- blankie wall tiles
- horizontal field
- horizontal debug vectors
- horizontal uncertainty report

## Viewer smoke test

The original `viewer_4d.html` was tested with the newly rebuilt data.

Result:

- viewer opens
- layer toggles work
- no console/file error observed
- subsidence bowl layer disabled because `Data/jakarta_bowl_outlines_wgs84.geojson` was not rebuilt

## Important note

This proves that the V6 package can rebuild the Jakarta viewer products from raw source files in `Data/1_OG`.

This is the main prerequisite before testing another RUM-based InSAR dataset.

## Remaining cleanup

The pipeline is now mostly template-structured.

The viewer is not yet fully template-clean. It still contains Jakarta-specific naming/configuration and should be refactored next.