# Pipeline Run Order Draft — Jakarta 4D RUM Template

## Rule

This file defines the intended run order before renaming scripts.

Raw source files for each dataset must be placed in:

`Data/1_OG/`

For Jakarta, this currently contains the original ENU source files.
For Groningen later, the equivalent Groningen JSON/CSV/PKL source files should also be placed in:

`Data/1_OG/`

Do not inspect Groningen yet.

---

## Important: prepared epoch file

Jakarta source data is velocity-based, not a real epoch-by-epoch vertical displacement product.

Therefore the template needs an explicit preprocessing step that converts a velocity product into a viewer-ready epoch file.

For Jakarta, the prepared epoch file currently used by the pipeline is:

`Data/jakarta_vertical_epochs_1b_linear_14d_synthsigma.json`

This file contains:

- `epochs`
- `epoch_decimal_year`
- `epoch_unix`
- `series`
- per-RUM `vertical_mm`
- per-RUM `sigma_mm`
- per-RUM `source_up_mm_yr`

The downstream vertical pipeline assumes this file already exists.

For Groningen later, there are two possibilities:

1. Groningen already contains time-series epochs  
   → map it into the same prepared epoch JSON format.

2. Groningen is velocity-only, like Jakarta  
   → synthesize epochs in the same way as Jakarta, then continue with the same pipeline.

This means the reusable template must include a step called something like:

`00_prepare_vertical_epochs.py`

before validation.

## Epoch synthesis clarification

Jakarta is velocity-based, not originally epoch-based.

The original epoch generator is:

`260426_generate_synthetic_epochs.py`

It creates:

- `jakarta_vertical_epochs_1a_linear_3mo.json`
- `jakarta_vertical_epochs_1b_linear_14d.json`
- `jakarta_vertical_epochs_2_dynamic_3mo.json`

from a GeoJSON containing RUM position, vertical velocity `up`, and vertical variance `var_up`.

The current `synth_sigma.py` is not the main epoch generator. It is an optional uncertainty enhancement step. It reads:

`Data/jakarta_vertical_epochs_1b_linear_14d.json`

and writes:

`Data/jakarta_vertical_epochs_1b_linear_14d_synthsigma.json`

It keeps `vertical_mm` unchanged and only synthesizes/replaces `sigma_mm`.

Therefore the clean pipeline must distinguish:

1. epoch generation from velocity;
2. optional sigma enhancement;
3. downstream packing/texture generation.

## Current historical script names

| Future clean name | Current script | Purpose |
|---|---|---|
| `00_validate_inputs.py` | `phase0_validate.py` | Validate source/intermediate input consistency before building |
| `01_build_footprints.py` | `phase1_footprints.py` | Build RUM footprints |
| `02_pack_vertical_series.py` | `phase2_pack_series.py` | Pack vertical displacement/sigma time series |
| `03_build_geometry_cache.py` | `phase3_geometry.py` | Build geometry/cache needed for tiling |
| `04_build_tile_index.py` | `phase4_tiler.py` | Assign RUMs/cells to tiles |
| `05_export_epoch_axis.py` | `phase7a_export_epochs.py` | Export epoch axis for viewer |
| `06_build_blank_cells.py` | `phase8_blank_cells.py` | Build blankie/no-data interior cells |
| `07_build_height_texture.py` | `phase9a_height_texture_rg_sigma.py` | Build height texture with displacement + sigma |
| `08_build_real_caps_b3dm.py` | `phase9b_b3dm_cap_uv.py` | Build real RUM cap b3dm tiles |
| `09_build_blank_caps_b3dm.py` | `phase9c_blank_caps_b3dm_with_normals_FIXED.py` | Build blankie cap b3dm tiles |
| `10_build_walls_b3dm.py` | `phase9d_walls_split_b3dm_upcolor_with_normals.py` | Build real and blankie wall b3dm tiles |
| `11_check_horizontal_inputs.py` | `h0_check_horizontal_inputs_v2.py` | Check source horizontal velocity/covariance fields |
| `12_build_horizontal_field.py` | `h1_build_horizontal_field.py` | Build viewer-ready horizontal_field.json |
| `13_check_horizontal_field.py` | `h2_check_horizontal_field.py` | Check horizontal field output |
| `14_check_horizontal_uncertainty.py` | `h7_0_check_horizontal_uncertainty.py` | Check covariance-derived uncertainty/shimmer logic |

---

## Current likely active run order

```bat
python pipeline\phase0_validate.py
python pipeline\phase1_footprints.py
python pipeline\phase2_pack_series.py
python pipeline\phase3_geometry.py
python pipeline\phase4_tiler.py
python pipeline\phase7a_export_epochs.py
python pipeline\phase8_blank_cells.py
python pipeline\phase9a_height_texture_rg_sigma.py
python pipeline\phase9b_b3dm_cap_uv.py
python pipeline\phase9c_blank_caps_b3dm_with_normals_FIXED.py
python pipeline\phase9d_walls_split_b3dm_upcolor_with_normals.py
python pipeline\h0_check_horizontal_inputs_v2.py
python pipeline\h1_build_horizontal_field.py
python pipeline\h2_check_horizontal_field.py
python pipeline\h7_0_check_horizontal_uncertainty.py