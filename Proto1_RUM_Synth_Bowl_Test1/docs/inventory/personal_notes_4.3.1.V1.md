## Synthetic Bowl Test — Validation of Horizontal Particle System

### Objective
Validate horizontal particle mechanics, interpolation behavior, 
and uncertainty encoding against known ground truth, independent 
of real data noise.

### Grid Spec
- 80×55 RUMs, 450m spacing, EPSG:23830
- ~4400 cells, anchored near Jakarta bounding box
- Full vertical + horizontal + covariance columns
- 288 epochs, linear vertical trend per district

### Physical Constraints
- Horizontal rim magnitude = |vertical rate| × 0.4 (elastic coupling)
- Uncertainty scales inversely with signal magnitude
- Exception: D6/D_certain are explicit uncertainty test cases

### Districts
| ID  | Name                  | Tests                                      |
|-----|-----------------------|--------------------------------------------|
| D1  | Symmetric bowl        | Primary validation, radial convergence     |
| D2  | Asymmetric bowl       | Directional asymmetry preservation         |
| D3  | Double bowl           | Flow splitting, saddle point behavior      |
| D4  | Uniform translation   | Parallel flow, artifact detection          |
| D5  | Slow zone             | Near-zero graceful behavior, respawn       |
| D6  | High uncertainty      | Shimmer amplitude response (inflated σ)    |
| D7  | Fault discontinuity   | Interpolation boundary, hard edge behavior |
| D8  | Blankie island        | Particle death + respawn at data gaps      |
| D9  | RUM size variants     | 225m / 450m / 900m resolution comparison  |
| D10 | Rate comparison       | -10 vs -40 mm/yr, velocity encoding check |
| D11 | Sinusoidal district   | Seasonal reversal, Ramon's request         |
| D12 | Low uncertainty       | Shimmer = near-zero on confident data      |
| D13 | Elongated bowl        | Fault-bounded aquifer, shape preservation  |

### Ghost Car Comparison
- Toggle: nearest / conservative bilinear v2 / both simultaneously
- Money shot: D7 fault district shows hard vs soft boundary behavior
- Key figure: interpolation method comparison → thesis methodology

### Blind Test Protocol
- Show synthetic viewer to naive participant
- Ask: describe what you see in each district
- Success = correct characterization without InSAR knowledge

### Pipeline
synth_phase0  generate synthetic CSV (all districts)
synth_phase1  → phase10a → velocity_field_synth.json
synth_phase2  viewer config (auto camera, labels, synth color scale)
synth_phase3  ghost car toggle implementation
synth_phase4  blind test run + record findings
synth_phase5  write findings → thesis methodology + results section

### Expected Outputs
- Validated particle mechanics (D1, D4, D5)
- Validated interpolation choice (D7, ghost car)
- Validated uncertainty encoding (D6, D12, triplet with D1)
- Validated boundary conditions (D8)
- Validated velocity encoding (D10)
- Thesis figures: convergence, fault comparison, uncertainty triplet
- Blind test results → perceptual validation section

### FINDINGS 
- For generalizability test we found that the pipeline is sensitive to Nan in source file(s)
- The first synthetic CSV represented the D8 no-data island as explicit coordinate rows with NaN velocity and covariance values. This caused an inconsistency in the template pipeline: tile assignment treated these rows as valid RUM cells, while packed epoch generation omitted them. Comparison with the Groningen input showed that missing RUM cells should be represented by absence from the source CSV rather than NaN-valued rows. The synthetic source was therefore converted to a pipeline-safe version by removing 36 NaN rows, leaving 4364 valid RUMs.

result of new pipeline bat(afrter fix)
D:\Kuliah\Q7\Thesis\260413_Gameplan1\4.Web\4.1.CesiumApp\4.3.1.V1_Synth_Bowl_Test\docs\inventory\bat run results.txt




New-Item -ItemType Directory -Force scripts | Out-Null

@'
import json
from pathlib import Path

ROOT = Path.cwd()

MAIN_TILESET = ROOT / "Data" / "tiles" / "tileset.json"

BLANK_CAP_DIR = ROOT / "Data" / "tiles_blank"
BLANK_WALL_DIR = ROOT / "Data" / "tiles_walls_blank"

BLANK_CAP_TILESET = BLANK_CAP_DIR / "tileset.json"
BLANK_WALL_TILESET = BLANK_WALL_DIR / "tileset.json"

BLANK_CAP_ASSIGN = BLANK_CAP_DIR / "blank_tile_assignments.json"
BLANK_WALL_ASSIGN = BLANK_WALL_DIR / "blank_wall_tile_assignments.json"

BLANK_CAP_DIR.mkdir(parents=True, exist_ok=True)
BLANK_WALL_DIR.mkdir(parents=True, exist_ok=True)

if MAIN_TILESET.exists():
    main = json.loads(MAIN_TILESET.read_text(encoding="utf-8"))
    root_bv = main.get("root", {}).get("boundingVolume", {
        "region": [0, 0, 0, 0, 0, 0]
    })
else:
    root_bv = {"region": [0, 0, 0, 0, 0, 0]}

empty_tileset = {
    "asset": {
        "version": "1.0",
        "generator": "empty-placeholder-for-no-blank-cells"
    },
    "geometricError": 0,
    "root": {
        "boundingVolume": root_bv,
        "geometricError": 0,
        "refine": "ADD",
        "children": []
    }
}

BLANK_CAP_TILESET.write_text(json.dumps(empty_tileset, indent=2), encoding="utf-8")
BLANK_WALL_TILESET.write_text(json.dumps(empty_tileset, indent=2), encoding="utf-8")

BLANK_CAP_ASSIGN.write_text(json.dumps({
    "note": "No blank cells in this dataset.",
    "blank_cells": [],
    "tiles": []
}, indent=2), encoding="utf-8")

BLANK_WALL_ASSIGN.write_text(json.dumps({
    "note": "No blank-cell walls in this dataset.",
    "blank_wall_edges": [],
    "tiles": []
}, indent=2), encoding="utf-8")

print("[OK] Empty blank cap tileset written:", BLANK_CAP_TILESET)
print("[OK] Empty blank wall tileset written:", BLANK_WALL_TILESET)
print("[OK] No-blank placeholders complete.")
'@ | Set-Content scripts\create_empty_blank_tilesets.py -Encoding UTF8

python scripts\create_empty_blank_tilesets.py