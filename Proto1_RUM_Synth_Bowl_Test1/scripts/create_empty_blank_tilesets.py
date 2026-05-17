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
