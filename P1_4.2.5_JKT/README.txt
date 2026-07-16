P1_4.3.2 → P1_4.3.3 — Amplitude-Aware Relief + Playback Speed Patch

Apply over the current working DeckGL folder.

Overwrite:
  src\main.js
  src\style.css
  scripts\build_jakarta_assets.mjs
  config\project_config.json
  index.html

Keep untouched:
  node_modules\
  data\jakarta_enu_estimates.csv
  public\data\jakarta\   (it will rebuild automatically)
  package.json
  package-lock.json

Then restart Vite:
  Ctrl + C
  npm run dev

What changed
------------
1. Blankies are flat grey moving support only.
   - No blankie spikes, dimples, or sigma visual.
   - Blankie tooltip no longer presents uncertainty as a data layer.

2. Live RUM relief is amplitude-aware.
   - Mean cap is always drawn under relief.
   - Effective relief = displayed uncertainty range × vertical exaggeration.
   - At 0 vertical exaggeration or 0 relief range, no relief mesh is drawn.
   - Low effective relief fades back toward the exact mean-cap colour.
   - Eight static GPU mesh-strength bands preserve the instanced architecture;
     no per-epoch cap geometry is rebuilt.

3. Relief defaults are softer.
   - Up gain 0.75, down gain 0.85.
   - Softer fixed tint ranges, especially for dimples.

4. Playback speed slider.
   - 0.25× to 4×.
   - Baseline playback remains config.playback_epochs_per_second.

Relevant config settings
------------------------
uncertainty_relief.visual_fade:
  start_effective_relief_m: 0.5
  full_effective_relief_m: 10.0
  minimum_render_weight: 0.01
  buckets: 8

viewer.playback_speed:
  default_multiplier: 1.0
  min_multiplier: 0.25
  max_multiplier: 4.0
  step_multiplier: 0.25

Validation performed
--------------------
- Real Jakarta asset build: 4,779 live RUMs + 456 blankies, 5,235 runtime rows, 288 epochs.
- Vite production build passed.
