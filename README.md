# MSc Thesis Project: InSAR 4D Viewer Prototypes

Interactive CesiumJS prototypes for RUM-based InSAR deformation visualization.

Recommended use: desktop or laptop browser with WebGL enabled.

## Online use

Open the GitHub Pages link and choose one of the viewers from the landing page:

- [`index.html`](index.html)

## Prototype1 viewers · new UI

The current Prototype1 viewers use the updated UI and the V4 canvas horizontal-particle setup.  
Each dataset has two horizontal uncertainty variants:

- **Monte Carlo**: realization-based particle paths sampled from horizontal velocity covariance.
- **Uncertainty shimmer**: earlier visual jitter/shimmer uncertainty cue.

| Dataset | Monte Carlo viewer | Uncertainty shimmer viewer |
|---|---|---|
| Jakarta | [`viz1_dev_v4_montecarlo.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_jakarta/viz1_dev_v4_montecarlo.html) | [`viz1_dev_v4_shimmer.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_jakarta/viz1_dev_v4_shimmer.html) |
| Groningen | [`viz1_dev_v4_montecarlo.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_gron/viz1_dev_v4_montecarlo.html) | [`viz1_dev_v4_shimmer.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_gron/viz1_dev_v4_shimmer.html) |
| Synthbowl | [`viz1_dev_v4_montecarlo.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_synthbowl/viz1_dev_v4_montecarlo.html) | [`viz1_dev_v4_shimmer.html`](4DViz_thesis/P1_4.1.5.V4_4D_MC_synthbowl/viz1_dev_v4_shimmer.html) |

### Occlusion checkpoint

This viewer records the current progress on the 3D particle occlusion experiment:

- [`Fix occlusion problem, current progress (Jakarta)`](4DViz_thesis/4.1.5.V3_4D_jkt_UI_PRIMIT%20CHECKPOINT/viz1_dev_v3_batch9.3.6.1.html)

## Old version Prototype1 viewers

These are the earlier Prototype1 viewers kept for comparison and project history:

1. [`Jakarta 4D RUM Viewer`](Proto1_RUM_jakarta/viewer_4d.html)
2. [`Groningen 4D RUM Viewer`](Proto1_RUM_groningen/viewer_4d.html)
3. [`Groningen tuned 4D RUM Viewer`](Proto1_RUM_groningen/viewer_4d_tuned.html)
4. [`Synthetic Bowl Test (No blank RUM)`](Proto1_RUM_Synth_Bowl_Test1/viewer_4d.html)
5. [`Synthetic Bowl Test (With Blank RUM)`](Proto1_RUM_Synth_Bowl_Test2withBlanks/viewer_4d.html)

Additional note:

- [`New toggle available for Synthetic Bowl Test`](synthetic_test_toggle_announcement.JPG)

## Template packages

The template folders contain reusable package structures and pipeline notes. They are not meant to be opened as finished viewers.

- `Proto1_RUM_TEMPLATE/` — earlier Prototype1 template package.
- `4DViz_thesis/P1_4.1.5.V4_4D_MC_template/` — new Prototype1 V4 template package with the updated UI and Monte Carlo / shimmer viewer structure.

## Repository

- [Open GitHub repository](https://github.com/bramantys/InSAR_4Dviz_MScthesis)

## Local use

Online use requires no installation.

For local use, download the repository and run a simple local server from the repository root. Do not double-click `index.html`.

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

## Notes

- The viewers load local JSON, B3DM, image, and JavaScript assets, so they must be served through a local server or GitHub Pages.
- Large generated data products are included for demonstration viewers. For development, use the template package and pipeline documentation.
