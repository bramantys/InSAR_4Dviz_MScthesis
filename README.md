# MSc Thesis Project: InSAR 4D Viewer Prototypes

**Thesis:** *Spatiotemporal Visualization of InSAR Ground Deformation*  
**Author:** Ridan Bramantya · 6188575  
**Status:** MSc thesis in progress

Interactive CesiumJS webportal prototypes for InSAR data visualization.

Recommended use: desktop or laptop browser with WebGL enabled.

## Online use

Open the GitHub Pages landing page and choose one of the available viewers:

- [`index.html`](index.html)

## Prototype1 - RUM based

Prototype 1 visualizes Region of Uniform Motion (RUM) products as animated 4D deformation surfaces with horizontal-motion particles.

The current viewers use one merged horizontal-particle mode switch. Monte Carlo realizations and uncertainty shimmer are available in the same viewer.

**New:** the Jakarta **Vertical Uncertainty V2** viewer adds low-poly RUM spikes as the updated vertical-uncertainty encoding while retaining the same dual-mode horizontal-particle interface.

| Dataset | Viewer |
|---|---|
| **Jakarta — Vertical Uncertainty V2 (new)** | [`Open viewer`](P1_4.1.5.V6_4D_JKT/viz1_dev_v4_dualmode.html) |
| Jakarta | [`Dual-mode viewer`](P1_4.1.5.V4_4D_MC_jakarta/viz1_dev_v4_dualmode.html) |
| Groningen | [`Dual-mode viewer`](P1_4.1.5.V4_4D_MC_gron/viz1_dev_v4_dualmode.html) |
| Synthbowl | [`Dual-mode viewer`](P1_4.1.5.V4_4D_MC_synthbowl/viz1_dev_v4_dualmode.html) |

### Prototype1 progress and experiments

Current development checkpoints and standalone sandboxes are retained for testing visual encodings before they are merged into the main viewer.

- [`Jakarta particle-occlusion checkpoint`](P1_4.1.5.V3_4D_jkt_UI_PRIMIT%20CHECKPOINT/viz1_dev_v3_batch9.3.6.1.html)
- [`Vertical uncertainty encoding sandbox`](P1_4.1.5.V4_4D_MC_jakarta/dimple_sandbox_proto1_clean_v8.html)

## Prototype2 - Parcel Based

Prototype 2 visualizes parcel-based seasonal soft-soil deformation in Krimpenerwaard.

The viewer separates:

- **Reversible deformation** — seasonal swelling and shrinkage;
- **Irreversible deformation** — accumulated long-term subsidence; and
- **Combined deformation** — both processes shown together through time.

The core visual concept is **breathing while drowning**: parcels move seasonally while their long-term reference level gradually subsides.

| Dataset | Viewer |
|---|---|
| Krimpenerwaard | [`Parcel viewer`](P2_4.2.1.V3.1_Krimpen/viz2_dev_v11.html) |
| **Krimpenerwaard - MC (new)** | [`Parcel viewer`](P2_4.2.1.V5.3_Krimpen_MC_Github/viz2_parcel_viewer.html) |

The current highlighted Prototype2 release is **Krimpenerwaard - MC**.

## Template packages

The template folders contain reusable project structures, pipeline scripts, configuration files, and documentation. They are intended for development and are not finished demonstration viewers.

- [`Proto1_RUM_TEMPLATE/`](Proto1_RUM_TEMPLATE/) — earlier Prototype 1 template package.
- [`P1_4.1.5.V4_4D_MC_template/`](P1_4.1.5.V4_4D_MC_template/) — current Prototype 1 template package.
- [`P2_4.2.1.V5.3_Krimpen_MC_Template/`](P2_4.2.1.V5.3_Krimpen_MC_Template/) — current Prototype 2 Krimpenerwaard Monte Carlo template package.

For Prototype 2 setup, input requirements, configuration, pipeline execution, run receipts, and viewer controls, read the documentation included inside the Krimpenerwaard MC template package.

## Repository

- [Open the GitHub repository](https://github.com/bramantys/InSAR_4Dviz_MScthesis)

## Local use

Online use requires no installation.

For local use, download or clone the repository and start a simple HTTP server from the repository root. Do not open the viewer HTML files directly with `file://`.

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/
```

Visual Studio Code users may instead open `index.html` with the **Live Server** extension.

## Notes

- The viewers load local JSON, B3DM, GLB, image, and JavaScript assets, so they must be served through GitHub Pages or a local HTTP server.
- CesiumJS and Three.js runtime files are bundled with the relevant packages; users do not need to install them separately.
- An internet connection may still be required for background map imagery.
- These viewers are research prototypes developed as part of an MSc thesis. They are not operational monitoring systems.
- Large generated products are included only where needed for demonstration. Use the template packages and their documentation for development.
