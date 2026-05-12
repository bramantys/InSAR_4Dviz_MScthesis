# README

## 1. Introductory Information
- **Title of the dataset**: 3D displacement estimates due to deep-seated hydrocarbon production in Groningen from satellite radar interferometry, 2015–2023
- **File descriptions**:  
  - `groningen_enu_estimates.csv`: Tabular dataset of 3D ground displacement velocity estimates (east, north, up) for the Groningen gas field area, derived from InSAR observations (2015–2023).  
  - `groningen_enu_estimates.json`: Machine-readable metadata describing the dataset (variables, units, spatial/temporal coverage, provenance, and processing overview).
- **File naming convention**: Files are named by content and format (`*_estimates.csv` for data; `*_estimates.json` for metadata).
- **File formats**: CSV (comma-separated values) for the data; JSON (JavaScript Object Notation) for the metadata.
- **Relationships between files**: The CSV contains the dataset; the JSON file contains metadata describing the CSV columns, units, coverage, and provenance.
- **Contact information**:  
  Ramon Hanssen  
  Delft University of Technology  
  r.f.hanssen@tudelft.nl  

## 2. Methodological Information
- **Data collection/generation methods**:  
  The dataset provides 3D ground displacement estimates for the Groningen gas field area derived from InSAR observations (2015–2023). The displacement vectors are estimated with an **augmented strapdown decomposition method**, combining ascending and descending Line-of-Sight (LoS) observations to derive meaningful 3D components.  

  The strapdown approach enables “3D-global / 2D-local” solutions by incorporating minimal contextual information about the physical behavior or spatial geometry of deformation. A local TLN (transversal, longitudinal, normal) frame is defined, and motion is restricted to the transversal–normal (TN) plane. Prior uncertainty in orientation can be included and propagated to the estimates, which supports visualization using vector fields and confidence ellipses [1,2].  

  Each data point corresponds to a **Region of Uniform Motion (RUM)**, a 500 × 500 m grid cell.

- **Instrument-/sensor-specific info**:  
  Sentinel-1 A/B C-band SAR data:  
  - Track 15 (ascending)  
  - Track 88 (ascending)  
  - Track 37 (descending)  
  - Track 139 (descending)  

- **Software/workflow (with versions)**:  
  The decomposition and visualization are implemented in a Python repository (TUDelftGeodesy/3D-Groningen).  
  [https://github.com/TUDelftGeodesy/3D-Groningen](https://github.com/TUDelftGeodesy/3D-Groningen)

- **Quality assurance / uncertainty handling**:  
  Variance and covariance estimates are included to quantify uncertainties and correlations among components.

## 3. Data-Specific Information
- **Column definitions (CSV)**:  
  - `lon`: Longitude of cell centroid (degrees, WGS84)  
  - `lat`: Latitude of cell centroid (degrees, WGS84)  
  - `d_e`: East displacement velocity estimate [mm/yr]  
  - `d_n`: North displacement velocity estimate [mm/yr]  
  - `d_u`: Up displacement velocity estimate [mm/yr]  
  - `var_de`: Variance of east component [(mm/yr)²]  
  - `var_dn`: Variance of north component [(mm/yr)²]  
  - `var_du`: Variance of up component [(mm/yr)²]  
  - `cov_de_dn`: Covariance between east & north [(mm/yr)²]  
  - `cov_de_du`: Covariance between east & up [(mm/yr)²]  
  - `cov_dn_du`: Covariance between north & up [(mm/yr)²]

- **Units of measurement**:  
  - Velocities: millimeters per year (mm/yr)  
  - Variances/covariances: (mm/yr)²  

- **Missing data codes**:  
  Missing or invalid values are represented as `NaN`.

- **Specialized formats/abbreviations**:  
  - **RUM** = Region of Uniform Motion (grid cells sized 500 × 500 m)  
  - **TLN frame** = Transversal, Longitudinal, Normal local coordinate frame  
  - **TN-plane** = plane in which motion is allowed (transversal + normal)

## 4. Sharing and Access Information
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)  
- **Restrictions**: None beyond standard license terms.  

## Resources / Example Code
This dataset is accompanied by a Python implementation to load, process, and visualize the 3D displacement vectors (including error ellipses). The code is available at:  
[https://github.com/TUDelftGeodesy/3D-Groningen](https://github.com/TUDelftGeodesy/3D-Groningen)  

The repository includes a Jupyter notebook `plot_3d_displ_groningen.ipynb` to demonstrate usage and plotting.

---

## Citation
If you use this dataset, please cite:  
Brouwer, Wietske S., and Ramon F. Hanssen. “Estimating three-dimensional displacements with InSAR: the strapdown approach.” *Journal of Geodesy* 98.12 (2024): 110.  
DOI: [https://doi.org/10.1007/s00190-024-01918-2](https://doi.org/10.1007/s00190-024-01918-2)

---

## References
1. Brouwer, Wietske S., and Ramon F. Hanssen. "Estimating three-dimensional displacements with InSAR: the strapdown approach." *Journal of Geodesy* 98.12 (2024): 110. DOI: [https://doi.org/10.1007/s00190-024-01918-2](https://doi.org/10.1007/s00190-024-01918-2)  
2. Brouwer, Wietske S., and Ramon F. Hanssen. "3D surface displacement estimation over the Groningen gas field, the Netherlands." Submitted to: *Netherlands Journal of Geoscience*. DOI: [https://doi.org/10.31223/X5775W](https://doi.org/10.31223/X5775W)  
