V4.3 canvas Monte Carlo taming/comparison

Install only:
- viz1_dev_v4_shimmer.html
- viz1_dev_v4_montecarlo.html
- _internal/js/horizontal_particles_engine.js

Do not overwrite pipeline/data/config files.

Key console commands:
window.__rumDev.getHParticleDiagnostics()
window.__rumDev.setHParticleUncertaintyMode("off")
window.__rumDev.setHParticleUncertaintyMode("shimmer")
window.__rumDev.setHParticleUncertaintyMode("montecarlo")
window.__rumDev.setHParticleMonteCarloSeed("jakarta-test-1")
window.__rumDev.setHParticleMonteCarloRandomized()
window.__rumDev.setHParticleMonteCarloScale(0.25)
window.__rumDev.setHParticleMonteCarloTuning({model:"directional", strength:0.35})
window.__rumDev.setHParticleMonteCarloTuning({model:"capped_full", strength:0.35})
window.__rumDev.setHParticleMonteCarloTuning({model:"full", strength:0.35})

Model meanings:
- directional: readable default; only perpendicular covariance component perturbs the mean path.
- capped_full: full covariance direction, capped to avoid extreme spaghetti.
- full: raw 2D covariance realization; scientifically direct but can look messy.
