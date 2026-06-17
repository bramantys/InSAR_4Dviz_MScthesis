# Batch D2A local test note

Replace both files:

```text
viz1_dev_v4_dualmode.html
_internal/js/horizontal_particles_engine.js
```

No pipeline rerun is needed.

## Quick test

```js
viewer.scene.debugShowFramesPerSecond = true;
__vuncPerf.setPass("AUTO");
__vuncPerf.report();
__vuncEpochPerf.reset();
```

Run playback for 20–30 seconds at the same city and near cameras used in D0.

Then run:

```js
__vuncEpochPerf.report();
__hParticleSurfaceCacheStats();
```

Record:

```text
City playback FPS:
City playback smoothness:
City __vuncEpochPerf.report():

Near playback FPS:
Near playback smoothness:
Near __vuncEpochPerf.report():

Does the halt-then-catch-up pattern remain? yes/no
```

Important: this patch targets frame pacing. A successful result may keep the same 9–10 FPS average but remove or reduce the regular playback hiccup.
