# Performance Batch D2A — Playback frame-pacing cleanup

## Goal

Accepted C4 had acceptable average FPS, but playback showed a repeating halt-and-catch-up pattern. The likely cause was synchronous per-epoch CPU work rather than the 4×4 relief geometry itself.

D2A is viewer-only. It does not change B3DM geometry, cap LOD, sigma thresholds, p98 clipping, horizontal glyph filtering, Cesium version, or visual semantics.

## Main fix

### Before

Every epoch change called:

```js
hHeightImageCtx.getImageData(idx, 0, 1, NUM_RUMS)
```

Then it decoded all height rows for the current epoch. With 5,077 rows and 8 epoch changes per second, this introduced repeated synchronous Canvas2D readbacks on the main thread.

### After

At startup, the horizontal particle engine decodes the full height texture once into epoch-major `Float32Array` caches:

```text
hDispEpochCache[epoch * NUM_RUMS + row]
hSigmaEpochCache[epoch * NUM_RUMS + row]
```

Each epoch change now only swaps the active view:

```js
hCurrentDispByRow = hDispEpochCache.subarray(start, end)
hCurrentSigmaByRow = hSigmaEpochCache.subarray(start, end)
```

Expected cache memory:

```text
5,077 rows × 114 epochs × 4 bytes × 2 arrays ≈ 4.4 MB
```

This intentionally trades a small amount of memory for smoother playback pacing.

## Secondary fix

The epoch UI update is now gated by discrete epoch index. The slider, label, bottom date label, and popup refresh are no longer rewritten on every Cesium clock tick when the epoch index has not changed.

## Diagnostics added

The viewer now exposes:

```js
__vuncEpochPerf.report()
__vuncEpochPerf.reset()
```

The particle engine also exposes:

```js
__hParticleSurfaceCacheStats()
```

Use these after a playback run to confirm that per-epoch particle-surface update time is near zero or at least stable.

## Test protocol

Use the same C4 cameras as D0.

1. Open accepted D2A viewer.
2. Confirm pass:

```js
viewer.scene.debugShowFramesPerSecond = true;
__vuncPerf.setPass("AUTO");
__vuncPerf.report();
```

3. Reset timing:

```js
__vuncEpochPerf.reset();
```

4. Play epochs for at least 20 seconds at the city camera and near camera.
5. Report:

```js
__vuncEpochPerf.report();
__hParticleSurfaceCacheStats();
```

6. Compare visually against C4:

```text
Before expected symptom:
halt → catch-up → halt → catch-up

D2A success:
epochs advance with more even pacing, even if average FPS remains similar
```

## Expected result

This patch is not primarily an average-FPS optimization. FPS may remain around the C4 range. The desired improvement is reduced playback hitching and more even epoch timing.

## Files changed

- `viz1_dev_v4_dualmode.html`
- `_internal/js/horizontal_particles_engine.js`

