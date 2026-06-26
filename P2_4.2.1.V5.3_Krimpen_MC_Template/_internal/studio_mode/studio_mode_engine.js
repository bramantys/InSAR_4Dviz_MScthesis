/*
 * Capture Mode v1 — Batch 1 foundation + viewer API smoke harness
 * -----------------------------------------------------------------
 * The engine remains viewer-agnostic. All Proto2-specific work is behind
 * window.viewerApi, injected by the Batch 1 patch inside viz2_dev_v12.html.
 */
(function captureModeBatch1Bootstrap() {
  'use strict';

  const VERSION = '1.0.0-batch9-temporal-persistence';
  const DEFAULTS = Object.freeze({
    outputWidth: 1920,
    outputHeight: 1080,
    fps: 30,
    probeDurationMs: 1600,
    defaultBitrate: 12_000_000,
    qualityProbeBitrate: 30_000_000,
    storyboardExportBitrate: 30_000_000,
    caption: 'Capture Mode — compositor probe',
    qualityCaption: 'Capture Mode — Batch 1 high-resolution probe',
    storyboardLibraryUrl: window.__STUDIO_MODE_LIBRARY_URL__ || 'http://127.0.0.1:5511',
    autosaveDelayMs: 280
  });

  const state = {
    compositor: null,
    compositorCtx: null,
    sourceCanvas: null,
    renderRafId: null,
    isRendering: false,
    recorder: null,
    activeStream: null,
    lastBlob: null,
    lastMimeType: null,
    lastError: null,
    studio: {
      mounted: false,
      active: false,
      apiReady: false,
      entrySnapshot: null,
      lastError: null,
      exiting: false,
      exitPromise: null,
      viewfinderMounted: false,
      viewfinderLayoutRaf: null,
      storyboard: [],
      selectedShotId: null,
      previewRun: null,
      compositorCaption: { text: '', centered: false },
      nextStoryboardItemId: 1,
      nextViewNumber: 1,
      projectName: '',
      projectCreatedAt: null,
      projectMeta: null,
      autosaveTimer: null,
      autosaveLoaded: false,
      library: {
        url: window.__STUDIO_MODE_LIBRARY_URL__ || 'http://127.0.0.1:5511',
        available: false,
        files: [],
        selectedFile: '',
        lastError: ''
      }
    }
  };

  function log(...args) { console.log('[StudioMode]', ...args); }
  function warn(...args) { console.warn('[StudioMode]', ...args); }

  function getViewerApi() {
    return window.viewerApi && typeof window.viewerApi === 'object'
      ? window.viewerApi
      : null;
  }

  async function awaitViewerApi(timeoutMs = 20_000) {
    const existing = getViewerApi();
    if (existing) return existing;

    if (window.viewerApiReady && typeof window.viewerApiReady.then === 'function') {
      const api = await window.viewerApiReady;
      if (api) return api;
      throw new Error('Proto2 finished loading without exposing window.viewerApi. Check the Batch 1 viewer patch.');
    }

    return new Promise((resolve, reject) => {
      let timer = null;
      const cleanup = () => {
        window.removeEventListener('viewer-api-ready', onReady);
        window.removeEventListener('viewer-api-error', onError);
        if (timer !== null) window.clearTimeout(timer);
      };
      const onReady = () => {
        cleanup();
        const api = getViewerApi();
        if (api) resolve(api);
        else reject(new Error('viewer-api-ready fired without a usable window.viewerApi object.'));
      };
      const onError = (event) => {
        cleanup();
        reject(new Error(event?.detail?.message || 'Proto2 viewer API failed to initialize.'));
      };
      window.addEventListener('viewer-api-ready', onReady, { once: true });
      window.addEventListener('viewer-api-error', onError, { once: true });
      timer = window.setTimeout(() => {
        cleanup();
        reject(new Error('Timed out waiting for Proto2 viewerApi.'));
      }, Math.max(1000, Number(timeoutMs) || 20_000));
    });
  }

  function findSourceCanvas() {
    const api = getViewerApi();
    if (api && typeof api.getCanvas === 'function') {
      const canvas = api.getCanvas();
      if (canvas instanceof HTMLCanvasElement) return canvas;
    }

    const cesiumContainer = document.getElementById('cesiumContainer');
    const directCanvas = cesiumContainer && cesiumContainer.querySelector('canvas');
    if (directCanvas instanceof HTMLCanvasElement) return directCanvas;

    const candidates = Array.from(document.querySelectorAll('canvas'))
      .filter((canvas) => canvas instanceof HTMLCanvasElement)
      .sort((a, b) => (b.width * b.height) - (a.width * a.height));
    return candidates[0] || null;
  }

  function chooseMimeType() {
    if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== 'function') return '';
    const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || '';
  }

  function calculateCenteredCrop(sourceCanvas, targetAspect) {
    const sourceWidth = Number(sourceCanvas.width) || 0;
    const sourceHeight = Number(sourceCanvas.height) || 0;
    if (sourceWidth <= 0 || sourceHeight <= 0) {
      throw new Error('The source canvas has no drawable pixel dimensions yet.');
    }

    const sourceAspect = sourceWidth / sourceHeight;
    let sx = 0;
    let sy = 0;
    let sw = sourceWidth;
    let sh = sourceHeight;
    if (sourceAspect > targetAspect) {
      sw = Math.round(sourceHeight * targetAspect);
      sx = Math.round((sourceWidth - sw) / 2);
    } else if (sourceAspect < targetAspect) {
      sh = Math.round(sourceWidth / targetAspect);
      sy = Math.round((sourceHeight - sh) / 2);
    }
    return { sx, sy, sw, sh };
  }

  function ensureCompositor(options = {}) {
    const outputWidth = Number(options.outputWidth) || DEFAULTS.outputWidth;
    const outputHeight = Number(options.outputHeight) || DEFAULTS.outputHeight;

    if (!state.compositor) {
      const canvas = document.createElement('canvas');
      canvas.id = 'studioModeCompositorCanvas';
      canvas.setAttribute('aria-hidden', 'true');
      canvas.style.cssText = 'position:fixed;left:-10000px;top:-10000px;width:1px;height:1px;opacity:0;pointer-events:none;z-index:-1';
      document.body.appendChild(canvas);
      const context = canvas.getContext('2d', { alpha: false, desynchronized: true });
      if (!context) throw new Error('Could not create the 2D compositor context.');
      state.compositor = canvas;
      state.compositorCtx = context;
    }

    if (state.compositor.width !== outputWidth || state.compositor.height !== outputHeight) {
      state.compositor.width = outputWidth;
      state.compositor.height = outputHeight;
    }
    return state.compositor;
  }

  function normalizeCaptionPayload(input) {
    const raw = typeof input === 'function' ? input() : input;
    if (raw && typeof raw === 'object') {
      return { text: String(raw.text || '').trim(), centered: Boolean(raw.centered) };
    }
    return { text: String(raw || '').trim(), centered: false };
  }

  function wrapCaptionLines(ctx, text, maxWidth, maxLines = 2) {
    const words = String(text).trim().split(/\s+/).filter(Boolean);
    const lines = [];
    let line = '';
    let wordIndex = 0;
    for (; wordIndex < words.length; wordIndex += 1) {
      const word = words[wordIndex];
      const next = line ? `${line} ${word}` : word;
      if (!line || ctx.measureText(next).width <= maxWidth) {
        line = next;
        continue;
      }
      lines.push(line);
      line = word;
      if (lines.length >= maxLines) {
        line = '';
        break;
      }
    }
    if (line && lines.length < maxLines) lines.push(line);
    const truncated = wordIndex < words.length;
    if (truncated && lines.length) {
      // Keep captions predictable at 1080p: two readable lines max.
      const last = lines.length - 1;
      while (ctx.measureText(`${lines[last]}…`).width > maxWidth && lines[last].length > 1) {
        lines[last] = lines[last].slice(0, -1);
      }
      lines[last] = `${lines[last]}…`;
    }
    return lines;
  }

  function drawCaption(ctx, width, height, input) {
    const payload = normalizeCaptionPayload(input);
    if (!payload.text) return;

    const centered = payload.centered;
    const cardWidth = centered ? Math.round(width * 0.72) : Math.round(width * 0.60);
    const cardHeight = centered ? Math.round(height * 0.145) : Math.round(height * 0.105);
    const x = centered ? Math.round((width - cardWidth) / 2) : Math.round(width * 0.055);
    const y = centered
      ? Math.round((height - cardHeight) / 2)
      : height - cardHeight - Math.round(height * 0.075);

    ctx.save();
    ctx.fillStyle = centered ? 'rgba(9, 14, 18, 0.84)' : 'rgba(14, 19, 24, 0.80)';
    ctx.fillRect(x, y, cardWidth, cardHeight);
    ctx.fillStyle = 'rgba(126, 245, 255, 1)';
    if (centered) {
      ctx.fillRect(x, y, cardWidth, Math.max(4, Math.round(height * 0.003)));
    } else {
      ctx.fillRect(x, y, Math.max(6, Math.round(width * 0.004)), cardHeight);
    }
    ctx.font = `${centered ? 700 : 600} ${Math.round(height * (centered ? 0.043 : 0.034))}px Arial, sans-serif`;
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.textAlign = centered ? 'center' : 'left';

    const lines = wrapCaptionLines(ctx, payload.text, cardWidth - Math.round(width * (centered ? 0.080 : 0.065)));
    const lineHeight = Math.round(height * (centered ? 0.048 : 0.038));
    const blockHeight = lines.length * lineHeight;
    let textY = y + (cardHeight - blockHeight) / 2 + lineHeight / 2;
    const textX = centered ? x + cardWidth / 2 : x + Math.round(width * 0.030);
    for (const line of lines) {
      ctx.fillText(line, textX, textY);
      textY += lineHeight;
    }
    ctx.restore();
  }

  function drawProbeBadge(ctx, width, height) {
    ctx.save();
    ctx.font = `700 ${Math.round(height * 0.020)}px Arial, sans-serif`;
    ctx.textBaseline = 'middle';
    const text = '16:9 · LIVE COMPOSITOR TEST';
    const pad = Math.round(height * 0.016);
    const textWidth = ctx.measureText(text).width;
    const x = width - textWidth - pad * 2 - Math.round(width * 0.045);
    const y = Math.round(height * 0.045);
    const h = Math.round(height * 0.050);
    ctx.fillStyle = 'rgba(14, 19, 24, 0.76)';
    ctx.fillRect(x, y, textWidth + pad * 2, h);
    ctx.fillStyle = '#7ef5ff';
    ctx.fillText(text, x + pad, y + h / 2);
    ctx.restore();
  }

  function drawFrame(options = {}) {
    const sourceCanvas = options.sourceCanvas || state.sourceCanvas || findSourceCanvas();
    if (!sourceCanvas) throw new Error('No Cesium canvas was found. Open the viewer and wait until it has rendered.');

    const compositor = ensureCompositor(options);
    const ctx = state.compositorCtx;
    const width = compositor.width;
    const height = compositor.height;
    const crop = options.sourceCrop
      || getStudioViewfinderState(sourceCanvas)?.canvasCrop
      || calculateCenteredCrop(sourceCanvas, width / height);
    ctx.save();
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);
    try {
      ctx.drawImage(sourceCanvas, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, width, height);
    } catch (error) {
      throw new Error('The Cesium canvas could not be copied into the compositor. This often means a basemap or imagery tile is not CORS-safe. ' + (error?.message || ''));
    }
    if (options.showProbeBadge !== false) drawProbeBadge(ctx, width, height);
    drawCaption(ctx, width, height, options.caption ?? DEFAULTS.caption);
    ctx.restore();
    state.sourceCanvas = sourceCanvas;
    return { compositor, crop, sourceCanvas };
  }

  function startRenderLoop(options = {}) {
    stopRenderLoop();
    state.isRendering = true;
    const tick = () => {
      if (!state.isRendering) return;
      try { drawFrame(options); }
      catch (error) {
        state.lastError = error;
        state.isRendering = false;
        warn('Compositor loop stopped:', error.message || error);
        return;
      }
      state.renderRafId = requestAnimationFrame(tick);
    };
    tick();
  }

  function stopRenderLoop() {
    state.isRendering = false;
    if (state.renderRafId !== null) {
      cancelAnimationFrame(state.renderRafId);
      state.renderRafId = null;
    }
  }

  function sleep(milliseconds) { return new Promise((resolve) => window.setTimeout(resolve, milliseconds)); }
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.style.display = 'none';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function timestampForFilename() { return new Date().toISOString().replace(/[:.]/g, '-'); }

  function getDiagnostics() {
    const sourceCanvas = findSourceCanvas();
    const api = getViewerApi();
    const mediaRecorderAvailable = typeof window.MediaRecorder !== 'undefined';
    return {
      version: VERSION,
      locationProtocol: window.location.protocol,
      secureContext: window.isSecureContext,
      documentHidden: document.hidden,
      sourceCanvasFound: !!sourceCanvas,
      sourceCanvasPixels: sourceCanvas ? { width: sourceCanvas.width, height: sourceCanvas.height } : null,
      sourceCanvasCssPixels: sourceCanvas ? { width: sourceCanvas.clientWidth, height: sourceCanvas.clientHeight } : null,
      mediaRecorderAvailable,
      captureStreamAvailable: !!HTMLCanvasElement.prototype.captureStream,
      preferredMimeType: chooseMimeType() || null,
      compositorReady: !!state.compositor,
      viewerApiAvailable: !!api,
      viewerApiVersion: api?.version || null,
      captureQuality: api?.getCaptureQualityState ? api.getCaptureQualityState() : null,
      studioActive: state.studio.active,
      viewfinder: getStudioViewfinderState(sourceCanvas),
      lastError: state.lastError ? String(state.lastError.message || state.lastError) : null
    };
  }

  async function runBatch0Probe(options = {}) {
    if (state.recorder && state.recorder.state !== 'inactive') throw new Error('A capture probe is already running.');
    if (!window.MediaRecorder) throw new Error('MediaRecorder is not available in this browser. Use a current Chrome, Edge, or Firefox build.');
    if (!HTMLCanvasElement.prototype.captureStream) throw new Error('canvas.captureStream() is not available in this browser.');

    const durationMs = Math.max(500, Number(options.durationMs) || DEFAULTS.probeDurationMs);
    const fps = Math.min(60, Math.max(12, Number(options.fps) || DEFAULTS.fps));
    const download = options.download !== false;
    const caption = options.caption ?? DEFAULTS.caption;
    const outputWidth = Math.max(2, Math.floor(Number(options.outputWidth) || DEFAULTS.outputWidth));
    const outputHeight = Math.max(2, Math.floor(Number(options.outputHeight) || DEFAULTS.outputHeight));
    const videoBitsPerSecond = Math.max(1_000_000, Number(options.videoBitsPerSecond) || DEFAULTS.defaultBitrate);
    const mimeType = chooseMimeType();

    state.lastError = null;
    const sourceCanvas = findSourceCanvas();
    if (!sourceCanvas) throw new Error('No Cesium canvas found yet. Wait for Proto2 to finish loading, then retry.');

    state.sourceCanvas = sourceCanvas;
    drawFrame({ caption, outputWidth, outputHeight });
    const compositor = ensureCompositor({ outputWidth, outputHeight });
    const stream = compositor.captureStream(fps);
    const chunks = [];
    let recorder;
    try {
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond })
        : new MediaRecorder(stream, { videoBitsPerSecond });
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      throw new Error(`MediaRecorder could not start: ${error.message || error}`);
    }

    state.recorder = recorder;
    state.activeStream = stream;
    state.lastMimeType = recorder.mimeType || mimeType || 'video/webm';
    const result = await new Promise((resolve, reject) => {
      recorder.addEventListener('dataavailable', (event) => { if (event.data && event.data.size > 0) chunks.push(event.data); });
      recorder.addEventListener('error', (event) => reject(event.error || new Error('Unknown MediaRecorder error')), { once: true });
      recorder.addEventListener('stop', () => {
        try {
          const blob = new Blob(chunks, { type: state.lastMimeType });
          if (!blob.size) throw new Error('MediaRecorder stopped but produced an empty file.');
          resolve(blob);
        } catch (error) { reject(error); }
      }, { once: true });

      startRenderLoop({ caption, outputWidth, outputHeight });
      recorder.start(250);
      window.setTimeout(() => { if (recorder.state !== 'inactive') recorder.stop(); }, durationMs);
    }).finally(() => {
      stopRenderLoop();
      stream.getTracks().forEach((track) => track.stop());
      state.activeStream = null;
      state.recorder = null;
    });

    state.lastBlob = result;
    const filename = `capture_mode_batch0_${timestampForFilename()}.webm`;
    if (download) downloadBlob(result, filename);
    const summary = {
      ok: true, filename, bytes: result.size, durationRequestedMs: durationMs, fps,
      mimeType: state.lastMimeType, videoBitsPerSecond,
      output: { width: compositor.width, height: compositor.height },
      diagnostics: getDiagnostics()
    };
    log('Batch 0 probe complete:', summary);
    return summary;
  }

  async function runBatch1QualityProbe(options = {}) {
    const api = await awaitViewerApi();
    const outputWidth = Math.max(2, Math.floor(Number(options.outputWidth) || DEFAULTS.outputWidth));
    const outputHeight = Math.max(2, Math.floor(Number(options.outputHeight) || DEFAULTS.outputHeight));
    const viewfinder = getStudioViewfinderState();
    const qualityOptions = {
      outputWidth,
      outputHeight,
      maxResolutionScale: Number(options.maxResolutionScale) || 3.5
    };
    if (viewfinder?.cssCrop) qualityOptions.cropCssRect = viewfinder.cssCrop;

    const quality = await api.beginCaptureQuality(qualityOptions);

    try {
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const probe = await runBatch0Probe({
        ...options,
        outputWidth,
        outputHeight,
        caption: options.caption ?? DEFAULTS.qualityCaption,
        videoBitsPerSecond: Number(options.videoBitsPerSecond) || DEFAULTS.qualityProbeBitrate
      });
      const result = {
        ...probe,
        captureQuality: quality,
        viewfinder: getStudioViewfinderState()
      };
      log('Batch 1 quality probe complete:', result);
      return result;
    } finally {
      await api.endCaptureQuality();
    }
  }



  // ---------------------------------------------------------------------------
  // Batch 7 — Storyboard WebM export
  // ---------------------------------------------------------------------------
  // The compositor remains the single capture source. It receives the dynamic
  // caption payload from the Preview runner, so browser-only DOM captions are
  // faithfully baked into the actual WebM.

  function createStoryboardFilename() {
    return `proto2_storyboard_${timestampForFilename()}.webm`;
  }

  function ensureMediaRecorderSupport() {
    if (!window.MediaRecorder) throw new Error('MediaRecorder is not available in this browser. Use a current Chrome, Edge, or Firefox build.');
    if (!HTMLCanvasElement.prototype.captureStream) throw new Error('canvas.captureStream() is not available in this browser.');
  }

  function createStoryboardRecordingSession(options = {}) {
    if (state.recorder && state.recorder.state !== 'inactive') {
      throw new Error('A recording is already active. Stop it before starting another export.');
    }
    ensureMediaRecorderSupport();

    const outputWidth = Math.max(2, Math.floor(Number(options.outputWidth) || DEFAULTS.outputWidth));
    const outputHeight = Math.max(2, Math.floor(Number(options.outputHeight) || DEFAULTS.outputHeight));
    const fps = Math.min(60, Math.max(12, Number(options.fps) || DEFAULTS.fps));
    const videoBitsPerSecond = Math.max(1_000_000, Number(options.videoBitsPerSecond) || DEFAULTS.storyboardExportBitrate);
    const sourceCanvas = findSourceCanvas();
    if (!sourceCanvas) throw new Error('No Cesium canvas found yet. Wait for Proto2 to finish loading, then retry export.');

    const frameOptions = {
      outputWidth,
      outputHeight,
      sourceCanvas,
      showProbeBadge: false,
      caption: () => state.studio.compositorCaption
    };
    state.sourceCanvas = sourceCanvas;
    drawFrame(frameOptions);
    const compositor = ensureCompositor({ outputWidth, outputHeight });
    const stream = compositor.captureStream(fps);
    const mimeType = chooseMimeType();
    const chunks = [];

    let recorder;
    try {
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond })
        : new MediaRecorder(stream, { videoBitsPerSecond });
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      throw new Error(`MediaRecorder could not start: ${error.message || error}`);
    }

    let resolveBlob;
    let rejectBlob;
    const done = new Promise((resolve, reject) => {
      resolveBlob = resolve;
      rejectBlob = reject;
    });

    const session = {
      recorder,
      stream,
      chunks,
      done,
      frameOptions,
      outputWidth,
      outputHeight,
      fps,
      videoBitsPerSecond,
      mimeType: recorder.mimeType || mimeType || 'video/webm',
      startedAt: Date.now(),
      finished: false,
      finishPromise: null,
      recorderError: null
    };

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener('error', (event) => {
      const error = event.error || new Error('Unknown MediaRecorder error');
      session.recorderError = error;
      rejectBlob(error);
      if (typeof options.onError === 'function') options.onError(error);
    }, { once: true });
    recorder.addEventListener('stop', () => {
      if (session.recorderError) return;
      try {
        const blob = new Blob(chunks, { type: session.mimeType });
        if (!blob.size) throw new Error('MediaRecorder stopped but produced an empty WebM.');
        resolveBlob(blob);
      } catch (error) {
        rejectBlob(error);
      }
    }, { once: true });

    state.recorder = recorder;
    state.activeStream = stream;
    state.lastMimeType = session.mimeType;
    startRenderLoop(frameOptions);
    recorder.start(250);
    return session;
  }

  async function finishStoryboardRecordingSession(session) {
    if (!session) return null;
    if (session.finishPromise) return session.finishPromise;
    session.finishPromise = (async () => {
      try {
        if (session.recorder.state !== 'inactive') session.recorder.stop();
        const blob = await session.done;
        state.lastBlob = blob;
        return blob;
      } finally {
        session.finished = true;
        stopRenderLoop();
        try { session.stream.getTracks().forEach((track) => track.stop()); } catch (_) { /* best effort */ }
        if (state.activeStream === session.stream) state.activeStream = null;
        if (state.recorder === session.recorder) state.recorder = null;
      }
    })();
    return session.finishPromise;
  }

  // ---------------------------------------------------------------------------
  // Batch 2 — Studio Mode dock shell
  // ---------------------------------------------------------------------------
  // The dock is rendered by this engine, but its mount point and layout CSS live
  // in the viewer HTML. This keeps viewer ownership (placement) separate from
  // studio ownership (behaviour), while all Proto2 internals stay behind viewerApi.

  function studioDom() {
    return {
      mount: document.getElementById('studioModeDockMount'),
      header: document.getElementById('studioModeToggle'),
      toolbox: document.getElementById('studioModeToolbox'),
      status: document.getElementById('studioModeStatus'),
      exit: document.getElementById('studioModeExit'),
      captureView: document.getElementById('studioModeCaptureView'),
      addIntro: document.getElementById('studioModeAddIntro'),
      clearStoryboard: document.getElementById('studioModeClearStoryboard'),
      preview: document.getElementById('studioModePreview'),
      record: document.getElementById('studioModeRecord'),
      projectName: document.getElementById('studioModeProjectName'),
      saveProject: document.getElementById('studioModeSaveProject'),
      loadProject: document.getElementById('studioModeLoadProject'),
      refreshLibrary: document.getElementById('studioModeRefreshLibrary'),
      libraryFiles: document.getElementById('studioModeLibraryFiles'),
      libraryState: document.getElementById('studioModeLibraryState'),
      storyboardCards: document.getElementById('studioModeStoryboardCards'),
      storyboardDuration: document.getElementById('studioModeStoryboardDuration'),
      storyboardTopDuration: document.getElementById('studioModeStoryboardTopDuration'),
      root: document.getElementById('leftControlRoot'),
      burger: document.getElementById('leftDrawerBurger')
    };
  }

  function setStudioStatus(message, tone = 'muted') {
    const { status } = studioDom();
    if (!status) return;
    status.textContent = message;
    status.dataset.tone = tone;
  }

  function setStudioControlsReady(ready) {
    const { header, mount } = studioDom();
    if (header) {
      header.disabled = !ready;
      header.title = ready ? 'Open Studio Mode' : 'Waiting for the Proto2 viewer to finish loading';
    }
    document.querySelectorAll('[data-studio-story-action]').forEach((button) => {
      button.disabled = !ready;
    });
    if (mount) mount.classList.toggle('studioModeApiReady', Boolean(ready));
  }

  function setStudioDockOpen(open) {
    const { mount, header, toolbox } = studioDom();
    if (!mount || !header || !toolbox) return;
    mount.classList.toggle('studioModeDockOpen', Boolean(open));
    header.setAttribute('aria-expanded', String(Boolean(open)));
    toolbox.hidden = !open;
  }

  function ensureExistingDrawerOpen() {
    const { root, burger } = studioDom();
    if (!root || root.classList.contains('drawerOpen')) return;
    if (burger) burger.click();
  }


  // ---------------------------------------------------------------------------
  // Batch 3 — 16:9 authoring frame
  // ---------------------------------------------------------------------------
  // The frame owns the exact crop that the compositor uses while Studio Mode is
  // active. The left drawer remains an authoring overlay; it is not recorded.
  // The visible, unobstructed scene at right is the actual exported source area.

  function studioViewfinderDom() {
    return {
      overlay: document.getElementById('studioModeViewfinderOverlay'),
      frame: document.getElementById('studioModeViewfinderFrame'),
      previewHud: document.getElementById('studioModePreviewHud'),
      previewProgress: document.getElementById('studioModePreviewProgress'),
      previewCaption: document.getElementById('studioModePreviewCaption')
    };
  }

  function ensureStudioViewfinder() {
    let { overlay, frame } = studioViewfinderDom();
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'studioModeViewfinderOverlay';
      overlay.setAttribute('aria-hidden', 'true');
      overlay.innerHTML = `
        <div id="studioModeViewfinderFrame" role="presentation">
          <div class="studioModeViewfinderCorner studioModeViewfinderCornerTL"></div>
          <div class="studioModeViewfinderCorner studioModeViewfinderCornerTR"></div>
          <div class="studioModeViewfinderCorner studioModeViewfinderCornerBL"></div>
          <div class="studioModeViewfinderCorner studioModeViewfinderCornerBR"></div>
          <div id="studioModePreviewHud" aria-hidden="true" hidden>
            <div id="studioModePreviewProgress"></div>
            <div id="studioModePreviewCaption"></div>
          </div>
        </div>`;
      document.body.appendChild(overlay);
      frame = overlay.querySelector('#studioModeViewfinderFrame');
    }
    state.studio.viewfinderMounted = Boolean(frame);
    return { overlay, frame };
  }

  function layoutStudioViewfinder() {
    const { overlay, frame } = ensureStudioViewfinder();
    const sourceCanvas = findSourceCanvas();
    const { root } = studioDom();
    if (!overlay || !frame || !sourceCanvas) return null;

    const canvasRect = sourceCanvas.getBoundingClientRect();
    const drawerRect = root?.getBoundingClientRect();
    const shotControl = document.getElementById('displayQuickPanel');
    const shotControlRect = shotControl && getComputedStyle(shotControl).display !== 'none'
      ? shotControl.getBoundingClientRect()
      : null;
    const inset = Math.max(12, Math.min(24, Math.round(Math.min(canvasRect.width, canvasRect.height) * 0.032)));
    const aspect = DEFAULTS.outputWidth / DEFAULTS.outputHeight;

    let left = Math.max(canvasRect.left + inset, (drawerRect?.right || canvasRect.left) + inset);
    let top = canvasRect.top + inset;
    let right = canvasRect.right - inset;
    let bottom = canvasRect.bottom - inset;

    // Batch 4.1: the live Vertical exaggeration control remains usable in
    // Studio Mode. Keep the composition frame below it instead of letting
    // authoring chrome intrude into the export crop.
    const controlOverlapsScene = shotControlRect
      && shotControlRect.width > 0
      && shotControlRect.height > 0
      && shotControlRect.right > left
      && shotControlRect.left < right
      && shotControlRect.bottom > top
      && shotControlRect.top < bottom;
    if (controlOverlapsScene) {
      top = Math.max(top, Math.min(bottom - 90, shotControlRect.bottom + inset));
    }

    // Narrow-window fallback: retain a usable frame even if the open drawer
    // temporarily consumes almost all of the available CSS viewport.
    if (right - left < 160 || bottom - top < 90) {
      left = canvasRect.left + inset;
      right = canvasRect.right - inset;
    }

    const availableWidth = Math.max(1, right - left);
    const availableHeight = Math.max(1, bottom - top);
    let width = availableWidth;
    let height = width / aspect;
    if (height > availableHeight) {
      height = availableHeight;
      width = height * aspect;
    }

    const x = left + (availableWidth - width) * 0.5;
    const y = top + (availableHeight - height) * 0.5;
    frame.style.left = `${Math.round(x)}px`;
    frame.style.top = `${Math.round(y)}px`;
    frame.style.width = `${Math.round(width)}px`;
    frame.style.height = `${Math.round(height)}px`;

    return {
      left: x,
      top: y,
      width,
      height,
      right: x + width,
      bottom: y + height
    };
  }

  function requestStudioViewfinderLayout() {
    if (!state.studio.active) return;
    if (state.studio.viewfinderLayoutRaf !== null) {
      cancelAnimationFrame(state.studio.viewfinderLayoutRaf);
    }
    state.studio.viewfinderLayoutRaf = requestAnimationFrame(() => {
      state.studio.viewfinderLayoutRaf = null;
      layoutStudioViewfinder();
    });
  }

  function showStudioViewfinder() {
    const { overlay } = ensureStudioViewfinder();
    if (!overlay) return null;
    overlay.classList.add('is-visible');
    const layout = layoutStudioViewfinder();
    requestAnimationFrame(() => layoutStudioViewfinder());
    return layout;
  }

  function hideStudioViewfinder() {
    const { overlay } = studioViewfinderDom();
    if (state.studio.viewfinderLayoutRaf !== null) {
      cancelAnimationFrame(state.studio.viewfinderLayoutRaf);
      state.studio.viewfinderLayoutRaf = null;
    }
    if (overlay) overlay.classList.remove('is-visible');
  }

  function getStudioViewfinderState(sourceCanvas = findSourceCanvas()) {
    const { overlay, frame } = studioViewfinderDom();
    if (!state.studio.active || !overlay?.classList.contains('is-visible') || !frame || !sourceCanvas) {
      return null;
    }

    const canvasRect = sourceCanvas.getBoundingClientRect();
    const frameRect = frame.getBoundingClientRect();
    if (canvasRect.width <= 0 || canvasRect.height <= 0 || frameRect.width <= 0 || frameRect.height <= 0) {
      return null;
    }

    const clippedLeft = Math.max(canvasRect.left, frameRect.left);
    const clippedTop = Math.max(canvasRect.top, frameRect.top);
    const clippedRight = Math.min(canvasRect.right, frameRect.right);
    const clippedBottom = Math.min(canvasRect.bottom, frameRect.bottom);
    const cssWidth = Math.max(0, clippedRight - clippedLeft);
    const cssHeight = Math.max(0, clippedBottom - clippedTop);
    if (cssWidth < 1 || cssHeight < 1) return null;

    const cssCrop = {
      left: clippedLeft - canvasRect.left,
      top: clippedTop - canvasRect.top,
      width: cssWidth,
      height: cssHeight
    };
    const scaleX = sourceCanvas.width / canvasRect.width;
    const scaleY = sourceCanvas.height / canvasRect.height;
    const sx = Math.max(0, Math.round(cssCrop.left * scaleX));
    const sy = Math.max(0, Math.round(cssCrop.top * scaleY));
    const sw = Math.max(1, Math.min(sourceCanvas.width - sx, Math.round(cssCrop.width * scaleX)));
    const sh = Math.max(1, Math.min(sourceCanvas.height - sy, Math.round(cssCrop.height * scaleY)));

    return {
      aspect: cssCrop.width / cssCrop.height,
      cssCrop,
      viewportRect: {
        left: frameRect.left,
        top: frameRect.top,
        width: frameRect.width,
        height: frameRect.height
      },
      canvasCrop: { sx, sy, sw, sh }
    };
  }

  async function runBatch3ViewfinderProbe(options = {}) {
    if (!state.studio.active) {
      throw new Error('Enter Studio Mode before running the Batch 3 frame-aligned probe.');
    }
    const viewfinder = getStudioViewfinderState();
    if (!viewfinder) {
      throw new Error('The Studio Mode viewfinder is not ready yet. Wait one frame and retry.');
    }
    return runBatch1QualityProbe({
      ...options,
      caption: options.caption ?? 'Studio Mode — 16:9 frame-aligned capture',
      videoBitsPerSecond: Number(options.videoBitsPerSecond) || DEFAULTS.qualityProbeBitrate
    });
  }

  function injectBatch5Styles() {
    if (document.getElementById('studioModeBatch5Styles')) return;
    const style = document.createElement('style');
    style.id = 'studioModeBatch5Styles';
    style.textContent = `
#studioModeDockMount .studioModeShotCard {
  cursor: pointer;
  transition: background-color 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}
#studioModeDockMount .studioModeShotCard:hover {
  background: rgba(255,255,255,0.062);
}
#studioModeDockMount .studioModeShotCard.is-expanded {
  background: rgba(255,255,255,0.072);
  border-color: rgba(126,245,255,0.30);
  box-shadow: 0 0 0 1px rgba(126,245,255,0.08) inset;
}
#studioModeDockMount .studioModeShotCardTop {
  grid-template-columns: 26px auto minmax(0, 1fr) auto 22px;
}
#studioModeDockMount .studioModeShotReorder {
  display: grid;
  grid-template-rows: 10px 10px;
  gap: 2px;
  align-self: stretch;
}
#studioModeDockMount .studioModeShotMove {
  width: 26px;
  min-width: 26px;
  height: 10px;
  padding: 0;
  border: 1px solid var(--ui-border-soft);
  border-radius: 4px;
  background: rgba(255,255,255,0.04);
  color: var(--ui-text-muted);
  font-size: 8px;
  line-height: 1;
  cursor: pointer;
}
#studioModeDockMount .studioModeShotMove:hover:not(:disabled) {
  background: var(--ui-bg-hover);
  color: var(--ui-text-strong);
}
#studioModeDockMount .studioModeShotMove:disabled {
  opacity: 0.35;
  cursor: default;
}
#studioModeDockMount .studioModeShotMeta {
  margin-top: 6px;
}
#studioModeDockMount .studioModeShotEditor {
  display: grid;
  gap: 7px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255,255,255,0.07);
}
#studioModeDockMount .studioModeShotEditor[hidden] {
  display: none;
}
#studioModeDockMount .studioModeShotEditorGrid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 76px;
  gap: 6px;
}
#studioModeDockMount .studioModeField {
  display: grid;
  gap: 4px;
}
#studioModeDockMount .studioModeFieldLabel {
  color: var(--ui-text-faint);
  font-size: 7.9px;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}
#studioModeDockMount .studioModeTextInput,
#studioModeDockMount .studioModeNumberInput,
#studioModeDockMount .studioModeTextarea {
  width: 100%;
  border: 1px solid var(--ui-border-soft);
  border-radius: 6px;
  background: rgba(9, 14, 18, 0.58);
  color: var(--ui-text-strong);
  font: inherit;
  font-size: 9px;
  line-height: 1.35;
  padding: 6px 7px;
  box-sizing: border-box;
}
#studioModeDockMount .studioModeTextInput:focus,
#studioModeDockMount .studioModeNumberInput:focus,
#studioModeDockMount .studioModeTextarea:focus {
  outline: none;
  border-color: rgba(126,245,255,0.6);
  box-shadow: 0 0 0 1px rgba(126,245,255,0.12);
}
#studioModeDockMount .studioModeTextarea {
  min-height: 52px;
  resize: vertical;
}
#studioModeDockMount .studioModeShotEditorNote {
  color: var(--ui-text-muted);
  font-size: 8.3px;
  line-height: 1.35;
}
#studioModeDockMount .studioModeFooterActions {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 6px;
  margin-top: 6px;
}
#studioModeDockMount .studioModeTertiaryAction {
  width: 100%;
  min-height: 26px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 6px;
  background: rgba(255,255,255,0.02);
  color: var(--ui-text-muted);
  font-size: 9px;
  font-weight: 700;
  cursor: pointer;
}
#studioModeDockMount .studioModeTertiaryAction:hover:not(:disabled) {
  background: var(--ui-bg-hover);
  color: var(--ui-text-strong);
}
#studioModeDockMount .studioModeTertiaryAction:disabled {
  opacity: 0.45;
  cursor: default;
}
#studioModeDockMount .studioModePreviewAction.is-previewing {
  background: rgba(255, 213, 79, 0.14);
  border-color: rgba(255, 213, 79, 0.38);
  color: var(--ui-warning);
}
#studioModeDockMount .studioModeRecordAction.is-recording {
  background: rgba(255, 112, 112, 0.14);
  border-color: rgba(255, 112, 112, 0.42);
  color: #ffb0b0;
}
#studioModeViewfinderFrame #studioModePreviewHud {
  position: absolute;
  inset: 0;
  display: block;
  pointer-events: none;
}
#studioModeViewfinderFrame #studioModePreviewHud[hidden] {
  display: none;
}
#studioModePreviewProgress {
  position: absolute;
  top: 10px;
  left: 12px;
  max-width: calc(100% - 24px);
  padding: 4px 6px;
  border-radius: 4px;
  background: rgba(9, 14, 18, 0.72);
  color: var(--ui-accent);
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.055em;
  line-height: 1.1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#studioModePreviewCaption {
  position: absolute;
  left: 5.5%;
  bottom: 7%;
  width: min(67%, 560px);
  max-width: calc(100% - 11%);
  padding: 9px 11px;
  border-left: 3px solid var(--ui-accent);
  border-radius: 0 6px 6px 0;
  background: rgba(9, 14, 18, 0.80);
  color: #fff;
  font-size: clamp(11px, 1.6vw, 20px);
  font-weight: 650;
  line-height: 1.28;
  white-space: pre-wrap;
  text-shadow: 0 1px 2px rgba(0,0,0,0.35);
}
#studioModePreviewCaption.is-empty {
  display: none;
}
#studioModePreviewCaption.is-centered {
  left: 14%;
  right: 14%;
  bottom: auto;
  top: 50%;
  width: auto;
  max-width: none;
  transform: translateY(-50%);
  padding: 16px 20px;
  border: 1px solid rgba(126,245,255,0.44);
  border-left-width: 3px;
  border-radius: 8px;
  background: rgba(9, 14, 18, 0.84);
  text-align: center;
  font-size: clamp(16px, 3vw, 32px);
  font-weight: 750;
}
#studioModeDockMount .studioModeTransitionHint {
  margin-top: 7px;
  color: var(--ui-text-faint);
  font-size: 8.3px;
  line-height: 1.35;
}
#studioModeDockMount .studioModeTransitionHint b {
  color: var(--ui-accent);
  font-weight: 800;
}
#studioModeDockMount .studioModeShotCard--transition {
  margin: 1px 7px;
  border-style: dashed;
  border-left: 3px solid rgba(185,215,255,0.70);
  background: rgba(185,215,255,0.035);
}
#studioModeDockMount .studioModeShotCard--transition.is-expanded {
  border-color: rgba(185,215,255,0.46);
  box-shadow: 0 0 0 1px rgba(185,215,255,0.08) inset;
}
#studioModeDockMount .studioModeShotCard--transition .studioModeShotCardTop {
  grid-template-columns: auto minmax(0, 1fr) auto;
}
#studioModeDockMount .studioModeShotCard--transition .studioModeShotType {
  color: #cbe0ff;
}
#studioModeDockMount .studioModeTransitionState {
  margin-top: 4px;
  color: var(--ui-text-faint);
  font-size: 8px;
  line-height: 1.28;
}
#studioModeDockMount .studioModeTransitionDurationField {
  max-width: 102px;
}
#studioModeDockMount .studioModeTemporalBlock {
  display: grid;
  gap: 6px;
  padding: 7px;
  border: 1px solid rgba(126,245,255,0.13);
  border-radius: 7px;
  background: rgba(126,245,255,0.025);
}
#studioModeDockMount .studioModeTemporalGrid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 6px;
}
#studioModeDockMount .studioModeEpochRangeInput {
  width: 100%;
  accent-color: var(--ui-accent);
}
#studioModeDockMount .studioModeEpochLabel {
  min-height: 12px;
  color: var(--ui-text-muted);
  font-size: 8px;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
#studioModeDockMount .studioModeTemporalHelp {
  color: var(--ui-text-faint);
  font-size: 8px;
  line-height: 1.3;
}
#studioModeDockMount .studioModeProjectLibrary {
  display: grid;
  gap: 6px;
  margin-top: 7px;
  padding-top: 7px;
  border-top: 1px solid rgba(255,255,255,0.07);
}
#studioModeDockMount .studioModeProjectLibraryHeader {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  color: var(--ui-text-faint);
  font-size: 7.8px;
  font-weight: 800;
  letter-spacing: 0.045em;
}
#studioModeDockMount .studioModeLibraryState {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ui-text-muted);
  font-weight: 650;
  letter-spacing: 0;
  text-transform: none;
}
#studioModeDockMount .studioModeLibraryRow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
}
#studioModeDockMount .studioModeLibraryRow--load {
  grid-template-columns: minmax(0, 1fr) auto auto;
}
#studioModeDockMount .studioModeLibraryInput,
#studioModeDockMount .studioModeLibrarySelect {
  min-width: 0;
  width: 100%;
  height: 26px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 6px;
  box-sizing: border-box;
  background: rgba(9,14,18,0.58);
  color: var(--ui-text-strong);
  padding: 0 7px;
  font: inherit;
  font-size: 8.7px;
}
#studioModeDockMount .studioModeLibraryButton {
  min-width: 48px;
  height: 26px;
  padding: 0 8px;
  border: 1px solid var(--ui-border-soft);
  border-radius: 6px;
  background: rgba(255,255,255,0.035);
  color: var(--ui-text-muted);
  font: inherit;
  font-size: 8.7px;
  font-weight: 750;
  cursor: pointer;
}
#studioModeDockMount .studioModeLibraryButton:hover:not(:disabled) {
  background: var(--ui-bg-hover);
  color: var(--ui-text-strong);
}
#studioModeDockMount .studioModeLibraryButton:disabled,
#studioModeDockMount .studioModeLibraryInput:disabled,
#studioModeDockMount .studioModeLibrarySelect:disabled {
  opacity: 0.45;
  cursor: default;
}
    `;
    document.head.appendChild(style);
  }

  function renderStudioDock() {
    const { mount } = studioDom();
    if (!mount || state.studio.mounted) return;

    mount.innerHTML = `
      <div class="studioModeDock" aria-label="Studio Mode">
        <button id="studioModeToggle" class="studioModeDockHeader" type="button" aria-expanded="false" disabled>
          <span class="studioModeDockHeaderMain">Studio Mode</span>
          <span class="studioModeDockHeaderMeta">16:9</span>
          <span class="studioModeDockChevron" aria-hidden="true">⌃</span>
        </button>
        <section id="studioModeToolbox" class="studioModeToolbox" hidden aria-label="Capture Mode toolbox">
          <div class="studioModeToolboxFixedTop">
            <div class="studioModeToolboxTopline">
              <span>CAPTURE MODE</span>
              <span id="studioModeStoryboardTopDuration">STORYBOARD · 00:00</span>
            </div>
            <div id="studioModeStatus" class="studioModeStatus" data-tone="muted">Studio shell ready.</div>

            <button id="studioModeCaptureView" class="studioModePrimaryAction" type="button" data-studio-story-action disabled>Capture View</button>
            <button id="studioModeAddIntro" class="studioModeSecondaryAction" type="button" data-studio-story-action disabled>+ Intro Preset</button>
            <div class="studioModeTransitionHint">Transitions are generated automatically between Views. Set <b>0 s</b> for a cut.</div>
          </div>

          <div id="studioModeStoryboardRegion" class="studioModeStoryboardRegion" aria-label="Storyboard list">
            <div class="studioModeStoryboardShell" aria-label="Storyboard">
              <div class="studioModeStoryboardLabel"><span>STORYBOARD</span><span id="studioModeStoryboardDuration">00:00</span></div>
              <div id="studioModeStoryboardCards" class="studioModeStoryboardCards"></div>
            </div>
          </div>

          <div class="studioModeDockFooter">
            <div class="studioModeRunGrid">
              <button id="studioModePreview" class="studioModePreviewAction" type="button" disabled title="Preview the current storyboard">Preview</button>
              <button id="studioModeRecord" class="studioModeRecordAction" type="button" disabled title="Play and save the 16:9 storyboard as a 1080p WebM">Save 1080p WebM</button>
            </div>
            <div class="studioModeProjectLibrary" aria-label="Storyboard project library">
              <div class="studioModeProjectLibraryHeader"><span>STORYBOARD LIBRARY</span><span id="studioModeLibraryState" class="studioModeLibraryState">Autosave initialising…</span></div>
              <div class="studioModeLibraryRow">
                <input id="studioModeProjectName" class="studioModeLibraryInput" type="text" maxlength="80" placeholder="Storyboard name" aria-label="Storyboard name" />
                <button id="studioModeSaveProject" class="studioModeLibraryButton" type="button" disabled>Save</button>
              </div>
              <div class="studioModeLibraryRow studioModeLibraryRow--load">
                <select id="studioModeLibraryFiles" class="studioModeLibrarySelect" aria-label="Saved storyboards" disabled><option value="">Start library to load saved projects</option></select>
                <button id="studioModeLoadProject" class="studioModeLibraryButton" type="button" disabled>Load</button>
                <button id="studioModeRefreshLibrary" class="studioModeLibraryButton" type="button" disabled title="Refresh the local storyboard folder">↻</button>
              </div>
            </div>
            <div class="studioModeFooterActions">
              <button id="studioModeClearStoryboard" class="studioModeTertiaryAction" type="button" disabled>Clear all</button>
              <button id="studioModeExit" class="studioModeExitAction" type="button">Exit Studio Mode</button>
            </div>
          </div>
        </section>
      </div>`;

    state.studio.mounted = true;
    bindStudioControls();
    renderStoryboard();
    setStudioDockOpen(false);
    setStudioHeaderState(false);
    setStudioStatus('Waiting for Proto2 viewer…', 'muted');
  }

  function setStudioHeaderState(active) {
    const { header } = studioDom();
    if (!header) return;
    const meta = header.querySelector('.studioModeDockHeaderMeta');
    const chevron = header.querySelector('.studioModeDockChevron');
    header.title = active ? 'Close Studio Mode' : 'Open Studio Mode';
    header.setAttribute('aria-label', active ? 'Close Studio Mode' : 'Open Studio Mode');
    if (meta) meta.textContent = active ? 'CLOSE' : '16:9';
    if (chevron) chevron.textContent = active ? '×' : '⌃';
  }

  const STORY_DEFAULTS = Object.freeze({
    viewDurationSec: 5,
    transitionDurationSec: 0,
    // 2.5 s map/title context + a 4.5 s curved approach/reveal by default.
    introDurationSec: 7
  });

  function clonePlain(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatStoryboardDuration(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function nextStoryboardId() {
    const id = `shot-${state.studio.nextStoryboardItemId}`;
    state.studio.nextStoryboardItemId += 1;
    return id;
  }

  function nextViewName() {
    // Never reuse a view label in the current authoring session. Deleting a
    // card should not make a later shot look like an earlier one.
    const number = state.studio.nextViewNumber;
    state.studio.nextViewNumber += 1;
    return `View ${String(number).padStart(2, '0')}`;
  }

  function sceneMapLabel(value) {
    if (value === 'bw') return 'B/W';
    if (value === 'map') return 'Map';
    if (value === 'satellite') return 'No map';
    return value ? String(value) : 'Map unavailable';
  }

  function viewSceneSummary(scene) {
    if (!scene) return 'Camera state unavailable';
    const date = scene.epochLabel || `Epoch ${Number(scene.epoch) + 1}`;
    const mode = scene.displayMode || 'display';
    const exag = Number.isFinite(Number(scene.verticalExag)) ? `${Number(scene.verticalExag).toFixed(1)}×` : 'exag —';
    return `${date} · ${mode} · ${exag} · ${sceneMapLabel(scene.mapLayerMode)}`;
  }

  function shotDescription(item) {
    if (item.type === 'view') return viewSceneSummary(item.scene);
    if (item.type === 'intro') return '2.5 s map context + title · curved approach and parcel reveal into the first View.';
    return 'Storyboard item';
  }

  function shotTypeLabel(type) {
    if (type === 'view') return 'VIEW';
    if (type === 'intro') return 'INTRO';
    return 'SHOT';
  }

  function defaultCaptionLabel(item) {
    if (item.type === 'intro') return 'Intro title';
    return 'View caption';
  }

  function defaultCaptionPlaceholder(item) {
    if (item.type === 'intro') return 'e.g. Ground deformation overview';
    return 'Optional lower-third caption for this View';
  }

  function clampDurationSeconds(value, fallback) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return Math.max(1, Math.round(Number(fallback) || 1));
    return Math.max(1, Math.min(600, Math.round(numeric)));
  }

  function clampTransitionDurationSeconds(value, fallback = STORY_DEFAULTS.transitionDurationSec) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return Math.max(0, Math.round(Number(fallback) || 0));
    return Math.max(0, Math.min(600, Math.round(numeric)));
  }

  function defaultDurationForType(type) {
    return type === 'intro' ? STORY_DEFAULTS.introDurationSec : STORY_DEFAULTS.viewDurationSec;
  }

  function findStoryboardIndex(itemId) {
    return state.studio.storyboard.findIndex((item) => item.id === itemId);
  }

  function followingViewItem(startIndex) {
    for (let index = startIndex + 1; index < state.studio.storyboard.length; index += 1) {
      const item = state.studio.storyboard[index];
      if (item?.type === 'view' && item.scene) return { item, index };
    }
    return null;
  }

  function firstViewItem() {
    return state.studio.storyboard.find((item) => item?.type === 'view' && item.scene) || null;
  }

  function ensureTransitionConfig(sourceView) {
    if (!sourceView || sourceView.type !== 'view') return { durationSec: 0, caption: '' };
    if (!sourceView.transitionToNext || typeof sourceView.transitionToNext !== 'object') {
      sourceView.transitionToNext = { durationSec: STORY_DEFAULTS.transitionDurationSec, caption: '' };
    }
    sourceView.transitionToNext.durationSec = clampTransitionDurationSeconds(
      sourceView.transitionToNext.durationSec,
      STORY_DEFAULTS.transitionDurationSec
    );
    sourceView.transitionToNext.caption = String(sourceView.transitionToNext.caption || '');
    return sourceView.transitionToNext;
  }

  function transitionSelectionId(sourceViewId) {
    return `transition:${sourceViewId}`;
  }

  function transitionSourceIdFromSelection(selectionId) {
    const prefix = 'transition:';
    return String(selectionId || '').startsWith(prefix) ? String(selectionId).slice(prefix.length) : null;
  }

  function followingViewForSourceId(sourceViewId) {
    const index = findStoryboardIndex(sourceViewId);
    return index < 0 ? null : followingViewItem(index);
  }

  function validSelectionId(selectionId) {
    const transitionSourceId = transitionSourceIdFromSelection(selectionId);
    if (transitionSourceId) return Boolean(followingViewForSourceId(transitionSourceId));
    return state.studio.storyboard.some((item) => item.id === selectionId);
  }

  function selectedStoryboardItemId() {
    if (!state.studio.storyboard.length) return null;
    const existing = validSelectionId(state.studio.selectedShotId) ? state.studio.selectedShotId : null;
    const latestView = [...state.studio.storyboard].reverse().find((item) => item.type === 'view') || state.studio.storyboard[state.studio.storyboard.length - 1];
    const selected = existing || latestView.id;
    state.studio.selectedShotId = selected;
    return selected;
  }

  function selectStoryboardItem(selectionId, options = {}) {
    if (!selectionId || !validSelectionId(selectionId)) return false;
    const changed = state.studio.selectedShotId !== selectionId;
    state.studio.selectedShotId = selectionId;
    if (changed && options.render !== false) renderStoryboard();
    return changed;
  }

  function canMoveStoryboardItemUp(index) {
    if (index <= 0) return false;
    const item = state.studio.storyboard[index];
    if (!item || item.type !== 'view') return false;
    return state.studio.storyboard[index - 1]?.type !== 'intro';
  }

  function canMoveStoryboardItemDown(index) {
    if (index < 0 || index >= state.studio.storyboard.length - 1) return false;
    const item = state.studio.storyboard[index];
    return Boolean(item?.type === 'view');
  }

  function moveStoryboardItem(itemId, direction) {
    const index = findStoryboardIndex(itemId);
    if (index < 0) return false;
    const delta = direction === 'up' ? -1 : 1;
    if (delta < 0 && !canMoveStoryboardItemUp(index)) return false;
    if (delta > 0 && !canMoveStoryboardItemDown(index)) return false;
    const targetIndex = index + delta;
    const [item] = state.studio.storyboard.splice(index, 1);
    state.studio.storyboard.splice(targetIndex, 0, item);
    state.studio.selectedShotId = item.id;
    scheduleStoryboardAutosave();
    renderStoryboard();
    setStudioStatus(`${item.name} moved ${direction}. Review its automatic transition.`, 'muted');
    return true;
  }

  function epochTimeline() {
    const meta = state.studio.projectMeta || {};
    const labels = Array.isArray(meta.epochLabels) ? meta.epochLabels.map((label) => String(label)) : [];
    const count = Math.max(1, Number(meta.epochCount) || labels.length || 1);
    return { count, labels };
  }

  function clampEpochIndex(value, fallback = 0) {
    const { count } = epochTimeline();
    const numeric = Number(value);
    const resolved = Number.isFinite(numeric) ? Math.floor(numeric) : Math.floor(Number(fallback) || 0);
    return Math.max(0, Math.min(count - 1, resolved));
  }

  function epochLabelForIndex(index) {
    const { labels } = epochTimeline();
    const clamped = clampEpochIndex(index);
    return labels[clamped] || `Epoch ${clamped + 1}`;
  }

  function ensureEpochRange(item) {
    const scene = item?.scene || {};
    const range = item?.epochRange && typeof item.epochRange === 'object'
      ? item.epochRange
      : (scene.epochRange && typeof scene.epochRange === 'object' ? scene.epochRange : {});
    const fallback = clampEpochIndex(scene.epoch, 0);
    const normalized = {
      from: clampEpochIndex(range.from, fallback),
      to: clampEpochIndex(range.to, fallback),
      behavior: String(range.behavior || (Number(range.from) === Number(range.to) ? 'hold' : 'play')).toLowerCase() === 'play' ? 'play' : 'hold'
    };
    if (item && typeof item === 'object') item.epochRange = normalized;
    return normalized;
  }

  function projectIdFromMeta(meta = state.studio.projectMeta) {
    const raw = meta?.projectId || meta?.project_id || window.location.pathname || 'proto2';
    return String(raw).replace(/[^a-z0-9._-]+/gi, '_').replace(/^_+|_+$/g, '') || 'proto2';
  }

  function defaultProjectName(meta = state.studio.projectMeta) {
    const raw = meta?.projectId || meta?.projectName || 'storyboard';
    return String(raw).replace(/[_-]+/g, ' ').trim() || 'storyboard';
  }

  function normalizedProjectName(input) {
    const value = String(input || '').trim();
    return value || defaultProjectName();
  }

  function safeStoryboardFilename(name) {
    const slug = String(name || 'storyboard')
      .trim()
      .replace(/[^a-z0-9._-]+/gi, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 72) || 'storyboard';
    return `${slug}.json`;
  }

  function storyboardStorageKey() {
    return `proto2.capture-mode.storyboard.v1.${projectIdFromMeta()}`;
  }

  function documentMetadata() {
    const meta = state.studio.projectMeta || {};
    return {
      projectId: projectIdFromMeta(meta),
      projectName: String(meta.projectName || meta.displayName || ''),
      epochCount: Number(meta.epochCount) || 0,
      epochStart: Array.isArray(meta.epochLabels) && meta.epochLabels.length ? meta.epochLabels[0] : null,
      epochEnd: Array.isArray(meta.epochLabels) && meta.epochLabels.length ? meta.epochLabels[meta.epochLabels.length - 1] : null
    };
  }

  function storyboardDocument() {
    const now = new Date().toISOString();
    if (!state.studio.projectCreatedAt) state.studio.projectCreatedAt = now;
    return {
      schema: 'proto2_capture_storyboard_v1',
      version: 1,
      createdAt: state.studio.projectCreatedAt,
      updatedAt: now,
      project: documentMetadata(),
      projectName: normalizedProjectName(state.studio.projectName),
      storyboard: clonePlain(state.studio.storyboard),
      nextStoryboardItemId: state.studio.nextStoryboardItemId,
      nextViewNumber: state.studio.nextViewNumber
    };
  }

  function maxCounterFromStoryboard(storyboard) {
    let shotCounter = 0;
    let viewCounter = 0;
    storyboard.forEach((item) => {
      const shotMatch = String(item?.id || '').match(/^shot-(\d+)$/i);
      if (shotMatch) shotCounter = Math.max(shotCounter, Number(shotMatch[1]) || 0);
      const viewMatch = String(item?.name || '').match(/^View\s+(\d+)$/i);
      if (viewMatch) viewCounter = Math.max(viewCounter, Number(viewMatch[1]) || 0);
    });
    return { shotCounter, viewCounter };
  }

  function normaliseLoadedStoryboard(rawStoryboard) {
    const input = Array.isArray(rawStoryboard) ? rawStoryboard : [];
    const intro = [];
    const views = [];
    const seenIds = new Set();
    input.forEach((raw, index) => {
      if (!raw || typeof raw !== 'object') return;
      const type = String(raw.type || '').toLowerCase();
      if (type !== 'view' && type !== 'intro') return;
      if (type === 'view' && (!raw.scene || typeof raw.scene !== 'object')) return;
      let id = String(raw.id || `${type}-${index + 1}`);
      if (seenIds.has(id)) id = `${id}-${index + 1}`;
      seenIds.add(id);
      const item = {
        id,
        type,
        name: String(raw.name || (type === 'intro' ? 'Orbit Intro' : `View ${String(index + 1).padStart(2, '0')}`)),
        durationSec: clampDurationSeconds(raw.durationSec, defaultDurationForType(type)),
        caption: String(raw.caption || '')
      };
      if (type === 'intro') {
        item.preset = String(raw.preset || 'orbit-v1');
        if (!intro.length) intro.push(item);
      } else {
        item.scene = clonePlain(raw.scene);
        item.epochRange = clonePlain(raw.epochRange || raw.scene.epochRange || { from: raw.scene.epoch, to: raw.scene.epoch, behavior: 'hold' });
        item.transitionToNext = {
          durationSec: clampTransitionDurationSeconds(raw.transitionToNext?.durationSec, STORY_DEFAULTS.transitionDurationSec),
          caption: String(raw.transitionToNext?.caption || '')
        };
        ensureEpochRange(item);
        views.push(item);
      }
    });
    return [...intro, ...views];
  }

  function hydrateStoryboardDocument(documentInput, options = {}) {
    if (!documentInput || typeof documentInput !== 'object') throw new Error('Storyboard file is not a valid project document.');
    if (documentInput.schema && documentInput.schema !== 'proto2_capture_storyboard_v1') {
      throw new Error(`Unsupported storyboard schema: ${documentInput.schema}`);
    }
    const incomingProjectId = String(documentInput.project?.projectId || '');
    const currentProjectId = projectIdFromMeta();
    if (incomingProjectId && currentProjectId && incomingProjectId !== currentProjectId && options.allowProjectMismatch !== true) {
      throw new Error(`This storyboard belongs to “${incomingProjectId}”, not the current project “${currentProjectId}”.`);
    }
    const storyboard = normaliseLoadedStoryboard(documentInput.storyboard);
    const counters = maxCounterFromStoryboard(storyboard);
    state.studio.storyboard = storyboard;
    state.studio.selectedShotId = storyboard.length ? (storyboard.find((item) => item.type === 'view') || storyboard[0]).id : null;
    state.studio.projectName = normalizedProjectName(documentInput.projectName || state.studio.projectName);
    state.studio.projectCreatedAt = String(documentInput.createdAt || new Date().toISOString());
    state.studio.nextStoryboardItemId = Math.max(Number(documentInput.nextStoryboardItemId) || 1, counters.shotCounter + 1);
    state.studio.nextViewNumber = Math.max(Number(documentInput.nextViewNumber) || 1, counters.viewCounter + 1);
    renderStoryboard();
    scheduleStoryboardAutosave();
    return storyboard.length;
  }

  function updateLibraryUi() {
    const { projectName, saveProject, loadProject, refreshLibrary, libraryFiles, libraryState } = studioDom();
    const library = state.studio.library;
    if (projectName && document.activeElement !== projectName) projectName.value = normalizedProjectName(state.studio.projectName);
    const previewing = isStoryboardPreviewing();
    if (libraryState) {
      libraryState.textContent = library.available
        ? `${library.files.length} file${library.files.length === 1 ? '' : 's'} · folder ready`
        : 'Autosave active · library offline';
      libraryState.title = library.available
        ? `Folder: _internal/studio_mode/storyboards`
        : 'Run _internal/studio_mode/start_storyboard_library.ps1 to save and load project files in the local folder.';
    }
    if (saveProject) saveProject.disabled = previewing || !state.studio.apiReady || !library.available;
    if (refreshLibrary) refreshLibrary.disabled = previewing || !state.studio.apiReady;
    if (libraryFiles) {
      libraryFiles.disabled = previewing || !library.available || !library.files.length;
      const current = library.selectedFile;
      const options = library.files.map((file) => `<option value="${escapeHtml(file.name)}">${escapeHtml(file.name)}</option>`).join('');
      libraryFiles.innerHTML = options || `<option value="">${library.available ? 'No saved storyboards yet' : 'Start local storyboard library'}</option>`;
      const selected = library.files.some((file) => file.name === current) ? current : (library.files[0]?.name || '');
      library.selectedFile = selected;
      libraryFiles.value = selected;
    }
    if (loadProject) loadProject.disabled = previewing || !library.available || !library.selectedFile;
  }

  function scheduleStoryboardAutosave() {
    if (!state.studio.apiReady) return;
    if (state.studio.autosaveTimer !== null) window.clearTimeout(state.studio.autosaveTimer);
    state.studio.autosaveTimer = window.setTimeout(() => {
      state.studio.autosaveTimer = null;
      try {
        localStorage.setItem(storyboardStorageKey(), JSON.stringify(storyboardDocument()));
        updateLibraryUi();
      } catch (error) {
        warn('Could not autosave storyboard locally:', error);
      }
    }, DEFAULTS.autosaveDelayMs);
  }

  function restoreStoryboardAutosave() {
    try {
      const raw = localStorage.getItem(storyboardStorageKey());
      if (!raw) return false;
      const documentInput = JSON.parse(raw);
      const count = hydrateStoryboardDocument(documentInput);
      state.studio.autosaveLoaded = true;
      if (count) setStudioStatus(`Restored ${count} storyboard item${count === 1 ? '' : 's'} from local autosave.`, 'ready');
      return Boolean(count);
    } catch (error) {
      warn('Could not restore local storyboard autosave:', error);
      return false;
    }
  }

  function libraryEndpoint(path = '') {
    const base = String(state.studio.library.url || DEFAULTS.storyboardLibraryUrl).replace(/\/+$/, '');
    return `${base}${path}`;
  }

  async function fetchStoryboardLibrary(path, options = {}) {
    const response = await fetch(libraryEndpoint(path), {
      cache: 'no-store',
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {})
      }
    });
    const bodyText = await response.text();
    let payload = null;
    try { payload = bodyText ? JSON.parse(bodyText) : null; } catch (_) { payload = null; }
    if (!response.ok) throw new Error(payload?.error || `Storyboard library request failed (${response.status}).`);
    return payload;
  }

  async function refreshStoryboardLibrary(options = {}) {
    const library = state.studio.library;
    try {
      await fetchStoryboardLibrary('/api/health');
      const payload = await fetchStoryboardLibrary('/api/storyboards');
      library.available = true;
      library.files = Array.isArray(payload?.files) ? payload.files.filter((file) => file && file.name) : [];
      library.lastError = '';
      if (options.status !== false) setStudioStatus('Storyboard folder ready.', 'ready');
    } catch (error) {
      library.available = false;
      library.files = [];
      library.selectedFile = '';
      library.lastError = error?.message || String(error);
      if (options.status === true) setStudioStatus('Storyboard folder is offline. Autosave still protects this browser session.', 'muted');
    }
    updateLibraryUi();
    return library.available;
  }

  async function saveStoryboardProject() {
    if (isStoryboardPreviewing()) return false;
    const { projectName } = studioDom();
    state.studio.projectName = normalizedProjectName(projectName?.value || state.studio.projectName);
    const online = state.studio.library.available || await refreshStoryboardLibrary({ status: false });
    if (!online) throw new Error('Storyboard folder is offline. In a second VS Code terminal, run _internal/studio_mode/start_storyboard_library.ps1. Local autosave is still active.');
    const filename = safeStoryboardFilename(state.studio.projectName);
    await fetchStoryboardLibrary(`/api/storyboards/${encodeURIComponent(filename)}`, {
      method: 'PUT',
      body: JSON.stringify(storyboardDocument(), null, 2)
    });
    state.studio.library.selectedFile = filename;
    await refreshStoryboardLibrary({ status: false });
    setStudioStatus(`Saved ${filename} to _internal/studio_mode/storyboards.`, 'ready');
    return filename;
  }

  async function loadStoryboardProject(filename = state.studio.library.selectedFile) {
    const selected = String(filename || '');
    if (!selected) throw new Error('Choose a saved storyboard first.');
    const payload = await fetchStoryboardLibrary(`/api/storyboards/${encodeURIComponent(selected)}`);
    const incomingProjectId = String(payload?.project?.projectId || '');
    const currentProjectId = projectIdFromMeta();
    if (incomingProjectId && currentProjectId && incomingProjectId !== currentProjectId) {
      const proceed = window.confirm(`This storyboard belongs to “${incomingProjectId}”, not “${currentProjectId}”. Load it anyway?`);
      if (!proceed) return false;
    }
    const count = hydrateStoryboardDocument(payload, { allowProjectMismatch: true });
    state.studio.library.selectedFile = selected;
    updateLibraryUi();
    setStudioStatus(`Loaded ${selected} · ${count} storyboard item${count === 1 ? '' : 's'}.`, 'ready');
    return true;
  }

  function updateStoryboardItem(itemId, field, rawValue) {
    const index = findStoryboardIndex(itemId);
    if (index < 0) return false;
    const item = state.studio.storyboard[index];
    if (field === 'name') {
      const next = String(rawValue ?? '').trim();
      if (next) item.name = next;
    } else if (field === 'durationSec') {
      item.durationSec = clampDurationSeconds(rawValue, item.durationSec || defaultDurationForType(item.type));
    } else if (field === 'caption') {
      item.caption = String(rawValue ?? '');
    } else if (item.type === 'view' && field === 'epochBehavior') {
      const range = ensureEpochRange(item);
      range.behavior = String(rawValue || '').toLowerCase() === 'play' ? 'play' : 'hold';
    } else if (item.type === 'view' && field === 'epochFrom') {
      const range = ensureEpochRange(item);
      range.from = clampEpochIndex(rawValue, range.from);
    } else if (item.type === 'view' && field === 'epochTo') {
      const range = ensureEpochRange(item);
      range.to = clampEpochIndex(rawValue, range.to);
    } else {
      return false;
    }
    scheduleStoryboardAutosave();
    renderStoryboard();
    return true;
  }

  function updateTransition(sourceViewId, field, rawValue) {
    const index = findStoryboardIndex(sourceViewId);
    const source = index >= 0 ? state.studio.storyboard[index] : null;
    const target = index >= 0 ? followingViewItem(index) : null;
    if (!source || source.type !== 'view' || !target) return false;
    const transition = ensureTransitionConfig(source);
    if (field === 'durationSec') transition.durationSec = clampTransitionDurationSeconds(rawValue, transition.durationSec);
    else if (field === 'caption') transition.caption = String(rawValue ?? '');
    else return false;
    scheduleStoryboardAutosave();
    renderStoryboard();
    return true;
  }

  function storyboardTotalSeconds() {
    let total = 0;
    state.studio.storyboard.forEach((item, index) => {
      total += Math.max(0, Number(item.durationSec) || 0);
      if (item.type === 'view' && followingViewItem(index)) {
        total += ensureTransitionConfig(item).durationSec;
      }
    });
    return total;
  }

  function isStoryboardPreviewing() {
    return Boolean(state.studio.previewRun && state.studio.previewRun.active);
  }

  function isStoryboardRecording() {
    return Boolean(state.studio.previewRun && state.studio.previewRun.active && state.studio.previewRun.mode === 'record');
  }

  function storyboardRunLabel(run) {
    return run?.mode === 'record' ? 'Export' : 'Preview';
  }

  function previewTextForItem(item) {
    const caption = String(item?.caption || '').trim();
    if (caption) return caption;
    if (item?.type === 'intro') return item.name && item.name !== 'Orbit Intro' ? item.name : '';
    return '';
  }

  function updatePreviewOverlay(item, index, total, options = {}) {
    const { previewHud, previewProgress, previewCaption } = studioViewfinderDom();
    const kind = options.kind || item?.type || 'shot';
    const label = options.label || item?.name || 'Storyboard Preview';
    const text = String(options.text !== undefined ? options.text : previewTextForItem(item));
    const centered = options.centered === true;
    state.studio.compositorCaption = { text, centered };

    if (!previewHud || !previewProgress || !previewCaption) return;
    previewHud.hidden = false;
    previewHud.setAttribute('aria-hidden', 'false');
    previewProgress.textContent = `${storyboardRunLabel(options.run)} · ${index + 1}/${total} · ${String(kind).toUpperCase()} · ${label}`;
    previewCaption.textContent = text;
    previewCaption.classList.toggle('is-empty', !text.trim());
    previewCaption.classList.toggle('is-centered', centered);
  }

  function clearPreviewOverlay() {
    const { previewHud, previewProgress, previewCaption } = studioViewfinderDom();
    state.studio.compositorCaption = { text: '', centered: false };
    if (!previewHud || !previewProgress || !previewCaption) return;
    previewHud.hidden = true;
    previewHud.setAttribute('aria-hidden', 'true');
    previewProgress.textContent = '';
    previewCaption.textContent = '';
    previewCaption.classList.remove('is-empty', 'is-centered');
  }

  function waitPreviewDuration(run, durationSec) {
    const milliseconds = Math.max(0, Number(durationSec) || 0) * 1000;
    return new Promise((resolve) => {
      if (run.cancelled || milliseconds <= 1) {
        resolve({ cancelled: Boolean(run.cancelled) });
        return;
      }
      let settled = false;
      const settle = (result) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        if (run.waitCancel === cancel) run.waitCancel = null;
        resolve(result);
      };
      const timer = window.setTimeout(() => settle({ cancelled: false }), milliseconds);
      const cancel = () => settle({ cancelled: true });
      run.waitCancel = cancel;
    });
  }

  function normalizedEpochRange(item) {
    return ensureEpochRange(item);
  }

  function buildPreviewTimeline() {
    const steps = [];
    state.studio.storyboard.forEach((item, index) => {
      if (item.type === 'intro') {
        steps.push({ kind: 'intro', item, targetView: followingViewItem(index)?.item || firstViewItem() });
        return;
      }
      if (item.type !== 'view' || !item.scene) return;
      steps.push({ kind: 'view', item });
      const next = followingViewItem(index);
      if (next) {
        steps.push({
          kind: 'transition',
          source: item,
          target: next.item,
          config: clonePlain(ensureTransitionConfig(item))
        });
      }
    });
    return steps;
  }

  async function previewViewItem(run, api, item, index, total, options = {}) {
    if (!item.scene) throw new Error(`${item.name} has no saved scene state.`);
    updatePreviewOverlay(item, index, total, { run, kind: 'view', label: item.name, text: previewTextForItem(item) });
    setStudioStatus(`${storyboardRunLabel(run)} · ${index + 1}/${total} — ${item.name}`, 'active');
    await api.applyStudioSceneState(item.scene, { camera: options.camera !== false });
    if (run.cancelled) return;
    const range = normalizedEpochRange(item);
    if (range.behavior === 'play' && range.from !== range.to) {
      await api.playEpochs(range.from, range.to, item.durationSec);
    } else {
      api.setEpoch(range.from);
      await waitPreviewDuration(run, item.durationSec);
    }
  }

  async function previewTransitionItem(run, api, source, target, config, index, total) {
    if (!source?.scene || !target?.scene?.camera) throw new Error('Automatic transition needs a saved source and destination View.');
    const durationSec = clampTransitionDurationSeconds(config?.durationSec, STORY_DEFAULTS.transitionDurationSec);
    const caption = String(config?.caption || '');
    const transitionLabel = `${source.name} → ${target.name}`;
    updatePreviewOverlay(source, index, total, {
      run,
      kind: durationSec > 0 ? 'travel' : 'cut',
      label: transitionLabel,
      text: caption
    });
    setStudioStatus(durationSec > 0
      ? `${storyboardRunLabel(run)} · ${index + 1}/${total} — travelling to ${target.name}`
      : `${storyboardRunLabel(run)} · ${index + 1}/${total} — cut to ${target.name}`,
    'active');

    // Deliberate sequencing: source View's display/epoch/layers remain active
    // during the whole flight. At the destination the target camera has fully
    // resolved, then the target View's visual state is applied.
    await api.flyToCameraState(target.scene.camera, durationSec);
    if (run.cancelled) return;
    await api.applyStudioSceneState(target.scene, { camera: false });
    run.arrivedAtViewId = target.id;
  }

  function introPhaseTiming(durationSec) {
    const totalDurationSec = clampDurationSeconds(durationSec, STORY_DEFAULTS.introDurationSec);
    // User-requested context/title dwell. It remains exactly 2.5 s whenever
    // the intro is long enough; short custom intros compress gracefully.
    const contextDurationSec = totalDurationSec >= 5
      ? 2.5
      : Math.max(0.8, Math.min(2.5, totalDurationSec * 0.5));
    const approachDurationSec = Math.max(0, totalDurationSec - contextDurationSec);
    // Reveal near the end of the approach. We intentionally make the visual
    // state swap clean and deterministic rather than attempting fragile
    // per-model alpha animation in the renderer.
    const revealDurationSec = approachDurationSec <= 0
      ? 0
      : Math.min(1.15, Math.max(0.65, approachDurationSec * 0.30));
    return { totalDurationSec, contextDurationSec, approachDurationSec, revealDurationSec };
  }

  async function previewIntroItem(run, api, item, targetView, index, total) {
    if (!targetView?.scene?.camera) throw new Error('Orbit Intro needs a first captured View with a saved camera.');
    if (typeof api.prepareStudioIntroContext !== 'function') {
      throw new Error('Viewer API is missing prepareStudioIntroContext(). Apply the Batch 8 viewer replacement.');
    }

    const timing = introPhaseTiming(item.durationSec);
    const title = previewTextForItem(item);
    if (run.introPreparedFor !== item.id) {
      await api.prepareStudioIntroContext(targetView.scene);
      run.introPreparedFor = item.id;
    }
    if (run.cancelled) return;

    updatePreviewOverlay(item, index, total, {
      run,
      kind: 'intro',
      label: 'Map context',
      text: title,
      centered: true
    });
    setStudioStatus(`${storyboardRunLabel(run)} · ${index + 1}/${total} — map context and title (${timing.contextDurationSec.toFixed(1)} s).`, 'active');
    await waitPreviewDuration(run, timing.contextDurationSec);
    if (run.cancelled) return;

    updatePreviewOverlay(item, index, total, {
      run,
      kind: 'intro',
      label: 'Curved approach',
      text: '',
      centered: true
    });
    setStudioStatus(`${storyboardRunLabel(run)} · ${index + 1}/${total} — curved approach to ${targetView.name}.`, 'active');

    if (timing.approachDurationSec <= 0.001) {
      await api.applyStudioSceneState(targetView.scene, { camera: true });
      run.arrivedAtViewId = targetView.id;
      return;
    }

    // Cesium computes the actual curved flight from the overview to the saved
    // View camera. We keep the map-only context active at first, then reveal
    // the target View's complete visual state during the final approach.
    const flight = api.flyToCameraState(targetView.scene.camera, timing.approachDurationSec);
    const mapOnlyApproachSec = Math.max(0, timing.approachDurationSec - timing.revealDurationSec);
    if (mapOnlyApproachSec > 0.001) {
      await waitPreviewDuration(run, mapOnlyApproachSec);
    }
    if (run.cancelled) {
      await flight;
      return;
    }

    if (timing.revealDurationSec > 0) {
      updatePreviewOverlay(item, index, total, {
        run,
        kind: 'intro',
        label: 'Parcel reveal',
        text: '',
        centered: true
      });
      setStudioStatus(`${storyboardRunLabel(run)} · ${index + 1}/${total} — revealing ${targetView.name}.`, 'active');
    }
    await api.applyStudioSceneState(targetView.scene, { camera: false });
    await flight;
    if (run.cancelled) return;

    // Re-apply after the camera settles so epoch, mode, map, exag, datum, and
    // layer state are exact immediately before View 01 begins its own hold.
    await api.applyStudioSceneState(targetView.scene, { camera: false });
    run.arrivedAtViewId = targetView.id;
  }

  async function runStoryboardPreview(options = {}) {
    requireStudioActive(options.mode === 'record' ? 'starting export' : 'starting Preview');
    if (!firstViewItem()) throw new Error('Capture at least one View before Preview or export.');
    if (isStoryboardPreviewing()) return state.studio.previewRun.promise;

    const api = await awaitViewerApi();
    if (typeof api.getStudioSceneState !== 'function' || typeof api.applyStudioSceneState !== 'function') {
      throw new Error('Viewer API is missing studio scene methods. Apply the Batch 6 viewer replacement.');
    }

    const timeline = buildPreviewTimeline();
    if (timeline.some((step) => step.kind === 'intro') && typeof api.prepareStudioIntroContext !== 'function') {
      throw new Error('Viewer API is missing Orbit Intro support. Apply the Batch 8 viewer replacement.');
    }
    const run = {
      active: true,
      mode: options.mode === 'record' ? 'record' : 'preview',
      cancelled: false,
      reason: null,
      waitCancel: null,
      arrivedAtViewId: null,
      initialScene: clonePlain(api.getStudioSceneState()),
      startedAt: Date.now(),
      promise: null
    };
    state.studio.previewRun = run;

    run.promise = (async () => {
      let runError = null;
      try {
        api.pauseEpochPlayback?.();
        api.setCameraInputEnabled?.(false);
        renderStoryboard();

        if (typeof options.beforeTimeline === 'function') {
          await options.beforeTimeline({ run, api, timeline });
        }

        for (let index = 0; index < timeline.length; index += 1) {
          if (run.cancelled) break;
          const step = timeline[index];
          if (step.kind === 'view') {
            const alreadyAtTarget = run.arrivedAtViewId === step.item.id;
            await previewViewItem(run, api, step.item, index, timeline.length, { camera: !alreadyAtTarget });
            run.arrivedAtViewId = null;
          } else if (step.kind === 'transition') {
            await previewTransitionItem(run, api, step.source, step.target, step.config, index, timeline.length);
          } else if (step.kind === 'intro') {
            await previewIntroItem(run, api, step.item, step.targetView, index, timeline.length);
          }
        }
      } catch (error) {
        runError = error;
        if (!run.cancelled) throw error;
      } finally {
        try { api.pauseEpochPlayback?.(); } catch (_) { /* best effort */ }
        try {
          if (typeof options.beforeRestore === 'function') {
            await options.beforeRestore({ run, api, timeline, error: runError });
          }
        } catch (exportStopError) {
          warn('Storyboard recording stopped with a warning:', exportStopError);
          if (!runError) runError = exportStopError;
        }
        try {
          if (run.initialScene) await api.applyStudioSceneState(run.initialScene);
        } catch (restoreError) {
          warn('Preview restored with a warning:', restoreError);
          if (!runError) runError = restoreError;
        }
        try { api.restoreCameraInput?.(); } catch (_) { /* best effort */ }
        clearPreviewOverlay();
        run.active = false;
        if (state.studio.previewRun === run) state.studio.previewRun = null;
        renderStoryboard();
        const verb = run.mode === 'record' ? 'Export' : 'Preview';
        if (runError && !run.cancelled) {
          setStudioStatus(`${verb} stopped with a warning: ${runError.message || runError}`, 'error');
        } else if (run.cancelled) {
          setStudioStatus(`${verb} stopped · authoring scene restored.`, 'muted');
        } else {
          setStudioStatus(`${verb} complete · authoring scene restored.`, 'ready');
        }
      }
      return {
        cancelled: run.cancelled,
        reason: run.reason,
        durationMs: Date.now() - run.startedAt
      };
    })();

    return run.promise;
  }

  async function prepareStoryboardExportStart(api, timeline) {
    const firstStep = timeline[0];
    if (!firstStep) return;
    if (firstStep.kind === 'intro' && firstStep.targetView?.scene) {
      if (typeof api.prepareStudioIntroContext !== 'function') {
        throw new Error('Viewer API is missing Orbit Intro support. Apply the Batch 8 viewer replacement.');
      }
      await api.prepareStudioIntroContext(firstStep.targetView.scene);
      const run = state.studio.previewRun;
      if (run) run.introPreparedFor = firstStep.item.id;
      updatePreviewOverlay(firstStep.item, 0, timeline.length, {
        run,
        kind: 'intro',
        label: 'Map context',
        text: previewTextForItem(firstStep.item),
        centered: true
      });
      return;
    }
    if (firstStep.kind === 'view' && firstStep.item?.scene) {
      await api.applyStudioSceneState(firstStep.item.scene, { camera: true });
      const range = normalizedEpochRange(firstStep.item);
      api.setEpoch(range.from);
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      updatePreviewOverlay(firstStep.item, 0, timeline.length, {
        run: state.studio.previewRun,
        kind: 'view',
        label: firstStep.item.name,
        text: previewTextForItem(firstStep.item)
      });
    }
  }

  async function recordStoryboard(options = {}) {
    requireStudioActive('starting export');
    if (!firstViewItem()) throw new Error('Capture at least one View before exporting.');
    if (isStoryboardPreviewing()) throw new Error('Stop the active Preview or export before starting a new export.');

    const api = await awaitViewerApi();
    if (typeof api.beginCaptureQuality !== 'function' || typeof api.endCaptureQuality !== 'function') {
      throw new Error('Viewer API is missing capture-quality methods. Apply the Batch 6 viewer replacement.');
    }

    const outputWidth = Math.max(2, Math.floor(Number(options.outputWidth) || DEFAULTS.outputWidth));
    const outputHeight = Math.max(2, Math.floor(Number(options.outputHeight) || DEFAULTS.outputHeight));
    const fps = Math.min(60, Math.max(12, Number(options.fps) || DEFAULTS.fps));
    const videoBitsPerSecond = Math.max(1_000_000, Number(options.videoBitsPerSecond) || DEFAULTS.storyboardExportBitrate);
    const viewfinder = getStudioViewfinderState();
    const qualityOptions = {
      outputWidth,
      outputHeight,
      maxResolutionScale: Number(options.maxResolutionScale) || 3.5
    };
    if (viewfinder?.cssCrop) qualityOptions.cropCssRect = viewfinder.cssCrop;

    let quality = null;
    let session = null;
    let blob = null;
    let playbackSummary = null;
    const filename = options.filename || createStoryboardFilename();
    state.lastError = null;

    try {
      quality = await api.beginCaptureQuality(qualityOptions);
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      playbackSummary = await runStoryboardPreview({
        mode: 'record',
        beforeTimeline: async ({ run, api: playbackApi, timeline }) => {
          if (run.cancelled) return;
          await prepareStoryboardExportStart(playbackApi, timeline);
          if (run.cancelled) return;
          session = createStoryboardRecordingSession({
            outputWidth,
            outputHeight,
            fps,
            videoBitsPerSecond,
            onError: (error) => {
              run.cancelled = true;
              run.reason = 'recorder error';
              run.recordingError = error;
              if (typeof run.waitCancel === 'function') run.waitCancel();
              try { playbackApi.pauseEpochPlayback?.(); } catch (_) { /* best effort */ }
              try {
                if (playbackApi.getCameraState && playbackApi.setCameraState) {
                  playbackApi.setCameraState(playbackApi.getCameraState());
                }
              } catch (_) { /* best effort */ }
            }
          });
          setStudioStatus('Export recording · compositor is capturing 1920×1080 WebM.', 'active');
        },
        beforeRestore: async ({ run }) => {
          if (session) blob = await finishStoryboardRecordingSession(session);
          if (run.recordingError) throw run.recordingError;
        }
      });

      if (playbackSummary.cancelled && !blob) {
        return {
          ok: false,
          cancelled: true,
          reason: playbackSummary.reason,
          filename: null,
          bytes: 0,
          output: { width: outputWidth, height: outputHeight },
          captureQuality: quality
        };
      }
      if (!blob) throw new Error('Export completed without a WebM blob.');
      if (playbackSummary.cancelled) {
        return {
          ok: false,
          cancelled: true,
          reason: playbackSummary.reason,
          filename: null,
          bytes: blob.size,
          output: { width: outputWidth, height: outputHeight },
          captureQuality: quality
        };
      }

      if (options.download !== false) downloadBlob(blob, filename);
      const summary = {
        ok: true,
        cancelled: false,
        filename,
        bytes: blob.size,
        durationMs: playbackSummary.durationMs,
        fps,
        mimeType: state.lastMimeType || session?.mimeType || 'video/webm',
        videoBitsPerSecond,
        output: { width: outputWidth, height: outputHeight },
        captureQuality: quality,
        viewfinder: getStudioViewfinderState(),
        storyboard: clonePlain(state.studio.storyboard)
      };
      log('Storyboard export complete:', summary);
      return summary;
    } finally {
      if (session && !session.finished) {
        try { await finishStoryboardRecordingSession(session); } catch (_) { /* surfaced by the run when relevant */ }
      }
      try { await api.endCaptureQuality(); } catch (error) { warn('Could not restore capture quality:', error); }
    }
  }

  async function stopStoryboardPreview(reason = 'stopped') {
    const run = state.studio.previewRun;
    if (!run) return false;
    run.cancelled = true;
    run.reason = reason;
    if (typeof run.waitCancel === 'function') run.waitCancel();
    const api = getViewerApi();
    try { api?.pauseEpochPlayback?.(); } catch (_) { /* best effort */ }
    try {
      if (api?.getCameraState && api?.setCameraState) api.setCameraState(api.getCameraState());
    } catch (_) { /* best effort */ }
    return run.promise;
  }

  function transitionDurationLabel(durationSec) {
    const seconds = clampTransitionDurationSeconds(durationSec, STORY_DEFAULTS.transitionDurationSec);
    return seconds <= 0 ? 'CUT' : `${seconds}s`;
  }

  function renderMoveControls(item, index, previewing) {
    if (item.type !== 'view') return '<span aria-hidden="true"></span>';
    const moveUpDisabled = !canMoveStoryboardItemUp(index);
    const moveDownDisabled = !canMoveStoryboardItemDown(index);
    return `
      <div class="studioModeShotReorder" aria-label="Reorder ${escapeHtml(item.name)}">
        <button class="studioModeShotMove" type="button" data-shot-move="up" data-shot-id="${escapeHtml(item.id)}" ${(previewing || moveUpDisabled) ? 'disabled' : ''} title="Move up">▲</button>
        <button class="studioModeShotMove" type="button" data-shot-move="down" data-shot-id="${escapeHtml(item.id)}" ${(previewing || moveDownDisabled) ? 'disabled' : ''} title="Move down">▼</button>
      </div>`;
  }

  function renderTemporalControls(item, previewing) {
    if (item.type !== 'view') return '';
    const range = ensureEpochRange(item);
    const { count } = epochTimeline();
    const isPlaying = range.behavior === 'play';
    const disabled = previewing || !isPlaying;
    return `
      <div class="studioModeTemporalBlock">
        <label class="studioModeField">
          <span class="studioModeFieldLabel">Epoch behavior</span>
          <select class="studioModeTextInput" data-shot-id="${escapeHtml(item.id)}" data-shot-field="epochBehavior" ${previewing ? 'disabled' : ''}>
            <option value="hold" ${!isPlaying ? 'selected' : ''}>Hold saved epoch</option>
            <option value="play" ${isPlaying ? 'selected' : ''}>Play epoch range</option>
          </select>
        </label>
        <div class="studioModeTemporalGrid">
          <label class="studioModeField">
            <span class="studioModeFieldLabel">Start</span>
            <input class="studioModeEpochRangeInput" type="range" min="0" max="${Math.max(0, count - 1)}" step="1" data-shot-id="${escapeHtml(item.id)}" data-shot-field="epochFrom" value="${range.from}" ${disabled ? 'disabled' : ''} />
            <span class="studioModeEpochLabel">${escapeHtml(epochLabelForIndex(range.from))}</span>
          </label>
          <label class="studioModeField">
            <span class="studioModeFieldLabel">End</span>
            <input class="studioModeEpochRangeInput" type="range" min="0" max="${Math.max(0, count - 1)}" step="1" data-shot-id="${escapeHtml(item.id)}" data-shot-field="epochTo" value="${range.to}" ${disabled ? 'disabled' : ''} />
            <span class="studioModeEpochLabel">${escapeHtml(epochLabelForIndex(range.to))}</span>
          </label>
        </div>
        <div class="studioModeTemporalHelp">${isPlaying ? 'This View duration is the playback duration. The exported WebM follows the same epoch range.' : 'Hold preserves the saved epoch. Choose Play epoch range to animate this View during Preview and export.'}</div>
      </div>`;
  }

  function renderViewOrIntroCard(item, index, selectedId, previewing) {
    const expanded = item.id === selectedId;
    const captionLabel = defaultCaptionLabel(item);
    const captionPlaceholder = defaultCaptionPlaceholder(item);
    const durationLabel = item.type === 'intro' ? 'Intro duration (s)' : 'View hold / play (s)';
    const detailNote = item.type === 'view'
      ? 'Captured scene state is locked for this View. To change camera or display setup, adjust the viewer and capture a new View. Its automatic transition is configured directly below this card.'
      : 'Only one Orbit Intro is allowed, and it stays at the top. It holds a 2.5-second map context and title, then curves toward the first View. Its parcels and saved visual state reveal during the final approach.';
    return `
      <article class="studioModeShotCard studioModeShotCard--${escapeHtml(item.type)}${expanded ? ' is-expanded' : ''}" data-shot-id="${escapeHtml(item.id)}" tabindex="0" aria-expanded="${expanded ? 'true' : 'false'}">
        <div class="studioModeShotCardTop">
          ${renderMoveControls(item, index, previewing)}
          <span class="studioModeShotType">${escapeHtml(shotTypeLabel(item.type))}</span>
          <span class="studioModeShotName">${escapeHtml(item.name)}</span>
          <span class="studioModeShotDuration">${escapeHtml(formatStoryboardDuration(item.durationSec))}</span>
          <button class="studioModeShotDelete" type="button" data-shot-delete="${escapeHtml(item.id)}" title="Delete ${escapeHtml(item.name)}" aria-label="Delete ${escapeHtml(item.name)}" ${previewing ? 'disabled' : ''}>×</button>
        </div>
        <div class="studioModeShotMeta">${escapeHtml(shotDescription(item))}</div>
        ${item.type === 'view' ? '<div class="studioModeShotState">camera + display state saved</div>' : ''}
        <div class="studioModeShotEditor" ${expanded ? '' : 'hidden'}>
          <div class="studioModeShotEditorGrid">
            <label class="studioModeField">
              <span class="studioModeFieldLabel">Shot name</span>
              <input class="studioModeTextInput" type="text" data-shot-id="${escapeHtml(item.id)}" data-shot-field="name" value="${escapeHtml(item.name)}" ${previewing ? 'disabled' : ''} />
            </label>
            <label class="studioModeField">
              <span class="studioModeFieldLabel">${durationLabel}</span>
              <input class="studioModeNumberInput" type="number" min="1" max="600" step="1" data-shot-id="${escapeHtml(item.id)}" data-shot-field="durationSec" value="${escapeHtml(item.durationSec)}" ${previewing ? 'disabled' : ''} />
            </label>
          </div>
          <label class="studioModeField">
            <span class="studioModeFieldLabel">${escapeHtml(captionLabel)}</span>
            <textarea class="studioModeTextarea" rows="2" data-shot-id="${escapeHtml(item.id)}" data-shot-field="caption" placeholder="${escapeHtml(captionPlaceholder)}" ${previewing ? 'disabled' : ''}>${escapeHtml(item.caption || '')}</textarea>
          </label>
          ${renderTemporalControls(item, previewing)}
          <div class="studioModeShotEditorNote">${escapeHtml(detailNote)}</div>
        </div>
      </article>`;
  }

  function renderTransitionCard(source, target, selectedId, previewing) {
    const transition = ensureTransitionConfig(source);
    const selectionId = transitionSelectionId(source.id);
    const expanded = selectionId === selectedId;
    const kind = transition.durationSec > 0 ? 'TRAVEL' : 'CUT';
    const summary = transition.durationSec > 0
      ? `${transition.durationSec}s camera movement · inherits ${source.name} display state`
      : `Instant jump · inherits ${source.name} until ${target.name} is reached`;
    return `
      <article class="studioModeShotCard studioModeShotCard--transition${expanded ? ' is-expanded' : ''}" data-transition-source-id="${escapeHtml(source.id)}" tabindex="0" aria-expanded="${expanded ? 'true' : 'false'}">
        <div class="studioModeShotCardTop">
          <span class="studioModeShotType">${kind}</span>
          <span class="studioModeShotName">${escapeHtml(source.name)} → ${escapeHtml(target.name)}</span>
          <span class="studioModeShotDuration">${escapeHtml(transitionDurationLabel(transition.durationSec))}</span>
        </div>
        <div class="studioModeShotMeta">${escapeHtml(summary)}</div>
        <div class="studioModeTransitionState">Auto transition. Source visual state remains active during travel; target camera and visual state resolve at arrival.</div>
        <div class="studioModeShotEditor" ${expanded ? '' : 'hidden'}>
          <label class="studioModeField studioModeTransitionDurationField">
            <span class="studioModeFieldLabel">Travel duration (s)</span>
            <input class="studioModeNumberInput" type="number" min="0" max="600" step="1" data-transition-source-id="${escapeHtml(source.id)}" data-transition-field="durationSec" value="${escapeHtml(transition.durationSec)}" ${previewing ? 'disabled' : ''} />
          </label>
          <label class="studioModeField">
            <span class="studioModeFieldLabel">Travel caption</span>
            <textarea class="studioModeTextarea" rows="2" data-transition-source-id="${escapeHtml(source.id)}" data-transition-field="caption" placeholder="Optional lower-third text during this travel" ${previewing ? 'disabled' : ''}>${escapeHtml(transition.caption || '')}</textarea>
          </label>
          <div class="studioModeShotEditorNote">Set duration to <b>0</b> for a clean cut. A positive duration lets Cesium calculate the camera path and resolve the destination heading and pitch.</div>
        </div>
      </article>`;
  }

  function renderStoryboard() {
    const { storyboardCards, storyboardDuration, storyboardTopDuration, addIntro, clearStoryboard, preview, record } = studioDom();
    const previewing = isStoryboardPreviewing();
    const recording = isStoryboardRecording();
    const total = storyboardTotalSeconds();
    const totalLabel = formatStoryboardDuration(total);
    if (storyboardDuration) storyboardDuration.textContent = totalLabel;
    if (storyboardTopDuration) storyboardTopDuration.textContent = `STORYBOARD · ${totalLabel}`;
    document.querySelectorAll('[data-studio-story-action]').forEach((button) => {
      button.disabled = !state.studio.apiReady || previewing;
    });
    if (addIntro) addIntro.disabled = !state.studio.apiReady || previewing || state.studio.storyboard.some((item) => item.type === 'intro');
    if (clearStoryboard) clearStoryboard.disabled = previewing || !state.studio.storyboard.length;
    if (preview) {
      preview.disabled = !state.studio.apiReady || recording || (!previewing && (!state.studio.active || !firstViewItem()));
      preview.textContent = previewing && !recording ? 'Stop Preview' : 'Preview';
      preview.title = previewing && !recording ? 'Stop the active storyboard preview' : 'Preview the current storyboard';
      preview.classList.toggle('is-previewing', previewing && !recording);
    }
    if (record) {
      record.disabled = !state.studio.apiReady || (!recording && (previewing || !state.studio.active || !firstViewItem()));
      record.textContent = recording ? 'Stop Export' : 'Save 1080p WebM';
      record.title = recording ? 'Stop the active export without saving a file' : 'Play and save the 16:9 storyboard as a 1080p WebM';
      record.classList.toggle('is-recording', recording);
    }
    updateLibraryUi();
    if (!storyboardCards) return;

    if (!state.studio.storyboard.length) {
      state.studio.selectedShotId = null;
      storyboardCards.innerHTML = `<div class="studioModeEmptyState">Capture Views to create your story. Transitions appear automatically between consecutive Views.</div>`;
      return;
    }

    const selectedId = selectedStoryboardItemId();
    const cards = [];
    state.studio.storyboard.forEach((item, index) => {
      if (item.type !== 'view' && item.type !== 'intro') return;
      cards.push(renderViewOrIntroCard(item, index, selectedId, previewing));
      if (item.type === 'view') {
        const next = followingViewItem(index);
        if (next) cards.push(renderTransitionCard(item, next.item, selectedId, previewing));
      }
    });
    storyboardCards.innerHTML = cards.join('');
  }

  function appendStoryboardItem(item) {
    state.studio.storyboard.push(item);
    state.studio.selectedShotId = item.id;
    scheduleStoryboardAutosave();
    renderStoryboard();
    return item;
  }

  function requireStudioActive(actionName) {
    if (!state.studio.active) throw new Error(`Open Studio Mode before ${actionName}.`);
  }

  function requireStudioEditable(actionName) {
    requireStudioActive(actionName);
    if (isStoryboardPreviewing()) throw new Error(`Stop the active Preview or export before ${actionName}.`);
  }

  async function captureStoryboardView() {
    requireStudioEditable('capturing a View');
    const api = await awaitViewerApi();
    if (typeof api.getStudioSceneState !== 'function') {
      throw new Error('Viewer API is missing getStudioSceneState(). Apply the Batch 6 viewer replacement.');
    }
    const scene = api.getStudioSceneState();
    const item = appendStoryboardItem({
      id: nextStoryboardId(),
      type: 'view',
      name: nextViewName(),
      durationSec: STORY_DEFAULTS.viewDurationSec,
      caption: '',
      transitionToNext: { durationSec: STORY_DEFAULTS.transitionDurationSec, caption: '' },
      scene: clonePlain(scene),
      epochRange: clonePlain(scene.epochRange || { from: scene.epoch, to: scene.epoch, behavior: 'hold' })
    });
    setStudioStatus(`Saved ${item.name} · ${scene.epochLabel || `epoch ${scene.epoch}`}. Add another View to reveal its automatic transition.`, 'ready');
    log('Storyboard View captured:', clonePlain(item));
    return clonePlain(item);
  }

  function addStoryboardIntro() {
    requireStudioEditable('adding an intro');
    const existing = state.studio.storyboard.find((item) => item.type === 'intro');
    if (existing) {
      setStudioStatus('Orbit Intro already exists at the top of this storyboard.', 'muted');
      return clonePlain(existing);
    }
    const item = {
      id: nextStoryboardId(),
      type: 'intro',
      name: 'Orbit Intro',
      durationSec: STORY_DEFAULTS.introDurationSec,
      caption: 'Ground deformation overview',
      preset: 'orbit-v2-map-context'
    };
    state.studio.storyboard.unshift(item);
    state.studio.selectedShotId = item.id;
    scheduleStoryboardAutosave();
    renderStoryboard();
    setStudioStatus('Orbit Intro added at the top.', 'ready');
    return clonePlain(item);
  }

  function removeStoryboardItem(itemId) {
    const index = findStoryboardIndex(itemId);
    if (index < 0) return false;
    const [removed] = state.studio.storyboard.splice(index, 1);
    if (state.studio.selectedShotId === itemId || transitionSourceIdFromSelection(state.studio.selectedShotId) === itemId) {
      const fallback = state.studio.storyboard[Math.min(index, state.studio.storyboard.length - 1)] || state.studio.storyboard[index - 1] || null;
      state.studio.selectedShotId = fallback ? fallback.id : null;
    }
    scheduleStoryboardAutosave();
    renderStoryboard();
    setStudioStatus(`${removed.name} removed. Automatic transitions updated.`, 'muted');
    return true;
  }

  function clearStoryboard() {
    const count = state.studio.storyboard.length;
    state.studio.storyboard.length = 0;
    state.studio.selectedShotId = null;
    scheduleStoryboardAutosave();
    renderStoryboard();
    setStudioStatus(count ? 'Storyboard cleared.' : 'Storyboard is already empty.', 'muted');
    return count;
  }

  function getStoryboard() {
    return clonePlain(state.studio.storyboard);
  }

  function bindStudioControls() {
    const { mount, header, exit, captureView, addIntro, clearStoryboard: clearBtn, preview, record, projectName, saveProject, loadProject, refreshLibrary, libraryFiles, storyboardCards } = studioDom();
    if (!mount || mount.dataset.studioControlsBound === 'true') return;
    mount.dataset.studioControlsBound = 'true';

    // Exactly one click route for each Studio control. No global click fixups.
    if (header) {
      header.onclick = async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          if (state.studio.active) await exitStudioMode();
          else await enterStudioMode();
        } catch (error) {
          handleStudioError(error);
        }
      };
    }

    if (exit) {
      exit.onclick = async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          await exitStudioMode();
        } catch (error) {
          handleStudioError(error);
        }
      };
    }

    if (captureView) {
      captureView.onclick = async () => {
        try { await captureStoryboardView(); }
        catch (error) { handleStudioError(error); }
      };
    }
    if (addIntro) {
      addIntro.onclick = () => {
        try { addStoryboardIntro(); }
        catch (error) { handleStudioError(error); }
      };
    }
    if (preview) {
      preview.onclick = async () => {
        try {
          if (isStoryboardRecording()) return;
          if (isStoryboardPreviewing()) await stopStoryboardPreview('user stopped preview');
          else await runStoryboardPreview();
        } catch (error) {
          handleStudioError(error);
        }
      };
    }
    if (record) {
      record.onclick = async () => {
        try {
          if (isStoryboardRecording()) await stopStoryboardPreview('user stopped export');
          else await recordStoryboard();
        } catch (error) {
          handleStudioError(error);
        }
      };
    }
    if (clearBtn) {
      clearBtn.onclick = () => {
        try {
          if (!state.studio.storyboard.length) return;
          if (window.confirm('Clear the entire storyboard?')) clearStoryboard();
        } catch (error) {
          handleStudioError(error);
        }
      };
    }
    if (projectName) {
      projectName.onchange = () => {
        state.studio.projectName = normalizedProjectName(projectName.value);
        scheduleStoryboardAutosave();
        updateLibraryUi();
      };
      projectName.oninput = () => { state.studio.projectName = projectName.value; };
    }
    if (saveProject) {
      saveProject.onclick = async () => {
        try { await saveStoryboardProject(); }
        catch (error) { handleStudioError(error); }
      };
    }
    if (refreshLibrary) {
      refreshLibrary.onclick = async () => {
        try { await refreshStoryboardLibrary({ status: true }); }
        catch (error) { handleStudioError(error); }
      };
    }
    if (libraryFiles) {
      libraryFiles.onchange = () => {
        state.studio.library.selectedFile = libraryFiles.value;
        updateLibraryUi();
      };
    }
    if (loadProject) {
      loadProject.onclick = async () => {
        try { await loadStoryboardProject(); }
        catch (error) { handleStudioError(error); }
      };
    }
    if (storyboardCards) {
      storyboardCards.onclick = (event) => {
        const element = event.target instanceof Element ? event.target : null;
        if (!element || isStoryboardPreviewing()) return;

        const deleteButton = element.closest('[data-shot-delete]');
        if (deleteButton) {
          event.preventDefault();
          removeStoryboardItem(deleteButton.getAttribute('data-shot-delete'));
          return;
        }

        const moveButton = element.closest('[data-shot-move]');
        if (moveButton) {
          event.preventDefault();
          moveStoryboardItem(moveButton.getAttribute('data-shot-id'), moveButton.getAttribute('data-shot-move'));
          return;
        }

        if (element.closest('input, textarea, select, label, button')) return;
        const transitionCard = element.closest('[data-transition-source-id]');
        if (transitionCard) {
          event.preventDefault();
          selectStoryboardItem(transitionSelectionId(transitionCard.getAttribute('data-transition-source-id')));
          return;
        }
        const card = element.closest('[data-shot-id]');
        if (card) {
          event.preventDefault();
          selectStoryboardItem(card.getAttribute('data-shot-id'));
        }
      };

      storyboardCards.onfocusin = (event) => {
        if (isStoryboardPreviewing()) return;
        const element = event.target instanceof Element ? event.target : null;
        const transitionCard = element ? element.closest('[data-transition-source-id]') : null;
        if (transitionCard) {
          selectStoryboardItem(transitionSelectionId(transitionCard.getAttribute('data-transition-source-id')));
          return;
        }
        const card = element ? element.closest('[data-shot-id]') : null;
        if (card) selectStoryboardItem(card.getAttribute('data-shot-id'));
      };

      storyboardCards.oninput = (event) => {
        if (isStoryboardPreviewing()) return;
        const element = event.target instanceof Element ? event.target.closest('[data-shot-field]') : null;
        const field = String(element?.getAttribute('data-shot-field') || '');
        if (!element || (field !== 'epochFrom' && field !== 'epochTo')) return;
        const label = element.parentElement?.querySelector('.studioModeEpochLabel');
        if (label) label.textContent = epochLabelForIndex(element.value);
      };

      storyboardCards.onchange = (event) => {
        if (isStoryboardPreviewing()) return;
        const element = event.target instanceof Element ? event.target : null;
        if (!element) return;
        const transitionField = element.closest('[data-transition-field]');
        if (transitionField) {
          updateTransition(
            transitionField.getAttribute('data-transition-source-id'),
            transitionField.getAttribute('data-transition-field'),
            'value' in transitionField ? transitionField.value : transitionField.textContent
          );
          return;
        }
        const fieldElement = element.closest('[data-shot-field]');
        if (!fieldElement) return;
        updateStoryboardItem(
          fieldElement.getAttribute('data-shot-id'),
          fieldElement.getAttribute('data-shot-field'),
          'value' in fieldElement ? fieldElement.value : fieldElement.textContent
        );
      };
    }
  }

  function handleStudioError(error) {
    const message = error?.message || String(error || 'Unknown Studio Mode error');
    state.studio.lastError = message;
    setStudioStatus(message, 'error');
    warn('Studio Mode:', message);
  }

  async function enterStudioMode() {
    if (state.studio.active) return state.studio.entrySnapshot;
    const api = await awaitViewerApi();
    ensureExistingDrawerOpen();

    state.studio.entrySnapshot = typeof api.onCaptureModeEnter === 'function'
      ? await api.onCaptureModeEnter()
      : null;
    if (typeof api.beginStudioAuthoringControls === 'function') api.beginStudioAuthoringControls();
    if (typeof api.hideMainUI === 'function') api.hideMainUI();

    state.studio.active = true;
    document.body.classList.add('studio-mode-active');
    showStudioViewfinder();
    const { mount, header } = studioDom();
    mount?.classList.add('studioModeActive');
    if (header) {
      header.querySelector('.studioModeDockHeaderMain').textContent = 'Studio Mode';
    }
    setStudioHeaderState(true);
    setStudioDockOpen(true);
    renderStoryboard();
    setStudioStatus('Studio active · capture Views. Transitions appear automatically.', 'active');
    log('Studio Mode entered.');
    return state.studio.entrySnapshot;
  }

  async function exitStudioMode() {
    if (state.studio.exitPromise) return state.studio.exitPromise;

    state.studio.exiting = true;
    state.studio.exitPromise = (async () => {
      const wasActive = Boolean(state.studio.active);

      // Stop Preview before closing the dock so it can restore the current
      // authoring scene and release camera input cleanly.
      await stopStoryboardPreview('studio exit');
      // Close the dock first. The user gets immediate visual feedback even if
      // restoring a GLB mode, imagery, or camera takes a few render frames.
      stopActiveCapture();
      state.studio.active = false;
      hideStudioViewfinder();
      document.body.classList.remove('studio-mode-active');
      const { mount } = studioDom();
      mount?.classList.remove('studioModeActive');
      setStudioDockOpen(false);
      setStudioHeaderState(false);
      setStudioStatus('Closing Studio Mode…', 'muted');

      if (!wasActive) {
        setStudioStatus('Studio shell ready.', 'muted');
        return null;
      }

      let restoreWarning = null;
      try {
        const api = await awaitViewerApi();
        if (typeof api.endStudioAuthoringControls === 'function') api.endStudioAuthoringControls();
        if (typeof api.onCaptureModeExit === 'function') {
          await api.onCaptureModeExit();
        } else if (typeof api.showMainUI === 'function') {
          api.showMainUI();
        }
      } catch (error) {
        restoreWarning = error;
        // A safety fallback: never strand the normal Proto2 interface because
        // a non-critical state-restoration call had a problem.
        try {
          const api = getViewerApi();
          if (api && typeof api.showMainUI === 'function') api.showMainUI();
        } catch (_) { /* best effort only */ }
      } finally {
        state.studio.entrySnapshot = null;
        state.studio.exiting = false;
        setStudioDockOpen(false);
        setStudioHeaderState(false);
      }

      if (restoreWarning) {
        warn('Studio Mode closed, but some viewer state may not have restored:', restoreWarning);
        setStudioStatus('Studio closed · normal UI restored with a warning.', 'error');
      } else {
        setStudioStatus('Studio shell ready.', 'muted');
        log('Studio Mode exited.');
      }
      return true;
    })();

    try {
      return await state.studio.exitPromise;
    } finally {
      state.studio.exitPromise = null;
    }
  }

  function isStudioModeActive() {
    return Boolean(state.studio.active);
  }

  async function initialiseStoryboardProject(api) {
    try {
      const metadata = typeof api.getStudioProjectMeta === 'function' ? await api.getStudioProjectMeta() : null;
      state.studio.projectMeta = metadata && typeof metadata === 'object' ? clonePlain(metadata) : { projectId: window.location.pathname, epochCount: 1, epochLabels: [] };
      state.studio.projectName = normalizedProjectName(state.studio.projectName || defaultProjectName(state.studio.projectMeta));
      restoreStoryboardAutosave();
      updateLibraryUi();
      // Non-blocking: the local library service is optional because autosave
      // must work even when the user only runs Live Server.
      refreshStoryboardLibrary({ status: false }).catch(() => {});
      scheduleStoryboardAutosave();
    } catch (error) {
      warn('Could not initialise storyboard persistence:', error);
      state.studio.projectMeta = { projectId: window.location.pathname, epochCount: 1, epochLabels: [] };
      state.studio.projectName = normalizedProjectName(state.studio.projectName || defaultProjectName());
      updateLibraryUi();
    }
  }

  function bootstrapStudioDock() {
    injectBatch5Styles();
    renderStudioDock();
    awaitViewerApi()
      .then(async (api) => {
        state.studio.apiReady = true;
        await initialiseStoryboardProject(api);
        setStudioControlsReady(true);
        renderStoryboard();
        if (!state.studio.autosaveLoaded) setStudioStatus('Ready · capture Views. Autosave is active.', 'ready');
      })
      .catch(handleStudioError);
  }

  function stopActiveCapture() {
    if (state.recorder && state.recorder.state !== 'inactive') state.recorder.stop();
    stopRenderLoop();
  }

  const StudioMode = window.StudioMode || {};
  StudioMode.version = VERSION;
  StudioMode.getDiagnostics = getDiagnostics;
  StudioMode.awaitViewerApi = awaitViewerApi;
  StudioMode.drawBatch0Frame = (options) => drawFrame(options);
  StudioMode.runBatch0Probe = runBatch0Probe;
  StudioMode.runBatch1QualityProbe = runBatch1QualityProbe;
  StudioMode.stop = stopActiveCapture;
  StudioMode.getLastBlob = () => state.lastBlob;
  StudioMode.enter = enterStudioMode;
  StudioMode.exit = exitStudioMode;
  StudioMode.isActive = isStudioModeActive;
  StudioMode.getViewfinderState = () => getStudioViewfinderState();
  StudioMode.runBatch3ViewfinderProbe = runBatch3ViewfinderProbe;
  StudioMode.getStoryboard = getStoryboard;
  StudioMode.clearStoryboard = clearStoryboard;
  StudioMode.saveStoryboardProject = saveStoryboardProject;
  StudioMode.loadStoryboardProject = loadStoryboardProject;
  StudioMode.refreshStoryboardLibrary = refreshStoryboardLibrary;
  StudioMode.getStoryboardDocument = storyboardDocument;
  StudioMode.preview = runStoryboardPreview;
  StudioMode.record = recordStoryboard;
  StudioMode.stopPreview = stopStoryboardPreview;
  StudioMode.stopRecord = (reason = 'stopped') => isStoryboardRecording() ? stopStoryboardPreview(reason) : false;
  StudioMode.isPreviewing = isStoryboardPreviewing;
  StudioMode.isRecording = isStoryboardRecording;
  StudioMode.mountStudioDock = bootstrapStudioDock;
  window.StudioMode = StudioMode;

  window.addEventListener('resize', () => {
    requestStudioViewfinderLayout();
  }, { passive: true });

  // Escape is a useful non-mouse escape hatch while the studio dock is active.
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !state.studio.active) return;
    event.preventDefault();
    if (isStoryboardPreviewing()) stopStoryboardPreview('escape').catch(handleStudioError);
    else exitStudioMode().catch(handleStudioError);
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrapStudioDock, { once: true });
  } else {
    bootstrapStudioDock();
  }

  window.dispatchEvent(new CustomEvent('studio-mode-ready', { detail: { version: VERSION } }));
  log(`Batch 9 ready (${VERSION}). Views can play epoch ranges; storyboards autosave locally and save/load through the local library folder.`);
})();

