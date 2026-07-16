const STORAGE_KEY = 'proto1.studioCaptureStoryboard.v1';
const PREVIEW_STEP_MS = 1750;
const PREVIEW_CAMERA_DURATION_MS = 1250;

function nowIso() {
  return new Date().toISOString();
}

function uid(prefix = 'view') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function safeText(value, fallback = '') {
  return String(value ?? fallback).replace(/[<>]/g, '').trim();
}

function clampIndex(index, length) {
  return Math.max(0, Math.min(Math.max(0, length - 1), Math.round(Number(index) || 0)));
}

function readStoryboard() {
  try {
    const raw = window.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    const views = Array.isArray(parsed?.views) ? parsed.views : Array.isArray(parsed) ? parsed : [];
    return views
      .filter((view) => view && view.camera && view.scene)
      .map((view, index) => ({
        id: String(view.id || uid('view')),
        title: safeText(view.title, `View ${index + 1}`),
        camera: view.camera,
        scene: view.scene,
        createdAt: String(view.createdAt || nowIso()),
        note: safeText(view.note, ''),
      }));
  } catch (error) {
    console.warn('[Studio capture] Failed to read storyboard:', error);
    return [];
  }
}

export function createStudioCaptureMode(options = {}) {
  return new StudioCaptureMode(options);
}

class StudioCaptureMode {
  constructor(options = {}) {
    this.elements = options.elements ?? {};
    this.api = options.api ?? {};
    this.views = readStoryboard();
    this.viewfinderVisible = false;
    this.previewing = false;
    this.previewIndex = -1;
    this.previewTimer = null;
    this.bindUi();
    this.render();
  }

  destroy() {
    if (this.previewTimer) window.clearTimeout(this.previewTimer);
    this.previewTimer = null;
    document.body.classList.remove('studio-viewfinder-on');
  }

  handleResize() {
    // CSS-driven viewfinder. This method exists as a future recording hook.
  }

  refresh() {
    this.render();
  }

  persist() {
    try {
      window.localStorage?.setItem(STORAGE_KEY, JSON.stringify({version: 1, views: this.views}));
    } catch (error) {
      console.warn('[Studio capture] Failed to save storyboard:', error);
      this.flash(`Storyboard save failed: ${error?.message ?? String(error)}`, 'error');
    }
  }

  flash(message, tone = 'info') {
    this.api.flashStatus?.(message, tone);
  }

  bindUi() {
    this.elements.viewfinderToggle?.addEventListener('click', () => this.toggleViewfinder());
    this.elements.captureButton?.addEventListener('click', () => this.captureCurrentView());
    this.elements.introButton?.addEventListener('click', () => this.addIntroView());
    this.elements.previewButton?.addEventListener('click', () => this.previewStoryboard());
    this.elements.clearButton?.addEventListener('click', () => this.clearStoryboard());
    this.elements.list?.addEventListener('click', (event) => {
      const button = event.target?.closest?.('[data-studio-capture-action]');
      if (!button) return;
      const action = button.dataset.studioCaptureAction;
      const id = button.dataset.studioCaptureId;
      if (action === 'goto') this.gotoView(id);
      if (action === 'delete') this.deleteView(id);
    });
  }

  toggleViewfinder(force = null) {
    this.viewfinderVisible = force === null ? !this.viewfinderVisible : Boolean(force);
    document.body.classList.toggle('studio-viewfinder-on', this.viewfinderVisible);
    if (this.elements.viewfinderToggle) {
      this.elements.viewfinderToggle.setAttribute('aria-pressed', this.viewfinderVisible ? 'true' : 'false');
      this.elements.viewfinderToggle.textContent = this.viewfinderVisible ? 'Frame on' : '16:9 frame';
      this.elements.viewfinderToggle.title = this.viewfinderVisible ? 'Hide 16:9 capture frame' : 'Show 16:9 capture frame';
    }
  }

  nextTitle(prefix = 'View') {
    return `${prefix} ${this.views.length + 1}`;
  }

  capturePayload(title, {cameraOverride = null, note = ''} = {}) {
    const camera = cameraOverride ?? this.api.getCameraState?.();
    const scene = this.api.getSceneState?.();
    if (!camera || !scene) return null;
    return {
      id: uid('view'),
      title,
      note,
      camera,
      scene,
      createdAt: nowIso(),
    };
  }

  captureCurrentView() {
    if (!this.api.isReady?.()) {
      this.flash('Studio capture: viewer still loading', 'error');
      return;
    }
    const view = this.capturePayload(this.nextTitle('View'), {note: 'Captured current view'});
    if (!view) return;
    this.views.push(view);
    this.persist();
    this.render();
    this.flash(`Captured ${view.title}`, 'ok');
  }

  addIntroView() {
    if (!this.api.isReady?.()) {
      this.flash('Studio capture: viewer still loading', 'error');
      return;
    }
    const introCamera = this.api.getIntroCameraState?.() ?? this.api.getCameraState?.();
    const view = this.capturePayload(this.nextTitle('Intro'), {
      cameraOverride: introCamera,
      note: 'Wide intro context view',
    });
    if (!view) return;
    view.title = this.views.length ? this.nextTitle('Intro') : 'Intro view';
    this.views.push(view);
    this.persist();
    this.render();
    this.flash('Intro view added', 'ok');
  }

  clearStoryboard() {
    if (!this.views.length) return;
    const ok = window.confirm?.('Clear all captured storyboard views?');
    if (ok === false) return;
    this.stopPreview();
    this.views = [];
    this.persist();
    this.render();
    this.flash('Storyboard cleared', 'ok');
  }

  deleteView(id) {
    const before = this.views.length;
    this.views = this.views.filter((view) => view.id !== id);
    if (this.views.length === before) return;
    this.persist();
    this.render();
  }

  viewById(id) {
    return this.views.find((view) => view.id === id) ?? null;
  }

  gotoView(id, {duration = 760} = {}) {
    const view = this.viewById(id);
    if (!view || !this.api.isReady?.()) return;
    this.api.applySceneState?.(view.scene);
    window.setTimeout(() => {
      this.api.applyCameraState?.(view.camera, {duration});
    }, 40);
    this.api.openStudioDrawer?.();
  }

  stopPreview() {
    if (this.previewTimer) window.clearTimeout(this.previewTimer);
    this.previewTimer = null;
    this.previewing = false;
    this.previewIndex = -1;
    this.render();
  }

  previewStoryboard() {
    if (!this.views.length || !this.api.isReady?.()) return;
    if (this.previewing) {
      this.stopPreview();
      return;
    }
    this.previewing = true;
    this.previewIndex = -1;
    this.toggleViewfinder(true);
    this.runPreviewStep();
  }

  runPreviewStep() {
    if (!this.previewing) return;
    this.previewIndex += 1;
    if (this.previewIndex >= this.views.length) {
      this.stopPreview();
      this.flash('Storyboard preview done', 'ok');
      return;
    }
    const view = this.views[this.previewIndex];
    this.render();
    this.gotoView(view.id, {duration: PREVIEW_CAMERA_DURATION_MS});
    this.previewTimer = window.setTimeout(() => this.runPreviewStep(), PREVIEW_STEP_MS);
  }

  metaForView(view, index) {
    const epoch = Number(view?.scene?.activeEpoch ?? 0);
    const epochCount = Number(this.api.getEpochCount?.() ?? 0);
    const epochNumber = epochCount ? `${clampIndex(epoch, epochCount) + 1}/${epochCount}` : `#${epoch + 1}`;
    const date = safeText(this.api.getEpochLabel?.(epoch), `Epoch ${epoch + 1}`);
    const mode = (view?.scene?.sceneMode === '2d') ? '2D' : '3D';
    const zoom = Number(view?.camera?.zoom);
    const pitch = Number(view?.camera?.pitch);
    const bearing = Number(view?.camera?.bearing);
    return `${index + 1}. ${mode} · ${date} · ${epochNumber} · z${Number.isFinite(zoom) ? zoom.toFixed(2) : '—'} · p${Number.isFinite(pitch) ? pitch.toFixed(0) : '—'} · b${Number.isFinite(bearing) ? bearing.toFixed(0) : '—'}`;
  }

  render() {
    if (this.elements.status) {
      this.elements.status.textContent = this.views.length
        ? `${this.views.length} view${this.views.length === 1 ? '' : 's'}`
        : 'No views';
    }
    if (this.elements.previewButton) {
      this.elements.previewButton.textContent = this.previewing ? 'Stop' : 'Preview';
      this.elements.previewButton.disabled = !this.views.length;
    }
    if (this.elements.clearButton) this.elements.clearButton.disabled = !this.views.length;
    if (this.elements.viewfinderToggle) {
      this.elements.viewfinderToggle.setAttribute('aria-pressed', this.viewfinderVisible ? 'true' : 'false');
      this.elements.viewfinderToggle.textContent = this.viewfinderVisible ? 'Frame on' : '16:9 frame';
    }
    if (!this.elements.list) return;
    if (!this.views.length) {
      this.elements.list.innerHTML = '<div class="studioCaptureEmpty">Captured camera views will appear here.</div>';
      return;
    }
    this.elements.list.innerHTML = this.views.map((view, index) => `
      <div class="studioCaptureCard ${this.previewing && index === this.previewIndex ? 'isPreviewing' : ''}">
        <div class="studioCaptureCardText">
          <strong>${safeText(view.title, `View ${index + 1}`)}</strong>
          <span>${safeText(this.metaForView(view, index))}</span>
        </div>
        <div class="studioCaptureCardActions">
          <button type="button" data-studio-capture-action="goto" data-studio-capture-id="${view.id}">Go</button>
          <button type="button" data-studio-capture-action="delete" data-studio-capture-id="${view.id}">Delete</button>
        </div>
      </div>
    `).join('');
  }
}
