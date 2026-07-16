import {PathLayer, SolidPolygonLayer, TextLayer} from '@deck.gl/layers';

const STORAGE_KEY = 'proto1.studioPolygonAnnotations.v1';
const MAX_POINTS = 32;
const MIN_POINTS = 3;
const DEFAULT_NAME = 'Subsidence bowl';
const POLYGON_TOP_Z = 0.08;
const LABEL_Z = 1.25;
// Keep the curtain as a subsurface marker: polygon/outline stays on the datum,
// while the curtain starts just below Z=0 so flat caps can hide it with depth.
const CURTAIN_TOP_Z = -0.75;
const CURTAIN_BOTTOM_Z = -2520;
const FILL_COLOR = [0, 221, 255, 48];
const FILL_COLOR_2D = [0, 226, 255, 54];
const OUTLINE_COLOR = [76, 245, 255, 242];
const PREVIEW_FILL_COLOR = [0, 226, 255, 36];
const PREVIEW_OUTLINE_COLOR = [143, 252, 255, 245];
const CURTAIN_COLOR = [0, 215, 255, 58];
const CURTAIN_EDGE_COLOR = [96, 248, 255, 172];
const LABEL_COLOR = [218, 253, 255, 255];
const LABEL_BACKGROUND_COLOR = [4, 26, 34, 215];

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function sanitizeText(value, fallback = '') {
  return String(value ?? fallback).replace(/[<>]/g, '').trim();
}

function normalizeCoordinate(coord) {
  const lon = Number(coord?.[0]);
  const lat = Number(coord?.[1]);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  return [lon, lat];
}

function closePath(points, z = null) {
  const clean = safeArray(points).map(normalizeCoordinate).filter(Boolean);
  if (!clean.length) return [];
  const mapped = clean.map(([lon, lat]) => z === null ? [lon, lat] : [lon, lat, z]);
  const first = mapped[0];
  const last = mapped[mapped.length - 1];
  if (first && last && (first[0] !== last[0] || first[1] !== last[1])) mapped.push([...first]);
  return mapped;
}

function polygonAreaCentroid(points) {
  const coords = safeArray(points).map(normalizeCoordinate).filter(Boolean);
  if (!coords.length) return [0, 0];
  let area2 = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < coords.length; i += 1) {
    const [x0, y0] = coords[i];
    const [x1, y1] = coords[(i + 1) % coords.length];
    const cross = x0 * y1 - x1 * y0;
    area2 += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  if (Math.abs(area2) > 1e-12) {
    return [cx / (3 * area2), cy / (3 * area2)];
  }
  const sum = coords.reduce((acc, [lon, lat]) => [acc[0] + lon, acc[1] + lat], [0, 0]);
  return [sum[0] / coords.length, sum[1] / coords.length];
}

function curtainOpacityScale(verticalExaggeration) {
  const value = Number(verticalExaggeration);
  if (!Number.isFinite(value)) return 1;
  if (value <= 0.15) return 0;
  if (value >= 2.5) return 1;
  return Math.max(0, Math.min(1, (value - 0.15) / 2.35));
}

function scaleColorAlpha(color, scale) {
  const out = [...color];
  out[3] = Math.round((Number(out[3]) || 0) * Math.max(0, Math.min(1, scale)));
  return out;
}

function edgeCurtains(points) {
  const coords = safeArray(points).map(normalizeCoordinate).filter(Boolean);
  if (coords.length < 2) return [];
  return coords.map((a, index) => {
    const b = coords[(index + 1) % coords.length];
    return {
      polygon: [
        [a[0], a[1], CURTAIN_TOP_Z],
        [b[0], b[1], CURTAIN_TOP_Z],
        [b[0], b[1], CURTAIN_BOTTOM_Z],
        [a[0], a[1], CURTAIN_BOTTOM_Z],
      ],
      path: [
        [a[0], a[1], CURTAIN_TOP_Z],
        [a[0], a[1], CURTAIN_BOTTOM_Z],
      ],
    };
  });
}

function parseStoredPolygons() {
  try {
    const raw = window.localStorage?.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return safeArray(parsed?.polygons ?? parsed)
      .map((item) => {
        const coordinates = safeArray(item.coordinates).map(normalizeCoordinate).filter(Boolean);
        if (coordinates.length < MIN_POINTS) return null;
        return {
          id: sanitizeText(item.id, `poly-${Date.now()}`) || `poly-${Date.now()}`,
          name: sanitizeText(item.name, DEFAULT_NAME) || DEFAULT_NAME,
          info: sanitizeText(item.info, ''),
          coordinates,
          visible: item.visible !== false,
          createdAt: sanitizeText(item.createdAt, new Date().toISOString()),
          updatedAt: sanitizeText(item.updatedAt, item.createdAt ?? new Date().toISOString()),
        };
      })
      .filter(Boolean);
  } catch (error) {
    console.warn('[Studio polygons] Failed to load polygon annotations:', error);
    return [];
  }
}

export function createStudioPolygonAnnotations(options = {}) {
  return new StudioPolygonAnnotations(options);
}

class StudioPolygonAnnotations {
  constructor(options = {}) {
    this.map = options.map ?? null;
    this.elements = options.elements ?? {};
    this.callbacks = {
      force2D: typeof options.force2D === 'function' ? options.force2D : () => {},
      requestRedraw: typeof options.requestRedraw === 'function' ? options.requestRedraw : () => {},
      openStudioDrawer: typeof options.openStudioDrawer === 'function' ? options.openStudioDrawer : () => {},
      flashStatus: typeof options.flashStatus === 'function' ? options.flashStatus : () => {},
    };
    this.polygons = parseStoredPolygons();
    this.showAll = true;
    this.drawing = false;
    this.naming = false;
    this.points = [];
    this.defaultName = DEFAULT_NAME;
    this.defaultInfo = '';
    this.revision = 0;
    this.bindUi();
    this.renderUi();
  }

  isDrawing() {
    return this.drawing;
  }

  setShowAll(value) {
    this.showAll = Boolean(value);
    this.renderUi();
    this.requestRedraw();
  }

  toggleShowAll() {
    this.setShowAll(!this.showAll);
  }

  resetVisualDefaults() {
    this.showAll = true;
    this.renderUi();
    this.requestRedraw();
  }

  startDrawing({sourceCell = null, name = DEFAULT_NAME, info = ''} = {}) {
    this.callbacks.force2D();
    this.callbacks.openStudioDrawer();
    this.defaultName = sanitizeText(name || sourceCell?.rumId || DEFAULT_NAME, DEFAULT_NAME) || DEFAULT_NAME;
    this.defaultInfo = sanitizeText(info, '');
    this.drawing = true;
    this.naming = false;
    this.points = [];
    this.elements.nameInput && (this.elements.nameInput.value = this.defaultName);
    this.elements.infoInput && (this.elements.infoInput.value = this.defaultInfo);
    document.body.classList.add('studio-polygon-drawing');
    this.map?.getCanvas?.().classList?.add('studioPolygonCrosshair');
    this.renderUi();
    this.callbacks.flashStatus?.('Polygon drawing · click map points, then Finish', 'info');
    this.requestRedraw();
  }

  cancelDrawing() {
    this.drawing = false;
    this.naming = false;
    this.points = [];
    document.body.classList.remove('studio-polygon-drawing');
    this.map?.getCanvas?.().classList?.remove('studioPolygonCrosshair');
    this.renderUi();
    this.requestRedraw();
  }

  undoPoint() {
    if (!this.drawing || !this.points.length) return;
    this.points.pop();
    this.naming = false;
    this.renderUi();
    this.requestRedraw();
  }

  addPoint(coord) {
    if (!this.drawing || this.naming) return false;
    const normalized = normalizeCoordinate(coord);
    if (!normalized) return false;
    if (this.points.length >= MAX_POINTS) {
      this.callbacks.flashStatus?.(`Polygon point limit reached (${MAX_POINTS})`, 'warn');
      return true;
    }
    this.points.push(normalized);
    this.renderUi();
    this.requestRedraw();
    return true;
  }

  handleMapClick(event) {
    if (!this.drawing || this.naming) return false;
    const lngLat = event?.lngLat;
    if (!lngLat) return false;
    event?.originalEvent?.preventDefault?.();
    return this.addPoint([lngLat.lng, lngLat.lat]);
  }

  finishDrawing() {
    if (!this.drawing || this.points.length < MIN_POINTS) return;
    this.naming = true;
    this.renderUi();
    window.setTimeout(() => this.elements.nameInput?.focus?.(), 0);
  }

  saveDrawing() {
    if (!this.drawing || this.points.length < MIN_POINTS) return false;
    const now = new Date().toISOString();
    const name = sanitizeText(this.elements.nameInput?.value, this.defaultName) || this.defaultName;
    const info = sanitizeText(this.elements.infoInput?.value, '');
    const polygon = {
      id: `poly-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      name,
      info,
      coordinates: this.points.map((point) => [...point]),
      visible: true,
      createdAt: now,
      updatedAt: now,
    };
    this.polygons.push(polygon);
    this.persist();
    this.cancelDrawing();
    this.callbacks.flashStatus?.(`Saved polygon · ${name}`, 'ok');
    return true;
  }

  persist() {
    try {
      window.localStorage?.setItem(STORAGE_KEY, JSON.stringify({version: 1, polygons: this.polygons}));
    } catch (error) {
      console.warn('[Studio polygons] Failed to save polygon annotations:', error);
    }
  }

  deletePolygon(id) {
    const before = this.polygons.length;
    this.polygons = this.polygons.filter((polygon) => polygon.id !== id);
    if (this.polygons.length === before) return;
    this.persist();
    this.renderUi();
    this.requestRedraw();
  }

  togglePolygon(id) {
    const polygon = this.polygons.find((item) => item.id === id);
    if (!polygon) return;
    polygon.visible = polygon.visible === false;
    polygon.updatedAt = new Date().toISOString();
    this.persist();
    this.renderUi();
    this.requestRedraw();
  }

  requestRedraw() {
    this.revision += 1;
    this.callbacks.requestRedraw?.();
  }

  bindUi() {
    this.elements.addButton?.addEventListener('click', () => this.startDrawing());
    this.elements.sceneToggle?.addEventListener('click', () => this.toggleShowAll());
    this.elements.undoButton?.addEventListener('click', () => this.undoPoint());
    this.elements.finishButton?.addEventListener('click', () => this.finishDrawing());
    this.elements.cancelButton?.addEventListener('click', () => this.cancelDrawing());
    this.elements.formCancelButton?.addEventListener('click', () => {
      this.naming = false;
      this.renderUi();
      this.requestRedraw();
    });
    this.elements.saveForm?.addEventListener('submit', (event) => {
      event.preventDefault();
      this.saveDrawing();
    });
    this.elements.list?.addEventListener('click', (event) => {
      const button = event.target?.closest?.('[data-studio-polygon-action]');
      if (!button) return;
      const action = button.dataset.studioPolygonAction;
      const id = button.dataset.studioPolygonId;
      if (action === 'toggle') this.togglePolygon(id);
      if (action === 'delete') this.deletePolygon(id);
    });
  }

  renderUi() {
    const count = this.polygons.length;
    if (this.elements.status) {
      const visibleCount = this.polygons.filter((polygon) => polygon.visible !== false).length;
      this.elements.status.textContent = this.drawing
        ? `${this.points.length}/${MAX_POINTS} points`
        : (count ? `${visibleCount}/${count} visible` : 'No saved polygons');
    }
    if (this.elements.sceneToggle) {
      this.elements.sceneToggle.classList.toggle('active', this.showAll);
      this.elements.sceneToggle.setAttribute('aria-pressed', this.showAll ? 'true' : 'false');
      this.elements.sceneToggle.textContent = this.showAll ? 'Polygon' : 'Polygon off';
      this.elements.sceneToggle.title = this.showAll ? 'Hide saved studio polygons' : 'Show saved studio polygons';
    }
    if (this.elements.drawBar) this.elements.drawBar.hidden = !this.drawing || this.naming;
    if (this.elements.saveForm) this.elements.saveForm.hidden = !this.drawing || !this.naming;
    if (this.elements.drawStatus) {
      this.elements.drawStatus.textContent = this.points.length < MIN_POINTS
        ? `Click map points · ${MIN_POINTS - this.points.length} more to finish`
        : `Click more points or Finish · ${this.points.length}/${MAX_POINTS}`;
    }
    if (this.elements.undoButton) this.elements.undoButton.disabled = !this.points.length;
    if (this.elements.finishButton) this.elements.finishButton.disabled = this.points.length < MIN_POINTS;
    this.renderList();
  }

  renderList() {
    const list = this.elements.list;
    if (!list) return;
    if (!this.polygons.length) {
      list.innerHTML = '<div class="studioPolygonEmpty">Saved polygons will appear here.</div>';
      return;
    }
    list.innerHTML = this.polygons.map((polygon) => `
      <div class="studioPolygonItem ${polygon.visible === false ? 'isHidden' : ''}">
        <div class="studioPolygonItemText">
          <strong>${sanitizeText(polygon.name, DEFAULT_NAME)}</strong>
          ${polygon.info ? `<span>${sanitizeText(polygon.info)}</span>` : '<span>No note</span>'}
        </div>
        <div class="studioPolygonItemActions">
          <button type="button" data-studio-polygon-action="toggle" data-studio-polygon-id="${polygon.id}">${polygon.visible === false ? 'Show' : 'Hide'}</button>
          <button type="button" data-studio-polygon-action="delete" data-studio-polygon-id="${polygon.id}">Delete</button>
        </div>
      </div>
    `).join('');
  }

  visiblePolygons() {
    if (!this.showAll) return [];
    return this.polygons.filter((polygon) => polygon.visible !== false && polygon.coordinates.length >= MIN_POINTS);
  }

  getLayers({sceneMode = '3d', verticalExaggeration = 1} = {}) {
    const is3d = sceneMode !== '2d';
    const curtainAlpha = curtainOpacityScale(verticalExaggeration);
    const visible = this.visiblePolygons();
    const labelData = visible.map((polygon) => ({...polygon, labelPosition: [...polygonAreaCentroid(polygon.coordinates), is3d ? LABEL_Z : 0]}));
    const curtainData = is3d && curtainAlpha > 0 ? visible.flatMap((polygon) => edgeCurtains(polygon.coordinates)) : [];
    const layers = [];

    if (visible.length) {
      layers.push(new SolidPolygonLayer({
        id: `studio-polygons-fill-${sceneMode}`,
        data: visible,
        pickable: false,
        filled: true,
        stroked: false,
        extruded: false,
        _full3d: is3d,
        getPolygon: (polygon) => polygon.coordinates.map(([lon, lat]) => is3d ? [lon, lat, POLYGON_TOP_Z] : [lon, lat]),
        getFillColor: () => is3d ? FILL_COLOR : FILL_COLOR_2D,
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: {getPolygon: [this.revision, sceneMode]},
      }));
      layers.push(new PathLayer({
        id: `studio-polygons-outline-${sceneMode}`,
        data: visible,
        pickable: false,
        getPath: (polygon) => closePath(polygon.coordinates, is3d ? POLYGON_TOP_Z + 0.05 : null),
        getColor: OUTLINE_COLOR,
        getWidth: 3.2,
        widthUnits: 'pixels',
        widthMinPixels: 2.2,
        capRounded: true,
        jointRounded: true,
        parameters: {depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: {getPath: [this.revision, sceneMode]},
      }));
    }

    if (curtainData.length) {
      layers.push(new SolidPolygonLayer({
        id: 'studio-polygons-curtain-fill',
        data: curtainData,
        pickable: false,
        filled: true,
        stroked: false,
        extruded: false,
        _full3d: true,
        getPolygon: (item) => item.polygon,
        getFillColor: () => scaleColorAlpha(CURTAIN_COLOR, curtainAlpha),
        parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: {getPolygon: [this.revision], getFillColor: [curtainAlpha]},
      }));
      layers.push(new PathLayer({
        id: 'studio-polygons-curtain-edges',
        data: curtainData,
        pickable: false,
        getPath: (item) => item.path,
        getColor: () => scaleColorAlpha(CURTAIN_EDGE_COLOR, curtainAlpha),
        getWidth: 1.15,
        widthUnits: 'pixels',
        widthMinPixels: 0.75,
        parameters: {depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: {getPath: [this.revision], getColor: [curtainAlpha]},
      }));
    }

    if (labelData.length) {
      layers.push(new TextLayer({
        id: `studio-polygons-labels-${sceneMode}`,
        data: labelData,
        pickable: false,
        getPosition: (polygon) => polygon.labelPosition,
        getText: (polygon) => polygon.name,
        getColor: LABEL_COLOR,
        getSize: 13,
        sizeUnits: 'pixels',
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        billboard: true,
        fontFamily: 'Arial, Helvetica, sans-serif',
        fontWeight: 800,
        background: true,
        getBackgroundColor: LABEL_BACKGROUND_COLOR,
        backgroundPadding: [4, 2],
        parameters: {depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: {getPosition: [this.revision, sceneMode], getText: [this.revision]},
      }));
    }

    if (this.drawing && this.points.length) {
      const previewPath = this.points.length > 1 ? [this.points,] : [];
      if (this.points.length >= MIN_POINTS) {
        layers.push(new SolidPolygonLayer({
          id: `studio-polygon-draft-fill-${sceneMode}`,
          data: [{coordinates: this.points}],
          pickable: false,
          filled: true,
          stroked: false,
          extruded: false,
          getPolygon: (item) => item.coordinates,
          getFillColor: PREVIEW_FILL_COLOR,
          parameters: {cullMode: 'none', depthWriteEnabled: false, depthCompare: 'always'},
          updateTriggers: {getPolygon: [this.revision]},
        }));
      }
      layers.push(new PathLayer({
        id: `studio-polygon-draft-outline-${sceneMode}`,
        data: [{path: this.points.length >= MIN_POINTS ? closePath(this.points) : this.points}],
        pickable: false,
        getPath: (item) => item.path,
        getColor: PREVIEW_OUTLINE_COLOR,
        getWidth: 3.6,
        widthUnits: 'pixels',
        widthMinPixels: 2,
        capRounded: true,
        jointRounded: true,
        parameters: {depthWriteEnabled: false, depthCompare: 'less-equal'},
        updateTriggers: {getPath: [this.revision], getColor: [curtainAlpha]},
      }));
      layers.push(new TextLayer({
        id: `studio-polygon-draft-points-${sceneMode}`,
        data: this.points.map((point, index) => ({point, label: String(index + 1)})),
        pickable: false,
        getPosition: (item) => item.point,
        getText: (item) => item.label,
        getColor: [3, 31, 38, 255],
        getSize: 10,
        sizeUnits: 'pixels',
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        billboard: true,
        background: true,
        getBackgroundColor: [126, 248, 255, 238],
        backgroundPadding: [3, 2],
        parameters: {depthWriteEnabled: false, depthCompare: 'always'},
        updateTriggers: {getPosition: [this.revision]},
      }));
    }

    return layers;
  }
}
