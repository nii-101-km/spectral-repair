/* createPointerController: explicit dependencies, state owned by this editor instance. */
function createPointerController({editor, root, canvas, $, drawing, selectAudition}) {
  const {invalidate, selectionChanged, updateView, clampView} = drawing;
  /* All gestures are local; drawing, zoom and undo never call generation endpoints. */
  function at(event, allowOutside = false) {
    const rect = canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const plot = allowOutside && editor.drag ? editor.drag.plot : editor.plots.find(p => y >= p.y && y <= p.y + p.h);
    if (!plot || (!allowOutside && (x < plot.x || x > plot.x + plot.w))) return null;
    const point = spectralPoint(Math.max(plot.x, Math.min(plot.x + plot.w, x)),
      Math.max(plot.y, Math.min(plot.y + plot.h, y)), plot, editor.view);
    return {...point, x, y, plot};
  }
  function paint(a, b) {
    editor.grid.stroke(a, b, Math.max(editor.grid.dt, +$('[data-role="brush-ms"]').value / 1000 || editor.grid.dt),
      Math.max(editor.grid.df, +$('[data-role="brush-hz"]').value || editor.grid.df), +$('[data-role="soft"]').value / 100,
      editor.tool === 'eraser' || $('[data-role="operation"]').value === 'subtract');
    editor.dirty = true; invalidate();
  }
  canvas.addEventListener('pointerdown', e => {
    if (!editor.grid || e.button !== 0) return;
    const point = at(e); if (!point) return;
    if (editor.drag) finishGesture(editor.drag.last);
    e.preventDefault(); canvas.focus({preventScroll: true}); canvas.setPointerCapture(e.pointerId);
    editor.drag = {start: point, last: point, view: {...editor.view}, plot: point.plot, pointerId: e.pointerId};
    if (['rectangle', 'brush', 'eraser'].includes(editor.tool)) editor.grid.begin();
    if (['brush', 'eraser'].includes(editor.tool)) paint(point, point);
    invalidate();
  });
  canvas.addEventListener('pointermove', e => {
    if (!editor.grid) return;
    if (editor.drag && e.pointerId !== editor.drag.pointerId) return;
    const p = at(e, !!editor.drag); if (!p) { editor.cursor = null; invalidate(); return; }
    editor.cursor = p;
    $('[data-role="position"]').textContent = `${p.t.toFixed(4)} с · ${p.f.toFixed(1)} Гц · бин ${editor.grid.cell(p.t, p.f).join(':')}`;
    if (editor.drag) {
      if (['brush', 'eraser'].includes(editor.tool)) paint(editor.drag.last, p);
      if (editor.tool === 'pan') {
        const before = editor.drag.view, dt = -(p.x - editor.drag.start.x) / p.plot.w * (before.t1 - before.t0);
        const df = (p.y - editor.drag.start.y) / p.plot.h * (before.f1 - before.f0);
        editor.view = clampView({t0: before.t0 + dt, t1: before.t1 + dt, f0: before.f0 + df, f1: before.f1 + df}); updateView();
      }
      editor.drag.last = p;
    }
    invalidate();
  });
  function finishGesture(p) {
    if (!editor.drag) return;
    const gesture = editor.drag;
    if (['brush', 'eraser'].includes(editor.tool)) paint(editor.drag.last, p);
    if (editor.tool === 'rectangle') editor.grid.rectangle(editor.drag.start, p, +$('[data-role="soft"]').value / 100,
      $('[data-role="operation"]').value === 'subtract');
    if (editor.tool === 'zoom' && Math.abs(p.x - editor.drag.start.x) > 3 && Math.abs(p.y - editor.drag.start.y) > 3) {
      editor.view = clampView({t0: Math.min(editor.drag.start.t, p.t), t1: Math.max(editor.drag.start.t, p.t),
        f0: Math.min(editor.drag.start.f, p.f), f1: Math.max(editor.drag.start.f, p.f)}); updateView();
    }
    editor.grid.finish(); editor.drag = null; selectionChanged();
    if (editor.tool === 'audition') selectAudition(gesture.start, p);
  }
  function endGesture(e) {
    if (!editor.drag || e.pointerId !== editor.drag.pointerId) return;
    finishGesture(at(e, true) || editor.drag.last);
  }
  canvas.addEventListener('pointerup', endGesture);
  canvas.addEventListener('lostpointercapture', () => { if (editor.drag) finishGesture(editor.drag.last); });
  canvas.addEventListener('pointercancel', () => {
    if (editor.drag && editor.grid.before) { editor.grid.values = editor.grid.before; editor.grid.before = null; }
    editor.drag = null; if (editor.grid) selectionChanged();
  });
  canvas.addEventListener('pointerleave', () => { if (!editor.drag) editor.cursor = null; invalidate(); });
  canvas.addEventListener('wheel', e => {
    if (editor.drag) { e.preventDefault(); return; }
    if (!editor.source) return; const point = at(e); if (!point) return; e.preventDefault();
    const factor = Math.exp(Math.max(-1, Math.min(1, e.deltaY / 500)));
    if (e.shiftKey || e.ctrlKey) {
      editor.view.f0 = point.f + (editor.view.f0 - point.f) * factor;
      editor.view.f1 = point.f + (editor.view.f1 - point.f) * factor;
    } else {
      editor.view.t0 = point.t + (editor.view.t0 - point.t) * factor;
      editor.view.t1 = point.t + (editor.view.t1 - point.t) * factor;
    }
    editor.view = clampView(editor.view); updateView();
  }, {passive: false});
  root.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click', () => {
    editor.tool = button.dataset.tool;
    root.dataset.currentTool = editor.tool;
    root.querySelectorAll('[data-tool]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    canvas.style.cursor = editor.tool === 'pan' ? 'grab' : 'crosshair';
    if (editor.grid) selectionChanged(); else invalidate();
  }));
  function action(name) {
    if (!editor.grid) return;
    if (name === 'reset') { editor.view = {t0: 0, t1: editor.source.duration, f0: 0, f1: editor.source.stft.sample_rate / 2}; updateView(); return; }
    if (name === 'undo') editor.grid.undo();
    else if (name === 'redo') editor.grid.redo();
    else editor.grid.transaction(() => {
      if (name === 'clear') editor.grid.values.fill(0);
      if (name === 'invert') for (let i = 0; i < editor.grid.values.length; i++) editor.grid.values[i] = 1 - editor.grid.values[i];
      if (name === 'soften') editor.grid.soften();
    });
    selectionChanged();
  }
  root.querySelectorAll('[data-action]').forEach(b => b.addEventListener('click', () => action(b.dataset.action)));
  canvas.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && ['z', 'y'].includes(e.key.toLowerCase())) {
      e.preventDefault(); action(e.shiftKey || e.key.toLowerCase() === 'y' ? 'redo' : 'undo');
    }
  });
  function applyView() {
    if (!editor.view) return;
    const candidate = Object.fromEntries(['t0', 't1', 'f0', 'f1'].map(k => [k, +$(`[data-view="${k}"]`).value]));
    if (Object.values(candidate).every(Number.isFinite) && candidate.t1 > candidate.t0 && candidate.f1 > candidate.f0) editor.view = clampView(candidate);
    updateView();
  }
  root.querySelectorAll('[data-view]').forEach(input => input.addEventListener('change', applyView));
  $('[data-role="apply-view"]').addEventListener('click', applyView);
  $('[data-role="channel"]').addEventListener('change', invalidate);
  $('[data-role="overlay"]').addEventListener('change', () => editor.grid ? selectionChanged() : invalidate());
  $('[data-role="operation"]').addEventListener('change', () => { if (editor.grid) selectionChanged(); });
  new ResizeObserver(invalidate).observe(canvas);

  return {finishGesture, action, applyView, at};
}
