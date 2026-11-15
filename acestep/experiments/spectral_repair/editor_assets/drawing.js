/* createDrawingController: explicit dependencies, state owned by this editor instance. */
function createDrawingController({editor, root, canvas, $, drawAudition, refreshPlayhead}) {
  const ctx = canvas.getContext('2d');
  const text = value => globalThis.niiLocale?.text(value) ?? value;
  function message(text) { $('[data-role="status"]').textContent = text; }
  function invalidate() { if (!editor.frame) editor.frame = requestAnimationFrame(draw); }
  function clampView(v) {
    const duration = editor.source.duration, nyquist = editor.source.stft.sample_rate / 2;
    const tw = Math.min(duration, Math.max(editor.grid.dt, v.t1 - v.t0));
    const fw = Math.min(nyquist, Math.max(editor.grid.df, v.f1 - v.f0));
    v.t0 = Math.max(0, Math.min(duration - tw, v.t0)); v.t1 = v.t0 + tw;
    v.f0 = Math.max(0, Math.min(nyquist - fw, v.f0)); v.f1 = v.f0 + fw;
    return v;
  }
  function updateView() {
    for (const key of ['t0', 't1', 'f0', 'f1']) $(`[data-view="${key}"]`).value = Number(editor.view[key].toFixed(4));
    invalidate();
  }
  function rebuildOverlay() {
    if (!editor.dirty || !editor.grid) return;
    const pixels = editor.overlayImage.data;
    for (let t = 0; t < editor.grid.T; t++) for (let f = 0; f < editor.grid.F; f++) {
      const p = ((editor.grid.F - 1 - f) * editor.grid.T + t) * 4;
      pixels[p] = 40; pixels[p + 1] = 225; pixels[p + 2] = 245;
      pixels[p + 3] = Math.round(editor.grid.values[t * editor.grid.F + f] * 140);
    }
    editor.overlay.getContext('2d').putImageData(editor.overlayImage, 0, 0); editor.dirty = false;
  }
  function draw() {
    editor.frame = 0;
    const width = canvas.clientWidth, height = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!width || !height) return;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    ctx.font = '12px system-ui'; ctx.fillStyle = getComputedStyle(root).color;
    if (!editor.source) { ctx.fillText(text('Загрузите WAV и постройте спектрограмму'), 66, height / 2); return; }
    rebuildOverlay();
    const mode = $('[data-role="channel"]').value;
    const channels = mode === 'both' ? editor.images.map((_, i) => i) : [Math.min(+mode, editor.images.length - 1)];
    const section = (height - 8) / channels.length; editor.plots = [];
    for (let c = 0; c < channels.length; c++) {
      const plot = {x: 65, y: c * section + 17, w: width - 82, h: section - 49}; editor.plots.push(plot);
      const sx = .5 + editor.view.t0 / editor.grid.dt, sy = editor.grid.F - .5 - editor.view.f1 / editor.grid.df;
      const sw = (editor.view.t1 - editor.view.t0) / editor.grid.dt, sh = (editor.view.f1 - editor.view.f0) / editor.grid.df;
      if (editor.images[channels[c]].complete) ctx.drawImage(editor.images[channels[c]], sx, sy, sw, sh, plot.x, plot.y, plot.w, plot.h);
      if ($('[data-role="overlay"]').checked) ctx.drawImage(editor.overlay, sx, sy, sw, sh, plot.x, plot.y, plot.w, plot.h);
      ctx.fillStyle = getComputedStyle(root).color;
      ctx.textAlign = 'left'; ctx.fillText(text(`Канал ${channels[c] + 1}`), plot.x, plot.y - 4);
      for (let i = 0; i <= 5; i++) {
        const f = editor.view.f1 - i / 5 * (editor.view.f1 - editor.view.f0), y = plot.y + plot.h * i / 5;
        ctx.textAlign = 'right'; ctx.fillText(text(`${Math.round(f)} Гц`), plot.x - 7, y + 4);
        ctx.textAlign = 'center';
        const t = editor.view.t0 + i / 5 * (editor.view.t1 - editor.view.t0);
        ctx.fillText(text(`${t.toFixed(2)} с`), plot.x + plot.w * i / 5, plot.y + plot.h + 19);
      }
      ctx.save(); ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.w, plot.h); ctx.clip();
      ctx.strokeStyle = '#73f4ff'; ctx.lineWidth = 1.5;
      const point = p => ({x: plot.x + (p.t - editor.view.t0) / (editor.view.t1 - editor.view.t0) * plot.w,
        y: plot.y + (editor.view.f1 - p.f) / (editor.view.f1 - editor.view.f0) * plot.h});
      if (editor.drag && ['rectangle', 'zoom', 'audition'].includes(editor.tool)) {
        const a = point(editor.drag.start), b = point(editor.drag.last); ctx.setLineDash([5, 4]);
        ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      }
      if (editor.cursor && ['brush', 'eraser'].includes(editor.tool)) {
        const p = point(editor.cursor); ctx.beginPath(); ctx.setLineDash([]);
        ctx.ellipse(p.x, p.y, +$('[data-role="brush-ms"]').value / 2000 / (editor.view.t1 - editor.view.t0) * plot.w,
          +$('[data-role="brush-hz"]').value / 2 / (editor.view.f1 - editor.view.f0) * plot.h, 0, 0, Math.PI * 2); ctx.stroke();
      }
      drawAudition(plot);
      ctx.restore();
    }
    refreshPlayhead();
  }
  function selectionChanged() {
    editor.dirty = true; invalidate();
    const b = editor.grid.bounds();
    const names = {rectangle: 'Прямоугольник', brush: 'Кисть', eraser: 'Ластик', zoom: 'Лупа', pan: 'Перемещение', audition: 'Прослушивание области'};
    const mode = names[editor.tool] + ($('[data-role="operation"]').value === 'subtract' && editor.tool !== 'eraser' ? ' · Вычитание' : '');
    const hidden = $('[data-role="overlay"]').checked ? '' : 'Маска скрыта · ';
    const empty = editor.tool === 'eraser' || $('[data-role="operation"]').value === 'subtract' ? 'маска пуста, стирать нечего' : 'маска пуста';
    message(hidden + mode + ' · ' + (b ? `${b.count.toLocaleString(globalThis.niiLocale?.language === 'en' ? 'en-US' : 'ru-RU')} бинов · ${b.start.toFixed(3)}–${b.end.toFixed(3)} с · история ${editor.grid.past.length}/${editor.grid.future.length}` : empty));
    $('[data-action="undo"]').disabled = !editor.grid.past.length;
    $('[data-action="redo"]').disabled = !editor.grid.future.length;
  }

  return {message, invalidate, clampView, updateView, selectionChanged, draw};
}
