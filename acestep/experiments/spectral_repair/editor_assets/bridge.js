/* createEditorBridge: explicit dependencies, state owned by this editor instance. */
function createEditorBridge({editor, root, canvas, $, versions, history, drawing, playback, seeking, audition, playhead, hostWindow, Observer}) {
  let lastRevision = null;
  const {invalidate, updateView, selectionChanged, message} = drawing;
  const {render:renderVersions, activate:activateVersion} = history;
  const cancelPlaybackSeek = seeking.cancel, stopAudition = audition.stop;
  const refreshPlayhead = playhead.refresh;
  function receive() {
    root.dataset.currentTool = editor.tool;
    let data;
    try { data = JSON.parse($('[data-role="payload"]').textContent || '{}'); }
    catch (_) { message('Не удалось прочитать данные спектрограммы'); return; }
    if (!data.signature) {
      cancelPlaybackSeek();
      stopAudition(true);
      playback.clearPending();
      editor.source = null; editor.grid = null; editor.images = []; playback.resetPreview();
      versions.signature = null; versions.items = []; renderVersions();
      refreshPlayhead();
      lastRevision = null; invalidate(); message('Загрузите WAV и постройте спектрограмму'); return;
    }
    if (data.revision === lastRevision) return;
    if (data.history_only) {
      if (!versions.merge(data.signature, data.history || [])) return;
      lastRevision = data.revision;
      activateVersion(data.history_select || versions.selected);
      root.querySelectorAll('[data-tool]').forEach(button =>
        button.setAttribute('aria-pressed', String(button.dataset.tool === editor.tool)));
      canvas.style.cursor = editor.tool === 'pan' ? 'grab' : 'crosshair';
      updateView(); selectionChanged(); return;
    }
    const changed = !editor.source || editor.source.signature !== data.signature;
    lastRevision = data.revision; editor.source = data;
    if (changed) {
      cancelPlaybackSeek();
      stopAudition(true);
      playback.clearPending();
      versions.reset(data);
      editor.grid = new SpectralGrid(data.shape, data.stft.hop_length / data.stft.sample_rate,
        data.stft.sample_rate / data.stft.fft_size, data.duration);
      editor.overlay = document.createElement('canvas'); editor.overlay.width = editor.grid.T; editor.overlay.height = editor.grid.F;
      editor.overlayImage = editor.overlay.getContext('2d').createImageData(editor.grid.T, editor.grid.F);
      editor.view = {t0: 0, t1: data.duration, f0: 0, f1: data.stft.sample_rate / 2};
      $('[data-role="channel"]').querySelector('[value="1"]').disabled = data.channels < 2;
      const ms = $('[data-role="brush-ms"]'), hz = $('[data-role="brush-hz"]');
      ms.min = editor.grid.dt * 1000; hz.min = editor.grid.df;
      ms.value = Math.max(+ms.value, +ms.min); hz.value = Math.max(+hz.value, +hz.min);
      updateView();
    }
    versions.merge(data.signature, data.history || []);
    if (data.mask_f32) {
      try {
        const mask = spectralDecode(data.mask_f32, editor.grid.values.length);
        editor.grid.transaction(() => { editor.grid.values = mask; });
      } catch (error) { message(error.message); return; }
    }
    root.querySelectorAll('[data-tool]').forEach(button =>
      button.setAttribute('aria-pressed', String(button.dataset.tool === editor.tool)));
    canvas.style.cursor = editor.tool === 'pan' ? 'grab' : 'crosshair';
    activateVersion(versions.selected);
    updateView();
    selectionChanged();
  }
  hostWindow.niiSpectralEditor = {
    serialize() {
      if (!editor.source || !editor.grid) return '{}';
      return JSON.stringify({signature: editor.source.signature, shape: editor.source.shape, stft: editor.source.stft,
        duration: editor.source.duration, offset: editor.source.offset, mask_f32: spectralBase64(editor.grid.values)});
    },
    rectangle(start, end, low, high, feather) {
      if (!editor.grid) { message('Сначала постройте спектрограмму'); return; }
      if (![start, end, low, high, feather].every(Number.isFinite) || start < 0 || start >= end ||
          end > editor.source.duration || low < 0 || low >= high || high > editor.source.stft.sample_rate / 2) {
        message('Проверьте границы прямоугольника'); return;
      }
      const softness = Math.min(1, Math.max(0, feather * 2 / (end - start)));
      editor.grid.transaction(() => editor.grid.rectangle({t: start, f: low}, {t: end, f: high}, softness,
        $('[data-role="operation"]').value === 'subtract'));
      selectionChanged();
    }
  };

  return {receive, start(){
    new Observer(receive).observe($('[data-role="payload"]'), {childList:true, characterData:true, subtree:true});
    receive(); refreshPlayhead();
  }};
}
