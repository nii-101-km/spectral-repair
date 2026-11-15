/* createPlayheadController: explicit dependencies, state owned by this editor instance. */
function createPlayheadController({editor, $, canvas, preview, playback, audition, getSeekDrag, frame}) {
  /* Lightweight DOM overlay: playback never rebuilds the spectrum or mask image. */
  const playheadLayer = $('[data-role="playheads"]'), seekRuler = $('[data-role="seek-ruler"]');
  let playheadFrame = 0;
  function playbackAudio() {
    if (audition.active && audition.box) return audition.audio;
    return playback.active && playback.active.isConnected ? playback.active : preview;
  }
  function playbackTime() {
    const audio = playbackAudio();
    const time = getSeekDrag() ? getSeekDrag().time : audio.currentTime + (audio === audition.audio && audition.box ? audition.box.t0 : 0);
    return Number.isFinite(time) ? Math.max(0, Math.min(editor.source.duration, time)) : 0;
  }
  function refreshPlayhead() {
    seekRuler.setAttribute('aria-disabled', String(!editor.source));
    if (!editor.source || !editor.view) {
      playheadLayer.replaceChildren();
      $('[data-role="seek-thumb"]').style.display = 'none';
      $('[data-role="playback-position"]').textContent = '0.00 с';
      seekRuler.setAttribute('aria-valuemax', '0'); seekRuler.setAttribute('aria-valuenow', '0');
      return;
    }
    const audio = playbackAudio(), time = playbackTime();
    if (playheadLayer.children.length !== editor.plots.length) {
      playheadLayer.replaceChildren();
      editor.plots.forEach(() => {
        const handle = document.createElement('button'); handle.type = 'button';
        handle.className = 'nii-playhead'; handle.tabIndex = -1;
        handle.setAttribute('aria-label', 'Перетащить позицию воспроизведения');
        playheadLayer.appendChild(handle);
      });
    }
    const visible = playheadInView(time, editor.view);
    editor.plots.forEach((plot, index) => {
      const handle = playheadLayer.children[index];
      handle.style.display = visible ? 'block' : 'none';
      handle.style.left = `${playheadX(time, plot, editor.view) - 7}px`;
      handle.style.top = `${plot.y}px`; handle.style.height = `${plot.h}px`;
      handle.title = `Перемотка: ${time.toFixed(2)} с`;
    });
    const fraction = Math.max(0, Math.min(1, (time-editor.view.t0)/(editor.view.t1-editor.view.t0)));
    const thumb = $('[data-role="seek-thumb"]');
    thumb.style.display = visible ? 'block' : 'none'; thumb.style.left = `${fraction*100}%`;
    seekRuler.setAttribute('aria-valuemax', String(editor.source.duration));
    seekRuler.setAttribute('aria-valuenow', time.toFixed(3));
    seekRuler.setAttribute('aria-valuetext', `${time.toFixed(2)} с${visible ? '' : ', вне вида'}`);
    $('[data-role="playback-position"]').textContent = `${time.toFixed(2)} / ${editor.source.duration.toFixed(2)} с · в песне ${(editor.source.offset+time).toFixed(2)} с${visible ? '' : ' · вне вида'}`;
    if (!audio.paused && !audio.ended && !playheadFrame) {
      playheadFrame = frame(() => { playheadFrame = 0; refreshPlayhead(); });
    }
  }
  function seekPlayback(time, audio = playbackAudio()) {
    if (!editor.source) return;
    time = Math.max(0, Math.min(editor.source.duration, time));
    if (audio === audition.audio && audition.box) {
      time = Math.max(audition.box.t0, Math.min(audition.box.t1, time));
      audition.audio.currentTime = time - audition.box.t0;
      refreshPlayhead(); return;
    }
    playback.seek(time, audio);
    refreshPlayhead();
  }
  function watchPlayhead(audio) {
    ['play','pause','timeupdate','seeking','seeked','loadedmetadata','ended','emptied','ratechange']
      .forEach(name => audio.addEventListener(name, refreshPlayhead));
  }

  return {audio:playbackAudio, time:playbackTime, refresh:refreshPlayhead, seek:seekPlayback, watch:watchPlayhead,
    layer:playheadLayer, ruler:seekRuler};
}
