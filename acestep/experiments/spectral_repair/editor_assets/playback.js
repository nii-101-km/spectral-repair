/* createPlaybackController: explicit dependencies, state owned by this editor instance. */
function createPlaybackController({editor, $, preview, registry, stopAudition, watchPlayhead, message, defer}) {
  let activeAudio = null, playingSelection = null, syncBusy = false, pendingVersionTime = null;
  const boundAudio = new WeakSet();
  $('[data-role="play-selection"]').addEventListener('click', () => {
    const selected = editor.grid && editor.grid.bounds();
    if (!selected) { message('Сначала нарисуйте маску'); return; }
    playingSelection = selected; activeAudio = null;
    $('[data-role="limit"]').checked = true; preview.currentTime = selected.start;
    preview.play().catch(error => message(error.message));
  });

  function bindAudio() {
    registry.all().forEach(audio => {
      if (boundAudio.has(audio)) return; boundAudio.add(audio);
      audio.addEventListener('play', () => {
        stopAudition();
        const sync = $('[data-role="sync"]').checked;
        if (sync && activeAudio && activeAudio !== audio && Number.isFinite(activeAudio.currentTime))
          audio.currentTime = Math.min(audio.duration || editor.source.duration, activeAudio.currentTime);
        if ($('[data-role="limit"]').checked) {
          playingSelection = editor.grid && editor.grid.bounds();
          if (playingSelection && (audio.currentTime < playingSelection.start || audio.currentTime >= playingSelection.end))
            audio.currentTime = playingSelection.start;
        }
        activeAudio = audio;
        if (sync) registry.all().forEach(other => {
          if (other !== audio) { other.pause(); if (Number.isFinite(other.duration)) other.currentTime = audio.currentTime; }
        });
      });
      audio.addEventListener('timeupdate', () => {
        if ($('[data-role="limit"]').checked && playingSelection && audio.currentTime >= playingSelection.end) audio.pause();
      });
      audio.addEventListener('seeked', () => {
        if (syncBusy || !$('[data-role="sync"]').checked) return;
        syncBusy = true;
        registry.all().forEach(other => {
          if (other !== audio && Number.isFinite(other.duration) && Math.abs(other.currentTime - audio.currentTime) > .05)
            other.currentTime = Math.min(other.duration, audio.currentTime);
        });
        defer(() => { syncBusy = false; });
      });
      watchPlayhead(audio);
    });
  }
  function switchVersion(item, isCurrent) {
    if (preview.getAttribute('src') !== item.preview_url) {
      const position = pendingVersionTime ?? (Number.isFinite(preview.currentTime) ? preview.currentTime : 0);
      const resume = !preview.paused;
      registry.all().forEach(audio => audio.pause()); activeAudio = null; playingSelection = null;
      pendingVersionTime = position;
      preview.onloadedmetadata = () => {
        if (!isCurrent()) return;
        preview.currentTime = Math.min(pendingVersionTime ?? position, preview.duration || editor.source.duration);
        pendingVersionTime = null;
        if (resume) preview.play().catch(error => message(error.message));
      };
      preview.src = item.preview_url; preview.load();
    }
  }

  function seek(time, audio) {
    if (audio === preview && pendingVersionTime !== null) pendingVersionTime = time;
    const targets = $('[data-role="sync"]').checked ? registry.all() : [audio];
    if (!targets.includes(audio)) targets.push(audio);
    activeAudio = audio;
    targets.forEach(target => {
      const duration = Number.isFinite(target.duration) ? target.duration : editor.source.duration;
      const position = Math.min(time, duration);
      if (Math.abs(target.currentTime-position) > .001) target.currentTime = position;
    });
  }
  return {bind:bindAudio, switchVersion, seek, get active(){return activeAudio;},
    clearPending(){pendingVersionTime=null;},
    resetPreview(){pendingVersionTime=null;preview.pause();preview.removeAttribute('src');}};
}
