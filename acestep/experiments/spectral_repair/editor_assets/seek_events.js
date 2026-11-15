/* createSeekController: explicit dependencies, state owned by this editor instance. */
function createSeekController({editor, canvas, playhead, message}) {
  let seekDrag = null;
  const {audio:playbackAudio, time:playbackTime, seek:seekPlayback, refresh:refreshPlayhead,
    layer:playheadLayer, ruler:seekRuler} = playhead;
  /* Seeking owns separate DOM targets; it never invokes mask gestures or undo transactions. */
  function seekEventTime(event) {
    const rect = canvas.getBoundingClientRect();
    const plot = editor.plots[0] || {x:65, w:canvas.clientWidth-82};
    return playheadTime(event.clientX, rect.left, plot, editor.view, editor.source.duration);
  }
  function beginPlaybackSeek(event, target) {
    if (!editor.source || !editor.view || editor.drag || seekDrag || event.button !== 0 || event.isPrimary === false) return;
    event.preventDefault(); event.stopPropagation(); seekRuler.focus({preventScroll:true});
    const audio = playbackAudio();
    seekDrag = {audio, resume:!audio.paused, pointerId:event.pointerId, target,
                time:seekEventTime(event)};
    audio.pause(); target.setPointerCapture(event.pointerId);
    seekPlayback(seekDrag.time, audio);
  }
  function movePlaybackSeek(event) {
    if (!seekDrag || event.pointerId !== seekDrag.pointerId) return;
    event.preventDefault(); seekDrag.time = seekEventTime(event);
    seekPlayback(seekDrag.time, seekDrag.audio);
  }
  function endPlaybackSeek(event, update = true) {
    if (!seekDrag || event.pointerId !== seekDrag.pointerId) return;
    if (update) movePlaybackSeek(event);
    const gesture = seekDrag; cancelPlaybackSeek();
    if (gesture.resume && gesture.audio.isConnected) gesture.audio.play().catch(error => message(error.message));
    refreshPlayhead();
  }
  [playheadLayer, seekRuler].forEach(target => {
    target.addEventListener('pointerdown', event => beginPlaybackSeek(event, target));
    target.addEventListener('pointermove', movePlaybackSeek);
    target.addEventListener('pointerup', endPlaybackSeek);
    target.addEventListener('pointercancel', event => endPlaybackSeek(event, false));
    target.addEventListener('lostpointercapture', event => endPlaybackSeek(event, false));
  });
  seekRuler.addEventListener('keydown', event => {
    if (!editor.source || !editor.view) return;
    const step = event.shiftKey ? 1 : .1;
    const positions = {ArrowLeft:playbackTime()-step, ArrowRight:playbackTime()+step,
                       Home:0, End:editor.source.duration};
    if (!(event.key in positions)) return;
    event.preventDefault(); seekPlayback(positions[event.key]);
  });

  function cancelPlaybackSeek() {
    if (!seekDrag) return;
    const gesture = seekDrag; seekDrag = null;
    if (gesture.target.hasPointerCapture(gesture.pointerId)) gesture.target.releasePointerCapture(gesture.pointerId);
  }

  return {cancel:cancelPlaybackSeek, get gesture(){return seekDrag;}};
}
