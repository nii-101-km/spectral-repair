/* Compose per-editor owners here; callback closures resolve deliberate initialization cycles. */
const root = element.querySelector('.nii-editor'), canvas = root.querySelector('canvas');
const $ = selector => root.querySelector(selector);
const editor = createEditorState(), versions = new SpectralVersions();
const preview = $('[data-role="audio"]');
const drawing = createDrawingController({editor, root, canvas, $,
  drawAudition:plot=>audition.draw(plot), refreshPlayhead:()=>playhead.refresh()});
const pointer = createPointerController({editor, root, canvas, $, drawing,
  selectAudition:(a,b)=>audition.select(a,b)});
const registry = createAudioRegistry({host:document.body, exclude:audio=>audio===audition.audio,
  onChange:()=>playback.bind(), frame:requestAnimationFrame, Observer:MutationObserver});
const audition = createAuditionController({editor, $, canvas, versions,
  allAudio:()=>registry.all(), invalidate:drawing.invalidate, refreshPlayhead:()=>playhead.refresh()});
const playback = createPlaybackController({editor, $, preview, registry,
  stopAudition:audition.stop, watchPlayhead:audio=>playhead.watch(audio),
  message:drawing.message, defer:queueMicrotask});
const playhead = createPlayheadController({editor, $, canvas, preview, playback, audition,
  getSeekDrag:()=>seeking.gesture, frame:requestAnimationFrame});
const seeking = createSeekController({editor, canvas, playhead, message:drawing.message});
const history = createVersionController({editor, versions, preview, $, playback,
  cancelPlaybackSeek:seeking.cancel, stopAudition:audition.stop,
  finishGesture:pointer.finishGesture, invalidate:drawing.invalidate});
const bridge = createEditorBridge({editor, root, canvas, $, versions, history, drawing,
  playback, seeking, audition, playhead, hostWindow:window, Observer:MutationObserver});
playback.bind();
bridge.start();
