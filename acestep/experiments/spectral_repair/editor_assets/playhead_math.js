/* Playback geometry uses CSS pixels and excerpt-relative seconds, never STFT bin rounding. */
function playheadX(time, plot, viewport) {
  return plot.x + (time - viewport.t0) / (viewport.t1 - viewport.t0) * plot.w;
}
function playheadTime(clientX, canvasLeft, plot, viewport, duration) {
  const fraction = Math.max(0, Math.min(1, (clientX - canvasLeft - plot.x) / plot.w));
  return Math.max(0, Math.min(duration, viewport.t0 + fraction * (viewport.t1 - viewport.t0)));
}
function playheadInView(time, viewport) {
  return Number.isFinite(time) && time >= viewport.t0 && time <= viewport.t1;
}
