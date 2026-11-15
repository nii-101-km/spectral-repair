/* STFT-coordinate mask operations, independent of canvas size, zoom and channels. */
class SpectralGrid {
  constructor(shape, dt, df, duration) {
    [this.T, this.F] = shape; this.dt = dt; this.df = df; this.duration = duration;
    this.values = new Float32Array(this.T * this.F);
    this.past = []; this.future = []; this.before = null;
  }
  begin() { if (!this.before) this.before = this.values.slice(); }
  finish() {
    if (!this.before) return false;
    const changed = this.values.some((v, i) => v !== this.before[i]);
    if (changed) {
      this.past.push(this.before); this.future = [];
      const limit = Math.max(1, Math.min(24, Math.floor(128 * 1024 * 1024 / this.values.byteLength)));
      while (this.past.length > limit) this.past.shift();
    }
    this.before = null; return changed;
  }
  undo() {
    if (!this.past.length) return;
    this.future.push(this.values); this.values = this.past.pop();
  }
  redo() {
    if (!this.future.length) return;
    this.past.push(this.values); this.values = this.future.pop();
  }
  transaction(fn) { this.begin(); fn(); this.finish(); }
  cell(t, f) {
    return [Math.max(0, Math.min(this.T - 1, Math.round(t / this.dt))),
      Math.max(0, Math.min(this.F - 1, Math.round(f / this.df)))];
  }
  write(t, f, value, subtract) {
    const i = t * this.F + f;
    this.values[i] = subtract ? Math.max(0, this.values[i] - value) : Math.max(this.values[i], value);
  }
  rectangle(a, b, soft, subtract = false) {
    soft = Math.max(0, Math.min(1, Number.isFinite(soft) ? soft : 0));
    const [at, af] = this.cell(a.t, a.f), [bt, bf] = this.cell(b.t, b.f);
    const loT = Math.min(at, bt), hiT = Math.max(at, bt);
    const loF = Math.min(af, bf), hiF = Math.max(af, bf);
    for (let t = loT; t <= hiT; t++) for (let f = loF; f <= hiF; f++) {
      const wt = soft ? Math.min(1, Math.min(t - loT + .5, hiT - t + .5) / Math.max(.5, soft * (hiT - loT + 1) / 2)) : 1;
      const wf = soft ? Math.min(1, Math.min(f - loF + .5, hiF - f + .5) / Math.max(.5, soft * (hiF - loF + 1) / 2)) : 1;
      this.write(t, f, wt * wf, subtract);
    }
  }
  stamp(point, seconds, hz, soft, subtract = false) {
    if (![seconds, hz].every(Number.isFinite) || seconds <= 0 || hz <= 0) return;
    soft = Math.max(0, Math.min(1, Number.isFinite(soft) ? soft : 0));
    const rt = seconds / 2, rf = hz / 2;
    const [t0, f0] = this.cell(point.t - rt, point.f - rf);
    const [t1, f1] = this.cell(point.t + rt, point.f + rf);
    let touched = false;
    for (let t = t0; t <= t1; t++) for (let f = f0; f <= f1; f++) {
      const d = Math.hypot((t * this.dt - point.t) / rt, (f * this.df - point.f) / rf);
      if (d > 1) continue;
      const weight = soft ? Math.min(1, (1 - d) / soft) : 1;
      if (weight <= 0) continue;
      this.write(t, f, weight, subtract);
      touched = true;
    }
    if (!touched && point.t >= 0 && point.t <= this.duration && point.f >= 0 && point.f <= (this.F - 1) * this.df) {
      const [t, f] = this.cell(point.t, point.f); this.write(t, f, 1, subtract);
    }
  }
  stroke(a, b, seconds, hz, soft, subtract = false) {
    if (![seconds, hz].every(Number.isFinite) || seconds <= 0 || hz <= 0) return;
    const steps = Math.max(1, Math.ceil(Math.hypot((b.t - a.t) / seconds, (b.f - a.f) / hz) * 5));
    for (let i = 0; i <= steps; i++) this.stamp({t: a.t + (b.t - a.t) * i / steps,
      f: a.f + (b.f - a.f) * i / steps}, seconds, hz, soft, subtract);
  }
  soften() {
    let a = this.values, b = new Float32Array(a.length);
    for (let t = 0; t < this.T; t++) for (let f = 0; f < this.F; f++) {
      const i = t * this.F + f;
      b[i] = (a[Math.max(0, t - 1) * this.F + f] + 2 * a[i] + a[Math.min(this.T - 1, t + 1) * this.F + f]) / 4;
    }
    a = b; b = new Float32Array(a.length);
    for (let t = 0; t < this.T; t++) for (let f = 0; f < this.F; f++) {
      const i = t * this.F + f;
      b[i] = (a[t * this.F + Math.max(0, f - 1)] + 2 * a[i] + a[t * this.F + Math.min(this.F - 1, f + 1)]) / 4;
    }
    this.values = b;
  }
  bounds() {
    let first = this.T, last = -1, count = 0;
    for (let t = 0; t < this.T; t++) for (let f = 0; f < this.F; f++) if (this.values[t * this.F + f] > 0) {
      first = Math.min(first, t); last = Math.max(last, t); count++;
    }
    return last < 0 ? null : {start: Math.max(0, (first - .5) * this.dt),
      end: Math.min(this.duration, (last + .5) * this.dt), count};
  }
}
function spectralPoint(x, y, plot, view) {
  return {t: view.t0 + (x - plot.x) / plot.w * (view.t1 - view.t0),
    f: view.f1 - (y - plot.y) / plot.h * (view.f1 - view.f0)};
}
function spectralBase64(array) {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  let string = '';
  for (let i = 0; i < bytes.length; i += 32768) string += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(string);
}
function spectralDecode(value, count) {
  const string = atob(value), bytes = new Uint8Array(string.length);
  for (let i = 0; i < string.length; i++) bytes[i] = string.charCodeAt(i);
  if (bytes.length !== count * 4) throw Error('Размер маски не совпадает с STFT');
  const result = new Float32Array(bytes.buffer);
  if (result.some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw Error('Маска должна содержать значения 0–1');
  return result;
}
