# Initial public source release checks

Date: 2026-11-14. Scope: preserve the existing editor, document installation and
prepare a source-only commit. No model weights downloaded, no new research,
no push or GitHub Release. Suggested commit: `Prepare Spectral Repair for initial public release`.

## PASS

- Baseline CPU suite: 151 tests. Release candidate: 154 tests, including two new
  clean-backend namespace/bootstrap regressions. The fix preserves backend import
  paths and handles a namespace without list `.insert()`.
- Installed dependency consistency: `uv pip check`, 143 packages compatible.
- Source distribution wheel builds successfully; wheel installs into a separate temporary
  venv, and imports succeed against the archived clean backend with installed dependencies reused.
- Updated editor builds and serves HTTP 200 on an isolated localhost port; the smoke
  server is closed afterwards and the working server is not stopped.
- DSP smoke: five 4-second synthetic cases, WAV/JSON/PNG export, ACE explicitly not run.
  [Recorded measurements](dsp-smoke.json).
- Existing demo: source/offset signature and exact mask verified; Damage 4 equals
  Repaint 22 input sample-for-sample. Original audio outside edited time-window
  support has zero final difference. STFT linearity error ≈3.17×10⁻¹⁷.
- Saved final assembly from the float32 candidate: maximum error ≈2.58×10⁻⁸ per sample.
- Browser checks on the running local editor: source loads, history selects Repaint
  22 with a ready 60-second audio player, zoom changes to 32.8–35.6 s / 0–2500 Hz,
  rectangle/brush add mask bins, eraser removes them, undo restores the previous mask.
- Frequency/time audition: selected 33.29–34.06 s / 965–1958 Hz; browser audio ready
  state 4, generated duration 0.768542 seconds, no media error. This checks media
  readiness, not a human listening judgment.
- Prior local export audit: all six saved WAV exports had empty ffprobe format and
  stream tags, with original sample data preserved. Those files remain outside Git.
- English/Russian README, MIT project license and upstream MIT notice prepared;
  model licenses and environment license metadata inventoried separately.

- Display localization: English default, Russian selection retained across reloads;
  switching languages leaves numeric inputs and user text unchanged. Rectangles,
  history labels, canvas axes, lazy settings, dropdown captions and callback status
  messages were checked in the browser. The English screenshot replaces the prior image.
  The localization regression also protects input values, filenames and raw payloads.

- Follow-up secret scan (2026-11-15): Gitleaks **8.30.1**, default rules, no custom
  config/baseline/ignore entries and `--ignore-gitleaks-allow`. Both staged-diff
  scanning and directory scanning of an exact staged snapshot returned exit 0,
  empty findings and “no leaks found”. Recursive decoding used the default depth;
  archive traversal was enabled to depth 2 for the snapshot scan. Logs were redacted.

## FAIL

No outstanding failed checks in the final source candidate. The clean-backend
namespace issue was fixed with a regression test; whitespace findings were corrected.

## NOT RUN / UNAVAILABLE

- Fresh online dependency provisioning on an empty machine: not run; this would
  install the large backend/CUDA environment. Existing environment, wheel installation
  smoke and clean-backend import checks are assessed separately.
- A fresh ACE GPU generation during release preparation: not run. Existing saved
  run is retained as evidence; CPU tests of model adapters use mocks. Export/release
  preparation does not require replacing or regenerating the demonstration.
- Cross-platform editor integration on Windows/macOS/ROCm/MPS/XPU: not run. CUDA
  selector limitations and tested Linux setup are documented.
- Human listening comparison: not performed by the agent; spectrograms and metrics
  are not substituted for that check.
- Independent legal clearance of the audio: not performed. The owner expressly
  authorized inclusion of this test material; it is not claimed to be MIT-licensed audio.

## Release boundaries

`var/`, environments, weights, caches, logs, local freeze paths and the full
`experiments/` directory stay local. `docs/demo/` contains the compact documented
experiment with six explicitly approved WAV files, replay scripts and no account data.
The broader local archive and unrelated recordings remain excluded. Public CPU
verification checks the supplied source, mask, damage and final assembly.

The code MIT license does not relicense third-party packages, model weights or
musical recordings. NVIDIA proprietary runtime packages and LGPL components exist
in the environment; no environment or binary bundle is distributed.

Git, final source scan, wheel smoke, DSP smoke, documentation links and whitespace
results are recorded in the companion [release audit](release-audit.json).
