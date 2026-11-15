# Third-party components and publication scope

Reviewed 2026-11-14. The root MIT license covers this project's own source and
modifications. It does not replace dependency, model or musical-content terms.
This is a source release, not an environment, binary or weights distribution.

| Component | Observed license / treatment |
|---|---|
| ACE-Step 1.5 code | MIT; retain [upstream notice](licenses/ACE-Step-MIT.txt); installed separately |
| ACE-Step Base and audio VAE | Upstream ACE-Step 1.5 model card declares MIT; no weights bundled |
| Qwen3-Embedding-0.6B | Apache-2.0 model card; no weights bundled |
| NumPy, SciPy, PyTorch | BSD-family terms; installed separately, including their own bundled-library notices |
| SoundFile | BSD-3-Clause wrapper; libsndfile has separate LGPL terms |
| Gradio, Transformers, Diffusers, Accelerate, PEFT | Apache-2.0; installed separately |
| Matplotlib | Matplotlib license (PSF-based); installed separately |
| Loguru, Einops, Setuptools | MIT; installed separately |
| Numba | BSD; installed separately |
| frozendict (transitive) | Installed metadata declares LGPL v3; not vendored |
| NVIDIA CUDA libraries (transitive) | Several installed packages declare proprietary NVIDIA terms; not redistributed or presented as MIT |
| nano-vllm (upstream backend dependency) | MIT in upstream backend; not copied into this repository |

[Dependency inventory](docs/dependency-licenses.json) records **all distributions
found in the tested environment**, their versions and available license metadata.
It is an environment inventory, not a claim that every installed package is used
by the editor. Long metadata license fields are truncated to 2000 characters;
the upstream license texts remain authoritative. Empty or contradictory metadata
is not proof of a permissive license. Native libraries packaged inside wheels can
have additional terms beyond their Python wrapper; do not redistribute the tested
venv as an MIT-only application bundle.

No third-party binaries, model weights, Python wheels or backend source archives
are included in the intended commit. The existing local backend, model symlink,
full experiment archive and environments are excluded through `.gitignore`.
A separate redistribution review is required before shipping a binary/environment bundle.

## Demo audio

The project owner explicitly authorized publication of the test recording and
its damaged/repaired variants as demonstration data. No account identity, private song URL,
screenshot or purchase evidence is included.

The source is Suno-generated instrumental test material. The six WAV files in
`docs/demo/audio/` preserve the saved audio samples, with ancillary file metadata
removed. They are supplied for demonstrating and reproducing this experiment with
the owner's permission. No separate open audio license has been selected; the
project's MIT license applies to code, not automatically to musical recordings.

The owner's publication instruction is recorded separately from a legal guarantee:
external service terms still apply. Metadata removal is a privacy operation, not
a rights-clearance or provenance-erasure guarantee. The source service is disclosed
here. This source release does not claim to remove any audible watermark.

## Sources

- [ACE-Step source and MIT license](https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE)
- [ACE-Step model card](https://huggingface.co/ACE-Step/Ace-Step1.5)
- [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [SoundFile and libsndfile licensing](https://github.com/bastibe/python-soundfile)
- [NVIDIA CUDA terms](https://docs.nvidia.com/cuda/eula/index.html)
- [Suno terms](https://suno.com/terms), revision August 10, 2026, effective September 3, 2026
- [Suno ownership explanation](https://help.suno.com/en/articles/2416769)
