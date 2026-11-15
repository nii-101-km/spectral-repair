"""Explicitly invoked single-fragment listening experiments using unchanged ACE APIs."""

from pathlib import Path
from uuid import uuid4

import soundfile as sf

from .ace import ACECandidate
from .baselines import interpolate, notch_candidate
from .core import repair
from .lab_parameters import assemble_named
from .metrics import Resources
from .provider_cache import ProviderCache
from .run_adapters import laboratory_request
from .run_contracts import RunRequest, RunResult
from .run_preparation import prepare_audio, validate_request
from .run_storage import save_result
from .visualize import save_preview


class Laboratory:
    """Cache one provider; serialize calls through Gradio's single-job queue."""

    def __init__(self, project_root: str, output: str):
        """Store paths without loading models until the user requests generation."""
        self.project_root = project_root
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self._providers = ProviderCache(
            project_root, factory=lambda root, model, device: ACECandidate(root, model, device))

    @property
    def provider(self) -> ACECandidate | None:
        """Expose the loaded provider for existing laboratory integrations."""
        return self._providers.provider

    @provider.setter
    def provider(self, value: ACECandidate | None) -> None:
        """Retain compatibility with callers assigning a provider explicitly."""
        self._providers.provider = value

    @property
    def provider_key(self) -> tuple[str, str] | None:
        """Expose the cached model/device pair without owning a second cache."""
        return self._providers.key

    @provider_key.setter
    def provider_key(self, value: tuple[str, str] | None) -> None:
        """Retain compatibility with callers resetting the cached pair."""
        self._providers.key = value

    def run(self, source, offset, duration, start, end, low, high, feather,
            defect, model, device, blend, expert, *values):
        """Generate one candidate, save A/B WAVs and an exact settings JSON per run."""
        request = laboratory_request(
            source, offset, duration, start, end, low, high, feather,
            defect, model, device, blend, expert, values)
        return self.execute(request).as_legacy()

    def execute(self, request: RunRequest) -> RunResult:
        """Process a named request and return saved artifacts with their report."""
        validate_request(request)
        params = assemble_named(request.generation, request.expert)
        random_seed = bool(request.generation["use_random_seed"])
        prepared = prepare_audio(request)
        original, damaged, mask = prepared.original, prepared.damaged, prepared.mask
        selection, sampler = request.selection, request.sampler
        config, method = request.config, request.method
        folder = self.output / uuid4().hex[:12]
        folder.mkdir()
        source_path = folder / "input.wav"
        sf.write(source_path, damaged, config.sample_rate, subtype="FLOAT")
        timer = Resources(cuda=method == "ace")
        if selection:
            from .dsp_controls import apply_dsp
            control = apply_dsp(damaged, mask, config, selection.dsp, selection.notch_hz)
        elif request.defect == "tone":
            notch = notch_candidate(damaged, 48000, (request.low + request.high) / 2)
            control, _ = repair(damaged, notch, mask, config, max_shift_ms=0)
        else:
            control = interpolate(damaged, mask, config)
        if method == "ace":
            provider = self._providers.get(request.model, request.device)
        if method == "dsp":
            candidate, metadata = None, {"not_run": "DSP selected"}
            result, diagnostics = control.copy(), {"algorithm": selection.dsp}
            if selection.dsp == "time":
                from .temporal_interpolation import ALGORITHM_VERSION
                diagnostics["algorithm_version"] = ALGORITHM_VERSION
            elif selection.dsp == "frequency":
                from .frequency_interpolation import ALGORITHM_VERSION
                diagnostics["algorithm_version"] = ALGORITHM_VERSION
        elif sampler.enabled:
            from .projection_adapter import generate_projected
            candidate, metadata = generate_projected(
                provider, str(source_path), mask, params,
                start_noise=sampler.start_noise, every=sampler.every, config=config,
                temporal_lock=sampler.temporal_lock, temporal_margin=sampler.temporal_margin,
                audio_condition=sampler.audio_condition, guidance_algorithm=sampler.guidance_algorithm)
        else:
            spans = selection.time_regions if selection else [(request.start, request.end)]
            candidate, metadata = provider.generate(str(source_path), spans, params)
        if method == "ace":
            result, diagnostics = repair(damaged, candidate, mask, config, request.blend)
        return save_result(
            folder=folder, request=request, original=original, damaged=damaged,
            control=control, candidate=candidate, result=result, mask=mask,
            provenance=prepared.provenance, region=prepared.region, params=params,
            random_seed=random_seed,
            metadata=metadata, diagnostics=diagnostics, timer=timer, preview=save_preview)
