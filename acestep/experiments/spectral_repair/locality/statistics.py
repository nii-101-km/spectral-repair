"""Descriptive grouping and clip-held-out channel-profile checks, without causal claims."""

import numpy as np


def describe(rows: list[dict]) -> dict:
    """Summarize independent clips separately from the many correlated perturbations."""
    controls = [r for r in rows if r["kind"] == "control"]
    trials = [r for r in rows if r["kind"] != "control"]
    groups = []
    for kind in sorted({r["kind"] for r in trials}):
        for duration in sorted({round(r["end"]-r["start"], 6) for r in trials if r["kind"] == kind}):
            selected = [r for r in trials if r["kind"] == kind
                        and abs((r["end"]-r["start"])-duration) < 1e-5]
            group = {"kind":kind,"duration":duration,"n":len(selected)}
            for key in ("fraction_in_support", "concentration_lift", "spread_90_seconds"):
                values = [r["mode"][key] for r in selected if r["mode"][key] is not None]
                group[key] = {"median":float(np.median(values)),
                              "q25":float(np.quantile(values,.25)),
                              "q75":float(np.quantile(values,.75))} if values else None
            groups.append(group)
    return {"trials":len(trials),"controls":len(controls),
            "clips":len({r["clip_id"] for r in trials}),
            "zero_waveform_change":sum(r["waveform_delta_rms"] <= 1e-12 for r in trials),
            "max_repeat_mode_error":max((r["mode"]["raw_max"] for r in controls),default=0),
            "max_repeat_coupled_error":max((r["coupled"]["raw_max"] for r in controls),default=0),
            "independent_posterior_l2_median":float(np.median(
                [r["independent_sample"]["raw_l2"] for r in controls])) if controls else None,
            "groups":groups,"band_classification":band_classification(trials)}


def band_classification(rows: list[dict]) -> dict:
    """Leave-one-clip-out nearest centroid; compare shuffled labels as a sanity check."""
    result = {}
    for kind in ("attenuate_band", "tone"):
        subset = [r for r in rows if r["kind"] == kind and r["mode"]["standardized_energy"] > 0]
        labels = sorted({r["band"] for r in subset})
        clips = sorted({r["clip_id"] for r in subset})
        if len(labels) < 2 or len(clips) < 2:
            result[kind] = {"status":"insufficient_clips_or_labels"}
            continue
        features = np.array([r["mode"]["channel_profile"] for r in subset])
        truth = np.array([r["band"] for r in subset])
        identifiers = np.array([r["clip_id"] for r in subset])
        def score(target):
            """Fit only on other clips; never split variants of one clip across folds."""
            correct, count = 0, 0
            confusion = np.zeros((len(labels),len(labels)), dtype=int)
            for clip in clips:
                train, test = identifiers != clip, identifiers == clip
                if not all(np.any(train & (target == label)) for label in labels):
                    continue
                centers = np.array([features[train & (target == label)].mean(axis=0) for label in labels])
                centers /= np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-15)
                for vector,label in zip(features[test],target[test]):
                    predicted = labels[int(np.argmax(centers @ vector))]
                    confusion[labels.index(label),labels.index(predicted)] += 1
                    correct += predicted == label; count += 1
            return {"accuracy":float(correct/count) if count else None,
                    "n":count,"confusion":confusion.tolist()}
        shuffled = truth.copy()
        rng = np.random.default_rng(113)
        for clip in clips:
            chosen = identifiers == clip
            shuffled[chosen] = rng.permutation(shuffled[chosen])
        result[kind] = {"status":"descriptive_only", "labels":labels,"chance":1/len(labels),
                        "held_out":score(truth),"shuffled":score(shuffled)}
    return result
