"""Fixed inverse-experiment cases and strictly held-out channel rankings."""

import hashlib
import json
from pathlib import Path

import numpy as np


def build_plan(config_path: str, smoke: bool) -> dict:
    """Build stages I–III; stage IV is gated on review of decoded-audio results."""
    path=Path(config_path).resolve();config=json.loads(path.read_text())
    previous=Path(config["previous_run"]).resolve()
    if not (previous/"DONE.json").is_file():raise ValueError("Previous series is incomplete")
    old=json.loads((previous/"manifest.json").read_text())
    original=old["plan"]
    clips=original["clips"][:1] if smoke else original["clips"]
    bands=[b for b in original["config"]["bands"] if b["id"]=="mid"] if smoke else original["config"]["bands"]
    if not bands:raise ValueError("Smoke requires a mid band")
    if config["duration"]<=0 or not 0<config["strength"]<=1:raise ValueError("Invalid intervention")
    for name in ("guards","coverage","alphas"):
        if not config[name] or not np.isfinite(config[name]).all():raise ValueError("Invalid factors")
    if min(config["guards"])<0 or not all(0<x<=1 for x in config["coverage"]):raise ValueError("Invalid guard/coverage")
    ranks=held_out_profiles(previous,original)
    pairs=[]
    for clip in clips:
        start=round(config["position"]*clip["duration"]-config["duration"]/2,9)
        end=start+config["duration"]
        if not 0<start<end<clip["duration"]:raise ValueError("Intervention lacks context")
        for band in bands:
            trial={"clip_id":clip["id"],"kind":"attenuate_band","start":start,"end":end,
                   "strength":config["strength"],"band":band["id"],"low":band["low"],"high":band["high"]}
            pairs.append({"id":f"{clip['id']}-{band['id']}","clip":clip,"trial":trial,
                          "variants":variants(config,smoke)})
    return {"schema":1,"mode":"smoke" if smoke else "main","config":config,
            "config_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"previous":str(previous),
            "previous_identity":old["identity"],"vae":original["vae"],"weights":original["weights"],
            "sample_rate":original["sample_rate"],"stride":original["stride"],
            "latent_channels":original["latent_channels"],"clips":clips,"pairs":pairs,"rankings":ranks,
            "records":sum(len(p["variants"])+1 for p in pairs)+len(clips),
            "decodes":sum(len(p["variants"])+1 for p in pairs)+2*len(clips),
            "encodes":len(clips)+len(pairs)}


def variants(config: dict, smoke: bool) -> list[dict]:
    """Keep oracle/test-dependent selection visibly separate from held-out selection."""
    result=[{"id":"A-full","family":"A","alpha":1.,"guard":None,"coverage":1.,"ranking":"all"}]
    guards=[.1] if smoke else config["guards"]
    for guard in guards:
        result.append({"id":f"B-{guard:g}","family":"B","alpha":1.,"guard":guard,"coverage":1.,"ranking":"all"})
    for ranking in ("oracle","heldout"):
        for coverage in ([.9] if smoke else config["coverage"]):
            for guard in guards:
                result.append({"id":f"C-{ranking}-{coverage:g}-{guard:g}","family":"C", "alpha":1.,
                               "guard":guard,"coverage":coverage,"ranking":ranking})
    for alpha in ([.5] if smoke else config["alphas"]):
        result.append({"id":f"D-{alpha:g}","family":"D","alpha":alpha,"guard":None,"coverage":1.,"ranking":"all"})
    return result


def held_out_profiles(previous: Path, old_plan: dict) -> dict:
    """Train channel rankings only on other clips, with raw delta energy recovered from maps."""
    rows=[json.loads(line) for line in (previous/"results.jsonl").read_text().splitlines()]
    scales={clip["id"]:np.array(json.loads((previous/f"{clip['id']}-reference.json").read_text())["channel_scale"])
            for clip in old_plan["clips"]}
    profiles=[]
    for row in rows:
        if row["kind"]!="attenuate_band":continue
        with np.load(previous/"maps"/f"{row['id']}.npz") as archive:
            energy=archive["standardized_energy"].sum(axis=0,dtype=float)*scales[row["clip_id"]]**2
        if energy.sum()>0:profiles.append((row["clip_id"],row["band"],energy/energy.sum()))
    result={}
    for clip in old_plan["clips"]:
        for band in old_plan["config"]["bands"]:
            selected=[p for c,b,p in profiles if c!=clip["id"] and b==band["id"]]
            if not selected:raise ValueError("No other-clip ranking data")
            result[f"{clip['id']}/{band['id']}"]=np.mean(selected,axis=0).tolist()
    return result
