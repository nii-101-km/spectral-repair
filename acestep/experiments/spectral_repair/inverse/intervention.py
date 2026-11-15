"""Causal latent interventions with declared temporal guards and channel subsets."""

import numpy as np


def apply(original: np.ndarray, delta: np.ndarray, trial: dict, variant: dict,
          stride_seconds: float, training_profile: np.ndarray) -> tuple[np.ndarray,dict]:
    """Never choose held-out channel counts using the current test fragment's delta."""
    energy=np.sum(delta.astype(float)**2,axis=0)
    profile=energy if variant["ranking"]!="heldout" else training_profile
    profile=profile/max(float(profile.sum()),1e-30)
    order=np.argsort(-profile,kind="stable")
    coverage=variant["coverage"]
    count=len(profile) if coverage==1 else min(len(profile),1+int(np.searchsorted(np.cumsum(profile[order]),coverage)))
    selected=order[:count]
    frame=np.ones(len(original),dtype=bool)
    if variant["guard"] is not None:
        left=np.arange(len(original))*stride_seconds;right=left+stride_seconds
        frame=(right>trial["start"]-variant["guard"]+1e-10)&(left<trial["end"]+variant["guard"]-1e-10)
    keep=np.zeros_like(delta,dtype=bool);keep[np.ix_(frame,selected)]=True
    correction=np.where(keep,delta*variant["alpha"],0).astype(original.dtype)
    result=(original+correction).astype(original.dtype)
    fraction=float(np.sum(np.where(keep,delta,0).astype(float)**2)/max(float(energy.sum()),1e-30))
    return result,{"selected_channels":selected.tolist(),"channel_count":count,
                   "selected_frames":int(frame.sum()),"retained_delta_energy":fraction,
                   "ranking_coverage":float(profile[selected].sum()),
                   "latent_change_l2":float(np.linalg.norm(correction))}
