"""Round-trip-aware waveform/STFT metrics with a single fixed alignment per clip."""

import numpy as np
from scipy import signal

from ..spectral import STFTConfig,analyze


def alignment(original: np.ndarray, decoded: np.ndarray, rate: int) -> dict:
    """Estimate stereo-common lag once on round trip; never optimize it per variant."""
    x=original-original.mean(axis=0);y=decoded-decoded.mean(axis=0)
    norm=float(np.linalg.norm(x)*np.linalg.norm(y));limit=round(.05*rate)
    if norm<1e-15:return {"lag":0,"correlation":0.,"accepted":False,"ambiguous":True}
    corr=sum(signal.correlate(y[:,c],x[:,c],method="fft") for c in range(2))
    center=len(x)-1;window=corr[center-limit:center+limit+1]/norm
    index=int(np.argmax(window));lag=index-limit;confidence=float(window[index])
    accepted=confidence>=.2 and abs(lag)<limit
    return {"lag":lag if accepted else 0,"estimated_lag":lag,"correlation":confidence,
            "accepted":accepted,"ambiguous":not accepted,"peak_at_limit":abs(lag)==limit}


def aligned(reference: np.ndarray, decoded: np.ndarray, lag: int) -> tuple[np.ndarray,np.ndarray,int]:
    """Crop a common valid interval, without padding edges using the original audio."""
    start=max(0,-lag);out=max(0,lag);count=min(len(reference)-start,len(decoded)-out)
    return reference[start:start+count],decoded[out:out+count],start


def ratio(numerator: float,denominator: float):
    """Do not invent an accuracy/locality ratio when the denominator is zero."""
    return float(numerator/denominator) if denominator>1e-30 else None


def measure(y: np.ndarray,x: np.ndarray,roundtrip: np.ndarray,dsp: np.ndarray,
            target_roundtrip: np.ndarray,trial: dict,rate: int,lag: int) -> dict:
    """Separate absolute decoder error, incremental intervention and target reproduction."""
    ref,audio,offset=aligned(x,y,lag)
    _,rt,_=aligned(x,roundtrip,lag)
    _,target,_=aligned(x,target_roundtrip,lag)
    desired=dsp[offset:offset+len(ref)]-ref
    delta=audio-rt;codec_target=target-rt
    time=np.arange(len(ref))/rate+offset/rate
    inside=(time>=trial["start"])&(time<=trial["end"])
    power=np.sum(delta**2,axis=1);energy=float(power.sum())
    config=STFTConfig(rate)
    spectrum=analyze(delta,config);spectral=np.sum(np.abs(spectrum)**2,axis=0)
    times=np.arange(len(spectral))*config.hop_length/rate+offset/rate
    freq=np.fft.rfftfreq(config.fft_size,1/rate)
    selected_time=(times>=trial["start"])&(times<=trial["end"])
    selected_frequency=(freq>=trial["low"])&(freq<=trial["high"])
    total=float(spectral.sum());joint=selected_time[:,None]&selected_frequency[None,:]
    width=trial["high"]-trial["low"]
    neighbor=((freq>=max(0,trial["low"]-width))&(freq<trial["low"]))|((freq>trial["high"])&(freq<=trial["high"]+width))
    rt_spec=analyze(rt,config);y_spec=rt_spec+spectrum
    rt_band=float(np.sum(np.abs(rt_spec[:,selected_time][:,:,selected_frequency])**2))
    y_band=float(np.sum(np.abs(y_spec[:,selected_time][:,:,selected_frequency])**2))
    return {"effect_rms":float(np.sqrt(np.mean(delta**2))),"effect_energy":energy,
            "temporal_fraction":ratio(float(power[inside].sum()),energy),
            "frequency_fraction":ratio(float(spectral[:,selected_frequency].sum()),total),
            "joint_fraction":ratio(float(spectral[joint].sum()),total),
            "neighbor_fraction":ratio(float(spectral[:,neighbor].sum()),total),
            "outside_time_energy":float(power[~inside].sum()),
            "outside_band_energy":float(spectral[:,~selected_frequency].sum()),
            "relative_direct_effect_error":ratio(float(np.linalg.norm(delta-desired)),float(np.linalg.norm(desired))),
            "relative_codec_effect_error":ratio(float(np.linalg.norm(delta-codec_target)),float(np.linalg.norm(codec_target))),
            "absolute_original_rmse":float(np.sqrt(np.mean((audio-ref)**2))),
            "absolute_dsp_rmse":float(np.sqrt(np.mean((audio-(ref+desired))**2))),
            "band_energy_ratio_to_roundtrip":ratio(y_band,rt_band),
            "roundtrip_rmse":float(np.sqrt(np.mean((rt-ref)**2))),
            "peak":float(np.max(np.abs(audio))),"aligned_samples":len(ref),
            "reference_offset_samples":offset,"lag_samples":lag}
