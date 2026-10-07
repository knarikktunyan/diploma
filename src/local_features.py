"""Experimental time-local features; the production ten-feature schema is unchanged."""
import numpy as np
from scipy.signal import savgol_filter

LOCAL_FEATURE_NAMES = ['window_rms_variation', 'window_mean_variation', 'crossing_period_variation']
FEATURE_GROUPS = {
    'baseline': [],
    'amplitude_offset': LOCAL_FEATURE_NAMES[:2],
    'frequency': LOCAL_FEATURE_NAMES[2:],
    'combined': LOCAL_FEATURE_NAMES,
}
LOCAL_FEATURE_VERSION = 1


def extract_local_features(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 32 or not np.isfinite(signal).all():
        raise ValueError('Local features require at least 32 finite one-dimensional samples')
    size = len(signal)//4
    starts = list(range(0, len(signal)-size+1, max(1, size//2)))
    if starts[-1] != len(signal)-size:
        starts.append(len(signal)-size)
    windows = np.array([signal[start:start+size] for start in starts])
    rms = np.sqrt(np.mean(windows**2, axis=1))
    rms_variation = np.ptp(rms)/rms.mean() if rms.mean() > 0 else 0.0
    std = np.std(signal)
    mean_variation = np.ptp(windows.mean(axis=1))/std if std > 0 else 0.0

    # Auxiliary crossing measurement only: the waveform and baseline features
    # retain their noise/harmonics. Fixed smoothing is not a fitted transform.
    centered = signal - np.mean(signal)
    smooth = savgol_filter(centered, window_length=11, polyorder=3)
    indices = np.flatnonzero((smooth[:-1] <= 0) & (smooth[1:] > 0))
    crossings = indices - smooth[indices]/(smooth[indices+1]-smooth[indices])
    periods = np.diff(crossings)
    variation = np.std(periods)/np.mean(periods) if len(periods) >= 2 else 0.0
    return dict(zip(LOCAL_FEATURE_NAMES, map(float, [rms_variation, mean_variation, variation])))
