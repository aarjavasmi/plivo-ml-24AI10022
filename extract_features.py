import numpy as np

from features import load_wav, speech_before, frame_energy_db, f0_contour

def _slope(y):
    """Linear slope of a 1D sequence via simple least squares. 0 if too short."""
    if len(y) < 2:
        return 0.0
    t = np.arange(len(y), dtype=np.float32)
    # polyfit degree 1 -> [slope, intercept]
    return float(np.polyfit(t, y, 1)[0])


def extract_features(x, sr, pause_start, pause_index, prev_pause_durs):
    """Causal features using only audio in [0, pause_start].

    prev_pause_durs: list of durations (seconds) of earlier pauses in this
    turn, already known at this point since their pause_end < pause_start.
    """
    # --- turn-level context (whole history so far, for normalization) ---
    hist_end = int(pause_start * sr)
    hist = x[:hist_end]
    if len(hist) < sr // 10:
        return np.zeros(11, dtype=np.float32)

    hist_e = frame_energy_db(hist, sr)
    hist_f0 = f0_contour(hist, sr)
    hist_voiced = hist_f0[hist_f0 > 0]

    turn_e_mean = hist_e.mean() if len(hist_e) else -60.0
    turn_e_std = hist_e.std() + 1e-6
    turn_f0_mean = hist_voiced.mean() if len(hist_voiced) else 0.0
    turn_f0_std = hist_voiced.std() + 1e-6

    # --- trailing window (local trend right before the pause) ---
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(11, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]

    # 1. energy slope into the pause (trailing ~0.5s of frames)
    e_tail = e[-15:] if len(e) >= 15 else e
    energy_slope = _slope(e_tail)

    # 2. final energy level, z-scored against this turn's own distribution
    final_energy_z = (e[-5:].mean() - turn_e_mean) / turn_e_std if len(e) else 0.0

    # 3. F0 slope over trailing voiced region (falling => eot cue)
    f0_slope = _slope(voiced[-10:]) if len(voiced) >= 3 else 0.0

    # 4. final voiced pitch, z-scored against this turn's own pitch range
    final_f0_z = (voiced[-3:].mean() - turn_f0_mean) / turn_f0_std if len(voiced) >= 3 else 0.0

    # 5. voiced fraction of trailing window (trailing silence/breath vs still talking)
    voiced_frac = float(voiced_mask.mean()) if len(voiced_mask) else 0.0

    # 6. voiced fraction of just the LAST ~0.3s (mid-word cutoff vs settled silence)
    tail_mask = voiced_mask[-8:] if len(voiced_mask) >= 8 else voiced_mask
    voiced_frac_tail = float(tail_mask.mean()) if len(tail_mask) else 0.0

    # 7. pitch range/variance in trailing window (flat vs varying)
    f0_std_local = float(voiced.std()) if len(voiced) >= 2 else 0.0

    # 8. structural: pause position within the turn
    pause_idx_f = float(pause_index)

    # 9. structural: cumulative speech duration so far
    cum_duration = float(pause_start)

    # 10-11. structural: previous pause durations in this turn
    prev_pause_mean = float(np.mean(prev_pause_durs)) if prev_pause_durs else 0.0
    prev_pause_last = float(prev_pause_durs[-1]) if prev_pause_durs else 0.0

    return np.array([
        energy_slope, final_energy_z, f0_slope, final_f0_z,
        voiced_frac, voiced_frac_tail, f0_std_local,
        pause_idx_f, cum_duration, prev_pause_mean, prev_pause_last,
    ], dtype=np.float32)
