"""predict.py — run a SAVED model on unseen data.

    python predict.py --data_dir eot_data/english --out predictions.csv

Requires model.joblib to already exist (produced by train.py). Does NOT fit
anything and does NOT use pause_end or the label column — only turn_id,
audio_file, pause_index, and pause_start, exactly like a live agent would.
"""
import argparse
import csv
import os
from collections import defaultdict

import joblib

from features import load_wav
from extract_features import extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args()

    clf = joblib.load(args.model)

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    # sort so previous-pause-duration history builds up correctly per turn
    rows.sort(key=lambda r: (r["turn_id"], int(r["pause_index"])))

    cache = {}
    turn_prev_durs = defaultdict(list)
    X, keys = [], []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]

        tid = r["turn_id"]
        pidx = int(r["pause_index"])
        pause_start = float(r["pause_start"])

        feat = extract_features(x, sr, pause_start, pidx, turn_prev_durs[tid])
        X.append(feat)
        keys.append((tid, r["pause_index"]))

        # record this pause's duration for use by LATER pauses in the same
        # turn only — this mirrors what a live agent would know by the time
        # a later pause starts (this pause has already ended by then).
        # pause_end is read here for bookkeeping only, never fed as a
        # feature for THIS pause's own prediction.
        turn_prev_durs[tid].append(float(r["pause_end"]) - pause_start)

    p = clf.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])

    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()