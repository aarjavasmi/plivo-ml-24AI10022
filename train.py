"""
    python train.py --data_dir eot_data/english --out predictions.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict

from features import load_wav, speech_before, frame_energy_db, f0_contour
from extract_features import extract_features  # or keep in this file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model_out", default="model.joblib")
    ap.add_argument("--C", type=float, default=0.5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    rows.sort(key=lambda r: (r["turn_id"], int(r["pause_index"])))

    cache = {}
    turn_prev_durs = defaultdict(list)
    X, y, groups, keys = [], [], [], []

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
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(tid)
        keys.append((tid, r["pause_index"]))

        turn_prev_durs[tid].append(float(r["pause_end"]) - pause_start)

    X, y = np.array(X), np.array(y)
    groups = np.array(groups)

    n_turns = len(set(groups))
    n_splits = min(5, n_turns)

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=args.C, class_weight="balanced"),
        )

    # --- HONEST dev-time predictions: out-of-sample via GroupKFold CV ---
    gkf = GroupKFold(n_splits=n_splits)
    p_oos = cross_val_predict(
        make_clf(), X, y, groups=groups, cv=gkf, method="predict_proba"
    )[:, 1]

    acc_oos = ((p_oos >= 0.5).astype(int) == y).mean()
    print(f"out-of-sample accuracy: {acc_oos:.3f} "
          f"(chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})  C={args.C}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p_oos):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} OUT-OF-SAMPLE predictions -> {args.out}  "
          f"(use this file with score.py during development)")

    # --- final model: refit on ALL data, save for predict.py ---
    final_clf = make_clf()
    final_clf.fit(X, y)
    joblib.dump(final_clf, args.model_out)
    print(f"saved final model (fit on all data) -> {args.model_out}  "
          f"(this is what predict.py will load)")


if __name__ == "__main__":
    main()