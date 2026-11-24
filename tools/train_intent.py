"""Train the learned intent gate and store its weights.

    python tools/train_intent.py            # train, evaluate, write weights
    python tools/train_intent.py --evaluate # evaluate the stored weights only

The classifier is a tiny logistic regression (eight features, 400 Hz loop), so
training takes a couple of seconds on a laptop CPU.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iars.sensors.intent_dataset import make_episodes  # noqa: E402
from iars.stabilization.intent import (  # noqa: E402
    DEFAULT_WEIGHTS,
    IntentClassifier,
    extract_features,
    heuristic_intent_gate,
)

FS = 400.0


def build(episodes):
    X, y = [], []
    for ep in episodes:
        feats, centres = extract_features(ep.gyro_dps, FS)
        X.append(feats)
        y.append(ep.label[centres])
    return np.vstack(X), np.concatenate(y)


def scores(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    acc = float(np.mean(pred == truth))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true", help="evaluate stored weights only")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_eps = make_episodes(args.episodes, seed=args.seed)
    test_eps = make_episodes(60, seed=args.seed + 10_000)
    Xtr, ytr = build(train_eps)
    Xte, yte = build(test_eps)

    if args.evaluate:
        clf = IntentClassifier.load(DEFAULT_WEIGHTS)
    else:
        clf = IntentClassifier.fit(Xtr, ytr)
        clf.save(DEFAULT_WEIGHTS)

    clf_s = scores(clf.predict(Xte), yte)

    # Reference: the 2024 threshold heuristic, sampled at the same window centres.
    h_pred, h_true = [], []
    for ep in test_eps:
        _, centres = extract_features(ep.gyro_dps, FS)
        gate = heuristic_intent_gate(ep.gyro_dps, FS)
        h_pred.append(gate[centres])
        h_true.append(ep.label[centres])
    heur_s = scores(np.concatenate(h_pred), np.concatenate(h_true))

    print(f"episodes: train={len(train_eps)} test={len(test_eps)}")
    print(f"windows : train={len(ytr)} test={len(yte)}  positive rate={yte.mean():.3f}")
    print()
    print(f"{'':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}")
    print("-" * 52)
    print(f"{'heuristic (2024)':<22}{heur_s['accuracy']:>10.3f}{heur_s['precision']:>11.3f}{heur_s['recall']:>9.3f}")
    print(f"{'learned classifier':<22}{clf_s['accuracy']:>10.3f}{clf_s['precision']:>11.3f}{clf_s['recall']:>9.3f}")
    print()
    print("feature weights (standardised space):")
    for name, coef in zip(clf.to_dict()["feature_names"], clf.coef):
        print(f"  {name:<16}{coef:+.3f}")
    if not args.evaluate:
        print(f"\nweights written to {DEFAULT_WEIGHTS}")


if __name__ == "__main__":
    main()
