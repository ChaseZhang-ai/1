"""Learned intent gate tests."""

import numpy as np
import pytest

from iars.config import StabilizationConfig
from iars.sensors.intent_dataset import generate_episode, make_episodes
from iars.stabilization.head_motion import HeadMotionSeparator
from iars.stabilization.intent import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    IntentClassifier,
    extract_features,
    heuristic_intent_gate,
    learned_intent_gate,
)

FS = 400.0


def _dataset(episodes):
    X, y = [], []
    for ep in episodes:
        f, centres = extract_features(ep.gyro_dps, FS)
        X.append(f)
        y.append(ep.label[centres])
    return np.vstack(X), np.concatenate(y)


def test_weights_are_packaged_and_well_formed():
    clf = IntentClassifier.load_default()
    assert DEFAULT_WEIGHTS.exists()
    assert clf.mean.shape == (len(FEATURE_NAMES),)
    assert clf.coef.shape == (len(FEATURE_NAMES),)
    assert 0.0 < clf.threshold < 1.0


def test_feature_extraction_shapes():
    ep = generate_episode(np.random.default_rng(1))
    X, centres = extract_features(ep.gyro_dps, FS)
    assert X.ndim == 2 and X.shape[1] == len(FEATURE_NAMES)
    assert len(centres) == len(X)
    assert np.all(np.isfinite(X))


def test_classifier_beats_heuristic_on_heldout_episodes():
    episodes = make_episodes(30, seed=424242)
    X, y = _dataset(episodes)
    clf = IntentClassifier.load_default()
    clf_acc = float(np.mean(clf.predict(X) == y))

    h_pred, h_true = [], []
    for ep in episodes:
        _, centres = extract_features(ep.gyro_dps, FS)
        h_pred.append(heuristic_intent_gate(ep.gyro_dps, FS)[centres])
        h_true.append(ep.label[centres])
    heur_acc = float(np.mean(np.concatenate(h_pred) == np.concatenate(h_true)))

    assert clf_acc > 0.90
    assert clf_acc > heur_acc


def test_stream_gate_is_per_sample_and_boolean():
    ep = generate_episode(np.random.default_rng(9))
    gate = learned_intent_gate(ep.gyro_dps, FS)
    assert gate.shape == (len(ep.gyro_dps),)
    assert gate.dtype == bool


def test_short_stream_does_not_crash():
    gate = learned_intent_gate(np.zeros((10, 3)), FS)
    assert gate.shape == (10,)


def test_learned_gate_drives_the_stabiliser():
    ep = generate_episode(np.random.default_rng(3))
    clf = IntentClassifier.load_default()
    sep = HeadMotionSeparator(StabilizationConfig(sample_rate_hz=FS), intent_classifier=clf)
    res = sep.stabilise(ep.gyro_dps)
    assert res.metrics["attenuation_db"] > 15.0
    assert res.intent.shape == (len(ep.gyro_dps),)
    assert res.intent.mean() > 0.0


def test_saved_roundtrip(tmp_path):
    clf = IntentClassifier.load_default()
    path = tmp_path / "w.json"
    clf.save(path)
    again = IntentClassifier.load(path)
    X = np.random.default_rng(0).standard_normal((12, len(FEATURE_NAMES)))
    assert np.allclose(clf.predict_proba(X), again.predict_proba(X))
