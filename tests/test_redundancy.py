"""Algorithm I - redundancy / fault-tolerant fusion tests."""

import numpy as np

from iars.config import RedundancyConfig
from iars.redundancy.fusion import RedundancyFusion


def test_isolates_single_stuck_channel():
    fusion = RedundancyFusion(RedundancyConfig())
    samples = {"baro": 1000.0, "gnss": 1000.6, "inertial": 999.5, "pitot": 1001.0}
    res = fusion.update(samples)
    assert "gnss" in res.healthy and "baro" in res.healthy
    assert abs(res.value - 1000.0) < 2.0
    assert not res.degraded


def test_isolates_a_stuck_gnss():
    fusion = RedundancyFusion(RedundancyConfig())
    # GNSS frozen 60 m away - the median/MAD detector must reject it.
    res = fusion.update({"baro": 1000.0, "gnss": 1060.0, "inertial": 1001.0, "pitot": 999.0})
    assert "gnss" in res.isolated
    assert "gnss" not in res.healthy
    assert abs(res.value - 1000.0) < 2.0


def test_flags_degraded_when_no_consensus():
    fusion = RedundancyFusion(RedundancyConfig())
    # Four mutually inconsistent sources: no consensus, so the output must be
    # flagged degraded rather than trusting an arbitrary pair.
    res = fusion.update({"baro": 0.0, "gnss": 10_000.0, "inertial": 20_000.0, "pitot": 30_000.0})
    assert res.degraded
    assert len(res.healthy) < 2


def test_bias_tracker_downweights_drifting_source():
    fusion = RedundancyFusion(RedundancyConfig())
    for i in range(400):
        truth = 1000.0 + i * 0.1
        res = fusion.update(
            {
                "baro": truth,
                "inertial": truth,
                "gnss": truth,
                "pitot": truth + 8.0,  # slowly recognised as biased
            }
        )
    # After the tracker settles, the biased source must not pull the estimate.
    assert abs(res.value - truth) < 3.0


def test_variance_shrinks_with_more_sources():
    fusion = RedundancyFusion(RedundancyConfig())
    two = RedundancyFusion(RedundancyConfig()).update({"baro": 100.0, "gnss": 100.0})
    four = RedundancyFusion(RedundancyConfig()).update(
        {"baro": 100.0, "gnss": 100.0, "inertial": 100.0, "pitot": 100.0}
    )
    assert four.variance < two.variance
