import numpy as np
from hydrus_research.optimization import encode_schedule, decode_schedule


def test_roundtrip():
    events = [{"amount": 1.5, "day": 10}, {"amount": 2.0, "day": 30}]
    theta = encode_schedule(events)
    assert theta.shape == (4,)
    events2 = decode_schedule(theta)
    assert events2[0]["amount"] == 1.5
    assert events2[1]["day"] == 30
