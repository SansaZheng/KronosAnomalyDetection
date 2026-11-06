import numpy as np
import pytest
from kronos_ad.joint import JointResidualCalibrator


def test_frozen_calibration_and_scale_floor():
    rng = np.random.default_rng(42)
    residual = rng.normal(size=(400, 4))
    local = np.ones_like(residual)
    fitted = JointResidualCalibrator().fit(residual, local, np.ones(4))
    before = fitted.state()
    ordinary = fitted.score(residual[:2], local[:2])
    extreme = fitted.score(np.r_[residual[:2], [[1e9] * 4]], np.ones((3, 4)))
    np.testing.assert_array_equal(ordinary, extreme[:2])
    assert fitted.state() == before
    assert extreme[-1] > 1e8
    assert np.isfinite(fitted.score(residual[:2], np.zeros((2, 4)))).all()


def test_ties_do_not_all_alarm_and_small_calibration_rejected():
    x = np.ones((200, 2))
    model = JointResidualCalibrator().fit(x, x, np.ones(2))
    assert not model.predict(x, x).any()
    with pytest.raises(ValueError, match="too small"):
        JointResidualCalibrator(risk=.001).fit(x, x, np.ones(2))
