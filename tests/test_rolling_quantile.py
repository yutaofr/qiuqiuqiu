import numpy as np

from qqq_cycle.core.rolling_quantile import rolling_quantile_diag_2d


def test_rolling_quantile_diag_2d_uses_expanding_then_rolling_window() -> None:
    delta_v = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    out = rolling_quantile_diag_2d(delta_v, window=3, quantile=0.5)

    assert out.shape == (4, 2, 2)
    np.testing.assert_allclose(np.diag(out[0]), [1.0, 100.0])
    np.testing.assert_allclose(np.diag(out[1]), [2.5, 250.0])
    np.testing.assert_allclose(np.diag(out[3]), [9.0, 900.0])
    assert out[3, 0, 1] == 0.0
    assert out[3, 1, 0] == 0.0


def test_rolling_quantile_diag_2d_ignores_nan_without_future_fill() -> None:
    delta_v = np.array(
        [
            [1.0, 2.0],
            [np.nan, 4.0],
            [3.0, np.nan],
        ]
    )

    out = rolling_quantile_diag_2d(delta_v, window=3, quantile=0.5)

    np.testing.assert_allclose(np.diag(out[2]), [5.0, 10.0])


def test_rolling_quantile_diag_2d_squares_raw_delta_v_internally() -> None:
    delta_v = np.array([[-2.0, 3.0], [4.0, -5.0]])

    out = rolling_quantile_diag_2d(delta_v, window=2, quantile=1.0)

    np.testing.assert_allclose(np.diag(out[1]), [16.0, 25.0])
