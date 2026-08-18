from __future__ import annotations

import unittest

import numpy as np

from evaluation.results.core import compute_prediction_metrics


def _arrays() -> dict[str, np.ndarray]:
    base = np.full((3, 2, 2), 0.40, dtype=np.float32)
    true = base.copy()
    true[0] += 0.01
    true[1] += 0.02
    true[2] += 0.03
    pred = base.copy()
    pred[0] += 0.005
    pred[1] += 0.015
    pred[2] += 0.025
    mask = np.ones_like(base)
    return {"pred": pred, "true": true, "mask": mask, "base": base}


class ResultsMetricTests(unittest.TestCase):
    def test_cube_and_step_metrics_use_persistence_and_response(self) -> None:
        cube, steps = compute_prediction_metrics(
            _arrays(),
            eligible_veg=np.ones((2, 2), dtype=bool),
            min_valid_target_coverage=0.0,
            min_valid_target_count=1,
            min_target_variance=0.0,
            response_threshold=0.001,
        )

        self.assertEqual(len(steps), 3)
        self.assertTrue(np.isclose(steps[-1]["observed_response"], 0.03))
        self.assertTrue(np.isclose(steps[-1]["predicted_response"], 0.025))
        self.assertTrue(np.isclose(cube["observed_r30"], 0.03))
        self.assertTrue(np.isclose(cube["predicted_r30"], 0.025))
        self.assertGreater(cube["mse_skill"], 0)
        self.assertGreater(cube["mae_gain"], 0)
        self.assertEqual(cube["persistence_beaten"], 1.0)
        self.assertEqual(cube["valid_target_coverage"], 1.0)

    def test_invalid_pixels_never_enter_metrics(self) -> None:
        arrays = _arrays()
        arrays["pred"][:, 0, 0] = 99.0
        arrays["true"][:, 0, 0] = -99.0
        arrays["mask"][:, 0, 0] = 0.0
        cube, _ = compute_prediction_metrics(
            arrays,
            eligible_veg=np.ones((2, 2), dtype=bool),
            min_valid_target_coverage=0.0,
            min_valid_target_count=1,
            min_target_variance=0.0,
        )
        self.assertLess(cube["mae"], 0.01)
        self.assertEqual(cube["n_valid_target_pixel_times"], 9)

    def test_response_error_is_not_reported_as_duplicate_delta_mse(self) -> None:
        cube, steps = compute_prediction_metrics(
            _arrays(),
            eligible_veg=np.ones((2, 2), dtype=bool),
            min_valid_target_coverage=0.0,
            min_valid_target_count=1,
            min_target_variance=0.0,
        )
        self.assertNotIn("mse_delta", cube)
        self.assertNotIn("mse_delta", steps[0])
        self.assertTrue(
            np.isclose(steps[0]["response_mean_abs_error"], 0.005)
        )


if __name__ == "__main__":
    unittest.main()
