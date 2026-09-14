import unittest

import torch

from pymeb import MEBResult, MixedPrecisionOptions, RipALMOptions, solve


class PublicAPITests(unittest.TestCase):
    def test_solve_accepts_integer_point_lists(self):
        result = solve([[0, 0], [2, 0]])

        self.assertIsInstance(result, MEBResult)
        self.assertTrue(result.converged)
        self.assertEqual(result.center.dtype, torch.float64)
        self.assertAlmostEqual(float(result.radius), 1.0, places=6)
        self.assertTrue(bool(result.contains([[0, 0], [2, 0]]).all()))

    def test_solve_accepts_input_balls(self):
        result = solve([[0.0], [2.0]], [0.25, 0.5])

        self.assertTrue(result.converged)
        self.assertTrue(bool(result.contains([[0.0], [2.0]], [0.25, 0.5]).all()))
        self.assertAlmostEqual(float(result.radius), 1.375, places=6)

    def test_options_must_match_method(self):
        with self.assertRaises(TypeError):
            solve([[0.0], [1.0]], options=MixedPrecisionOptions())
        with self.assertRaises(TypeError):
            solve(
                [[0.0], [1.0]],
                method="mp-ripalm",
                options=RipALMOptions(),
            )

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            solve([[0.0], [1.0]], method="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
