import unittest

import torch

from pymeb import (
    MEBProblem,
    MEBState,
    MixedPrecisionOptions,
    RipALMOptions,
    mixed_precision_ripalm,
    screen_by_boundary_ratio,
)


class MixedPrecisionTests(unittest.TestCase):
    def test_boundary_ratio_screening_keeps_near_boundary_balls(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [9.6], [10.0]],
            radii=[0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        state = MEBState.zeros(problem, r=10.0, c=torch.tensor([0.0]))

        kept, threshold = screen_by_boundary_ratio(problem, state, ratio=0.95)

        self.assertTrue(torch.equal(kept, torch.tensor([1, 2])))
        self.assertTrue(torch.allclose(threshold, torch.tensor(9.5)))

    def test_mixed_precision_solves_two_point_meb(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )
        options = MixedPrecisionOptions(
            screening_ratio=0.95,
            low_options=RipALMOptions(
                outer_tol=1e-4,
                gradient_tol=1e-4,
                max_outer=4,
                max_inner=4,
                fixed_outer_iterations=True,
                fixed_inner_iterations=True,
                print_history=False,
            ),
            high_options=RipALMOptions(
                outer_tol=1e-7,
                gradient_tol=1e-10,
                max_outer=40,
                max_inner=10,
                print_history=False,
            ),
            feasibility_abs_tol=1e-7,
            record_phase_timings=True,
        )

        result = mixed_precision_ripalm(problem, options=options)

        self.assertTrue(result.accepted)
        self.assertTrue(result.converged)
        self.assertEqual(result.low_result.state.dtype, torch.float32)
        self.assertEqual(result.high_result.state.dtype, torch.float64)
        self.assertEqual(result.state.dtype, torch.float64)
        self.assertTrue(torch.equal(result.full_problem.centers, problem.centers))
        self.assertTrue(torch.equal(result.full_problem.radii, problem.radii))
        self.assertEqual(result.reduced_problem.m, 2)
        self.assertLessEqual(float(result.max_violation), float(result.feasibility_tol))
        self.assertAlmostEqual(float(result.state.r), 1.0, places=6)
        self.assertIsNotNone(result.timings)
        assert result.timings is not None
        self.assertGreater(result.timings.total_sec, 0.0)
        self.assertGreater(result.timings.low_solve_sec, 0.0)
        self.assertGreater(result.timings.high_solve_sec, 0.0)
        self.assertTrue(
            torch.allclose(
                result.state.c,
                torch.tensor([1.0], dtype=torch.float64),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
