import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch

from pymeb import MEBProblem, RipALMOptions, ripalm, ripalm_residual
from pymeb.ripalm import _gradient_tol
from experiments.datasets import paper_meb_problem


class RipALMDriverTests(unittest.TestCase):
    def test_ripalm_solves_two_point_meb(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )

        result = ripalm(
            problem,
            options=RipALMOptions(
                outer_tol=1e-7,
                gradient_tol=1e-12,
                max_outer=40,
                max_inner=10,
                print_history=False,
            ),
        )

        self.assertTrue(result.converged)
        self.assertGreater(len(result.history), 0)
        self.assertEqual(result.history[-1].iteration, result.outer_iterations)
        self.assertEqual(
            result.line_search_evaluations,
            sum(item.line_search_evaluations for item in result.history),
        )
        self.assertEqual(
            result.accepted_full_steps,
            sum(item.accepted_full_steps for item in result.history),
        )
        self.assertGreaterEqual(result.line_search_evaluations, result.inner_iterations)
        self.assertLessEqual(result.accepted_full_steps, result.inner_iterations)
        self.assertTrue(torch.allclose(result.history[-1].residual.value, result.residual.value))
        self.assertLessEqual(float(result.residual.value), 1e-7)
        self.assertAlmostEqual(float(result.state.r), 1.0, places=6)
        self.assertTrue(torch.allclose(result.state.c, torch.tensor([1.0], dtype=torch.float64), atol=1e-6))
        self.assertEqual(result.state.u.shape, (2,))
        self.assertEqual(result.state.v.shape, (2, 1))

    def test_ripalm_residual_detects_nonoptimal_feasible_state(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )
        result = ripalm(
            problem,
            options=RipALMOptions(max_outer=3, outer_tol=0.0, print_history=False),
        )
        residual = ripalm_residual(problem, result.state)

        self.assertTrue(torch.allclose(residual.value, result.residual.value))
        self.assertGreaterEqual(float(residual.value), 0.0)

    def test_ripalm_prints_history_when_requested(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )
        output = StringIO()

        with redirect_stdout(output):
            result = ripalm(
                problem,
                options=RipALMOptions(max_outer=2, outer_tol=0.0, print_history=True),
            )

        text = output.getvalue()
        self.assertIn("iter", text)
        self.assertIn("residual", text)
        self.assertEqual(text.count("\n"), len(result.history) + 1)

    def test_gradient_tolerance_defaults_follow_dtype(self):
        self.assertEqual(_gradient_tol(torch.float32, None), 1e-3)
        self.assertEqual(_gradient_tol(torch.float64, None), 1e-8)
        self.assertEqual(_gradient_tol(torch.float64, 1e-6), 1e-6)

    def test_options_name_line_search_max_iter(self):
        options = RipALMOptions(line_search_max_iter=7)

        self.assertEqual(options.line_search_max_iter, 7)

    def test_sigma_growth_factor_is_recorded(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )

        result = ripalm(
            problem,
            options=RipALMOptions(
                sigma=2.0,
                sigma_growth_factor=5.0,
                max_outer=3,
                max_inner=1,
                fixed_outer_iterations=True,
                print_history=False,
            ),
        )

        sigmas = torch.tensor([float(item.sigma) for item in result.history])
        self.assertTrue(torch.allclose(sigmas, torch.tensor([2.0, 10.0, 50.0])))

    def test_sigma_growth_factor_must_be_positive(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )

        with self.assertRaises(ValueError):
            ripalm(
                problem,
                options=RipALMOptions(sigma_growth_factor=0.0, print_history=False),
            )

    def test_fixed_iteration_modes_use_full_budgets(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [2.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )

        result = ripalm(
            problem,
            options=RipALMOptions(
                max_outer=3,
                max_inner=2,
                fixed_outer_iterations=True,
                fixed_inner_iterations=True,
                print_history=False,
            ),
        )

        self.assertEqual(result.outer_iterations, 3)
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.inner_iterations, 6)
        self.assertTrue(all(item.inner_iterations == 2 for item in result.history))

    def test_ripalm_solves_generated_meb_without_stalling(self):
        problem = paper_meb_problem(20, 10, dtype=torch.float64)
        initial = float(problem.radii.add(problem.distances(problem.centers.mean(dim=0))).max())

        result = ripalm(
            problem,
            options=RipALMOptions(
                outer_tol=1e-8,
                max_outer=20,
                max_inner=20,
                print_history=False,
            ),
        )

        self.assertTrue(result.converged)
        self.assertLess(float(result.state.r), initial)
        self.assertLessEqual(float(result.residual.value), 1e-8)
        self.assertLessEqual(float(problem.violation(result.state).max().clamp_min(0.0)), 1e-8)


if __name__ == "__main__":
    unittest.main()
