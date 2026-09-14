import unittest

import torch

from pymeb import (
    MEBProblem,
    MEBState,
    check_relative_inexact_condition,
    evaluate_subproblem_candidate,
    line_search_newton_step,
    project_soc,
    semismooth_newton_direction,
    soc_jvp,
    subproblem_hessian_matrix,
)


class MEBCoreTests(unittest.TestCase):
    def test_problem_rejects_invalid_public_inputs(self):
        with self.assertRaises(TypeError):
            MEBProblem([[0, 1]], [0])
        with self.assertRaises(ValueError):
            MEBProblem(torch.empty(0, 2), torch.empty(0))

    def test_initial_feasible_state_shapes_and_slacks(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            radii=[0.0, 0.0, 0.0],
            dtype=torch.float64,
        )

        state = MEBState.initial_feasible(problem)

        self.assertEqual(problem.m, 3)
        self.assertEqual(problem.d, 2)
        self.assertEqual(state.u.shape, (3,))
        self.assertEqual(state.v.shape, (3, 2))
        self.assertEqual(state.primal_vector().shape, (3,))
        self.assertEqual(state.auxiliary_vector().shape, (3,))
        self.assertTrue(torch.all(problem.slack(state) >= -1e-12))
        self.assertTrue(torch.allclose(state.auxiliary_vector(), torch.zeros(3, dtype=torch.float64)))

    def test_subset_restricts_problem_and_state_consistently(self):
        problem = MEBProblem.from_tensors(
            centers=torch.arange(12, dtype=torch.float64).reshape(4, 3),
            radii=torch.arange(4, dtype=torch.float64),
        )
        state = MEBState.initial_feasible(problem).with_dual(
            u=torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64),
            v=torch.arange(12, dtype=torch.float64).reshape(4, 3),
        )

        reduced_problem, reduced_state = problem.restrict(torch.tensor([3, 1]), state)

        self.assertTrue(torch.equal(reduced_problem.original_indices, torch.tensor([3, 1])))
        self.assertTrue(torch.equal(reduced_problem.radii, torch.tensor([3.0, 1.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(reduced_state.u, torch.tensor([40.0, 20.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(reduced_state.v, state.v.index_select(0, torch.tensor([3, 1]))))

    def test_align_constraints_copies_old_multipliers_and_zeros_new_ones(self):
        full = MEBProblem.from_tensors(
            centers=torch.arange(15, dtype=torch.float64).reshape(5, 3),
            radii=torch.zeros(5, dtype=torch.float64),
        )
        source = full.subset(torch.tensor([1, 4]))
        target = full.subset(torch.tensor([4, 2, 1]))
        source_state = MEBState.initial_feasible(source).with_dual(
            u=torch.tensor([100.0, 400.0], dtype=torch.float64),
            v=torch.tensor(
                [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]],
                dtype=torch.float64,
            ),
        )

        aligned = source_state.align_constraints(source, target)

        self.assertTrue(torch.equal(aligned.u, torch.tensor([400.0, 0.0, 100.0], dtype=torch.float64)))
        self.assertTrue(
            torch.equal(
                aligned.v,
                torch.tensor(
                    [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    dtype=torch.float64,
                ),
            )
        )
        self.assertTrue(torch.allclose(aligned.primal_vector(), source_state.primal_vector()))
        self.assertTrue(torch.allclose(aligned.auxiliary_vector(), source_state.auxiliary_vector()))

    def test_screening_and_feasibility_helpers(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [10.0], [5.0]],
            radii=[0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        state = MEBState.zeros(problem, r=6.0, c=torch.tensor([0.0], dtype=torch.float64))

        kept = problem.screen(state, tol_red=0.5, min_keep=2)
        violation = problem.violation(state)
        violated = problem.violated_indices(state, tol=1e-8)

        self.assertTrue(torch.equal(kept, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(violated, torch.tensor([1])))
        self.assertAlmostEqual(float(violation.max()), 4.0)

    def test_dtype_conversion_for_two_phase_flow(self):
        full32 = MEBProblem.from_tensors(
            centers=[[0.0, 0.0], [1.0, 0.0]],
            radii=[0.0, 0.25],
            dtype=torch.float32,
        )
        low_state = MEBState.initial_feasible(full32)
        high_full = full32.to(dtype=torch.float64)
        high_state = low_state.to(dtype=torch.float64)

        high_state.validate_for(high_full)
        self.assertEqual(high_full.dtype, torch.float64)
        self.assertEqual(high_state.dtype, torch.float64)
        self.assertTrue(torch.equal(high_full.original_indices, full32.original_indices))

    def test_soc_projection_regions(self):
        z = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [-3.0, 1.0, 2.0],
                [1.0, 3.0, 4.0],
            ],
            dtype=torch.float64,
        )

        projected = project_soc(z)

        expected_boundary = torch.tensor([3.0, 1.8, 2.4], dtype=torch.float64)
        expected = torch.stack((z[0], torch.zeros(3, dtype=torch.float64), expected_boundary))

        self.assertTrue(torch.allclose(projected, expected))

    def test_cone_projection_does_not_track_autograd(self):
        z = torch.tensor([[1.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)

        projected = project_soc(z)

        self.assertFalse(projected.requires_grad)

    def test_soc_projection_derivative_components(self):
        z = torch.tensor([[1.0, 3.0, 4.0]], dtype=torch.float64)
        info = project_soc(z, return_info=True, return_jacobian=True)

        expected_jacobian = torch.tensor(
            [
                [0.5, 0.3, 0.4],
                [0.3, 0.564, -0.048],
                [0.4, -0.048, 0.536],
            ],
            dtype=torch.float64,
        )

        self.assertTrue(info.middle.item())
        self.assertTrue(torch.allclose(info.value, torch.tensor([[3.0, 1.8, 2.4]], dtype=torch.float64)))
        self.assertTrue(torch.equal(info.middle_indices, torch.tensor([0])))
        self.assertTrue(torch.allclose(info.middle_jacobian[0], expected_jacobian))

    def test_soc_matrix_free_jacobian_product_matches_dense_component(self):
        z = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [-3.0, 1.0, 2.0],
                [1.0, 3.0, 4.0],
            ],
            dtype=torch.float64,
        )
        info = project_soc(z, return_info=True, return_jacobian=True)
        da = torch.tensor(
            [
                [0.5, -1.0, 2.0],
                [1.0, 3.0, -2.0],
                [-0.25, 0.75, -1.5],
            ],
            dtype=torch.float64,
        )

        expected = torch.empty_like(da)
        expected[0] = da[0]
        expected[1] = torch.zeros_like(da[1])
        expected[2] = torch.einsum("ij,j->i", info.middle_jacobian[0], da[2])

        self.assertTrue(torch.equal(info.middle_indices, torch.tensor([2])))
        self.assertTrue(torch.allclose(soc_jvp(info, da), expected))

    def test_subproblem_candidate_evaluation_matches_manual_formula(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0, 0.0], [2.0, 0.0]],
            radii=[0.25, 0.5],
            dtype=torch.float64,
        )
        state = MEBState(
            r=2.0,
            c=torch.tensor([0.5, -0.25], dtype=torch.float64),
            u=torch.tensor([0.2, 0.1], dtype=torch.float64),
            v=torch.tensor([[0.05, -0.1], [0.2, 0.0]], dtype=torch.float64),
            alpha=torch.tensor(1.75, dtype=torch.float64),
            beta=torch.tensor([0.25, -0.1], dtype=torch.float64),
        )
        r = torch.tensor(1.6, dtype=torch.float64)
        c = torch.tensor([0.75, -0.15], dtype=torch.float64)
        sigma = torch.tensor(2.0, dtype=torch.float64)
        tau = torch.tensor(0.8, dtype=torch.float64)
        rho = torch.tensor(0.9, dtype=torch.float64)

        candidate = evaluate_subproblem_candidate(
            problem, state, r, c, sigma=sigma, tau=tau
        )

        cone_points = torch.cat(
            (
                (problem.radii - r + state.u / sigma).unsqueeze(1),
                problem.centers - c + state.v / sigma,
            ),
            dim=1,
        )
        dual_next = sigma * project_soc(cone_points)
        projected = project_soc(cone_points)
        u_next = dual_next[:, 0]
        v_next = dual_next[:, 1:]
        gradient = torch.cat(
            (
                (1.0 - u_next.sum() + (tau / sigma) * (r - state.r)).reshape(1),
                -v_next.sum(dim=0) + (tau / sigma) * (c - state.c),
            )
        )
        alpha_beta_gap = torch.cat(((state.alpha - r).reshape(1), state.beta - c))
        lhs = 2.0 * sigma * (alpha_beta_gap * gradient).sum()
        lhs = torch.abs(lhs + sigma.square() * (gradient * gradient).sum())
        dual_step = torch.cat(
            ((u_next - state.u).unsqueeze(1), v_next - state.v),
            dim=1,
        )
        primal_step_norm_sq = (r - state.r).square() + ((c - state.c) ** 2).sum()
        rhs = rho * ((dual_step * dual_step).sum() + tau * primal_step_norm_sq)
        dual_norm_sq = (state.u * state.u).sum() + (state.v * state.v).sum()
        objective = (
            r
            + 0.5 * sigma * (projected * projected).sum()
            - 0.5 * dual_norm_sq / sigma
            + 0.5 * tau * primal_step_norm_sq / sigma
        )
        check = check_relative_inexact_condition(
            problem, state, r, c, sigma=sigma, tau=tau, rho=rho, gradient_tol=1e9
        )

        self.assertTrue(torch.allclose(candidate.r, r))
        self.assertTrue(torch.allclose(candidate.c, c))
        self.assertTrue(torch.allclose(candidate.u, u_next))
        self.assertTrue(torch.allclose(candidate.v, v_next))
        self.assertTrue(torch.allclose(candidate.gradient, gradient))
        self.assertTrue(torch.allclose(candidate.objective, objective))
        self.assertTrue(torch.allclose(candidate.primal_vector(), torch.cat((r.reshape(1), c))))
        self.assertTrue(torch.allclose(check.candidate.u, u_next))
        self.assertTrue(torch.allclose(check.candidate.v, v_next))
        self.assertTrue(torch.allclose(check.candidate.gradient, gradient))
        self.assertTrue(torch.allclose(check.gradient_norm, torch.linalg.vector_norm(gradient)))
        self.assertTrue(bool(check.gradient_satisfied))
        self.assertEqual(bool(check.relative_satisfied), bool(lhs <= rhs))
        self.assertTrue(torch.allclose(check.lhs, lhs))
        self.assertTrue(torch.allclose(check.rhs, rhs))
        self.assertEqual(bool(check.satisfied), bool(lhs <= rhs) or bool(check.gradient_satisfied))

    def test_subproblem_candidate_does_not_track_autograd(self):
        centers = torch.tensor([[0.0], [2.0]], dtype=torch.float64, requires_grad=True)
        problem = MEBProblem.from_tensors(centers, [0.0, 0.0])
        state = MEBState.initial_feasible(problem)

        candidate = evaluate_subproblem_candidate(
            problem, state, state.r, state.c, sigma=10.0, tau=1e-4
        )

        self.assertFalse(candidate.r.requires_grad)
        self.assertFalse(candidate.c.requires_grad)
        self.assertFalse(candidate.gradient.requires_grad)

    def test_relative_inexact_condition_satisfaction_depends_on_rho(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [3.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )
        state = MEBState(
            r=2.0,
            c=torch.tensor([1.0], dtype=torch.float64),
            u=torch.zeros(2, dtype=torch.float64),
            v=torch.zeros(2, 1, dtype=torch.float64),
            alpha=torch.tensor(5.0, dtype=torch.float64),
            beta=torch.tensor([0.0], dtype=torch.float64),
        )
        r = torch.tensor(1.75, dtype=torch.float64)
        c = torch.tensor([1.25], dtype=torch.float64)

        loose = check_relative_inexact_condition(
            problem, state, r, c, sigma=1.0, tau=1.0, rho=50.0, gradient_tol=1e9
        )
        strict = check_relative_inexact_condition(
            problem, state, r, c, sigma=1.0, tau=1.0, rho=0.0, gradient_tol=0.0
        )
        loose_with_grad = check_relative_inexact_condition(
            problem, state, r, c, sigma=1.0, tau=1.0, rho=50.0, gradient_tol=0.0
        )

        self.assertTrue(bool(loose.satisfied))
        self.assertFalse(bool(strict.relative_satisfied))
        self.assertFalse(bool(strict.gradient_satisfied))
        self.assertFalse(bool(strict.satisfied))
        self.assertTrue(bool(loose_with_grad.relative_satisfied))
        self.assertFalse(bool(loose_with_grad.gradient_satisfied))
        self.assertTrue(bool(loose_with_grad.satisfied))

    def test_relative_inexact_condition_rejects_large_negative_signed_lhs(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [200.0]],
            radii=[0.0, 0.0],
            dtype=torch.float64,
        )
        state = MEBState.initial_feasible(problem)

        check = check_relative_inexact_condition(
            problem,
            state,
            state.r,
            state.c,
            sigma=10.0,
            tau=1e-4,
            rho=0.5,
            gradient_tol=1e-10,
        )

        self.assertAlmostEqual(float(check.gradient_norm), 1.0)
        self.assertGreater(float(check.lhs), 0.0)
        self.assertEqual(float(check.rhs), 0.0)
        self.assertFalse(bool(check.relative_satisfied))
        self.assertFalse(bool(check.satisfied))

    def test_subproblem_hessian_matrix_matches_gradient_finite_difference(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            radii=[0.0, 0.25, 0.5],
            dtype=torch.float64,
        )
        state = MEBState(
            r=1.0,
            c=torch.tensor([0.1, -0.2], dtype=torch.float64),
            u=torch.tensor([0.3, 0.1, 0.2], dtype=torch.float64),
            v=torch.tensor(
                [[0.1, -0.2], [0.05, 0.1], [-0.1, 0.05]],
                dtype=torch.float64,
            ),
        )
        sigma = torch.tensor(1.7, dtype=torch.float64)
        tau = torch.tensor(0.9, dtype=torch.float64)
        r = torch.tensor(1.4, dtype=torch.float64)
        c = torch.tensor([0.4, -0.35], dtype=torch.float64)
        direction = torch.tensor([0.2, -0.15, 0.1], dtype=torch.float64)
        eps = torch.tensor(1e-6, dtype=torch.float64)

        candidate = evaluate_subproblem_candidate(
            problem, state, r, c, sigma=sigma, tau=tau
        )
        hv = subproblem_hessian_matrix(candidate, sigma=sigma, tau=tau).matmul(direction)
        plus = evaluate_subproblem_candidate(
            problem,
            state,
            r + eps * direction[0],
            c + eps * direction[1:],
            sigma=sigma,
            tau=tau,
        )
        minus = evaluate_subproblem_candidate(
            problem,
            state,
            r - eps * direction[0],
            c - eps * direction[1:],
            sigma=sigma,
            tau=tau,
        )
        finite_difference = (plus.gradient - minus.gradient) / (2.0 * eps)

        self.assertTrue(torch.allclose(hv, finite_difference, atol=1e-6, rtol=1e-6))

    def test_subproblem_hessian_matrix_is_symmetric(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            radii=[0.0, 0.25, 0.5],
            dtype=torch.float64,
        )
        state = MEBState(
            r=1.0,
            c=torch.tensor([0.1, -0.2], dtype=torch.float64),
            u=torch.tensor([0.3, 0.1, 0.2], dtype=torch.float64),
            v=torch.tensor(
                [[0.1, -0.2], [0.05, 0.1], [-0.1, 0.05]],
                dtype=torch.float64,
            ),
        )
        sigma = torch.tensor(1.7, dtype=torch.float64)
        tau = torch.tensor(0.9, dtype=torch.float64)
        candidate = evaluate_subproblem_candidate(
            problem,
            state,
            torch.tensor(1.4, dtype=torch.float64),
            torch.tensor([0.4, -0.35], dtype=torch.float64),
            sigma=sigma,
            tau=tau,
        )

        matrix = subproblem_hessian_matrix(candidate, sigma=sigma, tau=tau)

        self.assertTrue(torch.allclose(matrix, matrix.T, atol=1e-12))

    def test_semismooth_newton_direction_solves_dense_system(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [3.0], [1.0]],
            radii=[0.0, 0.0, 0.25],
            dtype=torch.float64,
        )
        state = MEBState(
            r=2.0,
            c=torch.tensor([1.0], dtype=torch.float64),
            u=torch.tensor([0.2, 0.1, 0.05], dtype=torch.float64),
            v=torch.tensor([[0.1], [-0.05], [0.02]], dtype=torch.float64),
        )
        sigma = torch.tensor(1.3, dtype=torch.float64)
        tau = torch.tensor(0.7, dtype=torch.float64)
        candidate = evaluate_subproblem_candidate(
            problem,
            state,
            torch.tensor(1.6, dtype=torch.float64),
            torch.tensor([1.25], dtype=torch.float64),
            sigma=sigma,
            tau=tau,
        )

        newton = semismooth_newton_direction(candidate, sigma=sigma, tau=tau)
        residual = (
            subproblem_hessian_matrix(candidate, sigma=sigma, tau=tau).matmul(newton.direction)
            + candidate.gradient
        )

        self.assertTrue(bool(newton.converged))
        self.assertEqual(newton.iterations, 1)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-10))

    def test_line_search_newton_step_accepts_armijo_trial(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [3.0], [1.0]],
            radii=[0.0, 0.0, 0.25],
            dtype=torch.float64,
        )
        state = MEBState(
            r=2.0,
            c=torch.tensor([1.0], dtype=torch.float64),
            u=torch.tensor([0.2, 0.1, 0.05], dtype=torch.float64),
            v=torch.tensor([[0.1], [-0.05], [0.02]], dtype=torch.float64),
        )
        sigma = torch.tensor(1.3, dtype=torch.float64)
        tau = torch.tensor(0.7, dtype=torch.float64)
        candidate = evaluate_subproblem_candidate(
            problem,
            state,
            torch.tensor(1.6, dtype=torch.float64),
            torch.tensor([1.25], dtype=torch.float64),
            sigma=sigma,
            tau=tau,
        )
        newton = semismooth_newton_direction(candidate, sigma=sigma, tau=tau)

        result = line_search_newton_step(
            problem,
            state,
            candidate,
            newton,
            sigma=sigma,
            tau=tau,
            max_iter=8,
        )
        directional_derivative = (candidate.gradient * newton.direction).sum()
        armijo_rhs = (
            candidate.objective
            + 1e-4 * result.step_size * directional_derivative
        )

        self.assertTrue(bool(result.accepted))
        self.assertLessEqual(result.iterations, 8)
        self.assertTrue(torch.allclose(result.objective, result.candidate.objective))
        self.assertTrue(result.candidate.objective <= armijo_rhs)

    def test_line_search_newton_step_stops_at_first_accepted_trial(self):
        problem = MEBProblem.from_tensors(
            centers=[[0.0], [3.0], [1.0]],
            radii=[0.0, 0.0, 0.25],
            dtype=torch.float64,
        )
        state = MEBState(
            r=2.0,
            c=torch.tensor([1.0], dtype=torch.float64),
            u=torch.tensor([0.2, 0.1, 0.05], dtype=torch.float64),
            v=torch.tensor([[0.1], [-0.05], [0.02]], dtype=torch.float64),
        )
        sigma = torch.tensor(1.3, dtype=torch.float64)
        tau = torch.tensor(0.7, dtype=torch.float64)
        candidate = evaluate_subproblem_candidate(
            problem,
            state,
            torch.tensor(1.6, dtype=torch.float64),
            torch.tensor([1.25], dtype=torch.float64),
            sigma=sigma,
            tau=tau,
        )
        newton = semismooth_newton_direction(candidate, sigma=sigma, tau=tau)

        result = line_search_newton_step(
            problem,
            state,
            candidate,
            newton,
            sigma=sigma,
            tau=tau,
            max_iter=8,
        )

        self.assertTrue(bool(result.accepted))
        self.assertLess(result.iterations, 8)
        self.assertTrue(result.step_size <= 1.0)


if __name__ == "__main__":
    unittest.main()
