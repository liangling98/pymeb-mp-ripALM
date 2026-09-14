"""Small NVTX-scoped kernels for optional Nsight Compute measurements."""

from __future__ import annotations

import argparse
import torch

from .datasets import paper_meb_problem
from pymeb import MEBState, evaluate_subproblem_candidate, subproblem_hessian_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("evaluate", "hessian"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.target == "evaluate":
        problem = paper_meb_problem(100_000, 20, dtype=torch.float64, device="cuda")
        state = MEBState.initial_feasible(problem)
        for _ in range(2):
            evaluate_subproblem_candidate(
                problem, state, state.r, state.c, sigma=1.0, tau=1e-3
            )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("meb_evaluate")
        evaluate_subproblem_candidate(
            problem, state, state.r, state.c, sigma=1.0, tau=1e-3
        )
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    else:
        problem = paper_meb_problem(5_000, 500, dtype=torch.float64, device="cuda")
        state = MEBState.initial_feasible(problem)
        candidate = evaluate_subproblem_candidate(
            problem, state, state.r, state.c, sigma=1.0, tau=1e-3
        )
        for _ in range(2):
            subproblem_hessian_matrix(candidate, sigma=1.0, tau=1e-3)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("meb_hessian")
        subproblem_hessian_matrix(candidate, sigma=1.0, tau=1e-3)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
