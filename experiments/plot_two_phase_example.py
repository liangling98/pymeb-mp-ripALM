"""Plot a toy 2D MEB instance for the two-phase mixed-precision algorithm.

Run:

    python -m experiments.plot_two_phase_example
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import torch

from pymeb import MEBProblem, MixedPrecisionOptions, RipALMOptions, mixed_precision_ripalm


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "local"
PDF_PATH = OUTPUT_DIR / "meb_two_phase_toy_example.pdf"
PNG_PATH = OUTPUT_DIR / "meb_two_phase_toy_example.png"

FIGSIZE = (14.5, 5.8)
BALL_EDGEWIDTH = 1.4
MEB_LINEWIDTH = 2.6
CENTER_SIZE = 45
POINT_SIZE = 18

COLORS = {
    "all_ball": "#7aa6c2",
    "kept_ball": "#e68613",
    "dropped_ball": "#d4d8dd",
    "support_ball": "#c44536",
    "low_meb": "#2c7fb8",
    "threshold": "#f18f01",
    "high_meb": "#2a9d8f",
    "center": "#1f2933",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = _toy_problem()
    options = _toy_options()
    result = mixed_precision_ripalm(problem, options=options)

    low_state = result.low_result.state.to(dtype=torch.float64, device=problem.device)
    high_state = result.state.to(dtype=torch.float64, device=problem.device)
    kept = result.kept_indices.detach().cpu().tolist()
    kept_set = set(kept)

    support_tol = 5e-6
    support_mask = problem.slack(high_state, dtype=torch.float64).abs() <= support_tol
    support = torch.nonzero(support_mask, as_tuple=False).flatten().detach().cpu().tolist()
    support_set = set(support)

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, height_ratios=[5.0, 0.95])

    _plot_input_panel(axes[0, 0], problem)
    _plot_screening_panel(
        axes[0, 1],
        problem,
        low_state,
        screening_ratio=options.screening_ratio,
        threshold=float(result.screening_threshold),
        kept_set=kept_set,
    )
    _plot_refinement_panel(
        axes[0, 2],
        problem,
        high_state,
        kept_set=kept_set,
        support_set=support_set,
        accepted=result.accepted,
    )
    _add_legend_row(
        axes[1, 0],
        handles=[
            _ball_legend_handle("Input balls", COLORS["all_ball"], "#4d7187"),
        ],
    )
    _add_legend_row(
        axes[1, 1],
        handles=[
            _ball_legend_handle("Kept after screening", COLORS["kept_ball"], "#9a5a0a"),
            _ball_legend_handle("Dropped after screening", COLORS["dropped_ball"], "#a8b0b7"),
            _line_legend_handle("float32 MEB", COLORS["low_meb"], "-"),
            _line_legend_handle("screening threshold", COLORS["threshold"], "--"),
        ],
    )
    _add_legend_row(
        axes[1, 2],
        handles=[
            _ball_legend_handle("Active constraints", COLORS["support_ball"], "#8f2d23"),
            _ball_legend_handle("Inactive kept balls", COLORS["kept_ball"], "#9a5a0a"),
            _line_legend_handle("float64 MEB", COLORS["high_meb"], "-"),
            _center_legend_handle(),
        ],
    )

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved toy two-phase figure to {PDF_PATH}")
    print(f"Saved toy two-phase figure to {PNG_PATH}")
    print(
        "Summary:"
        f" kept {len(kept)}/{problem.m} constraints after float32 screening,"
        f" support size {len(support)},"
        f" accepted={result.accepted},"
        f" refinements={result.refinements}"
    )


def _toy_problem() -> MEBProblem:
    centers = [
        [-2.6, -0.4],
        [-1.8, 1.9],
        [0.4, 2.6],
        [2.7, 0.7],
        [1.8, -2.1],
        [-0.8, -2.4],
        [0.1, 0.2],
        [0.8, 0.7],
        [-0.7, 1.0],
        [1.2, -0.4],
    ]
    radii = [0.25, 0.18, 0.22, 0.20, 0.24, 0.17, 0.12, 0.10, 0.11, 0.09]
    return MEBProblem.from_tensors(centers=centers, radii=radii, dtype=torch.float64)


def _toy_options() -> MixedPrecisionOptions:
    return MixedPrecisionOptions(
        screening_ratio=0.94,
        low_options=RipALMOptions(
            outer_tol=1e-4,
            gradient_tol=1e-4,
            max_outer=20,
            max_inner=10,
            fixed_outer_iterations=True,
            fixed_inner_iterations=True,
            print_history=False,
        ),
        high_options=RipALMOptions(
            outer_tol=1e-8,
            gradient_tol=1e-10,
            max_outer=80,
            max_inner=20,
            fixed_inner_iterations=True,
            print_history=False,
        ),
        feasibility_abs_tol=1e-8,
        max_refinements=2,
    )


def _plot_input_panel(ax: plt.Axes, problem: MEBProblem) -> None:
    _draw_balls(ax, problem, facecolor=COLORS["all_ball"], edgecolor="#4d7187", alpha=0.26)
    _style_axes(ax)
    ax.set_title("1. Full 2D Instance", fontsize=14)


def _plot_screening_panel(
    ax: plt.Axes,
    problem: MEBProblem,
    state: torch.Tensor,
    *,
    screening_ratio: float,
    threshold: float,
    kept_set: set[int],
) -> None:
    for idx, (center, radius) in enumerate(zip(problem.centers, problem.radii)):
        if idx in kept_set:
            facecolor = COLORS["kept_ball"]
            edgecolor = "#9a5a0a"
            alpha = 0.38
            zorder = 3
        else:
            facecolor = COLORS["dropped_ball"]
            edgecolor = "#a8b0b7"
            alpha = 0.20
            zorder = 1
        _draw_ball(
            ax,
            center,
            float(radius),
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
        )

    _draw_meb(ax, state.c, float(state.r), color=COLORS["low_meb"], linestyle="-")
    _draw_meb(ax, state.c, threshold, color=COLORS["threshold"], linestyle="--")
    _draw_center(ax, state.c)
    _style_axes(ax)
    ax.set_title("2. Phase I: float32 Screening", fontsize=14)


def _plot_refinement_panel(
    ax: plt.Axes,
    problem: MEBProblem,
    state: torch.Tensor,
    *,
    kept_set: set[int],
    support_set: set[int],
    accepted: bool,
) -> None:
    for idx, (center, radius) in enumerate(zip(problem.centers, problem.radii)):
        if idx in support_set:
            facecolor = COLORS["support_ball"]
            edgecolor = "#8f2d23"
            alpha = 0.46
            zorder = 4
        elif idx in kept_set:
            facecolor = COLORS["kept_ball"]
            edgecolor = "#9a5a0a"
            alpha = 0.25
            zorder = 2
        else:
            continue
        _draw_ball(
            ax,
            center,
            float(radius),
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=zorder,
        )

    _draw_meb(ax, state.c, float(state.r), color=COLORS["high_meb"], linestyle="-")
    _draw_center(ax, state.c)
    _style_axes(ax)
    ax.set_title("3. Phase II: float64 Refinement", fontsize=14)


def _add_legend_row(ax: plt.Axes, *, handles: list[Line2D]) -> None:
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=2.4,
        columnspacing=1.4,
    )


def _draw_balls(
    ax: plt.Axes,
    problem: MEBProblem,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
) -> None:
    for center, radius in zip(problem.centers, problem.radii):
        _draw_ball(
            ax,
            center,
            float(radius),
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=2,
        )


def _draw_ball(
    ax: plt.Axes,
    center: torch.Tensor,
    radius: float,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
) -> None:
    center_np = center.detach().cpu()
    ax.add_patch(
        Circle(
            (float(center_np[0]), float(center_np[1])),
            radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=BALL_EDGEWIDTH,
            alpha=alpha,
            zorder=zorder,
        )
    )
    ax.scatter(
        [float(center_np[0])],
        [float(center_np[1])],
        s=POINT_SIZE,
        color=edgecolor,
        zorder=zorder + 1,
    )


def _draw_meb(
    ax: plt.Axes,
    center: torch.Tensor,
    radius: float,
    *,
    color: str,
    linestyle: str,
) -> None:
    center_np = center.detach().cpu()
    ax.add_patch(
        Circle(
            (float(center_np[0]), float(center_np[1])),
            radius,
            facecolor="none",
            edgecolor=color,
            linewidth=MEB_LINEWIDTH,
            linestyle=linestyle,
            zorder=5,
        )
    )


def _draw_center(ax: plt.Axes, center: torch.Tensor) -> None:
    center_np = center.detach().cpu()
    ax.scatter(
        [float(center_np[0])],
        [float(center_np[1])],
        s=CENTER_SIZE,
        color=COLORS["center"],
        marker="x",
        linewidths=1.8,
        zorder=6,
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.45, 3.45)
    ax.set_ylim(-3.1, 3.25)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.grid(True, linewidth=0.6, alpha=0.25)


def _ball_legend_handle(label: str, facecolor: str, edgecolor: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label=label,
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        markeredgewidth=1.4,
        markersize=10,
        alpha=0.85,
    )


def _line_legend_handle(label: str, color: str, linestyle: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        linestyle=linestyle,
        linewidth=MEB_LINEWIDTH,
        label=label,
    )


def _center_legend_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="x",
        linestyle="None",
        color=COLORS["center"],
        markeredgewidth=1.8,
        markersize=10,
        label="MEB center",
    )


if __name__ == "__main__":
    main()
