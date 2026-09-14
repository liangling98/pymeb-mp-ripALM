"""Solve and plot the paper's ten-ball, two-dimensional toy instance.

Install plotting support with ``python -m pip install -e ".[plot]"``.
Run ``python examples/quickstart.py`` to display the result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

from pymeb import solve


def main() -> None:
    # Same input balls as experiments/plot_two_phase_example.py in the paper.
    centers = torch.tensor(
        [
            [-2.6, -0.4], [-1.8, 1.9], [0.4, 2.6], [2.7, 0.7],
            [1.8, -2.1], [-0.8, -2.4], [0.1, 0.2], [0.8, 0.7],
            [-0.7, 1.0], [1.2, -0.4],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor(
        [0.25, 0.18, 0.22, 0.20, 0.24, 0.17, 0.12, 0.10, 0.11, 0.09],
        dtype=torch.float64,
    )
    result = solve(centers, radii)

    print("Paper toy example: 10 balls in 2D")
    print(f"  center: {result.center.tolist()}")
    print(f"  radius: {float(result.radius):.10f}")
    print(f"  maximum violation: {float(result.max_violation):.3e}")
    print(f"  KKT residual: {float(result.kkt_residual):.3e}")
    print(f"  converged: {result.converged}")
    enclosed = bool(result.contains(centers, radii).all())
    print(f"  encloses every input: {enclosed}")
    if not result.converged or not enclosed:
        raise RuntimeError("The toy solve did not pass convergence and enclosure checks.")

    # Highlight balls touching the enclosing circle, using the paper plot's tolerance.
    slack = result.radius - radii - torch.linalg.vector_norm(centers - result.center, dim=1)
    active = slack.abs() <= 5e-6
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    labeled = set()
    for center, radius, touches in zip(centers.tolist(), radii.tolist(), active.tolist()):
        color = "#c44536" if touches else "#7aa6c2"
        label = "Boundary balls" if touches else "Interior balls"
        ax.add_patch(Circle(
            center, radius, facecolor=color, edgecolor=color, alpha=0.45,
            linewidth=1.5, label=label if label not in labeled else None,
        ))
        ax.plot(*center, ".", color=color, markersize=5)
        labeled.add(label)

    center = result.center.tolist()
    radius = float(result.radius)
    ax.add_patch(Circle(
        center, radius, fill=False, edgecolor="#2a9d8f", linewidth=2.5,
        label="Minimum enclosing ball",
    ))
    ax.plot(*center, "x", color="#1f2933", markersize=9, label="MEB center")
    margin = 1.15 * radius
    ax.set(xlim=(center[0] - margin, center[0] + margin),
           ylim=(center[1] - margin, center[1] + margin),
           xlabel="$x_1$", ylabel="$x_2$",
           title=f"Paper toy example: minimum enclosing ball\nRadius = {radius:.6f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    plt.show()


if __name__ == "__main__":
    main()
