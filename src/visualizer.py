"""Visualization and Plot Generation Module.

Generates high-resolution figures for academic documentation and reports:
1. SVR model evaluation (predictions vs actual & 4x4 clash matrix heatmap).
2. Complete QUBO energy spectrum across all 256 configurations.
3. Quantum annealing solution distribution and convergence frequencies.
4. Multiprocessing parallel scaling (Speedup and Efficiency).
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set MPLCONFIGDIR to writable temp directory to eliminate Windows permission warnings
import tempfile
mpl_dir = Path(tempfile.gettempdir()) / "matplotlib_cache"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def plot_svr_evaluation(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    clash_matrix: np.ndarray,
    metrics: Dict[str, Any],
    save_path: Path = RESULTS_DIR / "svr_evaluation.png",
) -> None:
    """Plots SVR test set performance and the 4x4 pairwise clash matrix heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#1f77b4", edgecolor="k", s=35, label="Test Samples (180)")
    min_val = 0
    max_val = max(float(np.max(y_test)), float(np.max(y_pred))) + 5
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal Line (y = x)")
    ax1.set_xlabel("True Common Students (Ground Truth)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("SVR Predicted Common Students", fontsize=11, fontweight="bold")
    ax1.set_title(f"SVR Test Set Performance\nRMSE = {metrics['test_rmse']:.3f} | R² = {metrics['test_r2']:.3f}", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Heatmap of 4x4 Clash Matrix
    im = ax2.imshow(clash_matrix, cmap="YlOrRd", vmin=0, vmax=np.max(clash_matrix) * 1.1)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    exam_labels = ["Exam 1\n(CS Y3 Core)", "Exam 2\n(CS Y3 Elec)", "Exam 3\n(EE Y3 Core)", "Exam 4\n(CS Y4 Core)"]
    ax2.set_xticklabels(exam_labels, fontsize=9)
    ax2.set_yticklabels(exam_labels, fontsize=9)
    ax2.set_title("Predicted Pairwise Clash Matrix C\n(Common Enrolled Students)", fontsize=12, fontweight="bold")

    # Annotate cell numbers
    for i in range(4):
        for j in range(4):
            val = clash_matrix[i, j]
            color = "white" if val > 20 else "black"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted Clash Count", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_qubo_spectrum(
    bf_results: Dict[str, Any],
    save_path: Path = RESULTS_DIR / "qubo_energy_spectrum.png",
) -> None:
    """Plots the full energy spectrum of all 256 QUBO states, highlighting valid vs invalid."""
    states = bf_results["top_all_states"]
    # Re-evaluate all states from brute-force
    from src.qubo_builder import decode_solution

    all_sorted = bf_results.get("all_states_sorted", states)
    # Extract energies and validity
    energies = [s["total_energy"] for s in all_sorted]
    is_valid = [s["is_valid"] for s in all_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(len(energies))

    valid_x = [x for x, v in zip(x_indices, is_valid) if v]
    valid_e = [e for e, v in zip(energies, is_valid) if v]

    invalid_x = [x for x, v in zip(x_indices, is_valid) if not v]
    invalid_e = [e for e, v in zip(energies, is_valid) if not v]

    ax.scatter(invalid_x, invalid_e, color="#d62728", alpha=0.6, s=25, label="Invalid States (Constraint Violated, λ=100)")
    ax.scatter(valid_x, valid_e, color="#2ca02c", alpha=0.9, s=45, edgecolor="k", label="Valid States (Single-Slot Rule Met, 16 Total)")

    # Highlight ground state
    ground_e = bf_results["ground_energy"]
    ax.axhline(ground_e, color="blue", linestyle="--", linewidth=1.2, label=f"Global Ground State (E = {ground_e:.2f})")

    ax.set_xlabel("Sorted State Index (0 to 255)", fontsize=11, fontweight="bold")
    ax.set_ylabel("QUBO Energy E(x)", fontsize=11, fontweight="bold")
    ax.set_title("QUBO Energy Spectrum: Penalty Barrier & Valid State Space (Part b)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_quantum_distribution(
    q_summary: Dict[str, Any],
    save_path: Path = RESULTS_DIR / "annealing_solution_distribution.png",
) -> None:
    """Plots the frequency of top sampled solutions from the Quantum Annealer."""
    fig, ax = plt.subplots(figsize=(10, 5))

    top_sols = q_summary["top_5_solutions"]
    labels = [f"{s['bitstring']}\n(E={s['energy']:.2f})" for s in top_sols]
    counts = [s["count"] for s in top_sols]
    colors = ["#2ca02c" if s["is_valid"] else "#d62728" for s in top_sols]

    bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.55)

    for bar, s in zip(bars, top_sols):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval} ({s['frequency_pct']}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Candidate State Bitstring & Energy", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Appearance Count (out of {q_summary['num_reads']} reads)", fontsize=11, fontweight="bold")
    ax.set_title(f"Simulated Quantum Annealing Solution Distribution (Part c)\nGround State Twins Dominate Top Reads", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_parallel_scaling(
    scaling_table: List[Dict[str, Any]],
    save_path: Path = RESULTS_DIR / "parallel_speedup_efficiency.png",
) -> None:
    """Plots Speed-up and Efficiency curves for multiprocessing scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    workers = [r["workers"] for r in scaling_table]
    speedups = [r["speedup"] for r in scaling_table]
    efficiencies = [r["efficiency_pct"] for r in scaling_table]
    times = [r["time_seconds"] for r in scaling_table]

    # Subplot 1: Speedup
    ax1.plot(workers, speedups, "o-", color="#1f77b4", linewidth=2.0, markersize=7, label="Observed Speed-up (S_p)")
    ax1.plot(workers, workers, "k--", linewidth=1.5, label="Ideal Linear Speed-up (S_p = p)")
    ax1.set_xlabel("Number of Multiprocessing Workers (p)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speed-up S_p = T_1 / T_p", fontsize=11, fontweight="bold")
    ax1.set_title("Parallel Speed-up Scaling (Part d)", fontsize=12, fontweight="bold")
    ax1.set_xticks(workers)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for w, s, t in zip(workers, speedups, times):
        ax1.annotate(f"{s:.2f}x\n({t:.2f}s)", (w, s), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    # Subplot 2: Efficiency
    ax2.bar([str(w) for w in workers], efficiencies, color="#ff7f0e", edgecolor="k", width=0.45)
    ax2.axhline(100.0, color="k", linestyle="--", linewidth=1.2, label="100% Ideal Efficiency")
    ax2.set_xlabel("Number of Multiprocessing Workers (p)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Efficiency E_p (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Parallel Efficiency Scaling (Part d)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    for idx, (w, eff) in enumerate(zip(workers, efficiencies)):
        ax2.text(idx, eff + 2.0, f"{eff:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
