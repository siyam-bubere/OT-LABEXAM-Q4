"""Master Pipeline Runner for Question 4: ML + Quantum + Parallel Computing.

This script executes all subparts sequentially:
  [a] Machine Learning (SVR clash prediction, 600 synthetic rows, seed=012, 70/30 split)
  [b] QUBO Formulation (8x8 Q matrix, penalty method, brute force check of all 256 states)
  [c] Quantum Computing (Simulated Quantum Annealing, solution frequency, brute force comparison)
  [d] Parallel Computing (Independent restarts across 1, 2, and 4 workers, speedup, efficiency)
  [e] Academic Conclusion (Strict 3-line synthesis of final choices and learnings)

Also saves:
  - High-resolution plots in results/
  - Structured JSON metrics summary in results/metrics_summary.json
"""

import sys
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RANDOM_SEED,
    QUBO_PENALTY_LAMBDA,
    ANNEALING_NUM_READS,
    ANNEALING_NUM_SWEEPS,
    PARALLEL_WORKER_COUNTS,
    RESULTS_DIR,
)
from src.ml_svr import (
    generate_synthetic_enrolment_data,
    train_and_evaluate_svr,
    predict_clashes_for_exams,
)
from src.qubo_builder import (
    build_qubo_matrix,
    brute_force_solver,
)
from src.quantum_solver import (
    solve_qubo_quantum,
    compare_with_brute_force,
)
from src.parallel_runner import run_parallel_restarts
from src.conclusion import generate_three_line_conclusion
from src.visualizer import (
    plot_svr_evaluation,
    plot_qubo_spectrum,
    plot_quantum_distribution,
    plot_parallel_scaling,
)


def run_full_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """Runs end-to-end execution of Question 4 workflow."""
    if verbose:
        print("=" * 78)
        print("EXAM SCHEDULING OPTIMIZATION: MACHINE LEARNING + QUANTUM + PARALLEL")
        print("Course Outcome Coverage: CO1 (ML), CO2 (Quantum/QUBO), CO4 (Parallel HPC)")
        print("=" * 78)

    # -------------------------------------------------------------------------
    # PART A: Machine Learning (SVR) [01 mark, CO1]
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> PART A: Machine Learning (Support Vector Regression) [CO1]")
        print("-" * 78)

    df_synthetic, X, y = generate_synthetic_enrolment_data(seed=RANDOM_SEED)
    model, svr_metrics, X_train, X_test, y_test, y_pred = train_and_evaluate_svr(X, y, seed=RANDOM_SEED)
    clash_matrix, pair_preds, exam_summary = predict_clashes_for_exams(model)

    if verbose:
        print(f"Synthetic Dataset: Generated {len(df_synthetic)} rows with random seed = {RANDOM_SEED:03d}")
        print(f"Train/Test Partition: {svr_metrics['n_train']} train rows / {svr_metrics['n_test']} test rows (70/30 split)")
        print(f"SVR Model Performance: Test RMSE = {svr_metrics['test_rmse']:.4f} | Test R² = {svr_metrics['test_r2']:.4f}")
        print("\nSVR Clash Predictions for the 4 Exams (Pairwise Common Students):")
        for (e1, e2), count in pair_preds.items():
            print(f"  • Exam {e1} vs Exam {e2}: {count:.2f} clashing students")

        print("\n4x4 Symmetric Predicted Clash Matrix C:")
        print(clash_matrix)

        print("\nConflict Load Breakdown per Exam:")
        for eid, info in exam_summary.items():
            print(f"  • {info['name']} ({info['code']}, {info['dept']} Year {info['year']}, {info['type']}, Size={info['class_size']}): "
                  f"Total Potential Clash = {info['total_potential_clashes']:.2f}")

    # -------------------------------------------------------------------------
    # PART B: QUBO Formulation & Brute Force [01 mark, CO2]
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> PART B: QUBO Formulation & Brute Force Verification [CO2]")
        print("-" * 78)

    Q, offset = build_qubo_matrix(clash_matrix, lambda_penalty=QUBO_PENALTY_LAMBDA)
    bf_results = brute_force_solver(Q, offset, clash_matrix, lambda_penalty=QUBO_PENALTY_LAMBDA)

    if verbose:
        print(f"Decision Vector: 8 binary variables x = [x_11, x_12, x_21, x_22, x_31, x_32, x_41, x_42]^T")
        print(f"Constraint Penalty Parameter: λ = {QUBO_PENALTY_LAMBDA:.1f} (enforces x_i1 + x_i2 = 1)")
        print(f"Energy Function: E(x) = x^T Q x + {offset:.1f}")
        print(f"\nConstructed 8x8 Q Matrix:")
        print(Q)
        print(f"\nBrute Force Search Results:")
        print(f"  • Total configurations evaluated: {bf_results['total_states_evaluated']} (2^8)")
        print(f"  • Valid configurations meeting single-slot rule: {bf_results['total_valid_states']} (2^4)")
        print(f"  • Ground State Energy: {bf_results['ground_energy']:.2f}")
        print(f"  • Is Ground State Valid: {bf_results['is_ground_state_valid']}")

        print("\n  Top Optimal Solutions (Symmetric Ground States):")
        for s in bf_results["best_states"]:
            print(f"    - State [{s['bitstring']}]: Clash Energy = {s['total_energy']:.2f} | Slot 1: {s['slot_1']} | Slot 2: {s['slot_2']}")

    # -------------------------------------------------------------------------
    # PART C: Quantum Computing (Simulated Annealing) [1.5 marks, CO2]
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> PART C: Quantum Computing (Simulated Annealing on Simulator) [CO2]")
        print("-" * 78)

    raw_samples, q_summary = solve_qubo_quantum(
        Q=Q,
        offset=offset,
        num_reads=ANNEALING_NUM_READS,
        num_sweeps=ANNEALING_NUM_SWEEPS,
        seed=RANDOM_SEED,
    )
    comparison = compare_with_brute_force(q_summary, bf_results)

    if verbose:
        print(f"Quantum Annealing Execution: {q_summary['num_reads']} reads, {q_summary['num_sweeps']} sweeps.")
        print(f"Best Answer Found:   State [{q_summary['best_bitstring']}]")
        print(f"Best Answer Energy:  {q_summary['best_energy']:.2f}")
        print(f"Appearance Count:    {q_summary['best_appearance_count']}/{q_summary['num_reads']} reads ({q_summary['best_frequency_pct']}%)")
        print(f"Assigned Schedule:   Slot 1 = {q_summary['decoded']['slot_1']}, Slot 2 = {q_summary['decoded']['slot_2']}")
        print("\nComparison with Exhaustive Brute Force:")
        print(f"  • Brute Force Ground Energy:   {comparison['brute_force_ground_energy']:.2f}")
        print(f"  • Quantum Annealer Energy:     {comparison['quantum_best_energy']:.2f}")
        print(f"  • Energy Residual Gap:         {comparison['energy_gap']:.4f}")
        print(f"  • Exact Optimal State Found:   {comparison['is_optimal_found']}")
        print(f"  • Combined Ground State Freq:  {comparison['combined_ground_states_frequency']}")
        print(f"  • Verdict:                     {comparison['comparison_verdict']}")

    # -------------------------------------------------------------------------
    # PART D: Parallel Computing (multiprocessing.Pool) [01 mark, CO4]
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> PART D: Parallel Computing (Independent Annealing Restarts) [CO4]")
        print("-" * 78)

    parallel_results = run_parallel_restarts(
        Q=Q,
        offset=offset,
        worker_counts=PARALLEL_WORKER_COUNTS,
        num_restarts=16,
        reads_per_restart=40,
        sweeps_per_restart=200,
    )

    if verbose:
        print(f"Restart Portfolio: 16 independent random seeds executed via multiprocessing.Pool.")
        print(f"Minimum Energy Found Across All Restarts: {parallel_results['min_energy_found']:.2f}")
        print("\nParallel Scaling Benchmark Table:")
        print("| Workers (p) | Execution Time (s) | Speed-up (S_p) | Efficiency (E_p) |")
        print("|:-----------:|:------------------:|:--------------:|:----------------:|")
        for row in parallel_results["scaling_table"]:
            print(f"| {row['workers']:11d} | {row['time_seconds']:18.4f} | {row['speedup']:14.2f} | {row['efficiency_pct']:15.1f}% |")

    # -------------------------------------------------------------------------
    # PART E: Conclusion in 3 Lines [0.5 mark, CO4]
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> PART E: Academic Conclusion in 3 Lines [CO4]")
        print("-" * 78)

    best_s1 = q_summary["decoded"]["slot_1"]
    best_s2 = q_summary["decoded"]["slot_2"]
    max_sp = parallel_results["scaling_table"][-1]["speedup"]
    three_line_conclusion = generate_three_line_conclusion(
        best_slot_1=best_s1,
        best_slot_2=best_s2,
        min_clash=q_summary["best_energy"],
        max_speedup=max_sp,
        workers=4,
    )

    if verbose:
        print(three_line_conclusion)

    # -------------------------------------------------------------------------
    # Generate Visualizations & Persist Metrics
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-" * 78)
        print(">>> Generating Publication-Quality Figures & Saving Artifacts...")
        print("-" * 78)

    plot_svr_evaluation(y_test, y_pred, clash_matrix, svr_metrics)
    plot_qubo_spectrum(bf_results)
    plot_quantum_distribution(q_summary)
    plot_parallel_scaling(parallel_results["scaling_table"])

    all_metrics = {
        "part_a_ml": {
            "random_seed": RANDOM_SEED,
            "synthetic_rows": len(df_synthetic),
            "train_rows": svr_metrics["n_train"],
            "test_rows": svr_metrics["n_test"],
            "test_rmse": svr_metrics["test_rmse"],
            "test_r2": svr_metrics["test_r2"],
            "pairwise_clash_predictions": {f"{k[0]}_{k[1]}": v for k, v in pair_preds.items()},
            "clash_matrix": clash_matrix.tolist(),
        },
        "part_b_qubo": {
            "num_variables": 8,
            "lambda_penalty": QUBO_PENALTY_LAMBDA,
            "constant_offset": offset,
            "brute_force_ground_energy": bf_results["ground_energy"],
            "valid_states_count": bf_results["total_valid_states"],
            "is_ground_state_valid": bf_results["is_ground_state_valid"],
            "optimal_states": [
                {"bitstring": s["bitstring"], "energy": s["total_energy"], "slot_1": s["slot_1"], "slot_2": s["slot_2"]}
                for s in bf_results["best_states"]
            ],
        },
        "part_c_quantum": {
            "simulator": "simulated_annealing",
            "num_reads": q_summary["num_reads"],
            "num_sweeps": q_summary["num_sweeps"],
            "best_bitstring": q_summary["best_bitstring"],
            "best_energy": q_summary["best_energy"],
            "appearance_count": q_summary["best_appearance_count"],
            "frequency_pct": q_summary["best_frequency_pct"],
            "ground_state_exact_match": comparison["is_optimal_found"],
            "energy_gap": comparison["energy_gap"],
        },
        "part_d_parallel": {
            "scaling_benchmarks": parallel_results["scaling_table"],
            "num_restarts": parallel_results["num_restarts"],
        },
        "part_e_conclusion": three_line_conclusion.split("\n"),
    }

    metrics_file = RESULTS_DIR / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    if verbose:
        print(f"Metrics saved to: {metrics_file}")
        print(f"Plots saved to:   {RESULTS_DIR}")
        print("=" * 78)
        print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        print("=" * 78)

    return all_metrics


if __name__ == "__main__":
    mp.freeze_support()
    run_full_pipeline(verbose=True)
