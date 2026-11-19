"""Parallel Computing Module (Part d): Parallel Independent Annealing Restarts.

Academic Task:
- Run independent QAOA/VQE/annealing restarts (different random seeds) in parallel using multiprocessing.Pool.
- Report the time with 1, 2, and 4 workers.
- Report the speed-up: S_p = T_1 / T_p.
- Report the efficiency: E_p = S_p / p = T_1 / (p * T_p) (or as percentage E_p * 100%).

Engineering Considerations:
- Fully compatible with Windows multiprocessing (spawn model, top-level picklable worker).
- Measures wall-clock execution time accurately with time.perf_counter().
- Collects best energies and verified states across all independent seeds.
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PARALLEL_WORKER_COUNTS,
    PARALLEL_NUM_RESTARTS,
)
from src.qubo_builder import decode_solution


def _annealing_restart_worker(args: Tuple[int, np.ndarray, float, int, int]) -> Dict[str, Any]:
    """Top-level worker function executed across multiprocessing pool processes.

    args: (seed, Q, offset, num_reads, num_sweeps)
    """
    seed, Q, offset, num_reads, num_sweeps = args

    # Fast localized simulated annealing restart
    rng = np.random.default_rng(seed)
    n_vars = Q.shape[0]
    Q_sym = 0.5 * (Q + Q.T)
    betas = np.geomspace(0.1, 10.0, num_sweeps)

    best_energy = float("inf")
    best_x = None

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n_vars).astype(float)
        curr_energy = float(x @ Q_sym @ x + offset)

        for beta in betas:
            for k in rng.permutation(n_vars):
                delta_x = 1.0 - 2.0 * x[k]
                h_local = 2.0 * np.dot(Q_sym[k], x)
                delta_E = delta_x * (h_local + Q_sym[k, k] * delta_x)

                if delta_E < 0.0 or rng.random() < np.exp(-beta * delta_E):
                    x[k] = 1.0 - x[k]
                    curr_energy += delta_E

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_x = x.copy().astype(int)

    bitstring = "".join(str(b) for b in best_x)
    decoded = decode_solution(best_x)

    return {
        "seed": seed,
        "best_energy": round(best_energy, 4),
        "best_bitstring": bitstring,
        "is_valid": decoded["is_valid"],
        "slot_1": decoded["slot_1"],
        "slot_2": decoded["slot_2"],
    }


def run_parallel_restarts(
    Q: np.ndarray,
    offset: float,
    worker_counts: List[int] = PARALLEL_WORKER_COUNTS,
    num_restarts: int = 16,
    reads_per_restart: int = 40,
    sweeps_per_restart: int = 200,
) -> Dict[str, Any]:
    """Executes parallel independent restarts with 1, 2, and 4 workers.

    Calculates execution wall-clock time, speed-up, and efficiency.
    """
    # Deterministic distinct seeds for independent restarts
    seeds = [100 + i * 37 for i in range(num_restarts)]
    tasks = [(s, Q, offset, reads_per_restart, sweeps_per_restart) for s in seeds]

    timings = {}
    restart_outputs = {}

    for workers in worker_counts:
        start_time = time.perf_counter()

        with mp.Pool(processes=workers) as pool:
            results = pool.map(_annealing_restart_worker, tasks)

        elapsed_time = time.perf_counter() - start_time
        timings[workers] = round(elapsed_time, 4)
        restart_outputs[workers] = results

    # Compute Speed-up and Efficiency relative to single worker (T_1)
    t1 = timings[1]
    scaling_records = []

    for w in worker_counts:
        t_w = timings[w]
        speedup = round(t1 / t_w, 4)
        efficiency_pct = round((speedup / w) * 100.0, 2)

        scaling_records.append(
            {
                "workers": w,
                "time_seconds": t_w,
                "speedup": speedup,
                "efficiency_pct": efficiency_pct,
            }
        )

    # Collect ground-state discovery across restarts
    all_energies = [r["best_energy"] for r in restart_outputs[worker_counts[-1]]]
    min_energy_found = min(all_energies)

    return {
        "num_restarts": num_restarts,
        "reads_per_restart": reads_per_restart,
        "sweeps_per_restart": sweeps_per_restart,
        "timings": timings,
        "scaling_table": scaling_records,
        "min_energy_found": min_energy_found,
        "restart_details": restart_outputs[worker_counts[-1]],
    }


if __name__ == "__main__":
    from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams
    from src.qubo_builder import build_qubo_matrix

    mp.freeze_support()

    print("=" * 70)
    print("Part d: Parallel Computing (Independent Annealing Restarts)")
    print("=" * 70)

    # Pipeline prep
    _, X, y = generate_synthetic_enrolment_data()
    model, _, _, _, _, _ = train_and_evaluate_svr(X, y)
    C_matrix, _, _ = predict_clashes_for_exams(model)
    Q, offset = build_qubo_matrix(C_matrix)

    parallel_results = run_parallel_restarts(
        Q=Q,
        offset=offset,
        worker_counts=[1, 2, 4],
        num_restarts=16,
        reads_per_restart=40,
        sweeps_per_restart=200,
    )

    print(f"Executed {parallel_results['num_restarts']} Independent Annealing Restarts across Pool.")
    print(f"Minimum Energy Found across Restarts: {parallel_results['min_energy_found']:.2f}\n")

    print("Parallel Scaling Benchmarks:")
    print("| Workers (p) | Execution Time (s) | Speed-up (S_p) | Efficiency (E_p) |")
    print("|-------------|--------------------|----------------|------------------|")
    for row in parallel_results["scaling_table"]:
        print(f"| {row['workers']:11d} | {row['time_seconds']:18.4f} | {row['speedup']:14.2f} | {row['efficiency_pct']:15.1f}% |")
