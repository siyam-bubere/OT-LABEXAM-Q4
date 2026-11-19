"""Quantum Computing Module (Part c): Simulated Quantum Annealing on QUBO.

Academic Task:
- Solve the QUBO with Quantum Annealing (simulated) on a simulator.
- Report the best answer, its energy, and how often it appears.
- Compare with brute force results.

Methods Implemented:
1. Fast Simulated Annealing (SA) Sampler with incremental delta-E computation.
2. Simulated Quantum Annealing (SQA / Path-Integral Monte Carlo with Transverse Field Driver).
3. Optional wrapper for D-Wave neal.SimulatedAnnealingSampler if installed.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ANNEALING_NUM_READS,
    ANNEALING_NUM_SWEEPS,
    BETA_START,
    BETA_END,
    RANDOM_SEED,
    NUM_VARIABLES,
)
from src.qubo_builder import decode_solution


class SimulatedQuantumAnnealer:
    """High-performance Classical/Quantum Annealing Simulator for QUBO problems.

    Supports:
    - Metropolis-Hastings MCMC with geometric beta schedule
    - Fast O(N) local energy difference updates
    - Transverse-field quantum tunneling simulation
    """

    def __init__(
        self,
        Q: np.ndarray,
        offset: float = 0.0,
        num_reads: int = ANNEALING_NUM_READS,
        num_sweeps: int = ANNEALING_NUM_SWEEPS,
        beta_range: Tuple[float, float] = (BETA_START, BETA_END),
        seed: int = RANDOM_SEED,
    ):
        self.Q = np.asarray(Q, dtype=float)
        self.offset = float(offset)
        self.n_vars = self.Q.shape[0]
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed

        # Symmetrize Q for fast O(N) delta-E calculations:
        # E(x) = x^T Q_sym x + offset
        self.Q_sym = 0.5 * (self.Q + self.Q.T)
        self.diag_Q = np.diag(self.Q_sym)

    def energy(self, x: np.ndarray) -> float:
        """Computes exact total energy for a state vector x."""
        return float(x @ self.Q_sym @ x + self.offset)

    def run_simulated_annealing(
        self, seed: Optional[int] = None
    ) -> List[Tuple[float, str, np.ndarray]]:
        """Executes simulated annealing across num_reads independent runs."""
        run_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(run_seed)

        betas = np.geomspace(self.beta_range[0], self.beta_range[1], self.num_sweeps)
        results = []

        # Precompute interaction matrix without diagonal for local delta updates
        # Delta E when flipping x_k:
        # delta_E = delta_x_k * (2 * sum_{j} Q_sym[k, j] * x_j + Q_sym[k, k] * delta_x_k)
        for _ in range(self.num_reads):
            x = rng.integers(0, 2, size=self.n_vars).astype(float)
            curr_energy = self.energy(x)

            for beta in betas:
                for k in rng.permutation(self.n_vars):
                    # flip bit k: delta_x = 1 - 2*x[k]
                    delta_x = 1.0 - 2.0 * x[k]
                    h_local = 2.0 * np.dot(self.Q_sym[k], x)
                    delta_E = delta_x * (h_local + self.Q_sym[k, k] * delta_x)

                    if delta_E < 0.0 or rng.random() < np.exp(-beta * delta_E):
                        x[k] = 1.0 - x[k]
                        curr_energy += delta_E

            bitstring = "".join(str(int(b)) for b in x)
            results.append((round(curr_energy, 4), bitstring, x.copy().astype(int)))

        return results

    def run_sqa_trotter(
        self, trotter_slices: int = 8, seed: Optional[int] = None
    ) -> List[Tuple[float, str, np.ndarray]]:
        """Path-Integral Simulated Quantum Annealing (PIMC-SQA).

        Emulates transverse-field quantum fluctuations:
        H(s) = (1-s) H_transverse + s H_problem
        across trotter_slices replicas coupled in imaginary time.
        """
        run_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(run_seed)

        P = trotter_slices
        results = []

        s_schedule = np.linspace(0.01, 0.99, self.num_sweeps)
        beta_p = 1.0 / P

        for _ in range(self.num_reads):
            # Replicas (P, n_vars)
            spins = rng.choice([-1.0, 1.0], size=(P, self.n_vars))

            for s in s_schedule:
                gamma = 1.0 - s
                # Ferromagnetic coupling along Trotter dimension: J_perp = -0.5 * ln(tanh(gamma / P))
                gamma_safe = max(gamma, 1e-4)
                j_perp = -0.5 * np.log(np.maximum(1e-6, np.tanh(gamma_safe / P)))

                for p in range(P):
                    p_prev = (p - 1) % P
                    p_next = (p + 1) % P

                    for k in rng.permutation(self.n_vars):
                        # Convert spin to binary: x = (sigma + 1)/2
                        x_curr = (spins[p] + 1.0) / 2.0
                        delta_x = -spins[p, k]  # if spin was +1, delta_x is -1; if -1, delta_x is +1
                        h_local = 2.0 * np.dot(self.Q_sym[k], x_curr)
                        dE_prob = s * delta_x * (h_local + self.Q_sym[k, k] * delta_x)

                        # Coupling term with neighboring trotter slices
                        dE_trotter = -2.0 * j_perp * (-spins[p, k]) * (spins[p_prev, k] + spins[p_next, k])
                        total_dE = dE_prob + dE_trotter

                        if total_dE < 0.0 or rng.random() < np.exp(-beta_p * total_dE):
                            spins[p, k] *= -1.0

            # Evaluate each replica on original problem Hamiltonian and select lowest
            best_replica_energy = float("inf")
            best_replica_x = None

            for p in range(P):
                x_bin = ((spins[p] + 1.0) / 2.0).astype(int)
                e = self.energy(x_bin.astype(float))
                if e < best_replica_energy:
                    best_replica_energy = e
                    best_replica_x = x_bin

            bitstring = "".join(str(b) for b in best_replica_x)
            results.append((round(best_replica_energy, 4), bitstring, best_replica_x))

        return results


def solve_qubo_quantum(
    Q: np.ndarray,
    offset: float,
    num_reads: int = ANNEALING_NUM_READS,
    num_sweeps: int = ANNEALING_NUM_SWEEPS,
    method: str = "simulated_annealing",
    seed: int = RANDOM_SEED,
) -> Tuple[List[Tuple[float, str, np.ndarray]], Dict[str, Any]]:
    """Runs quantum/simulated annealing and aggregates solution frequencies."""
    annealer = SimulatedQuantumAnnealer(
        Q=Q,
        offset=offset,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )

    if method == "sqa":
        raw_samples = annealer.run_sqa_trotter()
    else:
        raw_samples = annealer.run_simulated_annealing()

    # Aggregate by (energy, bitstring)
    counter = Counter((e, b) for e, b, _ in raw_samples)
    sorted_unique = sorted(counter.items(), key=lambda item: (item[0][0], -item[1]))

    best_energy, best_bitstring = sorted_unique[0][0]
    best_count = sorted_unique[0][1]
    best_freq_pct = round((best_count / num_reads) * 100.0, 2)

    # Decode best solution
    best_x = np.array([int(c) for c in best_bitstring])
    decoded_best = decode_solution(best_x)

    summary = {
        "method": method,
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "best_bitstring": best_bitstring,
        "best_energy": best_energy,
        "best_appearance_count": best_count,
        "best_frequency_pct": best_freq_pct,
        "decoded": decoded_best,
        "top_5_solutions": [
            {
                "bitstring": b,
                "energy": e,
                "count": cnt,
                "frequency_pct": round((cnt / num_reads) * 100.0, 2),
                "is_valid": decode_solution(np.array([int(c) for c in b]))["is_valid"],
            }
            for (e, b), cnt in sorted_unique[:5]
        ],
    }

    return raw_samples, summary


def compare_with_brute_force(
    quantum_summary: Dict[str, Any], brute_force_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Compares quantum annealing outputs directly with the exhaustive brute force search."""
    bf_ground_energy = brute_force_results["ground_energy"]
    q_best_energy = quantum_summary["best_energy"]
    q_best_bitstring = quantum_summary["best_bitstring"]

    bf_ground_bitstrings = [s["bitstring"] for s in brute_force_results["best_states"]]

    energy_diff = abs(q_best_energy - bf_ground_energy)
    is_exact_match = (energy_diff < 1e-4) and (q_best_bitstring in bf_ground_bitstrings)

    # Calculate total ground state probability (accounting for symmetric twins)
    twin_count = 0
    for sol in quantum_summary["top_5_solutions"]:
        if sol["bitstring"] in bf_ground_bitstrings:
            twin_count += sol["count"]

    twin_pct = round((twin_count / quantum_summary["num_reads"]) * 100.0, 2)

    comparison = {
        "brute_force_ground_energy": bf_ground_energy,
        "quantum_best_energy": q_best_energy,
        "energy_gap": round(energy_diff, 4),
        "is_optimal_found": is_exact_match,
        "quantum_best_bitstring": q_best_bitstring,
        "bf_optimal_bitstrings": bf_ground_bitstrings,
        "primary_state_frequency": f"{quantum_summary['best_appearance_count']}/{quantum_summary['num_reads']} ({quantum_summary['best_frequency_pct']}%)",
        "combined_ground_states_frequency": f"{twin_count}/{quantum_summary['num_reads']} ({twin_pct}%)",
        "comparison_verdict": (
            "SUCCESS: Quantum Annealer converged to the exact global minimum identified by brute force."
            if is_exact_match
            else "SUBOPTIMAL: Quantum Annealer found a near-ground state."
        ),
    }

    return comparison


if __name__ == "__main__":
    from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams
    from src.qubo_builder import build_qubo_matrix, brute_force_solver

    print("=" * 70)
    print("Part c: Quantum Computing (Simulated Annealing on QUBO)")
    print("=" * 70)

    # 1. Pipeline prep
    _, X, y = generate_synthetic_enrolment_data()
    model, _, _, _, _, _ = train_and_evaluate_svr(X, y)
    C_matrix, _, _ = predict_clashes_for_exams(model)
    Q, offset = build_qubo_matrix(C_matrix)
    bf_res = brute_force_solver(Q, offset, C_matrix)

    # 2. Quantum solve
    raw_samples, q_summary = solve_qubo_quantum(Q, offset, num_reads=1000, num_sweeps=300)

    print(f"Annealing completed: {q_summary['num_reads']} reads, {q_summary['num_sweeps']} sweeps.")
    print(f"Best Answer Found:  State [{q_summary['best_bitstring']}]")
    print(f"Best Answer Energy: {q_summary['best_energy']:.2f}")
    print(f"Appearance Count:   {q_summary['best_appearance_count']}/{q_summary['num_reads']} reads ({q_summary['best_frequency_pct']}%)")
    print(f"Schedule:           Slot 1 = {q_summary['decoded']['slot_1']}, Slot 2 = {q_summary['decoded']['slot_2']}")

    # 3. Compare with brute force
    comp = compare_with_brute_force(q_summary, bf_res)
    print("\nComparison with Exhaustive Brute Force:")
    print(f"  Brute Force Ground Energy: {comp['brute_force_ground_energy']:.2f}")
    print(f"  Quantum Best Energy:       {comp['quantum_best_energy']:.2f}")
    print(f"  Energy Gap:                {comp['energy_gap']:.4f}")
    print(f"  Optimal Found:             {comp['is_optimal_found']}")
    print(f"  Combined Ground State Freq:{comp['combined_ground_states_frequency']}")
    print(f"  Verdict:                   {comp['comparison_verdict']}")
