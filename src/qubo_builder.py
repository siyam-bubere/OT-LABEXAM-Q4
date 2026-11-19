"""QUBO Formulation & Brute-Force Verification Module (Part b).

Academic Task:
- Turn the decision into a QUBO (energy function): write it down.
- Build its Q matrix from SVR predictions.
- Add a penalty for breaking the rule (each exam must be in exactly one time slot).
- Check by brute force that the lowest-energy answer is correct.

Mathematical Formulation:
Variables:
    x_{i, s} in {0, 1} for exam i in {1..4} and slot s in {1..2} (8 binary variables in total).
    Vector x = [x_11, x_12, x_21, x_22, x_31, x_32, x_41, x_42]^T

Constraint (Single-Slot Rule):
    For each exam i: x_i1 + x_i2 = 1
    Penalty term: P_i = lambda * (x_i1 + x_i2 - 1)^2
    Since x_{i,s}^2 = x_{i,s}:
        (x_i1 + x_i2 - 1)^2 = -x_i1 - x_i2 + 2 * x_i1 * x_i2 + 1
    Total Penalty:
        P(x) = lambda * sum_{i=1}^4 (-x_i1 - x_i2 + 2 * x_i1 * x_i2) + 4 * lambda

Objective (Minimize Clash Counts):
    If exam i and exam j sit in the same slot s:
        H_clash = sum_{s=1}^2 sum_{1 <= i < j <= 4} C_{ij} * x_{i, s} * x_{j, s}

Total QUBO Energy Function:
    E(x) = H_clash(x) + P(x)
         = x^T Q x + 4 * lambda
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    NUM_EXAMS,
    NUM_SLOTS,
    NUM_VARIABLES,
    QUBO_PENALTY_LAMBDA,
    EXAMS,
)


def var_index(exam_idx: int, slot_idx: int) -> int:
    """Maps (exam 0..3, slot 0..1) to binary vector index 0..7."""
    return 2 * exam_idx + slot_idx


def build_qubo_matrix(
    clash_matrix: np.ndarray,
    lambda_penalty: float = QUBO_PENALTY_LAMBDA,
    num_exams: int = NUM_EXAMS,
    num_slots: int = NUM_SLOTS,
    symmetric: bool = False,
) -> Tuple[np.ndarray, float]:
    """Constructs the Q matrix (upper-triangular or symmetric) and energy offset.

    Returns:
    - Q: (8, 8) numpy array
    - offset: constant scalar added to x^T Q x
    """
    n_vars = num_exams * num_slots
    Q = np.zeros((n_vars, n_vars), dtype=float)

    # 1. Constraint linear terms: -lambda * x_{i, s}
    for i in range(num_exams):
        for s in range(num_slots):
            k = var_index(i, s)
            Q[k, k] += -lambda_penalty

    # 2. Constraint quadratic terms: + 2 * lambda * x_{i, 1} * x_{i, 2}
    for i in range(num_exams):
        k1 = var_index(i, 0)
        k2 = var_index(i, 1)
        if symmetric:
            Q[k1, k2] += lambda_penalty
            Q[k2, k1] += lambda_penalty
        else:
            Q[k1, k2] += 2.0 * lambda_penalty

    # 3. Objective quadratic terms: + C_{ij} * x_{i, s} * x_{j, s}
    for i in range(num_exams):
        for j in range(i + 1, num_exams):
            c_val = float(clash_matrix[i, j])
            for s in range(num_slots):
                k_i = var_index(i, s)
                k_j = var_index(j, s)
                if symmetric:
                    Q[k_i, k_j] += c_val / 2.0
                    Q[k_j, k_i] += c_val / 2.0
                else:
                    Q[k_i, k_j] += c_val

    offset = float(num_exams * lambda_penalty)
    return Q, offset


def decode_solution(x: np.ndarray) -> Dict[str, Any]:
    """Interprets an 8-bit binary vector into exam-to-slot schedule."""
    x = np.asarray(x, dtype=int)
    slot_assignments = {1: [], 2: []}
    exam_slots = {}
    is_valid = True
    penalty_violations = 0

    for i in range(NUM_EXAMS):
        s1 = int(x[var_index(i, 0)])
        s2 = int(x[var_index(i, 1)])
        assigned = []
        if s1 == 1:
            assigned.append(1)
            slot_assignments[1].append(f"Exam {i+1}")
        if s2 == 1:
            assigned.append(2)
            slot_assignments[2].append(f"Exam {i+1}")

        exam_slots[f"Exam {i+1}"] = assigned
        if s1 + s2 != 1:
            is_valid = False
            penalty_violations += (s1 + s2 - 1) ** 2

    bitstring = "".join(str(b) for b in x)
    return {
        "bitstring": bitstring,
        "is_valid": is_valid,
        "violations": penalty_violations,
        "slot_1": slot_assignments[1],
        "slot_2": slot_assignments[2],
        "exam_slots": exam_slots,
    }


def brute_force_solver(
    Q: np.ndarray,
    offset: float,
    clash_matrix: np.ndarray,
    lambda_penalty: float = QUBO_PENALTY_LAMBDA,
) -> Dict[str, Any]:
    """Exhaustively evaluates all 2^8 = 256 possible configurations.

    Identifies global minimum, all valid states, and verifies correctness.
    """
    total_states = 1 << NUM_VARIABLES  # 256
    records = []

    for state_int in range(total_states):
        x = np.array([(state_int >> (NUM_VARIABLES - 1 - b)) & 1 for b in range(NUM_VARIABLES)], dtype=float)
        total_energy = float(x.T @ Q @ x + offset)

        decoded = decode_solution(x)

        # Compute raw clash and raw penalty for transparency
        raw_penalty = lambda_penalty * float(decoded["violations"])
        raw_clash = total_energy - raw_penalty

        records.append(
            {
                "state_int": state_int,
                "bitstring": decoded["bitstring"],
                "total_energy": round(total_energy, 4),
                "raw_clash": round(raw_clash, 4),
                "raw_penalty": round(raw_penalty, 4),
                "is_valid": decoded["is_valid"],
                "slot_1": decoded["slot_1"],
                "slot_2": decoded["slot_2"],
            }
        )

    # Sort all states by total energy ascending
    all_sorted = sorted(records, key=lambda r: r["total_energy"])
    valid_sorted = [r for r in all_sorted if r["is_valid"]]

    ground_energy = all_sorted[0]["total_energy"]
    best_states = [r for r in all_sorted if abs(r["total_energy"] - ground_energy) < 1e-6]

    return {
        "total_states_evaluated": total_states,
        "total_valid_states": len(valid_sorted),
        "ground_energy": ground_energy,
        "best_states": best_states,
        "top_all_states": all_sorted[:10],
        "top_valid_states": valid_sorted[:6],
        "is_ground_state_valid": best_states[0]["is_valid"],
    }


if __name__ == "__main__":
    from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams

    print("=" * 70)
    print("Part b: QUBO Formulation & Brute-Force Check")
    print("=" * 70)

    # Obtain predictions from Part a
    _, X, y = generate_synthetic_enrolment_data()
    model, _, _, _, _, _ = train_and_evaluate_svr(X, y)
    C_matrix, _, _ = predict_clashes_for_exams(model)

    Q, offset = build_qubo_matrix(C_matrix)
    print(f"Constructed {Q.shape[0]}x{Q.shape[1]} QUBO Q Matrix (lambda = {QUBO_PENALTY_LAMBDA}):")
    print(np.round(Q, 2))
    print(f"QUBO Constant Energy Offset: {offset:.2f}\n")

    bf_results = brute_force_solver(Q, offset, C_matrix)
    print(f"Brute Force Evaluated {bf_results['total_states_evaluated']} States.")
    print(f"Number of Valid Schedules: {bf_results['total_valid_states']} (2^4 = 16 expected).")
    print(f"Lowest Energy Ground State Value: {bf_results['ground_energy']:.2f}")
    print(f"Is Ground State Valid: {bf_results['is_ground_state_valid']}")

    print("\nOptimal Ground States (Symmetric Twins):")
    for s in bf_results["best_states"]:
        print(f"  Bitstring: {s['bitstring']} | Energy: {s['total_energy']:.2f} | Slot 1: {s['slot_1']} | Slot 2: {s['slot_2']}")

    print("\nTop 5 Lowest Energy Valid Configurations:")
    for rank, s in enumerate(bf_results["top_valid_states"][:5], 1):
        print(f"  #{rank}: State {s['bitstring']} | Clash: {s['raw_clash']:.2f} students | S1: {s['slot_1']}, S2: {s['slot_2']}")
