"""Unit tests for Quantum Annealing Solver (Part c)."""

import unittest
import numpy as np
from src.config import RANDOM_SEED
from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams
from src.qubo_builder import build_qubo_matrix, brute_force_solver
from src.quantum_solver import solve_qubo_quantum, compare_with_brute_force, SimulatedQuantumAnnealer


class TestQuantumSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, X, y = generate_synthetic_enrolment_data(seed=RANDOM_SEED)
        model, _, _, _, _, _ = train_and_evaluate_svr(X, y, seed=RANDOM_SEED)
        cls.clash_matrix, _, _ = predict_clashes_for_exams(model)
        cls.Q, cls.offset = build_qubo_matrix(cls.clash_matrix)
        cls.bf_res = brute_force_solver(cls.Q, cls.offset, cls.clash_matrix)

    def test_annealing_convergence(self):
        raw_samples, q_summary = solve_qubo_quantum(
            self.Q, self.offset, num_reads=200, num_sweeps=100, seed=RANDOM_SEED
        )
        self.assertEqual(len(raw_samples), 200)
        self.assertIn("best_bitstring", q_summary)
        self.assertIn("best_energy", q_summary)
        self.assertGreater(q_summary["best_appearance_count"], 0)

        # Ground energy must match brute force
        comp = compare_with_brute_force(q_summary, self.bf_res)
        self.assertAlmostEqual(q_summary["best_energy"], self.bf_res["ground_energy"], places=2)
        self.assertTrue(comp["is_optimal_found"])

    def test_sqa_trotter_sampler(self):
        annealer = SimulatedQuantumAnnealer(
            self.Q, self.offset, num_reads=50, num_sweeps=50, seed=RANDOM_SEED
        )
        sqa_samples = annealer.run_sqa_trotter(trotter_slices=4)
        self.assertEqual(len(sqa_samples), 50)
        min_sqa_energy = min(s[0] for s in sqa_samples)
        # Should reach near the ground state
        self.assertLessEqual(min_sqa_energy, self.bf_res["ground_energy"] + 5.0)


if __name__ == "__main__":
    unittest.main()
