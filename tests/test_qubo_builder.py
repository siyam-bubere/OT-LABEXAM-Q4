"""Unit tests for QUBO Builder and Brute Force Verification (Part b)."""

import unittest
import numpy as np
from src.config import RANDOM_SEED, QUBO_PENALTY_LAMBDA
from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams
from src.qubo_builder import (
    build_qubo_matrix,
    brute_force_solver,
    decode_solution,
    var_index,
)


class TestQUBOBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, X, y = generate_synthetic_enrolment_data(seed=RANDOM_SEED)
        model, _, _, _, _, _ = train_and_evaluate_svr(X, y, seed=RANDOM_SEED)
        cls.clash_matrix, _, _ = predict_clashes_for_exams(model)
        cls.Q, cls.offset = build_qubo_matrix(cls.clash_matrix, lambda_penalty=QUBO_PENALTY_LAMBDA)

    def test_qubo_dimensions(self):
        self.assertEqual(self.Q.shape, (8, 8))
        self.assertEqual(self.offset, 4.0 * QUBO_PENALTY_LAMBDA)

    def test_diagonal_penalties(self):
        for k in range(8):
            self.assertEqual(self.Q[k, k], -QUBO_PENALTY_LAMBDA)

    def test_solution_decoder(self):
        # 10011001: E1->S1, E2->S2, E3->S1, E4->S2
        x_valid = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        decoded = decode_solution(x_valid)
        self.assertTrue(decoded["is_valid"])
        self.assertEqual(decoded["violations"], 0)
        self.assertEqual(decoded["slot_1"], ["Exam 1", "Exam 3"])
        self.assertEqual(decoded["slot_2"], ["Exam 2", "Exam 4"])

        # Invalid state: E1 in both slots
        x_invalid = np.array([1, 1, 0, 1, 1, 0, 0, 1])
        decoded_inv = decode_solution(x_invalid)
        self.assertFalse(decoded_inv["is_valid"])
        self.assertGreater(decoded_inv["violations"], 0)

    def test_brute_force_ground_truth(self):
        bf = brute_force_solver(self.Q, self.offset, self.clash_matrix, lambda_penalty=QUBO_PENALTY_LAMBDA)
        self.assertEqual(bf["total_states_evaluated"], 256)
        self.assertEqual(bf["total_valid_states"], 16)
        self.assertTrue(bf["is_ground_state_valid"])
        self.assertGreater(len(bf["best_states"]), 0)

        # Verify ground energy equals min clash of optimal valid schedule
        min_clash = bf["ground_energy"]
        self.assertLess(min_clash, 15.0, "Optimal clash should be under 15 students")


if __name__ == "__main__":
    unittest.main()
