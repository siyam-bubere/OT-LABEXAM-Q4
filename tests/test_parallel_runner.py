"""Unit tests for Parallel Restarts Module (Part d)."""

import unittest
import multiprocessing as mp
from src.config import RANDOM_SEED
from src.ml_svr import generate_synthetic_enrolment_data, train_and_evaluate_svr, predict_clashes_for_exams
from src.qubo_builder import build_qubo_matrix
from src.parallel_runner import run_parallel_restarts, _annealing_restart_worker


class TestParallelRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, X, y = generate_synthetic_enrolment_data(seed=RANDOM_SEED)
        model, _, _, _, _, _ = train_and_evaluate_svr(X, y, seed=RANDOM_SEED)
        cls.clash_matrix, _, _ = predict_clashes_for_exams(model)
        cls.Q, cls.offset = build_qubo_matrix(cls.clash_matrix)

    def test_single_worker_execution(self):
        result = _annealing_restart_worker((123, self.Q, self.offset, 20, 50))
        self.assertIn("best_energy", result)
        self.assertIn("best_bitstring", result)
        self.assertTrue(result["is_valid"])

    def test_parallel_scaling_benchmarks(self):
        # Quick run for test suite
        res = run_parallel_restarts(
            self.Q,
            self.offset,
            worker_counts=[1, 2],
            num_restarts=4,
            reads_per_restart=20,
            sweeps_per_restart=50,
        )
        self.assertEqual(len(res["scaling_table"]), 2)
        self.assertEqual(res["scaling_table"][0]["workers"], 1)
        self.assertEqual(res["scaling_table"][0]["speedup"], 1.0)
        self.assertGreater(res["scaling_table"][1]["speedup"], 0.5)


if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()
