"""Unit tests for ML SVR module (Part a)."""

import unittest
import numpy as np
from src.config import RANDOM_SEED, DATASET_ROWS, EXAMS
from src.ml_svr import (
    generate_synthetic_enrolment_data,
    train_and_evaluate_svr,
    predict_clashes_for_exams,
    build_pair_feature,
)


class TestMLSVR(unittest.TestCase):
    def setUp(self):
        self.seed = RANDOM_SEED

    def test_synthetic_data_generation(self):
        df, X, y = generate_synthetic_enrolment_data(n_samples=DATASET_ROWS, seed=self.seed)
        self.assertEqual(len(df), DATASET_ROWS)
        self.assertEqual(X.shape, (DATASET_ROWS, 5))
        self.assertEqual(len(y), DATASET_ROWS)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue("common_students" in df.columns)
        self.assertTrue("same_dept" in df.columns)
        self.assertTrue("same_year" in df.columns)

    def test_svr_training_and_metrics(self):
        df, X, y = generate_synthetic_enrolment_data(n_samples=DATASET_ROWS, seed=self.seed)
        model, metrics, X_train, X_test, y_test, y_pred = train_and_evaluate_svr(X, y, seed=self.seed)

        self.assertEqual(metrics["n_train"], 420)
        self.assertEqual(metrics["n_test"], 180)
        self.assertGreater(metrics["test_r2"], 0.85, "SVR R^2 should exceed 0.85 on synthetic data")
        self.assertLess(metrics["test_rmse"], 6.0, "SVR RMSE should be below 6.0 students")

    def test_pairwise_predictions_and_clash_matrix(self):
        df, X, y = generate_synthetic_enrolment_data(n_samples=DATASET_ROWS, seed=self.seed)
        model, _, _, _, _, _ = train_and_evaluate_svr(X, y, seed=self.seed)
        clash_matrix, pair_preds, exam_summary = predict_clashes_for_exams(model, EXAMS)

        self.assertEqual(clash_matrix.shape, (4, 4))
        # Diagonal should be 0 (an exam does not clash with itself)
        for i in range(4):
            self.assertEqual(clash_matrix[i, i], 0.0)

        # Matrix should be symmetric: C_ij == C_ji
        np.testing.assert_allclose(clash_matrix, clash_matrix.T, atol=1e-5)

        # 6 pairs total
        self.assertEqual(len(pair_preds), 6)
        self.assertEqual(len(exam_summary), 4)

        # High clash expected between Exam 1 and Exam 2 (same dept CS, same year 3)
        self.assertGreater(clash_matrix[0, 1], 25.0, "CS Y3 core and elective should have large clash")


if __name__ == "__main__":
    unittest.main()
