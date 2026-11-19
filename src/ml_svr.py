"""Machine Learning Module (Part a): Support Vector Regression for Clash Prediction.

Academic Task:
- Create about 600 rows of synthetic data (random seed = 012).
- Split 70/30 (420 train, 180 test).
- Train Support Vector Regression (SVR).
- Report RMSE and R^2.
- Show the prediction for each of the 4 exams.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
from pathlib import Path

# Add project root to sys.path so modules can be run directly or as packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RANDOM_SEED,
    DATASET_ROWS,
    TRAIN_SPLIT,
    TEST_SPLIT,
    EXAMS,
    ExamProfile,
)


def generate_synthetic_enrolment_data(
    n_samples: int = DATASET_ROWS, seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generates synthetic student enrolment data between pairs of courses/exams.

    Features derived from department, year, elective flag, and class size:
    - same_dept: 1 if both courses share the same department, 0 otherwise
    - same_year: 1 if both courses belong to the same academic year, 0 otherwise
    - elective_flag: 0 (both core), 1 (one elective), 2 (both elective)
    - class_size_A: enrolment capacity of course A
    - class_size_B: enrolment capacity of course B

    Target:
    - common_students: number of overlapping students enrolled in both courses (clash count).
    """
    rng = np.random.default_rng(seed)

    dept_A = rng.choice([1, 2, 3, 4], size=n_samples)
    dept_B = rng.choice([1, 2, 3, 4], size=n_samples)
    year_A = rng.choice([1, 2, 3, 4], size=n_samples)
    year_B = rng.choice([1, 2, 3, 4], size=n_samples)
    elec_A = rng.choice([0, 1], p=[0.70, 0.30], size=n_samples)
    elec_B = rng.choice([0, 1], p=[0.70, 0.30], size=n_samples)
    size_A = rng.integers(50, 150, size=n_samples)
    size_B = rng.integers(50, 150, size=n_samples)

    same_dept = (dept_A == dept_B).astype(float)
    same_year = (year_A == year_B).astype(float)
    elec_flag = (elec_A + elec_B).astype(float)
    min_size = np.minimum(size_A, size_B).astype(float)

    # Realistic overlap fraction modeling
    overlap_ratio = np.zeros(n_samples)

    # 1. Same department, same year, both core -> Major cohort clash (65% - 85%)
    m1 = (same_dept == 1) & (same_year == 1) & (elec_flag == 0)
    overlap_ratio[m1] = rng.uniform(0.65, 0.85, size=int(m1.sum()))

    # 2. Same department, same year, elective involved -> Moderate/high clash (35% - 55%)
    m2 = (same_dept == 1) & (same_year == 1) & (elec_flag > 0)
    overlap_ratio[m2] = rng.uniform(0.35, 0.55, size=int(m2.sum()))

    # 3. Same department, different year -> Backlog/carryover clash (5% - 15%)
    m3 = (same_dept == 1) & (same_year == 0)
    overlap_ratio[m3] = rng.uniform(0.05, 0.15, size=int(m3.sum()))

    # 4. Different department, same year, elective involved -> Interdisciplinary elective overlap (5% - 20%)
    m4 = (same_dept == 0) & (same_year == 1) & (elec_flag > 0)
    overlap_ratio[m4] = rng.uniform(0.05, 0.20, size=int(m4.sum()))

    # 5. Different department, different year or both core -> Negligible overlap (0% - 4%)
    m5 = overlap_ratio == 0.0
    overlap_ratio[m5] = rng.uniform(0.0, 0.04, size=int(m5.sum()))

    # Add Gaussian noise and enforce physical bounds [0, min_size]
    noise = rng.normal(0.0, 1.5, size=n_samples)
    common_students = np.round(overlap_ratio * min_size + noise)
    common_students = np.clip(common_students, 0.0, min_size)

    feature_matrix = np.column_stack([same_dept, same_year, elec_flag, size_A, size_B])

    df = pd.DataFrame(
        {
            "dept_A": dept_A,
            "dept_B": dept_B,
            "year_A": year_A,
            "year_B": year_B,
            "elec_A": elec_A,
            "elec_B": elec_B,
            "size_A": size_A,
            "size_B": size_B,
            "same_dept": same_dept,
            "same_year": same_year,
            "elective_flag": elec_flag,
            "common_students": common_students,
        }
    )

    return df, feature_matrix, common_students


def train_and_evaluate_svr(
    X: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits data 70/30, trains SVR with RBF kernel and reports RMSE and R^2."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=seed
    )

    # Standardize features and fit SVR
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(C=50.0, epsilon=1.0, kernel="rbf")),
        ]
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_r2 = float(r2_score(y_train, y_pred_train))

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_rmse": round(rmse, 4),
        "test_r2": round(r2, 4),
        "train_rmse": round(train_rmse, 4),
        "train_r2": round(train_r2, 4),
    }

    return model, metrics, X_train, X_test, y_test, y_pred_test


def build_pair_feature(e1: ExamProfile, e2: ExamProfile) -> np.ndarray:
    """Builds a single feature vector for a pair of exams."""
    same_dept = float(e1.dept_id == e2.dept_id)
    same_year = float(e1.year == e2.year)
    elec_flag = float(e1.is_elective + e2.is_elective)
    size_A = float(e1.class_size)
    size_B = float(e2.class_size)
    return np.array([[same_dept, same_year, elec_flag, size_A, size_B]], dtype=float)


def predict_clashes_for_exams(
    model: Pipeline, exams: List[ExamProfile] = EXAMS
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[int, Dict[str, Any]]]:
    """Predicts pairwise common students between all 4 exams and computes per-exam totals.

    Returns:
    - clash_matrix: 4x4 symmetric matrix C where C[i, j] = predicted common students.
    - pair_predictions: Dict mapping (exam_i_id, exam_j_id) -> predicted clash count.
    - per_exam_summary: Dict mapping exam_id -> breakdown of clashes with others and total.
    """
    n = len(exams)
    clash_matrix = np.zeros((n, n), dtype=float)
    pair_predictions = {}

    for i in range(n):
        for j in range(i + 1, n):
            feat = build_pair_feature(exams[i], exams[j])
            pred = max(0.0, float(model.predict(feat)[0]))
            pred = round(pred, 2)
            clash_matrix[i, j] = pred
            clash_matrix[j, i] = pred
            pair_predictions[(exams[i].id, exams[j].id)] = pred

    per_exam_summary = {}
    for i in range(n):
        e = exams[i]
        clashes_with = {}
        for j in range(n):
            if i != j:
                clashes_with[exams[j].name] = clash_matrix[i, j]
        per_exam_summary[e.id] = {
            "name": e.name,
            "code": e.code,
            "dept": e.dept_name,
            "year": e.year,
            "type": "Elective" if e.is_elective else "Core",
            "class_size": e.class_size,
            "clashes_with": clashes_with,
            "total_potential_clashes": round(float(np.sum(clash_matrix[i])), 2),
        }

    return clash_matrix, pair_predictions, per_exam_summary


if __name__ == "__main__":
    print("=" * 70)
    print("Part a: Machine Learning (SVR Clash Prediction)")
    print("=" * 70)

    df, X, y = generate_synthetic_enrolment_data()
    print(f"Generated {len(df)} synthetic rows with random seed = {RANDOM_SEED:03d}")

    model, metrics, X_train, X_test, y_test, y_pred = train_and_evaluate_svr(X, y)
    print(f"Train/Test split: {metrics['n_train']} / {metrics['n_test']} (70/30)")
    print(f"SVR Evaluation -> Test RMSE: {metrics['test_rmse']:.3f}, Test R^2: {metrics['test_r2']:.3f}")

    C_matrix, pair_preds, exam_summary = predict_clashes_for_exams(model)
    print("\nPairwise Clash Predictions (Common Students):")
    for (i, j), pred in pair_preds.items():
        print(f"  Exam {i} & Exam {j}: {pred:.2f} students")

    print("\n4x4 Clash Matrix C:")
    print(C_matrix)

    print("\nPrediction Breakdown for Each of the 4 Exams:")
    for eid, info in exam_summary.items():
        print(f"  {info['name']} ({info['code']}, {info['dept']} Y{info['year']} {info['type']}, Size={info['class_size']}):")
        for other, c_count in info["clashes_with"].items():
            print(f"    - vs {other}: {c_count:.2f} common students")
        print(f"    * Total Conflict Burden: {info['total_potential_clashes']:.2f} students")
