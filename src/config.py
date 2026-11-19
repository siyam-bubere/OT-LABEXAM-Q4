"""Configuration constants and problem specifications for Q.4.

Problem Context:
- Place 4 exams into 2 time slots (8 binary variables: x_{i,s} for i in {1..4}, s in {1..2}).
- Enrolment features predict common student clash counts via SVR.
- QUBO minimization assigns exams to slots to minimize clashes subject to single-slot constraints.
- Quantum Annealing solves the QUBO.
- Multiprocessing evaluates parallel scaling (1, 2, 4 workers).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

# Academic Context & Seeds
RANDOM_SEED = 12  # Specified as '012' in question prompt
DATASET_ROWS = 600
TRAIN_SPLIT = 0.70
TEST_SPLIT = 0.30

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR = BASE_DIR / "tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 4 Exams Specifications
# Features: Department (1=CS, 2=EE, 3=ME, 4=IT), Year (1..4), Elective Flag (0=Core, 1=Elective), Class Size
@dataclass(frozen=True)
class ExamProfile:
    id: int
    name: str
    code: str
    dept_id: int
    dept_name: str
    year: int
    is_elective: int
    class_size: int

EXAMS: List[ExamProfile] = [
    ExamProfile(id=1, name="Exam 1", code="CS301", dept_id=1, dept_name="CS", year=3, is_elective=0, class_size=120),
    ExamProfile(id=2, name="Exam 2", code="CS305", dept_id=1, dept_name="CS", year=3, is_elective=1, class_size=80),
    ExamProfile(id=3, name="Exam 3", code="EE302", dept_id=2, dept_name="EE", year=3, is_elective=0, class_size=100),
    ExamProfile(id=4, name="Exam 4", code="CS401", dept_id=1, dept_name="CS", year=4, is_elective=0, class_size=90),
]

NUM_EXAMS = 4
NUM_SLOTS = 2
NUM_VARIABLES = NUM_EXAMS * NUM_SLOTS  # 8 yes/no binary choices

# QUBO Penalty Multiplier
# Must be sufficiently large so that breaking any constraint (x_i1 + x_i2 = 1)
# incurs higher energy than any valid clash combination.
QUBO_PENALTY_LAMBDA = 100.0

# Quantum Annealing Parameters
ANNEALING_NUM_READS = 1000
ANNEALING_NUM_SWEEPS = 300
BETA_START = 0.1
BETA_END = 10.0

# Parallel Restart Parameters
PARALLEL_WORKER_COUNTS = [1, 2, 4]
PARALLEL_NUM_RESTARTS = 32  # Number of independent random seeds
