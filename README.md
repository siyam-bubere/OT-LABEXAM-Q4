# Hybrid Machine Learning, Quantum QUBO Optimization & Parallel Computing for University Exam Scheduling

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)]()
[![Optimization](https://img.shields.io/badge/QUBO-8--Qubit%20Ising-purple.svg)]()
[![Course Outcomes](https://img.shields.io/badge/Outcomes-CO1%20%7C%20CO2%20%7C%20CO4-orange.svg)]()

A complete, production-grade implementation solving the **University Exam Scheduling Problem under Conflict Constraints** by integrating **Support Vector Regression (ML)**, **Quadratic Unconstrained Binary Optimization (QUBO)**, **Simulated Quantum Annealing (Quantum Computing)**, and **Multi-core Process Scaling (Parallel Computing)**.

---

## Table of Contents
1. [Project Title & Overview](#project-title--overview)
2. [Introduction](#introduction)
3. [Problem Statement (Question 4)](#problem-statement-question-4)
4. [Libraries and Tech Stack](#libraries-and-tech-stack)
5. [Assumptions](#assumptions)
6. [Mathematical Formulations and Algorithms](#mathematical-formulations-and-algorithms)
   - [Part a: Machine Learning (SVR Clash Prediction)](#part-a-machine-learning-svr-clash-prediction)
   - [Part b: QUBO Matrix Construction & Penalty Formulation](#part-b-qubo-matrix-construction--penalty-formulation)
   - [Part c: Quantum Computing (Simulated Annealing on QUBO)](#part-c-quantum-computing-simulated-annealing-on-qubo)
   - [Part d: Parallel Computing (Multiprocessing Pool Restarts)](#part-d-parallel-computing-multiprocessing-pool-restarts)
   - [Part e: Three-Line Academic Conclusion](#part-e-three-line-academic-conclusion)
7. [Repository Structure](#repository-structure)
8. [Step-by-Step Working & Execution Guide](#step-by-step-working--execution-guide)
9. [Detailed Benchmark Results & Metrics](#detailed-benchmark-results--metrics)
10. [Visual Plots & Verification Artifacts](#visual-plots--verification-artifacts)
11. [Unit Test Suite](#unit-test-suite)

---

## Project Title & Overview
**Project Title:** *Multi-Paradigm Exam Timetable Optimization: Predictive Support Vector Regression, QUBO Energy Minimization, Simulated Quantum Annealing, and Multi-Core Parallel Restarts.*

This project addresses a classic combinatorial NP-hard operations research challenge faced by academic examination offices: placing exams into non-conflicting time slots when exact student clash data cannot be surveyed directly and must be learned from enrolment features.

The repository demonstrates a complete 4-stage pipeline:
1. **Machine Learning (`CO1`)**: Train an RBF Support Vector Regressor (SVR) on 600 synthetic enrolment rows (`random_seed = 012`) to predict pairwise student conflict counts ($C_{ij}$).
2. **QUBO Formulation & Verification (`CO2`)**: Formulate the single-slot allocation rule and clash minimization objective into an $8 \times 8$ QUBO Hamiltonian ($2^8 = 256$ states) with quadratic penalty enforcement, proven by exhaustive brute force.
3. **Quantum Computing (`CO2`)**: Solve the QUBO using Simulated Quantum Annealing across 1000 reads, reporting the ground state energy, appearance frequency, and direct comparison against brute force.
4. **Parallel High-Performance Computing (`CO4`)**: Distribute independent multi-seed restarts across 1, 2, and 4 CPU workers using `multiprocessing.Pool`, reporting wall-clock time, speed-up ($S_p$), and parallel efficiency ($E_p$).

---

## Introduction
Modern university scheduling requires assigning academic examinations to a limited number of time slots such that no student is scheduled to sit two exams simultaneously. In practice:
- Pre-enrolment rosters are dynamic, partial, or private;
- Direct clash matrices are often unavailable until late in the semester;
- Departmental affinities, student seniority (year levels), course elective status, and cohort sizes strongly correlate with student overlap.

By using **Support Vector Regression (SVR)**, the exam cell predicts the expected student clash between any two courses from institutional features. Subsequently, the scheduling decision is transformed into a **Quadratic Unconstrained Binary Optimization (QUBO)** model compatible with quantum annealers (such as D-Wave QPUs) and classical quantum simulators. Finally, because quantum annealing is a meta-heuristic subject to quantum tunneling probabilities, **Parallel High-Performance Computing** is leveraged to launch concurrent restarts across distinct random seeds, dramatically boosting time-to-solution and ground-state discovery.

---

## Problem Statement (Question 4)
*Reproduced verbatim from the examination question sheet:*

> **Q.4 (5 marks) ML + Quantum + Parallel Computing**
> 
> An exam cell must place 4 exams into 2 time slots; students enrolled in two exams cannot sit them together, but the clash counts are only available from enrolment features.
> 
> **Idea:** a Support Vector Regression will predict number of common students between two exams from department, year, elective flag, class size; these predictions show how good or bad each choice is. Then a quantum algorithm must assign every exam to one of 2 options (8 yes/no choices in total), and parallel computing is used to speed up the repeated runs.
> 
> - **a.** Machine Learning: create about 600 rows of synthetic data (random seed = 012), split 70/30, train the Support Vector Regression and report RMSE and $R^2$. Show the prediction for each of the 4 exams. **[01 CO1]**
> - **b.** Turn the decision into a QUBO (energy function): write it down, build its Q matrix from your predictions, add a penalty for breaking the rule, and check by brute force that the lowest-energy answer is correct. **[01 CO2]**
> - **c.** Quantum Computing: solve the QUBO with Quantum annealing (simulated) on a simulator. Report the best answer, its energy and how often it appears; compare with brute force. **[1.5 CO2]**
> - **d.** Parallel Computing: run independent QAOA/VQE/annealing restarts (different random seeds) in parallel using `multiprocessing.Pool`. Report the time with 1, 2 and 4 workers, the speed-up and the efficiency. **[01 CO4]**
> - **e.** Conclusion in 3 lines: your final choice and what you learned. **[0.5 CO4]**
> 
> **END OF PAPER**

---

## Libraries and Tech Stack

| Category | Library / Tool | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `>= 3.9` (Tested on 3.13) | Core development platform |
| **Scientific Computing** | `numpy` | `>= 1.24.0` | Vectorized matrix operations, QUBO algebra |
| **Numerical Optimization**| `scipy` | `>= 1.10.0` | Scientific algorithms and distribution samplers |
| **Machine Learning** | `scikit-learn` | `>= 1.2.0` | Support Vector Regression (SVR), standard scaling |
| **Data Manipulation** | `pandas` | `>= 2.0.0` | Structured DataFrame handling and tabular metrics |
| **Data Visualization** | `matplotlib` | `>= 3.7.0` | Publication-quality plots and visual diagnostics |
| **Parallel Computing** | `multiprocessing` | Built-in | Process-level parallelism across multi-core CPUs |
| **Testing** | `unittest` | Built-in | Automated regression and unit test suite |

### Installation Instructions
Install the required dependencies via pip:
```bash
pip install -r requirements.txt
```
Or install the package in editable mode:
```bash
pip install -e .
```

---

## Assumptions
To ensure rigorous academic fidelity and reproducibility, the following domain assumptions are adopted:

1. **Discrete Scheduling Grid:**
   - 4 exams are indexed $i \in \{1, 2, 3, 4\}$.
   - 2 discrete, non-overlapping time slots are indexed $s \in \{1, 2\}$ (e.g., Morning Slot 1 and Afternoon Slot 2).
   - Each exam must be scheduled into **exactly one** slot (hard single-assignment rule).

2. **Binary State Space (8 Yes/No Choices):**
   - For each exam $i$ and slot $s$, a binary decision variable $x_{i, s} \in \{0, 1\}$ indicates whether exam $i$ is assigned to slot $s$.
   - $4 \text{ exams} \times 2 \text{ slots} = 8\text{ binary variables}$.
   - The state space contains $2^8 = 256$ total configurations, of which exactly $2^4 = 16$ satisfy the single-slot rule.

3. **Academic Profiles for the 4 Target Exams:**
   - **Exam 1 (`CS301`)**: CS Department (ID=1), 3rd Year, Core Course (`elective=0`), Class Size = 120.
   - **Exam 2 (`CS305`)**: CS Department (ID=1), 3rd Year, Elective Course (`elective=1`), Class Size = 80.
   - **Exam 3 (`EE302`)**: EE Department (ID=2), 3rd Year, Core Course (`elective=0`), Class Size = 100.
   - **Exam 4 (`CS401`)**: CS Department (ID=1), 4th Year, Core Course (`elective=0`), Class Size = 90.

4. **SVR Enrolment Dynamics:**
   - Courses within the same department and same academic year exhibit the highest common student overlap (up to 70–85% of cohort size).
   - Elective cross-enrollments produce moderate overlap (35–55%).
   - Backlog/carryover repeaters across different years produce minor overlap (5–15%).
   - Inter-departmental core courses exhibit near-zero overlap (0–4%).
   - The pairwise clash matrix is symmetric ($C_{ij} = C_{ji}$) with zero diagonal ($C_{ii} = 0$).

5. **QUBO Penalty Weighting ($\lambda$):**
   - To guarantee that any invalid schedule (leaving an exam unassigned or assigning it to both slots) has higher energy than any valid schedule, the penalty multiplier is chosen as $\lambda = 100.0$, which strictly exceeds the maximum possible clash sum ($\sum_{i<j} C_{ij} \approx 71.74$).

6. **Random Seed Standardization:**
   - The prompt specifies `random seed = 012`. In Python decimal integer literals, this is represented as `12` (`int("012") = 12`).

---

## Mathematical Formulations and Algorithms

### Part a: Machine Learning (SVR Clash Prediction)
The regression model predicts the continuous number of overlapping students $y \in [0, \min(N_A, N_B)]$ between course pair $(A, B)$ based on feature vector:
$$\mathbf{x}_{(A,B)} = \left[ \mathbb{I}_{(\text{dept}_A = \text{dept}_B)}, \; \mathbb{I}_{(\text{year}_A = \text{year}_B)}, \; (\text{elec}_A + \text{elec}_B), \; N_A, \; N_B \right]^T \in \mathbb{R}^5$$

The Support Vector Regressor solves the primal optimization problem:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{k=1}^{N_{\text{train}}} (\xi_k + \xi_k^*)$$
$$\text{subject to } \begin{cases} y_k - (\mathbf{w}^T \phi(\mathbf{x}_k) + b) \le \epsilon + \xi_k \\ (\mathbf{w}^T \phi(\mathbf{x}_k) + b) - y_k \le \epsilon + \xi_k^* \\ \xi_k, \xi_k^* \ge 0 \end{cases}$$
where $\phi(\cdot)$ is the non-linear feature mapping induced by the Radial Basis Function (RBF) kernel:
$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$

**Evaluation Metrics:**
$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{k=1}^{N_{\text{test}}} (y_k - \hat{y}_k)^2}, \qquad R^2 = 1 - \frac{\sum_{k=1}^{N_{\text{test}}} (y_k - \hat{y}_k)^2}{\sum_{k=1}^{N_{\text{test}}} (y_k - \bar{y})^2}$$

---

### Part b: QUBO Matrix Construction & Penalty Formulation
Let binary decision vector $\mathbf{x} \in \{0, 1\}^8$ be indexed by:
$$k(i, s) = 2(i - 1) + (s - 1), \quad i \in \{1, 2, 3, 4\}, \; s \in \{1, 2\}$$
$$\mathbf{x} = [x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}, x_{3,1}, x_{3,2}, x_{4,1}, x_{4,2}]^T$$

#### 1. Objective Function (Clash Minimization)
If two exams $i$ and $j$ ($i < j$) are scheduled into the same time slot $s$, a conflict penalty equal to $C_{ij}$ is incurred:
$$H_{\text{clash}}(\mathbf{x}) = \sum_{s=1}^2 \sum_{1 \le i < j \le 4} C_{ij} \, x_{i, s} x_{j, s}$$

#### 2. Constraint Penalty (Single-Slot Rule)
Each exam $i$ must sit in exactly one slot: $x_{i, 1} + x_{i, 2} = 1$. The penalty function is:
$$P_{\text{rule}}(\mathbf{x}) = \lambda \sum_{i=1}^4 (x_{i, 1} + x_{i, 2} - 1)^2$$
Since $x_{i, s} \in \{0, 1\}$, $x_{i, s}^2 = x_{i, s}$. Expanding:
$$(x_{i, 1} + x_{i, 2} - 1)^2 = x_{i, 1}^2 + x_{i, 2}^2 + 2 x_{i, 1} x_{i, 2} - 2 x_{i, 1} - 2 x_{i, 2} + 1 = - x_{i, 1} - x_{i, 2} + 2 x_{i, 1} x_{i, 2} + 1$$
Summing over all 4 exams:
$$P_{\text{rule}}(\mathbf{x}) = \lambda \sum_{i=1}^4 (- x_{i, 1} - x_{i, 2} + 2 x_{i, 1} x_{i, 2}) + 4\lambda$$

#### 3. Total QUBO Hamiltonian
$$E(\mathbf{x}) = H_{\text{clash}}(\mathbf{x}) + P_{\text{rule}}(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x} + \text{offset}$$
where $\text{offset} = 4\lambda = 400.0$, and the $8 \times 8$ matrix $Q$ has elements:
- **Diagonal entries:** $Q_{k, k} = -\lambda = -100.0$ for all $k \in \{0, \dots, 7\}$
- **Constraint off-diagonals:** $Q_{2i-2, 2i-1} = 2\lambda = 200.0$ (couples $x_{i, 1}$ and $x_{i, 2}$)
- **Clash off-diagonals:** $Q_{k(i, s), k(j, s)} = C_{ij}$ for $i < j$ (couples same slot $s$)
- **Remaining off-diagonals:** $0.0$

```
Upper-Triangular Q Matrix (8x8):
[[-100.0,  200.0,   39.70,    0.0,    2.05,    0.0,   10.08,    0.0],
 [   0.0, -100.0,    0.0,   39.70,    0.0,    2.05,    0.0,   10.08],
 [   0.0,    0.0, -100.0,  200.0,   10.42,    0.0,    7.72,    0.0],
 [   0.0,    0.0,    0.0, -100.0,    0.0,   10.42,    0.0,    7.72],
 [   0.0,    0.0,    0.0,    0.0, -100.0,  200.0,    2.06,    0.0],
 [   0.0,    0.0,    0.0,    0.0,    0.0, -100.0,    0.0,    2.06],
 [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0, -100.0,  200.0],
 [   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0, -100.0]]
Offset = 400.0
```

---

### Part c: Quantum Computing (Simulated Annealing on QUBO)
The QUBO is solved by simulating the quantum annealing physical process:
$$H(t) = A(t) H_{\text{driver}} + B(t) H_{\text{problem}}$$
In classical simulation, the Markov Chain Monte Carlo (MCMC) Metropolis-Hastings transition rule with geometric inverse temperature cooling $\beta \in [\beta_0, \beta_f]$ is applied:
$$\Delta E_k = E(\mathbf{x} \oplus \mathbf{e}_k) - E(\mathbf{x}) = \Delta x_k \left( 2 (Q_{\text{sym}} \mathbf{x})_k + Q_{kk} \Delta x_k \right)$$
$$\mathcal{P}(\mathbf{x} \to \mathbf{x} \oplus \mathbf{e}_k) = \min\left(1, \; \exp\left(-\beta \Delta E_k\right)\right)$$
This incremental update reduces spin-flip complexity to $\mathcal{O}(N)$, enabling 1000 reads with 300 sweeps each to complete in milliseconds.

---

### Part d: Parallel Computing (Multiprocessing Pool Restarts)
Stochastic quantum annealing is vulnerable to getting trapped in local minima in rugged energy landscapes. Independent restarts with different random initial states and random seeds $\mathcal{S} = \{s_1, s_2, \dots, s_K\}$ are mapped across CPU worker pools:
$$\text{Pool}(p).map(f_{\text{anneal}}, \mathcal{S})$$

**Parallel Benchmarks:**
- **Speed-up ($S_p$):** Ratio of single-worker runtime to $p$-worker runtime:
  $$S_p = \frac{T_1}{T_p}$$
- **Parallel Efficiency ($E_p$):** Processor utilization percentage:
  $$E_p = \frac{S_p}{p} = \frac{T_1}{p \cdot T_p} \times 100\%$$

---

### Part e: Three-Line Academic Conclusion
The final choice and core learnings synthesized into exactly 3 lines:
```text
1. Final Choice: Schedule Slot 1 = [Exam 2, Exam 4] and Slot 2 = [Exam 1, Exam 3], achieving the global minimum clash burden of 9.77 students.
2. ML & Quantum Learning: SVR-predicted clash matrices seamlessly map into an 8-variable QUBO Hamiltonian, where quadratic penalty terms enforce exact single-slot constraints and simulated quantum annealing converges directly to the brute-force global minimum.
3. Parallel Computing Learning: Running independent stochastic restarts via multiprocessing.Pool delivers practical parallel acceleration (1.46x speed-up on 4 workers), proving that parallelizing multi-seed quantum restarts maximizes ground-state hit rates while mitigating process-spawning overhead.
```

---

## Repository Structure

```
exam_scheduling_qubo/
├── README.md                           # Master project documentation (this file)
├── REPORT.md                           # Detailed academic submission report (CO1, CO2, CO4)
├── requirements.txt                    # Python dependency requirements
├── pyproject.toml                      # Modern packaging specification
├── .gitignore                          # Standard git exclusion patterns
├── run_pipeline.py                     # Master execution script (runs stages a -> e)
│
├── src/                                # Core modular source code
│   ├── __init__.py                     # Package initialization
│   ├── config.py                       # Configuration constants (seeds, exams, lambdas)
│   ├── ml_svr.py                       # Part a: Synthetic data, SVR model, clash prediction
│   ├── qubo_builder.py                 # Part b: QUBO Q matrix, penalty, brute-force solver
│   ├── quantum_solver.py               # Part c: Simulated quantum annealing, frequency analysis
│   ├── parallel_runner.py              # Part d: Multiprocessing pool restarts, speedup & efficiency
│   ├── conclusion.py                   # Part e: Strict 3-line conclusion generator
│   └── visualizer.py                   # High-resolution figure rendering
│
├── tests/                              # Automated unit test suite
│   ├── __init__.py
│   ├── test_ml_svr.py                  # Unit tests for SVR training & metrics
│   ├── test_qubo_builder.py            # Unit tests for QUBO formulation & 256-state search
│   ├── test_quantum_solver.py          # Unit tests for quantum annealing convergence
│   └── test_parallel_runner.py         # Unit tests for multiprocessing scaling
│
├── notebooks/                          # Interactive exploratory notebooks
│   └── exam_scheduling_q4.ipynb        # End-to-end Jupyter notebook with inline plots
│
└── results/                            # Generated artifacts and plots
    ├── metrics_summary.json            # Machine-readable output metrics
    ├── svr_evaluation.png              # SVR test scatter plot & 4x4 clash heatmap
    ├── qubo_energy_spectrum.png        # Complete 256-state QUBO energy spectrum
    ├── annealing_solution_distribution.png # Quantum solution histogram
    └── parallel_speedup_efficiency.png # Speed-up and efficiency curves
```

---

## Step-by-Step Working & Execution Guide

### Option 1: One-Command Master Pipeline
To execute all stages end-to-end, compute metrics, print tables, and generate plots:
```bash
python run_pipeline.py
```

### Option 2: Run Individual Stages
Run any stage independently from the project root:

```bash
# Part a: Train SVR and display 4-exam clash predictions
python src/ml_svr.py

# Part b: Construct 8x8 Q matrix and execute 256-state brute force check
python src/qubo_builder.py

# Part c: Execute 1000-read Quantum Annealing simulation
python src/quantum_solver.py

# Part d: Benchmark multi-core parallel scaling (1, 2, 4 workers)
python src/parallel_runner.py

# Part e: Print the strict 3-line academic conclusion
python src/conclusion.py
```

### Option 3: Run the Automated Unit Test Suite
To verify code correctness and numeric tolerances:
```bash
python -m unittest discover tests
```

---

## Detailed Benchmark Results & Metrics

### 1. Machine Learning SVR Evaluation (`CO1`)
- **Dataset Size:** 600 synthetic pairwise rows (`random_seed = 012`)
- **Partition:** 420 Train / 180 Test (70% / 30%)
- **Test RMSE:** **3.4376** students
- **Test $R^2$:** **0.9326**

**Predicted Pairwise Clash Matrix $C$ (Common Students):**
| Course Pair | Features Involved | Predicted Clash Count |
| :--- | :--- | :---: |
| **Exam 1 & Exam 2** | CS Department, Year 3 (Core vs Elective) | **39.70 students** |
| **Exam 1 & Exam 3** | CS vs EE Department, Year 3 (Both Core) | **2.05 students** |
| **Exam 1 & Exam 4** | CS Department, Year 3 vs Year 4 (Carryover) | **10.08 students** |
| **Exam 2 & Exam 3** | CS vs EE Department, Year 3 (Elective) | **10.42 students** |
| **Exam 2 & Exam 4** | CS Department, Year 3 vs Year 4 (Elective) | **7.72 students** |
| **Exam 3 & Exam 4** | EE vs CS Department, Year 3 vs Year 4 (Both Core) | **2.06 students** |

### 2. QUBO & Brute Force Check (`CO2`)
- **Total Configurations:** $2^8 = 256$ states
- **Valid Single-Slot Schedules:** Exactly $2^4 = 16$ states
- **Ground State Energy:** **9.77 students**
- **Optimal Schedule:**
  - **Slot 1:** `{Exam 2, Exam 4}` (Clash: $C_{2,4} = 7.72$)
  - **Slot 2:** `{Exam 1, Exam 3}` (Clash: $C_{1,3} = 2.05$)
  - **Total Clash Cost:** $7.72 + 2.05 = \mathbf{9.77}$ students
- **Symmetric Twin:** `Slot 1: {Exam 1, Exam 3}`, `Slot 2: {Exam 2, Exam 4}` ($E = 9.77$)

### 3. Quantum Annealing Simulator (`CO2`)
- **Total Reads:** 1000
- **Sweeps per Read:** 300
- **Ground State Hit Rate:** Primary twin `01100110` appeared **82 times (8.2%)**; symmetric twin `10011001` appeared **66 times (6.6%)**; combined ground state recovery = **14.8%**.
- **Energy Gap vs Brute Force:** $\Delta E = 0.0000$ (Exact global minimum recovered).

### 4. Parallel Computing Benchmark (`CO4`)
- **Portfolio:** 16 independent restarts mapped across worker pools.
- **Minimum Energy Found:** **9.77** across all worker configurations.

| Workers ($p$) | Execution Time ($T_p$) | Observed Speed-up ($S_p$) | Parallel Efficiency ($E_p$) |
| :---: | :---: | :---: | :---: |
| **1 Worker** | **19.8898 s** | **1.00x** (Baseline) | **100.0%** |
| **2 Workers** | **14.2796 s** | **1.39x** | **69.6%** |
| **4 Workers** | **13.5851 s** | **1.46x** | **36.6%** |

---

## Visual Plots & Verification Artifacts
All plots are rendered at 300 DPI and saved directly to `results/`:

1. **`results/svr_evaluation.png`**:
   - *Left:* Actual vs. Predicted common students on 180 test samples with $y=x$ ideal line.
   - *Right:* Annotated heatmap of the $4 \times 4$ pairwise conflict matrix $C$.
2. **`results/qubo_energy_spectrum.png`**:
   - The full energy landscape across all 256 states, clearly separating valid configurations (green) from constraint-violated configurations (red) due to the $\lambda=100.0$ penalty barrier.
3. **`results/annealing_solution_distribution.png`**:
   - Bar chart of top solutions sampled by the Quantum Annealer, illustrating that ground-state twins dominate the lowest-energy states.
4. **`results/parallel_speedup_efficiency.png`**:
   - Multi-core speed-up curve ($S_p$) vs ideal linear speedup alongside the parallel efficiency bar chart ($E_p$).

---

## Unit Test Suite
The repository includes a comprehensive `unittest` suite covering all modules:
- `test_synthetic_data_generation`: Confirms 600 rows, valid bounds, and feature columns.
- `test_svr_training_and_metrics`: Confirms $R^2 > 0.85$ and $\text{RMSE} < 6.0$.
- `test_pairwise_predictions_and_clash_matrix`: Confirms $4 \times 4$ matrix symmetry and zero diagonal.
- `test_qubo_dimensions`: Validates $(8, 8)$ shape and penalty off-diagonals.
- `test_solution_decoder`: Verifies bitstring-to-schedule interpretation and penalty violation counting.
- `test_brute_force_ground_truth`: Validates 256 state evaluations and 16 valid schedules.
- `test_annealing_convergence`: Confirms quantum simulator recovers the brute-force global minimum.
- `test_parallel_scaling_benchmarks`: Verifies worker execution and positive speed-up.

Run the test suite with:
```bash
python -m unittest discover tests
```
Result:
```text
...........
----------------------------------------------------------------------
Ran 11 tests in 6.701s

OK
```

---

## License
This project is released under the **MIT License**.
