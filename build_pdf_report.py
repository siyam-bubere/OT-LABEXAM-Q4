"""PDF Generation Script for Optimization Techniques Lab Exam - Question 4.

Author: Siyam Bubere (Reg ID: 241070012)
Creates a publication-quality multi-page PDF report with:
- Formal academic cover & metadata
- Mathematical formulations & algorithm descriptions
- Formatted tables & matrices
- High-resolution visual diagrams and benchmark plots
- Terminal execution screenshots
"""

import sys
import os
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas

RESULTS_DIR = PROJECT_ROOT / "results"
PDF_OUTPUT_PATH = PROJECT_ROOT / "REPORT_QUESTION_4.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54,
                11 * inch - 36,
                "Optimization Techniques Lab | Siyam Bubere (Reg ID: 241070012) | Q.4 Report",
            )
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * inch - 54, 45)

        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, footer_text)
        self.drawString(
            54, 32, "CONFIDENTIAL & ACADEMIC — Hybrid ML + Quantum QUBO + Parallel HPC"
        )
        self.restoreState()


def generate_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0d47a1"),
        alignment=0,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2e7d32"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0d47a1"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1b5e20"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=6,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1b1b1b"),
        backColor=colors.HexColor("#f5f5f5"),
        borderPadding=6,
        spaceAfter=6,
    )

    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=8,
    )

    story = []

    # ---------------------------------------------------------
    # HEADER BANNER & STUDENT DETAILS
    # ---------------------------------------------------------
    story.append(
        Paragraph("OPTIMIZATION TECHNIQUES LAB EXAMINATION — QUESTION 4", title_style)
    )
    story.append(
        Paragraph(
            "Hybrid Machine Learning, Quantum QUBO Minimization & Multi-Core Parallel Restarts",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#0d47a1"),
            spaceBefore=2,
            spaceAfter=10,
        )
    )

    # Student Details Table
    student_data = [
        [
            Paragraph("<b>Candidate Name:</b>", body_style),
            Paragraph("Siyam Bubere", body_style),
            Paragraph("<b>Registration ID:</b>", body_style),
            Paragraph("241070012", body_style),
        ],
        [
            Paragraph("<b>Class / Semester:</b>", body_style),
            Paragraph("TY B.Tech Computer Engg (Sem 5)", body_style),
            Paragraph("<b>Course / Lab:</b>", body_style),
            Paragraph("Optimization Techniques Lab", body_style),
        ],
        [
            Paragraph("<b>Exam Problem:</b>", body_style),
            Paragraph("Question 4 (5 Marks)", body_style),
            Paragraph("<b>Random Seed:</b>", body_style),
            Paragraph("<b>012</b> (Derived from Reg ID 241070012)", body_style),
        ],
        [
            Paragraph("<b>Course Outcomes:</b>", body_style),
            Paragraph(
                "CO1 (01 mark: ML) | CO2 (2.5 marks: QUBO & Quantum) | CO4 (1.5 marks: HPC)",
                body_style,
            ),
            "",
            "",
        ],
    ]

    t_student = Table(student_data, colWidths=[1.4 * inch, 2.3 * inch, 1.4 * inch, 2.2 * inch])
    t_student.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#90caf9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("SPAN", (1, 3), (3, 3)),
            ]
        )
    )
    story.append(t_student)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # EXECUTIVE SUMMARY OF RESULTS TABLE
    # ---------------------------------------------------------
    story.append(Paragraph("Executive Summary of Results", h2_style))
    exec_data = [
        ["Subpart / Component", "Methodology / Tool", "Benchmark Metric / Result", "CO Mapping"],
        ["a. Predictive ML", "Support Vector Regression (RBF)", "Test RMSE: 3.438 | Test R²: 0.933", "CO1 [1m]"],
        ["b. QUBO Formulation", "Quadratic Penalty Method (λ=100)", "8 vars, 256 states; Ground Energy: 9.77", "CO2 [1m]"],
        ["b. Brute Force Validation", "Exhaustive State Search (2^8)", "16 valid schedules; Ground Truth: 9.77", "CO2 [-]"],
        ["c. Quantum Annealing", "Simulated Annealing Simulator", "E = 9.77 (8.2% primary / 14.8% combined)", "CO2 [1.5m]"],
        ["d. Parallel Restarts", "multiprocessing.Pool (1, 2, 4)", "1w: 19.89s | 2w: 14.28s | 4w: 13.59s (1.46x)", "CO4 [1m]"],
        ["e. Optimal Choice", "Slot 1: {E2, E4} | Slot 2: {E1, E3}", "Min Clash: 9.77 students (>86% reduction)", "CO4 [0.5m]"],
    ]

    t_exec = Table(exec_data, colWidths=[1.8 * inch, 2.1 * inch, 2.4 * inch, 1.0 * inch])
    t_exec.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8f9fa")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8dc")),
                ("ALIGN", (3, 0), (3, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_exec)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SYSTEM ARCHITECTURE DIAGRAM
    # ---------------------------------------------------------
    story.append(Paragraph("System Architecture & Workflow", h1_style))
    story.append(
        Paragraph(
            "The end-to-end framework bridges predictive machine learning, unconstrained quantum optimization, and high-performance multi-core execution to resolve timetable conflicts:",
            body_style,
        )
    )

    arch_img_path = RESULTS_DIR / "system_architecture.png"
    if arch_img_path.exists():
        story.append(Image(str(arch_img_path), width=7.0 * inch, height=3.5 * inch))
        story.append(
            Paragraph("Figure 1: Complete 4-Stage Optimization Pipeline Flowchart", caption_style)
        )

    # ---------------------------------------------------------
    # PART A: MACHINE LEARNING (SVR)
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Part a: Machine Learning — Clash Prediction via SVR [CO1, 1 Mark]", h1_style))
    story.append(
        Paragraph(
            "<b>Objective:</b> Predict student conflict counts between courses using academic features before enrolment finalize.<br/>"
            "<b>Dataset:</b> 600 synthetic pairwise rows generated with <code>random_seed = 012</code> partitioned into 70% Train (420 rows) and 30% Test (180 rows).<br/>"
            "<b>Features:</b> <code>same_dept</code>, <code>same_year</code>, <code>elective_flag</code>, <code>class_size_A</code>, <code>class_size_B</code>.<br/>"
            "<b>Model Architecture:</b> Support Vector Regressor (SVR) with Standard Scaler and Radial Basis Function (RBF) kernel ($C=50.0, \\epsilon=1.0$).",
            body_style,
        )
    )

    svr_metrics_data = [
        ["Metric", "Training Set (420)", "Testing Set (180)", "Academic Interpretation"],
        ["Root Mean Squared Error (RMSE)", "2.8941 students", "3.4376 students", "Average deviation below 3.5 students"],
        ["Coefficient of Determination (R²)", "0.9512", "0.9326", "Explains 93.3% of clash variation"],
    ]
    t_svr = Table(svr_metrics_data, colWidths=[2.2 * inch, 1.4 * inch, 1.4 * inch, 2.3 * inch])
    t_svr.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8e6c9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8e9")]),
            ]
        )
    )
    story.append(t_svr)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Predictions for the 4 Target Examinations", h2_style))
    pred_data = [
        ["Exam Pair", "Course Names & Academic Context", "Shared Features", "Predicted Clash Count"],
        ["Exam 1 & Exam 2", "CS301 (Y3 Core) vs CS305 (Y3 Elective)", "Same Dept, Same Year (Core+Elec)", "39.70 students (CRITICAL)"],
        ["Exam 1 & Exam 3", "CS301 (CS Y3) vs EE302 (EE Y3 Core)", "Diff Dept, Same Year (Both Core)", "2.05 students (Low)"],
        ["Exam 1 & Exam 4", "CS301 (CS Y3) vs CS401 (CS Y4 Core)", "Same Dept, Diff Year (Carryover)", "10.08 students (Moderate)"],
        ["Exam 2 & Exam 3", "CS305 (CS Y3) vs EE302 (EE Y3)", "Diff Dept, Same Year (Elective)", "10.42 students (Moderate)"],
        ["Exam 2 & Exam 4", "CS305 (CS Y3) vs CS401 (CS Y4)", "Same Dept, Diff Year (Elective)", "7.72 students (Low)"],
        ["Exam 3 & Exam 4", "EE302 (EE Y3) vs CS401 (CS Y4)", "Diff Dept, Diff Year (Core)", "2.06 students (Low)"],
    ]
    t_pred = Table(pred_data, colWidths=[1.3 * inch, 2.6 * inch, 2.0 * inch, 1.4 * inch])
    t_pred.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b5e20")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8e6c9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fbe7")]),
            ]
        )
    )
    story.append(t_pred)
    story.append(Spacer(1, 8))

    svr_img_path = RESULTS_DIR / "svr_evaluation.png"
    if svr_img_path.exists():
        story.append(Image(str(svr_img_path), width=7.0 * inch, height=2.75 * inch))
        story.append(
            Paragraph("Figure 2: SVR Test Regression Scatter Plot (Left) & 4x4 Pairwise Conflict Matrix Heatmap (Right)", caption_style)
        )

    # ---------------------------------------------------------
    # PART B: QUBO FORMULATION & BRUTE FORCE
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Part b: QUBO Formulation & Brute Force Verification [CO2, 1 Mark]", h1_style))
    story.append(
        Paragraph(
            "<b>Decision Variables:</b> Let $x_{i, s} \\in \\{0, 1\\}$ denote whether Exam $i \\in \\{1..4\\}$ is assigned to Slot $s \\in \\{1..2\\}$.<br/>"
            "Vector $\\mathbf{x} = [x_{1,1}, x_{1,2}, x_{2,1}, x_{2,2}, x_{3,1}, x_{3,2}, x_{4,1}, x_{4,2}]^T \\in \\{0, 1\\}^8$ (8 binary choices).<br/><br/>"
            "<b>Constraint (Single-Slot Placement):</b> $\\sum_{s=1}^2 x_{i, s} = 1 \\implies (x_{i,1} + x_{i,2} - 1) = 0$.<br/>"
            "Using penalty multiplier $\\lambda = 100.0$: $P(\\mathbf{x}) = \\lambda \\sum_{i=1}^4 (x_{i,1} + x_{i,2} - 1)^2$.<br/>"
            "Expanding via idempotency ($x_{i,s}^2 = x_{i,s}$): $(x_{i,1} + x_{i,2} - 1)^2 = -x_{i,1} - x_{i,2} + 2 x_{i,1} x_{i,2} + 1$.<br/><br/>"
            "<b>Objective Function (Clash Minimization):</b> $H_{\\text{clash}}(\\mathbf{x}) = \\sum_{s=1}^2 \\sum_{i < j} C_{ij} x_{i,s} x_{j,s}$.<br/>"
            "<b>Total Energy Function:</b> $E(\\mathbf{x}) = H_{\\text{clash}}(\\mathbf{x}) + P(\\mathbf{x}) = \\mathbf{x}^T Q \\mathbf{x} + 400.0$.",
            body_style,
        )
    )

    story.append(Paragraph("Constructed 8x8 Q Matrix (Upper-Triangular, λ = 100.0)", h2_style))
    q_matrix_text = (
        "Q = [[-100.0,  200.0,   39.70,    0.00,    2.05,    0.00,   10.08,    0.00],\n"
        "     [   0.0, -100.0,    0.00,   39.70,    0.00,    2.05,    0.00,   10.08],\n"
        "     [   0.0,    0.0, -100.0,  200.00,   10.42,    0.00,    7.72,    0.00],\n"
        "     [   0.0,    0.0,    0.0, -100.00,    0.00,   10.42,    0.00,    7.72],\n"
        "     [   0.0,    0.0,    0.0,    0.00, -100.00,  200.00,    2.06,    0.00],\n"
        "     [   0.0,    0.0,    0.0,    0.00,    0.00, -100.00,    0.00,    2.06],\n"
        "     [   0.0,    0.0,    0.0,    0.00,    0.00,    0.00, -100.00,  200.00],\n"
        "     [   0.0,    0.0,    0.0,    0.00,    0.00,    0.00,    0.00, -100.00]]"
    )
    story.append(Paragraph(f"<pre>{q_matrix_text}</pre>", code_style))

    story.append(Paragraph("Exhaustive Brute Force Search Verification (256 States)", h2_style))
    story.append(
        Paragraph(
            "Every possible binary state was systematically checked to verify that the minimum energy corresponds to a valid timetable with minimum clashes:",
            body_style,
        )
    )

    bf_table_data = [
        ["State Rank", "Binary Bitstring", "QUBO Energy", "Single-Slot Valid", "Slot 1 Assignment", "Slot 2 Assignment", "Clash Cost"],
        ["#1 (Global Min)", "01100110", "9.77", "True (0 penalty)", "Exam 2, Exam 4", "Exam 1, Exam 3", "7.72 + 2.05 = 9.77"],
        ["#2 (Sym. Twin)", "10011001", "9.77", "True (0 penalty)", "Exam 1, Exam 3", "Exam 2, Exam 4", "2.05 + 7.72 = 9.77"],
        ["#3", "01100101", "14.19", "True", "Exam 2", "Exam 1, Exam 3, Exam 4", "14.19 students"],
        ["#4", "10011010", "14.19", "True", "Exam 1, Exam 3, Exam 4", "Exam 2", "14.19 students"],
        ["#5", "01101010", "20.20", "True", "Exam 2, Exam 3, Exam 4", "Exam 1", "20.20 students"],
        ["Penalized States", "e.g. 00000000", "400.00", "False (4 violations)", "No exams assigned", "+400.00 penalty barrier"],
    ]
    t_bf = Table(bf_table_data, colWidths=[1.1 * inch, 1.1 * inch, 0.9 * inch, 1.1 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
    t_bf.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90caf9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8f5e9"), colors.HexColor("#e8f5e9"), colors.white, colors.white, colors.white, colors.HexColor("#ffebee")]),
            ]
        )
    )
    story.append(t_bf)
    story.append(Spacer(1, 8))

    qubo_img_path = RESULTS_DIR / "qubo_energy_spectrum.png"
    if qubo_img_path.exists():
        story.append(Image(str(qubo_img_path), width=7.0 * inch, height=2.6 * inch))
        story.append(
            Paragraph("Figure 3: Complete QUBO Energy Spectrum (256 States) Separating Valid Schedules from Penalized States", caption_style)
        )

    # ---------------------------------------------------------
    # OPTIMAL SCHEDULE DIAGRAM
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Optimal Schedule Layout vs. Suboptimal Conflict", h1_style))
    story.append(
        Paragraph(
            "Separating Exam 1 and Exam 2 into distinct slots is the single most critical decision, averting <b>39.70 clashing students</b> and reducing overall conflicts by over 86%:",
            body_style,
        )
    )

    sched_img_path = RESULTS_DIR / "optimal_schedule_diagram.png"
    if sched_img_path.exists():
        story.append(Image(str(sched_img_path), width=7.0 * inch, height=2.7 * inch))
        story.append(
            Paragraph("Figure 4: Schedule Comparison — Suboptimal Grouping (Left) vs. QUBO Optimal Allocation (Right)", caption_style)
        )

    # ---------------------------------------------------------
    # PART C: QUANTUM COMPUTING (ANNEALING)
    # ---------------------------------------------------------
    story.append(Paragraph("Part c: Quantum Computing — Simulated Annealing on QUBO [CO2, 1.5 Marks]", h1_style))
    story.append(
        Paragraph(
            "<b>Algorithm:</b> Simulated Quantum Annealing on QUBO using Markov Chain Monte Carlo with geometric inverse temperature cooling schedule $\\beta \\in [0.1, 10.0]$ across 300 sweeps per read.<br/>"
            "<b>Performance:</b> 1000 independent reads executed in milliseconds via vectorized local $\\Delta E$ computation.",
            body_style,
        )
    )

    q_data = [
        ["Quantum Metric", "Observed Simulator Output", "Brute Force Ground Truth", "Comparative Verification"],
        ["Best Solution Bitstring", "01100110 (or twin 10011001)", "01100110 / 10011001", "EXACT MATCH (Global Minimum)"],
        ["Ground State Energy", "9.77 students", "9.77 students", "Energy Gap: ΔE = 0.0000"],
        ["Primary Hit Rate", "82 / 1000 reads (8.2%)", "N/A (Exhaustive)", "High convergence probability"],
        ["Combined Twin Hit Rate", "148 / 1000 reads (14.8%)", "N/A (Symmetric Ground)", "Dominates all valid states"],
        ["Timetable Schedule", "Slot 1: {E2, E4} | Slot 2: {E1, E3}", "Slot 1: {E2, E4} | Slot 2: {E1, E3}", "Zero constraint violations"],
    ]
    t_q = Table(q_data, colWidths=[1.8 * inch, 2.0 * inch, 1.9 * inch, 1.6 * inch])
    t_q.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1bee7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3e5f5")]),
            ]
        )
    )
    story.append(t_q)
    story.append(Spacer(1, 8))

    q_img_path = RESULTS_DIR / "annealing_solution_distribution.png"
    if q_img_path.exists():
        story.append(Image(str(q_img_path), width=7.0 * inch, height=2.6 * inch))
        story.append(
            Paragraph("Figure 5: Histogram of Top Solutions Sampled by Simulated Annealing (Ground State Twins Highlighted in Green)", caption_style)
        )

    # ---------------------------------------------------------
    # PART D: PARALLEL COMPUTING
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Part d: Parallel Computing — Multiprocessing Pool Scaling [CO4, 1 Mark]", h1_style))
    story.append(
        Paragraph(
            "<b>Multi-Core Setup:</b> 16 independent restarts (unique random seeds) distributed across worker pools using Python's <code>multiprocessing.Pool</code>.<br/>"
            "<b>Scaling Metrics:</b> Speed-up $S_p = T_1 / T_p$, Parallel Efficiency $E_p = \\frac{S_p}{p} \\times 100\\%$.",
            body_style,
        )
    )

    par_data = [
        ["Workers (p)", "Execution Time (T_p)", "Observed Speed-up (S_p)", "Parallel Efficiency (E_p)", "Amdahl's Law & Scaling Behavior"],
        ["1 Worker", "19.8898 seconds", "1.00x (Baseline)", "100.0%", "Purely sequential baseline execution"],
        ["2 Workers", "14.2796 seconds", "1.39x", "69.6%", "Strong scaling across dual cores"],
        ["4 Workers", "13.5851 seconds", "1.46x", "36.6%", "Bounded by Windows process spawning & IPC"],
    ]
    t_par = Table(par_data, colWidths=[1.1 * inch, 1.5 * inch, 1.4 * inch, 1.4 * inch, 1.9 * inch])
    t_par.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c2185b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f8bbd0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fce4ec")]),
            ]
        )
    )
    story.append(t_par)
    story.append(Spacer(1, 8))

    par_img_path = RESULTS_DIR / "parallel_speedup_efficiency.png"
    if par_img_path.exists():
        story.append(Image(str(par_img_path), width=7.0 * inch, height=2.6 * inch))
        story.append(
            Paragraph("Figure 6: Parallel Speed-up Curve (Left) & Parallel Efficiency Utilization Bar Chart (Right)", caption_style)
        )

    # ---------------------------------------------------------
    # TERMINAL EXECUTION SCREENSHOT
    # ---------------------------------------------------------
    story.append(Paragraph("Live Terminal Execution Snapshot", h2_style))
    term_img_path = RESULTS_DIR / "pipeline_execution_screenshot.png"
    if term_img_path.exists():
        story.append(Image(str(term_img_path), width=7.0 * inch, height=2.8 * inch))
        story.append(
            Paragraph("Figure 7: High-Resolution Screenshot of Terminal Output Executing <code>python run_pipeline.py</code>", caption_style)
        )

    # ---------------------------------------------------------
    # PART E: CONCLUSION IN 3 LINES
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Part e: Conclusion in 3 Lines [CO4, 0.5 Mark]", h1_style))
    story.append(
        Paragraph(
            "<b>Verbatim 3-Line Conclusion answering Question 4(e):</b>", body_style
        )
    )

    conclusion_box_text = (
        "<b>Line 1 (Final Choice):</b> Schedule Slot 1 = [Exam 2, Exam 4] and Slot 2 = [Exam 1, Exam 3], "
        "achieving the global minimum clash burden of 9.77 students.<br/><br/>"
        "<b>Line 2 (ML & Quantum Learning):</b> SVR-predicted clash matrices seamlessly map into an 8-variable QUBO Hamiltonian, "
        "where quadratic penalty terms enforce exact single-slot constraints and simulated quantum annealing converges directly to the brute-force global minimum.<br/><br/>"
        "<b>Line 3 (Parallel Computing Learning):</b> Running independent stochastic restarts via multiprocessing.Pool delivers practical "
        "parallel acceleration (1.46x speed-up on 4 workers), proving that parallelizing multi-seed quantum restarts maximizes ground-state hit rates while mitigating process-spawning overhead."
    )

    t_conc = Table([[Paragraph(conclusion_box_text, body_style)]], colWidths=[7.0 * inch])
    t_conc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f5e9")),
                ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#2e7d32")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(t_conc)
    story.append(Spacer(1, 14))

    # ---------------------------------------------------------
    # CODEBASE & UNIT TESTS VERIFICATION
    # ---------------------------------------------------------
    story.append(Paragraph("Codebase Verification & Test Suite Summary", h2_style))
    story.append(
        Paragraph(
            "The accompanying repository includes 11 automated unit tests verifying numeric tolerances, mathematical idempotency, and multiprocessing stability:",
            body_style,
        )
    )

    tests_summary = [
        ["Test Module", "Tests Executed", "Passing", "Coverage Focus"],
        ["test_ml_svr.py", "3 Unit Tests", "3 / 3 PASS", "600 synthetic rows, 70/30 split, R² > 0.85, symmetric C matrix"],
        ["test_qubo_builder.py", "4 Unit Tests", "4 / 4 PASS", "8x8 shape, diagonal penalties, 256 states check, ground truth = 9.77"],
        ["test_quantum_solver.py", "2 Unit Tests", "2 / 2 PASS", "Convergence to brute force ground truth, SQA Trotter sampler"],
        ["test_parallel_runner.py", "2 Unit Tests", "2 / 2 PASS", "Multi-core pool execution, speedup & efficiency calculation"],
        ["Total Suite", "11 Unit Tests", "11 / 11 PASS", "100% Pass Rate in 6.70 seconds"],
    ]
    t_tests = Table(tests_summary, colWidths=[1.8 * inch, 1.2 * inch, 1.2 * inch, 2.8 * inch])
    t_tests.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0bec5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#eceff1")]),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#c8e6c9")),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t_tests)
    story.append(Spacer(1, 14))

    # Submission footer sign-off
    signoff_text = (
        "<b>Submission Endorsement:</b><br/>"
        "I certify that this submission represents original analytical derivations, code implementations, "
        "and empirical benchmark results conducted for the Optimization Techniques Lab Exam (Question 4).<br/><br/>"
        "<b>Candidate:</b> Siyam Bubere &nbsp;|&nbsp; <b>Reg ID:</b> 241070012 &nbsp;|&nbsp; <b>Date:</b> November 19, 2026"
    )
    story.append(Paragraph(signoff_text, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report generated successfully at: {PDF_OUTPUT_PATH}")
    print(f"File size: {PDF_OUTPUT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    generate_pdf()
