<p align="center"><img src="../assets/readme/learning.svg" alt="Machine Learning Mini-Projects" width="100%"></p>

# Machine Learning Mini-Projects

**Worksheets and code connecting hand-built rules to learning and evaluation.**

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)

---

## At a glance

| Item | Description |
|---|---|
| Collection | Machine Learning Mini-Projects |
| Subject source inventory | 11 Python · 0 R · 0 notebooks |
| Verification level | Source inspection, selected saved artifacts, and focused regressions described in the repair log. |

## Scope and interpretation

Seven preserved PDFs explain the instructional sequence. Toy Healthy/Unhealthy labels and favorite-color encodings are teaching devices. Blood-pressure and credit-data experiments are educational and do not establish clinical or lending decision utility.

## Worksheet-to-code map

| Stage | Preserved worksheet | Focus |
|---|---|---|
| 1 | [Week1-Simple Linear Classifier.pdf](MachineLearning2025F/Week1-Simple%20Linear%20Classifier.pdf) | Weights, normalization, threshold labels; 10 training IDs and two held-out IDs. |
| 2 | [Week2-Simple Nonlinear Node.pdf](MachineLearning2025F/Week2-Simple%20Nonlinear%20Node.pdf) | Two hidden nodes and nonlinear score construction. |
| 3 | [Week3-Grid Search_Error Surface.pdf](MachineLearning2025F/Week3-Grid%20Search_Error%20Surface.pdf) | Search over weights/threshold and inspect an error surface. |
| 4 | [Week4-From Gradient Descent to Machine Learning.pdf](MachineLearning2025F/Week4-From%20Gradient%20Descent%20to%20Machine%20Learning.pdf) | Gradient updates, forward calculation, loss, and backpropagation. |
| 6 | [Week6-Blood Pressure regression.pdf](MachineLearning2025F/Week6-Blood%20Pressure%20regression.pdf) | Preprocessing and regression pipeline for systolic/diastolic targets. |
| 7 | [Week7-Data Visualization & Outlier Detection.pdf](MachineLearning2025F/Week7-Data%20Visualization%20%26%20Outlier%20Detection.pdf) | Exploratory plots, correlations, and outlier questions. |
| 7 dataset | [Week7Dataset-Interpretation.pdf](MachineLearning2025F/Week7Dataset-Interpretation.pdf) | Interpretation of the preserved credit-data exercise. |

Week 5 is represented by conversion/subset code and the newly completed visualization script rather than a separate listed worksheet. The plotting CLI accepts an explicit CSV and output directory, making its side effects visible.

## Evaluation questions worth following

The Week 6 source uses a train/validation split, training-only feature scaling, ridge/linear comparisons, and a NumPy one-hidden-layer MLP with activation/width/learning-rate exploration. A validation-selected model still needs a separate final evaluation before its score is presented as independent test performance.

Some earlier preprocessing is performed before splitting, and toy labels may be derived from input variables. Those design choices should be read in their instructional context. No new best-model score is reported by this documentation update.


## Project and study guides

| Guide | Question |
|---|---|
| [Machine Learning Worksheet Sequence](MachineLearning2025F/README.md) | How do weighted rules lead to search, gradient-based learning, and evaluation? |


## Reading and execution

Start with the guide matching your question, inspect its parameters and inputs, then follow links to source. The order is an explanatory reading path; it is not a guarantee that every script executes consecutively.

Read [the reproduction guide](../docs/REPRODUCIBILITY.md) before running code. Source files can load external data, contact services, write artifacts, run large searches, or control hardware. The focused validation suite does not reproduce all historical experiments.

---

[Portfolio](../README.md) · [Evidence & results](../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../docs/REPAIRS.md) · [Reproduction](../docs/REPRODUCIBILITY.md) · [Full code catalog](../docs/CODE_CATALOG.md)
