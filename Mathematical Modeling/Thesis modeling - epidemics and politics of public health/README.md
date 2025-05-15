# Optimal Intervention Strategies in Age-Structured SIR Models

This repository contains all Python code and computational resources used in the thesis:

**“Optimal Intervention Strategies in Age-Structured SIR Models: Epidemiological and Economic Analysis of COVID-19 in Korea”**

by Joon Chun (California State University, Los Angeles).

---

## Overview

This repository provides reproducible scripts and analysis tools for simulating, evaluating, and optimizing public health interventions against COVID-19.  
It covers the full workflow, from constructing age-structured epidemic models and parameter estimation to scenario simulations and economic cost-effectiveness analysis.

All scripts are organized to correspond to specific thesis sections, as summarized in the table below.

---

## Table of Contents

- [Folder Structure](#folder-structure)
- [Key Scripts and Correspondence to Thesis](#key-scripts-and-correspondence-to-thesis)
- [License](#license)
- [Contact](#contact)

---

## Folder Structure

/ (this repository root)
 ├── DataSets/          # Data files (Korean COVID-19, parameters, etc.)
 ├── Figures/           # Output images/plots (auto-generated)
 ├── *.py               # Core Python scripts for simulation and analysis
 └── README.md          # This file

---

## Key Scripts and Correspondence to Thesis

| Thesis Section | Description                                                  | Code File(s)                                                 |
| :------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|    3.1–3.2     | Construction of the transmission matrix and calculation of the basic reproduction number ($R_0$) | (3.1-1) Transmission Matrix.py<br>(3.1-2) Transmission Matrix plot real data roll over cumulative.py<br>(3.1-3) Create It data sets from Real data with ODEs.py<br>(3.1-4) Transmission Matrix parameter fitting.py |
|      3.3       | Estimates how Korea's COVID-19 policy levels affect transmission rates by calculating beta multipliers ("reduction factors") from data. | (3.2-1) R0 computation with Korea transmission matrix.py<br>(3.2-2) R0 computation.py<br>(3.2-3) R0 computation with reduction.py |
|      3.4       | Analysis of beta reduction multipliers across different intervention levels. | (5.1.1) Beta multiplier level1.py<br>(5.1.2) Beta multiplier level2.py<br>(5.1.3) Beta multiplier level3.py<br>(5.2.1) Beta multiplier analysis.py<br>(5.2.2) Beta multiplier visualization.py |
|     3.5-0      | Component-wise sensitivity analysis of the beta matrix.      | (3.3-0) Componentwise Beta matrix sensitive analysis.py      |
|      3.5       | Simulation of various quarantine scenarios in schools and workplaces. | (3.3-1) Quarantine at school 0 3 level graphs.py<br>(3.3-2) Quarantine at workplace 0 3 level graphs.py<br>(3.3-3) Quarantine at school and workplaces 0 3 level graphs.py<br>(3.3-4) Shutdown all the groups 0 3 level terminal level.py |
|      3.6       | Evaluation of vaccination strategies under different scenarios and capacities. | (4.1.6) Vaccination analysis base.py<br>(4.2.1) Vaccination scenario one.py<br>(4.2.2) Vaccination scenario two.py<br>(4.3.1) Vaccination cost effectiveness.py<br>(4.3.2) Vaccination policy eval.py<br>(4.4.1) Prioritize Vaccination.py<br>(4.4.2) Prioritize Vaccination keeping total capacity.py |
|      4.1       | Formulation of the economic model and estimation of associated costs. | (7.0.1) Economic model intro.py<br>(7.1.1) GDP loss calculation.py |
|      4.2       | Comprehensive economic analysis of non-pharmaceutical interventions (NPIs). | (8.2.0) NPI economic analysis.py                             |
|      4.3       | Cost-effectiveness analysis of combined intervention strategies. | (9.1.0) Combined intervention scenario.py<br>(9.1.1) Intervention cost summary.py<br>(9.2.1) Sensitivity economic policy.py |
|      4.4       | Optimization of intervention strategies for maximum economic benefit. | (9.3.1) Optimization main.py<br>(9.3.2) Optimization policy compare.py |



## License

This repository is for academic and research purposes.
 Feel free to cite, adapt, or extend with attribution.

------

## Contact

For questions or collaboration, contact:

- **Joon Chun**
- Email: jchun@calstatela.edu

------

## Citation

If you use this code for research, please cite the thesis:

> Chun, Joon. *Optimal Intervention Strategies in Age-Structured SIR Models: Epidemiological and Economic Analysis of COVID-19 in Korea*. M.S. Thesis, California State University, Los Angeles, 2025.