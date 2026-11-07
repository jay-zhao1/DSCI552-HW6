# Homework 6 compliance checklist

Source: supplied two-page Homework6.pdf and APS description. No AGENTS.md was found in the project or its applicable ancestor directories. No course clarification or local textbook was present initially.

- [x] 1(a): preserve official train/test; detect real CSV header and `na`; audit 170 numeric features, labels, order, counts, missingness and hashes.
- [x] 1(b)(i): discuss imputation; separate train/test medians for footnote main results; fold-fitted train-only preprocessing in CV; train-only sensitivity.
- [x] 1(b)(ii): all 170 sample CVs with ddof=1; zero/all-missing checks and stated fallback.
- [x] 1(b)(iii): complete correlation matrix, large labeled export, CSV and indexed report figure.
- [x] 1(b)(iv): top 13 CVs; full scatter matrix, readable report scatter panels and all class box plots; sampling/transform disclosure; answer significance question; all 170 retained for modeling.
- [x] 1(b)(v): class counts in each official split; imbalance discussion.
- [x] 1(c): uncompensated RF, train/test confusion, ROC/AUC/error; OOB versus test.
- [x] 1(d): cited weighting rationale; matched RF settings and all metrics; positive precision/recall/F1/AP.
- [x] 1(e): inspect installed XGBoost API and matching v3.4.1 docs; distinguish boosters/L1/model trees; transparent runnable interpretation.
- [x] 1(e): full-data stratified 5-fold alpha search (0, .001, .01, .1, 1, 10, 100), prespecified mean AP selection, CV error, full final evaluations, no official-test tuning or early stopping.
- [x] 1(f): full-data fold-local imputer -> scaler -> SMOTE -> XGB Pipeline; same alpha search; no downsampling; original-distribution validation/test; original-sample training evaluation; counts and timings.
- [x] Q2: second-edition ISLR 6.6.3 (a)-(e), checked against author-provided textbook, nested-set proof and tradeoff explanation.
- [x] Q3: second-edition ISLR 6.6.5 (a)-(d), ridge/lasso derivations and degeneracy qualifications.
- [x] Q4: ISLR 8.4.5, ten red probabilities; hard vote and probability average with computed diagram.
- [x] Q5: ISLR 9.7.3 (a)-(h), plotted data, hyperplanes, margin/support vectors, perturbation and nonseparability proof.
- [x] Q6: all four Extra Practice exercises completed in English and merged into the main report; independent notebook heading `Question 6 — Extra Practice`.
- [x] Q6 / ISLR 5.4.2 (a)-(h): first-edition pp.197-198, cross-checked second-edition pp.219-220; exact probabilities, all-integer curve through 100,000, actual 10,000-sample bootstrap simulation.
- [x] Q6 / ISLR 6.8.4 (a)-(e): first-edition pp.260-261; second-edition equivalent is 6.6.4; optimality proof, variance/bias formulas, computed ridge illustration and qualifications.
- [x] Q6 / ISLR 8.4.4 (a)-(b): first-edition Figure 8.12, second-edition Figure 8.14; reconstructed tree and partition; five mean values and all cuts checked on 6,561 points.
- [x] Q6 / ISLR 9.7.2 (a)-(d): verified both editions; circle/interior/exterior, four query classifications and linear lifted-space formula.
- [x] Final organization: main report exactly 20 pages including all references; Extra Practice on pp.16-19; no separate supplement required. Original 17-page report and notebook archived under results/pre_extra_practice.
- [x] Execution: only the eight new code cells were executed; Q1-5 code/outputs and original experiment checkpoints preserved. New figures rendered and reviewed; no model retraining, commits, pushes or external submissions.

Final execution, metrics, PDF pagination and visual QA are recorded in results/validation.json and results/report_layout.json after artifact generation.
