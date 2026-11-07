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
- [x] Q6: explicitly Extra Practice, not required in submission; record only: ISLR 5.4.2, 6.8.4, 8.4.4, 9.7.2. The original cross-edition numbering is retained verbatim and is not silently substituted.

Final execution, metrics, PDF pagination and visual QA are recorded in results/validation.json and results/report_layout.json after artifact generation.
