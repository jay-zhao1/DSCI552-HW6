# DSCI 552 Homework 6

Completed English submission for Questions 1(a)-(f), 2-5, and all four Question 6 Extra Practice exercises. Original files are preserved. `CHECKLIST.md` maps every formal subquestion to its deliverables. Question 6 is integrated into the main report on pages 16-19; no separate Extra Practice PDF is needed.

## Deliverables

- `DSCI552_Homework6.ipynb`: full experiment and figure code, explanations, executed outputs, all theoretical answers, complete 13-feature scatter matrix, all 170 CVs and detailed CV results.
- `DSCI552_Homework6_Report.pdf`: **20 actual PDF pages**, including references; no separate cover or appendix. Body text is 10.5 pt. Every page has been rendered for visual inspection.
- `scripts/experiments.py`: resumable full-data fitting and fold-local alpha searches.
- `scripts/figures.py`: training-only exploration, evaluation panels and theory diagrams.
- `scripts/content.py`: shared formal-answer content populated from saved results.
- `scripts/extra_practice.py`: all four additional answers, bootstrap simulation, ridge calculations, reconstructed trees/partitions and circular-boundary figures.
- `scripts/report_sections.py`: the combined 20-page organization; the two random-forest sections share settings and a unified metric table, with both original evaluation panels retained.
- `scripts/update_extra_notebook.py`: executes only the eight Question 6 code cells and preserves the original 20 code cells and their outputs.
- `scripts/build_report.py`, `scripts/build_notebook.py`: artifact generation and notebook execution.
- `scripts/verify.py`: independent metric, saved-estimator, fold, CV-ranking and feature-statistic checks.
- `results/`: actual metrics, per-fold and summary searches, serialized fitted models, probabilities, fold memberships, audit/hashes, environment, execution/validation records and page renders.
- `figures/`: high-resolution PNGs and vector PDF figures. `correlation_all_labels.pdf` is the large fully labeled 170-feature matrix; `scatter_all13.pdf` contains all pairs.
- `references/`: author-provided first/second-edition books, exact verified exercise-page images/text and installed/matching XGBoost API documentation. These are local reference materials, not additional report pages.
- `requirements.txt`: exact environment pins.

## Environment and commands

Actual run: Python 3.12.8 on macOS arm64, 8 logical CPUs and 16 GiB RAM. CPU histogram XGBoost; estimator threads/workers = 4; CV folds run sequentially. No GPU is required. The versions in `results/environment.json` are authoritative.

Run from this project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name python3 --display-name 'Python 3 (DSCI552 .venv)'
.venv/bin/python -u scripts/experiments.py
.venv/bin/python scripts/figures.py
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/extra_practice.py
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/finalize.py
```

For this update, the existing models were not retrained: `scripts/extra_practice.py`, `scripts/update_extra_notebook.py`, `scripts/build_report.py`, and `scripts/finalize.py` were run. The original 20 code cells and outputs are preserved; eight new self-contained cells were executed in a fresh kernel (their execution counts start at 1). To update only Question 6, run those four commands.

The notebook can also be opened and run top-to-bottom with the project `.venv` kernel. `build_notebook.py` explicitly executes all code cells with errors disallowed. With supplied checkpoints, it reuses the costly fits, regenerates figures, reruns mathematical calculations, and independently reproduces test probabilities from the serialized estimators. It does not present recomputed cached fits as newly trained experiments. All original full-data fits and all 70 fold evaluations were actually executed before the notebook was assembled.

`experiments.py` saves each fold immediately. It skips completed fold rows and skips a final model only when its completed JSON exists. Its initial JSON serialization issue for XGBoost's NaN missing-value parameter was fixed; a resumed invocation completed normally. The original log retains this history. JSON metadata now represents that parameter explicitly as a missing-value sentinel string; fitted models retain the real NaN setting.

For an independent **fresh fit without changing the supplied results**, create a separate directory with copies of `Homework6.pdf`, `Homework 6 Data/`, `scripts/`, and `requirements.txt`, create its environment, and run the commands there. An empty results directory triggers every forest and all 70 CV fits. Generated `tmp/pipeline_cache/` may be omitted from a submission/copy; it is a performance cache, not a source of results. Rebuilding artifacts uses the saved numerical results; no metrics are typed into tables by hand.

To read the PDF page count programmatically:

```bash
.venv/bin/python -c "from pypdf import PdfReader; print(len(PdfReader('DSCI552_Homework6_Report.pdf').pages))"
```

## Data and modeling assumptions

- Both CSV headers are at line 21; `na` denotes missing data. Official train/test sizes are 60,000/16,000, with 170 numeric predictors. Training labels: 59,000 neg and 1,000 pos; test labels: 15,625 neg and 375 pos. `pos=1`, `neg=0`; class is never a predictor.
- **Homework footnote convention:** main test results impute missing test values using test medians, separately from training medians. No labels enter preprocessing and no train/test statistics are pooled. This differs from ordinary deployment practice. A fixed-model train-only-imputation sensitivity analysis is separately saved for every model.
- **Inside CV:** raw training rows enter an imblearn Pipeline. The current training fold alone fits the median imputer and StandardScaler; SMOTE, when enabled, runs only on that training fold. Validation/test rows retain their natural distribution. No early stopping is used.
- All 170 training coefficients of variation use the sample SD (`ddof=1`) after training-median imputation. Zero means would yield undefined CVs and be excluded from ranking; entirely missing features would be flagged and retained with zero fallback. Neither occurs. The top 13 are for visualization only.
- Scatter plots use `log1p` axes, all 1,000 positives and a seeded 2,500-negative training subsample. Correlations and box plots use all 60,000 training rows. Log transformation is **not** used in the modeling pipeline.
- Both forests use 500 trees, seed 552 and otherwise matched settings. Compensation is `class_weight='balanced'`, with no forest resampling beyond ordinary bootstrap sampling. OOB errors use a 0.5 probability threshold and global training imputation, so they are approximate internal estimates, not preprocessing-cross-fitted errors.
- XGBoost: `gbtree`, `binary:logistic`, 250 rounds, depth 4, learning rate .08, `reg_lambda=1`, CPU `hist`, `max_bin=256`, `base_score=.5`, full row/feature sampling and seed 552. Both XGB procedures use train-fitted scaling for comparability. No class weight is used in either XGB procedure.
- Alpha candidates: 0, .001, .01, .1, 1, 10, 100. Stratified shuffled 5-fold CV selects maximum mean validation **average precision**, with smaller alpha breaking exact ties. CV error at 0.5 is reported too. Official test data do not select alpha, rounds, thresholds or any other setting.
- SMOTE uses standardized Euclidean geometry, `k_neighbors=5`, seed 552 and equal final class sizes: 59,000 per class / 118,000 total in the final fit. Each fold has 47,200 per class / 94,400 total after SMOTE. There is no downsampling. Training evaluations always use the original 60,000 observations.
- Every hard prediction uses `P(pos) >= 0.5` (including ties). Confusion matrices order true rows and predicted columns as `[neg, pos]`. ROC-AUC uses probabilities; AP means `average_precision_score`, not trapezoidal PR-AUC.

## XGBoost wording requiring instructor confirmation

The homework describes L1 logistic regression at each tree node. The installed XGBoost 3.4.1 API and matching official documentation do not supply that model through the ordinary tree booster. Here `reg_alpha` regularizes **scalar tree-leaf weights**; `binary:logistic` supplies logistic loss/link. `gblinear` is a global linear booster and is not a tree. No local clarification was found. The submission transparently uses standard boosted trees and tunes alpha, rather than claiming to implement a genuine logistic model tree. This is the only remaining interpretation issue.

## Actual results

Main official-test results following the footnote:

| Model | Selected alpha | Error | ROC-AUC | AP | Positive precision | Positive recall | Positive F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF: unweighted | - | 0.00775 | 0.99435 | 0.92609 | 0.93127 | 0.72267 | 0.81381 |
| RF: balanced | - | 0.00944 | 0.99418 | 0.88701 | 0.83735 | 0.74133 | 0.78642 |
| XGB: unweighted | 0.0 | 0.00637 | 0.99623 | 0.93199 | 0.95652 | 0.76267 | 0.84866 |
| XGB: SMOTE | 1.0 | 0.01600 | 0.99365 | 0.88362 | 0.60799 | 0.89333 | 0.72354 |

SMOTE detects more positives at the fixed threshold, at the expense of more false positives. Its train-only-imputation sensitivity result has error 0.01075 and AP 0.89974; this is materially different from the footnote main result and is not substituted for it. An AP-selected alpha of zero for plain XGBoost is the observed optimum of this grid, not an omitted search.

## Textbook verification

Author website: https://www.statlearning.com/ . The author-hosted second-edition download redirects to the authors' Google Drive PDF; the local file identifies the corrected printing as June 21, 2023. Verified second-edition printed pages: Q2 p.284 (PDF p.295); Q3 p.285 (PDF p.296); Q4 p.362 (PDF p.373); Q5 pp.398-399 (PDF pp.409-410). The first edition places the first two exercises in Section 6.8; the second edition matches the homework's Section 6.6. Q4/Q5 contents agree across both checked R editions. Answers are independently derived rather than copied from solution sites.

## Completed Question 6: edition verification and results

Each exercise was checked independently in both local R editions; this is not a blanket assumption about edition numbering.

| Homework reference | Primary source: first edition, seventh printing | Cross-check: second edition (June 2023 corrected printing) | Content |
|---|---|---|---|
| 5.4.2 | pp.197-198; PDF pp.212-213 | 5.4.2, pp.219-220; PDF pp.230-231 | Bootstrap inclusion, all (a)-(h) |
| 6.8.4 | pp.260-261; PDF pp.275-276 | Equivalent **6.6.4**, pp.284-285; PDF pp.295-296 | Ridge penalty, all (a)-(e) |
| 8.4.4 | pp.332-333; PDF pp.347-348; Figure 8.12 | 8.4.4, pp.361-362; PDF pp.372-373; Figure 8.14 | Tree from partition and partition from tree, (a)-(b) |
| 9.7.2 | p.368; PDF p.383 | 9.7.2, p.398; PDF p.409 | Circular boundary and feature lift, (a)-(d) |

There is no unresolved Question 6 statement ambiguity. Source excerpts, page images and edition metadata are saved in `references/extra_practice_sources/` and `references/*_extra_verified.txt`. The separate Question 6 notebook heading is exactly `Question 6 — Extra Practice`.

The actual bootstrap run uses seed 552, 10,000 samples of size 100 and j=4: 6,238 inclusions (0.6238), theoretical probability 0.6339676587, Monte Carlo SE 0.00481718. The difference is -2.11 SE; the seed was not changed to make the simulation closer. Every n=1,...,100000 is evaluated. Ridge monotonicity checks and an actual orthogonal-design illustration pass. Tree traversals and region formulas agree on 6,561 grid points including cut lines. The circle query classes are Blue, Red, Blue, Blue, with expanded scores 1, -3, 5, 48.

All four complete solutions are merged into the main report (20 pages including references). The duplicate region table remains in the notebook; the report includes both complete tree/region diagrams with all cutpoints and means. Original Q1-5 answers are retained; duplicated forest settings and descriptions are consolidated. The original 17-page report and notebook are backed up in `results/pre_extra_practice/`. The 21 original model, prediction and search files are checked against pre-update SHA-256 hashes.

## Verification records

`results/validation.json`: existing metric/model/fold checks and final artifact checks; `results/extra_practice_validation.json`: mathematical, preservation and page-limit checks for this update; `results/extra_practice_results.json`: actual new computations; `results/extra_notebook_execution.json`: eight new executed code cells, zero errors. `results/notebook_execution.json`: all code-cell execution counts and zero-error status. `results/report_layout.json`: actual pagination and layout metadata. `results/report_pages/`: each rendered page and contact sheets. Original data hashes are recorded in `results/data_audit.json`.
