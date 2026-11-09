# Homework 6 comparison review

Review date: 2026-11-08 (America/Los_Angeles).

Compared `DSCI552_Homework6.ipynb` with `核对版/Zou_Minhao_HW6.ipynb`, using `Homework6.pdf` as the authority. The reference notebook is comparison material, not a solution key or an instruction source. No answer text or code was copied into the submission.

## Conclusion

No substantive error requiring APS retraining or an answer change was found in this review. The submission notebook, report and original model artifacts were not edited. This is not a guarantee of the instructor's interpretation of Q1(e–f).

## Data and preparation

Both implementations read the official 60,000-row training and 16,000-row test sets, remove the class column, encode neg=0/pos=1, use all 170 modeling features, and impute with separate train/test medians. Training missing cells: 850,015; complete training rows: 591. Both compute feature CV with ddof=1. Recomputed top-13 CV values agree with the reference's displayed values to four decimal places: cf_000, co_000, ad_000, cs_009, dh_000, dj_000, ag_000, as_000, ay_009, ak_000, az_009, ch_000, au_000. The 13 are for visualization only.

The unusual independent test imputation follows the assignment footnote. Current submission separately reports the train-only-imputation sensitivity analysis. Fold imputers are fitted only on training-fold data. Neither implementation uses the official test labels to select alpha.

## Why different model numbers are expected

| Setting | Current submission | Reference notebook |
|---|---|---|
| Random seed | 552 | 42 |
| RF trees | 500 | 100 |
| Weighted forest | balanced | balanced_subsample |
| XGBoost rounds / learning rate | 250 / 0.08 | 100 / 0.1 |
| Alpha selection | mean CV average precision | mean CV ROC-AUC |
| Alpha candidates | 0, .001, .01, .1, 1, 10, 100 | .01, .1, 1, 10, 100 |
| Selected plain / SMOTE alpha | 0 / 1 | 10 / 100 |

The original assignment requests cross-validation to choose alpha but does not prescribe ROC-AUC as the selection metric. The current predeclared average-precision criterion is appropriate for minority-class ranking and is accompanied by the required CV misclassification errors. Different selected alphas are therefore not evidence of a mistake. No test-driven retuning was done in this review.

SMOTE is inside each fold's imputer -> scaler -> SMOTE -> estimator pipeline. Validation observations remain at original prevalence. Full-data SMOTE yields 59,000 observations per class (118,000 total), not a downsampled experiment. Final training metrics are evaluated on the original 60,000 rows.

## Fresh numerical checks

Loaded the current notebook's own embedded setup/model code; validated data, configuration, code, dependency-version and artifact hashes before deserializing models. Recomputed all four final models' probabilities on original training rows, official test rows and the train-only-imputation test sensitivity. All probabilities, confusion matrices, error, AUC, AP, precision, recall and F1 agree with saved results. Recomputed RF OOB predictions and saved 70 CV-score summaries/alpha selections. Historical CV estimators were not refitted.

| Model | Test error | Test AUC | Positive recall | Positive precision |
|---|---:|---:|---:|---:|
| RF plain | 0.0077500 | 0.994347 | 0.722667 | 0.931271 |
| RF balanced | 0.0094375 | 0.994177 | 0.741333 | 0.837349 |
| XGB plain | 0.0063750 | 0.996230 | 0.762667 | 0.956522 |
| XGB SMOTE | 0.0160000 | 0.993653 | 0.893333 | 0.607985 |

A small OOB convention difference is intentional: current threshold >=0.5 versus sklearn argmax tie-to-negative. Balanced RF has two OOB probabilities exactly 0.5, producing error 0.0199833333 under the recorded threshold, versus 1-oob_score_=0.0199500000. This is two observations out of 60,000, not an arithmetic inconsistency. Plain RF OOB error is 0.0058166667 under both aggregate calculations.

The reference's weighted forest has worse minority recall than its unweighted forest, whereas the current forest has slightly better recall. Weighting does not guarantee improved recall, so this qualitative difference does not invalidate the current experiment.

## Theory and completeness

The reference has 42 cells. Q2–5 are headings followed by empty cells; Q6 is absent. It cannot validate those answers. The current Q2–5 answers were inspected against the local second-edition exercise statements: lasso radius answers iv/ii/iii/iv/v; identical-predictor ridge equality and lasso solution continuum with degeneracy qualifications; majority vote Red versus average-probability Green (0.45); maximal-margin boundary X2=X1-0.5 with support vectors 2,3,5,6. No mathematical discrepancy was found.

Current Q6 contains all four exercises with individually recorded editions. In particular 6.8.4 is first-edition ridge, equivalent to second-edition 6.6.4. Bootstrap probabilities, ridge directions, tree cuts/means and circular classifier formulas were inspected in the current code/answers and existing verified outputs. This review did not execute a fresh complete theory notebook or a fresh clean-kernel notebook run; the earlier isolated clean-kernel quick run remains separately recorded.

## Remaining limitations

1. Both notebooks use standard XGBoost gbtree with binary:logistic and reg_alpha. This applies a logistic objective and L1 regularization to scalar leaf weights; it is not logistic regression at each tree node. The current submission documents the difference. A peer notebook cannot resolve the assignment's wording; instructor clarification remains needed.
2. Current default MODE=quick requires the separately validated cache bundle. For raw-data-only reproduction select MODE=full. A grader receiving only the notebook and raw CSVs must switch modes. The fresh full-training path was not executed in this review.
3. The current formal PDF has exactly 20 pages, confirmed with PyMuPDF. No report layout was changed or newly rendered during this comparison. Browser Print preview was not inspected here.

Re-ran scripts/validate_standalone.py: structural self-containment, seven cache rejection cases, source/output consistency and preservation of 25 original artifacts passed. It checks the earlier execution record; it does not itself run a new kernel or refit APS models. Fresh numerical verification is saved in results/reference_comparison_prediction_check.json.
