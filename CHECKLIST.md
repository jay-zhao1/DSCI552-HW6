# Homework 6 final checklist

Current authoritative verification: `results/oob_review_validation.json`; rendering review: `results/oob_visual_review.json`. Earlier numeric-only verification omitted balanced-OOB coverage and is superseded on that point. Earlier checklists/answers are preserved in `results/archive_before_final_bounded/`.

- [x] Original Homework6 and local course requirements read; no applicable AGENTS.md found in project/ancestor paths. No prescribed font/margin rule found.
- [x] Existing worktree changes, original CSVs, 25 historical experiment/data artifacts and schema-1 cache preserved; no reset, commit, push or platform submission.
- [x] Q1(a): official 60000/16000 datasets; preamble/header, na, labels, class counts, numeric types, 170-column order checked.
- [x] Q1(b)(i): missing-data methods/rationale; separate test-own medians per footnote; distinct train-only sensitivity; no pooling/test-label preprocessing.
- [x] Q1(b)(ii): all170 CVs with ddof=1; zero-mean/all-missing handling; top13 only for display.
- [x] Q1(b)(iii): all170 correlation matrix displayed.
- [x] Q1(b)(iv): all78 unordered top13 pairs appear once directly in PDF; full high-resolution matrix in notebook, legend/features/log1p/sample size/seed stated; scatter significance answered.
- [x] Q1(b)(v): both splits' counts and imbalance discussed.
- [x] Q1(c–d): both forests fit on all170 features; train/test CM/ROC/AUC/error; OOB comparison; balanced weights cited; minority metrics/AP and error tradeoff.
- [x] Q1(e): complete 7×5 plain CV, AP/tie rule, actual CV errors, all-row final fit and train/test outputs. Unrestricted alpha=0; alpha0 explicitly means no L1 penalty.
- [x] Q1(f): complete 7×5 SMOTE CV; fold-local imputer/scaler/SMOTE; original validation/test distribution; final118000 rows; training metrics on original60000; times/counts/settings and comparison.
- [x] Positive-alpha supplement: same existing CV restriction/rule selects 0.01; independent full-row no-SMOTE fit/model/predictions/metrics; clearly not original optimum or test-based tuning.
- [x] Q2 all(a–e), Q3 all(a–d), Q4 both aggregation rules, Q5 all(a–h) retained with derivations/figures.
- [x] Q6 allfour statements/editions checked: bootstrap(a–h), ridge(a–e), tree(a–b), circle(a–d); actual computations/plots retained.
- [x] Full isolated clean kernel: initially notebook + two CSVs only; 29 code cells, zero errors; 75 actual fits (2+70+2+1).
- [x] Quick isolated clean kernel: 29 code cells, zero errors; fresh schema2 cache validated; no fit. Missing/mismatch rejection exercised; no re-signing historical artifacts.
- [x] Notebook answers, timings, metrics, table differences and mode scope draw on current run; delivered outputs are the new coverage-aware quick execution; historical full outputs are archived.
- [x] PDF/Notebook/README alpha, model metrics, CV, CM, params and execution provenance cross-checked. Fresh/historical comparison saved without altering history.
- [x] Formal PDF 20 pages including necessary code, all answers, outputs and references; body10.4pt/code8.7pt unchanged; every page rendered and reviewed (see visual record).
- [ ] Browser Print preview directly inspected: not performed; programmatic count and PDF rendering are separate checks.
- [ ] Instructor clarification of genuine node-level logistic model tree versus standard gbtree leaf L1: still unresolved and accurately disclosed; positive alpha does not resolve it.

## Coverage-aware OOB correction

- [x] Every fitted tree's true bootstrap indices counted; root unique counts and treewise OOB probability reconstruction verified.
- [x] Plain zero-vote0, valid60000/60000; balanced zero-vote1000, valid59000/60000, positive coverage0/1000 explicitly reported.
- [x] All-zero/NaN/nonfinite/other invalid probability rows checked; warning attributed to balanced weighted sampling in installed1.9.1.
- [x] Historical1199/60000 retained; valid199/59000 explicitly negative-only, never treated as overall OOB or compared with full test error.
- [x] One separate500-tree balanced_subsample fit; complete both-class OOB coverage; valid457/60000 compared with official test error. Train/test CM/ROC/AUC/error and minority metrics included.
- [x] No original RF/XGB/SMOTE/CV refit; original cache/manifests/historical model hashes unchanged.
- [x] Independent OOB fingerprint/provenance, one-line post-fit-check migration audit, no-fit revalidation and cache rejection checks saved.
- [x] Current self-contained notebook clean-kernel quick execution:31 code cells, zero errors; original OOB scope claims corrected.
- [x] Final PDF20pages, font sizes unchanged, every page rendered and reviewed; all original and extra answers retained.
