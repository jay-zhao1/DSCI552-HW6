"""Author a self-contained notebook. Build-time sources are embedded, never imported by it."""
from pathlib import Path
import json,ast
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

SETUP='''from pathlib import Path
from datetime import datetime, timezone
import os, json, time, hashlib, shutil, platform, importlib.metadata
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, roc_curve, precision_recall_curve)
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from joblib import dump, load
from matplotlib.patches import Rectangle, Circle
from IPython.display import display, Markdown, Image

# Edit here, or set the environment variables before launching the kernel.
MODE = os.environ.get("DSCI552_MODE", "quick").lower()  # "quick" or "full"
CACHE_DIR = Path(os.environ.get("DSCI552_CACHE_DIR", "results/final_full_run/model_artifacts"))
OOB_CACHE_DIR = Path(os.environ.get("DSCI552_OOB_CACHE_DIR", "results/oob_review/artifacts_v2"))
OOB_MODE = os.environ.get("DSCI552_OOB_MODE", "fit_supplement" if MODE == "full" else "verify")
ROOT = Path.cwd()
DATA_DIR = Path("Homework 6 Data")
assert MODE in {"quick", "full"}
assert (DATA_DIR / "aps_failure_training_set.csv").is_file(), "Run beside the raw data directory."
SEED = 552
MODEL_NAMES = ["rf_plain", "rf_balanced", "xgb_plain", "xgb_smote", "xgb_positive"]
CONFIG = dict(seed=SEED, threshold=.5, alphas=[0., .001, .01, .1, 1., 10., 100.],
    supplement="plain search restricted alpha > 0, same AP and smaller-alpha tie rule",
    features=170, cv_selection="mean average precision; smaller-alpha tie break",
    imputation=dict(strategy="median", keep_empty_features=True,
        main_test="test-own medians", sensitivity="train medians", cv="training-fold only"),
    rf=dict(n_estimators=500, criterion="gini", max_features="sqrt", bootstrap=True,
            oob_score=True, max_depth=None, min_samples_leaf=1, random_state=SEED, n_jobs=4),
    xgb=dict(booster="gbtree", objective="binary:logistic", reg_lambda=1,
        n_estimators=250, max_depth=4, learning_rate=.08, subsample=1,
        colsample_bytree=1, min_child_weight=1, tree_method="hist", max_bin=256,
        n_jobs=4, random_state=SEED, eval_metric="logloss", base_score=.5),
    smote=dict(random_state=SEED, k_neighbors=5, sampling_strategy=1.0),
    cv=dict(n_splits=5, shuffle=True, random_state=SEED))
RUN_ROOT = Path("results/notebook_runs") / (MODE + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
R = RUN_ROOT
F = RUN_ROOT / "figures"
F.mkdir(parents=True, exist_ok=False)
print("Mode:", MODE, "| Output:", RUN_ROOT)
print("QUICK requires the separate matching cache bundle at", CACHE_DIR)
print("Missing cache? Select full mode to train using only the two raw CSVs.")
print("Full fits 2 forests + 70 CV estimators + 3 final XGBs + 1 OOB-safe supplemental forest.")
print("OOB_MODE:", OOB_MODE, "| separate matching OOB sidecar required for verify:", OOB_CACHE_DIR)
'''
EXECUTION='''(X, y), (T, yt), data_audit = load_data()
expected_signature = cache_signature()
if MODE == "full":
    model_dir = RUN_ROOT / "model_artifacts"
    model_dir.mkdir()
    fit_all_from_raw(model_dir, X, y, T, yt)
    seal_cache(model_dir, expected_signature,
        {"mode": "full", "trained_from_raw_this_run": True,
         "note": "Five final models and all 70 CV fits executed from raw data; no previous cache read."})
    provenance = validate_cache(model_dir, expected_signature)
else:
    model_dir = CACHE_DIR
    provenance = validate_cache(model_dir, expected_signature)
    print("Validated cache provenance:", provenance["provenance"])
    print("No original cached model or CV estimator is fitted in quick mode; OOB_MODE controls the separate supplemental forest.")
# Hash checks happen before deserializing any supplied cache model.
for filename in provenance["files"]:
    if not filename.endswith(".joblib"):
        shutil.copy2(model_dir / filename, R / filename)
validation, comparison = verify_results(model_dir, X, y, T, yt, MODE)
validation["validation_scope"] = validation["validation_scope"].replace(
    "metrics and RF OOB verified.", "metrics verified; legacy unmasked OOB arithmetic only, not coverage validation.")
oob_sidecar = (RUN_ROOT / "oob_artifacts") if MODE == "full" else OOB_CACHE_DIR
oob_audits, supplemental_forest = run_oob_review(model_dir, X, y, T, yt, R,
                                                OOB_MODE, oob_sidecar)
validation["oob_coverage_audits"] = oob_audits
validation["oob_scope"] = "Coverage-aware masks and per-class counts supersede legacy unmasked OOB claims."
validation["supplemental_rf_trained_this_run"] = OOB_MODE == "fit_supplement"
validation.update(mode=MODE, trained_from_raw_this_run=(MODE == "full"),
    cache_signature=expected_signature, cache_provenance=provenance["provenance"])
write_json(R / "notebook_validation.json", validation)
comparison.to_csv(R / "model_comparison.csv", index=False)
write_json(R / "data_audit.json", data_audit)
write_json(R / "environment.json", dict(python=platform.python_version(),
    platform=platform.platform(), seed=SEED, threads=4,
    versions={p: importlib.metadata.version(p) for p in
        ["numpy", "pandas", "scikit-learn", "xgboost", "imbalanced-learn", "matplotlib", "joblib"]}))
imputer, A, B, Bt = prepare(X, T)
D = pd.DataFrame(A, columns=X.columns)
mean, sd = D.mean(), D.std(ddof=1)
cv = sd / mean.replace(0, np.nan)
feature_table = pd.DataFrame(dict(mean=mean, sample_sd_ddof1=sd, cv=cv,
    missing_fraction=X.isna().mean(), zero_mean=mean.eq(0), all_missing=X.isna().all()))
feature_table.to_csv(R / "feature_cv.csv")
D.corr().to_csv(R / "correlation.csv")
write_json(R / "top13.json", cv.nlargest(13).index.tolist())
assert len(cv) == 170 and len(cv.nlargest(13)) == 13
print("Execution status:", MODE, "| original five models newly trained:", MODE == "full")
print("Supplemental forest newly trained:", OOB_MODE == "fit_supplement")
print("Final-model verification:", validation["validation_scope"])
display(pd.DataFrame(data_audit).T)
display(comparison[comparison.split == "test"])
'''
DISPLAY='''def show_answer_section(section):
    title, blocks = section
    display(Markdown("## " + title))
    for block in blocks:
        if block[0] in ["p", "h"]:
            text = block[1].replace("<b>", "**").replace("</b>", "**")
            display(Markdown(("### " if block[0] == "h" else "") + text))
        elif block[0] == "math":
            display(Markdown("$$\\n" + block[1] + "\\n$$"))
        elif block[0] == "table":
            head, rows = block[1:]
            table_text = "| " + " | ".join(head) + " |\\n| " + " | ".join(["---"] * len(head)) + " |\\n"
            table_text += "\\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
            display(Markdown(table_text))
        elif block[0] == "image":
            display(Image(filename=str(F / (block[1] + ".png")), width=1000))
'''
# The escaped newlines above must be real Python string escapes in the authored cell.
DISPLAY=DISPLAY.replace('\\\\n','\\n')

def strip_redundant_imports(source):
    lines=source.splitlines()
    remove=set()
    for node in ast.parse(source).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            remove.update(range(node.lineno-1,node.end_lineno))
    return '\n'.join(line for i,line in enumerate(lines) if i not in remove and line != "matplotlib.use('Agg')")

def build():
    nb=nbf.v4.new_notebook();c=[]
    def md(s):c.append(nbf.v4.new_markdown_cell(s))
    def code(s,tag=None):
        cell=nbf.v4.new_code_cell(s)
        if tag:cell.metadata['tags']=[tag]
        c.append(cell)
    md('''# DSCI 552 Homework 6

This notebook is self-contained: it imports no project modules and requires only the declared Python packages and the two raw APS CSV files for a **full** run. Paths are relative, and output directories are created automatically.

**Full training:** set `MODE="full"` below (or `DSCI552_MODE=full`). It creates a fresh run, fits both forests, performs both complete 7-alpha × 5-fold searches, and fits both final XGBoost models, a separate strictly-positive-alpha comparison and one OOB-safe supplemental forest. It reads no previous result or model cache.

**Quick check (default):** requires the separately supplied original `results/final_full_run/model_artifacts/` bundle and coverage-audit `results/oob_review/artifacts_v2/` sidecar. It verifies raw-data hashes, current implementation/configuration/package signatures and every artifact hash before loading models. It reproduces predictions on all original training/test rows, recomputes metrics/OOB and saved CV-score summaries/alpha selection, and regenerates plots/theory calculations. With default OOB_MODE="verify", it **does not fit any estimator**. The separate OOB_MODE="fit_supplement" explicitly trains only the supplemental forest; it is never a full APS rerun. Missing/mismatched caches raise explicit errors; no cache read is called retraining.

Only the submitted PDF is subject to the 20-page limit. The separately attached notebook/source pages are not added. The PDF contains necessary method/code/result evidence; the project source need not all be printed.

**XGBoost limitation:** binary logistic loss/link and L1 tree-leaf regularization do not create logistic regressions at nodes. No explanatory course FAQ/announcement was found among available local material; instructor confirmation remains needed. Standard gbtree results are retained.
''')
    code(SETUP,'configuration')
    md('## Self-contained modeling, full-training and cache-validation logic\n\nAll 170 features enter every estimator. Fold-local preprocessing/resampling, AP-based alpha selection and the 0.5 classification rule match the original experiment. The code-cell fingerprint is part of cache compatibility; changing it invalidates the existing cache.')
    engine=(ROOT/'scripts/notebook_engine_source.py').read_text().strip()
    code(engine,'model-engine')
    md('### Coverage-aware OOB audit\n\nLegacy unmasked OOB numbers are preserved for reproducibility; they are not valid overall estimates when coverage fails. The following code counts actual bootstrap exclusions, reconstructs tree-level OOB probabilities, checks both classes, and separately validates a balanced_subsample supplemental forest. Its independent fingerprinted sidecar leaves the original model/cache unchanged.')
    code((ROOT/'scripts/oob_audit_source.py').read_text().strip(),'oob-audit-engine')
    code(EXECUTION,'execute-mode')
    md('## Plotting and theoretical computations\n\nThese functions write only to the current relative run directory. Statistical plots use training rows; SMOTE training scores always use original, nonsynthetic rows.')
    fig=(ROOT/'scripts/figures.py').read_text().split("if __name__=='__main__':")[0]
    fig=fig.replace('from experiments import ROOT,load_data,SEED\n','').replace("F=ROOT/'figures';F.mkdir(exist_ok=True);R=ROOT/'results'\n",'')
    extra=(ROOT/'scripts/extra_practice.py').read_text().split('\ndef extra_notebook_cells():')[0]
    extra=extra.replace("ROOT=Path(__file__).resolve().parents[1]\n",'').replace("R=ROOT/'results';F=ROOT/'figures';SEED=552\n",'')
    code(strip_redundant_imports(fig)+'\n\n'+strip_redundant_imports(extra),'plot-theory-functions')
    code('preparation()\nmodel_figures()\nextra_results = run_extra()\nassert extra_results["all_passed"]\nprint("All figures and new theory computations generated for mode:", MODE)')
    md('## Answers and actual outputs\n\nModel numbers/timings below come from the selected run or verified historical cache, as stated by MODE. In quick mode, stored fit/CV timings describe the original fitting runs, not this verification run. Full per-feature and per-fold outputs follow the answers.')
    content=(ROOT/'scripts/content.py').read_text()
    content=content.replace("ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'\n",'')
    code(strip_redundant_imports(content)+'\n\n'+DISPLAY,'answer-functions')
    # Formal answers stay individually navigable, and numeric text is evaluated from this run.
    for index in range(17):
        code(f'show_answer_section(pages()[{index}])')
        if index == 6: code('show_answer_section(oob_supplement_answer())')
    code('show_answer_section(positive_answer())')
    md('## Question 6 — Extra Practice\n\nAll four statements were checked separately in the two local R editions. ISLR 6.8.4 denotes first-edition ridge (second-edition equivalent: 6.6.4); the other three references agree across editions, with tree Figure 8.12 becoming 8.14. Each full answer below states the verified source. No textbook file is needed to execute these inline solutions.')
    for index in range(4):code(f'show_answer_section(extra_pages()[{index}])')
    md('## Detailed outputs, mathematical checks and execution scope')
    code('''display(feature_table)
display(pd.read_csv(R / "correlation.csv", index_col=0))
display(Image(filename=str(F / "scatter_all13.png"), width=1500))
display(Image(filename=str(F / "correlation_all_labels.png"), width=1500))
for name in ["xgb_plain", "xgb_smote"]:
    print(name, "all 35 fold scores (fresh in full, stored in quick), mode:", MODE)
    display(pd.read_csv(R / f"{name}_cv_folds.csv"))
x = np.array([1., -1.]); response = np.array([2., -2.]); penalty = 2.
a = x @ x; b = x @ response
ridge = b / (2*a + penalty)
tstar = np.sign(b)*max(abs(b)-penalty/2, 0)/a
objectives = [np.sum((response-x*tstar)**2) + penalty*(abs(w*tstar)+abs((1-w)*tstar))
              for w in np.linspace(0, 1, 11)]
assert np.ptp(objectives) < 1e-12
print("Q3 ridge:", (ridge,ridge), "lasso sum:", tstar, "objective:", objectives[0])
p = np.array([.1,.15,.2,.2,.55,.6,.6,.65,.7,.75])
assert (p > .5).sum() == 6 and np.isclose(p.mean(), .45)
print("Q4 votes / mean:", (p>.5).sum(), p.mean())
print("Q5 independent geometry verification:", read_json(R / "q5_verification.json"))
assert extra_results["all_passed"] and validation["all_passed"]
print("Final execution status:", MODE, "| original five models trained this run:", MODE == "full")
print(validation["validation_scope"])
print("Supplemental forest trained this run:", OOB_MODE == "fit_supplement")
print(validation["oob_scope"])
if MODE == "quick":
    print("Full training was NOT executed. Saved CV estimators are NOT refitted.")
else:
    print("Full training executed: both forests, all 70 CV fits, both selected final XGB models, the positive-alpha supplement and OOB-safe supplemental forest.")
print("Outputs:", RUN_ROOT)
''')
    nb.cells=c;nb.metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3 (DSCI552)'},'language_info':{'name':'python'},'self_contained':True,'training_mode_default':'quick','cache_model_cell_sha256':__import__('hashlib').sha256(engine.encode()).hexdigest()}
    for cell in c:
        if cell.cell_type == 'code': ast.parse(cell.source)
    nbf.validate(nb);nbf.write(nb,ROOT/'DSCI552_Homework6.ipynb')
    print('Authored self-contained notebook:',len(c),'cells')
if __name__=='__main__':build()
