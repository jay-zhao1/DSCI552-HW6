"""Update current submission guidance without rewriting historical execution records."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def update():
    run=read('results/oob_quick_execution.json');out=ROOT/run['output_directory']
    oa=json.loads((out/'oob_coverage.json').read_text());sup=json.loads((out/'rf_balanced_subsample.json').read_text())
    base=(ROOT/'results/archive_before_oob_review/README.md').read_text()
    base=base.replace('**actual full-run outputs**','**fresh coverage-reviewed quick outputs**')
    base=base.replace('The delivered notebook retains the completed **full** execution outputs.', 'The earlier full snapshot is preserved locally as `results/final_full_run/Executed_full.ipynb`; it contains the original balanced-OOB warning. The current notebook contains the newly executed OOB-aware quick outputs.')
    base=base.replace('For **quick** reproduction:','For **quick** reproduction:')
    base=base.replace('This is a local cache/checkpoint bundle; it is not required for full training and is not added to Git by this revision.', 'Also supply the separately fingerprinted `results/oob_review/artifacts_v2/` OOB sidecar. These are local checkpoint bundles, not required for full training and not added to Git by this revision.')
    base=base.replace('**75 fits total**', '**76 fits total in the revised full mode**, including the OOB-safe supplemental forest')
    base=base.replace('and a separate strictly-positive-alpha XGB model:', 'a separate strictly-positive-alpha XGB model, and one balanced_subsample forest:')
    base=base.replace('recalculates metrics/OOB and summarizes saved CV scores/alpha selection.', 'recalculates original metrics and historical OOB arithmetic, then separately reconstructs coverage-aware OOB results and verifies the sixth, supplemental forest. It summarizes saved CV scores/alpha selection.')
    base=base.replace('Cache schema **2** was generated', 'The original-model cache schema **2** was generated')
    base=base.replace('OOB uses the same rule, so exact .5 ties can differ from sklearn argmax.', 'Valid OOB uses the same rule, so exact .5 ties can differ from sklearn argmax. The original balanced unmasked OOB value is INVALID as an overall estimate (details below).')
    base=base.replace('Recomputed all final predictions/metrics, OOB and 70 fold-score summaries;', 'Earlier verification reproduced all final predictions/metrics, unmasked OOB arithmetic and 70 fold-score summaries, but OMITTED OOB coverage validation;')
    base=base.replace('| Q1(c–d): forest code/results/OOB | 5–6 |','| Q1(c–d): original RF code/results; corrected OOB; supplemental RF | 5–8 |')
    base=base.replace('| Q1(e–f): model interpretation, fold-local pipeline, both alpha searches | 7–8 |','| Q1(e–f): interpretation/pipeline/CV; alongside RF coverage and supplement | 7–8 |')
    base=base.replace('`scripts/execute_standalone_notebook.py --mode full`', '`scripts/execute_standalone_notebook.py --mode full --revision new_name`')
    base=base.replace('.venv/bin/python scripts/validate_final_bounded.py', '.venv/bin/python scripts/validate_oob_review.py')
    base=base.replace('.venv/bin/python scripts/update_final_docs.py', '.venv/bin/python scripts/update_oob_docs.py')
    rows=['| Forest / group | Zero OOB votes | Valid / total | Coverage | Votes min/max/mean |', '|---|---:|---:|---:|---:|']
    for name,d in oa.items():
        for group,v in [('all',d['total']),*d['by_class'].items()]:
            rows.append(f"| {name} / {group} | {v['zero_votes']} | {v['valid']} / {v['n']} | {v['valid_fraction']:.6%} | {v['min_votes']}/{v['max_votes']}/{v['mean_votes']:.3f} |")
    correction=f'''\n## OOB correction: current authoritative review

The original `class_weight="balanced"` forest has **zero OOB votes for all 1000 positives**. Installed scikit-learn **1.9.1** uses stored sample/class weights as bootstrap draw probabilities. Each positive has draw probability 1/2000 and expected multiplicity 30 per tree; hence near-zero OOB exclusion probability. The generic "Some inputs do not have OOB scores" warning originates from this forest. It is not explained by merely having too few of the 500 trees. The plain forest covers every row.

The audit uses every fitted tree's `estimators_samples_`, checks reconstructed in-bag unique counts against fitted tree roots, counts exclusions per row, and independently reconstructs every stored OOB probability from tree predictions. It never assumes uniform sampling for the weighted forest. Both original models have no NaN/nonfinite probability rows or other invalid covered rows. Balanced has 1000 [0,0] rows, exactly the uncovered positives; plain has none.

'''+ '\n'.join(rows)+f'''

- Plain: valid OOB error **{oa['rf_plain']['valid_errors']}/{oa['rf_plain']['valid_denominator']}={oa['rf_plain']['valid_error']:.9f}**, a valid both-class comparison to test error 0.007750. Global imputation still makes OOB an approximate internal estimate.
- Original balanced: historical unmasked **1199/60000=0.019983333** is preserved. It treated the 1000 all-zero positive rows as negative predictions. Valid-only error is **{oa['rf_balanced']['valid_errors']}/{oa['rf_balanced']['valid_denominator']}={oa['rf_balanced']['valid_error']:.9f}**, but covers **NEGATIVES ONLY**. It is NOT an overall-distribution OOB estimate and MUST NOT be compared with the full test error 0.0094375. Removing those rows does not solve the missing-positive problem. The previous OOB/test-gap conclusion is withdrawn; original train/test predictions and all XGB/SMOTE results remain unchanged.
- **Separate supplement:** one `class_weight="balanced_subsample"` forest was actually fitted with all other settings unchanged (500 trees, seed552, all170 predictors). In this installed source it uses uniform bootstraps followed by balanced weights inside each tree. It has complete both-class OOB coverage and no OOB warning. Valid OOB={sup['oob']['valid_errors']}/60000={sup['oob']['valid_error']:.9f}; official test error={sup['test']['error']:.9f}, AUC={sup['test']['auc']:.9f}, precision={sup['test']['precision']:.6f}, recall={sup['test']['recall']:.6f}. Test minus OOB={sup['test']['error']-sup['oob']['valid_error']:+.9f}. Train CM={sup['train']['confusion_matrix']}; test CM={sup['test']['confusion_matrix']}. It is a different compensation method, not a repaired original model or a test-selected improvement.

This revision fitted **only that one supplemental forest**, not either original forest, XGB, SMOTE or any of the70 CV estimators. Current Notebook: isolated clean-kernel quick execution, **{run['code_cells']} code cells, zero errors**, {run['seconds']:.1f}s; original cache and independent OOB sidecar validated before loading. Treewise OOB reconstruction and the supplemental train/test/sensitivity probabilities were recomputed. Earlier full execution (75 fits) is historical; revised full mode would additionally fit the OOB-safe forest (76 total), but was not rerun in this bounded review.

The frozen original model-code fingerprint and cache remain byte-for-byte unchanged. Coverage logic has its own executed-cell fingerprint, raw-data/model/cache hashes, actual sklearn source hash, versions and supplemental parameter signature. A one-line post-fit assertion initially compared full estimator defaults to a partial parameter dictionary; this stopped verification after the single fit. The correction changed ONLY that assertion, kept the original sidecar, and required complete parameter/prediction/treewise-OOB revalidation with all fitting forbidden before creating the new sidecar. See `results/oob_review/cache_check_migration.json`; this was not blind re-signing or retraining. The earlier invalid arithmetic is never relabeled as a valid OOB estimate.

For default `MODE="quick"`, also provide `results/oob_review/artifacts_v2/`. Its separate `OOB_MODE="verify"` performs no fit. To train only that supplement when the original matching cache is available, explicitly set `OOB_MODE="fit_supplement"` and a **new** `OOB_CACHE_DIR`; it refuses to overwrite an existing sidecar. In revised full mode, no old cache is needed and the sidecar is created inside the new run. The default missing-cache errors explain both options.

Authoritative review records: `results/oob_review_validation.json`, `oob_quick_execution.json`, `oob_visual_review.json`; per-row votes/masks, class coverage and actual installed source excerpts are in `results/oob_quick_run/`. Report pages5–8 contain corrected code, per-class coverage, historical/valid denominators and supplemental CM/ROC/metrics. Existing Q1–6, all78 scatter pairs, positive-alpha comparison and 20-page format are retained. Body10.4/code8.7pt unchanged.

'''
    # Put the semantic correction before historical run details, not buried after them.
    i=base.index('## What was actually executed');base=base[:i]+correction+base[i:]
    (ROOT/'README.md').write_text(base)
    chk=(ROOT/'results/archive_before_oob_review/CHECKLIST.md').read_text()
    chk=chk.replace('Current authoritative verification: `results/final_bounded_validation.json`; rendering review: `results/final_visual_review.json`.', 'Current authoritative verification: `results/oob_review_validation.json`; rendering review: `results/oob_visual_review.json`. Earlier numeric-only verification omitted balanced-OOB coverage and is superseded on that point.')
    chk=chk.replace('delivered outputs are full-mode outputs.', 'delivered outputs are the new coverage-aware quick execution; historical full outputs are archived.')
    chk+='''\n## Coverage-aware OOB correction\n\n- [x] Every fitted tree's true bootstrap indices counted; root unique counts and treewise OOB probability reconstruction verified.\n- [x] Plain zero-vote0, valid60000/60000; balanced zero-vote1000, valid59000/60000, positive coverage0/1000 explicitly reported.\n- [x] All-zero/NaN/nonfinite/other invalid probability rows checked; warning attributed to balanced weighted sampling in installed1.9.1.\n- [x] Historical1199/60000 retained; valid199/59000 explicitly negative-only, never treated as overall OOB or compared with full test error.\n- [x] One separate500-tree balanced_subsample fit; complete both-class OOB coverage; valid457/60000 compared with official test error. Train/test CM/ROC/AUC/error and minority metrics included.\n- [x] No original RF/XGB/SMOTE/CV refit; original cache/manifests/historical model hashes unchanged.\n- [x] Independent OOB fingerprint/provenance, one-line post-fit-check migration audit, no-fit revalidation and cache rejection checks saved.\n- [x] Current self-contained notebook clean-kernel quick execution:31 code cells, zero errors; original OOB scope claims corrected.\n- [x] Final PDF20pages, font sizes unchanged, every page rendered and reviewed; all original and extra answers retained.\n'''
    (ROOT/'CHECKLIST.md').write_text(chk)
    print('Updated current README/CHECKLIST with coverage limits and actual execution scope')
if __name__=='__main__':update()
