"""One-time migration of original fitted artifacts into a strictly signed notebook cache.
The notebook does not import this helper. No APS estimator is fitted here.
"""
from pathlib import Path
from types import SimpleNamespace
import json,shutil,hashlib
from unittest.mock import patch
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def prepare():
    raise RuntimeError('Historical schema-1 migration is retired. For schema 2 execute the standalone full mode; never re-sign old artifacts.')
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4)
    setup=next(c.source for c in nb.cells if 'configuration' in c.metadata.get('tags',[]))
    engine=next(c.source for c in nb.cells if 'model-engine' in c.metadata.get('tags',[]))
    ns={};exec(setup,ns)
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]))
    exec(engine,ns)
    (x,y),(t,yt),_=ns['load_data']()
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    with patch.object(RandomForestClassifier,'fit',side_effect=AssertionError('No RF fitting')), \
         patch.object(XGBClassifier,'fit',side_effect=AssertionError('No XGB fitting')):
        checked,frame=ns['verify_results'](ROOT/'results',x,y,t,yt)
    out=ROOT/'results/notebook_cache_v1'
    if out.exists():
        # Never silently re-sign an altered cache.
        ns['validate_cache'](out,ns['cache_signature']())
        print('Matching signed cache already exists; no changes.');return
    out.mkdir()
    names=['cv_folds.npz']
    for n in ns['MODEL_NAMES']:names += [n+'.joblib',n+'.json',n+'_predictions.npz']
    for n in ['xgb_plain','xgb_smote']:names += [n+'_cv_folds.csv',n+'_cv_summary.csv']
    for f in names:shutil.copy2(ROOT/'results'/f,out/f)
    legacy_source=ROOT/'scripts/experiments.py'
    provenance=dict(mode='validated historical full-data experiment',
        trained_from_raw_this_run=False,baseline_commit='4baf619',
        migration_validation=checked['validation_scope'],
        original_model_source_sha256=hashlib.sha256(legacy_source.read_bytes()).hexdigest(),
        parameter_and_preprocessing_equivalence='All 4 parameter dictionaries; training imputer/scaler statistics; 5 fold memberships; original train/test/sensitivity predictions reproduced.',
        limitation='Historical CV fits are not rerun; their saved fold-score summaries and alpha selection are verified.')
    ns['seal_cache'](out,ns['cache_signature'](),provenance)
    ns['validate_cache'](out,ns['cache_signature']())
    (ROOT/'results/notebook_cache_migration.json').write_text(json.dumps(dict(all_passed=True,provenance=provenance,files=len(names)),indent=2))
    print('Created validated cache:',len(names),'artifacts; APS fit methods were forbidden.')
if __name__=='__main__':prepare()
