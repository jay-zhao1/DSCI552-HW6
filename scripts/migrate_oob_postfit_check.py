"""Audited one-line post-fit assertion fix; preserve the single actual supplemental fit."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,nbformat,os,ast
ROOT=Path(__file__).resolve().parents[1]
def migrate():
    old=(ROOT/'results/archive_before_oob_review/oob_audit_source_postfit_check_v1.py').read_text().strip()
    new=(ROOT/'scripts/oob_audit_source.py').read_text().strip()
    a="assert fitted.get_params()==expected['supplemental_parameters']"
    b="assert fitted.get_params()==RandomForestClassifier(**expected['supplemental_parameters']).get_params()"
    assert old.count(a)==1 and old.replace(a,b)==new
    # Exactly a post-fit check was changed: fitting, data, hyperparameters and metrics identical.
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',4);ns={}
    exec(next(c.source for c in nb.cells if 'configuration' in c.metadata.get('tags',[])),ns)
    engine=next(c.source for c in nb.cells if 'model-engine' in c.metadata.get('tags',[]))
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]));exec(engine,ns)
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[new]));exec(new,ns)
    origin=ROOT/'results/oob_review/artifacts';dest=ROOT/'results/oob_review/artifacts_v2'
    assert not dest.exists()
    parent=ns['read_json'](origin/'manifest.json');expected=ns['oob_sidecar_signature'](ROOT/'results/final_full_run/model_artifacts')
    oldsig=dict(expected,audit_code_sha256=__import__('hashlib').sha256(old.encode()).hexdigest())
    assert parent['signature']==oldsig
    for file,sha in parent['files'].items():assert ns['sha256_file'](origin/file)==sha
    dest.mkdir()
    for file in parent['files']:os.link(origin/file,dest/file)
    ns['write_json'](dest/'manifest.json',dict(signature=expected,files=parent['files'],
        provenance=dict(parent_manifest_sha256=ns['sha256_file'](origin/'manifest.json'),
            change='One-line post-fit constructor-default comparison fix only; no fitting/preprocessing change.',
            actual_fits=1,training_log='results/oob_review_training.log',
            requires_full_prediction_parameter_and_treewise_oob_revalidation=True)))
    (x,y),(t,yt),_=ns['load_data']()
    with patch.object(ns['RandomForestClassifier'],'fit',side_effect=AssertionError('No fit during migration')),patch.object(ns['XGBClassifier'],'fit',side_effect=AssertionError('No XGB fit')):
        audited,score=ns['run_oob_review'](ROOT/'results/final_full_run/model_artifacts',x,y,t,yt,ROOT/'results/oob_review','verify',dest)
    ns['write_json'](ROOT/'results/oob_review/cache_check_migration.json',dict(all_passed=True,
        source_change_verified_exactly_one_postfit_assertion=True,all_fits_forbidden_during_revalidation=True,
        all_artifact_hashes_identical=True,original_sidecar_preserved=True,
        parent_manifest_sha256=ns['sha256_file'](origin/'manifest.json'),
        parameters_all_predictions_and_actual_bootstrap_oob_recomputed=True))
    print(json.dumps(audited,indent=2))
if __name__=='__main__':migrate()
