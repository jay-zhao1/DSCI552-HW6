"""Validate coverage semantics, frozen historical artifacts, standalone outputs and paper."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,hashlib,ast,shutil,tempfile,os
import numpy as np,nbformat,pymupdf
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate():
    record=read(ROOT/'results/oob_quick_execution.json');assert record['clean_kernel'] and record['errors']==0 and not record['aps_models_retrained']
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',4);codes=[c for c in nb.cells if c.cell_type=='code']
    assert sha(ROOT/'DSCI552_Homework6.ipynb')==record['notebook_sha256']
    assert len(codes)==record['code_cells']==31
    assert [c.execution_count for c in codes]==list(range(1,32))
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    forbidden={'scripts','verify','audit_submission_core','experiments','figures','content','extra_practice'}
    for c in codes:
        assert '__file__' not in c.source
        for n in ast.walk(ast.parse(c.source)):
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in forbidden
            if isinstance(n,ast.Import):assert not any(x.name.split('.')[0] in forbidden for x in n.names)
    preserve=read(ROOT/'results/archive_before_oob_review/preservation.json')
    for path,digest in preserve.items():assert sha(ROOT/path)==digest,path
    ns={};exec(next(c.source for c in codes if 'configuration' in c.metadata.get('tags',[])),ns)
    engine=next(c.source for c in codes if 'model-engine' in c.metadata.get('tags',[]))
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]));exec(engine,ns)
    audit=next(c.source for c in codes if 'oob-audit-engine' in c.metadata.get('tags',[]))
    assert audit==(ROOT/'scripts/oob_audit_source.py').read_text().strip()
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[audit]));exec(audit,ns)
    original=ROOT/'results/final_full_run/model_artifacts';sidecar=ROOT/'results/oob_review/artifacts_v2'
    ns['validate_cache'](original,ns['cache_signature']())
    (x,y),(t,yt),_=ns['load_data']()
    temp=Path(tempfile.mkdtemp(prefix='oob_validation_',dir=ROOT/'tmp'))
    try:
        with patch.object(ns['RandomForestClassifier'],'fit',side_effect=AssertionError('RF fit forbidden')),patch.object(ns['XGBClassifier'],'fit',side_effect=AssertionError('XGB fit forbidden')):
            audits,score=ns['run_oob_review'](original,x,y,t,yt,temp,'verify',sidecar)
            # Actual rejection paths; private copies only, never mutate the real cache.
            negative={};parent=read(sidecar/'manifest.json')
            for case in ['signature','artifact']:
                d=temp/case;d.mkdir()
                for f in sidecar.iterdir():
                    if f.name in ['manifest.json','rf_balanced_subsample.json']:shutil.copy2(f,d/f.name)
                    else:os.link(f,d/f.name)
                if case=='signature':
                    damaged=read(d/'manifest.json');damaged['signature']['audit_code_sha256']='0'*64
                    (d/'manifest.json').write_text(json.dumps(damaged))
                else:
                    with (d/'rf_balanced_subsample.json').open('a') as f:f.write(' ')
                try:ns['run_oob_review'](original,x,y,t,yt,temp,'verify',d)
                except ValueError:negative[case+'_mismatch_rejected']=True
                else:raise AssertionError('Corrupt sidecar accepted')
    finally:shutil.rmtree(temp)
    out=ROOT/record['output_directory'];stored=read(out/'oob_coverage.json');assert audits==stored
    for name in ['rf_plain','rf_balanced','rf_balanced_subsample']:
        a=audits[name]
        assert a['treewise_probabilities_reconstructed'] and a['nan_rows']==a['nonfinite_rows']==a['other_invalid_rows']==0
    assert audits['rf_plain']['total']['zero_votes']==0 and audits['rf_plain']['valid_denominator']==60000
    assert audits['rf_balanced']['total']['zero_votes']==1000
    assert audits['rf_balanced']['by_class']['pos']['valid']==0 and audits['rf_balanced']['by_class']['neg']['valid']==59000
    assert audits['rf_balanced']['all_zero_probability_rows']==1000 and not audits['rf_balanced']['full_class_coverage']
    assert audits['rf_balanced']['valid_errors']==199 and audits['rf_balanced']['valid_denominator']==59000
    assert audits['rf_balanced_subsample']['full_class_coverage'] and not score['warnings']
    # Same uniform draws as plain when seeds/settings match, but different within-tree weights.
    assert audits['rf_plain']['bootstrap_indices_sha256']==audits['rf_balanced_subsample']['bootstrap_indices_sha256']
    probability_integrity={}
    for name in ['rf_plain','rf_balanced','rf_balanced_subsample']:
        location=sidecar if name=='rf_balanced_subsample' else original
        fitted=ns['load'](location/f'{name}.joblib');p=fitted.oob_decision_function_
        probability_integrity[name]=dict(out_of_range_rows=int(((p<0)|(p>1)).any(axis=1).sum()),
            nan_rows=int(np.isnan(p).any(axis=1).sum()),inf_rows=int(np.isinf(p).any(axis=1).sum()))
        assert all(v==0 for v in probability_integrity[name].values())
    model=ns['load'](original/'rf_balanced.joblib');prob=model._sample_weight/model._sample_weight.sum()
    theory={}
    for c,label in [(0,'neg'),(1,'pos')]:
        p=float(prob[y==c][0]);theory[label]=dict(draw_probability=p,expected_draws_per_tree=p*model._n_samples_bootstrap,
            expected_oob_votes_per_row=len(model.estimators_)*float(np.exp(model._n_samples_bootstrap*np.log1p(-p))))
    for name in ns['MODEL_NAMES']:
        old=read(ROOT/'results/final_full_run'/f'{name}.json');new=read(out/f'{name}.json');assert old==new
    assert read(out/'extra_practice_results.json')['all_passed']
    v=read(out/'notebook_validation.json');assert v['oob_scope'].startswith('Coverage-aware')
    assert not v['supplemental_rf_trained_this_run']
    assert read(out/'oob_review_execution.json')['new_fits']==0
    migration=read(ROOT/'results/oob_review/cache_check_migration.json');assert migration['all_passed'] and migration['all_fits_forbidden_during_revalidation']
    # Explicitly record that the initial fit succeeded before a post-fit assertion failure.
    assert 'AssertionError' in (ROOT/'results/oob_review_training.log').read_text()
    pdf=pymupdf.open(ROOT/'DSCI552_Homework6_Report.pdf');assert len(pdf)==20
    layout=read(ROOT/'results/report_layout.json');assert layout['body_font_pt']==10.4 and layout['code_font_pt']==8.7
    from code_report_content import report_sections
    sections=report_sections();norm=lambda s:' '.join(s.split());printed=0
    for page,(title,blocks) in zip(pdf,sections):
        assert title in page.get_text()
        for b in blocks:
            if b[0] in ['code','output']:
                for line in b[1].splitlines():
                    if line.strip():assert norm(line) in norm(page.get_text()),line
                if b[0]=='code':printed+=len(b[1].splitlines())
            if b[0]=='table':
                for row in b[2]:
                    for value in row:assert norm(str(value)) in norm(page.get_text()),(title,value)
        assert '\ufffd' not in page.get_text()
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox'];assert 0<=x0<x1<=612.1 and 0<=y0<y1<=792.1
    assert printed==layout['core_code_lines_printed']
    text='\n'.join(p.get_text() for p in pdf)
    for token in ['NEGATIVE-ONLY','1199/60000','199/59000','balanced_subsample','0.00761667','0.010625','6.8.4','5.4.2','8.4.4','9.7.2']:
        assert token in text,token
    coverage=read(out/'scatter_pair_coverage.json');assert coverage['count']==78 and coverage['complete']
    core=read(ROOT/'results/oob_submission_core_audit.json');assert core['all_passed'] and core['core_source_sha256']==sha(ROOT/'scripts/submission_core.py')
    docs=(ROOT/'README.md').read_text()
    for token in ['NEGATIVES ONLY','0.003372881','0.007616667','31','artifacts_v2']:
        assert token in docs,token
    visual=read(ROOT/'results/oob_visual_review.json');assert visual['pdf_sha256']==sha(ROOT/'DSCI552_Homework6_Report.pdf') and visual['reviewed_pages']==list(range(1,21))
    result=dict(all_passed=True,sklearn_version=ns['importlib'].metadata.version('scikit-learn'),
        original_model_cache_fingerprint_unchanged=True,preserved_files=len(preserve),
        original_forest_and_xgb_smote_cv_refits=0,new_fits_this_review=1,
        new_fit='500-tree balanced_subsample forest only',current_quick_fit_count=0,
        clean_kernel_code_cells=31,kernel_errors=0,coverage=audits,probability_integrity=probability_integrity,weighted_sampling_expectation=theory,
        warning_source='rf_balanced: exact installed-source zero-contribution trigger; all1000 positives have zero votes',
        balanced_overall_oob_available=False,historical_unmasked_error_retained=True,
        invalid_overall_gap_conclusion_withdrawn=True,sidecar_negative_tests=negative,
        supplemental_model=score,cache_migration=migration,
        notebook_self_contained=True,all_78_pairs_retained=True,all_q1_to_q6_retained=True,
        pdf_pages=20,body_font_pt=10.4,code_font_pt=8.7,pdf_all_code_lines_and_table_values_verified=True,
        visual_review=visual,browser_print_preview_checked=False)
    (ROOT/'results/oob_review_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['supplemental_model','visual_review','coverage']},indent=2))
    return result
if __name__=='__main__':validate()
