"""Check self-containment, cache rejection cases, output consistency and full-mode wiring."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,json,hashlib,tempfile,shutil,os
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def validate():
    if (ROOT/'results/final_quick_execution.json').exists():
        from validate_final_bounded import validate as validate_current
        return validate_current()
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4)
    setup=next(c.source for c in nb.cells if 'configuration' in c.metadata.get('tags',[]))
    engine=next(c.source for c in nb.cells if 'model-engine' in c.metadata.get('tags',[]))
    assert '__file__' not in '\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    forbidden={'scripts','verify','audit_submission_core','experiments','figures','content','extra_practice'}
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in forbidden
            elif isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in forbidden for a in node.names)
    ns={};exec(setup,ns)
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]))
    exec(engine,ns);expected=ns['cache_signature']();cache=ROOT/'results/notebook_cache_v1'
    ns['validate_cache'](cache,expected)
    cases={}
    for key in ['data','config','code','version']:
        altered=copy.deepcopy(expected)
        if key=='data':altered['data']['training']='0'*64
        if key=='config':altered['config']['xgb']['max_depth']=99
        if key=='code':altered['model_code_sha256']='0'*64
        if key=='version':altered['versions']['xgboost']='incompatible-test-version'
        try:ns['validate_cache'](cache,altered)
        except ValueError:cases[key+'_signature_mismatch_rejected']=True
        else:raise AssertionError(key)
    temp=Path(tempfile.mkdtemp(prefix='cache_negative_',dir=ROOT/'tmp'))
    try:
        try:ns['validate_cache'](temp/'absent',expected)
        except FileNotFoundError:cases['missing_cache_explicit_error']=True
        else:raise AssertionError('missing cache accepted')
        missing=temp/'missing';missing.mkdir();shutil.copy2(cache/'manifest.json',missing/'manifest.json')
        try:ns['validate_cache'](missing,expected)
        except ValueError:cases['missing_artifact_rejected']=True
        else:raise AssertionError('missing artifact accepted')
        changed=temp/'changed';changed.mkdir()
        for path in cache.iterdir():
            # The modified file is a private copy, never a hard link to the real cache.
            if path.name=='rf_plain.json':shutil.copy2(path,changed/path.name)
            else:os.link(path,changed/path.name)
        with (changed/'rf_plain.json').open('a') as f:f.write(' ')
        try:ns['validate_cache'](changed,expected)
        except ValueError:cases['altered_artifact_hash_rejected']=True
        else:raise AssertionError('altered artifact accepted')
    finally:shutil.rmtree(temp)
    tree=ast.parse(engine);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit_all_from_raw')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name) and n.func.id in ['load','read_json','validate_cache'] for n in calls)
    assert any(isinstance(n,ast.Assert) for n in fn.body)
    assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in calls)==3
    assert 'pipe.fit(x.iloc[tr], labels[tr])' in engine
    assert 'pipe.predict_proba(x.iloc[va])' in engine
    assert "if smote:" in engine and 'fit_resample(a_s, labels)' in engine
    execution=json.loads((ROOT/'results/standalone_notebook_execution.json').read_text())
    assert execution['clean_kernel'] and execution['errors']==0 and execution['mode']=='quick'
    codes=[c for c in nb.cells if c.cell_type=='code'];assert len(codes)==execution['code_cells']
    assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    out=ROOT/execution['output_directory']
    for name in ns['MODEL_NAMES']:
        before=json.loads((ROOT/'results'/f'{name}.json').read_text())
        after=json.loads((out/f'{name}.json').read_text())
        for field in ['train','test','test_train_only_imputation']:assert before[field]==after[field]
    fresh_extra=json.loads((out/'extra_practice_results.json').read_text())
    old_extra=json.loads((ROOT/'results/extra_practice_results.json').read_text())
    for field in ['bootstrap','tree','circle']:assert fresh_extra[field]==old_extra[field]
    # Current original data/model/result hashes, independent of the new cache manifest.
    preserved=json.loads((ROOT/'results/standalone_preservation.json').read_text())
    for file,digest in preserved.items():assert ns['sha256_file'](ROOT/file)==digest,file
    result=dict(all_passed=True,self_contained_no_project_imports=True,
        clean_kernel_quick_execution=True,executed_code_cells=len(codes),errors=0,
        cache_negative_tests=cases,full_mode_has_no_prior_cache_reads=True,
        full_mode_source_has_fresh_directory_guard=True,full_mode_training_executed=False,
        full_mode_validation_scope='Static control-flow, constructor and fold/preprocessing equivalence checks only; no new APS/CV fitting.',
        saved_original_artifacts_unchanged=len(preserved),fresh_predictions_and_metrics_checked=True,
        theory_results_recomputed_and_equal=True,aps_models_retrained=False)
    (ROOT/'results/standalone_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':validate()
