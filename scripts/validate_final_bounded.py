"""Verify the final standalone full/quick runs, supplementary fit and paper evidence."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,tempfile,shutil,os
import nbformat,numpy as np,pandas as pd,pymupdf
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def validate():
    if (ROOT/'results/oob_quick_execution.json').exists():
        from validate_oob_review import validate as validate_current
        return validate_current()
    core=read(ROOT/'results/final_submission_core_audit.json')
    assert core['all_passed'] and core['core_source_sha256']==digest(ROOT/'scripts/submission_core.py')
    full=read(ROOT/'results/final_full_execution.json')
    quick=read(ROOT/'results/final_quick_execution.json')
    assert full['notebook_sha256']==digest(ROOT/'DSCI552_Homework6.ipynb')
    assert full['clean_kernel'] and quick['clean_kernel']
    assert full['errors']==quick['errors']==0
    assert len(full['initial_files'])==3 and full['no_scripts_or_references']
    assert full['training_fit_counts']==dict(forest=2,cv=70,final_xgb=2,positive_alpha_supplement=1)
    assert not quick['aps_models_retrained']
    out=ROOT/full['output_directory'];cache=out/'model_artifacts'
    events=read(cache/'training_events.json');assert len(events)==75
    for name in ['xgb_plain','xgb_smote']:
        ev=[e for e in events if e['model']==name and e['kind']=='cv']
        assert len(ev)==35 and len({(e['alpha'],e['fold']) for e in ev})==35
        assert all(e['fit_rows']==48000 and e['validation_rows']==12000 for e in ev)
        frame=pd.read_csv(out/f'{name}_cv_folds.csv')
        assert len(frame)==35 and frame.alpha.nunique()==7
    summary=pd.read_csv(out/'xgb_plain_cv_summary.csv')
    historical=pd.read_csv(ROOT/'results/xgb_plain_cv_summary.csv')
    choose=lambda df:float(df.sort_values(['mean_ap','alpha'],ascending=[False,True]).iloc[0].alpha)
    pos=read(out/'xgb_positive.json');plain=read(out/'xgb_plain.json')
    assert plain['selected_alpha']==choose(summary)==choose(historical)
    assert pos['selected_alpha']==choose(summary[summary.alpha>0])==choose(historical[historical.alpha>0])
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',4)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert len(codes)==quick['code_cells']
    assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    forbidden={'scripts','verify','audit_submission_core','experiments','figures','content','extra_practice'}
    for c in codes:
        assert '__file__' not in c.source
        for n in ast.walk(ast.parse(c.source)):
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in forbidden
            if isinstance(n,ast.Import):assert not any(x.name.split('.')[0] in forbidden for x in n.names)
    # Avoid side effects of config execution: rewrite only its run-directory creation to a temporary cwd.
    stage=Path(tempfile.mkdtemp(prefix='final_validation_',dir=ROOT/'tmp'))
    old_cwd=Path.cwd()
    try:
        (stage/'Homework 6 Data').mkdir()
        for split in ['training','test']:
            f=f'aps_failure_{split}_set.csv';os.link(ROOT/'Homework 6 Data'/f,stage/'Homework 6 Data'/f)
        os.chdir(stage)
        setup=next(c.source for c in codes if 'configuration' in c.metadata.get('tags',[]))
        engine=next(c.source for c in codes if 'model-engine' in c.metadata.get('tags',[]))
        ns={};exec(setup,ns);ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]));exec(engine,ns)
        signature=ns['cache_signature']();manifest=ns['validate_cache'](cache,signature)
        assert signature['schema']==2 and manifest['provenance']['trained_from_raw_this_run']
        (x,y),(t,yt),_=ns['load_data']()
        prediction_check,_=ns['verify_results'](cache,x,y,t,yt,'quick')
        negative={}
        for key in ['data','config','model_code_sha256','versions']:
            changed=copy.deepcopy(signature)
            if isinstance(changed[key],dict):changed[key]['deliberately_invalid']=True
            else:changed[key]='invalid'
            try:ns['validate_cache'](cache,changed)
            except ValueError:negative[key+'_mismatch_rejected']=True
            else:raise AssertionError(key)
        try:ns['validate_cache'](stage/'absent',signature)
        except FileNotFoundError:negative['absent_cache_rejected']=True
        else:raise AssertionError('missing cache')
        damaged=stage/'damaged';damaged.mkdir();shutil.copy2(cache/'manifest.json',damaged/'manifest.json')
        try:ns['validate_cache'](damaged,signature)
        except ValueError:negative['missing_artifact_rejected']=True
        else:raise AssertionError('missing artifact')
        for f in cache.iterdir():
            if f.name=='manifest.json':continue
            if f.name=='xgb_positive.json':shutil.copy2(f,damaged/f.name)
            else:os.link(f,damaged/f.name)
        with (damaged/'xgb_positive.json').open('a') as stream:stream.write(' ')
        try:ns['validate_cache'](damaged,signature)
        except ValueError:negative['altered_artifact_rejected']=True
        else:raise AssertionError('altered artifact')
    finally:os.chdir(old_cwd);shutil.rmtree(stage)
    for mode,record in [('full',full),('quick',quick)]:
        run=ROOT/record['output_directory'];v=read(run/'notebook_validation.json')
        assert v['mode']==mode and v['trained_from_raw_this_run']==(mode=='full')
        assert ('not refitted' in v['validation_scope'])==(mode=='quick')
        cp=read(run/'scatter_pair_coverage.json');top=read(run/'top13.json')
        from itertools import combinations
        assert cp['count']==78 and {frozenset(p) for p in cp['pairs']}=={frozenset(p) for p in combinations(top,2)}
        for name in ns['MODEL_NAMES']:
            assert read(run/f'{name}.json')==read(out/f'{name}.json')
        assert read(run/'extra_practice_results.json')['all_passed']
    preserved=read(ROOT/'results/archive_before_final_bounded/historical_artifacts_sha256.json')
    for path,sha in preserved.items():assert digest(ROOT/path)==sha,path
    pdf=pymupdf.open(ROOT/'DSCI552_Homework6_Report.pdf');assert len(pdf)<=20
    layout=read(ROOT/'results/report_layout.json');assert layout['body_font_pt']==10.4 and layout['code_font_pt']==8.7
    from code_report_content import report_sections
    norm=lambda s:' '.join(s.split());sections=report_sections();printed=0
    assert len(pdf)==len(sections)==20
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
    txt='\n'.join(p.get_text() for p in pdf)
    for token in ['alpha > 0','ddof=1','6.8.4','5.4.2','8.4.4','9.7.2','logistic model tree']:
        assert token in txt,token
    assert len(list((ROOT/'results/report_pages').glob('page_*.png')))==20
    visual=read(ROOT/'results/final_visual_review.json')
    assert visual['pdf_sha256']==digest(ROOT/'DSCI552_Homework6_Report.pdf')
    assert visual['reviewed_pages']==list(range(1,21))
    assert not visual['clipping_overlap_blank_pages_or_formula_corruption_found']
    docs=(ROOT/'README.md').read_text()
    assert f"alpha={pos['selected_alpha']:g}" in docs and f"{len(pdf)} actual PDF pages" in docs
    for name in ns['MODEL_NAMES']:
        d=read(out/f'{name}.json')
        assert f"{d['test']['error']:.6f}" in docs and f"{d['test']['auc']:.6f}" in docs
    comparison=read(ROOT/'results/final_full_history_comparison.json')
    result=dict(all_passed=True,full_mode_executed=True,quick_mode_executed=True,
        full_initial_files=full['initial_files'],full_fits=full['training_fit_counts'],
        full_code_cells=full['code_cells'],quick_code_cells=quick['code_cells'],kernel_errors=0,
        notebook_self_contained=True,cache_schema=2,cache_from_fresh_full_run=True,
        cache_negative_tests=negative,prediction_check=prediction_check,
        historical_artifacts_unchanged=len(preserved),historical_comparison=comparison,
        positive_alpha=pos['selected_alpha'],original_alpha=plain['selected_alpha'],
        positive_metrics=pos,all_78_pairs_present=True,pdf_pages=len(pdf),body_font_pt=10.4,code_font_pt=8.7,
        all_pdf_code_lines_and_numeric_tables_verified=True,core_code_lines=printed,
        visual_qa=visual,
        browser_print_preview_checked=False,
        remaining_issue='Instructor clarification of node-level logistic model tree wording.')
    (ROOT/'results/final_bounded_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['positive_metrics','prediction_check','historical_comparison']},indent=2))
    return result
if __name__=='__main__':validate()
