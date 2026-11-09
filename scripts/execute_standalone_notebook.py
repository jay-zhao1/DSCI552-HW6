"""Execute full/quick in a fresh kernel, with strictly bounded input inventory."""
from pathlib import Path
import os,json,tempfile,shutil,time,argparse,hashlib
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]

def execute(mode='quick',revision='final'):
    assert mode in ['full','quick']
    assert revision.replace('_','').isalnum()
    destination=ROOT/'results'/f'{revision}_{mode}_run'
    assert not destination.exists(), f'Preserve existing run: {destination}'
    stage=Path(tempfile.mkdtemp(prefix=f'final_{mode}_',dir=ROOT/'tmp'))
    (stage/'Homework 6 Data').mkdir()
    for split in ['training','test']:
        name=f'aps_failure_{split}_set.csv'
        os.link(ROOT/'Homework 6 Data'/name,stage/'Homework 6 Data'/name)
    path=ROOT/'DSCI552_Homework6.ipynb';nb=nbformat.read(path,as_version=4)
    nbformat.write(nb,stage/path.name)
    if mode=='quick':
        cache=stage/'results/final_full_run/model_artifacts';cache.mkdir(parents=True)
        for f in (ROOT/'results/final_full_run/model_artifacts').iterdir():
            if f.is_file():os.link(f,cache/f.name)
        if any('oob-audit-engine' in c.metadata.get('tags',[]) for c in nb.cells):
            source=ROOT/'results/oob_review/artifacts_v2'
            target=stage/'results/oob_review/artifacts_v2';target.mkdir(parents=True)
            for f in source.iterdir():
                if f.is_file():os.link(f,target/f.name)
    initial_files=sorted(str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file())
    if mode=='full': assert len(initial_files)==3
    start=time.perf_counter();old_mode=os.environ.get('DSCI552_MODE')
    old_cache=os.environ.pop('DSCI552_CACHE_DIR',None)
    os.environ['DSCI552_MODE']=mode
    try:
        NotebookClient(nb,timeout=3600,kernel_name='python3',
            resources={'metadata':{'path':str(stage)}},allow_errors=False).execute()
    except Exception:
        nbformat.write(nb,ROOT/'results'/f'final_{mode}_failed.ipynb')
        raise
    finally:
        if old_mode is None:os.environ.pop('DSCI552_MODE',None)
        else:os.environ['DSCI552_MODE']=old_mode
        if old_cache is not None:os.environ['DSCI552_CACHE_DIR']=old_cache
    runs=list((stage/'results/notebook_runs').glob(f'{mode}_*'));assert len(runs)==1
    run=runs[0];shutil.copytree(run,destination)
    nbformat.write(nb,path)
    nbformat.write(nb,destination/f'Executed_{mode}.ipynb')
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    record=dict(executed=True,mode=mode,clean_kernel=True,
        isolated_directory=str(stage.relative_to(ROOT)),initial_files=initial_files,
        no_scripts_or_references=True,code_cells=len(codes),errors=0,
        aps_models_retrained=(mode=='full'),full_training_mode_executed=(mode=='full'),
        output_directory=str(destination.relative_to(ROOT)),seconds=time.perf_counter()-start,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation=json.loads((destination/'notebook_validation.json').read_text()))
    if mode=='full':
        events=json.loads((destination/'model_artifacts/training_events.json').read_text())
        assert len(events)==75 and sum(e['kind']=='cv' for e in events)==70
        record['training_fit_counts']={k:sum(e['kind']==k for e in events)
            for k in ['forest','cv','final_xgb','positive_alpha_supplement']}
        record['training_fit_counts']['oob_supplemental_forest']=int(record['validation'].get('supplemental_rf_trained_this_run',False))
    (ROOT/'results'/f'{revision}_{mode}_execution.json').write_text(json.dumps(record,indent=2))
    # All outputs now have a durable copy. Remove only this newly created staging dir.
    shutil.rmtree(stage)
    print(json.dumps({k:v for k,v in record.items() if k not in ['validation','initial_files']},indent=2))
    return record
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['full','quick'],default='quick')
    parser.add_argument('--revision',default='final')
    args=parser.parse_args();execute(args.mode,args.revision)
