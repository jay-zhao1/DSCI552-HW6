"""Audit saved forests first, then fit only the explicitly labeled supplement."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nbformat,json
ROOT=Path(__file__).resolve().parents[1]
def run(mode='verify',artifacts='results/oob_review/artifacts_v2'):
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',4);ns={}
    setup=next(c.source for c in nb.cells if 'configuration' in c.metadata.get('tags',[]))
    engine=next(c.source for c in nb.cells if 'model-engine' in c.metadata.get('tags',[]))
    exec(setup,ns);ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[engine]));exec(engine,ns)
    model_dir=ROOT/'results/final_full_run/model_artifacts'
    ns['validate_cache'](model_dir,ns['cache_signature']())
    audit=(ROOT/'scripts/oob_audit_source.py').read_text().strip()
    ns['get_ipython']=lambda:SimpleNamespace(history_manager=SimpleNamespace(input_hist_raw=[audit]));exec(audit,ns)
    (x,y),(t,yt),_=ns['load_data']()
    out=ROOT/'results/oob_review';out.mkdir(exist_ok=True)
    fit=ns['RandomForestClassifier'].fit;fit_calls=[]
    def guarded(self,*a,**kw):
        assert self.class_weight=='balanced_subsample'
        fit_calls.append(self.get_params());return fit(self,*a,**kw)
    with patch.object(ns['XGBClassifier'],'fit',side_effect=AssertionError('XGB fit forbidden')),patch.object(ns['RandomForestClassifier'],'fit',guarded):
        stats,score=ns['run_oob_review'](model_dir,x,y,t,yt,out,mode,ROOT/artifacts)
    assert len(fit_calls)==(1 if mode=='fit_supplement' else 0)
    print(json.dumps(dict(audits=stats,supplement=score,fit_calls=len(fit_calls)),indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['verify','fit_supplement'],default='verify')
    parser.add_argument('--artifacts',default='results/oob_review/artifacts_v2')
    args=parser.parse_args();run(args.mode,args.artifacts)
