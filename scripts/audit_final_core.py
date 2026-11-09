"""Validate printed core logic against fresh full-run models, forbidding further APS fits."""
from pathlib import Path
import ast,json,hashlib,runpy
import numpy as np,pandas as pd
from unittest.mock import patch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import load
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1]

def audit():
    with patch.object(RandomForestClassifier,'fit',side_effect=AssertionError('No RF retrain')), \
         patch.object(XGBClassifier,'fit',side_effect=AssertionError('No XGB retrain')):
        ns=runpy.run_path(str(ROOT/'scripts/submission_core.py'))
    R=ROOT/'results/final_full_run';F=R/'figures'
    for key,name in [('fig_a','core_tree_a'),('fig_b','core_tree_b'),('fig','core_circle')]:
        ns[key].savefig(F/f'{name}.png',dpi=210,bbox_inches='tight')
        ns[key].savefig(F/f'{name}.pdf',bbox_inches='tight')
    oldcv=pd.read_csv(R/'feature_cv.csv',index_col=0)
    assert np.allclose(ns['cv'],oldcv.cv,equal_nan=True)
    assert ns['top']==json.loads((R/'top13.json').read_text())
    folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=552).split(ns['X'],ns['y']))
    saved=np.load(R/'cv_folds.npz')
    for i,(_,va) in enumerate(folds,1):assert np.array_equal(va,saved[f'validation_{i}'])
    tree=ast.parse((ROOT/'scripts/submission_core.py').read_text())
    rfcall=next(x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='RandomForestClassifier' and any(k.arg=='class_weight' for k in x.keywords))
    constructors={}
    for name,weight in [('rf_plain',None),('rf_balanced','balanced')]:
        model=eval(compile(ast.Expression(rfcall),'<audited constructor>','eval'),dict(ns,weight=weight))
        original=load(R/'model_artifacts'/f'{name}.joblib')
        assert model.get_params()==original.get_params()
        constructors[name]=True
    for name,alpha in [(n,json.loads((R/f'{n}.json').read_text())['selected_alpha']) for n in ['xgb_plain','xgb_smote','xgb_positive']]:
        model=ns['make_xgb'](alpha);bundle=load(R/'model_artifacts'/f'{name}.joblib')
        a=model.get_params();b=bundle['model'].get_params()
        assert a.keys()==b.keys()
        for k in a:
            assert a[k]==b[k] or (isinstance(a[k],float) and np.isnan(a[k]) and np.isnan(b[k])),k
        scaler=ns['StandardScaler']().fit(ns['A'])
        assert np.allclose(scaler.mean_,bundle['scaler'].mean_) and np.allclose(scaler.scale_,bundle['scaler'].scale_)
        constructors[name]=True
    for name in ['rf_plain','rf_balanced','xgb_plain','xgb_smote','xgb_positive']:
        probabilities=np.load(R/f'{name}_predictions.npz');data=json.loads((R/f'{name}.json').read_text())
        for split,labels in [('train',ns['y']),('test',ns['yt']),('test_train_only',ns['yt'])]:
            calc=ns['evaluate'](labels,probabilities[split]);old=data['test_train_only_imputation' if split=='test_train_only' else split]
            for k in ['error','auc','average_precision','precision','recall','f1']:assert np.isclose(calc[k],old[k],atol=1e-12)
            assert calc['confusion_matrix'].tolist()==old['confusion_matrix']
    oldextra=json.loads((R/'extra_practice_results.json').read_text())
    assert ns['empirical']==oldextra['bootstrap']['empirical_probability']
    assert ns['classes'].tolist()==oldextra['circle']['requested_classes']
    from extra_practice import predict_a,predict_b
    v=np.linspace(-1,3,81);grid=np.array(np.meshgrid(v,v)).reshape(2,-1).T
    for name,pred in [('TA',predict_a),('TB',predict_b)]:
        assert np.allclose([ns['tree_value'](ns[name],x) for x in grid],pred(grid))
    manifest=json.loads((ROOT/'results/archive_before_final_bounded/historical_artifacts_sha256.json').read_text())
    for file,digest in manifest.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    result={'all_passed':True,'aps_fit_calls_forbidden_during_audit':True,'original_parameter_dictionaries_equal':constructors,'cv_fold_indices_equal':True,'imputation_and_scaling_logic_equal':True,'170_feature_cvs_and_top13_equal':True,'all_saved_metrics_recomputed':True,'extra_bootstrap_circle_and_tree_outputs_equal':True,'preserved_experiment_files':len(manifest),'core_source_sha256':hashlib.sha256((ROOT/'scripts/submission_core.py').read_bytes()).hexdigest(),'q3_check':{'ridge':float(ns['ridge']),'lasso_sum':float(ns['tstar']),'objective':float(ns['objectives'][0])}}
    (ROOT/'results/oob_submission_core_audit.json').write_text(json.dumps(result,indent=2))
    ns['plt'].close('all');print(json.dumps(result,indent=2));return result
if __name__=='__main__':audit()
