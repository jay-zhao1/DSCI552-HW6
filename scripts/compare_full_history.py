"""Compare fresh complete fits to preserved historical outputs without changing either."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def compare():
    fresh=ROOT/'results/final_full_run';checks={}
    for name in ['rf_plain','rf_balanced','xgb_plain','xgb_smote']:
        old=json.loads((ROOT/'results'/f'{name}.json').read_text())
        new=json.loads((fresh/f'{name}.json').read_text())
        op=np.load(ROOT/'results'/f'{name}_predictions.npz')
        npred=np.load(fresh/f'{name}_predictions.npz')
        row={'prediction_max_abs_difference':{s:float(np.max(np.abs(op[s]-npred[s]))) for s in op.files},
             'selected_alpha_unchanged':old.get('selected_alpha')==new.get('selected_alpha'),
             'metrics':{}}
        for split in ['train','test','test_train_only_imputation']:
            row['metrics'][split]={k:dict(old=old[split][k],fresh=new[split][k])
                for k in ['error','auc','average_precision','precision','recall','f1']
                if not np.isclose(old[split][k],new[split][k],atol=1e-12,rtol=0)}
            assert old[split]['confusion_matrix']==new[split]['confusion_matrix']
        if name.startswith('xgb'):
            a=pd.read_csv(ROOT/'results'/f'{name}_cv_folds.csv')
            b=pd.read_csv(fresh/f'{name}_cv_folds.csv')
            cols=['alpha','fold','error','auc','average_precision']
            row['cv_max_abs_score_difference']=float(np.max(np.abs(a[cols].to_numpy()-b[cols].to_numpy())))
        checks[name]=row
    preserved=json.loads((ROOT/'results/archive_before_final_bounded/historical_artifacts_sha256.json').read_text())
    for path,digest in preserved.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    report=dict(models=checks,historical_artifacts_unchanged=len(preserved),
        numerical_metrics_agree_to_1e_12=all(not any(d['metrics'].values()) and max(d['prediction_max_abs_difference'].values())<1e-12 for d in checks.values()),
        timings='Fresh fit/CV times are new measurements and are not expected to equal history.')
    (ROOT/'results/final_full_history_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':compare()
