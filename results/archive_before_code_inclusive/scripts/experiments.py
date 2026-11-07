"""Full-data APS experiments. Each completed stage is checkpointed; no test-driven tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json,time,platform,hashlib,importlib.metadata as md
import numpy as np,pandas as pd,psutil
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,precision_score,recall_score,f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Memory,dump
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results';R.mkdir(exist_ok=True)
SEED=552; ALPHAS=[0.,.001,.01,.1,1.,10.,100.]
def save(name,obj): (R/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def load_data():
 frames=[];info={}
 for split in ['training','test']:
  p=ROOT/'Homework 6 Data'/f'aps_failure_{split}_set.csv'
  with p.open() as f:
   for header,line in enumerate(f):
    if line.startswith('class,'): break
  df=pd.read_csv(p,skiprows=header,na_values=['na'])
  assert df['class'].isin(['pos','neg']).all()
  X=df.drop(columns='class').astype(float);y=df['class'].map({'pos':1,'neg':0}).to_numpy()
  assert X.shape[1]==170 and all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
  info[split]={'rows':len(X),'features':X.shape[1],'header_line_1based':header+1,'counts':{'neg':int((y==0).sum()),'pos':int((y==1).sum())},'missing_cells':int(X.isna().sum().sum()),'all_missing_columns':X.columns[X.isna().all()].tolist(),'dtype':str(X.dtypes.iloc[0]),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  frames.append((X,y))
 assert frames[0][0].columns.equals(frames[1][0].columns)
 assert len(frames[0][0])==60000 and len(frames[1][0])==16000
 save('data_audit.json',info);return *frames,info

def metrics(y,p):
 pred=(p>=.5).astype(int)
 return {'n':len(y),'error':float(np.mean(pred!=y)),'auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred)),'f1':float(f1_score(y,pred)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def xgb(alpha):return XGBClassifier(booster='gbtree',objective='binary:logistic',reg_alpha=alpha,reg_lambda=1,n_estimators=250,max_depth=4,learning_rate=.08,subsample=1,colsample_bytree=1,min_child_weight=1,tree_method='hist',max_bin=256,n_jobs=4,random_state=SEED,eval_metric='logloss',base_score=.5)

def run():
 start=time.perf_counter()
 save('environment.json',{'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),'ram_gib':psutil.virtual_memory().total/2**30,'available_ram_gib':psutil.virtual_memory().available/2**30,'versions':{m:md.version(m) for m in ['numpy','pandas','scikit-learn','xgboost','imbalanced-learn','matplotlib','nbformat','nbclient','reportlab']},'seed':SEED,'threads':4})
 (X,y),(T,yt),audit=load_data()
 imp=SimpleImputer(strategy='median',keep_empty_features=True); A=imp.fit_transform(X);B=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(T);Bt=imp.transform(T)
 assert np.isfinite(A).all() and np.isfinite(B).all()
 D=pd.DataFrame(A,columns=X.columns);mean=D.mean();sd=D.std(ddof=1);cv=sd/mean.replace(0,np.nan)
 tab=pd.DataFrame({'mean':mean,'sample_sd_ddof1':sd,'cv':cv,'missing_fraction':X.isna().mean(),'zero_mean':mean==0,'all_missing':X.isna().all()});tab.to_csv(R/'feature_cv.csv');D.corr().to_csv(R/'correlation.csv');save('top13.json',cv.nlargest(13).index.tolist())
 for balanced,name in [(False,'rf_plain'),(True,'rf_balanced')]:
  if (R/f'{name}.json').exists():continue
  t=time.perf_counter(); model=RandomForestClassifier(n_estimators=500,max_features='sqrt',bootstrap=True,oob_score=True,class_weight='balanced' if balanced else None,n_jobs=4,random_state=SEED)
  model.fit(A,y);pt=model.predict_proba(A)[:,1];pv=model.predict_proba(B)[:,1];ps=model.predict_proba(Bt)[:,1]
  out={'train':metrics(y,pt),'test':metrics(yt,pv),'test_train_only_imputation':metrics(yt,ps),'oob_error':float(np.mean((model.oob_decision_function_[:,1]>=.5)!=y)),'fit_and_evaluate_seconds':time.perf_counter()-t,'parameters':{k:('NaN (default missing sentinel)' if isinstance(v,float) and np.isnan(v) else v) for k,v in model.get_params().items()}}
  np.savez_compressed(R/f'{name}_predictions.npz',train=pt,test=pv,test_train_only=ps);dump(model,R/f'{name}.joblib',compress=3);save(f'{name}.json',out);print(name,out,flush=True)
 folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y))
 np.savez_compressed(R/'cv_folds.npz',**{f'validation_{i+1}':v for i,(_,v) in enumerate(folds)})
 cache=Memory(ROOT/'tmp'/'pipeline_cache',verbose=0)
 for smote,name in [(False,'xgb_plain'),(True,'xgb_smote')]:
  rows=[];pfile=R/f'{name}_cv_folds.csv'
  if pfile.exists():rows=pd.read_csv(pfile).to_dict('records')
  for alpha in ALPHAS:
   for fold,(tr,va) in enumerate(folds,1):
    if any(r['alpha']==alpha and r['fold']==fold for r in rows):continue
    t=time.perf_counter()
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler())]
    if smote:steps.append(('smote',SMOTE(random_state=SEED,k_neighbors=5,sampling_strategy=1.0)))
    pipe=Pipeline(steps+[('estimator',xgb(alpha))],memory=cache)
    pipe.fit(X.iloc[tr],y[tr]); p=pipe.predict_proba(X.iloc[va])[:,1]
    m=metrics(y[va],p);row={'alpha':alpha,'fold':fold,**{k:v for k,v in m.items() if k not in ['confusion_matrix']},'seconds':time.perf_counter()-t,'original_fold_n':len(tr),'resampled_fold_n':94400 if smote else len(tr)};rows.append(row);pd.DataFrame(rows).to_csv(pfile,index=False);print(name,'alpha',alpha,'fold',fold,'AP',row['average_precision'],'seconds',row['seconds'],flush=True)
  df=pd.DataFrame(rows);summary=df.groupby('alpha').agg(mean_ap=('average_precision','mean'),sd_ap=('average_precision','std'),mean_error=('error','mean'),sd_error=('error','std'),mean_auc=('auc','mean'),seconds=('seconds','sum')).reset_index();summary.to_csv(R/f'{name}_cv_summary.csv',index=False)
  best=float(summary.sort_values(['mean_ap','alpha'],ascending=[False,True]).iloc[0]['alpha'])
  if (R/f'{name}.json').exists():continue
  t=time.perf_counter();scaler=StandardScaler();As=scaler.fit_transform(A);Bs=scaler.transform(B);Bts=scaler.transform(Bt)
  tsm=time.perf_counter()
  if smote: Ar,yr=SMOTE(random_state=SEED,k_neighbors=5).fit_resample(As,y)
  else:Ar,yr=As,y
  smtime=time.perf_counter()-tsm;model=xgb(best);model.fit(Ar,yr)
  pt=model.predict_proba(As)[:,1];pv=model.predict_proba(Bs)[:,1];ps=model.predict_proba(Bts)[:,1]
  out={'selected_alpha':best,'selection_metric':'mean validation average precision (maximize; ties: smaller alpha)','cv':summary[summary.alpha==best].iloc[0].to_dict(),'train':metrics(y,pt),'test':metrics(yt,pv),'test_train_only_imputation':metrics(yt,ps),'resampling_seconds':smtime,'final_fit_and_evaluate_seconds':time.perf_counter()-t,'resampled_counts':{'neg':int((yr==0).sum()),'pos':int((yr==1).sum())},'parameters':{k:('NaN (default missing sentinel)' if isinstance(v,float) and np.isnan(v) else v) for k,v in model.get_params().items()}}
  np.savez_compressed(R/f'{name}_predictions.npz',train=pt,test=pv,test_train_only=ps);dump({'imputer':imp,'scaler':scaler,'model':model},R/f'{name}.joblib',compress=3);save(f'{name}.json',out);print(name,out,flush=True)
 save('run_complete.json',{'seconds_this_invocation':time.perf_counter()-start,'completed_models':['rf_plain','rf_balanced','xgb_plain','xgb_smote']})
if __name__=='__main__':run()
