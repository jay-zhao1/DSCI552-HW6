"""Rebuild publication figures from actual saved data and predictions."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.svm import SVC
from experiments import ROOT,load_data,SEED
F=ROOT/'figures';F.mkdir(exist_ok=True);R=ROOT/'results'
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'figure.dpi':130})
def finish(name):
 plt.savefig(F/f'{name}.png',dpi=190,bbox_inches='tight');plt.savefig(F/f'{name}.pdf',bbox_inches='tight');plt.close()
def preparation():
 (X,y),(T,yt),_=load_data();D=pd.DataFrame(SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(X),columns=X.columns)
 top=json.loads((R/'top13.json').read_text());corr=D.corr()
 fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(np.arange(0,170,10),np.arange(1,171,10));ax.set_yticks(np.arange(0,170,10),np.arange(1,171,10));ax.set(xlabel='Feature index in original CSV (1-based)',ylabel='Feature index in original CSV (1-based)',title='All 170 features: training Pearson correlation after median imputation');fig.colorbar(im,ax=ax,label='Pearson r',shrink=.85);fig.tight_layout();finish('correlation_overview')
 fig,ax=plt.subplots(figsize=(28,26));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(170),X.columns,rotation=90,fontsize=7);ax.set_yticks(range(170),X.columns,fontsize=7);fig.colorbar(im,ax=ax,shrink=.7);fig.tight_layout();finish('correlation_all_labels')
 pd.DataFrame({'index_1based':np.arange(1,171),'feature':X.columns}).to_csv(R/'feature_index.csv',index=False)
 rng=np.random.default_rng(SEED);idx=np.concatenate([rng.choice(np.where(y==0)[0],2500,replace=False),np.where(y==1)[0]])
 Z=np.log1p(D.iloc[idx][top]);colors=np.where(y[idx]==1,'#d34c49','#3378a6')
 def matrix(cols,name,size):
  n=len(cols);fig,axes=plt.subplots(n,n,figsize=size)
  for i,a in enumerate(cols):
   for j,b in enumerate(cols):
    ax=axes[i,j]
    if i==j:
     for c,color in [(0,'#3378a6'),(1,'#d34c49')]:ax.hist(Z.loc[:,a].to_numpy()[y[idx]==c],bins=20,histtype='step',color=color,density=True)
    else:ax.scatter(Z[b],Z[a],c=colors,s=1.5,alpha=.25,rasterized=True)
    ax.tick_params(labelsize=6)
    if i==n-1:ax.set_xlabel(b,fontsize=8)
    else:ax.set_xticklabels([])
    if j==0:ax.set_ylabel(a,fontsize=8)
    else:ax.set_yticklabels([])
  fig.suptitle('Training scatter matrix: log(1+x); blue neg, red pos',fontsize=13);fig.tight_layout(rect=[0,0,1,.98]);finish(name)
 matrix(top,'scatter_all13',(22,22));matrix(top[:6],'scatter_top6',(9,9))
 # Paper lower triangle: every unordered pair appears exactly once.
 from itertools import combinations
 fig,axes=plt.subplots(12,12,figsize=(7.4,7.4))
 pairs=[]
 for i in range(1,13):
  for j in range(12):
   ax=axes[i-1,j]
   if j>=i:
    ax.axis('off');continue
   ax.scatter(Z[top[j]],Z[top[i]],c=colors,s=.35,alpha=.22,rasterized=True)
   ax.set_xticks([]);ax.set_yticks([])
   if i==12:ax.set_xlabel(top[j],rotation=90,fontsize=9)
   if j==0:ax.set_ylabel(top[i],rotation=0,ha='right',va='center',fontsize=9)
   pairs.append([top[j],top[i]])
 assert {frozenset(v) for v in pairs}=={frozenset(v) for v in combinations(top,2)}
 assert len(pairs)==78
 from matplotlib.lines import Line2D
 handles=[Line2D([0],[0],marker='o',ls='',color=c,label=l,markersize=4)
          for c,l in [('#3378a6','neg'),('#d34c49','pos')]]
 fig.legend(handles=handles,loc='upper center',ncol=2,fontsize=10,frameon=False)
 fig.subplots_adjust(left=.15,right=.99,bottom=.12,top=.95,wspace=.08,hspace=.08)
 finish('scatter_report')
 (R/'scatter_pair_coverage.json').write_text(json.dumps(dict(count=78,pairs=pairs,
     complete=True,seed=SEED,n_negative=2500,n_positive=1000,transform='log1p'),indent=2))
 fig,axes=plt.subplots(5,3,figsize=(10,10))
 for ax,col in zip(axes.flat,top):
  ax.boxplot([np.log1p(D.loc[y==c,col]) for c in [0,1]],tick_labels=['neg','pos'],showfliers=False,patch_artist=True,boxprops={'facecolor':'#bbd6e6'});ax.set_title(col,fontsize=14);ax.set_ylabel('log(1+x)',fontsize=13);ax.tick_params(labelsize=12)
 for ax in list(axes.flat)[13:]:ax.axis('off')
 fig.suptitle('Top 13 CV features: all training rows; outliers hidden, not removed');fig.tight_layout(rect=[0,0,1,.97]);finish('boxplots13')
 # Q2 pedagogical shapes, not empirical APS estimates.
 s=np.linspace(0,4,300);fig,axes=plt.subplots(1,2,figsize=(9,3.6))
 axes[0].plot(s,1+5*np.exp(-s),label='Training RSS');axes[0].plot(s,2+(s-1.6)**2,label='Typical test RSS');axes[0].legend();axes[0].set(xlabel='L1 constraint radius s',ylabel='Schematic error',title='Relaxing the lasso constraint')
 axes[1].plot(s,3*np.exp(-s),label='Squared bias');axes[1].plot(s,.3+s*.7,label='Variance');axes[1].plot(s,np.ones_like(s),label='Irreducible error');axes[1].legend();axes[1].set(xlabel='L1 constraint radius s',title='Typical bias-variance tradeoff');fig.tight_layout();finish('q2_shapes')
 p=np.array([.1,.15,.2,.2,.55,.6,.6,.65,.7,.75]);fig,ax=plt.subplots(figsize=(9,3.5));ax.bar(np.arange(1,11),p,color=np.where(p>.5,'#d34c49','#4e9c78'));ax.axhline(.5,color='k',ls='--',label='Class threshold 0.5');ax.axhline(p.mean(),color='#7030a0',label=f'Mean = {p.mean():.2f}');ax.set(xlabel='Bootstrap tree',ylabel='P(Red | x)',ylim=(0,1));ax.legend(loc='upper left');fig.tight_layout();finish('q4_votes')
 points=np.array([[3,4],[2,2],[4,4],[1,4],[2,1],[4,3],[4,1]],dtype=float);labels=np.array([1,1,1,1,-1,-1,-1]);svm=SVC(C=1e6,kernel='linear',tol=1e-10).fit(points,labels)
 assert np.allclose(svm.coef_[0],[-2,2],atol=1e-3);assert set(svm.support_)=={1,2,4,5}
 (R/'q5_verification.json').write_text(json.dumps({'coef':svm.coef_.tolist(),'intercept':svm.intercept_.tolist(),'support_observations_1based':(svm.support_+1).tolist(),'margin':float(1/np.linalg.norm(svm.coef_[0]))},indent=2))
 fig,axes=plt.subplots(1,2,figsize=(10,5.7));xx=np.linspace(.5,4.8,200)
 for ax in axes:
  ax.scatter(points[:4,0],points[:4,1],c='#d34c49',s=65,label='Red');ax.scatter(points[4:,0],points[4:,1],c='#3378a6',s=65,label='Blue');
  for i,(a,b) in enumerate(points,1):ax.annotate(str(i),(a+.07,b+.09))
  ax.plot(xx,xx-.5,'k-',label='Optimal: X2 = X1 - 0.5');ax.plot(xx,xx,'k--',lw=1);ax.plot(xx,xx-1,'k--',lw=1);ax.set(xlim=(.5,4.8),ylim=(.3,4.8),xlabel='X1',ylabel='X2');ax.set_aspect('equal');ax.grid(alpha=.15)
 axes[0].scatter(points[[1,2,4,5],0],points[[1,2,4,5],1],s=180,facecolors='none',edgecolors='black',label='Support vectors');axes[0].plot(xx,xx-.2,color='#8d4ca6',ls=':',label='Other separator: X2 = X1 - 0.2');axes[0].set_title('(a)-(g): separator and margins');axes[0].legend(fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2)
 axes[1].scatter([3],[3],c='#3378a6',s=110,marker='X',label='New Blue (3,3)');axes[1].plot([2,4],[2,4],color='#d34c49',alpha=.5,lw=3,label='Red convex-hull segment');axes[1].set_title('(h): intersecting convex hulls');axes[1].legend(fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2);fig.tight_layout();finish('q5_geometry')

def model_figures():
 (X,y),(T,yt),_=load_data()
 for name in ['rf_plain','rf_balanced','xgb_plain','xgb_smote','xgb_positive','rf_balanced_subsample']:
  if not (R/f'{name}.json').exists():continue
  d=json.loads((R/f'{name}.json').read_text());p=np.load(R/f'{name}_predictions.npz');fig,axes=plt.subplots(1,3,figsize=(11,3.4))
  for ax,split in zip(axes[:2],['train','test']):
   cm=np.array(d[split]['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['neg','pos'],yticklabels=['neg','pos'],xlabel='Predicted',ylabel='True',title=split.title())
   for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',fontsize=12,color='white' if v>cm.max()/2 else 'black')
  for split,yy in [('train',y),('test',yt)]:
   fpr,tpr,_=roc_curve(yy,p[split]);axes[2].plot(fpr,tpr,label=f'{split}: AUC {d[split]["auc"]:.4f}')
  axes[2].plot([0,1],[0,1],'k--',lw=.7);axes[2].set(xlabel='False positive rate',ylabel='True positive rate',title=name.replace('_',' ').upper());axes[2].legend(fontsize=8,loc='lower right');fig.tight_layout();finish(name+'_evaluation')
 if all((R/f'{n}.json').exists() for n in ['rf_plain','rf_balanced','xgb_plain','xgb_smote']):
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for name in ['rf_plain','rf_balanced','xgb_plain','xgb_smote']:
   d=json.loads((R/f'{name}.json').read_text());p=np.load(R/f'{name}_predictions.npz')['test'];a,b,_=roc_curve(yt,p);axes[0].plot(a,b,label=name);prec,rec,_=precision_recall_curve(yt,p);axes[1].plot(rec,prec,label=f'{name}: AP={d["test"]["average_precision"]:.3f}')
  axes[0].set(xlabel='False positive rate',ylabel='Recall',title='Official test ROC');axes[1].set(xlabel='Recall',ylabel='Precision',title='Official test precision-recall');axes[1].axhline(yt.mean(),color='k',ls='--',lw=.8);axes[0].legend(fontsize=8);axes[1].legend(fontsize=8);fig.tight_layout();finish('model_comparison')
if __name__=='__main__':preparation();model_figures()
