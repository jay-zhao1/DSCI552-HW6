"""Twenty-page organization retaining all formal answers and adding four extra exercises."""
from content import pages,j,p,table,im
from extra_practice import extra_pages

def report_sections():
 original=pages();a=j('rf_plain');b=j('rf_balanced')
 rows=[]
 for name,d in [('RF plain',a),('RF balanced',b)]:
  for split in ['train','test']:
   rows.append([name+' / '+split]+[f'{d[split][k]:.5f}' for k in ['error','auc','average_precision','precision','recall','f1']])
 forest=('1(c)-(d). Random forests: unweighted and balanced',[
 p('Both use the full training set, 500 fully grown Gini trees, bootstrap samples, sqrt(170) candidate features, leaf size 1, seed 552 and 4 workers. The baseline has class_weight=None and no resampling. Following cost-sensitive forest methods [3,4], the other uses class_weight="balanced": n/(2 n_k), or 0.5085 for neg and 30 for pos (59:1 weight ratio). All other settings match. Weighting affects impurity and fitted class probabilities; it need not improve minority recall.'),
 p('Hard predictions use P(pos) >= 0.5, including ties; ROC/AUC use probabilities. Confusion matrices order true rows and predicted columns [neg, pos]. Training results are resubstitution scores and can be optimistic.'),
 table(['Model / split','Error','ROC-AUC','AP','Precision','Recall','F1'],rows),
 im('rf_plain_evaluation',510),im('rf_balanced_evaluation',510),
 p(f'1(c): OOB error {a["oob_error"]:.5f}, test error {a["test"]["error"]:.5f}, test minus OOB {a["test"]["error"]-a["oob_error"]:.5f}; fit/evaluation {a["fit_and_evaluate_seconds"]:.1f}s. 1(d): OOB error {b["oob_error"]:.5f}, test error {b["test"]["error"]:.5f}, difference {b["test"]["error"]-b["oob_error"]:.5f}; {b["fit_and_evaluate_seconds"]:.1f}s. OOB probabilities use only trees excluding each row from their bootstrap. Global training imputation makes OOB an approximate internal estimate; test prevalence also differs.'),
 p(f'Weighting raises test recall from {a["test"]["recall"]:.4f} to {b["test"]["recall"]:.4f}, but lowers precision from {a["test"]["precision"]:.4f} to {b["test"]["precision"]:.4f} and increases overall error. Low aggregate error can conceal missed positives because negatives dominate. The table reports positive precision, recall and F1, and average precision (AP), alongside required error and AUC.')])
 extras=extra_pages()
 # The fully annotated region diagram already contains the five means and cutpoints.
 # Keep the duplicate region table in the notebook, while avoiding repetition on paper.
 title,blocks=extras[2]
 extras[2]=(title,[q for q in blocks if q[0]!='table'])
 return original[:5]+[forest]+original[7:16]+extras+[original[16]]
