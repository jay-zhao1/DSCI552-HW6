"""Author and execute the submission notebook with explicit resumable computation."""
from pathlib import Path
import sys,json,re
import nbformat as nbf
from nbclient import NotebookClient
from content import pages,ROOT

def build(execute=True):
 nb=nbf.v4.new_notebook();cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s))
 def code(s):cells.append(nbf.v4.new_code_cell(s))
 md('# DSCI 552 Homework 6\n\nEnglish submission. All official data are used. This notebook includes the full experiment implementation and the formal answers. Expensive fits are checkpointed in `results/`; executing with existing checkpoints reuses them and independently verifies predictions against saved estimators. A fresh run is obtained in a copy of the project with generated experiment checkpoints removed (see README). No input file is changed. Seed = 552.\n\nThe report and notebook read the same actual outputs. Final test imputation follows the unusual homework footnote; cross-validation remains strictly training-fold fitted. The standard XGBoost implementation differs from the node-level logistic model tree described by the question; see 1(e).')
 code('from pathlib import Path\nimport os, sys, json\nROOT = Path.cwd()\nassert (ROOT / "Homework6.pdf").exists(), "Run from the project root"\nsys.path.insert(0, str(ROOT / "scripts"))\nprint("Python:", sys.version)\nprint("Project:", ROOT)')
 md('## Reproducible experiment code\n\nThe following is the complete implementation used for the full-data fits, including the leakage-safe CV Pipeline, candidate search, metrics, and checkpoints. `run()` skips only successfully saved stages. CV results use no official test observations or labels.')
 source=(ROOT/'scripts/experiments.py').read_text().split("if __name__=='__main__':run()")[0]
 code('__file__ = str(ROOT / "scripts" / "experiments.py")\n'+source)
 code('run()\nprint("All four full-data model stages and both 35-fit CV searches are present.")\nprint((R / "environment.json").read_text())')
 md('## Figure-generation code\n\nThe plotting subsample affects only scatter plots. Box plots and correlations use the complete training set. Models use all 170 predictors.')
 source=(ROOT/'scripts/figures.py').read_text().split("if __name__=='__main__':preparation();model_figures()")[0]
 code(source)
 code('preparation()\nmodel_figures()\nprint("All figures regenerated from training data and saved model predictions.")')
 for title,blocks in pages():
  md('## '+title)
  for block in blocks:
   if block[0] in ['p','h']:
    text=block[1].replace('<b>','**').replace('</b>','**')
    md(('### ' if block[0]=='h' else '')+text)
   elif block[0]=='math':md('$$\n'+block[1]+'\n$$')
   elif block[0]=='table':
    head,rows=block[1:];md('| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))
   elif block[0]=='image':code('from IPython.display import display, Image\ndisplay(Image(filename=str(ROOT / "figures" / "'+block[1]+'.png"), width=1000))')
 md('## Detailed outputs and supplementary plots\n\nThese are retained outside the page-limited report. The complete scatter matrix uses every pair among the 13 features. Numerical correlation and all 170 CVs are shown below.')
 code('display(pd.read_csv(R / "feature_cv.csv").to_string(index=False))\ndisplay(pd.read_csv(R / "correlation.csv", index_col=0))\ndisplay(Image(filename=str(ROOT / "figures/scatter_all13.png"), width=1500))\ndisplay(Image(filename=str(ROOT / "figures/correlation_all_labels.png"), width=1500))')
 code('for name in ["xgb_plain", "xgb_smote"]:\n    print(name, "all 35 fold evaluations")\n    display(pd.read_csv(R / f"{name}_cv_folds.csv"))\nfor name in ["rf_plain", "rf_balanced", "xgb_plain", "xgb_smote"]:\n    print(name, json.dumps(json.loads((R / f"{name}.json").read_text()), indent=2))')
 md('## Executed mathematical checks for Questions 2-5\n\nFor Question 2 the nested feasible-set argument is analytical; the plotted shapes are explicitly schematic. The next cell verifies the ridge/lasso algebra on a centered identical-column example, the ten-tree calculation, and the geometric margin/support vectors.')
 code('''# Centered example for Q3: x=(1,-1), y=(2,-2), lambda=2.
x = np.array([1., -1.]); yy = np.array([2., -2.]); lam = 2.
A0 = x @ x; C0 = x @ yy
ridge_beta = C0 / (2*A0 + lam)
tstar = np.sign(C0)*max(abs(C0)-lam/2, 0)/A0
splits = np.linspace(0,1,11)
lasso_objectives = [np.sum((yy-x*tstar)**2)+lam*(abs(a*tstar)+abs((1-a)*tstar)) for a in splits]
assert np.ptp(lasso_objectives)<1e-12
print("Q3 ridge coefficients:", (ridge_beta,ridge_beta), "lasso optimal sum:", tstar)
print("Q3 identical lasso objective across all 11 splits:", lasso_objectives[0])
probs = np.array([.1,.15,.2,.2,.55,.6,.6,.65,.7,.75])
assert (probs>.5).sum()==6 and np.isclose(probs.mean(), .45)
print("Q4 Red votes:", (probs>.5).sum(), "; average Red probability:", probs.mean())
print("Q5 independent SVM verification:", (R / "q5_verification.json").read_text())''')
 md('## Final independent checks\n\nRecompute all saved metrics, reload serialized estimators to reproduce official-test probabilities, and check fold membership, CV selection, feature CVs, shapes and correlation symmetry.')
 code('from verify import verify\nvalidation = verify()\nassert validation["all_passed"]\nprint("All necessary saved-result checks passed.")')
 from extra_practice import extra_notebook_cells, run_extra
 run_extra()
 cells.extend(extra_notebook_cells())
 from update_code_notebook import code_cells
 cells.extend(code_cells())
 nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3 (DSCI552 .venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}}
 path=ROOT/'DSCI552_Homework6.ipynb';nbf.write(nb,path)
 if execute:
  client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
  client.execute();nbf.write(nb,path)
  errors=[(i,o) for i,c in enumerate(nb.cells) if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
  counts=[c.execution_count for c in nb.cells if c.cell_type=='code'];assert not errors and all(c is not None for c in counts)
  (ROOT/'results/notebook_execution.json').write_text(json.dumps({'executed':True,'code_cells':len(counts),'execution_counts':counts,'errors':len(errors),'checkpoint_reuse':True,'figures_regenerated':True},indent=2))
  print('Executed notebook',len(counts),'code cells; no errors.')
if __name__=='__main__':build()
