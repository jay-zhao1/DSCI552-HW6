"""Validate added mathematics, original-work preservation, notebook outputs and report scope."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,nbformat,pymupdf
from extra_practice import ROOT,R

def verify_extra():
 manifest=json.loads((R/'extra_preservation_manifest.json').read_text())
 for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 old=nbformat.read(R/'pre_extra_practice/DSCI552_Homework6.ipynb',as_version=4);new=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4)
 oldcodes=[c for c in old.cells if c.cell_type=='code'];newcodes=[c for c in new.cells if c.cell_type=='code'];assert oldcodes==newcodes[:len(oldcodes)]
 assert '## Question 6 — Extra Practice' in '\n'.join(c.source for c in new.cells if c.cell_type=='markdown')
 extra=[c for c in new.cells if c.metadata.get('homework_section')=='question_6_extra_practice'];extracodes=[c for c in extra if c.cell_type=='code'];assert len(extracodes)==8
 assert all(c.execution_count is not None for c in newcodes);assert not [o for c in newcodes for o in c.outputs if o.output_type=='error']
 # Only the two explicitly outdated scope/cross-reference sentences may differ in old markdown.
 for a,b in zip(old.cells,new.cells):
  if a.cell_type=='markdown':
   expected=a.source.replace('Page 12 reports the train-only alternative.','The unified comparison reports the train-only alternative.').replace('Question 6 is explicitly Extra Practice and is recorded separately in CHECKLIST.md, excluded from this submitted report.','Question 6 contains all four completed Extra Practice exercises, with individually verified edition references, computations and diagrams, integrated before the report references.')
   assert expected==b.source
 d=json.loads((R/'extra_practice_results.json').read_text());assert d['all_passed']
 curve=pd.read_csv(R/'extra_bootstrap_curve.csv');assert len(curve)==100000 and curve.n.iloc[0]==1 and curve.n.iloc[-1]==100000
 for n in [5,100,10000]:assert np.isclose(curve.iloc[n-1].probability_included,1-(1-1/n)**n,atol=1e-12)
 assert np.all(np.diff(curve.probability_included)<0)
 b=d['bootstrap'];assert b['inclusion_count']/b['replicates']==b['empirical_probability'];assert b['replicates']==10000 and b['observation_j']==4
 ridge=pd.read_csv(R/'extra_ridge_curve.csv');assert np.all(np.diff(ridge.observed_training_rss)>=-1e-9);assert np.all(np.diff(ridge.prediction_variance)<=1e-9);assert np.all(np.diff(ridge.squared_bias_isotropic_test)>=-1e-9)
 assert np.allclose(ridge.expected_test_mse,ridge.squared_bias_isotropic_test+ridge.prediction_variance+ridge.irreducible_error)
 points=pd.read_csv(R/'extra_circle_points.csv');assert points['class'].tolist()==['Blue','Red','Blue','Blue'];assert points.expanded_score.tolist()==[1,-3,5,48]
 pdf=pymupdf.open(ROOT/'DSCI552_Homework6_Report.pdf');assert len(pdf)==20
 for i,name in [(5,'1(c)-(d).'),(11,'Q2'),(12,'Q3.'),(13,'Q4.'),(14,'Q5.'),(15,'5.4.2'),(11,'6.8.4'),(17,'8.4.4'),(18,'9.7.2'),(19,'References,')]:assert name in pdf[i].get_text()
 text='\n'.join(p.get_text() for p in pdf);assert 'excluded from this submitted report' not in text
 for page in pdf:
  assert '\ufffd' not in page.get_text()
  for block in page.get_text('blocks'):
   if len(block)>6 and block[6]==0:assert block[0]>=0 and block[1]>=0 and block[2]<=612.1 and block[3]<=792.1
 assert len(list((R/'report_pages').glob('page_*.png')))==20
 result={'all_passed':True,'formal_notebook_code_and_outputs_unchanged':True,'original_code_cells':len(oldcodes),'new_executed_code_cells':len(extracodes),'notebook_errors':0,'original_experiment_files_hash_verified':len(manifest),'model_retraining_performed':False,'all_four_extra_exercises_complete':True,'exercise_statement_ambiguities':[],'main_report_pages':len(pdf),'extra_practice_pages':[12,16,17,18,19],'organization':'merged into main report','bootstrap_and_ridge_numeric_checks_passed':True,'tree_grid_points_checked':d['tree']['grid_points_checked'],'circle_classifications_verified':True,'report_text_bounds_and_replacement_character_checks_passed':True}
 (R/'extra_practice_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify_extra()
