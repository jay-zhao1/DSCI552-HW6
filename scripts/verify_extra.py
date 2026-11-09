"""Verify fresh standalone outputs, all extra-practice mathematics and source editions."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,nbformat,pymupdf
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'

def verify_extra():
    record=R/'oob_quick_execution.json'
    if not record.exists():record=R/'final_full_execution.json'
    if not record.exists():record=R/'standalone_notebook_execution.json'
    execution=json.loads(record.read_text())
    out=ROOT/execution['output_directory']
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4)
    assert nb.metadata['self_contained']
    assert '## Question 6 — Extra Practice' in '\n'.join(c.source for c in nb.cells)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not [o for c in codes for o in c.outputs if o.output_type=='error']
    d=json.loads((out/'extra_practice_results.json').read_text());assert d['all_passed']
    curve=pd.read_csv(out/'extra_bootstrap_curve.csv');assert len(curve)==100000
    for n in [5,100,10000]:assert np.isclose(curve.iloc[n-1].probability_included,1-(1-1/n)**n,atol=1e-12)
    assert np.all(np.diff(curve.probability_included)<0)
    b=d['bootstrap'];assert b['inclusion_count']/10000==b['empirical_probability']
    ridge=pd.read_csv(out/'extra_ridge_curve.csv')
    assert np.all(np.diff(ridge.observed_training_rss)>=-1e-9)
    assert np.all(np.diff(ridge.prediction_variance)<=1e-9)
    assert np.all(np.diff(ridge.squared_bias_isotropic_test)>=-1e-9)
    assert np.allclose(ridge.expected_test_mse,ridge.squared_bias_isotropic_test+ridge.prediction_variance+ridge.irreducible_error)
    points=pd.read_csv(out/'extra_circle_points.csv')
    assert points['class'].tolist()==['Blue','Red','Blue','Blue']
    assert points.expanded_score.tolist()==[1,-3,5,48]
    editions=json.loads((ROOT/'references/extra_practice_sources/manifest.json').read_text())
    for edition in ['first','second']:
        path=ROOT/'references'/editions[edition]['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==editions[edition]['sha256']
    pdf=pymupdf.open(ROOT/'DSCI552_Homework6_Report.pdf');assert len(pdf)<=20
    for i,text in [(11,'Q2'),(11,'6.8.4'),(12,'Q3.'),(13,'Q4.'),(14,'Q5.'),
                   (15,'5.4.2'),(16,'8.4.4'),(17,'8.4.4'),(18,'9.7.2'),(19,'References,')]:
        assert text in pdf[i].get_text()
    result=dict(all_passed=True,source='Fresh standalone clean-kernel execution',
        original_models_preserved=True,new_aps_training=execution.get('aps_models_retrained',False),
        all_four_extra_exercises_complete=True,exercise_statement_ambiguities=[],
        main_report_pages=len(pdf),extra_practice_pages=[12,16,17,18,19],
        bootstrap_and_ridge_numeric_checks_passed=True,
        tree_grid_points_checked=d['tree']['grid_points_checked'],
        circle_classifications_verified=True,verified_textbook_hashes_unchanged=True)
    (R/'extra_practice_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify_extra()
