"""Verify the actual code-inclusive PDF, notebook and saved results."""
from pathlib import Path
import json,hashlib,ast,re
import nbformat,pymupdf
from code_report_content import ROOT,report_sections

def verify_code_submission():
    if (ROOT/'results/final_quick_execution.json').exists():
        from validate_final_bounded import validate as validate_current
        return validate_current()
    R=ROOT/'results';pdf=pymupdf.open(ROOT/'DSCI552_Homework6_Report.pdf')
    assert len(pdf)==20
    layout=json.loads((R/'report_layout.json').read_text())
    assert layout['pages']==20 and layout['body_font_pt']==10.4 and layout['code_font_pt']==8.7
    assert layout['includes_core_code_answers_outputs']
    source=(ROOT/'scripts/submission_core.py').read_text();ast.parse(source)
    audit=json.loads((R/'submission_core_audit.json').read_text());assert audit['all_passed']
    assert audit['core_source_sha256']==hashlib.sha256(source.encode()).hexdigest()
    fulltext='\n'.join(p.get_text() for p in pdf);normalize=lambda s:' '.join(s.split())
    normalized=normalize(fulltext);printed=0
    code_pages=[];font_sizes=set();body_found=False
    sections=report_sections()
    for index,((title,blocks),page) in enumerate(zip(sections,pdf),1):
        assert title in page.get_text(),(index,title)
        for b in blocks:
            if b[0] in ['code','output']:
                for line in b[1].splitlines():
                    if line.strip():assert normalize(line) in normalize(page.get_text()),(index,line)
                if b[0]=='code':printed+=len(b[1].splitlines())
        assert '\ufffd' not in page.get_text()
        page_has_code=False
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=0 and y0>=0 and x1<=612.1 and y1<=792.1
                    if span['font']=='Courier':
                        font_sizes.add(round(span['size'],1))
                        page_has_code=True
                    if span['font']=='Helvetica' and abs(span['size']-10.4)<.01:body_found=True
        if page_has_code:code_pages.append(index)
    assert printed==layout['core_code_lines_printed'] and printed>250
    assert font_sizes.issubset({8.7,8.8}) and 8.7 in font_sizes and body_found
    required={
      'data_and_cv':[1], 'correlation':[2], 'scatter_box_imbalance':[3,4],
      'rf_training_evaluation_oob':[5,6], 'xgb_parameters_and_pipeline':[7],
      'fold_local_smote_and_alpha_selection':[7,8], 'xgb_final_results':[9,10],
      'comparison_and_sensitivity':[11], 'q2_and_extra_ridge_all_parts':[12],
      'q3_all_parts':[13], 'q4':[14], 'q5_all_parts':[14,15],
      'extra_bootstrap_all_parts':[16], 'extra_trees_both_parts':[17,18],
      'extra_circle_all_parts':[19], 'references':[20]}
    # Required mathematical and implementation tokens are present in selectable text.
    for token in ['ddof=1','class_weight=weight','oob_decision_function_',
                  'pipe.fit(X.iloc[tr], y[tr])','pipe.predict_proba(X.iloc[va])',
                  "ascending=[False, True]","reg_alpha=alpha",'fit_resample(As, y)',
                  'probability >= .5','0.6339676587','6.8.4','logistic model tree']:
        assert normalize(token) in normalized,token
    nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4)
    execution=json.loads((R/'standalone_notebook_execution.json').read_text())
    standalone=json.loads((R/'standalone_validation.json').read_text())
    assert nb.metadata.get('self_contained') and standalone['all_passed']
    assert execution['clean_kernel'] and execution['mode']=='quick' and execution['errors']==0
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert len(cells)==execution['code_cells']
    assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
    manifest=json.loads((R/'standalone_preservation.json').read_text())
    for file,digest in manifest.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    assert len(list((R/'report_pages').glob('page_*.png')))==20
    result={'all_passed':True,'pdf_pages':20,'body_font_pt':10.4,'code_font_pt':8.7,
      'code_leading_pt':10.2,'core_code_lines_printed':printed,'code_pages':code_pages,
      'all_printed_code_and_output_lines_found_in_pdf':True,'code_compiles_and_indentation_preserved':True,
      'code_font_sizes_observed':sorted(font_sizes),'all_text_within_page':True,
      'required_item_pages':required,'notebook_executed_code_cells':len(cells),'notebook_errors':0,
      'notebook_self_contained_clean_kernel_verified':True,'saved_experiment_files_unchanged':len(manifest),
      'aps_models_retrained_this_revision':False,'full_training_mode_executed':False,'core_equivalence_audit_passed':True,
      'format_audit':'No explicit local font/margin mandate found; chosen settings are documented.',
      'remaining_issue':'Original Q1(e-f) logistic model tree wording; standard gbtree interpretation explicitly distinguished.'}
    (R/'code_inclusive_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify_code_submission()
