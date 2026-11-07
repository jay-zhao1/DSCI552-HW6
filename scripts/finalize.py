"""Check final artifact consistency and refresh README values from saved outputs."""
from pathlib import Path
import json,re,hashlib
import nbformat
from pypdf import PdfReader
import pymupdf
from content import ROOT,R,NAMES,j

def finalize():
 report=ROOT/'DSCI552_Homework6_Report.pdf';count=len(PdfReader(report).pages);assert count<=20
 nb=nbformat.read(ROOT/'DSCI552_Homework6.ipynb',as_version=4);codes=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in codes)
 assert not [o for c in codes for o in c.outputs if o.output_type=='error']
 assert j('validation')['all_passed']
 # Check actual source files against the initial audit hashes.
 audit=j('data_audit'); initial=json.loads((ROOT/'references/input_manifest.json').read_text())
 for split in ['training','test']:
  file=ROOT/'Homework 6 Data'/f'aps_failure_{split}_set.csv';assert hashlib.sha256(file.read_bytes()).hexdigest()==audit[split]['sha256']==initial[f'{split}_csv_sha256']
 rows=['| Model | Selected alpha | Error | ROC-AUC | AP | Positive precision | Positive recall | Positive F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name in NAMES:
  d=j(name);row=[NAMES[name],str(d.get('selected_alpha','-'))]+[f'{d["test"][k]:.5f}' for k in ['error','auc','average_precision','precision','recall','f1']];rows.append('| '+' | '.join(row)+' |')
 readme=ROOT/'README.md';s=readme.read_text();s=re.sub(r'\| Model \| Selected alpha .*?(?=\n\nSMOTE detects)', '\n'.join(rows),s,flags=re.S);s=re.sub(r'\*\*\d+ actual PDF pages\*\*',f'**{count} actual PDF pages**',s);readme.write_text(s)
 pdf=pymupdf.open(report)
 assert len(pdf)==17
 # Ensure the report sections did not silently shift onto unexpected extra pages.
 for i,t in [(0,'DSCI 552 Homework 6'),(12,'Question 2.'),(13,'Question 3.'),(14,'Question 4.'),(15,'Question 5.'),(16,'References,')]:assert t in pdf[i].get_text()
 assert len(list((R/'report_pages').glob('page_*.png')))==count
 record=j('validation');record.update({'notebook_executed_code_cells':len(codes),'notebook_errors':0,'pdf_pages':count,'pdf_within_20_page_limit':True,'original_csv_hashes_unchanged':True,'report_sections_in_expected_pages':True})
 (R/'validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':finalize()
