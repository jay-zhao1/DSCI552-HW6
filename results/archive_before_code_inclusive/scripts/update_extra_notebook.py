"""Execute only Question 6 cells and append them, preserving all Q1-5 code/outputs."""
import json,hashlib
import nbformat as nbf
from nbclient import NotebookClient
from extra_practice import ROOT,R,extra_notebook_cells

def update():
 path=ROOT/'DSCI552_Homework6.ipynb';nb=nbf.read(path,as_version=4)
 # Idempotent reruns replace only the independently authored Question 6 section.
 start=next((i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Question 6 — Extra Practice')),len(nb.cells))
 base=nb.cells[:start]
 old_code=[json.dumps(c,sort_keys=True) for c in base if c.cell_type=='code']
 for c in base:
  if c.cell_type=='markdown':
   c.source=c.source.replace('Page 12 reports the train-only alternative.','The unified comparison reports the train-only alternative.').replace('Question 6 is explicitly Extra Practice and is recorded separately in CHECKLIST.md, excluded from this submitted report.','Question 6 contains all four completed Extra Practice exercises, with individually verified edition references, computations and diagrams, integrated before the report references.')
 extra=nbf.v4.new_notebook(cells=extra_notebook_cells(),metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (DSCI552 .venv)','language':'python'}})
 NotebookClient(extra,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
 for c in extra.cells:
  c.metadata['homework_section']='question_6_extra_practice'
 codes=[c for c in extra.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in codes)
 assert not [o for c in codes for o in c.outputs if o.output_type=='error']
 nb.cells=base+extra.cells;nbf.validate(nb);nbf.write(nb,path)
 assert old_code==[json.dumps(c,sort_keys=True) for c in nb.cells[:len(base)] if c.cell_type=='code']
 record={'extra_code_cells':len(codes),'extra_execution_counts':[c.execution_count for c in codes],'errors':0,'execution_mode':'Only new Question 6 cells executed in a fresh kernel; original Q1-5 code and outputs preserved verbatim.','original_code_cells_unchanged':True,'original_code_cells':len(old_code),'total_code_cells':len(old_code)+len(codes)}
 (R/'extra_notebook_execution.json').write_text(json.dumps(record,indent=2))
 original=json.loads((R/'notebook_execution.json').read_text());original['extra_practice']=record;original['total_code_cells']=len(old_code)+len(codes);(R/'notebook_execution.json').write_text(json.dumps(original,indent=2))
 print(json.dumps(record,indent=2))
if __name__=='__main__':update()
