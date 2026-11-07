"""Append and execute the code-inclusive submission core without rerunning Q1 fits."""
from pathlib import Path
import json
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]

def code_cells():
    source=(ROOT/'scripts/submission_core.py').read_text()
    return [nbf.v4.new_markdown_cell('## Code-inclusive PDF: implementation and provenance\n\nThe current 20-page PDF includes the core Python implementation, all Q1–5 answers, four Q6 exercises, necessary figures and actual numeric outputs. Its body is 10.4 pt and monospaced code is 8.7 pt; no local course font/margin mandate was found. The following compact implementation is printed in the PDF and checked against the original full implementation above. Original APS models and search outputs are preserved. `RUN_TRAINING=False`: only preprocessing, plots and small theoretical examples execute in this section. APS fitting is additionally forbidden by the independent audit. The new cells were executed in a fresh kernel; previous executed cells and outputs remain intact.'),
      nbf.v4.new_code_cell('import matplotlib\nmatplotlib.use("Agg")\n'+source+'\nplt.close("all")\nprint("Core preprocessing, figures and theoretical computations executed; APS fitting disabled.")\nprint("Bootstrap:", empirical, "Q3 objective:", objectives[0], "Circle:", classes.tolist())'),
      nbf.v4.new_code_cell('from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path.cwd() / "scripts"))\nfrom audit_submission_core import audit\ncore_audit = audit()\nassert core_audit["all_passed"]')]

def update():
    path=ROOT/'DSCI552_Homework6.ipynb';nb=nbf.read(path,as_version=4)
    start=next((i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## Code-inclusive PDF:')),len(nb.cells))
    previous=nb.cells[:start];snapshot=json.dumps(previous,sort_keys=True)
    fragment=nbf.v4.new_notebook(cells=code_cells(),metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (DSCI552 .venv)','language':'python'}})
    NotebookClient(fragment,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
    for c in fragment.cells:c.metadata['homework_section']='code_inclusive_submission'
    code=[c for c in fragment.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type=='error']
    nb.cells=previous+fragment.cells;assert snapshot==json.dumps(nb.cells[:len(previous)],sort_keys=True)
    nb.metadata['code_inclusive_pdf']={'pages':20,'body_font_pt':10.4,'code_font_pt':8.7,'aps_refits_this_revision':False}
    nbf.validate(nb);nbf.write(nb,path)
    record={'new_code_cells':len(code),'errors':0,'previous_cells_preserved_verbatim':True,'previous_code_cells':sum(c.cell_type=='code' for c in previous),'total_code_cells':sum(c.cell_type=='code' for c in nb.cells),'aps_models_retrained':False,'execution_mode':'New compact core and independent audit only, in a fresh kernel.'}
    (ROOT/'results/code_inclusive_notebook_execution.json').write_text(json.dumps(record,indent=2))
    execution=json.loads((ROOT/'results/notebook_execution.json').read_text());execution['code_inclusive_revision']=record;execution['total_code_cells']=record['total_code_cells'];(ROOT/'results/notebook_execution.json').write_text(json.dumps(execution,indent=2))
    print(json.dumps(record,indent=2))
if __name__=='__main__':update()
