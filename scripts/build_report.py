"""Build a <=20-page submission including real core code, answers and outputs."""
from pathlib import Path
import hashlib,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,
                               TableStyle,Image,PageBreak,Preformatted)
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from pypdf import PdfReader
from content import ROOT
from code_report_content import report_sections
W=528
BODY=10.4;CODE=8.7
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=BODY,
                         leading=13.0,spaceAfter=6))
styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=11.5,
                         leading=14,spaceBefore=3,spaceAfter=5))
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=15.5,
                         leading=18.5,spaceAfter=8))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=9.2,leading=11))
styles.add(ParagraphStyle(name='CodeCustom',fontName='Courier',fontSize=CODE,leading=10.2,
                         leftIndent=4,rightIndent=4,spaceBefore=2,spaceAfter=7))
styles.add(ParagraphStyle(name='OutputCustom',fontName='Courier',fontSize=8.8,leading=10.3,
                         leftIndent=4,spaceBefore=2,spaceAfter=6))

def equation(tex):
    path=ROOT/'figures'/('equation_'+hashlib.sha256(tex.encode()).hexdigest()[:12]+'.png')
    fig=plt.figure(figsize=(12,1))
    fig.text(.5,.5,'$'+tex+'$',ha='center',va='center',fontsize=16)
    fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    a,b=PILImage.open(path).size;w=min(510,a/220*72)
    return Image(str(path),width=w,height=b/a*w)

def make_table(head,rows):
    data=[[Paragraph(str(c),styles['CellCustom']) for c in row] for row in [head]+rows]
    n=len(head)
    first=110 if head[0] in ['Model / test','Model / split','Model','Stage'] else 85
    if n==6 and head[0]=='Split': widths=[40,40,70,70,75,233]
    elif n==2 and head[0]=='Region conditions': widths=[360,168]
    else:widths=[first]+[(W-first)/(n-1)]*(n-1)
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf4')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#203b53')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#7a99b2')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),
        ('TOPPADDING',(0,0),(-1,-1),3.5),('BOTTOMPADDING',(0,0),(-1,-1),3.5)]))
    return t

def footer(c,d):
    c.saveState();c.setStrokeColor(colors.HexColor('#b9c7d2'));c.line(36,28,576,28)
    c.setFont('Helvetica',8.5);c.setFillColor(colors.HexColor('#556879'))
    c.drawString(36,17,'DSCI 552 | Homework 6 | code, answers and outputs')
    c.drawRightString(576,17,str(d.page));c.restoreState()

class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self,f):
        if hasattr(f,'section_title'):
            self.section_pages.append({'title':f.section_title,'page':self.page})

def build():
    selection=json.loads((ROOT/'results/final_run_selection.json').read_text())
    figure_dir=ROOT/selection['report_results']/'figures'
    sections=report_sections();story=[];heights=[];code_lines=0
    for index,(title,blocks) in enumerate(sections):
        heading=Paragraph(title,styles['TitleCustom']);heading.section_title=title
        parts=[heading]
        for block in blocks:
            kind=block[0]
            if kind=='p':parts.append(Paragraph(block[1].replace('&','&amp;'),styles['BodyCustom']))
            elif kind=='h':parts.append(Paragraph(block[1],styles['SectionCustom']))
            elif kind=='math':parts.extend([equation(block[1]),Spacer(1,5)])
            elif kind=='image':
                f=figure_dir/f'{block[1]}.png'
                if not f.exists(): f=ROOT/'figures'/f'{block[1]}.png'
                a,b=PILImage.open(f).size
                parts.extend([Image(str(f),width=block[2],height=b/a*block[2]),Spacer(1,5)])
            elif kind=='table':parts.extend([make_table(block[1],block[2]),Spacer(1,6)])
            elif kind in ['code','output']:
                from reportlab.pdfbase.pdfmetrics import stringWidth
                font='Courier';size=CODE if kind=='code' else 8.8
                assert all(stringWidth(line,font,size)<=W-12 for line in block[1].splitlines()),'Code line too wide'
                parts.append(Preformatted(block[1],styles['CodeCustom' if kind=='code' else 'OutputCustom']))
                if kind=='code':code_lines+=len(block[1].splitlines())
        height=sum(x.wrap(W,722)[1]+getattr(x,'spaceBefore',0)+getattr(x,'spaceAfter',0) for x in parts)
        heights.append({'section':index+1,'title':title,'estimated_height':height})
        story.extend(parts)
        if index<len(sections)-1:story.append(PageBreak())
    candidate=ROOT/'tmp/pdfs/code_inclusive_candidate.pdf';candidate.parent.mkdir(parents=True,exist_ok=True)
    doc=ReportDoc(str(candidate),pagesize=(612,792),leftMargin=36,rightMargin=36,
                  topMargin=32,bottomMargin=38,title='DSCI 552 Homework 6: Complete Submission',author='')
    doc.section_pages=[];doc.build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(candidate).pages)
    layout={'pages':count,'body_font_pt':BODY,'code_font_pt':CODE,'code_leading_pt':10.2,
            'margins_pt':{'left':36,'right':36,'top':32,'bottom':38},
            'core_code_lines_printed':code_lines,'sections':doc.section_pages,'estimates':heights,
            'includes_core_code_answers_outputs':True}
    (ROOT/'results/report_layout.json').write_text(json.dumps(layout,indent=2))
    print(json.dumps(layout,indent=2))
    assert count<=20,f'{count} pages: candidate preserved; final file not replaced'
    assert count==len(sections),'A section overflowed or an unexpected page was introduced'
    path=ROOT/'DSCI552_Homework6_Report.pdf';shutil.copy2(candidate,path)
    import pymupdf
    dest=ROOT/'results/report_pages';dest.mkdir(exist_ok=True);pdf=pymupdf.open(path)
    for stale in dest.glob('page_*.png'):
        if int(stale.stem.split('_')[1])>count:stale.unlink()
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(str(dest/f'page_{i+1:02d}.png'))
    for start in range(0,count,4):
        sheet=PILImage.new('RGB',(1224,1584),'#cccccc')
        for k in range(start,min(start+4,count)):
            im=PILImage.open(dest/f'page_{k+1:02d}.png');im.thumbnail((612,792))
            sheet.paste(im,((k-start)%2*612,(k-start)//2*792))
        sheet.save(dest/f'contact_{start+1:02d}.png')
    return layout
if __name__=='__main__':build()
