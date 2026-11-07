"""Generate a bounded report including Question 6 and render every page for QA."""
from pathlib import Path
import hashlib,json,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader
from content import ROOT
from report_sections import report_sections
W=528;styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.5,leading=14,spaceAfter=8));styles.add(ParagraphStyle(name='SectionCustom',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=5,spaceAfter=7));styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=17,leading=21,spaceAfter=12));styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=9,leading=11))
def equation(tex):
 path=ROOT/'figures'/('equation_'+hashlib.sha256(tex.encode()).hexdigest()[:12]+'.png')
 fig=plt.figure(figsize=(12,1));fig.text(.5,.5,'$'+tex+'$',ha='center',va='center',fontsize=16);fig.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.08);plt.close(fig)
 a,b=PILImage.open(path).size;w=min(510,a/220*72);return Image(str(path),width=w,height=b/a*w)
def make_table(head,rows):
 data=[[Paragraph(str(c),styles['CellCustom']) for c in row] for row in [head]+rows]
 # Wide first column for unified comparison.
 n=len(head);first=126 if head[0] in ['Model / test','Model / split','Model','Stage'] else (64 if n>5 else 90);widths=[first]+[(W-first)/(n-1)]*(n-1)
 t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf4')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#203b53')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7a99b2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t

def footer(c,d):
 c.saveState();c.setStrokeColor(colors.HexColor('#b9c7d2'));c.line(42,35,570,35);c.setFont('Helvetica',9);c.setFillColor(colors.HexColor('#556879'));c.drawString(42,23,'DSCI 552 | Homework 6');c.drawRightString(570,23,str(d.page));c.restoreState()
def build():
 story=[];heights=[]
 sections=report_sections()
 for index,(title,blocks) in enumerate(sections):
  parts=[Paragraph(title,styles['TitleCustom'])]
  for block in blocks:
   kind=block[0]
   if kind=='p':parts.append(Paragraph(block[1].replace('&','&amp;'),styles['BodyCustom']))
   elif kind=='h':parts.append(Paragraph(block[1],styles['SectionCustom']))
   elif kind=='math':parts.extend([equation(block[1]),Spacer(1,8)])
   elif kind=='image':
    file=ROOT/'figures'/f'{block[1]}.png';a,b=PILImage.open(file).size;parts.extend([Image(str(file),width=block[2],height=b/a*block[2]),Spacer(1,7)])
   elif kind=='table':parts.extend([make_table(block[1],block[2]),Spacer(1,9)])
  height=sum(x.wrap(W,720)[1]+getattr(x,'spaceBefore',0)+getattr(x,'spaceAfter',0) for x in parts)
  heights.append({'page':index+1,'title':title,'estimated_height':height});story.extend(parts)
  if index<len(sections)-1:story.append(PageBreak())
 path=ROOT/'DSCI552_Homework6_Report.pdf';doc=SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=46,title='DSCI 552 Homework 6',author='');doc.build(story,onFirstPage=footer,onLaterPages=footer)
 count=len(PdfReader(path).pages);(ROOT/'results'/'report_layout.json').write_text(json.dumps({'pages':count,'estimated_sections':heights},indent=2));print('PDF PAGES',count);print(heights);assert count<=20
 import pymupdf
 dest=ROOT/'results'/'report_pages';dest.mkdir(exist_ok=True);pdf=pymupdf.open(path)
 for stale in dest.glob('page_*.png'):
  if int(stale.stem.split('_')[1])>count:stale.unlink()
 for i,page in enumerate(pdf):page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(str(dest/f'page_{i+1:02d}.png'))
 for start in range(0,count,4):
  sheet=PILImage.new('RGB',(1224,1584),'#cccccc')
  for k in range(start,min(start+4,count)):
   im=PILImage.open(dest/f'page_{k+1:02d}.png');im.thumbnail((612,792));sheet.paste(im,((k-start)%2*612,(k-start)//2*792))
  sheet.save(dest/f'contact_{start+1:02d}.png')
if __name__=='__main__':build()
