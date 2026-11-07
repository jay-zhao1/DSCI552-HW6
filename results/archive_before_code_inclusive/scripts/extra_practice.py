"""Question 6: verified ISLR exercises, executed computations, figures and shared answers.
No APS fitting or existing checkpoint writes occur in this module.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=ROOT/'figures';SEED=552

def savefig(fig,name):
 fig.savefig(F/f'{name}.png',dpi=210,bbox_inches='tight');fig.savefig(F/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def predict_a(x):
 x=np.asarray(x);a,b=x[...,0],x[...,1]
 return np.where(a>=1,5,np.where(b>=1,15,np.where(a<0,3,np.where(b<0,10,0))))

def predict_b(x):
 x=np.asarray(x);a,b=x[...,0],x[...,1]
 return np.where(b<1,np.where(a<1,-1.8,.63),np.where(b<2,np.where(a<0,-1.06,.21),2.49))

def tree(ax,nodes,edges,title):
 ax.set(xlim=(-.04,1.04),ylim=(-.06,1.08));ax.axis('off');ax.set_title(title,fontsize=13)
 for parent,child,answer in edges:
  xp,yp,label=nodes[parent];xc,yc,_=nodes[child]
  ax.plot([xp,xc],[yp-.035,yc+.035],color='#5b6b7a',lw=1.1,zorder=0)
  ax.text((xp+xc)/2+(.02 if xc>xp else -.02),(yp+yc)/2,answer,fontsize=9,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
 for key,(x,y,label) in nodes.items():
  ax.text(x,y,label,ha='center',va='center',fontsize=12,bbox={'boxstyle':'round,pad=.2','facecolor':'#e7f0f6' if key.startswith('n') else '#fff3dc','edgecolor':'#718394','linewidth':.8})

def partition(ax,which):
 ax.set(xlim=(-1,3),ylim=(-1,3),xlabel='$X_1$',ylabel='$X_2$',xticks=[-1,0,1,2,3],yticks=[-1,0,1,2,3]);ax.set_aspect('equal');ax.tick_params(labelsize=10)
 if which=='a':
  regions=[(-1,-1,1,2,3),(0,-1,1,1,10),(0,0,1,1,0),(-1,1,2,2,15),(1,-1,2,4,5)]
  lines=[([1,1],[-1,3]),([-1,1],[1,1]),([0,0],[-1,1]),([0,1],[0,0])]
  ax.set_title('(a) Source partition (redrawn)',fontsize=13)
 else:
  regions=[(-1,-1,2,2,-1.8),(1,-1,2,2,.63),(-1,1,1,1,-1.06),(0,1,3,1,.21),(-1,2,4,1,2.49)]
  lines=[([-1,3],[1,1]),([1,1],[-1,1]),([-1,3],[2,2]),([0,0],[1,2])]
  ax.set_title('(b) Partition implied by source tree',fontsize=13)
 colors=['#dbe9f3','#f8e4be','#e4eddc','#e8def2','#f5dddd']
 for (x,y,w,h,v),c in zip(regions,colors):
  ax.add_patch(Rectangle((x,y),w,h,facecolor=c,edgecolor='none'));ax.text(x+w/2,y+h/2,f'{v:g}',ha='center',va='center',fontsize=14)
 for xx,yy in lines:ax.plot(xx,yy,color='#314c62',lw=1.5)


def run_extra():
 R.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':13})
 # Bootstrap: enumerate every requested n, and actually generate all 10,000 samples.
 n=np.arange(1,100001);p=np.empty(len(n),float);p[0]=1.;p[1:]=-np.expm1(n[1:]*np.log1p(-1/n[1:]))
 pd.DataFrame({'n':n,'probability_included':p}).to_csv(R/'extra_bootstrap_curve.csv',index=False)
 rng=np.random.default_rng(SEED);samples=rng.integers(1,101,size=(10000,100));contains=np.any(samples==4,axis=1);exact=float(p[99]);estimate=float(contains.mean());se=float(np.sqrt(exact*(1-exact)/len(contains)))
 bootstrap={'seed':SEED,'replicates':10000,'sample_size':100,'observation_j':4,'inclusion_count':int(contains.sum()),'empirical_probability':estimate,'exact_probability':exact,'monte_carlo_standard_error':se,'standardized_difference':(estimate-exact)/se,'exact_requested':{str(i):float(p[i-1]) for i in [5,100,10000]},'limit':float(1-np.exp(-1))}
 assert np.all(np.diff(p)<0) and abs(estimate-exact)<4*se
 fig,axes=plt.subplots(1,2,figsize=(9.6,2.9))
 for ax in axes:
  ax.plot(n,p,lw=1.8,label='$1-(1-1/n)^n$');ax.axhline(bootstrap['limit'],ls='--',color='#a14c4c',label='$1-e^{-1}$');ax.set(xlabel='Sample size n',ylabel='Inclusion probability');ax.grid(alpha=.15)
 axes[0].set(xlim=(1,100),title='Initial decrease (n = 1,...,100)');axes[1].set_xscale('log');axes[1].set(xlim=(1,100000),title='Every integer n through 100,000');axes[1].legend(fontsize=10);fig.tight_layout();savefig(fig,'extra_bootstrap')
 # Ridge illustration: one actual orthogonal-design fit; analytic risk is not a test estimate.
 d=np.array([4.,2.,1.]);X=np.diag(d);beta=np.array([2.,-1.,.7]);sigma2=1.;rng=np.random.default_rng(SEED);y=X@beta+rng.normal(size=3)
 lambdas=np.r_[0,np.logspace(-4,4,300)];coef=np.array([np.linalg.solve(X.T@X+lam*np.eye(3),X.T@y) for lam in lambdas]);rss=((y[None,:]-coef@X.T)**2).sum(axis=1)
 shrink=d[None,:]**2/(d[None,:]**2+lambdas[:,None]);bias2=(((shrink-1)*beta)**2).sum(axis=1);variance=(sigma2*d[None,:]**2/(d[None,:]**2+lambdas[:,None])**2).sum(axis=1);risk=bias2+variance+sigma2
 assert np.all(np.diff(rss)>=-1e-10) and np.all(np.diff(variance)<=1e-10) and np.all(np.diff(bias2)>=-1e-10)
 pd.DataFrame({'lambda':lambdas,'observed_training_rss':rss,'squared_bias_isotropic_test':bias2,'prediction_variance':variance,'irreducible_error':sigma2,'expected_test_mse':risk}).to_csv(R/'extra_ridge_curve.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(9.6,2.9))
 line1,=axes[0].plot(lambdas,rss,color='#2867b1',label='Training RSS (left)');riskax=axes[0].twinx();line2,=riskax.plot(lambdas,risk,color='#dd7931',label='Expected MSE (right)');riskax.set_ylabel('Expected test MSE',color='#dd7931',fontsize=10);axes[0].set(title='Computed orthogonal-design illustration',ylabel='Training RSS');axes[0].legend(handles=[line1,line2],fontsize=8,loc='upper left')
 for val,label in [(bias2,'Squared bias'),(variance,'Variance'),(np.ones(len(lambdas)),'Irreducible error')]:axes[1].plot(lambdas,val,label=label)
 axes[1].set(title='Analytic prediction-error components');axes[1].legend(fontsize=9)
 for ax in axes:ax.set_xscale('symlog',linthresh=.01);ax.set_xlabel('Ridge penalty lambda');ax.grid(alpha=.15)
 fig.tight_layout();savefig(fig,'extra_ridge')
 # Exact reconstruction of first-edition Figure 8.12, checked against second-edition 8.14.
 nodes_a={'n0':(.64,.97,'$X_1<1$'),'n1':(.39,.73,'$X_2<1$'),'l5':(.92,.73,'5'),'n2':(.20,.49,'$X_1<0$'),'l15':(.62,.49,'15'),'l3':(.04,.25,'3'),'n3':(.39,.25,'$X_2<0$'),'l10':(.29,.01,'10'),'l0':(.56,.01,'0')}
 edges_a=[('n0','n1','yes'),('n0','l5','no'),('n1','n2','yes'),('n1','l15','no'),('n2','l3','yes'),('n2','n3','no'),('n3','l10','yes'),('n3','l0','no')]
 nodes_b={'n0':(.48,.97,'$X_2<1$'),'n1':(.20,.65,'$X_1<1$'),'l1':(.04,.32,'-1.80'),'l2':(.34,.32,'0.63'),'n2':(.74,.65,'$X_2<2$'),'n3':(.60,.32,'$X_1<0$'),'l5':(.95,.32,'2.49'),'l3':(.49,.01,'-1.06'),'l4':(.75,.01,'0.21')}
 edges_b=[('n0','n1','yes'),('n0','n2','no'),('n1','l1','yes'),('n1','l2','no'),('n2','n3','yes'),('n2','l5','no'),('n3','l3','yes'),('n3','l4','no')]
 fig,axes=plt.subplots(1,2,figsize=(10,3.35));partition(axes[0],'a');tree(axes[1],nodes_a,edges_a,'(a) One valid corresponding tree');fig.tight_layout();savefig(fig,'extra_tree_a')
 fig,axes=plt.subplots(1,2,figsize=(10,3.35));tree(axes[0],nodes_b,edges_b,'(b) Source tree (redrawn)');partition(axes[1],'b');fig.tight_layout();savefig(fig,'extra_tree_b')
 points_a=np.array([[-.5,0],[.5,-.5],[.5,.5],[0,2],[2,0]]);targets_a=np.array([3,10,0,15,5]);points_b=np.array([[0,0],[2,0],[-.5,1.5],[.5,1.5],[0,2.5]]);targets_b=np.array([-1.8,.63,-1.06,.21,2.49])
 assert np.array_equal(predict_a(points_a),targets_a);assert np.allclose(predict_b(points_b),targets_b)
 # Independently traverse diagram edges and compare all piecewise values on a grid, including cuts.
 def traverse(point,nodes,edges):
  key='n0';mapping={(a,b):c for a,c,b in edges}
  while key.startswith('n'):
   label=nodes[key][2].replace('$','');lhs,rhs=label.split('<');dim=int(lhs.split('_')[1])-1;answer='yes' if point[dim]<float(rhs) else 'no';key=mapping[key,answer]
  return float(nodes[key][2])
 grid1=np.unique(np.r_[np.linspace(-1,3,81),0,1,2]);grid=np.array(np.meshgrid(grid1,grid1)).reshape(2,-1).T
 assert np.allclose([traverse(q,nodes_a,edges_a) for q in grid],predict_a(grid));assert np.allclose([traverse(q,nodes_b,edges_b) for q in grid],predict_b(grid))
 # Circular classification boundary, original four requested points and exact polynomial lift.
 points=np.array([[0,0],[-1,1],[2,2],[3,8]]);radius2=((points-np.array([-1,2]))**2).sum(axis=1);scores=radius2-4;classes=np.where(scores>0,'Blue','Red');expanded=points[:,0]**2+points[:,1]**2+2*points[:,0]-4*points[:,1]+1
 assert np.array_equal(scores,expanded);assert classes.tolist()==['Blue','Red','Blue','Blue']
 fig,axes=plt.subplots(1,2,figsize=(9.6,3.7))
 for ax in axes:
  ax.set_facecolor('#e5eefb');ax.add_patch(Circle((-1,2),2,facecolor='#f6d9db',edgecolor='#34495e',lw=1.5));ax.scatter([-1],[2],marker='+',color='black');ax.set_aspect('equal');ax.set(xlabel='$X_1$',ylabel='$X_2$');ax.grid(alpha=.12)
 axes[0].set(xlim=(-3.8,1.8),ylim=(-.8,4.8),title='(a)-(b) Boundary and signed regions');axes[0].text(-1,2.7,'Red: squared distance <= 4',ha='center',fontsize=9);axes[0].text(-1,4.35,'Blue: squared distance > 4',ha='center',fontsize=9);axes[0].annotate('center (-1,2)',xy=(-1,2),xytext=(-2.9,.7),fontsize=9,arrowprops={'arrowstyle':'->'});axes[0].plot([-1,1],[2,2],color='#34495e');axes[0].text(0,1.7,'radius 2',ha='center',fontsize=9)
 axes[1].set(xlim=(-3.8,4.4),ylim=(-1,9),title='(c) Classification of the four given points')
 for point,c in zip(points,classes):
  axes[1].scatter(*point,color='#c84a52' if c=='Red' else '#2867b1',s=35,zorder=4);axes[1].annotate(f'({int(point[0])},{int(point[1])})',point,xytext=(5,6),textcoords='offset points',fontsize=10)
 fig.tight_layout();savefig(fig,'extra_circle')
 pd.DataFrame({'X1':points[:,0],'X2':points[:,1],'squared_distance':radius2,'expanded_score':scores,'class':classes}).to_csv(R/'extra_circle_points.csv',index=False)
 result={'bootstrap':bootstrap,'ridge':{'design':'X = diag(4,2,1), beta = (2,-1,0.7), Gaussian noise variance = 1, known zero intercept, isotropic test covariates','seed':SEED,'response':y.tolist(),'training_rss_nondecreasing':True,'variance_nonincreasing':True,'squared_bias_nondecreasing':True},'tree':{'grid_points_checked':len(grid),'tree_and_partition_agree':True,'cut_convention':'less-than true branch; equality goes to false branch'},'circle':{'center':[-1,2],'radius':2,'requested_classes':classes.tolist(),'polynomial_coefficients_intercept_X1_X1sq_X2_X2sq':[1,2,1,-4,1]},'all_passed':True}
 (R/'extra_practice_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result


def extra_pages():
 # Tuple schema is shared with the existing report/notebook generators.
 def p(t):return ('p',t)
 def h(t):return ('h',t)
 def eq(t):return ('math',t)
 def im(n,w=510):return ('image',n,w)
 def table(a,b):return ('table',a,b)
 d=json.loads((R/'extra_practice_results.json').read_text());b=d['bootstrap']
 return [
 ('Question 6 - Extra Practice: ISLR 5.4.2',[
 p('Source verified separately: first edition, seventh printing, Section 5.4 Exercise 2, printed pp.197-198 (PDF pp.212-213). The second-edition 5.4.2, pp.219-220, has the same eight subparts. The task concerns inclusion of one fixed original observation j in n independent draws with replacement.'),
 p('<b>(a)</b> The first draw misses observation j with probability (n-1)/n = 1 - 1/n: exactly n-1 of the n equally likely indices differ from j. <b>(b)</b> The second draw has the same probability, 1 - 1/n; replacement restores the same sampling population.'),
 p('<b>(c)</b> All n draws are independent, so the probability that j is absent is the product of n identical miss probabilities. Taking the complement gives its inclusion probability:'),
 eq(r'P(j\notin B)=(1-1/n)^n,\qquad P(j\in B)=1-(1-1/n)^n'),
 table(['Subpart','n','Exact inclusion probability'],[[part,str(n),f'{b["exact_requested"][str(n)]:.10f}'] for part,n in [('(d)',5),('(e)',100),('(f)',10000)]]),
 p('<b>(g)</b> The code evaluates every integer n from 1 through 100,000 (the right panel uses a logarithmic x-axis). Inclusion decreases from 1 toward 1 - exp(-1) = 0.6321205588, approaching from above. Equivalently, about 36.8% of original observations are absent from a large bootstrap sample. The sample still has n draws; repeated indices explain the difference.'),im('extra_bootstrap',510),
 p(f'<b>(h)</b> With seed {b["seed"]}, we actually generated {b["replicates"]:,} bootstrap samples of size 100, and checked whether index j = 4 occurs. It appeared in {b["inclusion_count"]:,} samples: empirical probability {b["empirical_probability"]:.4f}, versus exact {b["exact_probability"]:.6f}. The Monte Carlo standard error is sqrt(p(1-p)/10000) = {b["monte_carlo_standard_error"]:.5f}; the difference is {b["standardized_difference"]:.2f} standard errors. This is consistent with simulation noise. Sampling uses NumPy\'s generator, so the seeded realization need not equal R\'s.')]),
 ('Question 6 - Extra Practice: ISLR 6.8.4',[
 p('Source: first edition, seventh printing, Section 6.8 Exercise 4, printed pp.260-261 (PDF pp.275-276). This is the ridge-penalty question. The equivalent second-edition exercise is 6.6.4, not 6.8.4. An unpenalized intercept is understood; lambda increases from zero.'),
 eq(r'\min_{\beta_0,\beta}\ R(\beta)+\lambda P(\beta),\quad R=\sum_i(y_i-\beta_0-x_i^T\beta)^2,\quad P=\sum_j\beta_j^2'),
 p('<b>(a) Training RSS: iii, steadily increases (nondecreasing).</b> Let R1,P1 and R2,P2 be the residual and penalty values at minimizers for 0 <= lambda1 < lambda2. Optimality gives R1 + lambda1 P1 <= R2 + lambda1 P2 and R2 + lambda2 P2 <= R1 + lambda2 P1. Adding implies P2 <= P1; substituting yields R2 - R1 >= lambda1(P1 - P2) >= 0. Equality is possible, for example when all fitted slopes are zero.'),
 p('<b>(b) Test RSS: ii, typically decreases initially and then increases.</b> Moderate shrinkage can reduce variance enough to improve prediction; excessive shrinkage causes underfitting. A U shape is the intended bias-variance pattern, not a theorem for every test set or underlying signal.'),
 p('<b>(c) Variance: iv, decreases.</b> For centered X = U D V-transpose with fixed design and homoskedastic noise, a ridge coefficient along right singular direction k has variance sigma squared times d_k squared / (d_k squared + lambda) squared. Each such component is nonincreasing; the unpenalized intercept contributes a constant term.'),
 eq(r'\operatorname{Var}(\hat\beta_{\lambda,k})=\frac{\sigma^2 d_k^2}{(d_k^2+\lambda)^2},\qquad \operatorname{Bias}_k=-\frac{\lambda}{d_k^2+\lambda}\beta_k'),
 p('<b>(d) Squared bias: iii, typically increases.</b> Larger lambda shrinks identifiable slope directions farther toward zero, increasing shrinkage bias. In the orthogonal coefficient norm (or isotropic test average), squared directional biases increase. Arbitrary pointwise squared prediction bias need not be monotone because directional contributions can cancel.'),
 p('<b>(e) Irreducible error: v, remains constant.</b> Changing regularization cannot change the response noise variance.'),im('extra_ridge',510),
 p('Computed illustration: X = diag(4,2,1), true slopes (2,-1,0.7), known zero intercept, unit Gaussian noise, seed 552. Training RSS comes from the realized response; expected test MSE is the analytic sum of squared bias, variance and noise for isotropic test covariates. These are not APS results or held-out empirical estimates. Monotonicity checks passed.')]),
 ('Question 6 - Extra Practice: ISLR 8.4.4',[
 p('Source: first edition, seventh printing, Section 8.4 Exercise 4, printed pp.332-333 (PDF pp.347-348), Figure 8.12. Checked independently against second-edition 8.4.4, pp.361-362, Figure 8.14: both partitions and all leaf values agree. The task asks (a) for a tree from the left partition and (b) for a partition from the right tree.'),
 p('<b>(a)</b> A valid first split is X1 < 1 (false leaf: 5). On the true branch split X2 < 1 (false leaf: 15). In the lower-left part split X1 < 0 (true leaf: 3). On its false branch split X2 < 0, yielding 10 if true and 0 otherwise. The picture is a redrawing of the source regions and our corresponding tree; the finite plotting window does not bound the actual outer regions.'),im('extra_tree_a',510),
 p('<b>(b)</b> The source tree first splits horizontally at X2 = 1, splits the lower half vertically at X1 = 1, then splits the upper half horizontally at X2 = 2 and the middle strip vertically at X1 = 0. The resulting five regions and their means are:'),im('extra_tree_b',510),
 table(['Region conditions','Mean Y'],[['X2 < 1 and X1 < 1','-1.80'],['X2 < 1 and X1 >= 1','0.63'],['1 <= X2 < 2 and X1 < 0','-1.06'],['1 <= X2 < 2 and X1 >= 0','0.21'],['X2 >= 2','2.49']]),
 p(f'Boundary convention: a less-than condition sends equality to the false branch. Independent tree traversal and piecewise region evaluation agree on {d["tree"]["grid_points_checked"]:,} grid points, including the cut lines. Each diagram has five leaves/regions.')]),
 ('Question 6 - Extra Practice: ISLR 9.7.2',[
 p('Source: first edition, seventh printing, Section 9.7 Exercise 2, printed p.368 (PDF p.383). The second-edition 9.7.2, p.398, has the same four subparts, curve, inequalities and query points.'),
 p('<b>(a)</b> The curve (1 + X1) squared + (2 - X2) squared = 4 is a circle centered at (-1,2) with radius 2. Its horizontal extremes are (-3,2) and (1,2), and its vertical extremes are (-1,0) and (-1,4).'),
 p('<b>(b)</b> The strict inequality > 4 denotes the exterior; <= 4 denotes the closed disk, including the boundary. The figure shades the exterior Blue and the closed disk Red, matching the rule in (c).'),im('extra_circle',510),
 h('(c) Classify all four specified observations'),
 table(['Point (X1,X2)','Squared distance from (-1,2)','Class'],[[f'({a},{b})',str((a+1)**2+(2-b)**2),'Blue' if (a+1)**2+(2-b)**2>4 else 'Red'] for a,b in [(0,0),(-1,1),(2,2),(3,8)]]),
 h('(d) A linear boundary in the expanded feature space'),
 eq(r'(1+X_1)^2+(2-X_2)^2-4=X_1^2+X_2^2+2X_1-4X_2+1'),
 p('Let z = (X1, X1 squared, X2, X2 squared). The same boundary is 1 + 2 z1 + z2 - 4 z3 + z4 = 0, an affine hyperplane with intercept 1 and coefficients (2,1,-4,1). Predict Blue if this score is positive and Red otherwise. Thus the boundary is nonlinear in the original two inputs but linear in these four transformed coordinates. The computed expanded scores (1,-3,5,48) agree exactly with squared distance minus 4.')])]


def extra_notebook_cells():
 import nbformat as nbf
 cells=[nbf.v4.new_markdown_cell('## Question 6 — Extra Practice\n\nAll four exercises are completed below. Each source was checked separately against both local R editions. The exact assignment reference 6.8.4 identifies the first edition; its second-edition equivalent is 6.6.4. The other three references agree across editions (with a changed tree-figure number). The original Q1–5 computations and outputs are retained; the following cells do not retrain any APS model. The additional cells were executed in a fresh kernel. For an end-to-end run, these cells are also self-contained after the earlier sections.')]
 source=Path(__file__).read_text().split('\ndef extra_notebook_cells():')[0]
 cells.append(nbf.v4.new_code_cell('from pathlib import Path\n__file__ = str(Path.cwd() / "scripts" / "extra_practice.py")\n'+source))
 cells.append(nbf.v4.new_code_cell('extra_results = run_extra()\nassert extra_results["all_passed"]'))
 for title,blocks in extra_pages():
  cells.append(nbf.v4.new_markdown_cell('### '+title.replace('Question 6 - Extra Practice: ','')))
  for block in blocks:
   if block[0] in ['p','h']:cells.append(nbf.v4.new_markdown_cell(('#### ' if block[0]=='h' else '')+block[1].replace('<b>','**').replace('</b>','**')))
   elif block[0]=='math':cells.append(nbf.v4.new_markdown_cell('$$\n'+block[1]+'\n$$'))
   elif block[0]=='table':
    head,rows=block[1:];cells.append(nbf.v4.new_markdown_cell('| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)))
   elif block[0]=='image':cells.append(nbf.v4.new_code_cell('from IPython.display import display, Image\ndisplay(Image(filename=str(F / "'+block[1]+'.png"), width=1000))'))
 cells.append(nbf.v4.new_code_cell('display(pd.read_csv(R / "extra_circle_points.csv"))\nprint("Bootstrap simulated inclusion count:", extra_results["bootstrap"]["inclusion_count"])\nprint("Tree/region grid points verified:", extra_results["tree"]["grid_points_checked"])\nassert extra_results["all_passed"]'))
 return cells

if __name__=='__main__':run_extra()
