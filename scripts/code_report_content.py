"""Exactly 20 paper sections containing real core code, answers and actual outputs."""
from pathlib import Path
import json
import pandas as pd
from content import ROOT,j,p,h,eq,im,table,pages
CORE=ROOT/'scripts/submission_core.py'

def snippets():
    chunks={};key=None
    for line in CORE.read_text().splitlines():
        if line.startswith('# %% '):
            key=line[5:];chunks[key]=[]
        elif key:chunks[key].append(line)
    return {k:'\n'.join(v).strip() for k,v in chunks.items() if k!='end'}

def code(*keys):return ('code','\n\n'.join(snippets()[k] for k in keys))
def output(text):return ('output',text)
def performance(d):
    return table(['Split','Error','ROC-AUC','AP','Precision','Recall','F1'],
      [[s]+[f'{d[s][k]:.6f}' for k in ['error','auc','average_precision','precision','recall','f1']] for s in ['train','test']])

def report_sections():
    old=pages();rf=j('rf_plain');rb=j('rf_balanced');xp=j('xgb_plain');xs=j('xgb_smote')
    ep=j('extra_practice_results');cv=pd.read_csv(ROOT/'results/feature_cv.csv',index_col=0)
    top=j('top13');env=j('environment');sections=[]
    topout=[]
    for i in range(0,13,4):topout.append(' | '.join(f'{f}: {cv.loc[f,"cv"]:.4f}' for f in top[i:i+4]))
    sections.append(('1(a)-(b)(ii). Data, imputation and feature statistics',[
      p('DSCI 552 Homework 6. Core Python code, English answers and actual outputs for Q1-6 are included. Original full-data models and 70 CV fits are reused; this revision does not repeat APS training. Common imports/functions are defined once. Seed = 552.'),
      code('imports','preparation'),
      table(['Official split','Shape','neg / pos','Missing cells'],[['Training','60000 x 170','59000 / 1000','850015'],['Test','16000 x 170','15625 / 375','228680']]),
      p('Both headers are on line 21 after 20 preamble lines; na denotes missingness. The 170 predictors are numeric float64, identically ordered, and exclude class. Median imputation retains rows and resists extreme counters; methods and the footnote convention are explained on page 5.'),
      p('All 170 CVs use median-imputed training rows and sample SD (ddof=1). Zero means give undefined CVs and are excluded from ranking; all-missing columns are retained with zero fallback and flagged. Neither occurs here. Main A/B imputation is separate; Bt is train-only sensitivity. CV fits preprocessing only on its training fold.'),
      output('Top 13 CV features (visualization only):\n'+'\n'.join(topout)),
      p('CV = sample SD / mean; near-zero means and rare extreme values inflate CV. All later models use every one of the 170 predictors.')]))
    sections.append(('1(b)(iii). Correlation and shared scatter-matrix code',[
      code('correlation'),im('correlation_overview',430),
      p('Pearson correlation uses all 60,000 median-imputed training rows. The report image uses 1-based original feature indices; code indices are zero-based. Exact names, all 170 labels and numerical entries are retained with the notebook. Related counters form strongly correlated blocks. Correlation is marginal and linear, does not establish causation, and can be changed by imputation.'),
      h('Complete 13-feature scatter matrix: helper used with page 3 data'),code('full_scatter')]))
    sections.append(('1(b)(iv). Scatter plots: the 13 highest-CV features',[
      code('scatter'),im('scatter_report',414),
      p('The reference is cf_000; all other 12 selected features appear above. Blue = neg; red = pos. The plot uses all 1,000 positives plus 2,500 randomly sampled negatives, seed 552. This deliberate display sample does not represent prevalence. log(1+x) is used only for visualization, never model fitting.'),
      p('The full matrix contains every pair and class-conditional diagonal histograms; the helper on page 2 takes Z, ys, top and colors. Scatter plots alone cannot establish statistical significance: overplotting, marginal versus conditional effects, and multiple comparisons prevent an inferential conclusion. High CV is not predictive importance.')]))
    sections.append(('1(b)(iv)-(v). Box plots and imbalance',[
      code('boxes'),im('boxplots13',438),
      p('Boxes use all original training rows, separated by class, after training-median imputation and log(1+x) for display. Outlier symbols are hidden; no rows are removed. Several rare-count features have quartiles at zero, so boxes collapse.'),
      p('Training: 59,000 neg and 1,000 pos (59:1; 1.667% positive). Official test: 15,625 neg and 375 pos (2.344% positive). Both are severely imbalanced. Always-negative prediction has error 1.667%/2.344% but zero positive recall. The 13 plotted predictors do not restrict modeling: all 170 are retained.')]))
    sections.append(('1(c)-(d). Common evaluation and forest training code',[
      code('evaluation','forest','roc_confusion'),
      p('Common missing-data methods are deletion, constant/mean/median/mode, nearest neighbors, iterative models and multiple imputation [2]. Median filling is fast and robust to extreme counters, but can distort dependencies and does not model missingness. The homework footnote main result uses separate medians for each split, without pooling or labels. This differs from the deployment train-fitted imputer; Bt implements that sensitivity. CV always fits on tr only. The baseline has no class weighting or resampling beyond ordinary bootstrap sampling. The second forest uses balanced weights n/(2 n_k): approximately 0.5085 for neg and 30 for pos, a 59:1 ratio. Cost-sensitive weighting and sampling are established approaches to imbalanced forests [3,4]. Other parameters match.'),
      p('Hard predictions use positive probability >= 0.5, including ties. ROC-AUC uses probabilities; AP is average_precision_score, not trapezoidal PR-AUC. Confusion matrices use true rows/predicted columns in [neg, pos] order. Positive precision, recall and F1 supplement error and AUC. OOB aggregates only trees that excluded the row. RUN_TRAINING=False preserves earlier fitted models; these training calls were not re-executed for this report revision.')]))
    forestrows=[]
    for name,d in [('Plain',rf),('Balanced',rb)]:
      for split in ['train','test']:forestrows.append([name+' / '+split]+[f'{d[split][k]:.6f}' for k in ['error','auc','average_precision','precision','recall','f1']])
    sections.append(('1(c)-(d). Forest outputs and OOB comparison',[
      table(['Model / split','Error','ROC-AUC','AP','Precision','Recall','F1'],forestrows),im('rf_plain_evaluation',520),im('rf_balanced_evaluation',520),
      table(['Forest','OOB error','Test error','Test - OOB','Fit/eval seconds'],[[name,f'{d["oob_error"]:.6f}',f'{d["test"]["error"]:.6f}',f'{d["test"]["error"]-d["oob_error"]:.6f}',f'{d["fit_and_evaluate_seconds"]:.1f}'] for name,d in [('Plain',rf),('Balanced',rb)]]),
      p('Global training imputation makes OOB an approximate internal estimate rather than fully cross-fitted preprocessing. Test prevalence differs. Weighting raises recall from 0.7227 to 0.7413 but lowers precision from 0.9313 to 0.8373, increasing overall error. Low error can conceal missed positives because negatives dominate. Deep-forest training scores are optimistic resubstitution results.')]))
    sections.append(('1(e)-(f). XGBoost interpretation and fold-local Pipeline',[
      p('Installed XGBoost '+env['versions']['xgboost']+' and its matching API/documentation were inspected [5,6]. The question describes node-level L1 logistic model trees. Standard gbtree instead uses feature-threshold splits and scalar leaf scores; binary:logistic supplies logistic loss/link, and reg_alpha penalizes leaf weights. It does not fit logistic regression at each node. gblinear is a global linear booster, not a tree. No course clarification of this wording was found. This submission uses the runnable gbtree interpretation and explicitly does not claim a genuine logistic model tree.'),
      eq(r'P(Y=1\mid x)=\sigma(\sum_t f_t(x)),\quad\Omega(f_t)=\gamma L_t+\frac{\lambda}{2}\sum_\ell w_{t\ell}^2+\alpha\sum_\ell |w_{t\ell}|'),
      code('xgb_pipeline'),
      p('The baseline Pipeline has no imbalance compensation. Both procedures share scaling and all estimator settings; scaling is unnecessary for trees but makes the SMOTE comparison consistent. Standardization reduces dominance of large-unit counters in Euclidean neighbor selection. It cannot remove skew, imputation artifacts or high-dimensional distance problems.'),
      p('Inside each fold, Pipeline.fit fits the imputer and scaler on tr only, resamples tr only when enabled, and fits the estimator. predict_proba transforms va without resampling. No official-test rows or labels enter tuning. Rounds (250), threshold (0.5) and all non-alpha settings are fixed in advance; no early stopping or nested parallel CV is used.')]))
    ca=pd.read_csv(ROOT/'results/xgb_plain_cv_summary.csv');cb=pd.read_csv(ROOT/'results/xgb_smote_cv_summary.csv')
    searchrows=[[f'{a.alpha:g}',f'{a.mean_ap:.6f}',f'{a.mean_error:.6f}',f'{b.mean_ap:.6f}',f'{b.mean_error:.6f}'] for a,b in zip(ca.itertuples(),cb.itertuples())]
    sections.append(('1(e)-(f). Alpha search: code and actual five-fold outputs',[
      code('search'),table(['Alpha','Plain mean AP','Plain CV error','SMOTE mean AP','SMOTE CV error'],searchrows),
      p('Stratified shuffled 5-fold CV uses all 60,000 training rows: 48,000 fitting and 12,000 validation rows per fold (200 validation positives). Maximize mean validation AP to emphasize rare-positive ranking; exact ties choose smaller alpha. Both searches use identical fold membership. Misclassification uses the fixed 0.5 rule; the official test set chooses nothing.'),
      p(f'Selected alpha: plain {xp["selected_alpha"]:g}, SMOTE {xs["selected_alpha"]:g}. Plain selected CV error {xp["cv"]["mean_error"]:.6f} (fold SD {xp["cv"]["sd_error"]:.6f}); SMOTE {xs["cv"]["mean_error"]:.6f} (SD {xs["cv"]["sd_error"]:.6f}). The 35 fits took {pd.read_csv(ROOT/"results/xgb_plain_cv_folds.csv").seconds.sum():.1f}s and {pd.read_csv(ROOT/"results/xgb_smote_cv_folds.csv").seconds.sum():.1f}s respectively. Alpha 0 is the observed plain-search optimum, not an omitted search. Fold SD is not a confidence interval; selected CV performance is a tuning estimate rather than unbiased nested validation.')]))
    sections.append(('1(e). Full training and uncompensated XGBoost results',[
      code('final_fit'),performance(xp),im('xgb_plain_evaluation',510),
      p(f'Selected alpha = {xp["selected_alpha"]:g}. No compensation is applied. Test error {xp["test"]["error"]:.6f} minus CV error {xp["cv"]["mean_error"]:.6f} is {xp["test"]["error"]-xp["cv"]["mean_error"]:.6f}. CV fits use 80% of training rows and train-only fold preprocessing; final fitting uses all rows and main testing follows the footnote. Prevalence/distribution differences also affect the gap. Final fit/evaluation took {xp["final_fit_and_evaluate_seconds"]:.2f}s in the saved original run. Training evaluation always uses original As, not synthetic data.')]))
    sections.append(('1(f). Full-data SMOTE + XGBoost results',[
      code('smote_call'),
      eq(r'x_{new}=x_i+u(x_{neighbor}-x_i),\qquad u\sim U(0,1)'),
      table(['Stage','neg','Original pos','Final pos','Total'],[['Each fold fit','47200','800','47200','94400'],['Full final fit','59000','1000','59000','118000'],['Validation (unchanged)','11800','200','200','12000'],['Test (unchanged)','15625','375','375','16000']]),
      performance(xs),im('xgb_smote_evaluation',510),
      p(f'SMOTE [7] uses five minority neighbors, seed 552, sampling_strategy=1.0, after fold-local imputation and standardization. There is no majority downsampling. Final full-data resampling took {xs["resampling_seconds"]:.3f}s; final scaling, resampling, fit/evaluation took {xs["final_fit_and_evaluate_seconds"]:.2f}s. The original full-data and fold fits were actually executed, not repeated in this revision. Original training/test distributions are used for all displayed evaluations.'),
      p(f'Alpha = {xs["selected_alpha"]:g}; test error {xs["test"]["error"]:.6f} minus selected CV error {xs["cv"]["mean_error"]:.6f} = {xs["test"]["error"]-xs["cv"]["mean_error"]:.6f}. Compared with plain XGB, recall rises from 0.7627 to 0.8933, but precision falls from 0.9565 to 0.6080: FP rises 13 to 216 while FN falls 89 to 40. Oversampling changes the fitted prior, so scores need not be calibrated population probabilities. The threshold was not optimized on the test set. Resampling before splitting would leak neighbor information [8].')]))
    names=[('rf_plain','RF plain'),('rf_balanced','RF balanced'),('xgb_plain','XGB plain'),('xgb_smote','XGB SMOTE')]
    rows=[[label]+[f'{j(n)["test"][k]:.5f}' for k in ['error','auc','average_precision','precision','recall','f1']] for n,label in names]
    sens=[[label,f'{j(n)["test"]["error"]:.6f}',f'{j(n)["test_train_only_imputation"]["error"]:.6f}',f'{j(n)["test"]["average_precision"]:.5f}',f'{j(n)["test_train_only_imputation"]["average_precision"]:.5f}'] for n,label in names]
    sections.append(('1(c)-(f). Unified comparison and imputation sensitivity',[
      code('comparison'),table(['Model / test','Error','ROC-AUC','AP','Precision','Recall','F1'],rows),im('model_comparison',478),
      table(['Model','Main error','Train-only error','Main AP','Train-only AP'],sens),
      p('Sensitivity holds every model and scaler fixed, replacing only test-median B with train-median Bt. No result in this table selects a model or threshold. SMOTE error falls from 1.600% to 1.075%; FP falls 216 to 125 while FN rises 40 to 47. This material change shows how synthetic-neighbor fits can amplify shifts in imputed values.'),
      p('ROC-AUC measures ranking across both classes; AP summarizes precision along minority recall and depends on prevalence. The dashed PR baseline is test prevalence 0.02344. The data description [9] gives cost 10 FP + 500 FN. Retrospective test costs are RF plain 52,200; RF balanced 49,040; XGB plain 44,630; XGB SMOTE 22,160. Costs did not guide tuning.')]))
    sections.append(('Q2 and Q6 ridge: all regularization subparts',[
      p('Q2: ISLR 2nd ed. 6.6.3 (a)-(e), p.284; same content as 1st ed. 6.8.3. Q6 Extra Practice: exact assignment reference ISLR 1st ed. 6.8.4 (a)-(e), pp.260-261; its 2nd-edition equivalent is 6.6.4 (not 6.8.4), pp.284-285. These are distinct lasso-radius and ridge-penalty questions. Intercepts are unpenalized.'),
      eq(r'Q2:\ \min R(\beta)\ \mathrm{s.t.}\ \sum_j|\beta_j|\leq s;\qquad Q6:\ \min R(\beta)+\lambda\sum_j\beta_j^2'),
      table(['Subpart','Q2: increase lasso radius s','Q6: increase ridge penalty lambda'],[['(a) Training RSS','iv: nonincreasing','iii: nondecreasing'],['(b) Test RSS','ii: typical U shape','ii: typical U shape'],['(c) Variance','iii: typically increases','iv: decreases'],['(d) Squared bias','iv: typically decreases','iii: typically increases'],['(e) Irreducible error','v: constant','v: constant']]),
      h('Q2 explanations (a)-(e)'),
      p('(a) Larger s enlarges the feasible set: the previous minimizer remains feasible, so training RSS cannot increase. It may plateau once least squares is feasible. At s=0, slopes are zero and only the sample-mean intercept remains. (b) Relaxing a tight constraint first reduces underfitting; excessive flexibility can increase variance and test error. (c) Less shrinkage permits greater sensitivity to sample noise. (d) Reduced shrinkage generally removes bias. (e) The response noise variance belongs to the generating process, not the fitting method.'),
      h('Q6 ridge explanations (a)-(e)'),
      p('(a) Let R1,P1 and R2,P2 be RSS and squared-coefficient penalty values at minimizers for 0 <= lambda1 < lambda2. Optimality gives R1+lambda1 P1 <= R2+lambda1 P2 and R2+lambda2 P2 <= R1+lambda2 P1. Adding implies P2 <= P1; substituting yields R2-R1 >= lambda1(P1-P2) >= 0. Equality is possible. (b) Moderate shrinkage can reduce variance enough to improve prediction; too much shrinkage underfits. (c) For centered fixed design X=UDV-transpose and homoskedastic noise, each identifiable singular-direction variance decreases as below. The intercept contributes a constant term.'),
      eq(r'\operatorname{Var}(\hat\beta_{\lambda,k})=\frac{\sigma^2d_k^2}{(d_k^2+\lambda)^2},\qquad \operatorname{Bias}_k=-\frac{\lambda}{d_k^2+\lambda}\beta_k'),
      p('(d) Increasing lambda shrinks slopes farther toward zero, generally increasing squared bias. This is monotone in the orthogonal coefficient norm (or isotropic test average), but arbitrary pointwise prediction biases can cancel across directions. (e) Regularization cannot change irreducible response noise. The U-shaped test errors and lasso bias/variance trends are intended statistical patterns, not universal monotonic theorems for every realized sample. Only the nested-set/optimality training-RSS statements are guaranteed. No numerical computation is required by either exercise; the notebook retains optional illustrative plots.')]))
    sections.append(('Q3. ISLR 6.6.5: identical predictors and solutions',old[13][1]+[code('q3_check'),output('Centered check: ridge slopes = (0.6666667, 0.6666667)\nLasso optimal sum = 1.5; all 11 same-sign splits give objective 3.5.')]))
    sections.append(('Q4. ISLR 8.4.5: bootstrap votes vs mean probability',[
      p('Both checked R editions identify this exercise as Section 8.4, Exercise 5. The ten red probabilities in the code are the given inputs.'),code('q4'),im('q4_votes',430),
      p('Majority vote: the first four trees vote Green and the final six vote Red, so the result is Red (6 of 10 votes). No probability is exactly 0.5. Mean probability: summing all probabilities gives 4.5, hence mean P(Red)=0.45 and P(Green)=0.55; the probability-average rule predicts Green.'),
      eq(r'\overline{p}_{Red}=4.5/10=0.45<0.5'),
      p('Hard voting discards confidence, whereas averaging retains numerical magnitudes. Several low red probabilities can outweigh more numerous moderately high ones. The rules need not agree. The Question 1 sklearn forests average probabilities before thresholding, as shown in their common evaluator.'),h('Shared plotting helper for Q5 (called on page 15)'),code('q5_plot')]))
    q5blocks=old[15][1]
    q5answer=[b for b in q5blocks if b[0]=='p'][2:]
    sections.append(('Q5. ISLR 9.7.3: maximal-margin geometry (a)-(h)',[
      p('Both checked R editions use Exercise 9.7.3. (a) The numbered coordinates are given in the executed code and figure. Red is +1, Blue is -1; the hard-margin approximation is an independent numerical verification, not an APS model.'),code('q5','q5_plot_call'),
      output('SVM: coefficients (-2, 2); intercept 1; support observations {2,3,5,6}.\nGeometric margin = 0.3535533906.'),im('q5_geometry',432),*q5answer]))
    b=ep['bootstrap']
    sections.append(('Q6 Extra Practice. ISLR 5.4.2: bootstrap inclusion',[
      p('Individually verified: 1st ed. 5.4.2, pp.197-198 (PDF 212-213); 2nd ed. 5.4.2, pp.219-220, has the same (a)-(h). Bootstrap draws are independent, uniform and with replacement.'),code('bootstrap'),
      p('(a) The first draw misses fixed observation j with probability (n-1)/n=1-1/n, since n-1 equally likely indices differ from j. (b) Replacement restores the same population, so the second draw has the same miss probability. (c) Independence multiplies the n miss probabilities; inclusion is the complement.'),
      eq(r'P(j\notin B)=(1-1/n)^n,\quad P(j\in B)=1-(1-1/n)^n'),
      table(['Part','n','Exact inclusion probability'],[[part,str(n),f'{b["exact_requested"][str(n)]:.10f}'] for part,n in [('(d)',5),('(e)',100),('(f)',10000)]]),im('extra_bootstrap',478),
      p(f'(g) Every integer n=1,...,100000 is evaluated; the right display uses a log axis. Inclusion decreases from 1 toward 1-exp(-1)=0.6321205588 from above. About 36.8% of original observations are absent even though there are n draws, because indices repeat. (h) The actual seeded 10,000-sample run includes j=4 in {b["inclusion_count"]} samples: {b["empirical_probability"]:.4f}, versus exact {b["exact_probability"]:.6f}. Monte Carlo SE is {b["monte_carlo_standard_error"]:.5f}; the difference is {b["standardized_difference"]:.2f} SE. The realization need not equal R\'s generator.')]))
    sections.append(('Q6 Extra Practice. ISLR 8.4.4: executable tree/region code',[
      p('Verified separately in 1st ed. 8.4.4, pp.332-333, Figure 8.12, and 2nd ed. 8.4.4, pp.361-362, Figure 8.14: all cuts and means agree. Each nested tuple is (zero-based feature index, cut, true branch, false branch); a scalar is a leaf mean. RA/RB store (left, bottom, width, height, mean) for a finite display window. Outer regions extend beyond the window.'),code('trees'),
      p('The code is executable plotting and traversal logic, not pseudocode. It produces the complete corresponding tree for (a), and the requested partition for (b), alongside both source structures redrawn for comparison. Equality follows the false branch of each strict less-than split. Outputs and geometric explanations follow on page 18.')]))
    sections.append(('Q6 Extra Practice. ISLR 8.4.4: diagrams and answers',[
      p('(a) Root X1 < 1: false gives 5. On the true branch, split X2 < 1: false gives 15. In the lower-left branch split X1 < 0: true gives 3; otherwise split X2 < 0, giving 10 if true and 0 if false. This valid recursive tree exactly matches the source partition.'),im('core_tree_a',520),
      p('(b) Root X2 < 1: in the lower half X1 < 1 gives -1.80, otherwise 0.63. In the upper half X2 < 2 selects the middle strip, where X1 < 0 gives -1.06 and otherwise 0.21. The uppermost region X2 >= 2 has mean 2.49.'),im('core_tree_b',520),
      table(['Region conditions','Mean Y'],[['X2 < 1, X1 < 1','-1.80'],['X2 < 1, X1 >= 1','0.63'],['1 <= X2 < 2, X1 < 0','-1.06'],['1 <= X2 < 2, X1 >= 0','0.21'],['X2 >= 2','2.49']]),
      p('Independent recursive traversal and the original piecewise formulas agree on all 6,561 grid points, including cut lines. Each tree and partition has five leaves/regions. No fitted statistical model is needed; these are direct constructions from the supplied figure.')]))
    sections.append(('Q6 Extra Practice. ISLR 9.7.2: nonlinear boundary',[
      p('Verified separately: 1st ed. 9.7.2, p.368; 2nd ed. 9.7.2, p.398, has the same (a)-(d). (a) The circle has center (-1,2), radius 2, horizontal extremes (-3,2)/(1,2) and vertical extremes (-1,0)/(-1,4). (b) >4 denotes the exterior; <=4 the closed disk. Exterior is Blue and the disk, including its boundary, is Red.'),code('circle'),im('core_circle',465),
      table(['Part (c) point','Squared distance','Class'],[['(0,0)','5','Blue'],['(-1,1)','1','Red'],['(2,2)','9','Blue'],['(3,8)','52','Blue']]),
      eq(r'(1+X_1)^2+(2-X_2)^2-4=X_1^2+X_2^2+2X_1-4X_2+1'),
      p('(d) In z=(X1, X1 squared, X2, X2 squared), the boundary is 1+2z1+z2-4z3+z4=0: an affine hyperplane with intercept 1 and coefficients (2,1,-4,1). Positive score predicts Blue; otherwise Red. The original-input boundary is nonlinear. Computed scores (1,-3,5,48) equal squared distance minus 4 exactly.')]))
    refs=old[16][1]
    sections.append(('References, execution provenance and submission scope',[
      *refs[:9],
      h('Execution and format'),
      p('Python '+env['python']+'; '+', '.join(f'{k} {v}' for k,v in env['versions'].items())+'. CPU only; seed 552; estimator threads 4 and sequential folds. Saved full-data fits, probability outputs and fold search files supply every numeric model table. No APS training was repeated for this revision. Refactored constructors, folds, preprocessing arrays and metrics are independently checked against those artifacts.'),
      p('The displayed code is real Python from submission_core.py; it includes data loading/imputation, all feature statistics, plots, common evaluation, both forests, fold-local SMOTE/Pipeline, alpha search/selection, final XGBoost fits and all computational exercises. RUN_TRAINING=False prevents retraining while retaining the original training logic. Optional code for environment discovery, caching, serialization, report layout and repeated validation remains in the project.'),
      p('Homework6.pdf, the local August 24, 2026 syllabus and course introduction were checked. No explicit font-size or margin rule was found there. Body 10.4 pt, monospaced code 8.7 pt and 36-pt side margins are layout choices, not claimed course mandates. All code, answers, necessary output, figures and references count in this PDF. The corrected 20-page scope follows the user-provided course submission requirement.'),
      p('Edition cross-check: Q2/Q3 use second-edition 6.6.3/6.6.5; Q4/Q5 and Q6 5.4.2/8.4.4/9.7.2 agree across checked editions. Q6 6.8.4 explicitly uses the first-edition ridge exercise, equivalent to second-edition 6.6.4. Full source-page evidence is retained. The remaining instructor interpretation issue concerns node-level logistic model trees versus standard XGBoost, accurately distinguished on page 7.')]))
    assert len(sections)==20
    return sections
