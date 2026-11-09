"""Exactly 20 paper sections containing real core code, answers and actual outputs."""
from pathlib import Path
import json
import pandas as pd
from content import ROOT,j,p,h,eq,im,table,pages
import content
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
    global R
    selection=json.loads((ROOT/'results/final_run_selection.json').read_text())
    R=ROOT/selection['report_results'];content.R=R
    old=pages();rf=j('rf_plain');rb=j('rf_balanced');xp=j('xgb_plain');xs=j('xgb_smote')
    ep=j('extra_practice_results');cv=pd.read_csv(R/'feature_cv.csv',index_col=0)
    top=j('top13');env=j('environment');sections=[]
    oa=j('oob_coverage');sup=j('rf_balanced_subsample')
    topout=[]
    for i in range(0,13,4):topout.append(' | '.join(f'{f}: {cv.loc[f,"cv"]:.4f}' for f in top[i:i+4]))
    sections.append(('1(a)-(b)(ii). Data, imputation and feature statistics',[
      p('DSCI 552 Homework 6: code, answers and actual full-run results for Q1-6. Common functions are defined once; seed 552.'),
      code('imports','preparation'),
      table(['Official split','Shape','neg / pos','Missing cells'],[['Training','60000 x 170','59000 / 1000','850015'],['Test','16000 x 170','15625 / 375','228680']]),
      p('Both headers are on line 21 after 20 preamble lines; na denotes missingness. The 170 predictors are numeric float64, identically ordered, and exclude class. Median imputation retains rows and resists extreme counters; methods and the footnote convention are explained on page 5.'),
      p('All 170 CVs use median-imputed training rows and sample SD (ddof=1). Zero means give undefined CVs and are excluded from ranking; all-missing columns are retained with zero fallback and flagged. Neither occurs here. Main A/B imputation is separate; Bt is train-only sensitivity. CV fits preprocessing only on its training fold.'),
      output('Top 13 CV features (visualization only):\n'+'\n'.join(topout)),
      p('CV = sample SD / mean. All 170 predictors enter modeling; the seeded sample below is for page 3 only.'),code('scatter')]))
    sections.append(('1(b)(iii). Correlation and shared scatter-matrix code',[
      code('correlation'),im('correlation_overview',430),
      p('All 60,000 imputed training rows and all 170 features enter Pearson correlation. Plot indices are 1-based; code indices are zero-based. Correlation is marginal/linear, does not establish causation and can change after imputation. Related counters form correlated blocks.'),
      code('full_scatter')]))
    sections.append(('1(b)(iv). Scatter plots: the 13 highest-CV features',[
      code('scatter_call'),im('scatter_report',510),
      p('The lower triangle directly displays all 78 unordered pairs of the 13 selected features, with each pair appearing once. Blue = neg; red = pos. The plot uses all 1,000 positives plus 2,500 randomly sampled negatives, seed 552. This deliberate display sample does not represent prevalence. log(1+x) is used only for visualization, never model fitting.'),
      p('Each panel uses its own log1p data range; numerical ticks are suppressed to keep feature names clear. The notebook also retains the full high-resolution matrix with ticks and diagonal histograms. Scatter plots alone cannot establish statistical significance: overplotting, marginal versus conditional effects, and multiple comparisons prevent an inferential conclusion. High CV is not predictive importance.')]))
    sections.append(('1(b)(iv)-(v). Box plots and imbalance',[
      code('boxes'),im('boxplots13',438),
      p('Boxes use all original training rows, separated by class, after training-median imputation and log(1+x) for display. Outlier symbols are hidden; no rows are removed. Several rare-count features have quartiles at zero, so boxes collapse.'),
      p('Training: 59,000 neg and 1,000 pos (59:1; 1.667% positive). Official test: 15,625 neg and 375 pos (2.344% positive). Both are severely imbalanced. Always-negative prediction has error 1.667%/2.344% but zero positive recall. The 13 plotted predictors do not restrict modeling: all 170 are retained.')]))
    sections.append(('1(c)-(d). Common evaluation and forest training code',[
      code('evaluation','oob_audit','forest','roc_confusion','forest_supplement'),
      p('Imputation methods include deletion, constant/mean/median/mode, nearest neighbors, iterative models and multiple imputation [2]. Median filling resists extremes but distorts dependencies. Separate split medians follow the footnote; Bt is train-only sensitivity, and CV fits only tr. The plain forest has no class weighting. In sklearn 1.9.1, balanced weights n/(2 n_k)=0.5085/30 drive bootstrap sampling; balanced_subsample weights within uniformly drawn trees. Both are established imbalance strategies [3,4]; other parameters match.'),
      p('Predictions use P(pos)>=0.5, including ties. ROC-AUC uses probabilities; AP is average_precision_score. CM rows are true and columns predicted [neg,pos]. OOB averages only trees excluding that row; both-class coverage must be checked. RUN_TRAINING=False disables only PDF-layout fitting; the original full run and separate supplemental fit were actually executed.')]))
    forestrows=[]
    for name,d in [('Plain',rf),('Balanced',rb)]:
      for split in ['train','test']:forestrows.append([name+' / '+split]+[f'{d[split][k]:.6f}' for k in ['error','auc','average_precision','precision','recall','f1']])
    sections.append(('1(c)-(d). Forest outputs and OOB comparison',[
      table(['Model / split','Error','ROC-AUC','AP','Precision','Recall','F1'],forestrows),im('rf_plain_evaluation',520),im('rf_balanced_evaluation',520),
      table(['Forest','Historical all-row','Valid OOB error','Valid n','Full-class OOB?'],
        [[label,f'{d["oob_error"]:.8f}',f'{oa[name]["valid_error"]:.8f}',str(oa[name]['valid_denominator']),
          'Yes' if oa[name]['full_class_coverage'] else 'NO: negative-only']
         for name,label,d in [('rf_plain','Plain',rf),('rf_balanced','Balanced',rb)]]),
      p(f'Plain: {oa["rf_plain"]["valid_errors"]}/60000 OOB errors; test error {rf["test"]["error"]:.8f}, gap {rf["test"]["error"]-oa["rf_plain"]["valid_error"]:+.8f}. Balanced: all 1000 positives lack OOB votes and have [0,0] probabilities. The historical 1199/60000 counts them all as negatives/errors. Its valid-row value is {oa["rf_balanced"]["valid_errors"]}/59000, NEGATIVE-ONLY; it cannot estimate overall error or be compared with full test error {rb["test"]["error"]:.8f}. Neither model has NaN or other invalid covered rows. Coverage/details and a separate valid supplement follow on pages 7-8.'),
      p('Existing train/test scores above remain unchanged. Weighting slightly increases minority recall but lowers precision and increases overall error. Global training imputation makes otherwise valid OOB an approximate internal estimate; training scores are optimistic resubstitution results.')]))
    sections.append(('1(e)-(f). Pipeline; coverage-aware RF OOB review',[
      p('XGBoost '+env['versions']['xgboost']+' gbtree uses feature-threshold splits and scalar leaf weights. binary:logistic is the logistic objective/link; reg_alpha penalizes leaf weights, not node-level logistic-regression coefficients [5,6]. gblinear is a global linear booster, not a model tree. Local course material gives no clarifying API mapping. The runnable standard-XGBoost interpretation is retained; a genuine logistic model tree remains an instructor question.'),
      eq(r'P(Y=1\mid x)=\sigma(\sum_t f_t(x)),\quad\Omega(f_t)=\gamma L_t+\frac{\lambda}{2}\sum_\ell w_{t\ell}^2+\alpha\sum_\ell |w_{t\ell}|'),
      code('xgb_pipeline'),
      p('The baseline has no imbalance compensation. Shared standardization makes SMOTE distances less dominated by large-unit counters; it cannot remove skew or high-dimensional distance limitations.'),
      p('Pipeline.fit fits preprocessing and SMOTE only on tr; predict_proba transforms va without resampling. Test data never enter tuning. Rounds250/threshold0.5 and other settings are fixed; no early stopping or nested parallel CV.'),
      h('RF OOB coverage audit and supplemental method'),
      table(['Forest / class','Zero votes','Valid / n','Coverage','Vote range / mean'],
        [[label+' / '+cls,str(oa[name]['by_class'][cls]['zero_votes']),
          f"{oa[name]['by_class'][cls]['valid']} / {oa[name]['by_class'][cls]['n']}",
          f"{oa[name]['by_class'][cls]['valid_fraction']:.1%}",
          f"{oa[name]['by_class'][cls]['min_votes']}-{oa[name]['by_class'][cls]['max_votes']} / {oa[name]['by_class'][cls]['mean_votes']:.2f}"]
         for name,label in [('rf_plain','Plain'),('rf_balanced','Balanced'),('rf_balanced_subsample','Supplement')]
         for cls in ['neg','pos']]),
      p('Actual estimators_samples_ indices were counted for every tree and all stored OOB probabilities independently reconstructed from tree predictions. Overall valid coverage: plain 60000/60000; balanced 59000/60000 (98.3333%); supplement 60000/60000. In sklearn 1.9.1, balanced sampling gives each positive probability 1/2000 per draw: 30 expected draws/tree and near-zero exclusion probability. The generic too-few-trees warning comes from balanced, not plain. Dropping missing positives does not solve the coverage problem.'),
      performance(sup),
      p(f'The separate balanced_subsample forest preserves all other settings, uses uniform bootstraps plus per-tree balanced impurity weights, and covers both classes. OOB={sup["oob"]["valid_errors"]}/60000={sup["oob"]["valid_error"]:.8f}; test error={sup["test"]["error"]:.8f}; gap={sup["test"]["error"]-sup["oob"]["valid_error"]:+.8f}. It is not a replacement or test-selected improvement. Confusion matrices and ROC follow on page 8.')]))
    ca=pd.read_csv(R/'xgb_plain_cv_summary.csv');cb=pd.read_csv(R/'xgb_smote_cv_summary.csv')
    searchrows=[[f'{a.alpha:g}',f'{a.mean_ap:.6f}',f'{a.mean_error:.6f}',f'{b.mean_ap:.6f}',f'{b.mean_error:.6f}'] for a,b in zip(ca.itertuples(),cb.itertuples())]
    sections.append(('1(e)-(f). CV outputs; supplemental forest evaluation',[
      code('search'),table(['Alpha','Plain mean AP','Plain CV error','SMOTE mean AP','SMOTE CV error'],searchrows),
      p('Both searches use all training rows, identical stratified five-fold splits and fold-local preprocessing. Select mean AP; smaller-alpha ties. No test-based selection.'),
      p(f'Selected plain alpha={xp["selected_alpha"]:g} (no L1 penalty), SMOTE alpha={xs["selected_alpha"]:g}. Earlier full-run CV times: {pd.read_csv(R/"xgb_plain_cv_folds.csv").seconds.sum():.1f}/{pd.read_csv(R/"xgb_smote_cv_folds.csv").seconds.sum():.1f}s. Selected CV scores are tuning estimates, not unbiased nested validation.'),
      h('Separate balanced_subsample forest: actual outputs'),
      im('rf_balanced_subsample_evaluation',520)]))
    pos=j('xgb_positive')
    sections.append(('1(e). Full training and uncompensated XGBoost results',[
      code('final_fit'),performance(xp),im('xgb_plain_evaluation',510),
      p(f'No compensation; alpha={xp["selected_alpha"]:g}. Test minus CV error = {xp["test"]["error"]-xp["cv"]["mean_error"]:+.6f}. CV uses 80% fitting rows and fold-local preprocessing; the final fit uses all rows and the test-imputation footnote. Current full-run final fit/evaluation: {xp["final_fit_and_evaluate_seconds"]:.2f}s. Training scores use original rows.'),
      h('Supplementary comparison restricted to alpha > 0'),code('positive_alpha'),
      table(['Split','alpha','CV error','Error','ROC-AUC','CM: [[TN,FP],[FN,TP]]'],
        [[split,f'{pos["selected_alpha"]:g}',f'{pos["cv"]["mean_error"]:.6f}',
          f'{pos[split]["error"]:.6f}',f'{pos[split]["auc"]:.6f}',
          str(pos[split]['confusion_matrix'])] for split in ['train','test']]),
      p(f'Restrict the same CV AP table to alpha > 0; the same smaller-alpha tie rule selects {pos["selected_alpha"]:g} (CV AP {pos["cv"]["mean_ap"]:.6f}). A separate all-60,000-row, no-SMOTE model was actually fitted with every other setting fixed. Test error difference versus alpha={xp["selected_alpha"]:g}: {pos["test"]["error"]-xp["test"]["error"]:+.6f}; AUC difference: {pos["test"]["auc"]-xp["test"]["auc"]:+.6f}. It does not replace the unrestricted optimum or resolve the node-level model-tree ambiguity.')]))
    sections.append(('1(f). Full-data SMOTE + XGBoost results',[
      code('smote_call'),
      eq(r'x_{new}=x_i+u(x_{neighbor}-x_i),\qquad u\sim U(0,1)'),
      table(['Stage','neg','Original pos','Final pos','Total'],[['Each fold fit','47200','800','47200','94400'],['Full final fit','59000','1000','59000','118000'],['Validation (unchanged)','11800','200','200','12000'],['Test (unchanged)','15625','375','375','16000']]),
      performance(xs),im('xgb_smote_evaluation',510),
      p(f'SMOTE [7] uses five minority neighbors, seed 552, sampling_strategy=1.0, after fold-local imputation and standardization. There is no majority downsampling. Final full-data resampling took {xs["resampling_seconds"]:.3f}s; final scaling, resampling, fit/evaluation took {xs["final_fit_and_evaluate_seconds"]:.2f}s. All full-data and fold fits were executed anew in the isolated full run. Original training/test distributions are used for all displayed evaluations.'),
      p(f'Alpha = {xs["selected_alpha"]:g}; test error {xs["test"]["error"]:.6f} minus selected CV error {xs["cv"]["mean_error"]:.6f} = {xs["test"]["error"]-xs["cv"]["mean_error"]:.6f}. Compared with plain XGB, recall changes {xp["test"]["recall"]:.4f} to {xs["test"]["recall"]:.4f}, precision {xp["test"]["precision"]:.4f} to {xs["test"]["precision"]:.4f}; FP {xp["test"]["confusion_matrix"][0][1]} to {xs["test"]["confusion_matrix"][0][1]}, FN {xp["test"]["confusion_matrix"][1][0]} to {xs["test"]["confusion_matrix"][1][0]}. Oversampling changes the fitted prior, so scores need not be calibrated population probabilities. The threshold was not optimized on the test set. Resampling before splitting would leak neighbor information [8].')]))
    names=[('rf_plain','RF plain'),('rf_balanced','RF balanced'),('xgb_plain','XGB plain'),('xgb_smote','XGB SMOTE')]
    rows=[[label]+[f'{j(n)["test"][k]:.5f}' for k in ['error','auc','average_precision','precision','recall','f1']] for n,label in names]
    sens=[[label,f'{j(n)["test"]["error"]:.6f}',f'{j(n)["test_train_only_imputation"]["error"]:.6f}',f'{j(n)["test"]["average_precision"]:.5f}',f'{j(n)["test_train_only_imputation"]["average_precision"]:.5f}'] for n,label in names]
    sections.append(('1(c)-(f). Unified comparison and imputation sensitivity',[
      code('comparison'),table(['Model / test','Error','ROC-AUC','AP','Precision','Recall','F1'],rows),im('model_comparison',478),
      table(['Model','Main error','Train-only error','Main AP','Train-only AP'],sens),
      p(f'Sensitivity holds every model and scaler fixed, replacing only test-median B with train-median Bt. No result in this table selects a model or threshold. SMOTE main versus train-only error: {xs["test"]["error"]:.6f} versus {xs["test_train_only_imputation"]["error"]:.6f}; FP {xs["test"]["confusion_matrix"][0][1]} versus {xs["test_train_only_imputation"]["confusion_matrix"][0][1]}, FN {xs["test"]["confusion_matrix"][1][0]} versus {xs["test_train_only_imputation"]["confusion_matrix"][1][0]}. This material change shows how synthetic-neighbor fits can amplify shifts in imputed values.'),
      p(f'ROC-AUC measures ranking across both classes; AP summarizes precision along minority recall and depends on prevalence. The dashed PR baseline is test prevalence 0.02344. The data description [9] gives cost 10 FP + 500 FN. Retrospective test costs (RF plain/balanced, XGB plain/SMOTE): {10*rf["test"]["confusion_matrix"][0][1]+500*rf["test"]["confusion_matrix"][1][0]}, {10*rb["test"]["confusion_matrix"][0][1]+500*rb["test"]["confusion_matrix"][1][0]}, {10*xp["test"]["confusion_matrix"][0][1]+500*xp["test"]["confusion_matrix"][1][0]}, {10*xs["test"]["confusion_matrix"][0][1]+500*xs["test"]["confusion_matrix"][1][0]}. Costs did not guide tuning.')]))
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
      p('Python '+env['python']+'; '+', '.join(f'{k} {v}' for k,v in env['versions'].items())+'. CPU only; seed 552; estimator threads 4 and sequential folds. Saved full-data fits, probability outputs and fold search files supply every numeric model table. The isolated full run fitted 2 forests, 70 CV estimators, 2 selected final XGB models and 1 positive-alpha supplement using only raw CSVs and the inline notebook. The self-contained notebook has full-from-raw and fingerprint-validated quick modes. A clean-kernel quick run reproduced all five original final-model train/test/sensitivity predictions and recomputed metrics; saved CV summaries/selection were checked without refitting CV estimators. The earlier full snapshot executed 75 fits. OOB review subsequently identified invalid balanced OOB coverage, retained the old arithmetic, and fitted only a separate balanced_subsample forest. The revised notebook was executed in quick mode with independent original-model and OOB-audit fingerprints. Historical models, XGB/SMOTE and all 70 CV results remain unchanged.'),
      p('The displayed code is real Python from submission_core.py; it includes data loading/imputation, all feature statistics, plots, common evaluation, both forests, fold-local SMOTE/Pipeline, alpha search/selection, final XGBoost fits and all computational exercises. RUN_TRAINING=False disables fitting only during PDF layout; the displayed fitting logic is exercised by full notebook execution. Optional code for environment discovery, caching, serialization, report layout and repeated validation remains in the project.'),
      p('Homework6.pdf, the local August 24, 2026 syllabus and course introduction were checked. No explicit font-size or margin rule was found there. Body 10.4 pt, monospaced code 8.7 pt and 36-pt side margins are layout choices, not claimed course mandates. All content in the formal PDF counts toward its 20-page limit. Separately attached notebook/source pages are not added, and the notice does not require printing the entire project. Necessary implementation and results remain here; supporting code stays in the notebook/project.'),
      p('Edition cross-check: Q2/Q3 use second-edition 6.6.3/6.6.5; Q4/Q5 and Q6 5.4.2/8.4.4/9.7.2 agree across checked editions. Q6 6.8.4 explicitly uses the first-edition ridge exercise, equivalent to second-edition 6.6.4. Full source-page evidence is retained. The remaining instructor interpretation issue concerns node-level logistic model trees versus standard XGBoost, accurately distinguished on page 7.')]))
    assert len(sections)==20
    return sections
