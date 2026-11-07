"""Real, compact core implementation printed in the PDF.
RUN_TRAINING stays False during report revision. Existing APS fits are reused.
The training branches implement the original experiment, without cache/I/O code.
"""
# %% imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics as M
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
SEED = 552
RUN_TRAINING = False  # Original APS fits are not rerun in this revision.
# %% preparation
def read_aps(split):
    path = Path('Homework 6 Data') / f'aps_failure_{split}_set.csv'
    with path.open() as stream:
        h = next(i for i, s in enumerate(stream) if s.startswith('class,'))
    frame = pd.read_csv(path, skiprows=h, na_values=['na'])
    assert frame['class'].isin(['pos', 'neg']).all()
    return (frame.drop(columns='class').astype(float),
            frame['class'].map({'neg': 0, 'pos': 1}).to_numpy())
X, y = read_aps('training')
T, yt = read_aps('test')
assert X.shape == (60000, 170) and T.shape == (16000, 170)
assert X.columns.equals(T.columns)
imp = SimpleImputer(strategy='median', keep_empty_features=True)
A = imp.fit_transform(X)
B = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(T)
Bt = imp.transform(T)  # Sensitivity only: train-fitted imputation.
D = pd.DataFrame(A, columns=X.columns)
cv = D.std(ddof=1) / D.mean().replace(0, np.nan)
top = cv.nlargest(13).index.tolist()
# %% correlation
corr = D.corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set(xticks=np.arange(0, 170, 10), yticks=np.arange(0, 170, 10))
fig.colorbar(im, ax=ax, label='Pearson correlation')
# %% full_scatter
def full_scatter(Z, ys, columns, colors):
    fig, axes = plt.subplots(13, 13, figsize=(22, 22))
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            ax = axes[i, j]
            if i == j:
                for c, color in [(0, '#3378a6'), (1, '#d34c49')]:
                    ax.hist(Z[a].to_numpy()[ys == c], bins=20,
                            histtype='step', density=True, color=color)
            else:
                ax.scatter(Z[b], Z[a], c=colors, s=1.5, alpha=.25)
            if j == 0: ax.set_ylabel(a, fontsize=8)
            if i == 12: ax.set_xlabel(b, fontsize=8)
    return fig
# %% scatter
rng = np.random.default_rng(SEED)
idx = np.r_[rng.choice(np.flatnonzero(y == 0), 2500, replace=False),
            np.flatnonzero(y == 1)]
Z = np.log1p(D.iloc[idx][top])
ys = y[idx]
colors = np.where(ys == 1, '#d34c49', '#3378a6')
# Full matrix: full_scatter(Z, ys, top, colors).
fig, axes = plt.subplots(4, 3, figsize=(9, 10))
for ax, col in zip(axes.flat, top[1:]):
    ax.scatter(Z[top[0]], Z[col], c=colors, s=3, alpha=.3)
    ax.set(xlabel=f'log(1+{top[0]})', ylabel=f'log(1+{col})')
fig.tight_layout()
# %% boxes
fig, axes = plt.subplots(5, 3, figsize=(10, 10))
for ax, col in zip(axes.flat, top):
    groups = [np.log1p(D.loc[y == c, col]) for c in [0, 1]]
    ax.boxplot(groups, tick_labels=['neg', 'pos'], showfliers=False)
    ax.set(title=col, ylabel='log(1+x)')
for ax in list(axes.flat)[13:]: ax.axis('off')
fig.tight_layout()
# %% evaluation
def evaluate(labels, probability):
    pred = (probability >= .5).astype(int)
    return dict(error=float(np.mean(pred != labels)),
        auc=M.roc_auc_score(labels, probability),
        average_precision=M.average_precision_score(labels, probability),
        precision=M.precision_score(labels, pred, zero_division=0),
        recall=M.recall_score(labels, pred), f1=M.f1_score(labels, pred),
        confusion_matrix=M.confusion_matrix(labels, pred, labels=[0, 1]))

def evaluate_splits(model, train, test, test_train_only):
    return {'train': evaluate(y, model.predict_proba(train)[:, 1]),
            'test': evaluate(yt, model.predict_proba(test)[:, 1]),
            'train_only_test': evaluate(yt,
                model.predict_proba(test_train_only)[:, 1])}
# %% forest
def train_forests():
    fitted, scores = {}, {}
    for name, weight in [('rf_plain', None), ('rf_balanced', 'balanced')]:
        model = RandomForestClassifier(n_estimators=500, criterion='gini',
            max_features='sqrt', max_depth=None, min_samples_leaf=1,
            bootstrap=True, oob_score=True, class_weight=weight,
            random_state=SEED, n_jobs=4)
        model.fit(A, y)
        scores[name] = evaluate_splits(model, A, B, Bt)
        oob = model.oob_decision_function_[:, 1]
        scores[name]['oob_error'] = float(np.mean((oob >= .5) != y))
        fitted[name] = model
    return fitted, scores
if RUN_TRAINING:
    forests, forest_scores = train_forests()
# %% roc_confusion
def draw_evaluation(labels, probability, cm_axis, roc_axis):
    cm = M.confusion_matrix(labels, probability >= .5, labels=[0, 1])
    M.ConfusionMatrixDisplay(cm, display_labels=['neg', 'pos']).plot(ax=cm_axis)
    fpr, tpr, _ = M.roc_curve(labels, probability)
    roc_axis.plot(fpr, tpr)
# %% xgb_pipeline
ALPHAS = [0., .001, .01, .1, 1., 10., 100.]
def make_xgb(alpha):
    return XGBClassifier(booster='gbtree', objective='binary:logistic',
        reg_alpha=alpha, reg_lambda=1, n_estimators=250, max_depth=4,
        learning_rate=.08, subsample=1, colsample_bytree=1, min_child_weight=1,
        tree_method='hist', max_bin=256, base_score=.5, eval_metric='logloss',
        random_state=SEED, n_jobs=4)
def fold_pipeline(alpha, use_smote):
    steps = [('imputer', SimpleImputer(strategy='median',
                                     keep_empty_features=True)),
             ('scaler', StandardScaler())]
    if use_smote:
        steps += [('smote', SMOTE(random_state=SEED, k_neighbors=5,
                                  sampling_strategy=1.0))]
    return Pipeline(steps + [('estimator', make_xgb(alpha))])
# %% search
def search_alpha(use_smote):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for alpha in ALPHAS:
        for fold, (tr, va) in enumerate(folds.split(X, y), 1):
            pipe = fold_pipeline(alpha, use_smote)
            pipe.fit(X.iloc[tr], y[tr])  # Only this training fold is fitted.
            p = pipe.predict_proba(X.iloc[va])[:, 1]  # Never resampled.
            rows.append(dict(alpha=alpha, fold=fold,
                ap=M.average_precision_score(y[va], p),
                error=float(np.mean((p >= .5) != y[va])),
                auc=M.roc_auc_score(y[va], p)))
    scores = pd.DataFrame(rows)
    summary = scores.groupby('alpha').agg(mean_ap=('ap', 'mean'),
        sd_ap=('ap', 'std'), mean_error=('error', 'mean'),
        sd_error=('error', 'std'), mean_auc=('auc', 'mean')).reset_index()
    best = summary.sort_values(['mean_ap', 'alpha'],
                               ascending=[False, True]).iloc[0]['alpha']
    return scores, summary, float(best)
if RUN_TRAINING:
    plain_folds, plain_summary, alpha0 = search_alpha(False)
    smote_folds, smote_summary, alpha1 = search_alpha(True)
# %% final_fit
def train_final(alpha, use_smote):
    import time
    scaler = StandardScaler()
    As = scaler.fit_transform(A)
    Bs, Bts = scaler.transform(B), scaler.transform(Bt)
    start = time.perf_counter()
    if use_smote:
        Ar, yr = SMOTE(random_state=SEED, k_neighbors=5,
                       sampling_strategy=1.0).fit_resample(As, y)
    else:
        Ar, yr = As, y
    sampling_seconds = time.perf_counter() - start
    model = make_xgb(alpha)
    model.fit(Ar, yr)
    # Evaluation always uses original rows, including for the training scores.
    scores = evaluate_splits(model, As, Bs, Bts)
    counts = np.bincount(yr, minlength=2)
    return model, scaler, scores, counts, sampling_seconds
if RUN_TRAINING:
    plain_final = train_final(alpha0, False)
# %% smote_call
if RUN_TRAINING:
    smote_final = train_final(alpha1, True)
    print(smote_final[3])  # Final class counts: [59000, 59000].
# %% comparison
def pr_curves(predictions):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, p in predictions.items():
        precision, recall, _ = M.precision_recall_curve(yt, p)
        ap = M.average_precision_score(yt, p)
        ax.plot(recall, precision, label=f'{name}: AP={ap:.3f}')
    ax.axhline(yt.mean(), ls='--', color='black')
    ax.set(xlabel='Recall', ylabel='Precision')
    ax.legend()
    return fig
# Bt (p.1) and evaluate_splits (p.5) produce the sensitivity evaluation.
# %% q3_check
x = np.array([1., -1.]); response = np.array([2., -2.]); penalty = 2.
a = x @ x; c = x @ response
ridge = c / (2*a + penalty)
tstar = np.sign(c) * max(abs(c) - penalty/2, 0) / a
objectives = [np.sum((response - x*tstar)**2) + penalty *
              (abs(w*tstar) + abs((1-w)*tstar)) for w in np.linspace(0, 1, 11)]
assert np.ptp(objectives) < 1e-12
# %% q4
prob = np.array([.1, .15, .2, .2, .55, .6, .6, .65, .7, .75])
red_votes = int((prob > .5).sum())
red_probability = float(prob.mean())
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(np.arange(1, 11), prob)
ax.axhline(.5, color='black', ls='--')
ax.axhline(red_probability, color='purple')
ax.set(xlabel='Bootstrap tree', ylabel='P(Red)', ylim=(0, 1))
# %% q5
from sklearn.svm import SVC
points = np.array([[3,4], [2,2], [4,4], [1,4], [2,1], [4,3], [4,1]])
labels = np.array([1, 1, 1, 1, -1, -1, -1])
svm = SVC(C=1e6, kernel='linear', tol=1e-10).fit(points, labels)
assert set(svm.support_ + 1) == {2, 3, 5, 6}
margin = 1 / np.linalg.norm(svm.coef_[0])
# %% q5_plot
def geometry_plot(points, labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    for cls, color in [(1, '#d34c49'), (-1, '#3378a6')]:
        ax.scatter(*points[labels == cls].T, c=color)
    for i, point in enumerate(points, 1): ax.annotate(str(i), point)
    xx = np.linspace(.5, 4.8, 200)
    for offset, style in [(-.5, '-'), (0, '--'), (-1, '--')]:
        ax.plot(xx, xx + offset, color='black', ls=style)
    ax.plot(xx, xx-.2, ':', color='purple')
    ax.scatter([3], [3], c='#3378a6', marker='X')  # Added Blue point for (h).
    ax.set(xlim=(.5, 4.8), ylim=(.3, 4.8), xlabel='X1', ylabel='X2', aspect='equal')
    return fig
# %% q5_plot_call
geometry_plot(points, labels)
# %% bootstrap
n = np.arange(1, 100001)
included = np.ones(len(n))
included[1:] = -np.expm1(n[1:] * np.log1p(-1/n[1:]))
rng = np.random.default_rng(SEED)
samples = rng.integers(1, 101, size=(10000, 100))
hits = np.any(samples == 4, axis=1)
empirical = float(hits.mean())
exact = float(included[99])
mcse = np.sqrt(exact * (1-exact) / 10000)
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.plot(n, included); ax.set_xscale('log')
ax.axhline(1-np.exp(-1), color='red', ls='--')
ax.set(xlabel='n (all integers 1 through 100000)', ylabel='Inclusion probability')
# %% trees
TA = (0, 1, (1, 1, (0, 0, 3, (1, 0, 10, 0)), 15), 5)
TB = (1, 1, (0, 1, -1.80, .63), (1, 2, (0, 0, -1.06, .21), 2.49))
def leaf(t): return not isinstance(t, tuple)
def leaves(t): return 1 if leaf(t) else leaves(t[2]) + leaves(t[3])
def tree_value(t, x):
    if leaf(t): return t
    j, cut, yes, no = t
    return tree_value(yes if x[j] < cut else no, x)
def draw_tree(ax, t, lo=0, hi=1, depth=0):
    x, y0 = (lo+hi)/2, -depth
    if leaf(t):
        text = f'{t:g}'
    else:
        j, cut, yes, no = t
        text = f'X{j+1} < {cut:g}'
        mid = lo + (hi-lo)*leaves(yes)/leaves(t)
        for branch, a, b, label in [(yes,lo,mid,'yes'), (no,mid,hi,'no')]:
            cx, cy = draw_tree(ax, branch, a, b, depth+1)
            ax.plot([x,cx], [y0,cy], color='#536879', zorder=0)
            ax.text((x+cx)/2, (y0+cy)/2, label, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
    ax.text(x, y0, text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=.2', facecolor='#e5edf5'))
    return x, y0
RA = [(-1,-1,1,2,3), (0,-1,1,1,10), (0,0,1,1,0),
      (-1,1,2,2,15), (1,-1,2,4,5)]
RB = [(-1,-1,2,2,-1.80), (1,-1,2,2,.63), (-1,1,1,1,-1.06),
      (0,1,3,1,.21), (-1,2,4,1,2.49)]
def tree_partition(t, regions):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    draw_tree(axes[0], t); axes[0].axis('off')
    for x, y0, w, h, mean in regions:
        axes[1].add_patch(plt.Rectangle((x,y0), w, h, facecolor='#e5edf5',
                                       edgecolor='#536879'))
        axes[1].text(x+w/2, y0+h/2, f'{mean:g}', ha='center', va='center')
    axes[1].set(xlim=(-1,3), ylim=(-1,3), xlabel='X1', ylabel='X2', aspect='equal')
    fig.tight_layout()
    return fig
fig_a = tree_partition(TA, RA)
fig_b = tree_partition(TB, RB)
# %% circle
query = np.array([[0,0], [-1,1], [2,2], [3,8]])
distance2 = ((query - [-1,2])**2).sum(axis=1)
score = query[:,0]**2 + query[:,1]**2 + 2*query[:,0] - 4*query[:,1] + 1
classes = np.where(score > 0, 'Blue', 'Red')
assert np.array_equal(score, distance2 - 4)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
for ax in axes:
    ax.set_facecolor('#e5eefb')
    ax.add_patch(plt.Circle((-1,2), 2, facecolor='#f6d9db', edgecolor='black'))
    ax.set(xlabel='X1', ylabel='X2', aspect='equal')
axes[0].set(xlim=(-3.8,1.8), ylim=(-.8,4.8))
axes[0].text(-1, 2.7, 'Red: distance squared <= 4', ha='center', fontsize=9)
axes[1].set(xlim=(-3.8,4.4), ylim=(-1,9))
for point, cls in zip(query, classes):
    axes[1].scatter(*point, c='red' if cls == 'Red' else 'blue')
    axes[1].annotate(f'({point[0]},{point[1]})', point, xytext=(4,5),
                     textcoords='offset points', fontsize=10)
fig.tight_layout()
# %% end
