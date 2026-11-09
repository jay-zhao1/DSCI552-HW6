"""Source embedded directly in the notebook; no project module imports at runtime."""

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return 'NaN missing sentinel'
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False))

def load_data():
    pairs, audit = [], {}
    for split in ['training', 'test']:
        path = DATA_DIR / f'aps_failure_{split}_set.csv'
        with path.open() as stream:
            header = next(i for i, line in enumerate(stream) if line.startswith('class,'))
        frame = pd.read_csv(path, skiprows=header, na_values=['na'])
        assert frame['class'].isin(['neg', 'pos']).all()
        x = frame.drop(columns='class').astype(float)
        labels = frame['class'].map({'neg': 0, 'pos': 1}).to_numpy()
        audit[split] = dict(rows=len(x), features=x.shape[1],
            header_line_1based=header+1, dtype=str(x.dtypes.iloc[0]),
            counts=dict(neg=int((labels == 0).sum()), pos=int((labels == 1).sum())),
            missing_cells=int(x.isna().sum().sum()),
            all_missing_columns=x.columns[x.isna().all()].tolist(), sha256=sha256_file(path))
        pairs.append((x, labels))
    assert pairs[0][0].shape == (60000, 170) and pairs[1][0].shape == (16000, 170)
    assert pairs[0][0].columns.equals(pairs[1][0].columns)
    return *pairs, audit

def prepare(x, test):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    train = imputer.fit_transform(x)
    main_test = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(test)
    sensitivity_test = imputer.transform(test)
    assert train.shape[1] == 170 and np.isfinite(train).all()
    return imputer, train, main_test, sensitivity_test

def metrics(labels, probability):
    pred = (probability >= CONFIG['threshold']).astype(int)
    return dict(n=len(labels), error=float(np.mean(pred != labels)),
        auc=float(roc_auc_score(labels, probability)),
        average_precision=float(average_precision_score(labels, probability)),
        precision=float(precision_score(labels, pred, zero_division=0)),
        recall=float(recall_score(labels, pred)), f1=float(f1_score(labels, pred)),
        confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1]).tolist())

def make_rf(balanced):
    return RandomForestClassifier(**CONFIG['rf'],
        class_weight='balanced' if balanced else None)

def xgb(alpha):
    return XGBClassifier(**CONFIG['xgb'], reg_alpha=float(alpha))

def fold_pipeline(alpha, smote):
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('scaler', StandardScaler())]
    if smote:
        steps.append(('smote', SMOTE(**CONFIG['smote'])))
    return Pipeline(steps + [('estimator', xgb(alpha))])

def training_folds(x, labels):
    return list(StratifiedKFold(**CONFIG['cv']).split(x, labels))

def choose_alpha(frame):
    summary = frame.groupby('alpha').agg(mean_ap=('average_precision', 'mean'),
        sd_ap=('average_precision', 'std'), mean_error=('error', 'mean'),
        sd_error=('error', 'std'), mean_auc=('auc', 'mean'), seconds=('seconds', 'sum')).reset_index()
    best = float(summary.sort_values(['mean_ap', 'alpha'],
        ascending=[False, True]).iloc[0]['alpha'])
    return summary, best

def cache_signature():
    return dict(schema=2, data={s: sha256_file(DATA_DIR / f'aps_failure_{s}_set.csv')
        for s in ['training', 'test']}, config=CONFIG,
        model_code_sha256=hashlib.sha256(MODEL_CODE.strip().encode()).hexdigest(),
        versions={p: importlib.metadata.version(p) for p in
            ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'imbalanced-learn', 'joblib']})

def validate_cache(directory, expected):
    directory = Path(directory)
    if not (directory / 'manifest.json').exists():
        raise FileNotFoundError(f'No validated cache at {directory}. Set MODE="full" '
            'to train from raw data, or supply the separately validated cache bundle.')
    manifest = read_json(directory / 'manifest.json')
    if manifest['signature'] != expected:
        raise ValueError('Cache does not match current raw data, parameters, code or '
                         'package versions. Select MODE="full" or the matching cache.')
    for file, digest in manifest['files'].items():
        if not (directory / file).is_file() or sha256_file(directory / file) != digest:
            raise ValueError(f'Missing or altered cached artifact: {file}')
    required = ['cv_folds.npz']
    for name in MODEL_NAMES:
        required += [f'{name}.joblib', f'{name}.json', f'{name}_predictions.npz']
    for name in ['xgb_plain', 'xgb_smote']:
        required += [f'{name}_cv_folds.csv', f'{name}_cv_summary.csv']
    if not set(required).issubset(manifest['files']):
        raise ValueError('Cache inventory is incomplete; use MODE="full".')
    return manifest

def seal_cache(directory, signature, provenance):
    files = {p.name: sha256_file(p) for p in Path(directory).iterdir()
             if p.suffix in ['.json', '.csv', '.npz', '.joblib'] and p.name != 'manifest.json'}
    write_json(Path(directory) / 'manifest.json',
               dict(signature=signature, files=files, provenance=provenance))

def choose_positive_alpha(summary):
    eligible = summary.loc[summary.alpha > 0]
    if eligible.empty:
        raise ValueError('No strictly positive alpha was searched.')
    return float(eligible.sort_values(['mean_ap', 'alpha'],
        ascending=[False, True]).iloc[0]['alpha'])

def fit_all_from_raw(directory, x, labels, test, test_labels):
    # Full mode never reads a cache or skips a fit. directory must be new/empty.
    directory = Path(directory)
    assert not any(directory.iterdir()), 'Full training requires a fresh directory.'
    events = []
    def fitted_event(**event):
        events.append(event)
        write_json(directory / 'training_events.json', events)
    imputer, a, b, bt = prepare(x, test)
    folds = training_folds(x, labels)
    np.savez_compressed(directory / 'cv_folds.npz',
                       **{f'validation_{i}': v for i, (_, v) in enumerate(folds, 1)})
    for balanced, name in [(False, 'rf_plain'), (True, 'rf_balanced')]:
        start = time.perf_counter()
        model = make_rf(balanced)
        model.fit(a, labels)
        fitted_event(model=name, kind="forest", rows=len(labels), features=a.shape[1])
        pt, pv, ps = [model.predict_proba(z)[:, 1] for z in [a, b, bt]]
        result = dict(train=metrics(labels, pt), test=metrics(test_labels, pv),
            test_train_only_imputation=metrics(test_labels, ps),
            oob_error=float(np.mean((model.oob_decision_function_[:, 1] >= .5) != labels)),
            fit_and_evaluate_seconds=time.perf_counter()-start, parameters=model.get_params())
        np.savez_compressed(directory / f'{name}_predictions.npz',
                            train=pt, test=pv, test_train_only=ps)
        dump(model, directory / f'{name}.joblib', compress=3)
        write_json(directory / f'{name}.json', result)
        print('Trained from raw:', name, flush=True)
    for smote, name in [(False, 'xgb_plain'), (True, 'xgb_smote')]:
        rows = []
        for alpha in CONFIG['alphas']:
            for fold, (tr, va) in enumerate(folds, 1):
                start = time.perf_counter()
                pipe = fold_pipeline(alpha, smote)
                pipe.fit(x.iloc[tr], labels[tr])
                fitted_event(model=name, kind="cv", alpha=alpha, fold=fold,
                             fit_rows=len(tr), validation_rows=len(va),
                             resampled_rows=94400 if smote else len(tr))
                probability = pipe.predict_proba(x.iloc[va])[:, 1]
                score = metrics(labels[va], probability)
                rows.append(dict(alpha=alpha, fold=fold,
                    **{k: v for k, v in score.items() if k != 'confusion_matrix'},
                    seconds=time.perf_counter()-start, original_fold_n=len(tr),
                    resampled_fold_n=94400 if smote else len(tr)))
        frame = pd.DataFrame(rows)
        summary, best = choose_alpha(frame)
        frame.to_csv(directory / f'{name}_cv_folds.csv', index=False)
        summary.to_csv(directory / f'{name}_cv_summary.csv', index=False)
        start = time.perf_counter()
        scaler = StandardScaler()
        a_s = scaler.fit_transform(a)
        b_s, bt_s = scaler.transform(b), scaler.transform(bt)
        sampling_start = time.perf_counter()
        if smote:
            a_r, y_r = SMOTE(**CONFIG['smote']).fit_resample(a_s, labels)
        else:
            a_r, y_r = a_s, labels
        sampling_seconds = time.perf_counter()-sampling_start
        model = xgb(best)
        model.fit(a_r, y_r)
        fitted_event(model=name, kind="final_xgb", alpha=best, rows=len(y_r))
        pt, pv, ps = [model.predict_proba(z)[:, 1] for z in [a_s, b_s, bt_s]]
        result = dict(selected_alpha=best,
            selection_metric='mean validation average precision; smaller-alpha tie break',
            cv=summary[summary.alpha == best].iloc[0].to_dict(),
            train=metrics(labels, pt), test=metrics(test_labels, pv),
            test_train_only_imputation=metrics(test_labels, ps),
            resampling_seconds=sampling_seconds,
            final_fit_and_evaluate_seconds=time.perf_counter()-start,
            resampled_counts=dict(neg=int((y_r == 0).sum()), pos=int((y_r == 1).sum())),
            parameters=model.get_params())
        np.savez_compressed(directory / f'{name}_predictions.npz',
                            train=pt, test=pv, test_train_only=ps)
        dump(dict(imputer=imputer, scaler=scaler, model=model),
             directory / f'{name}.joblib', compress=3)
        write_json(directory / f'{name}.json', result)
        print('Trained 35 CV fits + final from raw:', name, flush=True)
    # Supplement uses the same plain-search CV rows, restricted to alpha > 0.
    # No official test score enters this choice.
    plain_summary = pd.read_csv(directory / 'xgb_plain_cv_summary.csv')
    best = choose_positive_alpha(plain_summary)
    start = time.perf_counter()
    scaler = StandardScaler().fit(a)
    a_s, b_s, bt_s = [scaler.transform(z) for z in [a, b, bt]]
    model = xgb(best)
    model.fit(a_s, labels)
    fitted_event(model='xgb_positive', kind='positive_alpha_supplement',
                 alpha=best, rows=len(labels))
    pt, pv, ps = [model.predict_proba(z)[:, 1] for z in [a_s, b_s, bt_s]]
    result = dict(selected_alpha=best,
        label='Supplementary comparison restricted to alpha > 0; not original optimum',
        selection_metric='same mean validation AP, restricted alpha > 0; smaller-alpha ties',
        cv_source='xgb_plain_cv_folds.csv',
        cv=plain_summary[plain_summary.alpha == best].iloc[0].to_dict(),
        train=metrics(labels, pt), test=metrics(test_labels, pv),
        test_train_only_imputation=metrics(test_labels, ps),
        final_fit_and_evaluate_seconds=time.perf_counter()-start,
        parameters=model.get_params())
    np.savez_compressed(directory / 'xgb_positive_predictions.npz',
                        train=pt, test=pv, test_train_only=ps)
    dump(dict(imputer=imputer, scaler=scaler, model=model),
         directory / 'xgb_positive.joblib', compress=3)
    write_json(directory / 'xgb_positive.json', result)
    assert len(events) == 75
    print('Trained strictly positive alpha supplement:', best, flush=True)

def verify_results(directory, x, labels, test, test_labels, mode):
    imputer, a, b, bt = prepare(x, test)
    fold_file = np.load(Path(directory) / 'cv_folds.npz')
    for i, (_, va) in enumerate(training_folds(x, labels), 1):
        assert np.array_equal(va, fold_file[f'validation_{i}'])
    checks, rows = [], []
    for name in MODEL_NAMES:
        result = read_json(Path(directory) / f'{name}.json')
        saved = np.load(Path(directory) / f'{name}_predictions.npz')
        fitted = load(Path(directory) / f'{name}.joblib')
        if name.startswith('rf'):
            model = fitted
            assert model.get_params() == make_rf(name == 'rf_balanced').get_params()
            inputs = [a, b, bt]
            assert np.isclose(result['oob_error'],
                np.mean((model.oob_decision_function_[:, 1] >= .5) != labels))
        else:
            model = fitted['model']
            expected = xgb(result['selected_alpha']).get_params()
            actual = model.get_params()
            for key in expected:
                av, ev = actual[key], expected[key]
                assert av == ev or (isinstance(av, float) and np.isnan(av) and np.isnan(ev)), key
            assert np.allclose(fitted['imputer'].statistics_, imputer.statistics_, equal_nan=True)
            scaler = StandardScaler().fit(a)
            assert np.allclose(scaler.mean_, fitted['scaler'].mean_)
            assert np.allclose(scaler.scale_, fitted['scaler'].scale_)
            inputs = [scaler.transform(z) for z in [a, b, bt]]
        for split, true, data in zip(['train', 'test', 'test_train_only'],
                                    [labels, test_labels, test_labels], inputs):
            p = model.predict_proba(data)[:, 1]
            assert np.allclose(p, saved[split], atol=1e-12)
            calculated = metrics(true, p)
            previous = result['test_train_only_imputation' if split == 'test_train_only' else split]
            assert calculated['confusion_matrix'] == previous['confusion_matrix']
            for key in ['error', 'auc', 'average_precision', 'precision', 'recall', 'f1']:
                assert np.isclose(calculated[key], previous[key], atol=1e-12), (name, split, key)
            rows.append(dict(model=name, split=split,
                             **{k: v for k, v in calculated.items() if k != 'confusion_matrix'}))
        if name.startswith('xgb'):
            cv_name = 'xgb_plain' if name == 'xgb_positive' else name
            frame = pd.read_csv(Path(directory) / f'{cv_name}_cv_folds.csv')
            assert len(frame) == 35 and set(frame.alpha) == set(CONFIG['alphas'])
            assert frame.groupby('alpha').fold.nunique().eq(5).all()
            summary, best = choose_alpha(frame)
            if name == 'xgb_positive': best = choose_positive_alpha(summary)
            assert best == result['selected_alpha']
            stored = pd.read_csv(Path(directory) / f'{cv_name}_cv_summary.csv')
            for col in ['mean_ap', 'sd_ap', 'mean_error', 'sd_error', 'mean_auc']:
                assert np.allclose(summary[col], stored[col])
            selected = summary[summary.alpha == best].iloc[0]
            for col in ['mean_ap', 'sd_ap', 'mean_error', 'sd_error', 'mean_auc']:
                assert np.isclose(selected[col], result['cv'][col])
        checks.append(name + ': parameters, all three prediction sets, metrics and CV/OOB verified')
    scope = ('Full mode: 2 forests, all 70 CV estimators, 2 selected final XGB models '
             'and 1 strictly-positive-alpha supplementary model fitted from raw data this run.'
             if mode == 'full' else
             'Quick mode: no fitting. Validated cached final models re-evaluated; '
             'stored 70 CV fold scores summarized and alpha selection checked, not refitted.')
    return dict(all_passed=True, checks=checks, mode=mode,
        validation_scope=scope + ' All five final-model train/test/sensitivity predictions, '
        'metrics and RF OOB verified.'), pd.DataFrame(rows)

# Hash the actual executed cell, not a hard-coded expected digest.
MODEL_CODE = get_ipython().history_manager.input_hist_raw[-1].strip()
