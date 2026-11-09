"""Coverage-aware OOB review; independent sidecar, legacy fit/cache cell unchanged."""
def oob_coverage(model, a, labels):
    # estimators_samples_ reconstructs actual draws from fitted seeds and stored weights.
    n = len(labels)
    votes = np.zeros(n, dtype=np.int32)
    sums = np.zeros((n, 2))
    indices_hash = hashlib.sha256()
    for tree, inbag in zip(model.estimators_, model.estimators_samples_):
        counts = np.bincount(inbag, minlength=n)
        assert tree.tree_.n_node_samples[0] == np.count_nonzero(counts)
        indices_hash.update(np.asarray(inbag, dtype=np.int32).tobytes())
        mask = counts == 0
        votes += mask
        if mask.any():
            sums[mask] += tree.predict_proba(np.asarray(a[mask], dtype=np.float32),
                                            check_input=False)
    p = model.oob_decision_function_
    covered = votes > 0
    valid = covered & np.isfinite(p).all(axis=1) & np.isclose(p.sum(axis=1), 1)
    recomputed = np.zeros_like(sums)
    recomputed[covered] = sums[covered] / votes[covered, None]
    assert np.allclose(p, recomputed, atol=1e-12)
    errors = int(np.sum((p[valid, 1] >= .5) != labels[valid]))
    def group(mask):
        v = votes[mask]
        return dict(n=int(mask.sum()), valid=int(valid[mask].sum()),
            zero_votes=int((v == 0).sum()), valid_fraction=float(valid[mask].mean()),
            min_votes=int(v.min()), max_votes=int(v.max()), mean_votes=float(v.mean()))
    stats = dict(total=group(np.ones(n, dtype=bool)),
        by_class={name: group(labels == c) for c, name in [(0, 'neg'), (1, 'pos')]},
        all_zero_probability_rows=int(np.all(p == 0, axis=1).sum()),
        nan_rows=int(np.isnan(p).any(axis=1).sum()),
        nonfinite_rows=int((~np.isfinite(p).all(axis=1)).sum()),
        other_invalid_rows=int((covered & ~valid).sum()),
        historical_unmasked_error=float(np.mean((p[:, 1] >= .5) != labels)),
        valid_error=errors / int(valid.sum()) if valid.any() else None,
        valid_errors=errors, valid_denominator=int(valid.sum()),
        full_class_coverage=bool(valid.all()),
        warning_expected=bool((votes == 0).any()),
        bootstrap_indices_sha256=indices_hash.hexdigest(),
        treewise_probabilities_reconstructed=True)
    return stats, votes, valid

def oob_sidecar_signature(model_dir):
    import sklearn, inspect
    from sklearn.ensemble import _forest
    return dict(schema=1, legacy_signature=cache_signature(),
        legacy_manifest_sha256=sha256_file(Path(model_dir) / 'manifest.json'),
        original_rf_sha256={n: sha256_file(Path(model_dir) / f'{n}.joblib')
                           for n in ['rf_plain', 'rf_balanced']},
        audit_code_sha256=hashlib.sha256(OOB_CODE.encode()).hexdigest(),
        sklearn_version=sklearn.__version__,
        sklearn_forest_source_sha256=sha256_file(inspect.getsourcefile(_forest)),
        supplemental_parameters=dict(CONFIG['rf'], class_weight='balanced_subsample'))

def run_oob_review(model_dir, x, labels, test, test_labels, output, mode, sidecar):
    import inspect, warnings
    from sklearn.ensemble import _forest
    output, sidecar = Path(output), Path(sidecar)
    expected = oob_sidecar_signature(model_dir)
    _, a, b, bt = prepare(x, test)
    audits, arrays = {}, {}
    for name in ['rf_plain', 'rf_balanced']:
        fitted = load(Path(model_dir) / f'{name}.joblib')
        stats, votes, valid = oob_coverage(fitted, a, labels)
        audits[name] = stats
        arrays[name + '_votes'] = votes
        arrays[name + '_valid'] = valid
    if mode == 'fit_supplement':
        assert not sidecar.exists(), 'Use a new sidecar directory; never overwrite history.'
        sidecar.mkdir(parents=True)
        start = time.perf_counter()
        fitted = RandomForestClassifier(**expected['supplemental_parameters'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted.fit(a, labels)
        p = [fitted.predict_proba(z)[:, 1] for z in [a, b, bt]]
        stats, votes, valid = oob_coverage(fitted, a, labels)
        score = dict(train=metrics(labels, p[0]), test=metrics(test_labels, p[1]),
            test_train_only_imputation=metrics(test_labels, p[2]),
            oob=stats, fit_audit_evaluate_seconds=time.perf_counter()-start,
            parameters=fitted.get_params(), warnings=[str(w.message) for w in caught],
            rationale='Uniform bootstrap with per-tree balanced impurity weights permits both-class OOB coverage; supplementary, not a replacement.')
        dump(fitted, sidecar / 'rf_balanced_subsample.joblib', compress=3)
        np.savez_compressed(sidecar / 'rf_balanced_subsample_predictions.npz',
                            train=p[0], test=p[1], test_train_only=p[2])
        write_json(sidecar / 'rf_balanced_subsample.json', score)
        files={f.name:sha256_file(f) for f in sidecar.iterdir() if f.is_file()}
        write_json(sidecar / 'manifest.json', dict(signature=expected,files=files,
            provenance=dict(only_new_fit='rf_balanced_subsample',original_models_refitted=False,
                xgboost_or_smote_or_cv_refitted=False)))
    elif mode != 'verify':
        raise ValueError('OOB_MODE must be fit_supplement or verify.')
    if not (sidecar / 'manifest.json').exists():
        raise FileNotFoundError('OOB sidecar missing: supply matching artifacts, or explicitly select OOB_MODE=fit_supplement. Full mode trains its own supplement.')
    manifest=read_json(sidecar / 'manifest.json')
    if manifest['signature'] != expected:
        raise ValueError('OOB sidecar data/model/code/version signature mismatch; do not re-sign old artifacts.')
    for name,sha in manifest['files'].items():
        if not (sidecar/name).is_file() or sha256_file(sidecar/name)!=sha:
            raise ValueError('Missing/altered OOB sidecar artifact: '+name)
    fitted=load(sidecar/'rf_balanced_subsample.joblib')
    assert fitted.get_params()==RandomForestClassifier(**expected['supplemental_parameters']).get_params()
    score=read_json(sidecar/'rf_balanced_subsample.json')
    stats,votes,valid=oob_coverage(fitted,a,labels)
    assert stats==score['oob']
    for split,truth,z in [('train',labels,a),('test',test_labels,b),('test_train_only',test_labels,bt)]:
        fresh=fitted.predict_proba(z)[:,1]
        stored=np.load(sidecar/'rf_balanced_subsample_predictions.npz')[split]
        assert np.allclose(fresh,stored,atol=1e-12)
        old=score['test_train_only_imputation' if split=='test_train_only' else split]
        assert metrics(truth,fresh)==old
    name='rf_balanced_subsample'
    audits[name]=stats;arrays[name+'_votes']=votes;arrays[name+'_valid']=valid
    np.savez_compressed(output/'oob_votes_and_masks.npz',**arrays)
    pd.DataFrame({'class':labels,**{k:v for k,v in arrays.items() if k.endswith('_votes')}}).to_csv(output/'oob_per_sample.csv',index=False)
    write_json(output/'oob_coverage.json',audits)
    write_json(output/'rf_balanced_subsample.json',score)
    shutil.copy2(sidecar/'rf_balanced_subsample_predictions.npz',output/'rf_balanced_subsample_predictions.npz')
    source={n:inspect.getsource(getattr(_forest,n)) for n in
            ['_generate_sample_indices','_generate_unsampled_indices','_parallel_build_trees']}
    source['oob']=inspect.getsource(_forest.ForestClassifier._compute_oob_predictions)
    source['indices_property']=inspect.getsource(_forest.BaseForest._get_estimators_indices)
    source['class_weights']=inspect.getsource(_forest.ForestClassifier._validate_y_class_weight)
    write_json(output/'sklearn_oob_source.json',dict(signature=expected,source=source))
    write_json(output/'oob_review_execution.json',dict(mode=mode,
        new_fits=1 if mode=='fit_supplement' else 0,original_models_refitted=False,
        xgboost_or_smote_or_cv_refitted=False,sidecar=str(sidecar),
        legacy_oob_status='Historical unmasked arithmetic only; balanced value is not a valid overall OOB estimate.',
        coverage_checked_by_actual_indices=True))
    return audits,score

OOB_CODE = get_ipython().history_manager.input_hist_raw[-1].strip()
