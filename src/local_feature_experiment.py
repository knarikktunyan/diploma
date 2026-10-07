"""Isolated, preregistered feature comparison: prepare on training data, then evaluate once."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from classifier import create_models, save_confusion_matrix
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT, RANDOM_SEED
from generate_signals import generate_dataset
from local_features import FEATURE_GROUPS, LOCAL_FEATURE_NAMES, LOCAL_FEATURE_VERSION, extract_local_features
from prediction import analyze_signal, load_model
from review_errors import markdown_table, wilson_interval
from validate_dataset import validate_dataset

EXPERIMENT = PROJECT_ROOT / 'data/local_feature_experiment'
FRESH_TEST_SEED = 4242
FRESH_TEST_COUNT = 500


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_names(group):
    return FEATURE_NAMES + FEATURE_GROUPS[group]


def build_feature_table(rows, raw_root):
    records = []
    for class_name in CLASS_NAMES:
        subset = rows[rows['class']==class_name]
        for row in subset.to_dict('records'):
            analysis = analyze_signal(Path(raw_root)/class_name/row['filename'])
            # Preserve the existing extractor as the single source of baseline features.
            if all(name in row for name in FEATURE_NAMES):
                for name in FEATURE_NAMES:
                    if not np.isclose(row[name], analysis['features'][name]):
                        raise ValueError(f"Baseline features disagree for {row['filename']}")
            records.append(dict(**analysis['features'], **extract_local_features(analysis['signal']),
                                filename=row['filename'], **{'class': class_name}))
        print(f'{class_name}: {len(subset)} records featurized', flush=True)
    result = pd.DataFrame(records).set_index('filename')
    # Original training order matters for reproducible RF bootstrap sampling.
    return result.loc[rows['filename']].reset_index()


def prepare_experiment():
    if EXPERIMENT.exists() and any(EXPERIMENT.iterdir()):
        raise FileExistsError('Experiment already exists; preserve it instead of overwriting/tuning it')
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT/'data/dataset.csv'
    protected = {str(source.relative_to(PROJECT_ROOT)): digest(source)}
    protected.update({str(p.relative_to(PROJECT_ROOT)): digest(p) for p in (PROJECT_ROOT/'models').glob('*') if p.is_file()})
    protocol = dict(training_source='Original seed-42 training split only; reviewed 500-record holdout excluded',
                    training_cv='3-fold stratified macro F1, seed 42; same folds for every group',
                    feature_groups={k: feature_names(k) for k in FEATURE_GROUPS},
                    local_feature_version=LOCAL_FEATURE_VERSION,
                    local_parameters={'window_size':'floor(N/4)', 'hop':'floor(window_size/2)',
                                      'include_final_window':True,'crossing_smoothing_samples':11,'smoothing_polynomial':3},
                    models='Existing RF(200 trees) and SVM(StandardScaler→RBF SVC); no hyperparameter search',
                    fresh_test={'seed':FRESH_TEST_SEED,'count_per_class':FRESH_TEST_COUNT,'source':'Same unchanged synthetic generator'},
                    selection='Highest mean training-fold macro F1 per model, prefer earlier/smaller group for ties',
                    protected_baseline_sha256=protected, promotion='No active model/schema/GUI changes in this experiment')
    (EXPERIMENT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    dataset = pd.read_csv(source)
    train, _ = train_test_split(dataset, test_size=0.25, stratify=dataset['class'], random_state=RANDOM_SEED)
    split = pd.read_csv(PROJECT_ROOT/'data/feature_analysis/split.csv')
    expected = set(split.loc[split['split']=='train','filename'])
    if set(train['filename']) != expected or len(train) != 1500:
        raise ValueError('Training rows do not match the frozen original training population')
    table = build_feature_table(train, PROJECT_ROOT/'data/raw')
    table.to_csv(EXPERIMENT/'training_features.csv',index=False)
    metadata = pd.read_csv(PROJECT_ROOT/'data/raw/generation_metadata.csv').set_index('filename').loc[table['filename']]
    folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=RANDOM_SEED).split(table,table['class']))
    cv_rows, subtype_rows, selected = [], [], {}
    models_path = EXPERIMENT/'models'; models_path.mkdir()
    for model_name, template in create_models().items():
        candidates = []
        for group in FEATURE_GROUPS:
            columns = feature_names(group)
            scores, oof = [], np.empty(len(table),dtype=object)
            for fold, (training, validation) in enumerate(folds, start=1):
                model = clone(template)
                model.fit(table.iloc[training][columns],table.iloc[training]['class'])
                predicted = model.predict(table.iloc[validation][columns])
                scores.append(f1_score(table.iloc[validation]['class'],predicted,average='macro'))
                oof[validation] = predicted
            result = dict(model=model_name,group=group,features=len(columns),
                          cv_f1=float(np.mean(scores)),cv_fold_std=float(np.std(scores)),
                          fold_1=scores[0],fold_2=scores[1],fold_3=scores[2])
            cv_rows.append(result); candidates.append(result)
            for subtype in sorted(metadata['anomaly_type'].dropna().unique()):
                mask = (metadata['anomaly_type']==subtype).to_numpy()
                subtype_rows.append(dict(model=model_name,group=group,anomaly_type=subtype,
                                         support=int(mask.sum()),recall=float((oof[mask]=='anomalous').mean())))
            print(model_name,group,'CV macro F1:',result['cv_f1'],flush=True)
        winner = max(candidates,key=lambda item:item['cv_f1'])
        group = winner['group']; columns = feature_names(group)
        if group == 'baseline':
            artifact = load_model(PROJECT_ROOT/'models'/f'{model_name}.pkl')
            model = artifact['model']
        else:
            model = clone(template).fit(table[columns],table['class'])
        artifact = dict(model=model,features=columns,classes=CLASS_NAMES,feature_group=group,
                        local_feature_version=LOCAL_FEATURE_VERSION,
                        preprocessing={'remove_dc':False,'filter_signal':False,'normalize':False},
                        training_count=len(table),random_seed=RANDOM_SEED)
        joblib.dump(artifact,models_path/f'{model_name}.pkl')
        selected[model_name] = dict(group=group,features=columns,training_cv_f1=winner['cv_f1'],
                                   model_sha256=digest(models_path/f'{model_name}.pkl'))
    pd.DataFrame(cv_rows).to_csv(EXPERIMENT/'training_cv.csv',index=False)
    pd.DataFrame(subtype_rows).to_csv(EXPERIMENT/'training_cv_subtypes.csv',index=False)
    selection = dict(selected=selected,training_feature_sha256=digest(EXPERIMENT/'training_features.csv'),
                     protocol_sha256=digest(EXPERIMENT/'protocol.json'),fresh_test_generated=False)
    # Selection is persisted before the fresh test is generated or evaluated.
    (EXPERIMENT/'selection_locked.json').write_text(json.dumps(selection,indent=2))
    print('Training-only selection locked; fresh test has not been generated.',flush=True)


def load_experimental_model(filename):
    try:
        artifact = joblib.load(filename)
    except Exception as error:
        raise ValueError(f'Cannot load experimental model: {error}') from error
    if not isinstance(artifact, dict):
        raise ValueError('Experimental model artifact must be a dictionary')
    group = artifact.get('feature_group')
    if group not in FEATURE_GROUPS or artifact.get('features') != feature_names(group):
        raise ValueError('Incompatible experimental feature schema')
    if artifact.get('classes') != CLASS_NAMES or artifact.get('local_feature_version') != LOCAL_FEATURE_VERSION:
        raise ValueError('Incompatible experimental model metadata')
    if artifact.get('preprocessing') != {'remove_dc':False,'filter_signal':False,'normalize':False}:
        raise ValueError('Experimental model preprocessing does not match analysis')
    if not hasattr(artifact.get('model'), 'predict'):
        raise ValueError('Experimental artifact has no prediction model')
    return artifact


def predict_experimental_signal(filename, model_path):
    artifact = load_experimental_model(model_path)
    analysis = analyze_signal(filename)
    values = dict(**analysis['features'], **extract_local_features(analysis['signal']))
    prediction = artifact['model'].predict(pd.DataFrame([values],columns=artifact['features']))[0]
    if prediction not in CLASS_NAMES:
        raise ValueError('Unsupported predicted class')
    analysis['predicted_class'] = str(prediction).capitalize()
    return analysis


def evaluate_experiment():
    locked = json.loads((EXPERIMENT/'selection_locked.json').read_text())
    protocol = json.loads((EXPERIMENT/'protocol.json').read_text())
    if digest(EXPERIMENT/'protocol.json') != locked['protocol_sha256']:
        raise ValueError('Protocol changed after training selection')
    if digest(EXPERIMENT/'training_features.csv') != locked['training_feature_sha256']:
        raise ValueError('Training data changed after selection')
    for name, expected in protocol['protected_baseline_sha256'].items():
        if digest(PROJECT_ROOT/name) != expected: raise ValueError(f'Baseline changed: {name}')
    for name, info in locked['selected'].items():
        if digest(EXPERIMENT/'models'/f'{name}.pkl') != info['model_sha256']:
            raise ValueError('Selected model changed after locking')
    raw = EXPERIMENT/'fresh_test/raw'
    if raw.exists():
        raise FileExistsError('Fresh test already generated; do not repeatedly tune/evaluate on this test')
    generate_dataset(raw,count=protocol['fresh_test']['count_per_class'],seed=protocol['fresh_test']['seed'])
    baseline_features, summary = validate_dataset(raw,count=FRESH_TEST_COUNT,output_directory=EXPERIMENT/'fresh_test/validation')
    original_hashes = set(pd.read_csv(PROJECT_ROOT/'data/dataset_analysis/signal_manifest.csv')['sha256'])
    fresh_hashes = set(pd.read_csv(EXPERIMENT/'fresh_test/validation/signal_manifest.csv')['sha256'])
    if original_hashes & fresh_hashes: raise ValueError('Fresh test duplicates an original signal')
    table = build_feature_table(baseline_features,raw)
    table.to_csv(EXPERIMENT/'fresh_test/features.csv',index=False)
    metadata = pd.read_csv(raw/'generation_metadata.csv').set_index('filename').loc[table['filename']]
    results, subtype_results, predictions, roundtrip = [], [], [], []
    for name in create_models():
        for variant in ['baseline','selected']:
            if variant == 'baseline':
                artifact = load_model(PROJECT_ROOT/'models'/f'{name}.pkl')
                group = 'baseline'
            else:
                artifact = load_experimental_model(EXPERIMENT/'models'/f'{name}.pkl')
                group = artifact['feature_group']
            predicted = artifact['model'].predict(table[artifact['features']])
            report = classification_report(table['class'],predicted,labels=CLASS_NAMES,output_dict=True,zero_division=0)
            label = f'{name}_{variant}'
            (EXPERIMENT/f'{label}_report.json').write_text(json.dumps(report,indent=2))
            save_confusion_matrix(table['class'],predicted,EXPERIMENT/f'{label}_confusion_matrix.png',label)
            result = dict(model=name,variant=variant,feature_group=group,
                          accuracy=accuracy_score(table['class'],predicted),
                          precision=report['macro avg']['precision'],recall=report['macro avg']['recall'],
                          f1=report['macro avg']['f1-score'])
            results.append(result)
            predictions.append(pd.DataFrame({'model':name,'variant':variant,'filename':table['filename'],
                                             'true_class':table['class'],'predicted_class':predicted}))
            for subtype in sorted(metadata['anomaly_type'].dropna().unique()):
                mask = (metadata['anomaly_type']==subtype).to_numpy()
                correct = int((predicted[mask]=='anomalous').sum()); total=int(mask.sum())
                low,high = wilson_interval(correct,total)
                subtype_results.append(dict(model=name,variant=variant,anomaly_type=subtype,
                                            support=total,correct=correct,recall=correct/total,ci_low=low,ci_high=high))
            if variant == 'selected':
                for class_name in CLASS_NAMES:
                    signal = raw/class_name/f'{class_name}_001.csv'
                    actual = predict_experimental_signal(signal,EXPERIMENT/'models'/f'{name}.pkl')['predicted_class']
                    position = table.index[table['filename']==signal.name][0]
                    if actual.lower()!=predicted[position]: raise ValueError('Saved-model CSV roundtrip differs')
                    roundtrip.append(dict(model=name,filename=signal.name,predicted_class=actual))
            print(label,result,flush=True)
    results = pd.DataFrame(results); subtypes = pd.DataFrame(subtype_results)
    results.to_csv(EXPERIMENT/'fresh_test_results.csv',index=False)
    subtypes.to_csv(EXPERIMENT/'fresh_test_subtypes.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(EXPERIMENT/'fresh_test_predictions.csv',index=False)
    pd.DataFrame(roundtrip).to_csv(EXPERIMENT/'prediction_roundtrip.csv',index=False)
    write_experiment_report(results,subtypes,locked)
    verification=dict(status='passed',fresh_test=summary,original_signal_overlap=0,
                      training_only_selection=True,locked_before_test_generation=True,
                      saved_model_roundtrips=len(roundtrip),active_models_unchanged=True,
                      test_feature_sha256=digest(EXPERIMENT/'fresh_test/features.csv'))
    for name, expected in protocol['protected_baseline_sha256'].items():
        if digest(PROJECT_ROOT/name)!=expected: raise ValueError('Active baseline changed during experiment')
    (EXPERIMENT/'verification.json').write_text(json.dumps(verification,indent=2))


def write_experiment_report(results,subtypes,locked):
    docs=PROJECT_ROOT/'docs'; docs.mkdir(exist_ok=True)
    cv=pd.read_csv(EXPERIMENT/'training_cv.csv')
    text='''# Time-local feature experiment

This is a separate synthetic experiment. Production data, ten-feature models and
GUI are preserved. The historical 500-record holdout was excluded from fitting,
validation and this comparison because it had informed hypotheses.

## Fixed protocol and features

The candidate definitions were saved in protocol.json before any CV or fresh-test
generation. Use 1,500 original training records; identical seed-42 three-fold
stratified folds and the unchanged RF/SVM configurations; mean fold macro F1 for
selection. No hyperparameter search. SVM scaling is fitted inside each training fold.
Ties prefer the earlier/smaller predefined group. Audit metadata is used only for
subtype scoring, never as input. The original randomized training order is restored.

Additional per-signal features: range of overlapping quarter-record RMS values
normalized by their mean; range of window means normalized by whole-record standard
deviation; coefficient of variation of interpolated rising-crossing periods. The
last uses fixed 11-sample order-3 Savitzky–Golay smoothing of a centered auxiliary
copy. Original waveform and baseline feature definitions are unchanged. Fewer than
three rising crossings gives period variation zero; this is a limitation, not proof
of stationarity. Quarter-window means also depend on phase and record length.

## Training-only selection

'''+markdown_table(cv.round(4))+'\n\n'
    for name,info in locked['selected'].items():
        text+=f"- {name}: selected **{info['group']}**, {len(info['features'])} features, CV macro F1 {info['training_cv_f1']:.4f}.\n"
    text+='''
## Fresh evaluation

Only after selection/models were locked was a seed-4242 test dataset generated:
500 records per class, 2,000 total, using the unchanged synthetic generator.
All CSVs passed validation and no exact file hashes overlap the original dataset.
There are no paired original/fresh waveforms. Both frozen baseline and selected
models use the same original 1,500 training records and exactly the same fresh
2,000 test records. Baseline evaluation loads the actual active saved models.
No test record is used to refit/select a model, scaler or feature group.

'''+markdown_table(results.round(4))+'''\n
Precision, recall and F1 are macro averages. Historical 89.8%/87.2% accuracies belong
to a different, previously reviewed 500-record test set; use the paired baseline
and selected rows above for this experiment's comparison.

## Subtype evaluation

'''+markdown_table(subtypes.round(4))+'''\n
Confidence intervals are descriptive 95% Wilson intervals for anomalous recall.
Scores apply only to the synthetic population; no real-circuit claim or formal
significance claim is made. Improved synthetic results may reflect generator-specific
patterns. Noise/harmonics can disturb crossings; short records limit period estimates.

## Artifacts and reproducibility

Experiment data, protocol, locked selection, CV results, saved candidate models,
class reports, confusion matrices and roundtrip checks are in
../data/local_feature_experiment/. Candidate models are experimental schema artifacts,
not drop-in replacements for the active ten-feature loader/GUI.

Run `venv/bin/python src/local_feature_experiment.py --prepare` in an empty experiment
directory, then `--evaluate` once. The runner refuses to overwrite experiments or
regenerate/repeatedly evaluate an existing fresh test. Rerunning development should
use an explicitly separate protocol/cohort; do not delete this evidence to tune on
the same test. The fresh test is now evaluated and must be treated accordingly in
any subsequent redesign. Generic CSV preprocessing/analysis is reused.

No candidate is promoted to production in this stage. The next decision is whether
the measured gains justify schema-versioned prediction integration, followed by
GUI/export compatibility checks and preservation of the original baseline. Such
integration should retain separate baseline artifacts and explicit feature metadata.
'''
    (docs/'local_feature_experiment.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    stage=parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare',action='store_true')
    stage.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare_experiment()
    else: evaluate_experiment()
