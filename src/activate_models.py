"""Verify and activate frozen schema-2 candidates without fitting any model."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import PROJECT_ROOT
from feature_schema import feature_names
from prediction import load_model, predict_signal

EXPERIMENT = PROJECT_ROOT / 'data/local_feature_experiment'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate_models():
    protocol = json.loads((EXPERIMENT / 'protocol.json').read_text())
    locked = json.loads((EXPERIMENT / 'selection_locked.json').read_text())
    verification = json.loads((EXPERIMENT / 'verification.json').read_text())
    if verification['status'] != 'passed':
        raise ValueError('The candidate experiment has not passed verification')
    if digest(EXPERIMENT / 'protocol.json') != locked['protocol_sha256']:
        raise ValueError('Experiment protocol changed')
    if digest(EXPERIMENT / 'training_features.csv') != locked['training_feature_sha256']:
        raise ValueError('Experiment training features changed')
    for name, expected in protocol['protected_baseline_sha256'].items():
        # Selection changes deliberately; baseline dataset/model files stay frozen.
        if name != 'models/selected_model.json' and digest(PROJECT_ROOT / name) != expected:
            raise ValueError(f'Baseline artifact changed: {name}')
    features_path = EXPERIMENT / 'fresh_test/features.csv'
    if digest(features_path) != verification['test_feature_sha256']:
        raise ValueError('Frozen test feature table changed')
    table = pd.read_csv(features_path)
    expected_predictions = pd.read_csv(EXPERIMENT / 'fresh_test_predictions.csv')
    destination = PROJECT_ROOT / 'models/time_local'
    destination.mkdir(exist_ok=True)
    checks = []
    for name, selection in locked['selected'].items():
        source = EXPERIMENT / 'models' / f'{name}.pkl'
        if digest(source) != selection['model_sha256']:
            raise ValueError('Frozen candidate model changed')
        artifact = joblib.load(source)
        if artifact['features'] != feature_names(2):
            raise ValueError('Only the validated combined schema-2 candidates can be activated')
        artifact.update(feature_schema_version=2, model_name=name,
                        training_cv_f1=selection['training_cv_f1'],
                        training_source='Original seed-42 training split: 1500 records',
                        experiment='data/local_feature_experiment',
                        candidate_sha256=selection['model_sha256'])
        pending = destination / f'{name}.pending.pkl'
        joblib.dump(artifact, pending)
        loaded = load_model(pending)
        predicted = loaded['model'].predict(table[loaded['features']])
        expected = expected_predictions.query('model == @name and variant == "selected"').set_index('filename')
        if not np.array_equal(predicted, expected.loc[table['filename'], 'predicted_class'].to_numpy()):
            raise ValueError('Activated candidate differs from frozen test predictions')
        # Verify the complete CSV → analysis → saved model → export path on every subtype.
        metadata = pd.read_csv(EXPERIMENT / 'fresh_test/raw/generation_metadata.csv')
        examples = metadata.groupby('class', sort=False).head(1)
        anomalies = metadata[metadata['class'] == 'anomalous'].groupby('anomaly_type').head(1)
        examples = pd.concat([examples, anomalies]).drop_duplicates('filename')
        for row in examples.to_dict('records'):
            csv = EXPERIMENT / 'fresh_test/raw' / row['class'] / row['filename']
            result = predict_signal(csv, pending)
            if result['predicted_class'].lower() != expected.loc[row['filename'], 'predicted_class']:
                raise ValueError('Raw-CSV prediction differs from frozen prediction')
        pending.replace(destination / f'{name}.pkl')
        checks.append(dict(model=name, frozen_predictions_matched=len(table),
                           raw_csv_roundtrips=len(examples),
                           artifact_sha256=digest(destination / f'{name}.pkl')))
        print(f'{name}: {len(table)} frozen predictions and {len(examples)} CSV checks passed', flush=True)
    # Choose by the already locked TRAINING CV score, never fresh-test accuracy.
    selected = max(locked['selected'], key=lambda name: locked['selected'][name]['training_cv_f1'])
    selection_path = PROJECT_ROOT / 'models/selected_model.json'
    backup = PROJECT_ROOT / 'models/baseline_selection.json'
    if not backup.exists():
        if digest(selection_path) != protocol['protected_baseline_sha256']['models/selected_model.json']:
            raise ValueError('Original baseline selection was changed before backup')
        backup.write_bytes(selection_path.read_bytes())
    selection = dict(model=selected, artifact=f'time_local/{selected}.pkl',
                     feature_schema_version=2,
                     selection='Highest locked training-only three-fold macro F1',
                     experiment='data/local_feature_experiment')
    pending_selection = selection_path.with_suffix('.pending.json')
    pending_selection.write_text(json.dumps(selection, indent=2))
    pending_selection.replace(selection_path)
    if load_model()['feature_schema_version'] != 2:
        raise ValueError('Default model activation failed')
    output = PROJECT_ROOT / 'data/final_evaluation'
    output.mkdir(exist_ok=True)
    importance = pd.Series(load_model(destination / 'random_forest.pkl')['model'].feature_importances_,
                           index=feature_names(2)).sort_values()
    importance.rename('importance').to_csv(output / 'feature_importance.csv')
    figure = Figure(figsize=(10, 7), layout='constrained')
    axis = figure.subplots()
    axis.barh(importance.index, importance.values)
    axis.set(xlabel='Impurity-based importance', title='Activated Random Forest — 13 features')
    figure.savefig(output / 'feature_importance.png')
    for name, expected in protocol['protected_baseline_sha256'].items():
        if name != 'models/selected_model.json' and digest(PROJECT_ROOT / name) != expected:
            raise ValueError('Baseline changed during activation')
    report = dict(status='passed', selected_model=selected, feature_schema_version=2,
                  checks=checks, baseline_dataset_and_models_preserved=True,
                  refitting_performed=False, test_used_for_selection=False)
    (output / 'activation_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    activate_models()
