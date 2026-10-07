"""GUI-independent signal analysis, model loading, prediction and CSV export."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT
from feature_schema import analysis_features, feature_names
from local_features import LOCAL_FEATURE_VERSION, extract_local_features
from preprocessing import load_signal, preprocess_signal


def analyze_signal(filename, feature_schema_version=1):
    time, signal = load_signal(filename)
    return analyze_loaded_signal(time, signal, filename, feature_schema_version)


def analyze_loaded_signal(time, signal, filename, feature_schema_version=1):
    """Analyze a validated in-memory snapshot using the same preprocessing/features."""
    signal = preprocess_signal(time, signal)
    time = np.asarray(time, dtype=float)
    if len(time) < 2:
        raise ValueError('At least two samples are needed for frequency analysis')
    intervals = np.diff(time)
    if not np.allclose(intervals, intervals.mean(), rtol=1e-3, atol=0):
        raise ValueError('FFT analysis requires uniformly sampled time values; resample first')
    sampling_rate = 1 / intervals.mean()
    return {'filename': str(filename), 'time': time, 'signal': signal,
            'sampling_rate': sampling_rate,
            'feature_schema_version': feature_schema_version,
            'features': analysis_features(signal, sampling_rate, feature_schema_version)}


def load_model(filename=None):
    if filename is None:
        selection = json.loads((PROJECT_ROOT / 'models/selected_model.json').read_text())
        if selection['model'] not in ('random_forest', 'svm'):
            raise ValueError('Unsupported selected model')
        filename = PROJECT_ROOT / 'models' / (selection['model'] + '.pkl')
        if 'artifact' in selection:
            filename = PROJECT_ROOT / 'models' / selection['artifact']
            if not filename.resolve().is_relative_to((PROJECT_ROOT / 'models').resolve()):
                raise ValueError('Selected artifact must be inside the models directory')
    # Only load trusted local joblib files.
    try:
        import joblib
        artifact = joblib.load(filename)
    except Exception as error:
        raise ValueError(f'Cannot load model {filename}: {error}') from error
    if not isinstance(artifact, dict):
        raise ValueError('Model metadata must be a dictionary')
    version = artifact.get('feature_schema_version', 1)
    if artifact.get('features') != feature_names(version) or artifact.get('classes') != CLASS_NAMES:
        raise ValueError('Model metadata is incompatible with the current features/classes')
    if version == 2 and artifact.get('local_feature_version') != LOCAL_FEATURE_VERSION:
        raise ValueError('Local feature definitions do not match the model metadata')
    if artifact.get('preprocessing') != {'remove_dc': False, 'filter_signal': False, 'normalize': False}:
        raise ValueError('Model preprocessing does not match the prediction pipeline')
    if not hasattr(artifact.get('model'), 'predict'):
        raise ValueError('Saved model does not support prediction')
    model = artifact['model']
    if list(getattr(model, 'feature_names_in_', [])) != artifact['features']:
        raise ValueError('Fitted model feature order differs from its metadata')
    if set(getattr(model, 'classes_', [])) != set(CLASS_NAMES):
        raise ValueError('Fitted model classes differ from its metadata')
    artifact['feature_schema_version'] = version
    artifact.setdefault('model_name', Path(filename).stem)
    return artifact


def classify_analysis(analysis, model_path=None):
    artifact = load_model(model_path)
    if not set(FEATURE_NAMES).issubset(analysis['features']):
        raise ValueError('Analysis is missing baseline features')
    if artifact['feature_schema_version'] == 2:
        # Always calculate from this analysis snapshot, never reread a changed CSV.
        analysis['features'].update(extract_local_features(analysis['signal']))
        analysis['feature_schema_version'] = 2
    values = pd.DataFrame([analysis['features']], columns=artifact['features'])
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Prediction features must be finite')
    predicted = artifact['model'].predict(values)[0]
    if predicted not in CLASS_NAMES:
        raise ValueError('Model returned an unsupported class')
    analysis['model_name'] = artifact['model_name']
    analysis['model_feature_schema_version'] = artifact['feature_schema_version']
    return str(predicted).capitalize()


def predict_signal(filename, model_path=None):
    analysis = analyze_signal(filename)
    analysis['predicted_class'] = classify_analysis(analysis, model_path)
    return analysis


def export_analysis(analysis, filename):
    row = {'filename': analysis['filename'], 'sampling_rate': analysis['sampling_rate'],
           'feature_schema_version': analysis.get('feature_schema_version', 1),
           'model_name': analysis.get('model_name', ''),
           'model_feature_schema_version': analysis.get('model_feature_schema_version', ''),
           **analysis['features'], 'predicted_class': analysis.get('predicted_class', '')}
    pd.DataFrame([row]).to_csv(filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('signal')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    result = predict_signal(args.signal, args.model)
    print(f"Predicted class: {result['predicted_class']}")
    if args.export:
        export_analysis(result, args.export)
