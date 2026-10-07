import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT, SIGNALS_PER_CLASS
from features import extract_features
from preprocessing import load_signal, preprocess_signal, validate_signal
from prediction import analyze_signal, export_analysis


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(500) / 100000
        self.signal = np.sin(2 * np.pi * 1000 * self.time)

    def test_valid_and_default_preservation(self):
        self.assertTrue(validate_signal(self.time, self.signal))
        np.testing.assert_array_equal(preprocess_signal(self.time, self.signal), self.signal)

    def test_invalid_signals(self):
        for t, y in [([], []), ([0, 1], [1]), ([0, 0], [1, 2]),
                     ([1, 0], [1, 2]), ([0, 1], [1, np.nan]),
                     ([0, np.inf], [1, 2]), ([0, 1], ['bad', 'data']),
                     ([[0, 1]], [[1, 2]])]:
            with self.subTest(time=t, signal=y):
                self.assertFalse(validate_signal(t, y))
                with self.assertRaises(ValueError):
                    preprocess_signal(t, y)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'signal.csv'
            for text in ['time,value\n0,1\n', 'time,amplitude\n0,bad\n',
                         'time,amplitude\n', 'time,amplitude\n0,NaN\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_signal(path)
            with self.assertRaises(FileNotFoundError):
                load_signal(Path(folder) / 'missing.csv')

    def test_features_and_harmonics(self):
        y = self.signal + 0.2 * np.sin(2*np.pi*2000*self.time) + 0.1 * np.sin(2*np.pi*3000*self.time)
        features = extract_features(y, 100000)
        self.assertEqual(list(features), FEATURE_NAMES)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))
        self.assertAlmostEqual(features['dominant_frequency'], 1000)
        self.assertAlmostEqual(features['second_harmonic_ratio'], 0.2)
        self.assertAlmostEqual(features['third_harmonic_ratio'], 0.1)

    def test_invalid_feature_inputs(self):
        for y, rate in [([], 100000), ([1], 100000), ([1, np.nan], 100000), ([1, 2], 0)]:
            with self.assertRaises(ValueError):
                extract_features(y, rate)

    def test_zero_signal(self):
        values = extract_features(np.zeros(500), 100000)
        self.assertEqual(values['second_harmonic_ratio'], 0)
        self.assertEqual(values['third_harmonic_ratio'], 0)

    def test_analysis_and_export(self):
        result = analyze_signal(PROJECT_ROOT / 'data/raw/normal/normal_001.csv')
        self.assertAlmostEqual(result['sampling_rate'], 100000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.csv'
            export_analysis(result, path)
            exported = pd.read_csv(path)
            self.assertTrue(set(FEATURE_NAMES).issubset(exported.columns))
            self.assertEqual(len(exported), 1)

    def test_loaded_signal_analysis_matches_file_analysis(self):
        from prediction import analyze_loaded_signal
        path = PROJECT_ROOT / 'data/raw/distorted/distorted_001.csv'
        time, signal = load_signal(path)
        before = signal.copy()
        loaded = analyze_loaded_signal(time, signal, path)
        from_file = analyze_signal(path)
        self.assertEqual(loaded['features'], from_file['features'])
        np.testing.assert_array_equal(signal, before)
        with self.assertRaises(ValueError):
            analyze_loaded_signal([0, 0], [1, 2], 'invalid.csv')

    def test_nonuniform_sampling_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'signal.csv'
            pd.DataFrame({'time': [0, 1, 3], 'amplitude': [1, 2, 3]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'uniformly'):
                analyze_signal(path)

    def test_existing_dataset_consistency(self):
        data = pd.read_csv(PROJECT_ROOT / 'data/dataset.csv')
        self.assertEqual(len(data), len(CLASS_NAMES) * SIGNALS_PER_CLASS)
        self.assertEqual(data['class'].value_counts().to_dict(), {name: SIGNALS_PER_CLASS for name in CLASS_NAMES})
        self.assertEqual(list(data.columns), FEATURE_NAMES + ['filename', 'class'])
        self.assertEqual(set(data['class']), set(CLASS_NAMES))
        for _, row in data.iterrows():
            t, y = load_signal(PROJECT_ROOT / 'data/processed' / row['class'] / row['filename'])
            actual = extract_features(y, 1 / np.diff(t).mean())
            for name in FEATURE_NAMES:
                self.assertTrue(np.isclose(actual[name], row[name]), (row['filename'], name))


@unittest.skipUnless(importlib.util.find_spec('sklearn') and importlib.util.find_spec('joblib'),
                     'scikit-learn and joblib are required')
class ClassificationTests(unittest.TestCase):
    def test_training_saving_and_prediction(self):
        from classifier import train_and_evaluate
        from prediction import predict_signal, load_model, export_analysis
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'analysis'
            models = Path(folder) / 'models'
            results = train_and_evaluate(output_directory=output, model_directory=models)
            self.assertEqual(set(results['model']), {'random_forest', 'svm'})
            for name in results['model']:
                for class_name in CLASS_NAMES:
                    result = predict_signal(PROJECT_ROOT / 'data/raw' / class_name / f'{class_name}_001.csv',
                                            models / f'{name}.pkl')
                    self.assertIn(result['predicted_class'].lower(), CLASS_NAMES)
                    exported = Path(folder) / 'prediction.csv'
                    export_analysis(result, exported)
                    self.assertEqual(pd.read_csv(exported)['predicted_class'].iloc[0], result['predicted_class'])
                self.assertEqual(load_model(models / f'{name}.pkl')['features'], FEATURE_NAMES)
                self.assertTrue((output / f'{name}_confusion_matrix.png').exists())
            import joblib
            svm = joblib.load(models / 'svm.pkl')['model']
            self.assertEqual(svm.named_steps['standardscaler'].n_samples_seen_,
                             int(0.75 * len(pd.read_csv(PROJECT_ROOT / 'data/dataset.csv'))))
            split = pd.read_csv(output / 'split.csv')
            self.assertEqual(set(split[split['split'] == 'train']['row_index']) &
                             set(split[split['split'] == 'test']['row_index']), set())
            self.assertEqual(split.groupby(['split', 'class']).size().nunique(), 2)
            self.assertTrue((output / 'feature_importance.png').exists())

    def test_model_loading_errors(self):
        import joblib
        from prediction import load_model
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pkl'
            with self.assertRaisesRegex(ValueError, 'Cannot load model'):
                load_model(path)
            path.write_text('invalid model data')
            with self.assertRaisesRegex(ValueError, 'Cannot load model'):
                load_model(path)
            joblib.dump({'features': ['filename'], 'classes': CLASS_NAMES}, path)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_model(path)


if __name__ == '__main__':
    unittest.main()
