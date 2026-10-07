"""Production schema compatibility, snapshot prediction, and export provenance."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import CLASS_NAMES, FEATURE_NAMES
from feature_schema import feature_names
from prediction import analyze_loaded_signal, classify_analysis, export_analysis, load_model


class ModelSchemaTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(500) / 100000
        self.signal = np.sin(2 * np.pi * 1000 * self.time)

    def artifact(self, version):
        analysis = analyze_loaded_signal(self.time, self.signal, 'snapshot.csv', version)
        columns = feature_names(version)
        model = RandomForestClassifier(n_estimators=3, random_state=42)
        # Four distinct feature rows: only loader/schema behavior is tested here.
        training = pd.DataFrame([{key: value * scale for key, value in analysis['features'].items()}
                                 for scale in [0.8, 1, 1.2, 1.5]], columns=columns)
        model.fit(training, CLASS_NAMES)
        return dict(model=model, features=columns, classes=CLASS_NAMES,
                    feature_schema_version=version, local_feature_version=1, model_name='test',
                    preprocessing={'remove_dc': False, 'filter_signal': False, 'normalize': False})

    def test_legacy_and_versioned_models_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            for version in [1, 2]:
                artifact = self.artifact(version)
                if version == 1:
                    artifact.pop('feature_schema_version')  # Original model format.
                path = Path(folder) / 'model.pkl'
                joblib.dump(artifact, path)
                analysis = analyze_loaded_signal(self.time, self.signal, 'snapshot.csv')
                analysis['predicted_class'] = classify_analysis(analysis, path)
                self.assertIn(analysis['predicted_class'].lower(), CLASS_NAMES)
                self.assertEqual(list(analysis['features']), feature_names(version))
                exported = Path(folder) / 'export.csv'
                export_analysis(analysis, exported)
                row = pd.read_csv(exported).iloc[0]
                self.assertEqual(row['model_feature_schema_version'], version)
                self.assertEqual(row['feature_schema_version'], version)
                self.assertEqual(row['model_name'], 'test')
                self.assertEqual(list(analysis['features'])[:10], FEATURE_NAMES)

    def test_bad_versions_and_fitted_order_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pkl'
            for change in [{'feature_schema_version': 99}, {'local_feature_version': 99},
                           {'features': list(reversed(feature_names(2)))}]:
                artifact = self.artifact(2)
                artifact.update(change)
                joblib.dump(artifact, path)
                with self.assertRaises(ValueError):
                    load_model(path)
            artifact = self.artifact(2)
            artifact['model'].feature_names_in_ = np.array(list(reversed(feature_names(2))))
            joblib.dump(artifact, path)
            with self.assertRaisesRegex(ValueError, 'order'):
                load_model(path)

    def test_snapshot_is_used_for_local_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pkl'
            artifact = self.artifact(2)
            joblib.dump(artifact, path)
            analysis = analyze_loaded_signal(self.time, self.signal, 'nonexistent.csv')
            expected = analyze_loaded_signal(self.time, self.signal, 'nonexistent.csv', 2)
            classify_analysis(analysis, path)
            self.assertEqual(analysis['features'], expected['features'])
            np.testing.assert_array_equal(analysis['signal'], self.signal)

    def test_short_signals_keep_legacy_analysis_but_reject_local_schema(self):
        analysis = analyze_loaded_signal([0, 0.001], [1, 2], 'short.csv')
        self.assertEqual(list(analysis['features']), FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, '32'):
            analyze_loaded_signal([0, 0.001], [1, 2], 'short.csv', 2)


if __name__ == '__main__':
    unittest.main()
