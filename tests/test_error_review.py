import sys
from pathlib import Path
import unittest
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from review_errors import baseline_artifact_path, join_review_data, wilson_interval


class ErrorReviewTests(unittest.TestCase):
    def setUp(self):
        self.dataset = pd.DataFrame({'filename': ['normal_001.csv', 'noisy_001.csv'],
                                     'class': ['normal', 'noisy'], 'rms': [0.7, 0.8]})
        self.metadata = self.dataset[['filename', 'class']].copy()
        self.metadata['frequency'] = [1000, 900]
        self.split = self.dataset[['filename', 'class']].copy()
        self.split['split'] = ['test', 'train']
        self.predictions = pd.DataFrame({'model': ['random_forest', 'svm'],
            'filename': ['normal_001.csv', 'normal_001.csv'],
            'true_class': ['normal', 'normal'], 'predicted_class': ['normal', 'distorted']})

    def test_wilson_interval(self):
        low, high = wilson_interval(19, 19)
        self.assertAlmostEqual(low, 0.831820, places=5)
        self.assertAlmostEqual(high, 1)
        low, high = wilson_interval(0, 19)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, 0.168180, places=5)
        for correct, total in [(1, 0), (-1, 19), (20, 19)]:
            with self.assertRaises(ValueError):
                wilson_interval(correct, total)

    def test_historical_selection_uses_preserved_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(baseline_artifact_path('selected_model.json', root), root / 'selected_model.json')
            (root / 'baseline_selection.json').write_text('{"model": "random_forest"}')
            self.assertEqual(baseline_artifact_path('selected_model.json', root), root / 'baseline_selection.json')
            self.assertEqual(baseline_artifact_path('svm.pkl', root), root / 'svm.pkl')

    def test_join_preserves_prediction_count_and_outcomes(self):
        joined = join_review_data(self.dataset, self.metadata, self.split, self.predictions)
        self.assertEqual(len(joined), 2)
        self.assertEqual(joined['correct'].tolist(), [True, False])
        self.assertEqual(joined['frequency'].tolist(), [1000, 1000])

    def test_training_predictions_rejected(self):
        wrong = self.predictions.copy()
        wrong.loc[0, ['filename', 'true_class']] = ['noisy_001.csv', 'noisy']
        with self.assertRaisesRegex(ValueError, 'test split'):
            join_review_data(self.dataset, self.metadata, self.split, wrong)

    def test_duplicate_and_missing_metadata_rejected(self):
        duplicate = pd.concat([self.predictions, self.predictions.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            join_review_data(self.dataset, self.metadata, self.split, duplicate)
        missing = self.metadata[self.metadata['class'] == 'noisy']
        with self.assertRaisesRegex(ValueError, 'generation metadata'):
            join_review_data(self.dataset, missing, self.split, self.predictions)


if __name__ == '__main__':
    unittest.main()
