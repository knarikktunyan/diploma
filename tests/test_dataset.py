"""Small generator/validator tests; production data is never regenerated here."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from config import CLASS_NAMES, PROJECT_ROOT, SAMPLING_RATE, SIGNAL_DURATION
from generate_signals import ANOMALY_TYPES, generate_dataset, generate_signal
from preprocessing import validate_signal
from validate_dataset import validate_dataset


class DatasetTests(unittest.TestCase):
    def test_signal_reproducibility_and_validity(self):
        for class_name in CLASS_NAMES:
            t1, y1, parameters = generate_signal(class_name, np.random.default_rng(123))
            t2, y2, repeated = generate_signal(class_name, np.random.default_rng(123))
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(y1, y2)
            self.assertEqual(parameters, repeated)
            self.assertTrue(validate_signal(t1, y1))
            self.assertEqual(len(t1), round(SAMPLING_RATE * SIGNAL_DURATION))
            self.assertTrue(800 <= parameters['frequency'] <= 1200)
            self.assertTrue(0.8 <= parameters['amplitude'] <= 1.2)
            self.assertTrue(-0.05 <= parameters['dc_offset'] <= 0.05)
            _, different, _ = generate_signal(class_name, np.random.default_rng(124))
            self.assertFalse(np.array_equal(y1, different))

    def test_all_anomaly_subtypes(self):
        for subtype in ANOMALY_TYPES:
            time, signal, parameters = generate_signal('anomalous', np.random.default_rng(31), subtype)
            base_time, base, _ = generate_signal('normal', np.random.default_rng(31))
            start, end = parameters['change_start'], parameters['change_end']
            self.assertTrue(0 < start < end <= len(time))
            self.assertTrue(validate_signal(time, signal))
            np.testing.assert_array_equal(time, base_time)
            np.testing.assert_allclose(signal[:start], base[:start])
            self.assertFalse(np.allclose(signal[start:end], base[start:end]))
            if subtype == 'temporary_amplitude':
                self.assertLess(end, len(time))
                np.testing.assert_allclose(signal[end:], base[end:])

    def test_small_dataset_and_exact_reproduction(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'first', Path(folder) / 'second'
            generate_dataset(first, count=3, seed=91)
            generate_dataset(second, count=3, seed=91)
            features, summary = validate_dataset(first, count=3)
            self.assertEqual(summary['total'], 12)
            self.assertEqual(summary['unique_signals'], 12)
            self.assertEqual(len(features), 12)
            for filename in first.rglob('*.csv'):
                self.assertEqual(filename.read_bytes(), (second / filename.relative_to(first)).read_bytes())
            original = (first / 'normal/normal_001.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                generate_dataset(first, count=3)
            self.assertEqual((first / 'normal/normal_001.csv').read_bytes(), original)

    def test_original_class_entry_points_match_full_generator(self):
        import generate_signals as generator
        with tempfile.TemporaryDirectory() as folder:
            separate, together = Path(folder) / 'separate', Path(folder) / 'together'
            for class_name in CLASS_NAMES:
                getattr(generator, f'generate_{class_name}_signals')(1, output_root=separate)
            generate_dataset(together, count=1)
            for class_name in CLASS_NAMES:
                filename = f'{class_name}/{class_name}_001.csv'
                self.assertEqual((separate / filename).read_bytes(), (together / filename).read_bytes())

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            generate_dataset(root, count=1)
            (root / 'normal/normal_001.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'expected exactly'):
                validate_dataset(root, count=1)

    def test_malformed_signals_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            generate_dataset(root, count=1)
            path = root / 'normal/normal_001.csv'
            original = pd.read_csv(path)
            nonfinite = original.copy(); nonfinite.loc[1, 'amplitude'] = np.inf
            nan = original.copy(); nan.loc[1, 'amplitude'] = np.nan
            wrong_time = original.copy(); wrong_time['time'] *= 2
            nonnumeric = original.astype({'amplitude': 'object'}); nonnumeric.loc[1, 'amplitude'] = 'invalid'
            for case in [original.iloc[:-1], original.iloc[:0], nonfinite, nan,
                         wrong_time, nonnumeric, original.rename(columns={'amplitude': 'value'})]:
                with self.subTest(shape=case.shape):
                    case.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        validate_dataset(root, count=1)
            original.to_csv(path, index=False)

    def test_duplicate_signal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            generate_dataset(root, count=2)
            first = root / 'normal/normal_001.csv'
            (root / 'normal/normal_002.csv').write_bytes(first.read_bytes())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                validate_dataset(root, count=2)

    def test_development_reference_remains_recoverable(self):
        reference = PROJECT_ROOT / 'data/development_reference'
        data = pd.read_csv(reference / 'dataset.csv')
        self.assertEqual(data['class'].value_counts().to_dict(), {c: 20 for c in CLASS_NAMES})
        manifest = json.loads((reference / 'original_sha256.json').read_text())
        for original, expected in manifest.items():
            relative = original.removeprefix('data/')
            path = reference / relative
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__':
    unittest.main()
