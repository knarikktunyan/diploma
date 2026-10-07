"""Validate a synthetic CSV dataset and save descriptive statistics/examples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT, SAMPLING_RATE, SIGNAL_DURATION, SIGNALS_PER_CLASS
from features import extract_features
from preprocessing import validate_signal


def validate_dataset(root, count=SIGNALS_PER_CLASS, output_directory=None):
    root = Path(root)
    if count < 1:
        raise ValueError('Expected class count must be positive')
    expected_time = np.arange(round(SAMPLING_RATE * SIGNAL_DURATION)) / SAMPLING_RATE
    records, manifest, seen = [], [], set()
    for class_name in CLASS_NAMES:
        directory = root / class_name
        expected_names = {f'{class_name}_{i:03d}.csv' for i in range(1, count+1)}
        files = sorted(directory.glob('*.csv'))
        if {f.name for f in files} != expected_names:
            raise ValueError(f'{class_name}: expected exactly {count} consistently numbered CSVs')
        for filename in files:
            data = pd.read_csv(filename)
            if list(data.columns) != ['time', 'amplitude']:
                raise ValueError(f'{filename}: expected time,amplitude columns')
            try:
                time = data['time'].to_numpy(dtype=float)
                signal = data['amplitude'].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{filename}: nonnumeric signal') from error
            if not validate_signal(time, signal):
                raise ValueError(f'{filename}: empty, nonfinite or non-increasing signal')
            if len(time) != len(expected_time):
                raise ValueError(f'{filename}: expected {len(expected_time)} samples')
            if not np.allclose(time, expected_time, rtol=0, atol=1e-12):
                raise ValueError(f'{filename}: invalid synthetic sampling interval or time grid')
            fingerprint = hashlib.sha256(signal.tobytes()).hexdigest()
            if fingerprint in seen:
                raise ValueError(f'{filename}: duplicate signal amplitudes')
            seen.add(fingerprint)
            records.append(dict(**extract_features(signal, SAMPLING_RATE), filename=filename.name,
                                **{'class': class_name}))
            manifest.append(dict(filename=filename.name, **{'class': class_name},
                                 sha256=hashlib.sha256(filename.read_bytes()).hexdigest()))
        print(f'{class_name}: {len(files)} CSVs validated', flush=True)
    features = pd.DataFrame(records)
    summary = dict(root=str(root), classes={c: count for c in CLASS_NAMES}, total=len(features),
                   samples_per_signal=len(expected_time), sampling_rate=SAMPLING_RATE,
                   csv_validation='passed', signal_length_validation='passed',
                   time_validation='passed', finite_numeric_validation='passed',
                   unique_signals=len(seen), source='Synthetic experimental dataset')
    if output_directory is not None:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        features.to_csv(output / 'validated_features.csv', index=False)
        pd.DataFrame(manifest).to_csv(output / 'signal_manifest.csv', index=False)
        statistics = []
        for class_name, group in features.groupby('class'):
            for name in FEATURE_NAMES:
                statistics.append(dict(**{'class': class_name}, feature=name,
                                       **group[name].describe().to_dict()))
        pd.DataFrame(statistics).to_csv(output / 'feature_statistics.csv', index=False)
        features.groupby('class')[FEATURE_NAMES].mean().to_csv(output / 'class_feature_means.csv')
        save_examples(root, output)
        metadata_path = root / 'generation_metadata.csv'
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            expected_pairs = set(zip(features['class'], features['filename']))
            actual_pairs = set(zip(metadata['class'], metadata['filename']))
            if len(metadata) != len(features) or expected_pairs != actual_pairs:
                raise ValueError('Generation metadata does not match the validated signal files')
            metadata.groupby('class')[['frequency', 'amplitude', 'dc_offset']].agg(
                ['mean', 'std', 'min', 'max']).to_csv(output / 'base_parameter_statistics.csv')
            anomalies = metadata[metadata['class'] == 'anomalous']
            anomalies['anomaly_type'].value_counts().to_csv(output / 'anomaly_subtype_counts.csv')
            for subtype, group in anomalies.groupby('anomaly_type'):
                save_signal_plot(root / 'anomalous' / group.iloc[0]['filename'],
                                 output / f'anomalous_{subtype}_example.png', subtype)
            save_parameter_comparison(metadata, output)
        (output / 'validation.json').write_text(json.dumps(summary, indent=2))
    return features, summary


def save_signal_plot(filename, destination, title):
    data = pd.read_csv(filename)
    time, signal = data['time'].to_numpy(), data['amplitude'].to_numpy()
    figure = Figure(figsize=(11, 4), layout='constrained')
    waveform, spectrum = figure.subplots(1, 2)
    waveform.plot(time, signal, linewidth=1)
    waveform.set(title=title, xlabel='Time (s)', ylabel='Amplitude')
    frequencies = np.fft.rfftfreq(len(signal), 1/SAMPLING_RATE)
    spectrum.plot(frequencies, np.abs(np.fft.rfft(signal)))
    spectrum.set(xlabel='Frequency (Hz)', ylabel='FFT magnitude', xlim=(0, 6000))
    waveform.grid(True, alpha=0.3)
    spectrum.grid(True, alpha=0.3)
    figure.savefig(destination)


def save_examples(root, output):
    for class_name in CLASS_NAMES:
        save_signal_plot(root / class_name / f'{class_name}_001.csv',
                         output / f'{class_name}_example.png', class_name.capitalize())


def save_parameter_comparison(metadata, output):
    figure = Figure(figsize=(12, 4), layout='constrained')
    axes = figure.subplots(1, 3)
    for axis, parameter in zip(axes, ['frequency', 'amplitude', 'dc_offset']):
        for class_name in CLASS_NAMES:
            values = metadata[metadata['class'] == class_name][parameter]
            axis.hist(values, bins=20, density=True, histtype='step', label=class_name)
        axis.set(xlabel=f'Base {parameter}', ylabel='Density')
    axes[-1].legend()
    figure.savefig(output / 'base_parameter_overlap.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data/raw')
    parser.add_argument('--count', type=int, default=SIGNALS_PER_CLASS)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/dataset_analysis')
    args = parser.parse_args()
    _, summary = validate_dataset(args.root, args.count, args.output)
    print(json.dumps(summary, indent=2))
