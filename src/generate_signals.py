"""Reproducible synthetic signals; generation refuses to overwrite existing CSVs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import (CLASS_NAMES, PROJECT_ROOT, RANDOM_SEED, SAMPLING_RATE,
                    SIGNAL_DURATION, SIGNALS_PER_CLASS)

# Retain the original module's public parameter names.
sampling_rate = SAMPLING_RATE
duration = SIGNAL_DURATION
number_of_signals = SIGNALS_PER_CLASS
ANOMALY_TYPES = ['amplitude_increase', 'amplitude_decrease', 'temporary_amplitude',
                 'dc_offset_change', 'frequency_change']


def generate_base_signal(rng):
    """Same distributions for every class, with small natural noise and offset."""
    distribution = rng.choice(['uniform', 'triangular'])
    frequency = (rng.uniform(800, 1200) if distribution == 'uniform'
                 else rng.triangular(800, 1000, 1200))
    amplitude = rng.uniform(0.8, 1.2)
    phase = rng.uniform(0, 2 * np.pi)
    dc_offset = rng.uniform(-0.05, 0.05)
    natural_noise_std = rng.uniform(0.001, 0.01)
    time = np.arange(round(SAMPLING_RATE * SIGNAL_DURATION)) / SAMPLING_RATE
    carrier = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    natural_noise = rng.normal(0, natural_noise_std, len(time))
    parameters = dict(frequency=frequency, frequency_distribution=distribution,
                      amplitude=amplitude, phase=phase, dc_offset=dc_offset,
                      natural_noise_std=natural_noise_std)
    return time, carrier, natural_noise, parameters


def generate_signal(class_name, rng, anomaly_type=None):
    time, carrier, natural_noise, parameters = generate_base_signal(rng)
    signal = carrier + parameters['dc_offset'] + natural_noise
    frequency = parameters['frequency']
    amplitude = parameters['amplitude']
    if class_name == 'noisy':
        noise_std = rng.uniform(0.10, 0.35)
        signal += rng.normal(0, noise_std, len(time))
        parameters['noise_std'] = noise_std
    elif class_name == 'distorted':
        # Ratios are relative to the fundamental; harmonic phases also vary.
        for harmonic, low, high in [(2, 0.10, 0.25), (3, 0.05, 0.15)]:
            ratio, phase = rng.uniform(low, high), rng.uniform(0, 2*np.pi)
            signal += amplitude * ratio * np.sin(2*np.pi*harmonic*frequency*time + phase)
            parameters[f'harmonic_{harmonic}_ratio'] = ratio
            parameters[f'harmonic_{harmonic}_phase'] = phase
        fourth_ratio = rng.uniform(0.02, 0.06) if rng.random() < 0.25 else 0.0
        fourth_phase = rng.uniform(0, 2*np.pi)
        signal += amplitude * fourth_ratio * np.sin(2*np.pi*4*frequency*time + fourth_phase)
        parameters['harmonic_4_ratio'] = fourth_ratio
        parameters['harmonic_4_phase'] = fourth_phase
    elif class_name == 'anomalous':
        subtype = anomaly_type or str(rng.choice(ANOMALY_TYPES))
        if subtype not in ANOMALY_TYPES:
            raise ValueError(f'Unknown anomaly type: {subtype}')
        start = int(rng.integers(int(0.20*len(time)), int(0.65*len(time))))
        end = len(time)
        parameters.update(anomaly_type=subtype, change_start=start)
        if subtype in ['amplitude_increase', 'amplitude_decrease', 'temporary_amplitude']:
            if subtype == 'amplitude_increase':
                factor = rng.uniform(1.5, 2.5)
            elif subtype == 'amplitude_decrease':
                factor = rng.uniform(0.25, 0.60)
            else:
                factor = rng.uniform(1.5, 2.5) if rng.random() < 0.5 else rng.uniform(0.25, 0.60)
                end = min(start + int(rng.integers(int(0.15*len(time)), int(0.30*len(time)))), len(time)-1)
            # Change the carrier, preserving the shared natural offset/noise.
            signal[start:end] += (factor - 1) * carrier[start:end]
            parameters['amplitude_factor'] = factor
        elif subtype == 'dc_offset_change':
            change = amplitude * rng.uniform(0.30, 0.70) * rng.choice([-1, 1])
            signal[start:] += change
            parameters['offset_change'] = change
        else:
            factor = rng.uniform(0.55, 0.75) if rng.random() < 0.5 else rng.uniform(1.40, 1.75)
            new_frequency = frequency * factor
            # Continuous phase at the transition avoids an artificial phase jump.
            phase_at_change = 2*np.pi*frequency*time[start] + parameters['phase']
            changed = amplitude*np.sin(phase_at_change + 2*np.pi*new_frequency*(time[start:] - time[start]))
            signal[start:] += changed - carrier[start:]
            parameters['changed_frequency'] = new_frequency
        parameters['change_end'] = end
    elif class_name != 'normal':
        raise ValueError(f'Unknown signal class: {class_name}')
    return time, signal, parameters


def generate_class(class_name, count, rng, output_root):
    if count < 1:
        raise ValueError('Signals per class must be positive')
    directory = Path(output_root) / class_name
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing signals: {directory}')
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, count+1):
        time, signal, parameters = generate_signal(class_name, rng)
        filename = f'{class_name}_{index:03d}.csv'
        pd.DataFrame({'time': time, 'amplitude': signal}).to_csv(directory / filename, index=False)
        rows.append(dict(filename=filename, **{'class': class_name}, **parameters))
    print(f'{class_name}: {count} signals generated', flush=True)
    return rows


def class_rng(class_name, rng=None):
    if rng is not None:
        return rng
    streams = np.random.SeedSequence(RANDOM_SEED).spawn(len(CLASS_NAMES))
    return np.random.default_rng(streams[CLASS_NAMES.index(class_name)])


# Keep the original per-class entry points for callers and small experiments.
def generate_normal_signals(number_of_signals, rng=None, output_root=None):
    return generate_class('normal', number_of_signals, class_rng('normal', rng),
                          output_root or PROJECT_ROOT / 'data/raw')


def generate_noisy_signals(number_of_signals, rng=None, output_root=None):
    return generate_class('noisy', number_of_signals, class_rng('noisy', rng),
                          output_root or PROJECT_ROOT / 'data/raw')


def generate_distorted_signals(number_of_signals, rng=None, output_root=None):
    return generate_class('distorted', number_of_signals, class_rng('distorted', rng),
                          output_root or PROJECT_ROOT / 'data/raw')


def generate_anomalous_signals(number_of_signals, rng=None, output_root=None):
    return generate_class('anomalous', number_of_signals, class_rng('anomalous', rng),
                          output_root or PROJECT_ROOT / 'data/raw')


def generate_dataset(output_root=None, count=SIGNALS_PER_CLASS, seed=RANDOM_SEED):
    if count < 1:
        raise ValueError('Signals per class must be positive')
    output_root = Path(output_root or PROJECT_ROOT / 'data/raw_candidate')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Generate into an empty directory: {output_root}')
    # Independent class streams with identical base-parameter distributions.
    streams = np.random.SeedSequence(seed).spawn(len(CLASS_NAMES))
    rows = []
    for class_name, stream in zip(CLASS_NAMES, streams):
        rows.extend(generate_class(class_name, count, np.random.default_rng(stream), output_root))
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_root / 'generation_metadata.csv', index=False)
    (output_root / 'generation_config.json').write_text(json.dumps(dict(
        seed=seed, signals_per_class=count, sampling_rate=SAMPLING_RATE,
        duration=SIGNAL_DURATION, generator_version=2,
        source='Synthetic signals, not circuit simulator output'), indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=SIGNALS_PER_CLASS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/raw_candidate')
    args = parser.parse_args()
    generate_dataset(args.output, args.count, args.seed)
