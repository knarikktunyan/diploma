# Signal Classification System

University diploma project for analysis of uniformly sampled CSV signals with
`time` (seconds) and `amplitude` columns. The active dataset contains 2,000 synthetic
experimental signals, 500 per class. The original 80-signal development set is
preserved under `data/development_reference/`. It is not SPICE/Synopsys validation data.

The active desktop/prediction models now use schema 2 (thirteen features): Random
Forest by default and SVM as an option. The original ten-feature dataset and
models remain preserved. See [final validation](docs/final_validation.md) for
deployment checks, thesis tables/figures and scientific limitations, and
[real simulation validation](docs/real_simulation_validation.md) for the remaining
data/provenance requirement.

## Environment and commands

Run from the project directory using Python 3.12 and the existing virtual environment:

```bash
venv/bin/python -m pip install -r requirements.txt
MPLBACKEND=Agg venv/bin/python -m unittest discover -s tests -v
MPLBACKEND=Agg venv/bin/python src/classifier.py
venv/bin/python src/prediction.py data/raw/normal/normal_001.csv --export data/prediction.csv
```

Install only missing dependencies if the environment already has the other libraries.
The classifier requires scikit-learn and joblib. PyQt6 is required for the desktop GUI.

Existing processing commands (do not run generation merely to train models):

```bash
venv/bin/python src/dataset_info.py
venv/bin/python src/preprocessing.py
venv/bin/python src/features.py
MPLBACKEND=Agg venv/bin/python src/analyze_features.py
```

## Synthetic dataset generation and validation

`src/config.py` defines `RANDOM_SEED = 42`, `SIGNALS_PER_CLASS = 500`,
`SAMPLING_RATE = 100000`, and `SIGNAL_DURATION = 0.005`. Every CSV has 500 samples
and exactly `time,amplitude` columns. Filenames run from `_001.csv` to `_500.csv`.

All classes share the same base distributions: frequency 800–1200 Hz, randomly
chosen uniform or triangular distribution; amplitude 0.8–1.2; phase 0–2π; DC offset
−0.05 to +0.05; and very small Gaussian natural noise (σ 0.001–0.01). Each class has
an independent NumPy Generator stream derived from the configured seed. There are
no paired/reused base waveforms across classes.

- Normal retains the clean base signal.
- Noisy adds Gaussian noise with σ 0.10–0.35.
- Distorted adds second/third harmonics at 0.10–0.25 and 0.05–0.15 **relative to
  fundamental amplitude**, with random phases. About 25% also have a small fourth
  harmonic (relative amplitude 0.02–0.06).
- Anomalous randomly uses amplitude increase, amplitude decrease, temporary
  amplitude change, DC-offset change, or phase-continuous frequency change.
  Start locations vary between 20% and 65% of the record; temporary changes return
  to the baseline. Increase factors are 1.5–2.5; decrease factors 0.25–0.60;
  DC shifts are ±0.30–0.70 times the base amplitude; frequency factors are
  0.55–0.75 or 1.40–1.75.

The generated anomaly subtype counts are 102 increases, 101 decreases, 89 temporary
changes, 115 DC shifts, and 93 frequency changes. Do not interpret this synthetic
mixture as a measured distribution of circuit faults.

Generation refuses to overwrite a nonempty destination. Generate a candidate,
validate it, preserve the current data/models, and only then activate it at the
existing `data/raw` path. CLI count/seed overrides also support small experiments:

```bash
venv/bin/python src/generate_signals.py --count 500 --seed 42 --output data/raw_candidate
MPLBACKEND=Agg venv/bin/python src/validate_dataset.py --root data/raw_candidate --count 500
# For an independent quick experiment, use --count 20 and a different empty output path.
```

Generation is not run by training, tests, prediction or the GUI. After activating
validated raw data, run the existing preprocessing and feature scripts, validate
processed data, and run feature analysis before training. The present expansion
followed this sequence. `data/raw/generation_metadata.csv` and `generation_config.json`
record the generation parameters and seed. They are audit records and are **never
ML inputs**. `data/dataset_analysis/` contains signal checksums, validation summaries,
feature statistics, base-parameter overlap histograms, and waveform/spectrum
examples for all four classes and five anomaly subtypes.

The original reference is recoverable from `data/development_reference/raw`,
`processed`, `dataset.csv`, `feature_analysis`, and `models`. The old generator and
config are retained in its `src` folder as historical snapshots. Its original file
hashes are recorded in `original_sha256.json`; this reference was checked after
activation of the larger dataset. Old models can be explicitly used with the
prediction CLI's `--model` option without changing the active models.

## Machine learning

`src/config.py` retains the ten-feature baseline and class definition.
`src/feature_schema.py` defines ordered versioned schemas: version 1 has the
original ten features; version 2 adds the three validated time-local features.
Filename is excluded from model inputs.
Default preprocessing preserves amplitude, noise and harmonics.

Training uses a reproducible stratified 75%/25% split: 1,500 training and 500 test
signals, with 375/125 per class. Random Forest
(200 trees) and an RBF SVM are evaluated on the identical held-out 500 signals.
SVM uses a StandardScaler pipeline; each scaler is fitted only on its training
fold. Training-only three-fold macro F1 selects the default model, so the holdout
is not used for selection. No hyperparameter search is performed.

The baseline training command remains available for reproducing the original
experiment. After a successful baseline training run:

- `models/baseline_training/random_forest.pkl` and `svm.pkl` contain models, ordered features,
  supported classes, preprocessing settings and random seed.
- `models/baseline_training/selected_model.json` identifies that baseline selection.
- `data/baseline_training/` receives per-class JSON reports, saved confusion-matrix
  images, model comparison CSV, split CSV, and Random Forest importance CSV/image.

Baseline retraining does not overwrite active models, selection or historical
evaluation outputs. The original baseline models remain at `models/random_forest.pkl`
and `models/svm.pkl`; explicit `--model` paths retain legacy prediction support.

The saved models are the training-only models that were evaluated. The holdout
is not subsequently used to refit them. Load only trusted joblib files, using the
same package environment that saved them.

`src/prediction.py` exposes `analyze_signal`, `load_model`, `classify_analysis`,
`predict_signal`, and `export_analysis` independently of a GUI. Prediction returns
Normal, Noisy, Distorted, or Anomalous. It infers sampling rate from CSV time values
and rejects nonuniform sampling rather than silently applying an invalid FFT.
Missing files, missing columns, nonnumeric values and invalid signals raise errors.

## Baseline experimental results and scientific limits

The following ten-feature results describe the original expanded-dataset
experiment. Current thirteen-feature results use a different fresh cohort and are
documented below and in the final validation report.

All 2,000 raw and 2,000 processed signals passed CSV, length, monotonic/uniform
sampling and finite-number validation. All 2,000 raw signals are distinct. The
2,000×12 feature dataset matched raw and processed feature recalculation.
Base frequency, amplitude and offset distributions overlap across classes.
Average measured second/third harmonic ratios are normal 0.0527/0.0271,
noisy 0.0596/0.0355, distorted 0.1320/0.0600, anomalous 0.0694/0.0332.
These FFT ratios are affected by spectral leakage and 200 Hz bin resolution;
they are not identical to the nominal generator ratios.

FFT bins remain 200 Hz apart for the current signals. Spectral energy retains the
original unnormalized FFT definition and depends on signal length and sampling.
Predictions for very different acquisition conditions require separate validation.
RF impurity-based importance can be affected by correlated features; it is not a
causal explanation. Even a 500-signal synthetic holdout cannot establish performance on
real circuit simulations.

ML training uses scikit-learn 1.9.1 and joblib 1.6.0. The larger dataset's
holdout metrics are below; precision/recall/F1 are macro averages:

| Model | Accuracy | Precision | Recall | F1 | Training CV F1 |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8980 | 0.9027 | 0.8980 | 0.8979 | 0.8785 |
| SVM | 0.8720 | 0.8813 | 0.8720 | 0.8720 | 0.8424 |

Random Forest remains the default, selected from training-only three-fold CV.
The original 80-signal dataset gave 90% accuracy for both models on 20 test signals.
The new accuracies are lower by 0.2 and 2.8 percentage points, respectively.
Dataset diversity, sample size and test cases all changed; these are separate
experiments, not a controlled comparison on an identical holdout. The old 20-case
accuracy also had much greater sampling uncertainty. No generator parameters or
hyperparameters were tuned in response to the new test scores.

Anomalous holdout recall is 0.808 for RF and 0.760 for SVM. Frequency-change recall
is only 0.455/0.409 (RF/SVM), and temporary-amplitude recall is 0.609/0.435. Both
models detect all tested amplitude increases and DC shifts, but these subgroups
contain only 19 and 31 holdout signals, respectively. Subtype evaluation is
saved in `data/feature_analysis/anomaly_subtype_evaluation.csv`; it is descriptive
holdout analysis, not used for selection/tuning. The existing global features do
not explicitly locate changes within a signal. RMS, standard deviation and
spectral energy are strongly correlated; RF impurity importance is therefore not
a causal or independent contribution estimate. Current highest RF importances
are peak-to-peak (0.1415), second harmonic ratio (0.1377), and maximum (0.1337).

All 33 tests pass with zero failures and skips. The suite preserves the original
12 tests and adds eight generator, validation, per-class API, and reference-recovery
tests, plus five error-review, four time-local feature, and four schema-compatibility
tests. Count assumptions now use
configuration instead of a fixed 80, and SVM's fitted scaler must contain exactly
1,500 training samples. Temporary integration-test retraining reproduced the new
metrics. New active models are installed only after candidate saved models pass
holdout-metric reproduction, metadata checks and raw-CSV prediction checks.
The original trained models remain in `data/development_reference/models`.
Saved-model verification reproduced all holdout metrics and checked 36 raw CSVs
per model, including seven random examples per class plus all five anomaly types.
RF correctly classifies each `_001.csv` example; SVM classifies `distorted_001.csv`
as Normal. The checks verify functionality/schema consistency, not perfect
classification. Default CLI prediction and CSV export were also verified after
activation. `experiment_manifest.json` records dataset/model hashes and package
versions; `old_vs_new_model_comparison.csv` records the historical comparison.

## Running the GUI

From the project directory:

```bash
venv/bin/python main.py
```

With the virtual environment activated, the equivalent command is `python main.py`.
Install PyQt6 into that environment first if it is missing:

```bash
venv/bin/python -m pip install PyQt6
```

The resizable desktop window provides:

1. **Load Signal**: select a CSV with `time` and `amplitude` columns. The waveform
   appears in the Signal tab. Invalid files show a message and retain the previous
   valid signal.
2. **Analyze**: display all thirteen features and the unwindowed FFT magnitude in the
   Spectrum tab. Analysis uses the loaded in-memory signal, not a later reread of
   the file. Uniform sampling and at least 32 samples are required for schema-2 analysis.
3. **Classify**: use the existing default model (currently Random Forest), or select
   SVM. Classification automatically analyzes if necessary. A large label displays
   NORMAL, NOISY, DISTORTED, or ANOMALOUS. Changing the model or analyzing again
   clears the previous prediction.
4. **Export**: save features, filename, sampling rate, predicted class, feature-schema
   version and model identity as CSV.
   Analysis without classification exports an empty predicted-class field. Existing
   output files require confirmation; overwriting the loaded signal is prevented.

The plots have Matplotlib navigation toolbars for zooming/panning. The GUI calls
`load_signal`, `analyze_loaded_signal`, `classify_analysis`, and `export_analysis`;
model loading and feature calculation remain in the existing src pipeline.
It does not train models or regenerate the dataset.

### GUI compatibility checks

The user manually verified the GUI before dataset expansion. The schema upgrade
adds three feature rows and uses the selected-model loader with versioned
artifacts; the existing layout and visualization remain. A scripted compatibility
check is available:

```bash
# With a working desktop display:
venv/bin/python tests/check_gui.py
# For widget/rendering checks without a desktop:
QT_QPA_PLATFORM=offscreen venv/bin/python tests/check_gui.py
```

Offscreen checks exercise all listed widget operations and both model choices.
A native Wayland launch during the earlier dataset stage failed before window
creation with `Failed to create wl_display (Operation not permitted)`.
Visible desktop operation after the schema upgrade was not reverified here.

This check runs the real Qt event loop and buttons with file-dialog answers
supplied by the test. It checks loading, thirteen features, waveform/spectrum plotting,
both models, CSV export, overwrite refusal and error handling; it saves a JSON
report and screenshot under `data/dataset_analysis`. Offscreen verification does
not establish native-dialog interaction or visible desktop operation. Display
errors in WSL require checking the desktop/WSLg environment rather than changing
the GUI framework.

## Experimental error review

[Experimental results and error review](docs/experimental_results.md) documents
class/subtype performance, Wilson recall intervals, error destinations, illustrative
correct/missed records, feature overlap and limitations of the synthetic experiment.
The reproducible descriptive analysis command is:

```bash
MPLBACKEND=Agg venv/bin/python src/review_errors.py
```

Outputs are in `data/error_analysis/`, including every heldout error, numerical
confusion matrices, subtype-recall plots, example plots and an input-hash manifest.
The review does not train models, change features or regenerate data. The four new
tests check interval calculations, joins, duplicate/missing metadata and rejection
of training predictions in heldout analysis.

Frequency-change and temporary-amplitude subtypes account for 21/24 RF and 26/30
SVM missed anomalies. Every missed case in those two groups has a dominant-frequency
feature in the normal 800–1200 Hz range, but correct cases also occur in that range;
this is an observed overlap, not proof of a causal explanation.

The follow-up experiment below predefines time-local candidate features,
selects them on training-only validation, and uses a fresh untouched test set.
This reviewed holdout has informed hypotheses and must not be reused as an unbiased
final test for improvements inspired by it.
No real SPICE/Synopsys connection or additional model architecture was added.

## Time-local feature experiment

The isolated experiment in `src/local_feature_experiment.py` compares the original
ten features with three additional measurements: windowed RMS variation, windowed
mean variation, and rising-crossing period variation. Definitions are centralized
in `src/local_features.py`; existing feature extraction is reused unchanged.

The original 1,500 training records and identical stratified three-fold splits
select a feature group separately for RF and SVM using macro F1. Scaling remains
inside the SVM pipeline. Selection and model hashes are locked before generating
an independent seed-4242 test cohort of 500 signals per class. The previously
reviewed 500-record holdout is excluded. Audit filenames and anomaly metadata are
never model inputs.

```bash
# Run only for a new, empty experiment directory; existing evidence is protected.
MPLBACKEND=Agg venv/bin/python src/local_feature_experiment.py --prepare
MPLBACKEND=Agg venv/bin/python src/local_feature_experiment.py --evaluate
```

See [the experiment report](docs/local_feature_experiment.md) for measured results
and limitations. Protocol, validation, CV results, predictions, confusion matrices
and saved candidate models are under `data/local_feature_experiment/`.
On the same fresh 2,000 records, RF accuracy increased from 87.80% to 96.75%
and SVM from 84.10% to 92.15%. Both selected all three additional features using
training-only CV. All eight saved-model CSV roundtrips passed. These results apply
to the synthetic generator, not to real circuit simulations.
`predict_experimental_signal` reloads a candidate artifact and analyzes a generic
CSV through the existing pipeline. Experimental artifacts have their own ordered
feature metadata and are not replacements for the production ten-feature loader.
The active dataset, models and GUI remained unchanged during that isolated
experiment. Its validated candidates are now integrated through the versioned
schema described below. The fresh test must not be
used for subsequent feature tuning after its evaluation.

## Activated models and architecture

`models/time_local/random_forest.pkl` and `svm.pkl` are schema-2 copies of the
frozen experiment candidates, with ordered feature metadata and provenance.
`models/selected_model.json` points to the selected RF artifact;
`models/baseline_selection.json` preserves the original selection. Activation uses
the already locked training-only CV scores, verifies frozen predictions and CSV
roundtrips, and does not refit on either test cohort:

```bash
MPLBACKEND=Agg venv/bin/python src/activate_models.py
```

The prediction loader validates schema, fitted feature order, classes and
preprocessing. Legacy artifacts with no explicit schema version are interpreted
as version 1. GUI analysis displays schema 2; command-line prediction computes
the features required by the loaded model. Exports include schema versions and
model identity. Analysis functions retain version-1 defaults for existing feature
dataset builders and historical experiment scripts. Schema-2 signals need at
least 32 samples; shorter signals can still use legacy analysis/models.

```text
CSV → validation/preprocessing → versioned features → saved RF/SVM → prediction
                                      ↓                              ↓
                              waveform/spectrum                CSV export
                                      └──────── PyQt6 GUI ────────────┘
```

Generic CSV loading does not establish real-circuit accuracy. No real simulation
validation has been performed without identified traces and independently
established labels. The validation procedure is documented separately.
