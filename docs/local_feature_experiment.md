# Time-local feature experiment

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

| model | group | features | cv_f1 | cv_fold_std | fold_1 | fold_2 | fold_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | 10 | 0.8785 | 0.018 | 0.8915 | 0.853 | 0.8909 |
| random_forest | amplitude_offset | 12 | 0.932 | 0.0089 | 0.9249 | 0.9264 | 0.9445 |
| random_forest | frequency | 11 | 0.9254 | 0.011 | 0.9099 | 0.9344 | 0.932 |
| random_forest | combined | 13 | 0.9552 | 0.011 | 0.9397 | 0.9621 | 0.964 |
| svm | baseline | 10 | 0.8424 | 0.0004 | 0.8422 | 0.843 | 0.842 |
| svm | amplitude_offset | 12 | 0.8962 | 0.0019 | 0.8936 | 0.8975 | 0.8976 |
| svm | frequency | 11 | 0.8658 | 0.0031 | 0.8627 | 0.87 | 0.8647 |
| svm | combined | 13 | 0.9092 | 0.005 | 0.9066 | 0.9163 | 0.9049 |

- random_forest: selected **combined**, 13 features, CV macro F1 0.9552.
- svm: selected **combined**, 13 features, CV macro F1 0.9092.

## Fresh evaluation

Only after selection/models were locked was a seed-4242 test dataset generated:
500 records per class, 2,000 total, using the unchanged synthetic generator.
All CSVs passed validation and no exact file hashes overlap the original dataset.
There are no paired original/fresh waveforms. Both frozen baseline and selected
models use the same original 1,500 training records and exactly the same fresh
2,000 test records. Baseline evaluation loads the actual active saved models.
No test record is used to refit/select a model, scaler or feature group.

| model | variant | feature_group | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | baseline | 0.878 | 0.8815 | 0.878 | 0.8753 |
| random_forest | selected | combined | 0.9675 | 0.9677 | 0.9675 | 0.9675 |
| svm | baseline | baseline | 0.841 | 0.848 | 0.841 | 0.8389 |
| svm | selected | combined | 0.9215 | 0.9227 | 0.9215 | 0.9217 |

Precision, recall and F1 are macro averages. Historical 89.8%/87.2% accuracies belong
to a different, previously reviewed 500-record test set; use the paired baseline
and selected rows above for this experiment's comparison.

RF accuracy improved by 8.95 percentage points and SVM by 8.05 points on this
same cohort. RF's remaining 65 errors include 62 normal/distorted confusions,
one normal signal classified as anomalous, and two anomalous signals classified
as noisy. Improved anomaly detection therefore does not resolve all harmonic
class overlap. RF frequency-change recall increased from 40/102 to 102/102;
temporary-amplitude recall increased from 37/83 to 83/83. These perfect subgroup
scores describe this finite synthetic cohort, not guaranteed future performance.

## Subtype evaluation

| model | variant | anomaly_type | support | correct | recall | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | amplitude_decrease | 99 | 80 | 0.8081 | 0.7196 | 0.8735 |
| random_forest | baseline | amplitude_increase | 121 | 108 | 0.8926 | 0.8248 | 0.9361 |
| random_forest | baseline | dc_offset_change | 95 | 91 | 0.9579 | 0.8967 | 0.9835 |
| random_forest | baseline | frequency_change | 102 | 40 | 0.3922 | 0.303 | 0.4892 |
| random_forest | baseline | temporary_amplitude | 83 | 37 | 0.4458 | 0.3436 | 0.5527 |
| random_forest | selected | amplitude_decrease | 99 | 99 | 1.0 | 0.9626 | 1.0 |
| random_forest | selected | amplitude_increase | 121 | 121 | 1.0 | 0.9692 | 1.0 |
| random_forest | selected | dc_offset_change | 95 | 93 | 0.9789 | 0.9265 | 0.9942 |
| random_forest | selected | frequency_change | 102 | 102 | 1.0 | 0.9637 | 1.0 |
| random_forest | selected | temporary_amplitude | 83 | 83 | 1.0 | 0.9558 | 1.0 |
| svm | baseline | amplitude_decrease | 99 | 77 | 0.7778 | 0.6864 | 0.8484 |
| svm | baseline | amplitude_increase | 121 | 105 | 0.8678 | 0.796 | 0.9169 |
| svm | baseline | dc_offset_change | 95 | 92 | 0.9684 | 0.9112 | 0.9892 |
| svm | baseline | frequency_change | 102 | 34 | 0.3333 | 0.2494 | 0.4294 |
| svm | baseline | temporary_amplitude | 83 | 38 | 0.4578 | 0.3549 | 0.5645 |
| svm | selected | amplitude_decrease | 99 | 99 | 1.0 | 0.9626 | 1.0 |
| svm | selected | amplitude_increase | 121 | 121 | 1.0 | 0.9692 | 1.0 |
| svm | selected | dc_offset_change | 95 | 94 | 0.9895 | 0.9428 | 0.9981 |
| svm | selected | frequency_change | 102 | 97 | 0.951 | 0.8903 | 0.9789 |
| svm | selected | temporary_amplitude | 83 | 82 | 0.988 | 0.9349 | 0.9979 |

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
