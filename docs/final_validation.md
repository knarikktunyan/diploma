# Final implementation and validation

The desktop pipeline now uses the validated 13-feature Random Forest by default,
with a 13-feature SVM option. Both use the original 1,500 training records; no
refitting was performed during activation. Version-1 ten-feature artifacts remain
supported through explicit model paths. The baseline dataset and models are intact.

## Experimental evidence

Feature groups and the model choice were selected by training-only three-fold
macro F1. Definitions and model hashes were locked before generating the seed-4242
test cohort. The previously reviewed historical holdout was excluded from this
comparison. These are measurements on synthetic signals, not real circuit traces.

| model | variant | feature_group | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | baseline | 0.878 | 0.8815 | 0.878 | 0.8753 |
| random_forest | selected | combined | 0.9675 | 0.9677 | 0.9675 | 0.9675 |
| svm | baseline | baseline | 0.841 | 0.848 | 0.841 | 0.8389 |
| svm | selected | combined | 0.9215 | 0.9227 | 0.9215 | 0.9217 |

Precision, recall and F1 are macro averages. The four rows use exactly the same
2,000 fresh test records. Historical 89.8%/87.2% scores use a different 500-record
cohort and must not be treated as a paired comparison.

## Uncertainty and paired outcomes

| model | variant | support | correct | accuracy | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest | baseline | 2000 | 1756 | 0.878 | 0.8629 | 0.8916 |
| random_forest | selected | 2000 | 1935 | 0.9675 | 0.9588 | 0.9744 |
| svm | baseline | 2000 | 1682 | 0.841 | 0.8243 | 0.8564 |
| svm | selected | 2000 | 1843 | 0.9215 | 0.9089 | 0.9325 |

| model | test_count | errors_corrected | new_errors | both_correct | both_wrong |
| --- | --- | --- | --- | --- | --- |
| random_forest | 2000 | 196 | 17 | 1739 | 48 |
| svm | 2000 | 175 | 14 | 1668 | 143 |

Intervals are descriptive 95% Wilson accuracy intervals. Signals share one
synthetic generator; these intervals do not quantify uncertainty across circuits,
simulators or unseen fault mechanisms. Paired outcomes describe corrections and
new errors; no formal significance claim or independent real-world replication
is made. The evaluated fresh set must not be reused for further feature tuning.

## Compatibility verification

- Tests: 33 passed, 0 failed, 0 skipped.

- Both activated models reproduced all 2,000 frozen predictions; CSV roundtrips
  cover every class and every anomaly subtype. Original baseline hashes match.
- GUI checks use the real Qt event loop with the offscreen platform and supplied
  file-dialog responses. Loading, thirteen-feature analysis, waveform/spectrum,
  RF/SVM predictions, CSV export, overwrite refusal and error handling passed.
- Visible desktop/native-dialog operation was not reverified in this environment.
  The user previously verified desktop operation before the schema upgrade.
- Exports include ordered feature values, predicted class, model identity and
  feature/model schema versions. Prediction operates on the loaded snapshot.

## Scientific limitations

The new features improve within-record anomaly detection in this synthetic
population. Normal/distorted overlap remains. Crossing measurements can be affected
by noise/harmonics and too few cycles; window means depend on phase and duration.
The existing FFT has 200 Hz bin spacing for these records. Spectral energy and
amplitude features depend on length and physical units. Correlated features make
RF impurity importance descriptive rather than causal; do not interpret importance
as an independent contribution estimate. No adaptive feature/model tuning was
performed on the fresh test results.

## Real simulation validation

No identified real SPICE/Synopsys traces were available. Real-data validation is
unfinished. See [the input/provenance procedure](real_simulation_validation.md).
Do not describe the synthetic results as circuit-simulation validation.

## Thesis artifacts and reproduction

`data/final_evaluation/` contains CSV/LaTeX metric tables, per-class results,
accuracy intervals, paired outcomes, accuracy and RF-importance figures,
activation verification and a source/artifact/package-version manifest.
Detailed class reports, confusion matrices and subtype intervals are preserved
in `data/local_feature_experiment/`. See also [the experiment report](local_feature_experiment.md).

From the project root, run tests and offscreen GUI checks before reporting:

```bash
MPLBACKEND=Agg venv/bin/python -m unittest discover -s tests -v
QT_QPA_PLATFORM=offscreen venv/bin/python tests/check_gui.py
MPLBACKEND=Agg venv/bin/python src/final_report.py --tests-passed 33

```

The test count records the completed test run; report generation does not run
tests itself. Run `src/activate_models.py` only for deployment/compatibility checks
of the frozen candidates. It never retrains or selects using fresh-test scores.
Baseline retraining writes separate `models/baseline_training` and
`data/baseline_training` outputs and does not change the active selection.
