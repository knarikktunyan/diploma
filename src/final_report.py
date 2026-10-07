"""Export thesis tables/figures and an audit manifest from frozen evaluation outputs."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd
from matplotlib.figure import Figure

from config import CLASS_NAMES, PROJECT_ROOT
from review_errors import baseline_artifact_path, markdown_table, wilson_interval


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_report(tests_passed):
    experiment = PROJECT_ROOT / 'data/local_feature_experiment'
    output = PROJECT_ROOT / 'data/final_evaluation'
    activation = json.loads((output / 'activation_verification.json').read_text())
    gui = json.loads((PROJECT_ROOT / 'data/dataset_analysis/gui_verification.json').read_text())
    if activation['status'] != 'passed' or gui['status'] != 'passed':
        raise ValueError('Activation and GUI checks must pass before final reporting')
    if gui.get('feature_schema_version') != 2 or gui.get('feature_count') != 13:
        raise ValueError('Run the updated GUI schema compatibility check first')
    for check in activation['checks']:
        if digest(PROJECT_ROOT / 'models/time_local' / f"{check['model']}.pkl") != check['artifact_sha256']:
            raise ValueError('Activated model changed after verification')
    protocol = json.loads((experiment / 'protocol.json').read_text())
    for relative, expected in protocol['protected_baseline_sha256'].items():
        path = (baseline_artifact_path('selected_model.json') if relative == 'models/selected_model.json'
                else PROJECT_ROOT / relative)
        if digest(path) != expected:
            raise ValueError(f'Preserved baseline changed: {relative}')
    comparison = pd.read_csv(experiment / 'fresh_test_results.csv')
    predictions = pd.read_csv(experiment / 'fresh_test_predictions.csv')
    intervals, paired, class_rows = [], [], []
    for model in ['random_forest', 'svm']:
        rows = predictions[predictions['model'] == model]
        baseline = rows[rows['variant'] == 'baseline'].set_index('filename')
        selected = rows[rows['variant'] == 'selected'].set_index('filename').loc[baseline.index]
        if not baseline['true_class'].equals(selected['true_class']):
            raise ValueError('Paired predictions have different targets')
        before = baseline['true_class'] == baseline['predicted_class']
        after = selected['true_class'] == selected['predicted_class']
        paired.append(dict(model=model, test_count=len(before),
                           errors_corrected=int((~before & after).sum()),
                           new_errors=int((before & ~after).sum()),
                           both_correct=int((before & after).sum()),
                           both_wrong=int((~before & ~after).sum())))
        for variant, correct in [('baseline', before), ('selected', after)]:
            low, high = wilson_interval(int(correct.sum()), len(correct))
            intervals.append(dict(model=model, variant=variant, support=len(correct),
                                  correct=int(correct.sum()), accuracy=float(correct.mean()),
                                  ci_low=low, ci_high=high))
            report = json.loads((experiment / f'{model}_{variant}_report.json').read_text())
            for label in CLASS_NAMES:
                class_rows.append(dict(model=model, variant=variant, **{'class': label}, **report[label]))
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame(intervals).to_csv(output / 'accuracy_intervals.csv', index=False)
    pd.DataFrame(paired).to_csv(output / 'paired_prediction_outcomes.csv', index=False)
    pd.DataFrame(class_rows).to_csv(output / 'per_class_metrics.csv', index=False)
    # Standalone LaTeX table without an additional templating dependency.
    lines = [r'\begin{tabular}{llrrrr}', r'\hline',
             r'Model & Features & Accuracy & Precision & Recall & F1 \\', r'\hline']
    for row in comparison.to_dict('records'):
        label = 'Random Forest' if row['model'] == 'random_forest' else 'SVM'
        count = 10 if row['variant'] == 'baseline' else 13
        scores = ' & '.join(f"{row[key]:.4f}" for key in ['accuracy', 'precision', 'recall', 'f1'])
        lines.append(f'{label} & {count} & {scores} ' + r'\\')
    lines.extend([r'\hline', r'\end{tabular}'])
    (output / 'model_comparison.tex').write_text('\n'.join(lines) + '\n')
    figure = Figure(figsize=(9, 5), layout='constrained')
    axis = figure.subplots()
    labels = ['RF: 10 features', 'RF: 13 features', 'SVM: 10 features', 'SVM: 13 features']
    values = comparison['accuracy'].to_numpy() * 100
    bars = axis.bar(labels, values, color=['#8ba6b9', '#24577a', '#8ba6b9', '#24577a'])
    axis.bar_label(bars, fmt='%.2f%%')
    axis.set(ylim=(0, 105), ylabel='Accuracy (%)', title='Same fresh synthetic test cohort: 2,000 signals')
    figure.savefig(output / 'accuracy_comparison.png')
    sources = [p for directory in ['src', 'gui', 'tests'] for p in (PROJECT_ROOT / directory).glob('*.py')]
    sources += [PROJECT_ROOT / 'main.py', PROJECT_ROOT / 'requirements.txt']
    artifacts = [PROJECT_ROOT / 'data/dataset.csv',
                 experiment / 'protocol.json', experiment / 'selection_locked.json',
                 experiment / 'training_features.csv', experiment / 'fresh_test/features.csv',
                 experiment / 'fresh_test_predictions.csv']
    artifacts += [p for p in (PROJECT_ROOT / 'models').rglob('*') if p.is_file()]
    manifest = dict(python=platform.python_version(),
                    packages={name: version(name) for name in ['numpy', 'scipy', 'pandas', 'scikit-learn',
                                                              'matplotlib', 'PyQt6', 'joblib']},
                    tests_passed=tests_passed, tests_failed=0, tests_skipped=0,
                    gui_platform=gui['platform'], real_simulation_validation='Not performed: no identified real traces',
                    source_sha256={str(p.relative_to(PROJECT_ROOT)): digest(p) for p in sources},
                    artifact_sha256={str(p.relative_to(PROJECT_ROOT)): digest(p) for p in artifacts})
    (output / 'reproducibility_manifest.json').write_text(json.dumps(manifest, indent=2))
    text = '''# Final implementation and validation

The desktop pipeline now uses the validated 13-feature Random Forest by default,
with a 13-feature SVM option. Both use the original 1,500 training records; no
refitting was performed during activation. Version-1 ten-feature artifacts remain
supported through explicit model paths. The baseline dataset and models are intact.

## Experimental evidence

Feature groups and the model choice were selected by training-only three-fold
macro F1. Definitions and model hashes were locked before generating the seed-4242
test cohort. The previously reviewed historical holdout was excluded from this
comparison. These are measurements on synthetic signals, not real circuit traces.

'''+markdown_table(comparison.round(4))+'''

Precision, recall and F1 are macro averages. The four rows use exactly the same
2,000 fresh test records. Historical 89.8%/87.2% scores use a different 500-record
cohort and must not be treated as a paired comparison.

## Uncertainty and paired outcomes

'''+markdown_table(pd.DataFrame(intervals).round(4))+'\n\n'+markdown_table(pd.DataFrame(paired))+'''

Intervals are descriptive 95% Wilson accuracy intervals. Signals share one
synthetic generator; these intervals do not quantify uncertainty across circuits,
simulators or unseen fault mechanisms. Paired outcomes describe corrections and
new errors; no formal significance claim or independent real-world replication
is made. The evaluated fresh set must not be reused for further feature tuning.

## Compatibility verification

'''+f'- Tests: {tests_passed} passed, 0 failed, 0 skipped.\n'+'''
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
'''+f'MPLBACKEND=Agg venv/bin/python src/final_report.py --tests-passed {tests_passed}\n'+'''
```

The test count records the completed test run; report generation does not run
tests itself. Run `src/activate_models.py` only for deployment/compatibility checks
of the frozen candidates. It never retrains or selects using fresh-test scores.
Baseline retraining writes separate `models/baseline_training` and
`data/baseline_training` outputs and does not change the active selection.
'''
    (PROJECT_ROOT / 'docs/final_validation.md').write_text(text)
    print(f'Final tables, figures, manifest and report saved in {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests-passed', type=int, required=True)
    args = parser.parse_args()
    if args.tests_passed < 1:
        parser.error('--tests-passed must record a completed successful test run')
    generate_report(args.tests_passed)
