"""Descriptive review of the frozen synthetic holdout; no fitting or model changes."""
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT, SAMPLING_RATE
from validate_dataset import save_signal_plot


def wilson_interval(correct, total):
    """95% Wilson interval for a descriptive binomial success rate."""
    if total < 1 or not 0 <= correct <= total:
        raise ValueError('Require 0 <= correct <= total and total > 0')
    z = NormalDist().inv_cdf(0.975)
    rate = correct / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    radius = z*np.sqrt(rate*(1-rate)/total + z*z/(4*total*total))/denominator
    low = 0.0 if correct == 0 else max(0.0, center-radius)
    high = 1.0 if correct == total else min(1.0, center+radius)
    return low, high


def join_review_data(dataset, metadata, split, predictions):
    """Ensure every prediction is a unique test row before attaching audit metadata."""
    keys = ['filename', 'class']
    for name, frame in [('dataset', dataset), ('metadata', metadata), ('split', split)]:
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate signal identifiers in {name}')
    if predictions.duplicated(['model', 'filename', 'true_class']).any():
        raise ValueError('Duplicate heldout predictions')
    if not set(predictions['true_class']).issubset(CLASS_NAMES) or not set(predictions['predicted_class']).issubset(CLASS_NAMES):
        raise ValueError('Unsupported prediction labels')
    if set(predictions['model']) != {'random_forest', 'svm'}:
        raise ValueError('Both supported models must be present')
    expected = set(map(tuple, split.loc[split['split'] == 'test', keys].to_numpy()))
    for model, group in predictions.groupby('model'):
        actual = set(zip(group['filename'], group['true_class']))
        if actual != expected:
            raise ValueError(f'{model}: predictions do not match the test split')
    enriched = predictions.rename(columns={'true_class': 'class'}).merge(
        dataset, on=keys, how='left', validate='many_to_one', indicator=True)
    if not (enriched['_merge'] == 'both').all():
        raise ValueError('Prediction signal missing from dataset')
    enriched = enriched.drop(columns='_merge').merge(
        metadata, on=keys, how='left', validate='many_to_one', indicator=True)
    if not (enriched['_merge'] == 'both').all():
        raise ValueError('Prediction signal missing from generation metadata')
    enriched = enriched.drop(columns='_merge')
    enriched['correct'] = enriched['class'] == enriched['predicted_class']
    return enriched


def markdown_table(frame):
    # Avoid adding the optional tabulate dependency used by pandas.to_markdown.
    columns = [str(c) for c in frame.columns]
    rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    rows.extend('| ' + ' | '.join(str(value) for value in row) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def baseline_artifact_path(name, models_root=None):
    """Resolve historical selection after schema-2 activation, retaining hash checks."""
    root = models_root or PROJECT_ROOT / 'models'
    backup = root / 'baseline_selection.json'
    if name == 'selected_model.json' and backup.is_file():
        return backup
    return root / name


def review_errors():
    data_root = PROJECT_ROOT / 'data'
    analysis = data_root / 'feature_analysis'
    output = data_root / 'error_analysis'
    output.mkdir(exist_ok=True)
    docs = PROJECT_ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    dataset = pd.read_csv(data_root / 'dataset.csv')
    metadata = pd.read_csv(data_root / 'raw/generation_metadata.csv')
    split = pd.read_csv(analysis / 'split.csv')
    predictions = pd.read_csv(analysis / 'heldout_predictions.csv')
    comparison = pd.read_csv(analysis / 'model_comparison.csv')
    rows = join_review_data(dataset, metadata, split, predictions)
    # Validate frozen artifacts and stored reports before interpreting errors.
    manifest = json.loads((data_root / 'dataset_analysis/experiment_manifest.json').read_text())
    if hashlib.sha256((data_root/'dataset.csv').read_bytes()).hexdigest() != manifest['dataset_sha256']:
        raise ValueError('Dataset no longer matches the frozen experiment')
    for name, digest in manifest['models_sha256'].items():
        if hashlib.sha256(baseline_artifact_path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Model artifact changed: {name}')
    class_rows, subtype_rows, examples = [], [], []
    for model, group in rows.groupby('model'):
        report = classification_report(group['class'], group['predicted_class'], labels=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        stored = json.loads((analysis/f'{model}_report.json').read_text())
        if not np.isclose(report['accuracy'], stored['accuracy']):
            raise ValueError(f'{model}: stored accuracy disagrees with predictions')
        for class_name in CLASS_NAMES:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                if not np.isclose(report[class_name][metric], stored[class_name][metric]):
                    raise ValueError(f'{model}: stored class report is inconsistent')
            subgroup = group[group['class'] == class_name]
            correct = int(subgroup['correct'].sum())
            low, high = wilson_interval(correct, len(subgroup))
            class_rows.append(dict(model=model, **{'class': class_name}, support=len(subgroup),
                                   correct=correct, errors=len(subgroup)-correct,
                                   precision=report[class_name]['precision'], recall=correct/len(subgroup),
                                   f1=report[class_name]['f1-score'], recall_ci_low=low, recall_ci_high=high))
        matrix = confusion_matrix(group['class'], group['predicted_class'], labels=CLASS_NAMES)
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output/f'{model}_confusion_matrix.csv')
        for subtype, subgroup in group[group['class']=='anomalous'].groupby('anomaly_type'):
            correct = int(subgroup['correct'].sum())
            low, high = wilson_interval(correct, len(subgroup))
            subtype_rows.append(dict(model=model, anomaly_type=subtype, support=len(subgroup),
                                     correct=correct, missed=len(subgroup)-correct, recall=correct/len(subgroup),
                                     recall_ci_low=low, recall_ci_high=high))
        # Explicitly selected first filename per outcome, not a representative sample estimate.
        for subtype in ['frequency_change', 'temporary_amplitude']:
            for correct in [True, False]:
                candidates = group[(group['anomaly_type']==subtype)&(group['correct']==correct)].sort_values('filename')
                if not candidates.empty:
                    example = candidates.iloc[0]
                    label = 'correct' if correct else 'missed'
                    image_name = f'{model}_{subtype}_{label}.png'
                    save_signal_plot(data_root/'raw/anomalous'/example['filename'], output/image_name,
                                     f"{model}: {example['filename']} → {example['predicted_class']}")
                    examples.append(dict(model=model, subtype=subtype, outcome=label, filename=example['filename'],
                                         predicted=example['predicted_class'],
                                         change_start_seconds=example['change_start']/SAMPLING_RATE,
                                         image=image_name))
    classes, subtypes = pd.DataFrame(class_rows), pd.DataFrame(subtype_rows)
    classes.to_csv(output/'class_metrics.csv', index=False)
    subtypes.to_csv(output/'anomaly_subtype_metrics.csv', index=False)
    rows.to_csv(output/'heldout_audit.csv', index=False)
    rows[~rows['correct']].to_csv(output/'misclassified_signals.csv', index=False)
    pd.DataFrame(examples).to_csv(output/'selected_examples.csv', index=False)
    rows[~rows['correct']].groupby(['model','class','anomaly_type','predicted_class'], dropna=False).size().rename(
        'errors').reset_index().to_csv(output/'error_destinations.csv', index=False)

    # Descriptive overlap with training-normal feature ranges. These are not decision rules.
    training = dataset.merge(split[['filename','class','split']], on=['filename','class'], validate='one_to_one')
    normal = training[(training['class']=='normal')&(training['split']=='train')]
    overlap_rows = []
    for (model, subtype, correct), group in rows[rows['class']=='anomalous'].groupby(['model','anomaly_type','correct']):
        for feature in FEATURE_NAMES:
            low, high = normal[feature].quantile([0.05, 0.95])
            overlap_rows.append(dict(model=model, anomaly_type=subtype, correct=correct, feature=feature,
                                     support=len(group), median=float(group[feature].median()),
                                     normal_train_q05=float(low), normal_train_q95=float(high),
                                     fraction_inside_normal_train_range=float(group[feature].between(low, high).mean())))
    pd.DataFrame(overlap_rows).to_csv(output/'feature_overlap.csv', index=False)
    figure = Figure(figsize=(10, 5), layout='constrained')
    axis = figure.subplots()
    subtype_names = sorted(subtypes['anomaly_type'].unique())
    x = np.arange(len(subtype_names))
    for offset, model in [(-0.18, 'random_forest'), (0.18, 'svm')]:
        subset = subtypes[subtypes['model']==model].set_index('anomaly_type').loc[subtype_names]
        values = subset['recall'].to_numpy()
        error = np.vstack([values-subset['recall_ci_low'].to_numpy(), subset['recall_ci_high'].to_numpy()-values])
        axis.bar(x+offset, values, 0.36, label=model, yerr=np.maximum(error, 0), capsize=3)
    axis.set(xticks=x, xticklabels=[name.replace('_','\n') for name in subtype_names],
             ylim=(0,1.08), ylabel='Anomalous recall', title='Holdout subtype recall with 95% Wilson intervals')
    axis.legend()
    axis.grid(axis='y', alpha=0.3)
    figure.savefig(output/'anomaly_subtype_recall.png')
    inputs = [data_root/'dataset.csv', data_root/'raw/generation_metadata.csv',
              analysis/'split.csv', analysis/'heldout_predictions.csv', analysis/'model_comparison.csv']
    inputs.extend(baseline_artifact_path(name) for name in manifest['models_sha256'])
    (output/'review_manifest.json').write_text(json.dumps({
        'scope': 'Descriptive frozen-holdout review; no training or tuning',
        'input_sha256': {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs},
        'test_records_per_model': int(rows.groupby('model').size().iloc[0]),
        'new_features_or_models': False}, indent=2))
    write_report(docs/'experimental_results.md', comparison, classes, subtypes, rows, examples)
    print('Saved descriptive error review to', output)
    print(subtypes.to_string(index=False))
    return classes, subtypes


def write_report(path, comparison, classes, subtypes, rows, examples):
    errors = rows[~rows['correct']].groupby('model').size().to_dict()
    summary = comparison[['model','accuracy','precision','recall','f1','training_cv_f1']].round(4)
    subtype_table = subtypes[['model','anomaly_type','support','correct','missed','recall','recall_ci_low','recall_ci_high']].round(4)
    class_table = classes[['model','class','support','errors','precision','recall','f1']].round(4)
    weak = rows[rows['anomaly_type'].isin(['frequency_change', 'temporary_amplitude'])]
    frequency_overlap = []
    for (model, subtype, correct), group in weak.groupby(['model', 'anomaly_type', 'correct']):
        frequency_overlap.append(dict(model=model, subtype=subtype,
                                      outcome='correct' if correct else 'missed', support=len(group),
                                      dominant_frequency_in_800_1200=int(group['dominant_frequency'].between(800,1200).sum())))
    overlap_table = pd.DataFrame(frequency_overlap)
    missed = rows[(rows['class']=='anomalous') & (~rows['correct'])]
    miss_counts = missed.groupby(['model','anomaly_type']).size()
    totals = missed.groupby('model').size()
    rf_freq = int(miss_counts.get(('random_forest','frequency_change'), 0))
    svm_freq = int(miss_counts.get(('svm','frequency_change'), 0))
    rf_temp = int(miss_counts.get(('random_forest','temporary_amplitude'), 0))
    svm_temp = int(miss_counts.get(('svm','temporary_amplitude'), 0))
    text = f'''# Synthetic signal classification: experimental results and error review

## Scope and experimental protocol

This report describes a frozen synthetic experiment, not real SPICE/Synopsys
measurements. It adds descriptive analysis without generating data, fitting models,
changing feature definitions, or modifying the GUI.

The dataset has 2,000 independent generated records, 500 per class. Each record has
500 samples at 100 kHz (5 ms); the FFT bin spacing is 200 Hz. Classes share base
frequency/amplitude/phase/offset distributions. The generator seed is 42. The
original 80-signal dataset and models remain in data/development_reference/.

The existing seed-42 stratified split contains 1,500 training and 500 test records
(375/125 per class). Inputs are the same ten numerical features; filename and
all generation metadata are excluded. Preprocessing preserves noise, harmonics
and amplitude changes. SVM scaling stays inside its Pipeline. Three-fold macro
F1 cross-validation uses training data only; it selected Random Forest. The
stored models remain fitted only to the training set. No tuning used this holdout.

Dataset/model hashes and package versions are recorded in
../data/dataset_analysis/experiment_manifest.json. This review checks those hashes,
checks test membership and joins, and reproduces the stored classification reports
from the frozen predictions before writing any interpretation.

## Overall and per-class results

Precision, recall and F1 in the overall table are macro averages.

{markdown_table(summary)}

Random Forest makes {errors['random_forest']} errors out of 500; SVM makes
{errors['svm']} errors. The original 80-record experiment achieved 90% accuracy
for both models on 20 test records. Changes are −0.2 percentage points for RF and
−2.8 for SVM. Different sample sizes, signal distributions and test cases mean
these are separate experiments, not a controlled comparison on identical data.

{markdown_table(class_table)}

The largest per-class weakness is anomalous recall. Normal/distorted confusion
also occurs, so harmonic features help but do not separate every record.
Confusion-matrix CSVs and every misclassified signal are available under
../data/error_analysis/; they are derived from exactly the saved holdout predictions.

## Anomaly subtype results and uncertainty

{markdown_table(subtype_table)}

![Subtype recall](../data/error_analysis/anomaly_subtype_recall.png)

The intervals are descriptive 95% Wilson intervals for recall, treating records
as independent binomial outcomes. They quantify finite-sample uncertainty within
this synthetic holdout; they do not measure simulator/domain uncertainty. No
multiple-comparison correction or claim of statistically significant model
superiority is made. Even 19/19 or 31/31 detections do not establish perfect
sensitivity. Counts are small, and subtype proportions are generator choices.

Frequency changes account for {rf_freq} of RF's {totals['random_forest']} missed anomalies and {svm_freq} of SVM's {totals['svm']}.
Temporary-amplitude changes contribute another {rf_temp} RF misses and {svm_temp} SVM misses.
Together those two subtypes account for {rf_freq+rf_temp}/{totals['random_forest']} RF and {svm_freq+svm_temp}/{totals['svm']} SVM anomaly misses.
These are observed counts. Reasons below are hypotheses, not causal conclusions.

## Plausible limitations of the current features

{markdown_table(overlap_table)}

All missed frequency-change and temporary-amplitude cases in this experiment have
a dominant-frequency feature inside the normal 800–1200 Hz range. The table also
shows correctly detected cases in that range, so this overlap alone is not a
sufficient explanation or a decision rule.

The time features describe the whole record. They do not encode where a change
occurs, and aggregate values can overlap with clean records after a short event.
Dominant frequency is a single peak for the whole record, not a time-resolved
frequency trajectory. Coarse 200 Hz bins and a record of roughly 4–6 base cycles
can hide some frequency changes. The measured harmonic ratios also depend on
finite-record spectral leakage and phase; nominal generator ratios need not equal
measured FFT ratios. These facts suggest possible mechanisms but do not prove why
an individual tree/SVM decision was made.

feature_overlap.csv reports medians and the fraction of each anomaly/outcome group
inside each feature's 5th–95th percentile training-normal range. It is a descriptive
one-feature-at-a-time comparison, not a classifier, calibrated anomaly detector,
or evidence of joint feature-space overlap. A few values outside a range do not
establish that a model should detect the record.

RMS, standard deviation and spectral energy are strongly correlated. RF impurity
importance therefore should not be read as independent or causal contributions.
Largest recorded importances are peak-to-peak 0.1415, second harmonic ratio 0.1377,
and maximum 0.1337; they were learned, not manually imposed.

## Illustrative cases

The first filename in each model/subtype/correctness group is selected deterministically.
These plots are illustrations, not a representative sample or extra performance estimate.
Change locations in selected_examples.csv come from generation audit metadata;
that metadata is never available to the classifier.

'''
    for example in examples:
        text += f"- {example['model']}, {example['subtype']}, {example['outcome']}: [{example['filename']}](../data/error_analysis/{example['image']}), predicted {example['predicted']}, change starts at {example['change_start_seconds']:.5f} s.\n"
    text += '''
## Reproducibility and interpretation limits

Run `MPLBACKEND=Agg venv/bin/python src/review_errors.py` from the project directory.
Tables, audit CSVs and plots are rebuilt from the frozen artifacts; model fitting
is not invoked. Dataset/model integrity and report consistency are required.

The dataset remains narrow: one sample rate, one record length, sinusoidal bases,
Gaussian noise, a limited harmonic construction and five synthetic anomaly forms.
No conclusions about actual circuit faults, simulation engines or acquisition
conditions follow from these scores. The current GUI's model loading/schema is
unchanged; this stage does not provide new visible-desktop validation.

## Plan for a separate next experiment

1. Freeze this report, baseline artifacts and their hashes.
2. Predefine a small candidate set of time-local features for amplitude/offset and
   frequency changes. Preserve the existing ten-feature baseline for comparison.
3. Select candidates and hyperparameters using training-only cross-validation,
   ideally recording subtype scores on training validation folds. Generation
   metadata may be used for scoring strata, never as an input feature.
4. Evaluate the frozen baseline and candidate on a newly reserved synthetic test
   dataset, or a genuinely untouched real-simulation dataset when available. This
   holdout has now informed hypotheses and must not be reused as an unbiased final
   test for those improvements. Keep any historical results on it explicitly
   labeled exploratory.
5. Report class/subtype metrics, sample counts and uncertainty, then validate
   prediction/GUI compatibility before activating a model with a changed schema.

No candidate features, new model, new dataset or simulator integration are
implemented in this review stage. The observations justify a focused subsequent
experiment rather than changing generation parameters to raise accuracy.
'''
    path.write_text(text)


if __name__ == '__main__':
    review_errors()
