"""Train and compare models on one reproducible, stratified holdout."""
import json
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT, RANDOM_SEED


def load_dataset(filename):
    data = pd.read_csv(filename)
    missing = set(FEATURE_NAMES + ["class"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    features = data[FEATURE_NAMES].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Dataset features must be finite")
    if set(data['class']) != set(CLASS_NAMES):
        raise ValueError("Dataset must contain exactly the four supported classes")
    if data['class'].value_counts().min() < 5:
        raise ValueError("At least five examples per class are required")
    return features, data['class']


def create_models():
    return {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        'svm': make_pipeline(StandardScaler(), SVC(kernel='rbf')),
    }


def save_confusion_matrix(target, predicted, filename, title):
    matrix = confusion_matrix(target, predicted, labels=CLASS_NAMES)
    figure = Figure(figsize=(7, 6))
    axis = figure.subplots()
    plot = axis.imshow(matrix, cmap='Blues')
    figure.colorbar(plot, ax=axis)
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_NAMES,
             yticklabels=CLASS_NAMES, xlabel='Predicted class', ylabel='True class', title=title)
    for i in range(4):
        for j in range(4):
            axis.text(j, i, str(matrix[i, j]), ha='center', va='center')
    figure.tight_layout()
    figure.savefig(filename)


def train_and_evaluate(dataset_path=None, output_directory=None, model_directory=None):
    dataset_path = dataset_path or PROJECT_ROOT / 'data/dataset.csv'
    output = output_directory or PROJECT_ROOT / 'data/baseline_training'
    # Baseline experiments must not overwrite the activated schema-2 model selection.
    models_path = model_directory or PROJECT_ROOT / 'models/baseline_training'
    output.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
    # Choose using training-only cross-validation; holdout metrics are evaluation only.
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    results = []
    for name, model in create_models().items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=folds, scoring='f1_macro').mean()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        report = classification_report(y_test, predicted, labels=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        print(f'\n{name}\n' + classification_report(y_test, predicted, labels=CLASS_NAMES, zero_division=0))
        (output / f'{name}_report.json').write_text(json.dumps(report, indent=2))
        save_confusion_matrix(y_test, predicted, output / f'{name}_confusion_matrix.png', name)
        results.append(dict(model=name, accuracy=accuracy_score(y_test, predicted),
                            precision=report['macro avg']['precision'],
                            recall=report['macro avg']['recall'], f1=report['macro avg']['f1-score'],
                            training_cv_f1=float(cv_f1)))
        # Preserve the evaluated training-only model; never refit on the holdout.
        joblib.dump(dict(model=model, features=FEATURE_NAMES, classes=CLASS_NAMES,
                         preprocessing={'remove_dc': False, 'filter_signal': False, 'normalize': False},
                         random_seed=RANDOM_SEED), models_path / f'{name}.pkl')
        if name == 'random_forest':
            importance = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values()
            importance.rename('importance').to_csv(output / 'feature_importance.csv')
            figure = Figure(figsize=(9, 6))
            axis = figure.subplots()
            axis.barh(importance.index, importance.values)
            axis.set(xlabel='Impurity-based importance', title='Random Forest feature importance')
            figure.tight_layout()
            figure.savefig(output / 'feature_importance.png')
    comparison = pd.DataFrame(results)
    comparison.to_csv(output / 'model_comparison.csv', index=False)
    selected = max(results, key=lambda row: row['training_cv_f1'])['model']
    (models_path / 'selected_model.json').write_text(json.dumps({'model': selected,
        'selection': 'Highest macro F1 in training-only 3-fold cross-validation'}, indent=2))
    source = pd.read_csv(dataset_path)
    split = pd.DataFrame({'row_index': X.index, 'class': y,
                          'split': ['train' if i in X_train.index else 'test' for i in X.index]})
    if 'filename' in source.columns:
        split['filename'] = source['filename']
    split.to_csv(output / 'split.csv', index=False)
    print(comparison.to_string(index=False))
    print(f'Selected model: {selected} (training cross-validation)')
    return comparison


if __name__ == '__main__':
    train_and_evaluate()
