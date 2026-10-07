"""Scripted Qt smoke check; run separately with a display or QT_QPA_PLATFORM=offscreen."""
import sys, tempfile, json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QPushButton, QMessageBox
from gui.main_window import MainWindow
from prediction import predict_signal
from config import CLASS_NAMES, FEATURE_NAMES, PROJECT_ROOT
from feature_schema import CURRENT_FEATURE_SCHEMA, feature_names
import pandas as pd

app = QApplication(sys.argv)
window = MainWindow()
window.show()


def checks():
    report = {'platform': app.platformName(), 'application_starts': True,
              'feature_schema_version': CURRENT_FEATURE_SCHEMA,
              'feature_count': len(feature_names(CURRENT_FEATURE_SCHEMA)),
              'verification_method': 'Scripted Qt event-loop checks; file-dialog responses supplied by test'}
    try:
        buttons = {button.text(): button for button in window.findChildren(QPushButton)}
        with tempfile.TemporaryDirectory() as folder, patch('gui.main_window.QMessageBox.warning') as warning:
            for name in ['Analyze', 'Classify', 'Export']:
                buttons[name].click()
            assert warning.call_count == 3
            report['no_signal_errors'] = 'passed'
            warning.reset_mock()
            predictions = []
            for class_name in CLASS_NAMES:
                filename = PROJECT_ROOT / 'data/raw' / class_name / f'{class_name}_001.csv'
                with patch('gui.main_window.QFileDialog.getOpenFileName', return_value=(str(filename), 'CSV')):
                    buttons['Load Signal'].click()
                assert window.filename == str(filename)
                assert window.analysis is None and window.result_label.text() == 'Not classified'
                assert len(window.waveform_axis.lines[0].get_xdata()) == 500
                buttons['Analyze'].click()
                expected_features = feature_names(CURRENT_FEATURE_SCHEMA)
                assert window.feature_table.rowCount() == len(expected_features)
                assert all(window.feature_table.item(i, 1).text() != '—' for i in range(len(expected_features)))
                assert len(window.spectrum_axis.lines[0].get_xdata()) == 251
                window.tabs.setCurrentIndex(1); app.processEvents()
                for index in [0, 1]:
                    window.model_selector.setCurrentIndex(index)
                    buttons['Classify'].click()
                    expected = predict_signal(filename, window.model_selector.currentData())['predicted_class']
                    assert window.result_label.text() == expected.upper()
                    predictions.append(dict(signal=class_name, model=window.model_selector.currentText(), predicted=expected))
                exported = Path(folder) / f'{class_name}_result.csv'
                with patch('gui.main_window.QFileDialog.getSaveFileName', return_value=(str(exported), 'CSV')):
                    buttons['Export'].click()
                saved = pd.read_csv(exported)
                assert set(expected_features).issubset(saved.columns)
                assert saved['feature_schema_version'].iloc[0] == CURRENT_FEATURE_SCHEMA
                assert saved['model_feature_schema_version'].iloc[0] == CURRENT_FEATURE_SCHEMA
                assert saved['predicted_class'].iloc[0] == window.analysis['predicted_class']
                original = exported.read_bytes()
                with patch('gui.main_window.QFileDialog.getSaveFileName', return_value=(str(exported), 'CSV')), \
                     patch('gui.main_window.QMessageBox.question', return_value=QMessageBox.StandardButton.No):
                    buttons['Export'].click()
                assert exported.read_bytes() == original
            assert warning.call_count == 0
            valid_filename = window.filename
            for content in ['time,value\n0,1\n', 'time,amplitude\n', 'time,amplitude\n0,NaN\n']:
                bad = Path(folder) / 'bad.csv'; bad.write_text(content)
                with patch('gui.main_window.QFileDialog.getOpenFileName', return_value=(str(bad), 'CSV')):
                    buttons['Load Signal'].click()
                assert window.filename == valid_filename
            assert warning.call_count == 3
            with patch('gui.main_window.classify_analysis', side_effect=FileNotFoundError('missing model')):
                buttons['Classify'].click()
            assert warning.call_count == 4
            assert 'Trained model could not be found' in str(warning.call_args)
            report.update(signal_loading='passed', waveform='passed', feature_analysis='passed',
                          frequency_spectrum='passed', classification='passed', export='passed',
                          invalid_input='passed', overwrite_protection='passed', predictions=predictions)
        window.model_selector.setCurrentIndex(0)
        buttons['Classify'].click()
        app.processEvents()
        window.grab().save(str(PROJECT_ROOT / 'data/dataset_analysis/gui_verification.png'))
        report['status'] = 'passed'
        (PROJECT_ROOT / 'data/dataset_analysis/gui_verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        app.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        app.exit(1)

QTimer.singleShot(0, checks)
sys.exit(app.exec())
