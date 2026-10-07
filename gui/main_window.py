"""Simple PyQt6 interface over the existing signal-analysis pipeline."""
from pathlib import Path
import sys

# Existing src scripts use direct imports; retain their established structure.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from config import PROJECT_ROOT
from feature_schema import CURRENT_FEATURE_SCHEMA, feature_names
from preprocessing import load_signal
from prediction import analyze_loaded_signal, classify_analysis, export_analysis

FEATURE_LABELS = {
    'mean': 'Mean', 'rms': 'RMS', 'max': 'Maximum', 'min': 'Minimum',
    'peak_to_peak': 'Peak-to-Peak', 'std': 'Standard Deviation',
    'dominant_frequency': 'Dominant Frequency', 'spectral_energy': 'Spectral Energy',
    'second_harmonic_ratio': 'Second Harmonic Ratio',
    'third_harmonic_ratio': 'Third Harmonic Ratio',
    'window_rms_variation': 'Window RMS Variation',
    'window_mean_variation': 'Window Mean Variation',
    'crossing_period_variation': 'Crossing Period Variation',
}
GUI_FEATURE_NAMES = feature_names(CURRENT_FEATURE_SCHEMA)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filename = None
        self.time = None
        self.signal = None
        self.analysis = None
        self.setWindowTitle('Signal Analysis and Classification System')
        self.resize(1080, 780)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        heading = QLabel('Signal Analysis and Classification System')
        heading.setStyleSheet('font-size: 20px; font-weight: bold;')
        layout.addWidget(heading)
        controls = QHBoxLayout()
        for label, callback in [('Load Signal', self.load_signal), ('Analyze', self.analyze),
                                ('Classify', self.classify), ('Export', self.export)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addStretch()
        controls.addWidget(QLabel('Model:'))
        self.model_selector = QComboBox()
        # None uses selected_model.json through the existing default loader.
        self.model_selector.addItem('Random Forest (default)', None)
        self.model_selector.addItem('SVM', PROJECT_ROOT / 'models/time_local/svm.pkl')
        self.model_selector.currentIndexChanged.connect(self.clear_classification)
        controls.addWidget(self.model_selector)
        layout.addLayout(controls)
        self.file_label = QLabel('No signal loaded')
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.tabs = QTabWidget()
        self.waveform_canvas, self.waveform_axis = self.add_plot_tab('Signal')
        self.spectrum_canvas, self.spectrum_axis = self.add_plot_tab('Spectrum')
        layout.addWidget(self.tabs, stretch=1)
        self.draw_waveform()
        self.draw_spectrum()

        bottom = QHBoxLayout()
        feature_group = QGroupBox('Signal Features')
        feature_layout = QVBoxLayout(feature_group)
        self.feature_table = QTableWidget(len(GUI_FEATURE_NAMES), 2)
        self.feature_table.setHorizontalHeaderLabels(['Feature', 'Value'])
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feature_table.setMinimumHeight(300)
        for row, name in enumerate(GUI_FEATURE_NAMES):
            self.feature_table.setItem(row, 0, QTableWidgetItem(FEATURE_LABELS[name]))
            self.feature_table.setItem(row, 1, QTableWidgetItem('—'))
        feature_layout.addWidget(self.feature_table)
        bottom.addWidget(feature_group, stretch=2)
        result_group = QGroupBox('Classification Result')
        result_layout = QVBoxLayout(result_group)
        result_layout.addWidget(QLabel('Predicted Class:'))
        self.result_label = QLabel('Not classified')
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet('font-size: 26px; font-weight: bold; color: #24577a;')
        result_layout.addWidget(self.result_label, stretch=1)
        self.result_model_label = QLabel('')
        result_layout.addWidget(self.result_model_label)
        bottom.addWidget(result_group, stretch=1)
        layout.addLayout(bottom)
        self.statusBar().showMessage('Load a CSV signal to begin.')

    def add_plot_tab(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        figure = Figure(figsize=(7, 3), layout='constrained')
        canvas = FigureCanvasQTAgg(figure)
        axis = figure.subplots()
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas)
        self.tabs.addTab(page, title)
        return canvas, axis

    def show_error(self, title, error):
        QMessageBox.warning(self, title, str(error))

    def require_signal(self):
        if self.signal is None:
            self.show_error('No signal loaded', 'Please load a signal first.')
            return False
        return True

    def clear_classification(self):
        if self.analysis is not None:
            self.analysis.pop('predicted_class', None)
            self.analysis.pop('model_name', None)
            self.analysis.pop('model_feature_schema_version', None)
        self.result_label.setText('Not classified')
        self.result_model_label.setText('')

    def load_signal(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load Signal', str(PROJECT_ROOT / 'data/raw'), 'CSV files (*.csv);;All files (*)')
        if not filename:
            return
        try:
            time, signal = load_signal(filename)
        except Exception as error:
            self.show_error('Invalid signal file',
                            f"Invalid signal file. The CSV must contain 'time' and 'amplitude' columns.\n\n{error}")
            return
        # Commit only a successful load; failed loads leave the previous signal intact.
        self.filename, self.time, self.signal = filename, time, signal
        self.analysis = None
        self.clear_classification()
        for row in range(len(GUI_FEATURE_NAMES)):
            self.feature_table.item(row, 1).setText('—')
        self.file_label.setText(f'Loaded: {filename} ({len(signal)} samples)')
        self.draw_waveform()
        self.draw_spectrum()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage('Signal loaded. Click Analyze or Classify.')

    def draw_waveform(self):
        axis = self.waveform_axis
        axis.clear()
        if self.signal is not None:
            axis.plot(self.time, self.signal, linewidth=1)
        axis.set(title='Signal Waveform', xlabel='Time (s)', ylabel='Amplitude')
        axis.grid(True, alpha=0.3)
        self.waveform_canvas.draw()

    def draw_spectrum(self):
        axis = self.spectrum_axis
        axis.clear()
        axis.set(title='Frequency Spectrum', xlabel='Frequency (Hz)', ylabel='Magnitude')
        axis.grid(True, alpha=0.3)
        if self.analysis is not None:
            # Visualization only: the original unwindowed rFFT convention.
            signal = self.analysis['signal']
            frequencies = np.fft.rfftfreq(len(signal), 1 / self.analysis['sampling_rate'])
            axis.plot(frequencies, np.abs(np.fft.rfft(signal)), linewidth=1)
        else:
            axis.text(0.5, 0.5, 'Click Analyze to view the spectrum.',
                      transform=axis.transAxes, ha='center', va='center')
        self.spectrum_canvas.draw()

    def analyze(self):
        if not self.require_signal():
            return False
        try:
            analysis = analyze_loaded_signal(self.time, self.signal, self.filename, CURRENT_FEATURE_SCHEMA)
        except Exception as error:
            self.show_error('Analysis failed', error)
            return False
        self.analysis = analysis
        self.clear_classification()
        for row, name in enumerate(GUI_FEATURE_NAMES):
            value = analysis['features'][name]
            text = f'{value:.4f}' if name != 'spectral_energy' else f'{value:.6g}'
            if name == 'dominant_frequency':
                text = f'{value:.2f} Hz'
            self.feature_table.item(row, 1).setText(text)
        self.draw_spectrum()
        self.statusBar().showMessage(f'Analysis complete. All {len(GUI_FEATURE_NAMES)} features are displayed.')
        return True

    def classify(self):
        if not self.require_signal():
            return
        if self.analysis is None and not self.analyze():
            return
        self.clear_classification()
        model_path = self.model_selector.currentData()
        try:
            # Check file presence for a clear GUI message; loading/prediction stays in src.
            required = ([PROJECT_ROOT / 'models/selected_model.json']
                        if model_path is None else [model_path])
            if any(not path.is_file() for path in required):
                raise FileNotFoundError('Trained model could not be found.')
            result = classify_analysis(self.analysis, model_path)
        except Exception as error:
            if isinstance(error, FileNotFoundError) or 'No such file or directory' in str(error):
                error = 'Trained model could not be found.'
            self.show_error('Classification failed', error)
            return
        self.analysis['predicted_class'] = result
        self.result_label.setText(result.upper())
        self.result_model_label.setText(f'Model: {self.model_selector.currentText()}')
        self.statusBar().showMessage(f'Predicted class: {result}')

    def export(self):
        if self.analysis is None:
            self.show_error('No analysis available', 'Please load and analyze or classify a signal first.')
            return
        default = Path(self.filename).with_name(Path(self.filename).stem + '_analysis.csv')
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Analysis', str(default), 'CSV files (*.csv)',
            options=QFileDialog.Option.DontConfirmOverwrite)
        if not filename:
            return
        path = Path(filename)
        if path.suffix.lower() != '.csv':
            path = path.with_suffix('.csv')
        if self.filename and path.resolve() == Path(self.filename).resolve():
            self.show_error('Export failed', 'Choose a different filename to preserve the loaded signal.')
            return
        # Also covers overwrite after adding the .csv extension.
        if path.exists():
            answer = QMessageBox.question(self, 'Confirm overwrite', f'{path.name} already exists. Replace it?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            export_analysis(self.analysis, path)
        except Exception as error:
            self.show_error('Export failed', error)
            return
        self.statusBar().showMessage(f'Analysis exported to {path}')
