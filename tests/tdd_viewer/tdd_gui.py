import sys
import os
import json
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QComboBox, 
                               QFileDialog, QSlider, QCheckBox, QSplitter, QFrame,
                               QDoubleSpinBox, QGroupBox)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pem_utilities.path_helper import get_repo_paths
class TddDataLoader:
    def __init__(self):
        self.metadata = {}
        self.datasets = {}
        self.current_data = None
        self.base_dir = ""
        self.unique_times = []

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        self.base_dir = os.path.dirname(json_path)
        # Reset state when loading new JSON
        self.datasets = {}
        self.current_data = None
        self.unique_times = []
        return self.metadata

    def load_tdd_file(self, filename):
        if filename in self.datasets:
            self.current_data = self.datasets[filename]
            return self.current_data

        filepath = os.path.join(self.base_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Read TDD file. 
        # Format: Scenario_Time(s), Tap#, Delay(nsec), Tap_Real, Tap_Imag
        names = ['Time', 'Tap_Index', 'Delay', 'Real', 'Imag']
        
        # Use pandas to read csv, skipping comment lines starting with '#'
        self.data = pd.read_csv(filepath, comment='#', names=names, header=None)
        
        # Calculate Magnitude and Phase
        # Magnitude = sqrt(Real^2 + Imag^2)
        self.data['Magnitude'] = np.sqrt(self.data['Real']**2 + self.data['Imag']**2)
        
        self.datasets[filename] = self.data
        self.current_data = self.data
        
        # Get unique time steps for the slider
        if len(self.unique_times) == 0:
            self.unique_times = np.sort(self.data['Time'].unique())
        
        return self.data

    def get_data_at_index(self, time_index, filename=None):
        if filename:
            data = self.datasets.get(filename)
        else:
            data = self.current_data
            
        if data is None or time_index >= len(self.unique_times):
            return None
        
        current_time = self.unique_times[time_index]
        subset = data[data['Time'] == current_time]
        return current_time, subset
    
    @property
    def data(self):
        return self.current_data
    
    @data.setter
    def data(self, value):
        self.current_data = value

class TddViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loader = TddDataLoader()
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.step_size = 1
        self.c = 0.299792458 # Speed of light in m/ns
        self.init_ui()
        
        # Try to load default file if it exists in the expected location relative to this script
        # The user mentioned path_to_tdd_file in tdd_viewer.py
        # We can try to find it.
        paths = get_repo_paths()

        default_path = os.path.join(paths.output,'TapDelayExports','tap_delay_output_summary.json')
        if os.path.exists(default_path):
            self.load_json_file(default_path)
        
    def init_ui(self):
        self.setWindowTitle("TDD Viewer")
        self.resize(1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Top Controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load JSON Summary")
        self.load_btn.clicked.connect(self.browse_json)
        controls_layout.addWidget(self.load_btn)
        
        controls_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.load_channel)
        self.channel_combo.setMinimumWidth(200)
        controls_layout.addWidget(self.channel_combo)
        
        self.db_check = QCheckBox("Plot in dB")
        self.db_check.setChecked(True)
        self.db_check.toggled.connect(self.update_plot)
        controls_layout.addWidget(self.db_check)
        
        controls_layout.addStretch()

        # Y-Axis Controls
        y_axis_group = QGroupBox("Y-Axis Limits")
        y_axis_layout = QHBoxLayout()
        y_axis_group.setLayout(y_axis_layout)
        
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        self.auto_scale_check.toggled.connect(self.toggle_y_axis_controls)
        self.auto_scale_check.toggled.connect(self.update_plot)
        y_axis_layout.addWidget(self.auto_scale_check)
        
        y_axis_layout.addWidget(QLabel("Min:"))
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-200, 200)
        self.y_min_spin.setValue(-100)
        self.y_min_spin.setEnabled(False)
        self.y_min_spin.valueChanged.connect(self.update_plot)
        y_axis_layout.addWidget(self.y_min_spin)
        
        y_axis_layout.addWidget(QLabel("Max:"))
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-200, 200)
        self.y_max_spin.setValue(0)
        self.y_max_spin.setEnabled(False)
        self.y_max_spin.valueChanged.connect(self.update_plot)
        y_axis_layout.addWidget(self.y_max_spin)
        
        controls_layout.addWidget(y_axis_group)

        # X-Axis Controls
        x_axis_group = QGroupBox("X-Axis Limits")
        x_axis_layout = QHBoxLayout()
        x_axis_group.setLayout(x_axis_layout)
        
        self.x_unit_combo = QComboBox()
        self.x_unit_combo.addItems(["Time (ns)", "Range (m)"])
        self.x_unit_combo.currentIndexChanged.connect(self.on_x_unit_changed)
        x_axis_layout.addWidget(self.x_unit_combo)

        self.x_auto_scale_check = QCheckBox("Auto Scale")
        self.x_auto_scale_check.setChecked(True)
        self.x_auto_scale_check.toggled.connect(self.toggle_x_axis_controls)
        self.x_auto_scale_check.toggled.connect(self.update_plot)
        x_axis_layout.addWidget(self.x_auto_scale_check)
        
        x_axis_layout.addWidget(QLabel("Min:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(0, 1000) # Adjust range as needed, maybe dynamically
        self.x_min_spin.setValue(0)
        self.x_min_spin.setEnabled(False)
        self.x_min_spin.valueChanged.connect(self.update_plot)
        x_axis_layout.addWidget(self.x_min_spin)
        
        x_axis_layout.addWidget(QLabel("Max:"))
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0, 1000) # Adjust range as needed
        self.x_max_spin.setValue(100)
        self.x_max_spin.setEnabled(False)
        self.x_max_spin.valueChanged.connect(self.update_plot)
        x_axis_layout.addWidget(self.x_max_spin)
        
        controls_layout.addWidget(x_axis_group)
        
        main_layout.addLayout(controls_layout)
        
        # Plot Area
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        
        # Bottom Controls (Time Slider)
        slider_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_animation)
        slider_layout.addWidget(self.play_btn)

        slider_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1, 1000.0)
        self.speed_spin.setSingleStep(10)
        self.speed_spin.setValue(10.0)
        self.speed_spin.setSuffix("x")
        self.speed_spin.valueChanged.connect(self.update_speed)
        slider_layout.addWidget(self.speed_spin)
        
        slider_layout.addWidget(QLabel("Time Step:"))
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.time_slider)
        
        self.time_label = QLabel("Time: 0.000000 s")
        self.time_label.setMinimumWidth(150)
        slider_layout.addWidget(self.time_label)
        
        main_layout.addLayout(slider_layout)
        
        # Info Label
        self.info_label = QLabel("Ready")
        main_layout.addWidget(self.info_label)

    def toggle_animation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.play_btn.setText("Pause")
            self.update_speed()

    def update_speed(self):
        speed = self.speed_spin.value()
        base_interval = 100  # Base interval for 1x speed (100ms)
        min_interval = 33    # Minimum interval (approx 30fps)
        
        target_interval = base_interval / speed
        
        if target_interval >= min_interval:
            interval = int(target_interval)
            self.step_size = 1
        else:
            interval = int(min_interval)
            # Calculate how many frames to skip to maintain effective speed
            # effective_speed = step_size * (1000 / min_interval)
            # target_speed = speed * (1000 / base_interval)
            # step_size = target_speed / (1000 / min_interval) = speed * (min_interval / base_interval)
            self.step_size = max(1, int(np.ceil(speed * (min_interval / base_interval))))
            
        if self.timer.isActive():
            self.timer.stop()
            self.timer.start(interval)
        elif self.play_btn.text() == "Pause": # If supposed to be playing
             self.timer.start(interval)

    def next_frame(self):
        current_val = self.time_slider.value()
        max_val = self.time_slider.maximum()
        
        next_val = current_val + self.step_size
        
        if next_val <= max_val:
            self.time_slider.setValue(next_val)
        else:
            self.time_slider.setValue(0)  # Loop back to start

    def toggle_y_axis_controls(self, checked):
        self.y_min_spin.setEnabled(not checked)
        self.y_max_spin.setEnabled(not checked)

    def toggle_x_axis_controls(self, checked):
        self.x_min_spin.setEnabled(not checked)
        self.x_max_spin.setEnabled(not checked)

    def on_x_unit_changed(self, index):
        # Update spinbox limits and values
        factor = self.c if index == 1 else 1.0/self.c
        
        self.x_min_spin.blockSignals(True)
        self.x_max_spin.blockSignals(True)
        
        # Convert current values
        self.x_min_spin.setValue(self.x_min_spin.value() * factor)
        self.x_max_spin.setValue(self.x_max_spin.value() * factor)
        
        # Convert ranges
        self.x_min_spin.setMaximum(self.x_min_spin.maximum() * factor)
        self.x_max_spin.setMaximum(self.x_max_spin.maximum() * factor)
        
        self.x_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(False)
        
        self.update_plot()

    def browse_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON Summary", "", "JSON Files (*.json)")
        if path:
            self.load_json_file(path)

    def load_json_file(self, path):
        try:
            metadata = self.loader.load_json(path)
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            if 'Exported Files' in metadata:
                items = ["All Channels"] + list(metadata['Exported Files'].keys())
                self.channel_combo.addItems(items)
            self.channel_combo.blockSignals(False)
            
            self.info_label.setText(f"Loaded {path}")
            
            # Load first channel automatically
            if self.channel_combo.count() > 1: # > 1 because "All Channels" is first
                self.channel_combo.setCurrentIndex(1) # Select first real channel
                self.load_channel()
                
        except Exception as e:
            self.info_label.setText(f"Error loading JSON: {str(e)}")

    def load_channel(self):
        channel_key = self.channel_combo.currentText()
        if not channel_key:
            return
            
        try:
            if channel_key == "All Channels":
                # Load all files
                for key, filename in self.loader.metadata['Exported Files'].items():
                    self.loader.load_tdd_file(filename)
                
                # Set slider range based on unique_times
                num_steps = len(self.loader.unique_times)
                self.time_slider.setRange(0, num_steps - 1)
                self.time_slider.setValue(0)
                
                # Update X-axis limits based on max delay across all datasets
                max_delay = 0
                for df in self.loader.datasets.values():
                    max_delay = max(max_delay, df['Delay'].max())
                
                # Apply unit conversion if needed
                if self.x_unit_combo.currentText() == "Range (m)":
                    max_val = max_delay * self.c
                else:
                    max_val = max_delay

                self.x_min_spin.setRange(0, max_val * 1.1)
                self.x_max_spin.setRange(0, max_val * 10.1)
                self.x_max_spin.setValue(max_val)
                
                self.info_label.setText(f"Loaded all channels ({num_steps} time steps)")
                self.update_plot()
            else:
                filename = self.loader.metadata['Exported Files'][channel_key]
                self.loader.load_tdd_file(filename)
                
                # Update slider range
                num_steps = len(self.loader.unique_times)
                self.time_slider.setRange(0, num_steps - 1)
                self.time_slider.setValue(0)
                
                # Update X-axis spinbox ranges based on data
                max_delay = self.loader.data['Delay'].max()
                
                # Apply unit conversion if needed
                if self.x_unit_combo.currentText() == "Range (m)":
                    max_val = max_delay * self.c
                else:
                    max_val = max_delay

                self.x_min_spin.setRange(0, max_val * 1.1)
                self.x_max_spin.setRange(0, max_val * 10.1)
                self.x_max_spin.setValue(max_val)
                
                self.info_label.setText(f"Loaded channel: {channel_key} ({num_steps} time steps)")
                self.update_plot()
            
        except Exception as e:
            self.info_label.setText(f"Error loading channel data: {str(e)}")

    def convert_to_db(self, x):
        """Convert magnitude to dB scale, avoiding log(0) issues."""
        return 20 * np.log10(np.fmax(np.abs(x), 1.e-10))

    def update_plot(self):
        channel_key = self.channel_combo.currentText()
        if self.loader.data is None and channel_key != "All Channels":
            return
            
        idx = self.time_slider.value()
        
        self.ax.clear()
        
        if self.db_check.isChecked():
            ylabel = 'Magnitude (dB)'
            bottom = -200 # Floor for dB plot
        else:
            ylabel = 'Magnitude (Linear)'
            bottom = 0
            
        is_range = (self.x_unit_combo.currentText() == "Range (m)")
        x_factor = self.c if is_range else 1.0
        xlabel = "Range (m)" if is_range else "Delay (ns)"

        current_time = 0
        
        if channel_key == "All Channels":
            # Plot all channels
            for i, (key, filename) in enumerate(self.loader.metadata['Exported Files'].items()):
                result = self.loader.get_data_at_index(idx, filename=filename)
                if result:
                    current_time, subset = result
                    delays = subset['Delay'] * x_factor
                    magnitudes = subset['Magnitude']
                    
                    if self.db_check.isChecked():
                        y_data = self.convert_to_db(magnitudes)
                        y_data = np.maximum(y_data, bottom)
                    else:
                        y_data = magnitudes
                    
                    # Use different colors for each channel
                    self.ax.stem(delays, y_data, bottom=bottom, 
                               linefmt=f'C{i}-', markerfmt=f'C{i}o', basefmt=' ',
                               label=key)
            
            self.ax.legend()
            
        else:
            # Plot single channel
            result = self.loader.get_data_at_index(idx)
            if result is None:
                return
                
            current_time, subset = result
            delays = subset['Delay'] * x_factor
            magnitudes = subset['Magnitude']
            
            if self.db_check.isChecked():
                y_data = self.convert_to_db(magnitudes)
                y_data = np.maximum(y_data, bottom)
            else:
                y_data = magnitudes
                
            # Stem plot
            markerline, stemlines, baseline = self.ax.stem(delays, y_data, bottom=bottom, 
                                                         linefmt='-', markerfmt='ro', basefmt='k-')
        
        self.time_label.setText(f"Time: {current_time:.6f} s")
        
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f"Impulse Response at t={current_time:.6f}s")
        self.ax.grid(True, alpha=0.3)
        
        if not self.auto_scale_check.isChecked():
            self.ax.set_ylim(self.y_min_spin.value(), self.y_max_spin.value())
        elif self.db_check.isChecked():
            self.ax.set_ylim(bottom=bottom)
            
        if not self.x_auto_scale_check.isChecked():
            self.ax.set_xlim(self.x_min_spin.value(), self.x_max_spin.value())
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TddViewer()
    window.show()
    sys.exit(app.exec())
