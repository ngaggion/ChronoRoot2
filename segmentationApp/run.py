import os
import sys

# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import tempfile
import subprocess
import json
import shutil
import glob
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QFileDialog, QCheckBox,
                             QMessageBox, QTextEdit, QComboBox, QInputDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from datetime import datetime
from pathlib import Path

class SegmentationWorker(QThread):
    """Worker thread for running segmentation using nnUNet wrapper"""
    finished = pyqtSignal(str, str)  # folder_path, message
    error = pyqtSignal(str, str)     # folder_path, error_message
    progress = pyqtSignal(str, str)  # folder_path, status_update

    def __init__(self, input_path, robot_name, species="arabidopsis", fast_mode=False, conda_env="base"):
        super().__init__()
        self.input_path = Path(input_path)
        self.robot_name = robot_name
        self.species = species
        self.fast_mode = fast_mode
        self.conda_env = conda_env
        self.script_dir = Path(__file__).parent.resolve()
        self.seg_folder = self.input_path / 'Segmentation'
        self.info_file = self.seg_folder / 'segmentation_info.json'

    def _update_info_file(self, data_to_update):
        """Helper to read, update, and write the JSON info file."""
        try:
            os.makedirs(self.seg_folder, exist_ok=True)
            info_data = {}
            if self.info_file.exists():
                with open(self.info_file, 'r') as f:
                    info_data = json.load(f)
            
            info_data.update(data_to_update)
            
            with open(self.info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
        except Exception as e:
            self.progress.emit(str(self.input_path), f"Warning: Could not write info file: {e}")

    def run(self):
        try:
            folder_name = self.input_path.name
            self.progress.emit(str(self.input_path), "Starting segmentation...")

            self._update_info_file({
                'robot_name': self.robot_name,
                'conda_env': self.conda_env,
                'species': self.species,
                'fast_mode': self.fast_mode,
                'segmentation_start_time': datetime.now().isoformat(),
                'folder_path': str(self.input_path),
                'segmentation_status': 'started'
            })

            model_name = "Arabidopsis" if self.species == "arabidopsis" else "Tomato"
            model_path = self.script_dir / "models" / model_name

            if not model_path.exists():
                self.error.emit(str(self.input_path), f"Model not found at: {model_path}")
                return

            output_path = self.seg_folder / "Fold_0"
            os.makedirs(output_path, exist_ok=True)

            model_args = {
                'model_path': str(model_path),
                'device': 'cuda',
                'verbose': False,
                'use_gaussian': True,
                'use_mirroring': not self.fast_mode,
                'tile_step_size': 0.5
            }
            
            # Using repr() ensures all objects (paths, dict) are
            # serialized as valid Python code literals.
            wrapper_script = f"""
import sys
import os
sys.path.append({repr(str(self.script_dir))})
from nnUNet_wrapper import nnUNetv2

model_args = {repr(model_args)}
model = nnUNetv2(**model_args)

results = model.predict_from_folder(
    input_dir={repr(str(self.input_path))},
    output_dir={repr(str(output_path))},
    save_as_png=True
)
print(f"Segmentation completed: {{len(results)}} images processed")
            """
            temp_script_path = None
            try:
                # Use tempfile for a clean, safe temporary script
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(wrapper_script)
                    temp_script_path = f.name
                
                conda_prefix = f"conda run -n {self.conda_env}"
                # Quote the script path to handle spaces
                cmd = f'{conda_prefix} python "{temp_script_path}"'

                self.progress.emit(str(self.input_path), "Running nnUNet...")
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                
                self.progress.emit(str(self.input_path), f"Command output: {stdout[:200]}...")

                if process.returncode == 0:
                    seg_files = len(list(output_path.glob('*.png')))
                    if seg_files > 0:
                        self._update_info_file({
                            'segmentation_status': 'completed',
                            'segmentation_completion_time': datetime.now().isoformat()
                        })
                        mode_str = " (fast mode)" if self.fast_mode else ""
                        self.finished.emit(str(self.input_path),
                                         f"Segmentation completed for {folder_name} ({self.species}{mode_str}): {seg_files} files")
                    else:
                        self.error.emit(str(self.input_path), f"Segmentation failed for {folder_name}: No output files generated")
                else:
                    error_msg = f"Segmentation failed for {folder_name} (return code {process.returncode}): {stderr[:300]}"
                    self.error.emit(str(self.input_path), error_msg)

            finally:
                # Ensure temp file is always cleaned up
                if temp_script_path and os.path.exists(temp_script_path):
                    os.remove(temp_script_path)
                    
        except Exception as e:
            self.error.emit(str(self.input_path), f"Error running segmentation: {str(e)}")
    
class PostprocessWorker(QThread):
    """Worker thread for running postprocessing"""
    finished = pyqtSignal(str, str)  # folder_path, message
    error = pyqtSignal(str, str)     # folder_path, error_message
    progress = pyqtSignal(str, str)  # folder_path, status_update

    def __init__(self, input_path, robot_name, species="arabidopsis", alpha_parameter=0.9, conda_env="base"):
        super().__init__()
        self.input_path = Path(input_path)
        self.robot_name = robot_name
        self.species = species
        self.alpha_parameter = alpha_parameter
        self.conda_env = conda_env
        self.seg_folder = self.input_path / 'Segmentation'
        self.info_file = self.seg_folder / 'postprocess_info.json'

    def _update_info_file(self, data_to_update):
        """Helper to read, update, and write the JSON info file."""
        try:
            os.makedirs(self.seg_folder, exist_ok=True)
            info_data = {}
            if self.info_file.exists():
                with open(self.info_file, 'r') as f:
                    info_data = json.load(f)
            
            info_data.update(data_to_update)
            
            with open(self.info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
        except Exception as e:
            self.progress.emit(str(self.input_path), f"Warning: Could not write info file: {e}")

    def clean_postprocess_folders(self):
        """Clean existing postprocess folders"""
        try:
            for folder in ['Ensemble', 'Ensemble_color']:
                folder_path = self.seg_folder / folder
                if folder_path.exists():
                    shutil.rmtree(folder_path)
        except Exception as e:
            pass

    def run(self):
        try:
            folder_name = self.input_path.name
            self.progress.emit(str(self.input_path), "Starting postprocessing...")

            fold_0_path = self.seg_folder / 'Fold_0'
            if not fold_0_path.exists():
                self.error.emit(str(self.input_path), f"Cannot postprocess {folder_name}: No segmentation folder found")
                return

            seg_files = len(list(fold_0_path.glob('*.png')))
            if seg_files == 0:
                self.error.emit(str(self.input_path), f"Cannot postprocess {folder_name}: No segmentation files found")
                return

            self.clean_postprocess_folders()
            
            self._update_info_file({
                'robot_name': self.robot_name,
                'species': self.species,
                'alpha_parameter': self.alpha_parameter,
                'postprocess_start_time': datetime.now().isoformat(),
                'folder_path': str(self.input_path),
                'postprocess_status': 'started'
            })

            script_dir = Path(__file__).parent.resolve()
            postprocess_path = script_dir / "postprocess.py"
            
            alpha = self.alpha_parameter
            if not alpha:
                alpha = 0.85 if self.species == "arabidopsis" else 0.99
            
            conda_prefix = f"conda run -n {self.conda_env}"
            
            # This command is correct. It runs an existing script,
            # so it doesn't need a temp file.
            cmd = f'{conda_prefix} python "{postprocess_path}" "{self.input_path}" --method {self.species} --alpha {alpha} --seg_path Segmentation'

            self.progress.emit(str(self.input_path), "Running postprocessing...")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                ensemble_path = self.seg_folder / 'Ensemble'
                output_files = len(list(ensemble_path.glob('*.png'))) if ensemble_path.exists() else 0
                
                if output_files > 0:
                    self._update_info_file({
                        'postprocess_status': 'completed',
                        'postprocess_completion_time': datetime.now().isoformat()
                    })
                    self.finished.emit(str(self.input_path),
                                     f"Postprocessing completed for {folder_name} ({self.species}, α={alpha}): {output_files} files")
                else:
                    self.error.emit(str(self.input_path), f"Postprocessing failed for {folder_name}: No output files generated")
            else:
                error_msg = f"Postprocessing failed for {folder_name}: {stderr[:300]}"
                self.error.emit(str(self.input_path), error_msg)

        except Exception as e:
            self.error.emit(str(self.input_path), f"Error running postprocessing: {str(e)}")

class nnUNetMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robots = {}  # robot_name -> {path: str, folders: dict}
        self.folder_data = {}  # folder_path -> data dict
        self.processing_queue = []  # Simple list for serial processing
        self.alpha_parameter = 0.9  # Default alpha value
        self.conda_env = "base"  # Default conda environment
        self.fast_mode = False  # Default fast mode setting
        self.species = "arabidopsis"  # Default species
        
        self.current_segmentation_worker = None
        self.current_postprocess_worker = None
        
        self.init_ui()
        self.load_settings()
        self.setup_timer()
        
    def init_ui(self):
        self.setWindowTitle("nnUNet Segmentation Monitor")
        self.setGeometry(100, 100, 1200, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Load robot button
        self.load_button = QPushButton("Load Robot")
        self.load_button.clicked.connect(self.load_robot)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # Status labels
        self.robot_count_label = QLabel("Robots: 0")
        self.queue_info_label = QLabel("Queue: 0 | Processing: None")
        
        # Alpha parameter button (updated with loaded value)
        self.alpha_button = QPushButton(f"Alpha: {self.alpha_parameter}")
        self.alpha_button.clicked.connect(self.set_alpha_parameter)
        self.alpha_button.setStyleSheet("background-color: #2196F3; color: white;")
        
        # Conda environment button (updated with loaded value)
        self.conda_button = QPushButton(f"Conda: {self.conda_env}")
        self.conda_button.clicked.connect(self.set_conda_env)
        self.conda_button.setStyleSheet("background-color: #4CAF50; color: white;")
        
        # Fast mode checkbox (replaces nnUNet path)
        self.fast_mode_checkbox = QCheckBox("Fast Mode")
        self.fast_mode_checkbox.setChecked(self.fast_mode)
        self.fast_mode_checkbox.setToolTip("Disable test-time augmentation for faster processing")
        self.fast_mode_checkbox.stateChanged.connect(self.update_fast_mode)
        layout.addWidget(self.fast_mode_checkbox)
        
        # Species selection
        layout.addWidget(QLabel("Species:"))
        self.species_combo = QComboBox()
        self.species_combo.addItems(["arabidopsis", "tomato"])
        self.species_combo.setCurrentText(self.species)
        self.species_combo.currentTextChanged.connect(self.update_species)
        layout.addWidget(self.species_combo)
        
        # Queue control button
        self.clear_queue_button = QPushButton("Clear Queue")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        self.clear_queue_button.setStyleSheet("background-color: #f44336; color: white;")
        
        # Manual refresh button
        self.manual_refresh_button = QPushButton("Refresh Now")
        self.manual_refresh_button.clicked.connect(self.manual_refresh)
        self.manual_refresh_button.setStyleSheet("background-color: #9C27B0; color: white;")
        
        # Auto-refresh toggle
        self.auto_refresh_button = QPushButton("Auto-Refresh: ON")
        self.auto_refresh_button.clicked.connect(self.toggle_auto_refresh)
        self.auto_refresh_enabled = True
        
        header_layout.addWidget(self.load_button)
        header_layout.addWidget(self.robot_count_label)
        header_layout.addWidget(QLabel("|"))
        header_layout.addWidget(self.queue_info_label)
        header_layout.addWidget(self.alpha_button)
        header_layout.addWidget(self.species_combo)
        header_layout.addWidget(self.fast_mode_checkbox)
        header_layout.addWidget(self.clear_queue_button)
        header_layout.addStretch()
        header_layout.addWidget(self.manual_refresh_button)
        header_layout.addWidget(self.auto_refresh_button)
        header_layout.addWidget(self.conda_button)
        
        layout.addLayout(header_layout)
        
        # Filter section
        filter_layout = QHBoxLayout()
        
        self.robot_filter = QComboBox()
        self.robot_filter.addItem("All Robots")
        self.robot_filter.currentTextChanged.connect(self.update_table)
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Not Started", "Queued", "Segmenting", 
                                    "Segmented", "Postprocessing", "Complete", "Error"])
        self.status_filter.currentTextChanged.connect(self.update_table)
        
        filter_layout.addWidget(QLabel("Filter by Robot:"))
        filter_layout.addWidget(self.robot_filter)
        filter_layout.addWidget(QLabel("Filter by Status:"))
        filter_layout.addWidget(self.status_filter)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Table for folder information
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Robot", "Folder Name", "Total Images", "Segmentation %", 
            "Postprocess %", "Status", "Alpha Used", "Actions", "Delete"
        ])
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 120)
        
        layout.addWidget(self.table)
        
        # Log section
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("background-color: #f0f0f0; font-family: monospace; font-size: 10px;")
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_text)
        
    def setup_timer(self):
        """Setup single timer for auto-refresh and queue processing every 0.5 seconds"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_update)
        self.timer.start(500)  # 0.5 seconds
        
    def timer_update(self):
        """Handle both refresh and queue processing"""
        if self.auto_refresh_enabled:
            self.refresh_data()
        self.process_queue()
        
    def manual_refresh(self):
        """Manual refresh of all data"""
        self.refresh_data()
        self.log_message("Manual refresh completed")
        
    def set_alpha_parameter(self):
        """Set the alpha parameter for postprocessing"""
        value, ok = QInputDialog.getDouble(self, 'Alpha Parameter', 
                                         'Enter alpha value for postprocessing (0.0-1.0):', 
                                         self.alpha_parameter, 0.0, 1.0, 2)
        if ok:
            self.alpha_parameter = value
            self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
            self.save_settings()
            self.log_message(f"Alpha parameter set to {self.alpha_parameter}")
    
    def set_conda_env(self):
        """Set the conda environment name"""
        env_name, ok = QInputDialog.getText(self, 'Conda Environment', 
                                          'Enter conda environment name:', 
                                          text=self.conda_env)
        if ok and env_name.strip():
            self.conda_env = env_name.strip()
            self.conda_button.setText(f"Conda: {self.conda_env}")
            self.save_settings()
            self.log_message(f"Conda environment set to '{self.conda_env}'")
    
    def update_species(self, text):
        """Update the species setting"""
        self.species = text
        # Update alpha default based on species
        if self.species == "arabidopsis":
            self.alpha_parameter = 0.85
            self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
        else:  # tomato
            self.alpha_parameter = 0.99
            self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
        self.save_settings()
        self.log_message(f"Species set to: {self.species}")
    
    def update_fast_mode(self, state):
        """Update fast mode setting"""
        self.fast_mode = (state == Qt.Checked)
        mode_str = "enabled" if self.fast_mode else "disabled"
        self.save_settings()
        self.log_message(f"Fast mode {mode_str}")
        
    def save_settings(self):
        """Save configuration to file"""
        try:
            config = {
                'conda_env': self.conda_env,
                'alpha': self.alpha_parameter,
                'species': self.species,
                'fast_mode': self.fast_mode
            }
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.log_message(f"Failed to save config: {str(e)}")

    def load_settings(self):
        """Load configuration from file"""
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.conda_env = config.get('conda_env', 'base')
                self.species = config.get('species', 'arabidopsis')
                self.alpha_parameter = config.get('alpha', 0.85)
                self.fast_mode = config.get('fast_mode', False)
                
                # Update UI elements
                self.conda_button.setText(f"Conda: {self.conda_env}")
                self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
                self.species_combo.setCurrentText(self.species)
                self.fast_mode_checkbox.setChecked(self.fast_mode)
                
                # Scan loaded robots
                for robot_name in self.robots.keys():
                    self.scan_robot_folders(robot_name)
                    self.log_message(f"Loaded robot: {robot_name}")
            
    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off"""
        self.auto_refresh_enabled = not self.auto_refresh_enabled
        if self.auto_refresh_enabled:
            self.auto_refresh_button.setText("Auto-Refresh: ON")
            self.auto_refresh_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.log_message("Auto-refresh enabled")
        else:
            self.auto_refresh_button.setText("Auto-Refresh: OFF")
            self.auto_refresh_button.setStyleSheet("background-color: #f44336; color: white;")
            self.log_message("Auto-refresh disabled")
    
    def load_segmentation_info(self, folder_path):
        """Load segmentation info from file"""
        try:
            info_file = os.path.join(folder_path, 'Segmentation', 'segmentation_info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None

    def load_postprocess_info(self, folder_path):
        """Load postprocessing info from file"""
        try:
            info_file = os.path.join(folder_path, 'Segmentation', 'postprocess_info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return None
    
    def needs_re_postprocess(self, folder_path, current_status):
        """Check if folder needs re-postprocessing due to alpha change"""
        if current_status != 'Complete':
            return False
            
        info = self.load_postprocess_info(folder_path)
        if info is None:
            return True  # No info file, might need re-processing

        stored_alpha = info.get('alpha_parameter', 0.9)
        # Handle case where stored_alpha might be string
        try:
            stored_alpha_float = float(stored_alpha)
            return abs(stored_alpha_float - self.alpha_parameter) > 0.001
        except (ValueError, TypeError):
            return True # If stored value is invalid, recommend re-processing
    
    def delete_segmentation_folder(self, folder_path):
        """Delete the segmentation folder completely"""
        try:
            seg_folder = os.path.join(folder_path, 'Segmentation')
            if os.path.exists(seg_folder):
                shutil.rmtree(seg_folder)
                self.log_message(f"Deleted segmentation folder for {os.path.basename(folder_path)}")
                return True
        except Exception as e:
            self.log_message(f"Error deleting segmentation folder: {str(e)}")
            QMessageBox.warning(self, "Delete Error", f"Could not delete segmentation folder: {str(e)}")
            return False
        return True
    
    def load_robot(self):
        """Load a new robot folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Robot Folder")
        if folder_path:
            robot_name, ok = QInputDialog.getText(self, 'Robot Name', 
                                                f'Enter name for robot (default: {os.path.basename(folder_path)}):')
            if not ok:
                return
            if not robot_name:
                robot_name = os.path.basename(folder_path)
            
            # Check if robot already exists
            if robot_name in self.robots:
                reply = QMessageBox.question(self, 'Robot Exists', 
                                           f'Robot "{robot_name}" already exists. Replace?',
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.No:
                    return
            
            self.robots[robot_name] = {'path': folder_path, 'folders': {}}
            self.scan_robot_folders(robot_name)
            self.update_robot_filter()
            self.log_message(f"Loaded robot: {robot_name} from {folder_path}")
    
    def scan_robot_folders(self, robot_name):
        """Scan for folders containing images in a specific robot"""
        if robot_name not in self.robots:
            return
            
        robot_path = self.robots[robot_name]['path']
        
        try:
            folders = [f for f in os.listdir(robot_path) 
                      if os.path.isdir(os.path.join(robot_path, f))]
            
            self.robots[robot_name]['folders'] = {}
            for folder in folders:
                folder_path = os.path.join(robot_path, folder)
                full_folder_key = f"{robot_name}::{folder_path}"
                folder_data = self.analyze_folder(folder_path, robot_name)
                self.robots[robot_name]['folders'][folder] = folder_data
                self.folder_data[full_folder_key] = folder_data
            
            self.update_table()
            
        except Exception as e:
            self.log_message(f"Error scanning robot {robot_name}: {str(e)}")
    
    def analyze_folder(self, folder_path, robot_name):
        """Analyze a folder to get image count and segmentation progress"""
        data = {
            'path': folder_path,
            'robot': robot_name,
            'total_images': 0,
            'segmentation_progress': 0,
            'postprocess_progress': 0, 
            'status': 'Not Started',
            'last_error': '',
            'stored_alpha': None,
            'needs_re_postprocess': False, 
            'is_actively_processing': False
        }
        
        try:
            # Count total images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            total_images = 0
            for ext in image_extensions:
                total_images += len(glob.glob(os.path.join(folder_path, ext)))
                total_images += len(glob.glob(os.path.join(folder_path, ext.upper())))
            
            data['total_images'] = total_images
            
            if total_images == 0:
                data['status'] = 'No Images'
                return data
            
            # Check if in queue
            if any(item['path'] == folder_path for item in self.processing_queue):
                data['status'] = 'Queued'
                return data
            
            # Check if currently processing
            is_currently_segmenting = self.current_segmentation_worker and self.current_segmentation_worker.input_path == Path(folder_path)
            is_currently_postprocessing = self.current_postprocess_worker and self.current_postprocess_worker.input_path == Path(folder_path)
            data['is_actively_processing'] = is_currently_segmenting or is_currently_postprocessing
            
            seg_info = self.load_segmentation_info(folder_path)
            post_info = self.load_postprocess_info(folder_path)
            
            if post_info:
                data['stored_alpha'] = post_info.get('alpha_parameter', 'Unknown')
            
            # Check segmentation progress
            seg_folder = os.path.join(folder_path, 'Segmentation')
            if os.path.exists(seg_folder):
                # Check for error files (seg or postprocess)
                error_files = (glob.glob(os.path.join(seg_folder, '*.error')) + 
                               glob.glob(os.path.join(seg_folder, '*.error')))
                if error_files:
                    error_content = ""
                    try:
                        with open(error_files[0], 'r') as f:
                            error_content = f.read()
                    except:
                        pass
                    
                    data['status'] = 'Error'
                    data['last_error'] = error_content[:100]
                    return data
                
                # Check if segmentation is officially completed
                segmentation_officially_complete = (seg_info and 
                                                  seg_info.get('segmentation_status') == 'completed')
                
                # Check 'Ensemble' folder (output of postprocess worker)
                # Note: The folder name is still "Ensemble" as per PostprocessWorker
                ensemble_folder = os.path.join(seg_folder, 'Ensemble')
                if os.path.exists(ensemble_folder):
                    postprocess_files = 0
                    for ext in ['*.nii.gz', '*.nii', '*.npz', '*.png']:
                        postprocess_files += len(glob.glob(os.path.join(ensemble_folder, ext)))
                    
                    if postprocess_files > 0:
                        data['postprocess_progress'] = min(100, int((postprocess_files / total_images) * 100))
                        
                        postprocess_officially_complete = (post_info and 
                                                         post_info.get('postprocess_status') == 'completed')
                        
                        if data['postprocess_progress'] >= 90 and postprocess_officially_complete:
                            data['segmentation_progress'] = 100
                            data['status'] = 'Complete'
                            data['needs_re_postprocess'] = self.needs_re_postprocess(folder_path, 'Complete')
                            return data
                        else:
                            data['segmentation_progress'] = 100 if segmentation_officially_complete else min(100, data['postprocess_progress'])
                            data['status'] = 'Postprocessing' if is_currently_postprocessing else 'Processing'
                
                # Check Fold_0 for segmentation files
                fold_0_path = os.path.join(seg_folder, 'Fold_0')
                if os.path.exists(fold_0_path):
                    seg_files = 0
                    for ext in ['*.nii.gz', '*.nii', '*.npz', '*.png']:
                        seg_files += len(glob.glob(os.path.join(fold_0_path, ext)))
                    
                    if seg_files > 0:
                        data['segmentation_progress'] = min(100, int((seg_files / total_images) * 100))
                        
                        # Only mark as "Segmented" if seg is complete and postprocess hasn't started
                        if segmentation_officially_complete and data['postprocess_progress'] == 0:
                            data['status'] = 'Segmented'
                        elif data['segmentation_progress'] > 0:
                            data['status'] = 'Segmenting' if is_currently_segmenting else 'Processing'
                
                # If seg folder exists but no clear progress
                if data['segmentation_progress'] == 0 and data['postprocess_progress'] == 0:
                    if is_currently_segmenting:
                        data['status'] = 'Segmenting'
                    elif is_currently_postprocessing: 
                        data['status'] = 'Postprocessing' 
                    else:
                        data['status'] = 'Processing'
            
            # Override status if currently processing
            if is_currently_segmenting:
                data['status'] = 'Segmenting'
            elif is_currently_postprocessing: 
                data['status'] = 'Postprocessing' 
        
        except Exception as e:
            data['status'] = 'Error'
            data['last_error'] = str(e)[:100]
        
        return data
    
    def update_robot_filter(self):
        """Update robot filter dropdown"""
        current_selection = self.robot_filter.currentText()
        self.robot_filter.clear()
        self.robot_filter.addItem("All Robots")
        for robot_name in self.robots.keys():
            self.robot_filter.addItem(robot_name)
        
        # Restore selection
        index = self.robot_filter.findText(current_selection)
        if index >= 0:
            self.robot_filter.setCurrentIndex(index)
        
        self.robot_count_label.setText(f"Robots: {len(self.robots)}")
    
    def update_table(self):
        """Update the table with current folder data"""
        robot_filter = self.robot_filter.currentText()
        status_filter = self.status_filter.currentText()
        
        filtered_data = {}
        for key, data in self.folder_data.items():
            if robot_filter != "All Robots" and data['robot'] != robot_filter:
                continue
            if status_filter != "All Status" and data['status'] != status_filter:
                continue
            filtered_data[key] = data
        
        self.table.setRowCount(len(filtered_data))
        
        for row, (folder_key, data) in enumerate(filtered_data.items()):
            folder_path = data['path']
            folder_name = os.path.basename(folder_path)
            
            self.table.setItem(row, 0, QTableWidgetItem(data['robot']))
            self.table.setItem(row, 1, QTableWidgetItem(folder_name))
            self.table.setItem(row, 2, QTableWidgetItem(str(data['total_images'])))
            
            seg_progress = QProgressBar()
            seg_progress.setValue(data['segmentation_progress'])
            seg_progress.setFormat(f"{data['segmentation_progress']}%")
            self.table.setCellWidget(row, 3, seg_progress)
            
            post_progress = QProgressBar()
            post_progress.setValue(data['postprocess_progress'])
            post_progress.setFormat(f"{data['postprocess_progress']}%")
            self.table.setCellWidget(row, 4, post_progress)
            
            # Status with colors
            status_item = QTableWidgetItem(data['status'])
            if data['status'] == 'Complete':
                status_item.setBackground(QColor(144, 238, 144))  # Light green
            elif data['status'] == 'Segmented':
                status_item.setBackground(QColor(255, 255, 0))    # Yellow
            elif data['status'] in ['Segmenting', 'Postprocessing']:
                status_item.setBackground(QColor(173, 216, 230))  # Light blue
            elif data['status'] == 'Queued':
                status_item.setBackground(QColor(255, 255, 0))    # Yellow
            elif data['status'] == 'Error':
                status_item.setBackground(QColor(255, 182, 193))  # Light red
            elif data['status'] == 'Not Started':
                status_item.setBackground(QColor(211, 211, 211))  # Light gray
            
            if data['last_error']:
                status_item.setToolTip(data['last_error'])
            self.table.setItem(row, 5, status_item)
            
            # Alpha used
            alpha_text = str(data.get('stored_alpha', '—'))
            alpha_item = QTableWidgetItem(alpha_text)
            if data.get('needs_re_postprocess', False):
                alpha_item.setBackground(QColor(255, 255, 0))  # Yellow
                alpha_item.setToolTip(f"Current alpha: {self.alpha_parameter}, Stored alpha: {alpha_text}")
            self.table.setItem(row, 6, alpha_item)
            
            # Action button
            if data['status'] in ['Not Started', 'Error']:
                action_button = QPushButton("Add to Queue")
                action_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot, 'both'))
                self.table.setCellWidget(row, 7, action_button)
            elif data['status'] == 'Segmented':
                postprocess_button = QPushButton("Postprocess")
                postprocess_button.setStyleSheet("background-color: #ff9800; color: white;")
                postprocess_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot, 'postprocess'))
                self.table.setCellWidget(row, 7, postprocess_button)
            elif data['status'] == 'Complete' and data.get('needs_re_postprocess', False):
                re_process_button = QPushButton("Re-process")
                re_process_button.setStyleSheet("background-color: #ff9800; color: white;")
                re_process_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot, 'postprocess'))
                self.table.setCellWidget(row, 7, re_process_button)
            elif data['status'] == 'Queued':
                remove_button = QPushButton("Remove")
                remove_button.clicked.connect(lambda checked, path=folder_path: self.remove_from_queue(path))
                self.table.setCellWidget(row, 7, remove_button)
            elif data['status'] in ['Segmenting', 'Postprocessing']:
                if data.get('is_actively_processing', False):
                    processing_label = QLabel("Running...")
                    processing_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                    self.table.setCellWidget(row, 7, processing_label)
                else:
                    action_button = QPushButton("Add to Queue")
                    action_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot, 'both'))
                    self.table.setCellWidget(row, 7, action_button)
            else:
                self.table.setCellWidget(row, 7, QLabel("—"))
            
            # Delete button
            delete_button = QPushButton("Delete Seg")
            delete_button.setStyleSheet("background-color: #f44336; color: white; font-size: 10px;")
            delete_button.clicked.connect(lambda checked, path=folder_path: self.confirm_delete_segmentation(path))
            self.table.setCellWidget(row, 8, delete_button)
    
    def confirm_delete_segmentation(self, folder_path):
        """Confirm and delete segmentation folder"""
        folder_name = os.path.basename(folder_path)
        reply = QMessageBox.question(
            self, 
            'Confirm Delete', 
            f'Delete the entire segmentation folder for "{folder_name}"?\n\n'
            f'This will remove all segmentation and postprocessing results.\n'
            f'This action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.delete_segmentation_folder(folder_path):
                self.log_message(f"Deleted segmentation folder for {folder_name}")
                self.refresh_data()
    
    def add_to_queue(self, folder_path, robot_name, operation_type):
        """Add folder to processing queue"""
        folder_name = os.path.basename(folder_path)
        
        if operation_type == 'both':
            seg_folder = os.path.join(folder_path, 'Segmentation')
            if os.path.exists(seg_folder):
                post_info = self.load_postprocess_info(folder_path)
                stored_alpha = post_info.get('alpha_parameter', 'Unknown') if post_info else 'Unknown'
                current_alpha = self.alpha_parameter
                
                reply = QMessageBox.question(
                    self, 
                    'Segmentation Folder Exists', 
                    f'"{folder_name}" already has a segmentation folder.\n\n'
                    f'To proceed, the existing segmentation will be DELETED:\n'
                    f'• Current Alpha: {current_alpha}\n'
                    f'• Previous Alpha: {stored_alpha}\n\n'
                    f'Delete existing segmentation and restart?\n'
                    f'This action cannot be undone.',
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    self.log_message(f"Cancelled adding {folder_name} to queue - user chose to keep existing segmentation")
                    return
                
                if not self.delete_segmentation_folder(folder_path):
                    self.log_message(f"Failed to delete existing segmentation for {folder_name}")
                    return
        
        queue_item = {'path': folder_path, 'robot': robot_name, 'operation': operation_type}
        if queue_item not in self.processing_queue:
            self.processing_queue.append(queue_item)
            operation_text = "segmentation + postprocess" if operation_type == 'both' else operation_type
            self.log_message(f"Added {folder_name} ({robot_name}) to queue for {operation_text}")
            self.update_queue_info()
            self.refresh_data()
    
    def remove_from_queue(self, folder_path):
        """Remove folder from processing queue"""
        self.processing_queue = [item for item in self.processing_queue if item['path'] != folder_path]
        self.log_message(f"Removed {os.path.basename(folder_path)} from queue")
        self.update_queue_info()
        self.refresh_data()
    
    def clear_queue(self):
        """Clear the processing queue"""
        if self.processing_queue:
            reply = QMessageBox.question(self, 'Clear Queue', 
                                       'Clear the processing queue?\n\nNote: Currently running operations cannot be stopped.',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.processing_queue.clear()
                self.log_message("Queue cleared (running operations will continue)")
                self.update_queue_info()
                self.refresh_data()
    
    def process_queue(self):
        """Process the next item in queue if not currently processing"""
        if not self.current_segmentation_worker and not self.current_postprocess_worker and self.processing_queue:
            next_item = self.processing_queue.pop(0)
            folder_path = next_item['path']
            robot_name = next_item['robot']
            operation = next_item['operation']
            
            if operation == 'both':
                self.start_segmentation(folder_path, robot_name, chain_postprocess=True)
            elif operation == 'segment':
                self.start_segmentation(folder_path, robot_name, chain_postprocess=False)
            elif operation == 'postprocess': # This will now work
                self.start_postprocess(folder_path, robot_name)
            
            self.update_queue_info()
            self.refresh_data()
    
    def start_segmentation(self, folder_path, robot_name, chain_postprocess=False):
        """Start segmentation worker"""
        self.current_segmentation_worker = SegmentationWorker(
            folder_path, robot_name, self.species, self.fast_mode, self.conda_env
        )
        self.current_segmentation_worker.finished.connect(
            lambda path, msg: self.on_segmentation_finished(path, msg, chain_postprocess)
        )
        self.current_segmentation_worker.error.connect(self.on_segmentation_error)
        self.current_segmentation_worker.progress.connect(self.on_progress_update)
        self.current_segmentation_worker.start()
        
        mode_str = " (fast)" if self.fast_mode else ""
        self.log_message(f"Started segmentation for {os.path.basename(folder_path)} ({robot_name}) - {self.species}{mode_str}")
    
    def start_postprocess(self, folder_path, robot_name):
        """Start postprocess worker"""
        self.current_postprocess_worker = PostprocessWorker(
            folder_path, robot_name, self.species, self.alpha_parameter, self.conda_env
        )
        self.current_postprocess_worker.finished.connect(self.on_postprocess_finished)
        self.current_postprocess_worker.error.connect(self.on_postprocess_error)
        self.current_postprocess_worker.progress.connect(self.on_progress_update)
        self.current_postprocess_worker.start()
        
        self.log_message(f"Started postprocessing for {os.path.basename(folder_path)} ({robot_name}) - {self.species}, α={self.alpha_parameter}")
    
    def update_queue_info(self):
        """Update queue information display"""
        queue_size = len(self.processing_queue)
        current = "None"
        if self.current_segmentation_worker:
            current = f"Segmenting: {os.path.basename(self.current_segmentation_worker.input_path)}"
        elif self.current_postprocess_worker:
            current = f"Postprocessing: {os.path.basename(self.current_postprocess_worker.input_path)}"
        
        self.queue_info_label.setText(f"Queue: {queue_size} | Processing: {current}")
    
    def on_segmentation_finished(self, folder_path, message, chain_postprocess=False):
        """Handle segmentation completion"""
        self.log_message(message)
        robot_name = self.current_segmentation_worker.robot_name
        self.current_segmentation_worker = None
        
        if chain_postprocess:
            fold_0_path = os.path.join(folder_path, 'Segmentation', 'Fold_0')
            if os.path.exists(fold_0_path):
                seg_files = len(glob.glob(os.path.join(fold_0_path, '*.png')))
                if seg_files > 0:
                    self.start_postprocess(folder_path, robot_name)
                else:
                    self.log_message(f"Skipping postprocessing for {os.path.basename(folder_path)} - segmentation produced no files")
                    self.update_queue_info()
            else:
                self.log_message(f"Skipping postprocessing for {os.path.basename(folder_path)} - segmentation folder missing")
                self.update_queue_info()
        else:
            self.update_queue_info()
        
        self.refresh_data()
    
    def on_segmentation_error(self, folder_path, message):
        """Handle segmentation error"""
        self.log_message(f"SEGMENTATION ERROR: {message}")
        
        try:
            seg_folder = os.path.join(folder_path, 'Segmentation')
            os.makedirs(seg_folder, exist_ok=True)
            error_file = os.path.join(seg_folder, 'segmentation.error')
            with open(error_file, 'w') as f:
                f.write(f"{datetime.now()}: {message}")
        except:
            pass
        
        self.current_segmentation_worker = None
        self.update_queue_info()
        self.refresh_data()
    
    def on_postprocess_finished(self, folder_path, message):
        """Handle postprocessing completion"""
        self.log_message(message)
        self.current_postprocess_worker = None
        self.update_queue_info()
        self.refresh_data()
    
    def on_postprocess_error(self, folder_path, message):
        """Handle postprocessing error"""
        self.log_message(f"POSTPROCESS ERROR: {message}")
        
        try:
            seg_folder = os.path.join(folder_path, 'Segmentation')
            error_file = os.path.join(seg_folder, 'postprocess.error')
            with open(error_file, 'w') as f:
                f.write(f"{datetime.now()}: {message}")
        except:
            pass
        
        self.current_postprocess_worker = None
        self.update_queue_info()
        self.refresh_data()
    
    def on_progress_update(self, folder_path, status):
        """Handle progress updates from workers"""
        self.log_message(f"{os.path.basename(folder_path)}: {status}")
    
    def refresh_data(self):
        """Refresh folder data and update table"""
        if self.auto_refresh_enabled:
            for robot_name in self.robots.keys():
                self.scan_robot_folders(robot_name)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def main():
    app = QApplication(sys.argv)
    window = nnUNetMonitorUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()