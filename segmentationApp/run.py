import os
import sys
import subprocess
import json
import shutil
import glob
from datetime import datetime
from pathlib import Path

# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QFileDialog, QCheckBox,
                             QMessageBox, QTextEdit, QComboBox, QInputDialog, QAbstractItemView)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QDir
from PyQt5.QtGui import QColor

# --- Constants ---
APP_NAME = "chronoroot"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "segmentationInterfaceConfig.json")

# Ensure config directory exists
os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)

def get_available_models():
    """Scans the local models/ directory for available species/models."""
    models_dir = Path(__file__).parent.resolve() / "models"
    if not models_dir.exists():
        return []
    return [d.name for d in models_dir.iterdir() if d.is_dir()]

class CLIWorker(QThread):
    """Unified worker that runs the CLI backend"""
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(str, str)

    def __init__(self, input_path, robot_name, model, alpha, 
                 postprocess_only=False, resume=False, fast_mode=False, conda_env="ChronoRoot"):
        super().__init__()
        self.input_path = Path(input_path)
        self.robot_name = robot_name
        self.model = model
        self.alpha = alpha
        self.resume = resume
        self.postprocess_only = postprocess_only
        self.fast_mode = fast_mode
        self.conda_env = conda_env
        self.process = None 
        self._is_killed = False

    def run(self):
        try:
            # Test if there is a torch installation in the conda env
            if not self.postprocess_only:
                test_cmd = [
                    "conda", "run", "--no-capture-output", "-n", self.conda_env,
                    "python", "-c", "import torch; print(torch.__version__)"
                ]
                test_process = subprocess.Popen(
                    test_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = test_process.communicate()
                if test_process.returncode != 0:
                    self.error.emit(str(self.input_path), 
                                    f"Conda environment '{self.conda_env}' is missing PyTorch. \nYou are running in monitor mode only.\nInstall the full ChronoRoot application to enable segmentation.")
                    return 
            
            folder_name = self.input_path.name
            op_type = "Postprocessing" if self.postprocess_only else "Segmentation"
            self.progress.emit(str(self.input_path), f"Starting {op_type}...")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                "conda", "run", "--no-capture-output", "-n", self.conda_env,
                "python", "cli.py",
                str(self.input_path),
                "--model", self.model,
                "--alpha", str(self.alpha),
                "--device", "cuda"
            ]

            if self.postprocess_only:
                cmd.append("--postprocess-only")
            if self.resume:
                cmd.append("--resume")
            if self.fast_mode:
                cmd.append("--fast")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merged streams to catch errors in real-time
                text=True,
                bufsize=1, # Line-buffered
                env=env,
                universal_newlines=True,
                start_new_session=True
            )

            for line in iter(self.process.stdout.readline, ''):
                if self._is_killed:
                    break
                
                clean_line = line.strip()
                if clean_line:
                    self.progress.emit(str(self.input_path), clean_line)

            self.process.stdout.close()
            rc = self.process.wait()

            if rc == 0:
                self.finished.emit(str(self.input_path), f"{op_type} complete: {folder_name}")
            else:
                self.error.emit(str(self.input_path), f"Error (Code {rc}) - Check log above")

        except Exception as e:
            self.error.emit(str(self.input_path), f"System Error: {str(e)}")
            
    def stop(self):
        """Force kills the subprocess group safely."""
        self._is_killed = True
        if self.process:
            try:
                import signal
                os.killpg(self.process.pid, signal.SIGTERM)
            except Exception as e:
                self.error.emit(str(self.input_path), f"Failed to terminate process: {str(e)}")
                self.process.terminate()
        self._mark_metadata_terminated()
        self.finished.emit(str(self.input_path), "Process terminated by user.")

    def _mark_metadata_terminated(self):
        """Writes 'Error: Terminated' to the metadata file."""
        meta_file = self.input_path / 'Segmentation' / 'segmentation_metadata.json'
        
        # We only want to edit the file if it already exists
        if not meta_file.exists():
            return

        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            
            if data['segmentation_status'] == "Success":
                data['postprocessing_status'] = "Error: Terminated"
            else:
                data['segmentation_status'] = "Error: Terminated"

            with open(meta_file, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            self.error.emit(str(self.input_path), f"Failed to mark termination in metadata: {str(e)}")


class nnUNetMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # State
        self.robots = {}  
        self.robot_paths = {} 
        self.folder_data = {}
        self.processing_queue = []
        self.current_worker = None
        
        # Configuration Defaults
        self.alpha_parameter = 0.85
        self.conda_env = "ChronoRoot"
        self.fast_mode = False
        self.species = "arabidopsis"
        
        self.init_ui()
        self.load_settings()
        self.setup_timer()
        
    def init_ui(self):
        self.setWindowTitle("nnUNet Segmentation Monitor")
        self.setGeometry(100, 100, 1400, 600) 
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # --- Header ---
        header_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Robot")
        self.load_button.setToolTip("Select and load one or more robot folders to monitor")
        self.load_button.clicked.connect(self.load_robot)
        self.load_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 6px; font-weight: bold; }")
        
        # New: Remove Robot Button
        self.remove_robot_button = QPushButton("Remove Robot")
        self.remove_robot_button.setToolTip("Unload the currently filtered robot from the view") 
        self.remove_robot_button.clicked.connect(self.remove_robot)
        self.remove_robot_button.setStyleSheet("QPushButton { background-color: #607D8B; color: white; padding: 6px;}")

        self.robot_count_label = QLabel("Robots: 0")
        self.queue_info_label = QLabel("Queue: 0 | Processing: None")
        
        self.alpha_button = QPushButton(f"Alpha: {self.alpha_parameter}")
        self.alpha_button.setToolTip("Click to adjust the Alpha parameter for postprocessing") 
        self.alpha_button.clicked.connect(self.set_alpha_parameter)
        self.alpha_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white;}")

        self.conda_button = QPushButton(f"Conda: {self.conda_env}")
        self.conda_button.setToolTip("Set the Conda environment name to be used for segmentation")
        self.conda_button.clicked.connect(self.set_conda_env)
        
        self.fast_mode_checkbox = QCheckBox("Fast Mode")
        self.fast_mode_checkbox.setToolTip("Enable fast mode (by disabling test-time augmentations)")
        self.fast_mode_checkbox.stateChanged.connect(self.update_fast_mode)

        # New: Hide Empty Folders Checkbox
        self.hide_empty_checkbox = QCheckBox("Hide Empty Folders")
        self.hide_empty_checkbox.setToolTip("Hide folders that contain no images or data")
        self.hide_empty_checkbox.stateChanged.connect(self.refresh_data) 
        
        self.species_combo = QComboBox()
        self.species_combo.addItems(get_available_models() or ["No models found"])
        self.species_combo.currentTextChanged.connect(self.update_species)
        
        self.clear_queue_button = QPushButton("Clear Queue")
        self.clear_queue_button.setToolTip("Remove all pending tasks from the processing queue") 
        self.clear_queue_button.clicked.connect(self.clear_queue)
        self.clear_queue_button.setStyleSheet("QPushButton { background-color: #f44336; color: white;}")
        
        self.auto_refresh_enabled = True
        
        header_items = [
            self.load_button, self.remove_robot_button, 
            self.robot_count_label, QLabel("|"), self.queue_info_label,
            self.alpha_button, QLabel("Model:"), self.species_combo, 
            self.fast_mode_checkbox, self.hide_empty_checkbox,
            self.clear_queue_button, self.conda_button
        ]
        for item in header_items:
            header_layout.addWidget(item) if isinstance(item, QWidget) else None
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # --- Filters ---
        filter_layout = QHBoxLayout()
        self.robot_filter = QComboBox()
        self.robot_filter.addItem("All Robots")
        self.robot_filter.currentTextChanged.connect(self.update_table)
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Not Started", "Queued", "Segmenting", 
                                    "Stalled", "Segmented", "Postprocessing", 
                                    "Complete", "Different Alpha", "Different Model", "Error"])
        self.status_filter.currentTextChanged.connect(self.update_table)
        
        filter_layout.addWidget(QLabel("Filter Robot:"))
        filter_layout.addWidget(self.robot_filter)
        filter_layout.addWidget(QLabel("Filter Status:"))
        filter_layout.addWidget(self.status_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(10) 
        # Updated Headers to match new logic
        self.table.setHorizontalHeaderLabels([
            "Robot", "Folder Name", "Images", "Segmentation %", 
            "Postprocessing %", "Model - Alpha", "Status", "Actions", 
            "Remove View", "Clear Results"
        ])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch) # Robot Name stretches
        self.table.setColumnWidth(1, 100)  # Folder Name
        self.table.setColumnWidth(2, 60)  # Images
        self.table.setColumnWidth(3, 120)  # Segmentation %
        self.table.setColumnWidth(4, 120)  # Postprocessing %
        self.table.setColumnWidth(5, 120) # Model - Alpha
        self.table.setColumnWidth(6, 140) # Status
        self.table.setColumnWidth(7, 320) # Actions (Wide for text)
        self.table.setColumnWidth(8, 100) # Remove View
        self.table.setColumnWidth(9, 100) # Clear Results
        
        layout.addWidget(self.table)
        
        # --- Log ---
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("QTextEdit { background-color: #f0f0f0; font-family: monospace; font-size: 10px;}")
        layout.addWidget(self.log_text)
        
        # --- End of UI Setup ---    
        return 
    
    # --- Logic ---
    
    def analyze_folder(self, folder_path, robot_name):
        """
        Analyzes folder status.
        Legacy paths: Segmentation/Fold_0 (Seg), Segmentation/Ensemble (Post)
        """
        data = {
            'path': folder_path, 
            'robot': robot_name, 
            'total_images': 0,
            'seg_progress': 0,
            'post_progress': 0,
            'status': 'Not Started', 
            'stored_alpha': None,
            'model': None
        }
        
        # --- 1. Determine Image Count & Metadata ---
        meta_file = Path(folder_path) / 'Segmentation' / 'segmentation_metadata.json'
        meta = None
        
        # Try reading metadata first
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    data['total_images'] = meta.get('n_images', 0)                    
                    data['stored_alpha'] = meta.get('alpha_used', None)
                    data['model'] = meta.get('model', None)
            except: pass

        # Fallback to file count
        if data['total_images'] == 0:
            pngs = glob.glob(os.path.join(folder_path, '*.png'))
            data['total_images'] = len(pngs)

        if data['total_images'] == 0:
            data['status'] = 'No Images'
            return data

        # --- 2. Determine Status ---
        
        # Check if currently queued/running
        is_queued = any(item['path'] == folder_path for item in self.processing_queue)
        is_running = (self.current_worker is not None and 
                      os.path.abspath(self.current_worker.input_path) == os.path.abspath(folder_path))

        if is_running:            
            if self.current_worker.postprocess_only:
                data['status'] = 'Postprocessing'
                data['seg_progress'] = 100.0
                if meta:
                     data['post_progress'] = meta.get('postprocessing_progress', 0.0)
                else:
                     data['post_progress'] = 0.0
            else:
                data['status'] = 'Segmenting'
                if meta:
                    data['seg_progress'] = meta.get('segmentation_progress', 0.0)
                    if meta.get('segmentation_status') == 'Success':
                         data['seg_progress'] = 100.0
                else:
                    data['seg_progress'] = 0.0
                data['post_progress'] = 0.0

            data['stored_alpha'] = self.current_worker.alpha
            data['model'] = self.current_worker.model
            return data
        
        if is_queued:
            data['status'] = 'Queued'
            data['post_progress'] = 0.0
            for item in self.processing_queue:
                if item['path'] == folder_path:
                    data['stored_alpha'] = item['alpha']
                    data['model'] = item['model']
                    if item['operation'] == 'postprocess':
                        data['seg_progress'] = 100.0
                    elif item['operation'] == 'resume':
                        data['seg_progress'] = meta.get('segmentation_progress', 0.0) if meta else 0.0
                    else:
                        data['seg_progress'] = 0.0
            return data
        
        elif meta:
            segmentation_status = meta.get('segmentation_status', 'Not started')
            last_segmentation_time = meta.get('last_segmentation_time', None)
            segmentation_progress = meta.get('segmentation_progress', 0.0)
            segmentation_average_time = meta.get('segmentation_average_time_per_image', 0)
            
            postprocessing_status = meta.get('postprocessing_status', 'Not started')
            last_postprocessing_time = meta.get('last_postprocessing_time', None)
            postprocessing_progress = meta.get('postprocessing_progress', 0.0)
            postprocessing_average_time = meta.get('postprocessing_average_time_per_image', 0)
            
            data['stored_alpha'] = meta.get('alpha_used', None)
            data['model'] = meta.get('model', None)
            
            # Check for errors first
            if "Error" in segmentation_status:
                data['status'] = 'Error (Seg)'
                data['seg_progress'] = segmentation_progress
                data['post_progress'] = 0.0
                return data
            if "Error" in postprocessing_status:
                data['status'] = 'Error (Post)'
                data['seg_progress'] = 100.0
                data['post_progress'] = postprocessing_progress
                return data
            
            if segmentation_status == "Started":
                current_time = datetime.now()
                if last_segmentation_time:
                    last_time = datetime.strptime(last_segmentation_time, "%Y-%m-%d %H:%M:%S")
                    delta = (current_time - last_time).total_seconds()
                    if delta > segmentation_average_time  * 10: 
                        data['status'] = 'Stalled'
                        data['seg_progress'] = segmentation_progress
                        data['post_progress'] = 0.0
                    else:
                        data['status'] = 'Segmenting'
                        data['seg_progress'] = segmentation_progress
                        data['post_progress'] = 0.0
                else:
                    # check segmentation date
                    segmentation_date = meta['segmentation_date']
                    # failsafe if it is from the last 5 minutes, assume running
                    seg_time = datetime.strptime(segmentation_date, "%Y-%m-%d %H:%M:%S")
                    delta = (current_time - seg_time).total_seconds()
                    if delta < 300:
                        data['status'] = 'Segmenting'
                        data['seg_progress'] = segmentation_progress
                        data['post_progress'] = 0.0
                    else:
                        data['status'] = 'Stalled'
                        data['seg_progress'] = segmentation_progress
                        data['post_progress'] = 0.0
                return data
            
            elif segmentation_status == "Success" and postprocessing_status == "Started":
                current_time = datetime.now()
                if last_postprocessing_time:
                    last_time = datetime.strptime(last_postprocessing_time, "%Y-%m-%d %H:%M:%S")
                    delta = (current_time - last_time).total_seconds()
                    if delta > postprocessing_average_time * 10:
                        data['status'] = 'Stalled'
                        data['seg_progress'] = 100.0
                        data['post_progress'] = postprocessing_progress
                    else:
                        data['status'] = 'Postprocessing'
                        data['seg_progress'] = 100.0
                        data['post_progress'] = postprocessing_progress
                else:
                    # check postprocessing date
                    postprocessing_date = meta['postprocessing_date']
                    # failsafe if it is from the last 30 seconds, assume running
                    post_time = datetime.strptime(postprocessing_date, "%Y-%m-%d %H:%M:%S")
                    delta = (current_time - post_time).total_seconds()
                    if delta < 30:
                        data['status'] = 'Postprocessing'
                        data['seg_progress'] = 100.0
                        data['post_progress'] = postprocessing_progress
                    else:
                        data['status'] = 'Error (Post)'
                        data['seg_progress'] = 100.0
                        data['post_progress'] = postprocessing_progress
                
                return data
            
            if not is_running and segmentation_status == "Success" and postprocessing_status == "Success":
                data['seg_progress'] = 100.0
                data['post_progress'] = 100.0
                data['status'] = "Complete"
                
                if data['stored_alpha'] is not None and abs(data['stored_alpha'] - self.alpha_parameter) > 0.01:
                    data['status'] = 'Different Alpha'
                    if data['model'] is not None and data['model'] != self.species:
                        data['status'] = 'Different Model and Alpha'
                elif data['model'] is not None and data['model'] != self.species:
                    data['status'] = 'Different Model'
                
                return data
            
        else:
            # --- Legacy Fallback Logic ---
            # Check for Fold_0 (Segmentation)
            fold0_path = os.path.join(folder_path, "Segmentation", "Fold_0")
            seg_count = len(glob.glob(os.path.join(fold0_path, "*.png"))) if os.path.exists(fold0_path) else 0
            
            # Check for Ensemble (Postprocessing)
            ensemble_path = os.path.join(folder_path, "Segmentation", "Ensemble")
            post_count = len(glob.glob(os.path.join(ensemble_path, "*.png"))) if os.path.exists(ensemble_path) else 0
            
            # Legacy Status Determination
            if post_count >= data['total_images']:
                data['status'] = "Complete (Legacy)"
                data['seg_progress'] = 100.0
                data['post_progress'] = 100.0
            elif seg_count >= data['total_images']:
                data['status'] = "Segmented (Legacy)"
                data['seg_progress'] = 100.0
                data['post_progress'] = int((post_count / data['total_images']) * 100)
            elif seg_count > 0:
                data['status'] = "Incomplete (Legacy)"
                data['seg_progress'] = int((seg_count / data['total_images']) * 100)
                data['post_progress'] = 0.0
            else:
                data['status'] = 'Not Started'

        return data

    def update_table(self):
        self.table.setUpdatesEnabled(False)
        try:
            """Renders the analyzed folder data into the table."""
            filtered_rows = []
            robot_filter = self.robot_filter.currentText()
            status_filter = self.status_filter.currentText()
            
            for key, data in self.folder_data.items():
                if robot_filter != "All Robots" and data['robot'] != robot_filter:
                    continue
                
                s = data['status']
                if status_filter != "All Status":
                    if status_filter == "Error" and "Error" not in s: continue
                    elif status_filter == "Complete" and "Complete" not in s: continue
                    elif status_filter == "Queued" and s != "Queued": continue
                    elif status_filter == "Stalled" and s != "Stalled": continue
                    elif status_filter == "Different Model and Alpha" and s != "Different Model and Alpha": continue
                    elif status_filter == "Different Alpha" and s != "Different Alpha": continue
                    elif status_filter == "Different Model" and s != "Different Model": continue
                    elif status_filter not in ["Error", "Complete"] and s != status_filter: continue
                
                filtered_rows.append(data)

            self.table.setRowCount(len(filtered_rows))

            # Status Color Map
            colors = {
                'Complete': QColor(200, 255, 200),         
                'Complete (Legacy)': QColor(210, 255, 210),
                'Segmented': QColor(200, 230, 255),
                'Segmented (Legacy)': QColor(200, 230, 255),     
                'Segmenting': QColor(255, 255, 224),       
                'Postprocessing': QColor(230, 200, 255),   
                'Queued': QColor(255, 255, 200),           
                'Stalled': QColor(255, 200, 100),          
                'Different Alpha': QColor(255, 240, 150), 
                'Different Model': QColor(255, 240, 150),
                'Different Model and Alpha': QColor(255, 240, 150),
                'Error (Seg)': QColor(255, 200, 200),      
                'Error (Post)': QColor(255, 200, 200),
                'Incomplete (Legacy)': QColor(255, 220, 180),  
                'No Images': QColor(240, 240, 240)         
            }

            for row, data in enumerate(filtered_rows):
                self.table.setItem(row, 0, QTableWidgetItem(data['robot']))
                self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(data['path'])))
                self.table.setItem(row, 2, QTableWidgetItem(str(data['total_images'])))
                
                # Segmentation %
                p_bar = QProgressBar()
                p_bar.setValue(int(data['seg_progress']))
                p_bar.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(row, 3, p_bar)

                # Postprocessing %
                p_bar_post = QProgressBar()
                p_bar_post.setValue(int(data['post_progress']))
                p_bar_post.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(row, 4, p_bar_post)

                # Model - Alpha
                alpha = f"{data['stored_alpha']:.2f}" if data['stored_alpha'] is not None else "Unknown"
                model = f"{data['model']}" if 'model' in data and data['model'] is not None else "N/A"
                ma_item = QTableWidgetItem(f"{model} - {alpha}")
                ma_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 5, ma_item)
                
                # Status
                s_item = QTableWidgetItem(data['status'])
                bg_color = colors.get(data['status'], QColor(255, 255, 255))
                s_item.setBackground(bg_color)
                if data['status'] == 'Stalled': s_item.setForeground(QColor(200, 0, 0)) # Red text for visibility
                self.table.setItem(row, 6, s_item)

                # Col 6: Context Actions
                self._set_row_action(row, data)
                
                # Col 7: Remove from View (UI only)
                rm_btn = QPushButton("Remove")
                rm_btn.setToolTip("Stop monitoring this folder (does not delete files)") 
                rm_btn.setFixedSize(80, 25)
                rm_btn.setStyleSheet("QPushButton { color: #555;}")
                rm_btn.clicked.connect(lambda _, p=data['path']: self.remove_folder_from_ui(p))
                
                w_rm = QWidget()
                l_rm = QHBoxLayout(w_rm)
                l_rm.setContentsMargins(0, 0, 0, 0) 
                l_rm.setAlignment(Qt.AlignCenter)
                l_rm.addWidget(rm_btn)
                self.table.setCellWidget(row, 8, w_rm)

                # Col 8: Clear Results (Disk deletion)
                del_btn = QPushButton("Clear Results")
                del_btn.setToolTip("Permanently delete the 'Segmentation' output folder")
                del_btn.setFixedSize(90, 25)
                del_btn.setStyleSheet("QPushButton { background-color: #ffebee; border: 1px solid #ffcdd2; color: #c62828; }")
                del_btn.clicked.connect(lambda _, p=data['path']: self.confirm_clear_results(p))
                
                w_del = QWidget()
                l_del = QHBoxLayout(w_del)
                l_del.setContentsMargins(0, 0, 0, 0) 
                l_del.setAlignment(Qt.AlignCenter)  
                l_del.addWidget(del_btn)
                self.table.setCellWidget(row, 9, w_del)
        finally:
            self.table.setUpdatesEnabled(True)
    
    def _set_row_action(self, row, data):
        """Sets action buttons, supporting dual-choice for configuration mismatches."""
        status = data['status']
        path = data['path']
        robot = data['robot']
        alpha = self.alpha_parameter
        model = self.species
        
        # Container Widget
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5) 
        layout.setAlignment(Qt.AlignCenter)
        
        buttons_to_add = []

        if status == 'Different Model' or status == 'Different Model and Alpha':
            # Choice 1: Full Rerun (Update Model + Alpha)
            btn_seg = QPushButton("Rerun Pipeline")
            btn_seg.setToolTip(f"Rerun Segmentation ({model}) & Postprocessing ({alpha})")
            btn_seg.setFixedSize(110, 25)
            btn_seg.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-weight: bold;}")
            btn_seg.clicked.connect(lambda: self.add_to_queue(path, robot, 'both', model, alpha))
            buttons_to_add.append(btn_seg)
            
            # Choice 2: Postprocess Only (Ignore Model, Update Alpha)
            btn_post = QPushButton("Postproc Only")
            btn_post.setToolTip(f"Keep existing masks, rerun postprocessing with alpha {alpha}")
            btn_post.setFixedSize(110, 25)
            btn_post.setStyleSheet("QPushButton { background-color: #FFC107; color: black;}")
            btn_post.clicked.connect(lambda: self.add_to_queue(path, robot, 'postprocess', model, alpha))
            buttons_to_add.append(btn_post)

        # SCENARIO: Alpha Mismatch Only (Model is correct)
        elif status == 'Different Alpha':
            btn = QPushButton("Update Alpha")
            btn.setToolTip(f"Rerun postprocessing with new alpha: {alpha}")
            btn.setFixedSize(225, 25) 
            btn.setStyleSheet("QPushButton { background-color: #FFEB3B; color: black;}")
            btn.clicked.connect(lambda: self.add_to_queue(path, robot, 'postprocess', model, alpha))
            buttons_to_add.append(btn)

        # SCENARIO: Stalled
        elif status == 'Stalled':
            btn = QPushButton("Resume")
            btn.setToolTip("Resume segmentation from where it stalled") 
            btn.setFixedSize(225, 25)
            btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold;}")
            btn.clicked.connect(lambda: self.add_to_queue(path, robot, 'resume', model, alpha))
            buttons_to_add.append(btn)

        # SCENARIO: Standard Start 
        elif status in ['Not Started']:
            btn = QPushButton("Start Pipeline")
            btn.setToolTip(f"Run full segmentation ({model}) and postprocessing ({alpha})") 
            btn.setFixedSize(225, 25)
            btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold;}")
            btn.clicked.connect(lambda: self.add_to_queue(path, robot, 'both', model, alpha))
            buttons_to_add.append(btn)
            
        # SCENARIO: Error in Segmentation, allow full rerun or resume
        elif status == 'Error (Seg)':
            btn_full = QPushButton("Restart Pipeline")
            btn_full.setToolTip(f"Rerun full segmentation ({model}) & postprocessing ({alpha})")
            btn_full.setFixedSize(110, 25)
            btn_full.setStyleSheet("QPushButton { background-color: #F44336; color: white; font-weight: bold;}")
            btn_full.clicked.connect(lambda: self.add_to_queue(path, robot, 'both', model, alpha))
            buttons_to_add.append(btn_full)
            
            btn_resume = QPushButton("Resume")
            btn_resume.setToolTip("Resume segmentation from last correctly processed image")
            btn_resume.setFixedSize(110, 25)
            btn_resume.setStyleSheet("QPushButton { background-color: #FF5722; color: white;}")
            btn_resume.clicked.connect(lambda: self.add_to_queue(path, robot, 'resume', model, alpha))
            buttons_to_add.append(btn_resume)
        
        # SCENARIO: Standard Rerun
        elif status in ['Segmented', 'Segmented (Legacy)', 'Error (Post)', 'Complete (Legacy)']:
            btn = QPushButton("Rerun Postprocessing")
            btn.setToolTip(f"Re-calculate postprocessing using Alpha {alpha}")
            btn.setFixedSize(225, 25)
            btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white;}")
            btn.clicked.connect(lambda: self.add_to_queue(path, robot, 'postprocess', model, alpha))
            buttons_to_add.append(btn)

        # SCENARIO: Queued
        elif status == 'Queued':
            btn = QPushButton("Cancel")
            btn.setToolTip("Remove this item from the processing queue")
            btn.setFixedSize(225, 25)
            btn.clicked.connect(lambda: self.remove_from_queue(path))
            buttons_to_add.append(btn)

        # SCENARIO: Running
        elif 'Processing' in status or 'Segmenting' in status or 'Postprocessing' in status:
            # if running locally allow to kill
            if (self.current_worker and os.path.abspath(self.current_worker.input_path) == os.path.abspath(path)):
                btn = QPushButton("Kill Process")
                btn.setToolTip("Force stop the currently running process")
                btn.setFixedSize(225, 25)
                btn.setStyleSheet("QPushButton { background-color: #f44336; color: white;}")
                btn.clicked.connect(lambda: self.current_worker.stop())
                buttons_to_add.append(btn)
            else:
                lbl = QLabel("Running...")
                lbl.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(row, 7, lbl) 
                self.table.setItem(row, 7, None)
                return

        # --- Render ---
        if buttons_to_add:
            self.table.setItem(row, 7, None) 
            for btn in buttons_to_add:
                layout.addWidget(btn)
            self.table.setCellWidget(row, 7, container)
        else:
            self.table.removeCellWidget(row, 7)
            item = QTableWidgetItem("—")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 7, item)
            
    # --- Standard Boilerplate methods (Load, Save, Queue, etc.) ---

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_update)
        self.timer.start(2000)

    def timer_update(self):
        # Throttle logic: fast update if working, slow if idle
        is_working = bool(self.current_worker or self.processing_queue)
        self.timer.setInterval(1000 if is_working else 3000)
        
        if self.auto_refresh_enabled:
            self.refresh_data()
        self.process_queue()

    def load_robot(self):
        """
        Opens a custom file dialog allowing multiple folder selection.
        Always uses the folder name as the Robot Name (no prompts).
        """
                
        # 1. Setup Custom Dialog for Multi-Select
        dialog = QFileDialog(self, "Select Robot Folder(s)")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        
        # Enable multi-selection in the non-native dialog view
        for view in dialog.findChildren(QAbstractItemView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # 2. Execute and Process
        if dialog.exec_():
            folder_paths = dialog.selectedFiles()
            
            # Filter valid directories
            folder_paths = [p for p in folder_paths if os.path.isdir(p)]
            
            if not folder_paths:
                return

            loaded_count = 0
            
            # 3. Iterate and Load (Same logic for 1 or 100 folders)
            for folder_path in folder_paths:
                # Normalize path removes trailing slashes so basename works correctly
                r_name = os.path.basename(os.path.normpath(folder_path))
                
                # Register and Discover
                self.robots[r_name] = {'path': folder_path}
                self.discover_robot_paths(r_name)
                loaded_count += 1
            
            if loaded_count > 0:
                self.log_message(f"Loaded {loaded_count} robot(s).")
                self.update_robot_filter()

    def discover_robot_paths(self, robot_name):
        """Performs disk discovery (os.listdir) and populates the monitoring list."""
        if robot_name not in self.robots: return
        r_path = self.robots[robot_name]['path']
        
        found_paths = set()
        try:
            items = os.listdir(r_path)
            for item in items:
                full_path = os.path.join(r_path, item)
                if os.path.isdir(full_path):
                    found_paths.add(full_path)
        except Exception as e:
            self.log_message(f"Discovery Error: {e}")
            return

        # Initialize or update the checklist
        self.robot_paths[robot_name] = found_paths
        
        # Immediately analyze the newly found paths
        self.scan_robot_folders(robot_name)
        
    def scan_robot_folders(self, robot_name):
        if robot_name not in self.robot_paths: return
        
        paths_to_check = self.robot_paths[robot_name]
        data_changed = False # Flag to track changes
        
        for f_path in paths_to_check:
            # Analyze (Reads metadata/counts files)
            folder_data = self.analyze_folder(f_path, robot_name)
            
            # Update Data Store
            key = f"{robot_name}::{f_path}"
            
            # Only mark as changed if the data is actually different
            if key not in self.folder_data or self.folder_data[key] != folder_data:
                self.folder_data[key] = folder_data
                data_changed = True
            
        # Only trigger the UI redraw if something actually changed
        if data_changed:
            self.update_table()

    def remove_robot(self):
        """Removes the currently selected robot from the UI."""
        target_robot = self.robot_filter.currentText()
        if target_robot == "All Robots":
            QMessageBox.warning(self, "Selection Error", "Please select a specific robot using the filter to remove it from the view.")
            return

        reply = QMessageBox.question(self, 'Remove Robot', 
                                     f"Remove '{target_robot}' from view?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 1. Clear Data
            keys_to_remove = [k for k, v in self.folder_data.items() if v['robot'] == target_robot]
            for k in keys_to_remove:
                del self.folder_data[k]
            
            # 2. Clear Checklist
            if target_robot in self.robot_paths:
                del self.robot_paths[target_robot]
            
            # 3. Clear Registry
            if target_robot in self.robots:
                del self.robots[target_robot]
            
            self.update_robot_filter()
            self.update_table()

    def remove_folder_from_ui(self, path):
        """Removes a folder from the monitoring list (robot_paths)."""
        # 1. Find which robot owns this path
        target_robot = None
        for r_name, paths in self.robot_paths.items():
            if path in paths:
                target_robot = r_name
                break
        
        # 2. Remove from the checklist (stop monitoring)
        if target_robot:
            self.robot_paths[target_robot].discard(path)
        
        # 3. Remove from current data view
        key_to_remove = None
        for k, v in self.folder_data.items():
            if v['path'] == path:
                key_to_remove = k
                break
        
        if key_to_remove:
            del self.folder_data[key_to_remove]
        
        # 4. If a robot has no more paths, remove it entirely
        if target_robot and not self.robot_paths[target_robot]:
            del self.robot_paths[target_robot]
            if target_robot in self.robots:
                del self.robots[target_robot]
        
        self.update_robot_filter()
        self.update_table()

    def confirm_clear_results(self, path):
        """Destructive action: Deletes Segmentation folder and metadata."""
        msg = (f"Confirm Action for: {os.path.basename(path)}\n\n"
               "This will delete all generated masks in the 'Segmentation' folder "
               "and reset the metadata.\n\n"
               "This cannot be undone.\n"
               "Source images will NOT be affected.")
               
        reply = QMessageBox.question(self, 'Delete Segmentation Results', msg, 
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Delete Segmentation Folder
            seg_dir = os.path.join(path, 'Segmentation')
            if os.path.exists(seg_dir):
                try:
                    shutil.rmtree(seg_dir)
                    self.log_message(f"Deleted segmentation folder: {os.path.basename(path)}")
                except Exception as e:
                    self.log_message(f"Error deleting folder: {e}")
            
            # Refresh to update status to "Not Started"
            self.refresh_data()
            
    def add_to_queue(self, path, robot, op, model, alpha):
        item = {'path': path, 'robot': robot, 'operation': op, 'model': model, 'alpha': alpha}
        if item not in self.processing_queue:
            self.processing_queue.append(item)
            self.log_message(f"Added to queue: {os.path.basename(path)} ({op}), alpha={alpha})")
            self.refresh_data()
        self.update_queue_label()

    def remove_from_queue(self, path):
        self.processing_queue = [x for x in self.processing_queue if x['path'] != path]
        self.refresh_data()
        self.update_queue_label()

    def ensure_legacy_metadata(self, folder_path):
        """
        Creates metadata for legacy folders so the CLI knows segmentation is done.
        """
        meta_file = Path(folder_path) / 'Segmentation' / 'segmentation_metadata.json'
        if meta_file.exists():
            return

        # Check legacy segmentation folder
        fold0_path = os.path.join(folder_path, "Segmentation", "Fold_0")
        pngs = glob.glob(os.path.join(fold0_path, "*.png")) if os.path.exists(fold0_path) else []
        
        if not pngs: return # Cannot verify legacy segmentation
        
        # Create minimal metadata
        meta = {
            "input_path": folder_path,
            "output_path": folder_path,
            "model": self.species, 
            "fast_mode": self.fast_mode,
            "segmentation_status": "Success", # Crucial for skipping segmentation
            "segmentation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_images": len(pngs),
            "processed_images": len(pngs),
            "processed_files": "All",
            "segmentation_progress": 100.0,
            "postprocessing_progress": 0.0,
            "average_time_per_image": 0.0,
            "postprocessing_status": "Not started"
        }
        
        try:
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=4)
            self.log_message(f"Created legacy metadata for {os.path.basename(folder_path)}")
        except Exception as e:
            self.log_message(f"Failed to create legacy metadata: {e}")
            
    def process_queue(self):
        if self.current_worker or not self.processing_queue: return
        
        item = self.processing_queue.pop(0)
        if item['operation'] == 'postprocess':
            self.ensure_legacy_metadata(item['path'])
            
        self.current_worker = CLIWorker(
            item['path'], item['robot'], self.species, item['alpha'],
            postprocess_only=(item['operation'] == 'postprocess'),
            resume=(item['operation'] == 'resume'),
            fast_mode=self.fast_mode, conda_env=self.conda_env
        )
        self.current_worker.finished.connect(self.on_worker_done)
        self.current_worker.error.connect(self.on_worker_done)
        self.current_worker.progress.connect(
            lambda p, m: self.log_message(f"[{os.path.basename(p)}] {m}")
        )
        self.current_worker.start()
        self.update_queue_label()

    def on_worker_done(self, path, msg):
        self.log_message(msg)
        
        # Wait for the thread to fully exit its run() method before unreferencing it.
        if self.current_worker is not None:
            self.current_worker.quit()
            self.current_worker.wait() 
        
        self.current_worker = None
        self.refresh_data()
        self.update_queue_label()

    def update_queue_label(self):
        q_len = len(self.processing_queue)
        curr = f"Processing: {self.current_worker.input_path.name}" if self.current_worker else "Idle"
        self.queue_info_label.setText(f"Queue: {q_len} | {curr}")

    def refresh_data(self):
        for r_name in self.robots:
            self.scan_robot_folders(r_name)

    def update_robot_filter(self):
        curr = self.robot_filter.currentText()
        self.robot_filter.clear()
        self.robot_filter.addItem("All Robots")
        self.robot_filter.addItems(list(self.robots.keys()))
        self.robot_filter.setCurrentText(curr)
        self.robot_count_label.setText(f"Robots: {len(self.robots)}")

    def set_alpha_parameter(self):
        val, ok = QInputDialog.getDouble(self, 'Alpha', 'Value:', self.alpha_parameter, 0.0, 1.0, 2)
        if ok: 
            self.alpha_parameter = val
            self.alpha_button.setText(f"Alpha: {val}")
            self.save_settings()

    def set_conda_env(self):
        text, ok = QInputDialog.getText(self, 'Conda Env', 'Name:', text=self.conda_env)
        if ok: 
            self.conda_env = text
            self.conda_button.setText(f"Conda Environment: {text}")
            self.save_settings()

    def update_fast_mode(self, state):
        self.fast_mode = (state == Qt.Checked)
        self.save_settings()

    def update_species(self, text):
        self.species = text
        if "arabidopsis" in text.lower(): self.alpha_parameter = 0.85
        elif "tomato" in text.lower(): self.alpha_parameter = 0.60
        self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
        self.save_settings()

    def save_settings(self):
        try:
            with open(GLOBAL_CONFIG_FILE, 'w') as f:
                json.dump({'conda_env': self.conda_env, 'alpha': self.alpha_parameter, 
                           'species': self.species, 'fast_mode': self.fast_mode}, f)
        except: pass

    def load_settings(self):
        if os.path.exists(GLOBAL_CONFIG_FILE):
            try:
                with open(GLOBAL_CONFIG_FILE, 'r') as f:
                    c = json.load(f)
                    self.conda_env = c.get('conda_env', 'ChronoRoot')
                    self.species = c.get('species', 'arabidopsis')
                    self.alpha_parameter = c.get('alpha', 0.85)
                    self.fast_mode = c.get('fast_mode', False)
                    
                    self.conda_button.setText(f"Conda Environment: {self.conda_env}")
                    self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
                    self.species_combo.setCurrentText(self.species)
                    self.fast_mode_checkbox.setChecked(self.fast_mode)
            except: pass

    def clear_queue(self):
        self.processing_queue.clear()
        self.update_queue_label()
        self.refresh_data()

    def log_message(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
    
    def closeEvent(self, event):
        """
        Intercepts the window close event.
        If a worker is running, it blocks the close and asks for confirmation.
        """
        if self.current_worker and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self, 
                'Process Running',
                (f"A task is currently running:\n\n"
                 f"Folder: {os.path.basename(str(self.current_worker.input_path))}\n\n"
                 f"Are you sure you want to quit? This will FORCE KILL the process."),
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.log_message("Force quitting application...")
                
                # 1. Kill the subprocess
                self.current_worker.stop()
                
                # 2. Wait for thread to clean up (max 3 seconds to avoid freezing)
                self.current_worker.wait(3000)
                
                event.accept()  # Close the window
            else:
                event.ignore()  # Cancel the close, keep window open
        else:
            event.accept()

    def setup_tooltip_style(self):
        """
        Applies a homogeneous style to all QToolTips globally.
        We apply this to QApplication to ensure it overrides system defaults.
        """
        tooltip_style = """
        QToolTip {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #cccccc;
            padding: 5px;
            border-radius: 3px;
            font-size: 12px;
            font-family: sans-serif;
        }
        """
        # Access the global application instance
        app = QApplication.instance()
        if app:
            # Append to existing app styles rather than overwriting
            app.setStyleSheet(app.styleSheet() + tooltip_style)
        
def main():
    app = QApplication(sys.argv)
    window = nnUNetMonitorUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()