#!/usr/bin/env python3
import sys
import os
import subprocess
import json
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QFileDialog, 
                             QMessageBox, QTextEdit, QComboBox, QInputDialog)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import glob
from datetime import datetime

class SegmentationWorker(QThread):
    """Worker thread for running single segmentation"""
    finished = pyqtSignal(str, str)  # folder_path, message
    error = pyqtSignal(str, str)     # folder_path, error_message
    progress = pyqtSignal(str, str)  # folder_path, status_update
    
    def __init__(self, input_path, robot_name, alpha_parameter=0.9):
        super().__init__()
        self.input_path = input_path
        self.robot_name = robot_name
        self.alpha_parameter = alpha_parameter
        
    def run(self):
        try:
            folder_name = os.path.basename(self.input_path)
            self.progress.emit(self.input_path, "Starting segmentation...")
            
            # Save segmentation info before starting
            self.save_segmentation_info()
            
            # Create segmentation command
            cmd = f"""
            export nnUNet_raw="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_raw"
            export nnUNet_preprocessed="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_preprocessed"
            export nnUNet_results="/app/Segmentation/ChronoRoot_nnUNet/nnUNet_results"
            
            python name_handling.py "{self.input_path}"
            
            output_path="{self.input_path}/Segmentation/Fold_0"
            mkdir -p "$output_path"
            nnUNetv2_predict_chrono -i "{self.input_path}" -o "$output_path" -d 789 -c 2d -f 0 --save_probabilities
            python name_handling.py "{self.input_path}" --revert_seg --segpath "$output_path"
            python name_handling.py "{self.input_path}" --revert
            python ensemble_multiclass.py "{self.input_path}" --alpha {self.alpha_parameter}
            """
            
            self.progress.emit(self.input_path, "Running nnUNet...")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.update_segmentation_info_complete()
                self.finished.emit(self.input_path, f"Segmentation completed for {folder_name}")
            else:
                error_msg = f"Segmentation failed for {folder_name}: {stderr[:200]}"
                self.error.emit(self.input_path, error_msg)
                
        except Exception as e:
            self.error.emit(self.input_path, f"Error running segmentation: {str(e)}")
    
    def save_segmentation_info(self):
        """Save segmentation information to file"""
        try:
            seg_folder = os.path.join(self.input_path, 'Segmentation')
            os.makedirs(seg_folder, exist_ok=True)
            
            info_file = os.path.join(seg_folder, 'segmentation_info.json')
            info_data = {
                'alpha_parameter': self.alpha_parameter,
                'robot_name': self.robot_name,
                'start_time': datetime.now().isoformat(),
                'folder_path': self.input_path,
                'status': 'started'
            }
            
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
                
        except Exception as e:
            # Don't fail segmentation if we can't save info
            pass
    
    def update_segmentation_info_complete(self):
        """Update segmentation info to mark as complete"""
        try:
            info_file = os.path.join(self.input_path, 'Segmentation', 'segmentation_info.json')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    info_data = json.load(f)
                
                info_data['status'] = 'completed'
                info_data['completion_time'] = datetime.now().isoformat()
                
                with open(info_file, 'w') as f:
                    json.dump(info_data, f, indent=2)
        except Exception as e:
            # Don't fail segmentation if we can't update info
            pass

class nnUNetMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.robots = {}  # robot_name -> {path: str, folders: dict}
        self.folder_data = {}  # folder_path -> data dict
        self.processing_queue = []  # Simple list for serial processing
        self.current_worker = None
        self.alpha_parameter = 0.9  # Default alpha value
        self.init_ui()
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
        
        # Alpha parameter button
        self.alpha_button = QPushButton(f"Alpha: {self.alpha_parameter}")
        self.alpha_button.clicked.connect(self.set_alpha_parameter)
        self.alpha_button.setStyleSheet("background-color: #2196F3; color: white;")
        
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
        header_layout.addWidget(self.clear_queue_button)
        header_layout.addStretch()
        header_layout.addWidget(self.manual_refresh_button)
        header_layout.addWidget(self.auto_refresh_button)
        
        layout.addLayout(header_layout)
        
        # Filter section
        filter_layout = QHBoxLayout()
        
        self.robot_filter = QComboBox()
        self.robot_filter.addItem("All Robots")
        self.robot_filter.currentTextChanged.connect(self.update_table)
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Not Started", "Queued", "Processing", "Complete", "Error"])
        self.status_filter.currentTextChanged.connect(self.update_table)
        
        filter_layout.addWidget(QLabel("Filter by Robot:"))
        filter_layout.addWidget(self.robot_filter)
        filter_layout.addWidget(QLabel("Filter by Status:"))
        filter_layout.addWidget(self.status_filter)
        filter_layout.addStretch()
        
        layout.addLayout(filter_layout)
        
        # Table for folder information
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Robot", "Folder Name", "Total Images", "Segmentation %", 
            "Ensemble %", "Status", "Alpha Used", "Actions"
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
        """Set the alpha parameter for ensemble"""
        value, ok = QInputDialog.getDouble(self, 'Alpha Parameter', 
                                         'Enter alpha value for ensemble (0.0-1.0):', 
                                         self.alpha_parameter, 0.0, 1.0, 2)
        if ok:
            self.alpha_parameter = value
            self.alpha_button.setText(f"Alpha: {self.alpha_parameter}")
            self.log_message(f"Alpha parameter set to {self.alpha_parameter}")
    
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
    
    def needs_re_segmentation(self, folder_path, current_status):
        """Check if folder needs re-segmentation due to alpha change"""
        if current_status != 'Complete':
            return False
            
        info = self.load_segmentation_info(folder_path)
        if info is None:
            return True  # No info file, might need re-segmentation
            
        stored_alpha = info.get('alpha_parameter', 0.9)
        return abs(stored_alpha - self.alpha_parameter) > 0.001  # Small tolerance for float comparison
    
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
            'ensemble_progress': 0,
            'status': 'Not Started',
            'last_error': '',
            'stored_alpha': None,
            'needs_resegmentation': False,
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
            
            # Load segmentation info
            seg_info = self.load_segmentation_info(folder_path)
            if seg_info:
                data['stored_alpha'] = seg_info.get('alpha_parameter', 'Unknown')
            
            # Check if currently processing (but still check progress)
            is_currently_processing = self.current_worker and self.current_worker.input_path == folder_path
            data['is_actively_processing'] = is_currently_processing
            
            # Check segmentation progress
            seg_folder = os.path.join(folder_path, 'Segmentation')
            if os.path.exists(seg_folder):
                # Check for actual error files (only .error files, ignore warnings)
                error_files = glob.glob(os.path.join(seg_folder, '*.error'))
                if error_files:
                    # Only treat as error if it contains actual error content
                    is_real_error = False
                    error_content = ""
                    try:
                        with open(error_files[0], 'r') as f:
                            error_content = f.read()
                            # Check if it's a real error (not just warnings)
                            if any(keyword in error_content.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                                is_real_error = True
                    except:
                        is_real_error = True  # If we can't read it, assume it's an error
                    
                    if is_real_error:
                        data['status'] = 'Error'
                        data['last_error'] = error_content[:100]
                        return data
                
                # Check ensemble folder first (final step)
                ensemble_folder = os.path.join(seg_folder, 'Ensemble')
                if os.path.exists(ensemble_folder):
                    # Count ensemble files (including .png)
                    ensemble_files = 0
                    for ext in ['*.nii.gz', '*.nii', '*.npz', '*.png']:
                        ensemble_files += len(glob.glob(os.path.join(ensemble_folder, ext)))
                    
                    if ensemble_files > 0:
                        data['ensemble_progress'] = min(100, int((ensemble_files / total_images) * 100))
                        if data['ensemble_progress'] == 100:
                            data['segmentation_progress'] = 100
                            data['status'] = 'Complete'
                            # Check if needs re-segmentation due to alpha change
                            data['needs_resegmentation'] = self.needs_re_segmentation(folder_path, 'Complete')
                            return data
                        else:
                            data['status'] = 'Processing'
                
                # Check Fold_0 for segmentation files
                fold_0_path = os.path.join(seg_folder, 'Fold_0')
                if os.path.exists(fold_0_path):
                    seg_files = 0
                    for ext in ['*.nii.gz', '*.nii', '*.npz', '*.png']:
                        seg_files += len(glob.glob(os.path.join(fold_0_path, ext)))
                    
                    if seg_files > 0:
                        data['segmentation_progress'] = min(100, int((seg_files / total_images) * 100))
                        if data['segmentation_progress'] > 0:
                            data['status'] = 'Processing'
                
                # If segmentation folder exists but no files found
                if data['segmentation_progress'] == 0 and data['ensemble_progress'] == 0:
                    data['status'] = 'Processing'
            
            # Override status if currently processing (but keep the progress percentages)
            if is_currently_processing:
                data['status'] = 'Processing'
        
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
        
        # Restore selection if still valid
        index = self.robot_filter.findText(current_selection)
        if index >= 0:
            self.robot_filter.setCurrentIndex(index)
        
        # Update counts
        self.robot_count_label.setText(f"Robots: {len(self.robots)}")
    
    def update_table(self):
        """Update the table with current folder data"""
        # Apply filters
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
            
            # Robot name
            self.table.setItem(row, 0, QTableWidgetItem(data['robot']))
            
            # Folder name
            self.table.setItem(row, 1, QTableWidgetItem(folder_name))
            
            # Total images
            self.table.setItem(row, 2, QTableWidgetItem(str(data['total_images'])))
            
            # Segmentation progress
            seg_progress = QProgressBar()
            seg_progress.setValue(data['segmentation_progress'])
            seg_progress.setFormat(f"{data['segmentation_progress']}%")
            self.table.setCellWidget(row, 3, seg_progress)
            
            # Ensemble progress
            ens_progress = QProgressBar()
            ens_progress.setValue(data['ensemble_progress'])
            ens_progress.setFormat(f"{data['ensemble_progress']}%")
            self.table.setCellWidget(row, 4, ens_progress)
            
            # Status with colors
            status_item = QTableWidgetItem(data['status'])
            if data['status'] == 'Complete':
                status_item.setBackground(QColor(144, 238, 144))  # Light green
            elif data['status'] == 'Processing':
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
            if data.get('needs_resegmentation', False):
                alpha_item.setBackground(QColor(255, 255, 0))  # Yellow background
                alpha_item.setToolTip(f"Current alpha: {self.alpha_parameter}, Stored alpha: {alpha_text}")
            self.table.setItem(row, 6, alpha_item)
            
            # Action button
            if data['status'] in ['Not Started', 'Error']:
                action_button = QPushButton("Add to Queue")
                action_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot))
                self.table.setCellWidget(row, 7, action_button)
            elif data['status'] == 'Complete' and data.get('needs_resegmentation', False):
                re_segment_button = QPushButton("Re-segment")
                re_segment_button.setStyleSheet("background-color: #ff9800; color: white;")
                re_segment_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.re_segment_folder(path, robot))
                self.table.setCellWidget(row, 7, re_segment_button)
            elif data['status'] == 'Queued':
                remove_button = QPushButton("Remove")
                remove_button.clicked.connect(lambda checked, path=folder_path: self.remove_from_queue(path))
                self.table.setCellWidget(row, 7, remove_button)
            elif data['status'] == 'Processing':
                if data.get('is_actively_processing', False):
                    # Show that it's actively processing but no cancel option
                    processing_label = QLabel("Running...")
                    processing_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                    self.table.setCellWidget(row, 7, processing_label)
                else:
                    # Processing but not currently active (partial files exist)
                    action_button = QPushButton("Add to Queue")
                    action_button.clicked.connect(lambda checked, path=folder_path, robot=data['robot']: self.add_to_queue(path, robot))
                    self.table.setCellWidget(row, 7, action_button)
            else:
                self.table.setCellWidget(row, 7, QLabel("—"))
    
    def add_to_queue(self, folder_path, robot_name):
        """Add folder to processing queue"""
        folder_name = os.path.basename(folder_path)
        
        # Check if segmentation folder already exists
        seg_folder = os.path.join(folder_path, 'Segmentation')
        if os.path.exists(seg_folder):
            # Get stored alpha for information
            seg_info = self.load_segmentation_info(folder_path)
            stored_alpha = seg_info.get('alpha_parameter', 'Unknown') if seg_info else 'Unknown'
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
            
            # Delete existing segmentation folder
            if not self.delete_segmentation_folder(folder_path):
                self.log_message(f"Failed to delete existing segmentation for {folder_name}")
                return
        
        # Add to queue
        queue_item = {'path': folder_path, 'robot': robot_name}
        if queue_item not in self.processing_queue:
            self.processing_queue.append(queue_item)
            self.log_message(f"Added {folder_name} ({robot_name}) to queue")
            self.update_queue_info()
            self.refresh_data()
    
    def re_segment_folder(self, folder_path, robot_name):
        """Re-segment folder (delete existing segmentation first)"""
        folder_name = os.path.basename(folder_path)
        current_alpha = self.alpha_parameter
        
        # Get stored alpha for confirmation
        seg_info = self.load_segmentation_info(folder_path)
        stored_alpha = seg_info.get('alpha_parameter', 'Unknown') if seg_info else 'Unknown'
        
        reply = QMessageBox.question(
            self, 
            'Confirm Re-segmentation', 
            f'Re-segment "{folder_name}" due to alpha parameter change?\n\n'
            f'This will DELETE the current segmentation and restart with:\n'
            f'• Current Alpha: {current_alpha}\n'
            f'• Previous Alpha: {stored_alpha}\n\n'
            f'This action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.delete_segmentation_folder(folder_path):
                # Directly add to queue since we already confirmed deletion
                queue_item = {'path': folder_path, 'robot': robot_name}
                if queue_item not in self.processing_queue:
                    self.processing_queue.append(queue_item)
                    self.log_message(f"Re-segmentation queued for {folder_name} with alpha={current_alpha}")
                    self.update_queue_info()
                    self.refresh_data()
            else:
                self.log_message(f"Failed to prepare {folder_name} for re-segmentation")
    
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
                                       'Clear the processing queue?\n\nNote: Currently running segmentation cannot be stopped.',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.processing_queue.clear()
                self.log_message("Queue cleared (running segmentation will continue)")
                self.update_queue_info()
                self.refresh_data()
    
    def process_queue(self):
        """Process the next item in queue if not currently processing"""
        if not self.current_worker and self.processing_queue:
            next_item = self.processing_queue.pop(0)
            folder_path = next_item['path']
            robot_name = next_item['robot']
            
            self.current_worker = SegmentationWorker(folder_path, robot_name, self.alpha_parameter)
            self.current_worker.finished.connect(self.on_segmentation_finished)
            self.current_worker.error.connect(self.on_segmentation_error)
            self.current_worker.progress.connect(self.on_segmentation_progress)
            self.current_worker.start()
            
            self.log_message(f"Started processing {os.path.basename(folder_path)} ({robot_name}) with alpha={self.alpha_parameter}")
            self.update_queue_info()
            self.refresh_data()
    
    def update_queue_info(self):
        """Update queue information display"""
        queue_size = len(self.processing_queue)
        current = "None"
        if self.current_worker:
            current = os.path.basename(self.current_worker.input_path)
        
        self.queue_info_label.setText(f"Queue: {queue_size} | Processing: {current}")
    
    def on_segmentation_finished(self, folder_path, message):
        """Handle segmentation completion"""
        self.log_message(message)
        self.current_worker = None
        self.update_queue_info()
        self.refresh_data()
    
    def on_segmentation_error(self, folder_path, message):
        """Handle segmentation error"""
        self.log_message(f"ERROR: {message}")
        
        # Save error to file
        try:
            seg_folder = os.path.join(folder_path, 'Segmentation')
            os.makedirs(seg_folder, exist_ok=True)
            error_file = os.path.join(seg_folder, 'segmentation.error')
            with open(error_file, 'w') as f:
                f.write(f"{datetime.now()}: {message}")
        except:
            pass
        
        self.current_worker = None
        self.update_queue_info()
        self.refresh_data()
    
    def on_segmentation_progress(self, folder_path, status):
        """Handle segmentation progress updates"""
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
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def main():
    app = QApplication(sys.argv)
    window = nnUNetMonitorUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
