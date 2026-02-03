import glob
import os
import sys
import platform
import shutil

# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import json
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
                           QFileDialog, QGroupBox, QMessageBox, QScrollArea,
                           QTabWidget, QTableWidget, QTableWidgetItem, QMenu, QComboBox, QDialog)
from PyQt5.QtCore import Qt, QTimer, Qt
from PyQt5.QtGui import QColor, QPixmap, QIntValidator, QDoubleValidator

class GroupEntry(QWidget):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Group name input
        name_layout = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(f'Group {index+1}')
        
        layout.addWidget(QLabel(f'Group {index+1} Name:'))
        layout.addWidget(self.name_edit)
        
        # Add seed count input with label
        layout.addWidget(QLabel('Number of Seeds:'))
        self.seed_count_edit = QLineEdit()
        self.seed_count_edit.setPlaceholderText('Optional')
        self.seed_count_edit.setFixedWidth(100)  # Make it compact
        # Only allow integers to be entered
        self.seed_count_edit.setValidator(QIntValidator(0, 999999))
        layout.addWidget(self.seed_count_edit)
        
        self.delete_btn = QPushButton('Remove')
        layout.addWidget(self.delete_btn)
        
    def get_seed_count(self):
        """Return the seed count if entered, None otherwise"""
        text = self.seed_count_edit.text().strip()
        return int(text) if text else None


class AnalysisTab(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.group_entries = []
        self.initUI()
    
    def setup_project_fields(self):
        # Project directory
        proj_dir_layout = QHBoxLayout()
        self.proj_dir_edit = QLineEdit()
        self.proj_dir_edit.textChanged.connect(self.on_project_dir_changed)  # Add this connection
        proj_dir_btn = QPushButton('Browse')
        proj_dir_btn.clicked.connect(self.browse_project_dir)
        proj_dir_layout.addWidget(QLabel('Project Directory:'))
        proj_dir_layout.addWidget(self.proj_dir_edit)
        proj_dir_layout.addWidget(proj_dir_btn)
        return proj_dir_layout


    def initUI(self):
        layout = QVBoxLayout()
        
        # Project Directory Selection
        proj_group = QGroupBox('Project Settings')
        proj_layout = QVBoxLayout()
        
        # Add project directory fields using the new setup method
        proj_layout.addLayout(self.setup_project_fields())
        
        # Video path
        video_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        video_path_btn = QPushButton('Browse')
        video_path_btn.clicked.connect(self.browse_video_path)
        video_layout.addWidget(QLabel('Video Directory:'))
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(video_path_btn)
        proj_layout.addLayout(video_layout)
        
        # Analysis identifier
        identifier_layout = QHBoxLayout()
        self.identifier_edit = QLineEdit()
        self.identifier_edit.setPlaceholderText('analysis_name')
        identifier_layout.addWidget(QLabel('Analysis Identifier:'))
        identifier_layout.addWidget(self.identifier_edit)
        proj_layout.addLayout(identifier_layout)

        # Time delta field
        time_settings = QHBoxLayout()
        self.time_delta_edit = QLineEdit()
        self.time_delta_edit.setPlaceholderText('15')
        time_settings.addWidget(QLabel('Time between slices (minutes):'))
        time_settings.addWidget(self.time_delta_edit)

        # Time before pictures
        added_time = QHBoxLayout()
        self.add_time_edit = QLineEdit()
        self.add_time_edit.setPlaceholderText('0')
        added_time.addWidget(QLabel('Extra time before first picture (hours):'))
        added_time.addWidget(self.add_time_edit)
        
        # Cut down germination plot time
        germination_time_layout = QHBoxLayout()
        self.germination_time_edit = QLineEdit()
        self.germination_time_edit.setPlaceholderText('0 (Leave 0 for full duration)')
        germination_time_layout.addWidget(QLabel('End germination plot time (hours):'))
        germination_time_layout.addWidget(self.germination_time_edit)
        
        # Put them both in next to each other
        time_layout = QHBoxLayout()
        time_layout.addLayout(time_settings)
        time_layout.addLayout(added_time)
        time_layout.addLayout(germination_time_layout)
        proj_layout.addLayout(time_layout)

        # Calibration Group Box
        calib_group = QGroupBox('Calibration Settings')
        calib_layout = QHBoxLayout()

        # QR Code toggle and calibration options
        self.qr_checkbox = QCheckBox('Video has QR codes for calibration')
        self.qr_checkbox.stateChanged.connect(self.toggle_calibration_mode)
        calib_layout.addWidget(self.qr_checkbox)

        # Manual calibration widget (hidden by default)
        self.manual_calib_widget = QWidget()
        manual_calib_layout = QHBoxLayout()
        
        # Calibration helper button
        self.calibrate_btn = QPushButton('Open Calibration Helper')
        self.calibrate_btn.clicked.connect(self.open_calibration_helper)
        manual_calib_layout.addWidget(self.calibrate_btn)
        
        # Known distance input
        known_dist_layout = QHBoxLayout()
        self.known_dist_edit = QLineEdit()
        self.known_dist_edit.setPlaceholderText('10')
        self.known_dist_edit.setValidator(QDoubleValidator(0.0, 1000.0, 2))
        known_dist_layout.addWidget(QLabel('Known distance (mm):'))
        known_dist_layout.addWidget(self.known_dist_edit)
        manual_calib_layout.addLayout(known_dist_layout)

        # Pixel distance input
        pixel_dist_layout = QHBoxLayout()
        self.pixel_dist_edit = QLineEdit()
        self.pixel_dist_edit.setPlaceholderText('240')
        self.pixel_dist_edit.setValidator(QIntValidator(1, 10000))
        pixel_dist_layout.addWidget(QLabel('Corresponding pixels:'))
        pixel_dist_layout.addWidget(self.pixel_dist_edit)
        manual_calib_layout.addLayout(pixel_dist_layout)

        self.manual_calib_widget.setLayout(manual_calib_layout)
        calib_layout.addWidget(self.manual_calib_widget)
        calib_group.setLayout(calib_layout)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        layout.addWidget(calib_group)
        
        # Process customization options
        process_group = QGroupBox('Processing Options')
        process_vertical_layout = QVBoxLayout()
        
        process_layout = QHBoxLayout()
        
        # 1 Do germination analysis
        self.germination_checkbox = QCheckBox('Perform germination analysis')
        self.germination_checkbox.setChecked(True)
        process_layout.addWidget(self.germination_checkbox)
        
        # 2 Perform plant growth analysis
        self.plant_growth_checkbox = QCheckBox('Perform plant growth analysis')
        self.plant_growth_checkbox.setChecked(True)
        process_layout.addWidget(self.plant_growth_checkbox)
        self.plant_growth_checkbox.stateChanged.connect(self.toggle_plant_growth_options)
        
        # 3 Store tracking visualization
        self.show_tracking_checkbox = QCheckBox('Store tracking visualization')
        self.show_tracking_checkbox.setChecked(False)
        process_layout.addWidget(self.show_tracking_checkbox)
        
        # 4 Store germination for each video
        self.store_each_video_checkbox = QCheckBox('Store germination plots for each video separately')
        self.store_each_video_checkbox.setChecked(False)
        process_layout.addWidget(self.store_each_video_checkbox)
        
        process_vertical_layout.addLayout(process_layout)
        
        self.plant_growth_widget = QWidget()
        plant_process_layout = QHBoxLayout() # Use VBox to stack the rows
        
        # Metric selection row
        self.check_hypocotyl = QCheckBox('Hypocotyl Length')
        self.check_main_root = QCheckBox('Main Root Length')
        self.check_total_root = QCheckBox('Total Root Length')
        self.check_plant_area = QCheckBox('Plant Area')
        self.check_root_area = QCheckBox('Dense Root Area')
        
        for cb in [self.check_hypocotyl, self.check_main_root, self.check_total_root, 
                   self.check_plant_area, self.check_root_area]:
            cb.setChecked(True)
            plant_process_layout.addWidget(cb)
        
        # FPCA row
        self.fpca_checkbox = QCheckBox('Perform FPCA analysis')
        plant_process_layout.addWidget(self.fpca_checkbox)
        plant_process_layout.addWidget(QLabel('Components:'))
        self.fpca_components_edit = QLineEdit("2")
        self.fpca_components_edit.setFixedWidth(40)
        self.fpca_components_edit.setValidator(QIntValidator(2, 10))
        plant_process_layout.addWidget(self.fpca_components_edit)
        self.fpca_normalize_checkbox = QCheckBox('Normalize FPCA data')
        self.fpca_normalize_checkbox.setChecked(False)
        plant_process_layout.addWidget(self.fpca_normalize_checkbox)
        
        self.plant_growth_widget.setLayout(plant_process_layout)
        
        process_vertical_layout.addWidget(self.plant_growth_widget)
        
        process_group.setLayout(process_vertical_layout)     
        self.plant_growth_widget.setLayout(plant_process_layout)
        layout.addWidget(process_group)
        
        # Group Names
        group_group = QGroupBox('Group Names')
        self.group_layout = QVBoxLayout()
        
        # Scroll area for groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.group_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        
        # Add Group button
        self.add_group_btn = QPushButton('Add Group')
        self.add_group_btn.clicked.connect(self.add_group)
        self.group_layout.addWidget(self.add_group_btn)
        
        group_group.setLayout(QVBoxLayout())
        group_group.layout().addWidget(scroll)
        layout.addWidget(group_group)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Preview Video Button
        self.preview_btn = QPushButton('Preview Video')
        self.preview_btn.clicked.connect(self.preview_video)
        buttons_layout.addWidget(self.preview_btn)
        
        # Process Button
        self.process_btn = QPushButton('Process Video')
        self.process_btn.clicked.connect(self.process_video)
        buttons_layout.addWidget(self.process_btn)

        # Generate Report Button
        self.generate_report_btn = QPushButton('Generate Report')
        self.generate_report_btn.clicked.connect(self.generate_report)
        buttons_layout.addWidget(self.generate_report_btn)
        
        # Add name mapping button to the buttons_layout
        self.name_mapping_btn = QPushButton('Edit Name Mapping')
        self.name_mapping_btn.clicked.connect(self.edit_name_mapping)
        buttons_layout.addWidget(self.name_mapping_btn)

        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # Add initial groups
        for _ in range(3):
            self.add_group()

        # Initialize calibration mode
        self.toggle_calibration_mode()
        self.toggle_plant_growth_options()

    def edit_name_mapping(self):
        """Open dialog to edit name mapping for visualization"""
        if not self.proj_dir_edit.text() or not os.path.exists(self.proj_dir_edit.text()):
            QMessageBox.warning(self, 'Error', 'Please select a valid project directory first!')
            return
            
        dialog = NameMappingDialog(self.proj_dir_edit.text(), self)
        dialog.exec_()

    def on_project_dir_changed(self):
        """Handle project directory changes and update all tabs"""
        project_dir = self.proj_dir_edit.text()
        if os.path.exists(project_dir):
            self.main_window.set_project_dir(project_dir)

    def toggle_calibration_mode(self):
        """Toggle between QR and manual calibration modes"""
        has_qr = self.qr_checkbox.isChecked()
        self.manual_calib_widget.setVisible(not has_qr)
    
    def toggle_plant_growth_options(self):
        """Enable or disable plant growth analysis options"""
        enabled = self.plant_growth_checkbox.isChecked()
        self.plant_growth_widget.setVisible(enabled)

    def open_calibration_helper(self):
        """Opens a helper window to assist with manual calibration"""
        if not self.video_path_edit.text():
            QMessageBox.warning(self, 'Error', 'Please select a video directory first!')
            return

        # Create command line arguments
        args = [
            "python",
            "calibration_helper.py",
            "--video-dir", self.video_path_edit.text()
        ]

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            QMessageBox.information(
                self,
                "Calibration Helper",
                "Calibration helper window has been opened.\n"
                "Please measure the pixel distance between two points\n"
                "of known physical distance in your image."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting calibration helper: {str(e)}")

    def validate_inputs(self):
        """Updated validation to include calibration checks"""
        if not self.basic_validation():
            return False

        # Validate calibration settings
        if not self.qr_checkbox.isChecked():
            if not self.known_dist_edit.text() or not self.pixel_dist_edit.text():
                QMessageBox.warning(self, 'Error', 'Please provide both known distance and pixel distance for manual calibration!')
                return False
            try:
                known_dist = float(self.known_dist_edit.text())
                pixel_dist = int(self.pixel_dist_edit.text())
                if known_dist <= 0 or pixel_dist <= 0:
                    QMessageBox.warning(self, 'Error', 'Calibration values must be positive numbers!')
                    return False
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid calibration values!')
                return False

        return True

    def basic_validation(self):
        """Original validation logic moved to separate method"""
        if not os.path.exists(self.proj_dir_edit.text()):
            QMessageBox.warning(self, 'Error', 'Project directory does not exist!')
            return False
            
        if not os.path.exists(self.video_path_edit.text()):
            QMessageBox.warning(self, 'Error', 'Video directory does not exist!')
            return False
            
        identifier = self.identifier_edit.text().strip()
        if not identifier:
            QMessageBox.warning(self, 'Error', 'Please provide an analysis identifier!')
            return False
            
        if not identifier.replace('_', '').isalnum():
            QMessageBox.warning(self, 'Error', 'Identifier must contain only letters, numbers, and underscores!')
            return False
            
        analysis_dir = os.path.join(self.proj_dir_edit.text(), 'analysis', identifier)
        if os.path.exists(analysis_dir):
            QMessageBox.warning(self, 'Error', 'Analysis with this identifier already exists!')
            return False
            
        group_names = [entry.name_edit.text().strip() for entry in self.group_entries]
        if not all(group_names):
            QMessageBox.warning(self, 'Error', 'All group names must be filled!')
            return False
            
        if len(set(group_names)) != len(group_names):
            QMessageBox.warning(self, 'Error', 'Group names must be unique!')
            return False
            
        return True

    def add_group(self):
        group_entry = GroupEntry(len(self.group_entries))
        group_entry.delete_btn.clicked.connect(lambda: self.remove_group(group_entry))
        self.group_entries.append(group_entry)
        self.group_layout.insertWidget(len(self.group_entries)-1, group_entry)
        
    def remove_group(self, group_entry):
        if len(self.group_entries) > 1:
            self.group_entries.remove(group_entry)
            group_entry.deleteLater()
            for i, entry in enumerate(self.group_entries):
                entry.index = i
        else:
            QMessageBox.warning(self, 'Warning', 'At least one group is required!')
            
    def browse_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Project Directory')
        if dir_path:
            self.proj_dir_edit.setText(dir_path)
            # Update the main window's project directory
            self.main_window.set_project_dir(dir_path)  # This will update all tabs
            
    def browse_video_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Video Directory')
        if dir_path:
            self.video_path_edit.setText(dir_path)
            
    def process_video(self):
        if not self.validate_inputs():
            return
                
        # Create analysis directory structure
        project_dir = self.proj_dir_edit.text()
        analysis_base_dir = os.path.join(project_dir, 'analysis')
        os.makedirs(analysis_base_dir, exist_ok=True)
        
        identifier = self.identifier_edit.text().strip()
        
        # Get time delta, default to 15 if empty or invalid
        try:
            time_delta = float(self.time_delta_edit.text() or '15')
        except ValueError:
            time_delta = 15
            
        video_folder = self.video_path_edit.text()
        segmentation_dir = os.path.join(video_folder, 'Segmentation')
        
        # Check for PNG images
        images = glob.glob(os.path.join(video_folder, "*.png"))
        
        # Check if there is no images, then look for a file called "segmentation_metadata.json"
        if not images:
            metadata_path = os.path.join(video_folder, 'Segmentation', 'segmentation_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                images = glob.glob(os.path.join(metadata["input_path"], "*.png")) 
                video_folder = metadata["input_path"]

        if not images:
            QMessageBox.warning(
                None, 'Error', 
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return
        
        # Check for segmentation files in Segmentation/Ensemble folder
        seg_folder = os.path.join(segmentation_dir, "Ensemble")
        seg_files = glob.glob(os.path.join(seg_folder, "*.png")) if os.path.exists(seg_folder) else []
        
        if not seg_files:
            # Images exist but no segmentation found
            QMessageBox.warning(
                None, 'Error',
                f'Found {len(images)} images but no segmentation files!\n The images may not have been properly segmented.'
            )
            return
        
        # Collect parameters including seed counts
        params = {
            'project_dir': project_dir,
            'video_dir': video_folder,
            'segmentation_dir': segmentation_dir,
            'analysis_id': identifier,
            'has_qr': self.qr_checkbox.isChecked(),
            'show_tracking': self.show_tracking_checkbox.isChecked(),
            'time_delta': time_delta,
            'group_names': [entry.name_edit.text().strip() for entry in self.group_entries],
            'seed_counts': [entry.get_seed_count() for entry in self.group_entries]
        }
        
        # Create command line arguments
        args = [
            "python",
            "process_video.py",
            "--video-dir", params['video_dir'],
            "--segmentation-dir", params['segmentation_dir'],
            "--project-dir", params['project_dir'],
            "--analysis-id", params['analysis_id'],
            "--time-delta", str(params['time_delta'])
        ]

        # Add calibration parameters
        if params['has_qr']:
            args.append("--has-qr")
        else:
            # Get manual calibration values
            try:
                known_dist = float(self.known_dist_edit.text())
                pixel_dist = int(self.pixel_dist_edit.text())
                args.extend([
                    "--known-distance", str(known_dist),
                    "--pixel-distance", str(pixel_dist)
                ])
            except (ValueError, AttributeError) as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    "Please provide valid calibration values:\n"
                    "- Known distance (mm) as a decimal number\n"
                    "- Pixel distance as a whole number\n"
                    "Or enable QR code calibration."
                )
                return

        # Add optional flags
        if params['show_tracking']:
            args.append("--show-tracking")

        # Add group names and their corresponding seed counts
        group_info = []
        for name, count in zip(params['group_names'], params['seed_counts']):
            group_info.append(name)
            if count is not None:
                group_info.append(str(count))
            else:
                group_info.append("0")  # Use 0 to indicate no count provided
                
        args.extend(["--group-info"] + group_info)

        try:
            # Launch the processing script
            process = subprocess.Popen(
                " ".join(args),
                shell=True,
                preexec_fn=os.setsid
            )
            
            QMessageBox.information(
                self,
                "Processing Started",
                f"Video processing has been started in a separate window.\n"
                f"Results will be saved in: {os.path.join(analysis_base_dir, identifier)}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting processing: {str(e)}")

    def preview_video(self):
        
        # Get time delta, default to 15 if empty or invalid
        try:
            time_delta = float(self.time_delta_edit.text() or '15')
        except ValueError:
            time_delta = 15

        video_folder = self.video_path_edit.text()
        segmentation_dir = os.path.join(video_folder, 'Segmentation')
        
        # Check for PNG images
        images = glob.glob(os.path.join(video_folder, "*.png"))
        
        # Check if there is no images, then look for a file called "segmentation_metadata.json"
        if not images:
            metadata_path = os.path.join(video_folder, 'Segmentation', 'segmentation_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                images = glob.glob(os.path.join(metadata["input_path"], "*.png")) 
                video_folder = metadata["input_path"]

        if not images:
            QMessageBox.warning(
                None, 'Error', 
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return
        
        # Check for segmentation files in Segmentation/Ensemble folder
        seg_folder = os.path.join(segmentation_dir, "Ensemble")
        seg_files = glob.glob(os.path.join(seg_folder, "*.png")) if os.path.exists(seg_folder) else []
        
        if not seg_files:
            # Images exist but no segmentation found
            QMessageBox.warning(
                None, 'Error',
                f'Found {len(images)} images but no segmentation files!\n The images may not have been properly segmented.'
            )
            return
        
        # Collect parameters
        params = {
            'video_dir': video_folder,
            'segmentation_dir': segmentation_dir,
            'time_delta': time_delta
        }
        
        
        # Create command line arguments
        args = [
            "python",
            "preview_video.py",
            "--video-dir", params['video_dir'],
            "--segmentation-dir", params['segmentation_dir'],
            "--time-delta", str(params['time_delta'])
        ]

        try:
            # Launch the processing script
            process = subprocess.Popen(
                " ".join(args),
                shell=True,
                preexec_fn=os.setsid
            )
            
            QMessageBox.information(
                self,
                "Preview Started",
                f"Video preview has been started in a separate window.\n"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting processing: {str(e)}")
    
        return


    def generate_report(self):
        """
        Generate report on all completed experiments.
        """
        if not self.proj_dir_edit.text():
            QMessageBox.warning(self, 'Error', 'Please select a project directory first!')
            return
            
        project_dir = self.proj_dir_edit.text()
        
        # Get time delta, default to 15 if empty or invalid
        try:
            time_delta = float(self.time_delta_edit.text() or '15')
        except ValueError:
            time_delta = 15

        # Get add time, default 0 if empty or invalid
        try:
            add_time = int(self.add_time_edit.text() or '0')
        except ValueError:
            add_time = 0
        
        # Verify there are completed analyses
        analysis_dir = os.path.join(project_dir, 'analysis')
        if not os.path.exists(analysis_dir):
            QMessageBox.warning(self, 'Error', 'No analysis directory found!')
            return
            
        analyses = [d for d in os.listdir(analysis_dir) 
                if os.path.isdir(os.path.join(analysis_dir, d))]
        
        if not analyses:
            QMessageBox.warning(self, 'Error', 'No analyses found to process!')
            return
        
        # Check for name mapping file
        mapping_file = os.path.join(project_dir, 'name_mapping.json')
        
        # 1. Collect Active Metrics/Parts
        active_parts = []
        mapping = [
            ("HypocotylLength", self.check_hypocotyl),
            ("MainRootLength", self.check_main_root),
            ("TotalRootLength", self.check_total_root),
            ("Area", self.check_plant_area),
            ("DenseRootArea", self.check_root_area)
        ]
        for name, cb in mapping:
            if cb.isChecked():
                active_parts.append(name)        
        
        # 2. Collect FPCA Settings
        do_fpca = self.fpca_checkbox.isChecked()
        fpca_comps = self.fpca_components_edit.text() or "2"
        
        # 3. Handle Germination Cutoff
        germ_cut = self.germination_time_edit.text() or "0"

        # 4. Construct Command Line Arguments
        args = [
            "python", "generate_report.py",
            "--project-dir", project_dir,
            "--dt", str(time_delta),
            "--add-time-before-photo", str(add_time),
            "--germination-time-cut", str(germ_cut),
            "--do-germination", str(self.germination_checkbox.isChecked()),
            "--germination-each-video", str(self.store_each_video_checkbox.isChecked()),
            "--do-plant-growth", str(self.plant_growth_checkbox.isChecked()),
            "--selected-metrics", ",".join(active_parts),
            "--do-fpca", str(do_fpca),
            "--fpca-components", fpca_comps,
            "--normalize-fpca", str(self.fpca_normalize_checkbox.isChecked())
        ]
        
        # Add name mapping if it exists
        if os.path.exists(mapping_file):
            args.extend(["--name-mapping", mapping_file])
        
        try:
            # Launch the processing script as a separate process
            process = subprocess.Popen(
                " ".join(args),
                shell=True,
                preexec_fn=os.setsid
            )
            
            mapping_msg = " with name mapping" if os.path.exists(mapping_file) else ""
            QMessageBox.information(
                self,
                "Report Generation Started",
                f"Report generation has been started{mapping_msg}.\n"
                f"Results will be saved in: {os.path.join(project_dir, 'results')}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Error starting report generation: {str(e)}"
            )

class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.sort_column = 0  # Default sort column
        self.sort_order = Qt.AscendingOrder
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_results)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Control buttons layout
        control_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_btn = QPushButton('Refresh Results')
        self.refresh_btn.clicked.connect(self.refresh_results)
        control_layout.addWidget(self.refresh_btn)
        
        # Auto-refresh toggle
        self.auto_refresh_btn = QPushButton('Auto Refresh: Off')
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.clicked.connect(self.toggle_auto_refresh)
        control_layout.addWidget(self.auto_refresh_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            'Analysis ID', 'Groups', 'Num Groups', 'Start Time', 'Completion Time', 'Status'
        ])
        
        # Enable sorting
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().sortIndicatorChanged.connect(self.on_sort_changed)
        
        # Enable selection of entire rows
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        
        # Enable context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        
        # Make headers stretch
        self.table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.table)
        self.setLayout(layout)
        
    def toggle_auto_refresh(self):
        if self.auto_refresh_btn.isChecked():
            self.auto_refresh_btn.setText('Auto Refresh: On')
            self.timer.start(5000)  # Refresh every 5 seconds
        else:
            self.auto_refresh_btn.setText('Auto Refresh: Off')
            self.timer.stop()
    
    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None:
            return
            
        menu = QMenu(self)
        
        # Add actions
        open_folder_action = menu.addAction("Open Results Folder")
        view_metadata_action = menu.addAction("View Metadata")
        
        # Show menu and get selected action
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        
        if action == open_folder_action:
            self.open_results_folder(item.row())
        elif action == view_metadata_action:
            self.view_metadata(item.row())
            
    def open_results_folder(self, row):
        analysis_id = self.table.item(row, 0).text()
        folder_path = os.path.join(self.project_dir, 'analysis', analysis_id)
        
        if os.path.exists(folder_path):
            # Open folder in file explorer
            try:
                path = os.path.abspath(os.path.expanduser(folder_path))
                is_container = any(k in os.environ for k in ['APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER'])
                
                # --- STRATEGY 1: D-Bus ---
                if is_container and shutil.which("dbus-send"):
                    try:
                        subprocess.run([
                            "dbus-send", "--session", "--dest=org.freedesktop.FileManager1",
                            "--type=method_call", "/org/freedesktop/FileManager1",
                            "org.freedesktop.FileManager1.ShowItems", 
                            f"array:string:file://{path}", "string:''"
                        ], timeout=2, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                        return
                    except:
                        pass

                # --- STRATEGY 2: Standard Openers ---
                cmd = None
                if platform.system() == "Darwin":
                    cmd = "open"
                elif platform.system() == "Windows":
                    os.startfile(path)
                    return
                else:
                    cmd = "xdg-open"
                    
                if cmd and shutil.which(cmd):
                    subprocess.Popen([cmd, path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    return
                else:
                    print(f"Error opening folder: No suitable opener found for {path}")

            except Exception as e:
                # Final safety net to prevent app crash
                print(f"Error opening folder: {e}")
        
    def view_metadata(self, row):
        analysis_id = self.table.item(row, 0).text()
        metadata_path = os.path.join(self.project_dir, 'analysis', analysis_id, 'metadata.json')
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Show metadata in a message box
                msg = QMessageBox()
                msg.setWindowTitle(f"Metadata - {analysis_id}")
                msg.setText("\n".join([f"{k}: {v}" for k, v in metadata.items()]))
                msg.exec_()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error reading metadata: {str(e)}")
    
    def on_sort_changed(self, logical_index, order):
        self.sort_column = logical_index
        self.sort_order = order
        self.refresh_results()
        
    def set_project_dir(self, dir_path):
        self.project_dir = dir_path
        self.refresh_results()
        
    def create_table_item(self, text, color=None):
        item = QTableWidgetItem(text)
        if color:
            item.setBackground(QColor(color))
        return item
        
    def refresh_results(self):
        if not self.project_dir:
            return
            
        analysis_dir = os.path.join(self.project_dir, 'analysis')
        if not os.path.exists(analysis_dir):
            return
            
        # Store current sort state
        sort_column = self.table.horizontalHeader().sortIndicatorSection()
        sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        
        # Temporarily disable sorting to improve performance
        self.table.setSortingEnabled(False)
        
        # Clear current table
        self.table.setRowCount(0)
        
        # Get all analysis folders
        analyses = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d))]
        
        self.table.setRowCount(len(analyses))
        
        for row, analysis_id in enumerate(analyses):
            analysis_path = os.path.join(analysis_dir, analysis_id)
            metadata_path = os.path.join(analysis_path, 'metadata.json')
            
            # Initialize empty row
            for col in range(self.table.columnCount()):
                self.table.setItem(row, col, self.create_table_item(""))
                
            # Set analysis ID
            self.table.setItem(row, 0, self.create_table_item(analysis_id))
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Fill table row with color coding
                    self.table.setItem(row, 0, self.create_table_item(metadata['analysis_id']))
                    self.table.setItem(row, 1, self.create_table_item(', '.join(metadata['group_names'])))
                    self.table.setItem(row, 2, self.create_table_item(str(metadata['num_groups'])))
                    
                    # Use start_time from metadata
                    if 'start_time' in metadata:
                        self.table.setItem(row, 3, self.create_table_item(metadata['start_time']))
                    else:
                        # Fallback to creation time if metadata is from old version
                        start_time = datetime.fromtimestamp(os.path.getctime(analysis_path)).strftime("%Y-%m-%d %H:%M:%S")
                        self.table.setItem(row, 3, self.create_table_item(start_time))
                    
                    if metadata.get('status') == 'Complete':
                        self.table.setItem(row, 4, self.create_table_item(metadata['completion_time']))
                        self.table.setItem(row, 5, self.create_table_item("Complete", "#90EE90"))  # Light green
                    elif metadata.get('status') == 'In Progress':
                        self.table.setItem(row, 4, self.create_table_item("--"))
                        self.table.setItem(row, 5, self.create_table_item("In Progress", "#FFF68F"))  # Light yellow
                    else:
                        self.table.setItem(row, 4, self.create_table_item("--"))
                        self.table.setItem(row, 5, self.create_table_item("Unknown", "#FFB6C1"))  # Light red
                        
                except Exception as e:
                    self.table.setItem(row, 5, self.create_table_item("Error", "#FFB6C1"))  # Light red
            else:
                # No metadata file exists
                self.table.setItem(row, 5, self.create_table_item("No Metadata", "#FFB6C1"))  # Light red
        
        # Re-enable sorting
        self.table.setSortingEnabled(True)
        
        # Restore sort state
        self.table.sortItems(sort_column, sort_order)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()

class ReportsTab(QWidget):
    PARAMETERS = [
        "Germination",
        "Area",
        "DenseRootArea",
        "HypocotylLength", 
        "MainRootLength", 
        "TotalRootLength"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_dir = None
        self.current_image_index = 0
        self.image_paths = []
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Parameter selector
        self.parameter_selector = QComboBox()
        self.parameter_selector.addItems(self.PARAMETERS)
        controls_layout.addWidget(QLabel('Parameter:'))
        controls_layout.addWidget(self.parameter_selector)

        # Connect parameter selector
        self.parameter_selector.currentTextChanged.connect(self.update_plot_types)

        # Plot type selector (will be populated dynamically)
        self.plot_type = QComboBox()
        self.plot_type.currentTextChanged.connect(self.update_image_list)
        controls_layout.addWidget(QLabel('Plot Type:'))
        controls_layout.addWidget(self.plot_type)

        # Image selector
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self.display_selected_image)
        controls_layout.addWidget(QLabel('Select Image:'))
        controls_layout.addWidget(self.image_selector)
        
        # Navigation buttons
        self.prev_btn = QPushButton('←')
        self.next_btn = QPushButton('→')
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.prev_btn.setFixedWidth(40)
        self.next_btn.setFixedWidth(40)
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.next_btn)
        
        # Refresh button
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.clicked.connect(self.refresh_images)
        controls_layout.addWidget(self.refresh_btn)
        
        # Current path label for debugging
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        controls_layout.addWidget(self.path_label)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        layout.addWidget(self.scroll_area)
        
        self.setLayout(layout)
        self.update_button_states()
        
    def set_project_dir(self, dir_path):
        self.project_dir = dir_path
        self.refresh_images()
        
    def get_image_folder(self):
        if not self.project_dir:
            print("No project directory set")
            return None
            
        results_dir = os.path.join(self.project_dir, 'results')
        if not os.path.exists(results_dir):
            return None
            
        current_parameter = self.parameter_selector.currentText()
        current_plot_type = self.plot_type.currentText()
        
        if not current_parameter or not current_plot_type:
            return None
            
        image_folder = os.path.join(
            results_dir,
            current_parameter,
            current_plot_type
        )
        
        return image_folder if os.path.exists(image_folder) else None

    def update_plot_types(self):
        if not self.project_dir:
            return
                
        current_parameter = self.parameter_selector.currentText()
        if not current_parameter:
            return
                
        parameter_dir = os.path.join(self.project_dir, 'results', current_parameter)
        
        plot_types = []
        if os.path.exists(parameter_dir):
            for folder in os.listdir(parameter_dir):
                folder_path = os.path.join(parameter_dir, folder)
                if os.path.isdir(folder_path) and not folder.endswith('ProcessedData'):
                    plot_types.append(folder)
        
        # Block signals to prevent multiple redundant image updates
        self.plot_type.blockSignals(True)
        self.plot_type.clear()
        self.plot_type.addItems(plot_types)
        self.plot_type.blockSignals(False)
        
        if plot_types:
            # Default to the first one if current selection is gone
            self.plot_type.setCurrentIndex(0)
        
        # Manually trigger image list update
        self.update_image_list()

    def update_image_list(self):
        self.image_paths = []
        image_folder = self.get_image_folder()
        
        if image_folder and os.path.exists(image_folder):
            for f in os.listdir(image_folder):
                if f.lower().endswith('.png'):
                    full_path = os.path.join(image_folder, f)
                    self.image_paths.append(full_path)
            
            self.image_paths.sort()
        
        # Update image selector
        self.image_selector.blockSignals(True)
        self.image_selector.clear()
        if self.image_paths:
            self.image_selector.addItems([os.path.basename(p) for p in self.image_paths])
            self.image_selector.blockSignals(False)
            self.image_selector.setCurrentIndex(0) # Force selection of first image
        else:
            self.image_selector.blockSignals(False)
            self.image_label.setText("No images found in the selected directory")
            self.image_label.setPixmap(QPixmap()) # Clear old image
        
        self.current_image_index = 0
        self.display_current_image()
        self.update_button_states()
        
    def display_selected_image(self):
        self.current_image_index = self.image_selector.currentIndex()
        self.display_current_image()
        self.update_button_states()
        
    def display_current_image(self):
        if not self.image_paths or self.current_image_index < 0:
            self.image_label.setText("No images available")
            self.path_label.setText("No image selected")
            return
            
        if self.current_image_index >= len(self.image_paths):
            self.current_image_index = len(self.image_paths) - 1
            
        image_path = self.image_paths[self.current_image_index]
        self.path_label.setText(f"Current image: {image_path}")
                    
        # Load and display image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale image to fit the window while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.scroll_area.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            error_msg = f"Error loading image: {image_path}"
            self.image_label.setText(error_msg)
            
    def show_previous(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_selector.setCurrentIndex(self.current_image_index)
            
    def show_next(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.image_selector.setCurrentIndex(self.current_image_index)
            
    def update_button_states(self):
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < len(self.image_paths) - 1)
        
    def refresh_images(self):
        if not self.project_dir:
            return
        self.update_plot_types() 
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_current_image()


class NameMappingDialog(QDialog):
    def __init__(self, project_dir, parent=None):
        super().__init__(parent)
        self.project_dir = project_dir
        self.mapping_file = os.path.join(project_dir, 'name_mapping.json')
        self.mapping = {}
        self.initUI()
        self.load_existing_mapping()
        self.load_group_names()
        
    def initUI(self):
        self.setWindowTitle('Group Name Mapping')
        self.setMinimumWidth(500)
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "Map original group names to display names for visualization. "
            "Leave blank to use original name."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Scrollable area for mappings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.mapping_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Buttons
        buttons = QHBoxLayout()
        self.save_btn = QPushButton('Save Mapping')
        self.save_btn.clicked.connect(self.save_mapping)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)
        
        self.setLayout(layout)
    
    def load_existing_mapping(self):
        # Load existing mapping if any
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
            except:
                self.mapping = {}
    
    def load_group_names(self):
        # Find all unique group names across analyses
        group_names = set()
        analysis_dir = os.path.join(self.project_dir, 'analysis')
        
        if os.path.exists(analysis_dir):
            for analysis_id in os.listdir(analysis_dir):
                group_info_path = os.path.join(analysis_dir, analysis_id, 'group_info.json')
                if os.path.exists(group_info_path):
                    try:
                        with open(group_info_path, 'r') as f:
                            group_info = json.load(f)
                            if 'group_names' in group_info:
                                for name in group_info['group_names']:
                                    group_names.add(str(name).strip())
                    except:
                        pass
        
        # Clear existing layout
        while self.mapping_layout.count():
            item = self.mapping_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create mapping entries
        for name in sorted(group_names):
            row = QHBoxLayout()
            row.addWidget(QLabel(f'Original: {name}'))
            edit = QLineEdit()
            edit.setPlaceholderText(f'Display name for {name}')
            
            # Set existing mapping if any
            if name in self.mapping:
                edit.setText(self.mapping[name])
                
            row.addWidget(edit)
            row.addWidget(QLabel())  # Spacer
            
            # Store the widgets for later retrieval
            setattr(self, f'edit_{name}', edit)
            
            self.mapping_layout.addLayout(row)
            
        # Add stretch to bottom
        self.mapping_layout.addStretch()
    
    def save_mapping(self):
        # Collect mappings from UI
        new_mapping = {}
        for name in self.mapping.keys():
            edit = getattr(self, f'edit_{name}', None)
            if edit and edit.text().strip():
                new_mapping[name] = edit.text().strip()
        
        # Find any new mappings we added during this session
        for child in self.findChildren(QLineEdit):
            if child.text().strip():
                # Extract the original name from placeholder text
                placeholder = child.placeholderText()
                if placeholder.startswith('Display name for '):
                    orig_name = placeholder[17:]  # Length of 'Display name for '
                    new_mapping[orig_name] = child.text().strip()
        
        # Save mapping to file
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(new_mapping, f, indent=2)
            self.mapping = new_mapping
            QMessageBox.information(self, 'Success', 'Name mapping saved successfully!')
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error saving mapping: {str(e)}')

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import subprocess

class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set the background color to white
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Logo Logic (Pillow for high quality)
        self.logo_label = QLabel()
        ico_path = "../logo.ico"
        try:
            with Image.open(ico_path) as img:
                img = img.convert("RGBA").resize((200, 200), Image.Resampling.LANCZOS)
                data = img.tobytes("raw", "RGBA")
                qimg = QImage(data, img.size[0], img.size[1], QImage.Format_RGBA8888)
                self.logo_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            self.logo_label.setPixmap(QPixmap(ico_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.logo_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)

        # Title
        title = QLabel("ChronoRoot")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; background-color: transparent;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        # Short Description
        description = QLabel("An open-source platform for high-throughput phenotyping of plant root systems.")
        description.setStyleSheet("font-size: 14px; color: #34495e; background-color: transparent; margin-bottom: 5px;")
        layout.addWidget(description, alignment=Qt.AlignCenter)

        # Website Link
        web_link = QLabel('<a href="https://chronoroot.github.io/">https://chronoroot.github.io/</a>')
        web_link.setOpenExternalLinks(True)
        web_link.setStyleSheet("font-size: 13px; background-color: transparent; margin-bottom: 20px;")
        layout.addWidget(web_link, alignment=Qt.AlignCenter)

        # Update Button
        self.update_btn = QPushButton("Check for Updates")
        self.update_btn.setFixedWidth(250)
        self.update_btn.setCursor(Qt.PointingHandCursor)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; border-radius: 5px;
                padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.update_btn.clicked.connect(self.update_software)
        layout.addWidget(self.update_btn, alignment=Qt.AlignCenter)

        # Last Commit Info
        self.commit_label = QLabel(f"Last update: {self.get_last_commit_time()}")
        self.commit_label.setStyleSheet("color: #95a5a6; background-color: transparent; margin-top: 15px;")
        layout.addWidget(self.commit_label, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def get_git_hash(self):
        """Returns the current git commit hash (language independent)."""
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except:
            return None

    def get_last_commit_time(self):
        """Fetches the ISO date of the last local git commit."""
        try:
            cmd = ["git", "log", "-1", "--format=%cd", "--date=short"]
            return subprocess.check_output(cmd).decode().strip()
        except:
            return "Unknown"

    def update_software(self):
        """Performs a git pull and compares hashes to detect updates."""
        try:
            self.update_btn.setText("Checking...")
            self.update_btn.setEnabled(False)
            QApplication.processEvents()

            old_hash = self.get_git_hash()
            subprocess.check_call(["git", "pull"], stderr=subprocess.STDOUT)
            new_hash = self.get_git_hash()

            if old_hash == new_hash:
                QMessageBox.information(self, "Update", "ChronoRoot is already up to date!")
            else:
                QMessageBox.information(self, "Update Success", 
                    "Update downloaded successfully!\nPlease restart the application to apply changes.")
                self.commit_label.setText(f"Last update: {self.get_last_commit_time()}")

        except Exception:
            QMessageBox.critical(self, "Update Error", 
                "Failed to update. Please check your internet connection or git installation.")
        
        finally:
            self.update_btn.setText("Check for Updates")
            self.update_btn.setEnabled(True)

        
class ScreeningGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_dir = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('ChronoRoot Screening Interface')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create and store tabs as instance variables
        self.analysis_tab = AnalysisTab(self)  # passing self as main_window
        self.results_tab = ResultsTab()
        self.reports_tab = ReportsTab()
        self.about_tab = AboutTab()
        
        # Add tabs
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.reports_tab, "Reports")
        self.tabs.addTab(self.about_tab, "About")
        
    def set_project_dir(self, dir_path):
        """Update project directory for all tabs"""
        self.project_dir = dir_path
        # Update Results tab
        if hasattr(self, 'results_tab'):
            self.results_tab.set_project_dir(dir_path)
            self.results_tab.refresh_results()  # Explicitly refresh the results
        # Update Reports tab
        if hasattr(self, 'reports_tab'):
            self.reports_tab.set_project_dir(dir_path)
            self.reports_tab.refresh_images()  # Explicitly refresh the reports

def main():
    app = QApplication(sys.argv)
    ex = ScreeningGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()