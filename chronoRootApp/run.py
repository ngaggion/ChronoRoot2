import os
import sys
import platform
        
# Suppress Qt and OpenGL warnings
os.environ['QT_LOGGING_RULES'] = '*=false'
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

# --- CONFIGURATION CONSTANTS ---
APP_NAME = "chronoroot"
PROJECT_CONFIG_NAME = "project_config.json"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "mainInterfaceConfig.json")

# Ensure global config directory exists
os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox

import subprocess
import json
import pathlib
import re
import shutil
import glob
import os
from PIL import Image

def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resizeEvent(self, event):
        if self.pixmap():
            pixmap = self.pixmap().scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(pixmap)
        super().resizeEvent(event)

    def set_pixmap(self, pixmap, size = None):
        if size is None:
            size = self.size() 
        scaled_pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class Ui_ChronoRootAnalysis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_dir = None  # Add this before setupUi
        self.selected_plant = None  # Add this before setupUi
        self.setupUi(self)
        
    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog
        return QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", options=options)

    def saveFieldsIntoJson(self):            
        data = {}

        for field in [self.rpiField, self.cameraField, self.plantField, self.processingLimitField, 
                      self.processingLimitField_3, self.emergenceDistanceField, self.captureIntervalField,
                      self.everyXhourField, self.everyXhourFieldFourier, self.everyXhourFieldAngles, self.numComponentsFPCAField]:
            if field.text().isdigit():
                data[field.objectName()] = int(field.text())
            
            if field.text() == "":
                data[field.objectName()] = ""
                
        data.update({field.objectName(): field.text() for field in [self.identifierField, self.videoField, self.projectField,
                                                                    self.everyXhourField, self.everyXhourFieldFourier, 
                                                                    self.everyXhourFieldAngles, self.numComponentsFPCAField]})
        data.update({field.objectName(): field.isChecked() for field in [self.saveImagesButton, 
                                                                         self.videoHasQRbutton,
                                                                         self.saveImagesConvex, 
                                                                         self.doConvex, self.doFourier, self.doLateralAngles,
                                                                         self.doFPCA, self.normFPCA, self.averagePerPlantStats]})
        
        data["daysConvexHull"] = self.daysConvexField.text()
        data["daysAngles"] = self.daysAnglesField.text()

        # map values for compatibility with 1_analysis.py
        data["rpi"] = data["rpiField"]
        data["cam"] = data["cameraField"]
        data["plant"] = data["plantField"]
        data["identifier"] = data["identifierField"]
        data["Images"] = data["videoField"]
        data["processingLimit"] = data["processingLimitField"]
        data["timeStep"] = data["captureIntervalField"]
        data["MainFolder"] = data["projectField"]
        data["saveImages"] = data["saveImagesButton"]
        data["videoHasQR"] = data["videoHasQRbutton"]
        data["emergenceDistance"] = data["emergenceDistanceField"]

        if data["processingLimit"] != "":
            data['Limit'] = int(data["processingLimit"] * 24 * 60 / int(data['timeStep']))
        else:
            data['Limit'] = 0
            
        data['knownDistance'] = self.knownDistanceField.text()
        data['pixelDistance'] = self.pixelDistanceField.text()

        # 1. Save to Global Config (User Preferences)
        try:
            with open(GLOBAL_CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving global config: {e}")

        # 2. Save to Project Config (Reproducibility)
        project_path = self.projectField.text()
        if project_path and os.path.isdir(project_path):
            try:
                proj_cfg_path = os.path.join(project_path, PROJECT_CONFIG_NAME)
                with open(proj_cfg_path, "w") as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Error saving project config: {e}")

    def loadJsonIntoFields(self):
        # Determine which file to load
        project_cfg = os.path.join(self.projectField.text(), PROJECT_CONFIG_NAME)
        
        # Hierarchy: Project File > Global File
        if os.path.exists(project_cfg):
            json_path = project_cfg
        elif os.path.exists(GLOBAL_CONFIG_FILE):
            json_path = GLOBAL_CONFIG_FILE
        else:
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                for field in [self.rpiField, self.cameraField, self.plantField, self.processingLimitField, 
                      self.processingLimitField_3, self.emergenceDistanceField, self.captureIntervalField,
                      self.everyXhourField, self.everyXhourFieldFourier, self.everyXhourFieldAngles, self.numComponentsFPCAField]:
                    if field.objectName() in data:
                        field.setText(str(data[field.objectName()]))

                for field in [self.identifierField, self.videoField, self.projectField]:
                    if field.objectName() in data:
                        field.setText(data[field.objectName()])

                for field in [self.saveImagesButton, self.videoHasQRbutton,
                            self.saveImagesConvex, self.doConvex, self.doFourier, self.doLateralAngles,
                            self.doFPCA, self.normFPCA, self.averagePerPlantStats]:
                    if field.objectName() in data:
                        field.setChecked(data[field.objectName()])

                self.knownDistanceField.setText(str(data['knownDistance']))
                self.pixelDistanceField.setText(str(data['pixelDistance']))
            
                if "daysConvexHull" in data:
                    self.daysConvexField.setText(str(data["daysConvexHull"]))
                if "daysAngles" in data:
                    self.daysAnglesField.setText(str(data["daysAngles"]))
        except Exception as e:
            print(f"Error loading config: {e}")

    def refresh_table(self):
        # Store current sort order and column
        current_sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        current_sort_column = self.table.horizontalHeader().sortIndicatorSection()

        self.table.setSortingEnabled(False)

        self.table.clearContents()
        self.table.setRowCount(0)

        # Get the data from the database
        AnalysisFolder = os.path.join(self.projectField.text(), "Analysis")
        pathlib_dir = pathlib.Path(AnalysisFolder)

        data_files = pathlib_dir.glob('*/*/*/*/*')
        data_files = [str(file) for file in data_files]
        data_files = sorted(data_files, key=lambda x: natural_keys(x))

        data = []

        self.plant_dropdown.clear()

        for file in data_files:
            rel_path = os.path.relpath(file, AnalysisFolder)
            split = rel_path.split(os.path.sep)
            variety = split[0]
            rpi = split[1]
            camera = split[2]
            plant = split[3]
            results = split[4]

            # read the error rate from the log file first line
            if os.path.exists(os.path.join(file, "log.txt")):
                with open(os.path.join(file, "log.txt"), 'r') as f:
                    date = f.readline().replace("Analysis completed: ", "")

                    # error rate is in the last line
                    lines = f.readlines()
                    last_line = lines[-1]
                    error_rate = float(last_line.split(":")[-1].strip())
                    error_rate = round(error_rate, 4)
                status = "Finished"
            else:
                date = ""
                error_rate = ""
                status = "Not finished"

            data.append([variety, rpi, camera, plant, results, error_rate, status, date, file])

            self.plant_dropdown.addItem(file)

        self.table.setRowCount(len(data))

        for row, row_data in enumerate(data):
            for col, cell_data in enumerate(row_data[:-1]):  # Ignore the last element (path)
                item = QTableWidgetItem(str(cell_data))
                item.path = row_data[-1]  # Store the path in the item
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setSortingEnabled(True)  

        # Restore the sort order and column
        self.table.sortItems(current_sort_column, current_sort_order)

        return

    def universal_open(self, path):
        try:
            path = os.path.abspath(os.path.expanduser(path))
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
        
        # Ask the user to open manually if all methods fail
        QtWidgets.QMessageBox.information(None, 'Info', f'Please open the folder manually:\n{path}')
        
    def open_report_folder(self):
        report_path = os.path.join(self.projectField.text(), "Report")
        self.universal_open(report_path)
        
    def open_selected_path_tab3(self):
        self.universal_open(self.selected_plant)
        
    def open_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            item = self.table.item(selected_rows[0].row(), 0)
            self.universal_open(item.path)

    def remove_selected_path(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        # Removing the plant means moving it to a folder called "Removed"
        # This is done to avoid losing the data in case the user wants to recover it
        # Also keep the same folder structure, from the Analysis folder
        removed_path = self.projectField.text() + "/Removed"
        removed_path = os.path.join(removed_path, os.path.relpath(path, self.projectField.text() + "/Analysis"))

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Remove existing destination if it exists
        if os.path.exists(removed_path):
            shutil.rmtree(removed_path)

        # Move to removed folder
        shutil.move(path, removed_path)
        
        self.refresh_table()
        
        return

    def set_default_parameters(self):
        """Set default values for important fields"""
        self.emergenceDistanceField.setText("2")
        self.processingLimitField.setText("0")
        self.processingLimitField_3.setText("0")
        self.captureIntervalField.setText("15")
        self.captureIntervalField_3.setText("15")
        self.emergenceDistanceField_2.setText("2")
        self.everyXhourField.setText("6")
        self.everyXhourFieldFourier.setText("6")
        self.everyXhourFieldAngles.setText("6")
        self.numComponentsFPCAField.setText("2")

    def validate_numeric_input(self, field):
        """Validate numeric input fields"""
        try:
            text = field.text()
            # Don't allow empty fields
            if text.strip() == "":
                if field in [self.emergenceDistanceField, self.emergenceDistanceField_2]:
                    field.setText("2")
                elif field in [self.processingLimitField, self.processingLimitField_3]:
                    field.setText("0")
                elif field in [self.captureIntervalField, self.captureIntervalField_3]:
                    field.setText("15")
                return
                
            value = float(text)
            if field == self.emergenceDistanceField or field == self.emergenceDistanceField_2:
                if value <= 0:
                    field.setText("2")
            elif field in [self.processingLimitField, self.processingLimitField_3]:
                if value < 0:
                    field.setText("0")
            elif field in [self.captureIntervalField, self.captureIntervalField_3]:
                if value <= 0:
                    field.setText("15")
        except ValueError:
            if field in [self.emergenceDistanceField, self.emergenceDistanceField_2]:
                field.setText("2")
            elif field in [self.processingLimitField, self.processingLimitField_3]:
                field.setText("0")
            elif field in [self.captureIntervalField, self.captureIntervalField_3]:
                field.setText("15")

    def setup_field_validation(self):
        """Set up validation for numeric fields"""
        # Connect validation to editingFinished signal
        self.emergenceDistanceField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.emergenceDistanceField))
        self.emergenceDistanceField_2.editingFinished.connect(
            lambda: self.validate_numeric_input(self.emergenceDistanceField_2))
        self.processingLimitField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.processingLimitField))
        self.processingLimitField_3.editingFinished.connect(
            lambda: self.validate_numeric_input(self.processingLimitField_3))
        self.captureIntervalField.editingFinished.connect(
            lambda: self.validate_numeric_input(self.captureIntervalField))
        self.captureIntervalField_3.editingFinished.connect(
            lambda: self.validate_numeric_input(self.captureIntervalField_3))

    def get_image_paths(self):
        if not os.path.exists(os.path.join(self.selected_plant, "log.txt")):
            return None, None, None, None
        
        metadata = json.load(open(os.path.join(self.selected_plant, "metadata.json"), 'r'))
        bbox = metadata["bounding box"]
        overlayPath = metadata["folders"]["images"] + "/SegMulti/"
        
        variety = self.selected_plant.split(os.path.sep)[-5]
        rpi = self.selected_plant.split(os.path.sep)[-4]
        camera = self.selected_plant.split(os.path.sep)[-3]
        plant = self.selected_plant.split(os.path.sep)[-2]

        filename = variety + "_" + rpi + "_" + camera + "_" + plant + ".png"
        image2_path = os.path.join(self.selected_plant, filename)
        
        if not os.path.exists(image2_path):
            image2_path = None

        # list all images in the folder with pathlib, then sort them
        pathlib_dir = pathlib.Path(overlayPath)
        image_files = pathlib_dir.glob('*.png')
        image_files = [str(file) for file in image_files]
        image_files = sorted(image_files, key=lambda x: natural_keys(x))

        if len(image_files) == 0:
            return "Image not found", image2_path, overlayPath, None
        
        overlay = image_files[-1]
        image1_path = metadata["ImagePath"] + '/' + overlay.split(os.path.sep)[-1]

        return image1_path, image2_path, overlay, bbox

    def update_image_labels(self):
        # Add safety check
        if not hasattr(self, 'plant_dropdown') or self.plant_dropdown is None:
            return
        
        self.selected_plant = self.plant_dropdown.currentText()
        image1_path, image2_path, overlay, bbox = self.get_image_paths()
        
        # Check if image paths exist
        if image1_path is None:
            self.image_label1.clear()
            size = QtCore.QSize(250, 560)
            pixmap2 = QtGui.QPixmap("placeholder_figures/plant_placeholder.png")
            self.image_label1.set_pixmap(pixmap2, size)
            self.image_label1.show()
        elif not os.path.exists(image1_path) or not os.path.exists(overlay):
            self.image_label1.clear()
            size = QtCore.QSize(250, 560)
            pixmap2 = QtGui.QPixmap("placeholder_figures/plant_placeholder_2.png")
            self.image_label1.set_pixmap(pixmap2, size)
            self.image_label1.show()
        else:
            self.image_label1.clear()
            
            try:
                # Open image with PIL
                image = Image.open(image1_path)
                
                # Crop using PIL's crop method: (left, top, right, bottom)
                image = image.crop((bbox[2], bbox[0], bbox[3], bbox[1]))
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Check if overlay should be applied
                if self.overlay_checkbox.isChecked() and os.path.exists(overlay):
                    image_overlay = Image.open(overlay).convert("RGB")
                    
                    if image.size == image_overlay.size:
                        image = Image.blend(image, image_overlay, alpha=0.5)
                
                # Convert PIL Image to QPixmap
                image_bytes = image.tobytes()
                qImg = QtGui.QImage(image_bytes, image.width, image.height, 
                                image.width * 3, QtGui.QImage.Format_RGB888)
                pixmap1 = QtGui.QPixmap.fromImage(qImg)
                
                size = QtCore.QSize(250, 560)
                self.image_label1.set_pixmap(pixmap1, size)
                self.image_label1.show()
                
            except Exception as e:
                self.image_label1.setText(f"Analysis is not yet finished. \nRefresh to update\nError: {str(e)}")
                self.image_label1.setAlignment(QtCore.Qt.AlignCenter)
                self.image_label1.show()
        
        # Check if image2_path exists
        if image2_path is not None and os.path.exists(image2_path):
            size = QtCore.QSize(400, 400)
            pixmap2 = QtGui.QPixmap(image2_path)
            self.image_label2.set_pixmap(pixmap2, size)
            self.image_label2.show()
        else:
            self.image_label2.clear()
            size = QtCore.QSize(400, 400)
            pixmap2 = QtGui.QPixmap("placeholder_figures/plant_report_placeholder.png")
            self.image_label2.set_pixmap(pixmap2, size)
            self.image_label2.show()
        
        return

    def remove_selected_plant(self):
        path = self.selected_plant

        removed_path = self.projectField.text() + "/Removed"
        removed_path = os.path.join(removed_path, os.path.relpath(path, self.projectField.text() + "/Analysis"))

        if not os.path.exists(os.path.dirname(removed_path)):
            os.makedirs(os.path.dirname(removed_path))

        # Open the directory in the file explorer
        if os.name == 'nt':
            os.system(f'move "{path}" "{removed_path}"')
        elif sys.platform == 'darwin':
            os.system(f'mv "{path}" "{removed_path}"')
        else:
            os.system(f'mv "{path}" "{removed_path}"')
        
        self.refresh_table()
        
        return
    
    def analysis(self):
        """Run analysis with validation"""
        
        # Get and validate video folder
        video_folder = self.videoField.text()
        
        if not video_folder:
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please specify a video folder first!')
            return
            
        if not os.path.exists(video_folder):
            QtWidgets.QMessageBox.warning(None, 'Error', 'Video folder does not exist!\nPlease check the path.')
            return
        
        # Check for PNG images
        images = glob.glob(os.path.join(video_folder, "*.png"))
        
        if not images:
            QtWidgets.QMessageBox.warning(
                None, 'Error', 
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return
        
        # Check for segmentation files (required for analysis)
        seg_folder = os.path.join(video_folder, "Segmentation", "Ensemble")
        seg_files = glob.glob(os.path.join(seg_folder, "*.png")) if os.path.exists(seg_folder) else []
        
        if not seg_files:
            QtWidgets.QMessageBox.warning(
                None, 'Error',
                f'Found {len(images)} images but no segmentation files!\n\n'
                'Segmentation is required for analysis.\n'
                'Please ensure the images have been properly segmented first.'
            )
            return
        
        # Validate calibration settings
        if not self.videoHasQRbutton.isChecked():                
            if not self.knownDistanceField.text() or not self.pixelDistanceField.text():
                QtWidgets.QMessageBox.warning(
                    None, 'Error', 
                    'Please provide both known distance and pixel distance for manual calibration,\n'
                    'or enable "Video has QR codes"!'
                )
                return
            
            try:
                known_dist = float(self.knownDistanceField.text())
                pixel_dist = int(self.pixelDistanceField.text())
                if known_dist <= 0 or pixel_dist <= 0:
                    QtWidgets.QMessageBox.warning(None, 'Error', 'Calibration values must be positive numbers!')
                    return
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    None, 'Error', 
                    'Invalid calibration values!\nKnown distance must be a number, pixel distance must be an integer.'
                )
                return
        
        # All validations passed, run analysis
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "1_analysis.py"])
        
    def getBBOX(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "1_analysis.py", "--getbbox"])

    def rerunAnalysis(self):
        metadata_path = os.path.join(self.selected_plant, "metadata.json")
        subprocess.Popen(["python", "1_analysis.py", "--config", metadata_path, "--rerun"])

    def rerunAnalysis_table(self):
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        selected_row = selected_rows[0].row()
        item = self.table.item(selected_row, 0)
        path = item.path

        metadata_path = os.path.join(path, "metadata.json")
        subprocess.Popen(["python", "1_analysis.py", "--config", metadata_path, "--rerun"])

    def preview(self):
        """Preview with validation for images and segmentation"""
        
        # Save fields first
        self.saveFieldsIntoJson()
        
        # Get and validate video folder path
        video_folder = self.videoField.text()
        
        if not video_folder:
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please specify a video folder first!')
            return
        
        if not os.path.exists(video_folder):
            QtWidgets.QMessageBox.warning(None, 'Error', 'Video folder does not exist!\nPlease check the path to the folder.')
            return
        
        # Check for PNG images
        images = glob.glob(os.path.join(video_folder, "*.png"))
        
        if not images:
            QtWidgets.QMessageBox.warning(
                None, 'Error', 
                'No images found in the video folder!\nPlease check the path to the folder where the images are located.'
            )
            return
        
        # Check for segmentation files in Segmentation/Ensemble folder
        seg_folder = os.path.join(video_folder, "Segmentation", "Ensemble")
        seg_files = glob.glob(os.path.join(seg_folder, "*.png")) if os.path.exists(seg_folder) else []
        
        if not seg_files:
            # Images exist but no segmentation found
            QtWidgets.QMessageBox.warning(
                None, 'Error',
                f'Found {len(images)} images but no segmentation files!\n The images may not have been properly segmented.'
            )
            return
        
        # Launch preview
        subprocess.Popen(["python", "1_analysis.py", "--preview"])

    def PostProcess(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "2_postprocess.py"])
    
    def report(self):
        self.saveFieldsIntoJson()
        subprocess.Popen(["python", "3_generateReport.py"])   

    def reviewPlant(self):
        path = self.selected_plant
        subprocess.Popen(["python", "4_reviewPlant.py", "--path", path])

    def syncProjectFolderField(self):
        projectFolder = self.projectField.text()
        projectFolder2 = self.projectField_2.text()

        if self.central_widget.sender() == self.projectField:
            self.projectField_2.setText(projectFolder)
        elif self.central_widget.sender() == self.projectField_2:
            self.projectField.setText(projectFolder2)
    
    def syncCaptureIntervalField(self):
        captureInterval = self.captureIntervalField.text()
        captureInterval2 = self.captureIntervalField_3.text()

        if self.central_widget.sender() == self.captureIntervalField:
            self.captureIntervalField_3.setText(captureInterval)
        elif self.central_widget.sender() == self.captureIntervalField_3:
            self.captureIntervalField.setText(captureInterval2)

    def syncProcessingLimitField(self):
        processingLimit = self.processingLimitField.text()
        processingLimit2 = self.processingLimitField_3.text()

        if self.central_widget.sender() == self.processingLimitField:
            self.processingLimitField_3.setText(processingLimit)
        elif self.central_widget.sender() == self.processingLimitField_3:
            self.processingLimitField.setText(processingLimit2)
                    
    def setupUi(self, chrono_root_analysis):
        chrono_root_analysis.setObjectName("ChronoRootAnalysis")
        chrono_root_analysis.resize(811, 600)
        self.central_widget = QtWidgets.QWidget(chrono_root_analysis)
        self.central_widget.setObjectName("centralwidget")
        
        self.setup_tabs()
        self.setup_tab1_elements()
        self.setup_tab2_elements()
        self.setup_tab3_elements()
        self.setup_tab4_elements()
        self.setup_tab5_elements()

        self.setup_field_validation()
        self.set_default_parameters()

        chrono_root_analysis.setCentralWidget(self.central_widget)
        self.statusbar = QtWidgets.QStatusBar(chrono_root_analysis)
        self.statusbar.setObjectName("statusbar")
        chrono_root_analysis.setStatusBar(self.statusbar)

        self.retranslate_ui(chrono_root_analysis)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(chrono_root_analysis)

        #self.refresh_table()

    def setup_tabs(self):
        self.tab_widget = QtWidgets.QTabWidget(self.central_widget)
        self.tab_widget.setGeometry(QtCore.QRect(0, 0, 811, 621))
        
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tab_widget.setFont(font)
        self.tab_widget.setObjectName("tabWidget")
        
        return

    def read_config_from_file(self):
        options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog
        file_filter = "JSON Files (*.json);;All Files (*)"
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Configuration File", "", file_filter, options=options)
    
        if not json_path:
            return  # If no file was selected, exit the function

        with open(json_path, 'r') as file:
            data = json.load(file)

        for field in [self.rpiField, self.cameraField, self.plantField, self.processingLimitField, 
                      self.processingLimitField_3, self.emergenceDistanceField, self.captureIntervalField,
                      self.everyXhourField, self.everyXhourFieldFourier, self.everyXhourFieldAngles, self.numComponentsFPCAField]:
            if field.objectName() in data:
                field.setText(str(data[field.objectName()]))

        for field in [self.identifierField, self.videoField, self.projectField]:
            if field.objectName() in data:
                field.setText(data[field.objectName()])

        for field in [self.saveImagesButton, self.videoHasQRbutton,
                      self.saveImagesConvex, self.doConvex, self.doFourier, self.doLateralAngles,
                      self.doFPCA, self.normFPCA, self.averagePerPlantStats]:
            if field.objectName() in data:
                field.setChecked(data[field.objectName()])

        if "daysConvexHull" in data:
            self.daysConvexField.setText(str(data["daysConvexHull"]))
        if "daysAngles" in data:
            self.daysAnglesField.setText(str(data["daysAngles"]))


    def setup_tab1_elements(self):

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.tab_widget.addTab(self.tab1, "")
    
        self.videoField = QtWidgets.QLineEdit(self.tab1)
        self.videoField.setGeometry(QtCore.QRect(190, 100, 441, 31))
        self.videoField.setObjectName("videoField")

        self.loadVideo = QtWidgets.QPushButton(self.tab1)
        self.loadVideo.setGeometry(QtCore.QRect(10, 100, 161, 31))
        self.loadVideo.setObjectName("loadVideo")
        self.loadVideo.clicked.connect(lambda: self.videoField.setText(self.openFileNameDialog()))

        self.loadProject = QtWidgets.QPushButton(self.tab1)
        self.loadProject.setGeometry(QtCore.QRect(10, 50, 161, 31))
        self.loadProject.setObjectName("loadProject")
        self.loadProject.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.projectField = QtWidgets.QLineEdit(self.tab1)
        self.projectField.setGeometry(QtCore.QRect(190, 50, 441, 31))
        self.projectField.setObjectName("projectField")
        self.projectField.textChanged.connect(self.syncProjectFolderField)

        self.rpiField = QtWidgets.QLineEdit(self.tab1)
        self.rpiField.setGeometry(QtCore.QRect(190, 150, 51, 31))
        self.rpiField.setObjectName("rpiField")

        self.cameraField = QtWidgets.QLineEdit(self.tab1)
        self.cameraField.setGeometry(QtCore.QRect(190, 200, 51, 31))
        self.cameraField.setObjectName("cameraField")

        self.plantField = QtWidgets.QLineEdit(self.tab1)
        self.plantField.setGeometry(QtCore.QRect(190, 250, 51, 31))
        self.plantField.setObjectName("plantField")

        self.identifierField = QtWidgets.QLineEdit(self.tab1)
        self.identifierField.setGeometry(QtCore.QRect(190, 300, 151, 31))
        self.identifierField.setObjectName("identifierField")

        self.saveImagesButton = QtWidgets.QCheckBox(self.tab1)
        self.saveImagesButton.setGeometry(QtCore.QRect(10, 380, 161, 31))
        self.saveImagesButton.setObjectName("saveImagesButton")
        
        self.videoHasQRbutton = QtWidgets.QCheckBox(self.tab1)
        self.videoHasQRbutton.setGeometry(QtCore.QRect(10, 410, 161, 31))
        self.videoHasQRbutton.setObjectName("videoHasQRbutton")

        self.captureIntervalField = QtWidgets.QLineEdit(self.tab1)
        self.captureIntervalField.setGeometry(QtCore.QRect(190, 500, 51, 31))
        self.captureIntervalField.setObjectName("captureIntervalField")
        self.captureIntervalField.textChanged.connect(self.syncCaptureIntervalField)

        self.processingLimitField = QtWidgets.QLineEdit(self.tab1)
        self.processingLimitField.setGeometry(QtCore.QRect(190, 450, 51, 31))
        self.processingLimitField.setObjectName("processingLimitField")
        self.processingLimitField.textChanged.connect(self.syncProcessingLimitField)

        # Add emergence distance field to tab1
        self.emergenceDistanceField_2 = QtWidgets.QLineEdit(self.tab1)
        self.emergenceDistanceField_2.setGeometry(QtCore.QRect(190, 550, 51, 31))
        self.emergenceDistanceField_2.setObjectName("emergenceDistanceField_2")
        
        self.emergenceDistanceLabel = QtWidgets.QLabel(self.tab1)
        self.emergenceDistanceLabel.setGeometry(QtCore.QRect(10, 550, 161, 31))
        self.emergenceDistanceLabel.setObjectName("emergenceDistanceLabel")
        
        self.emergenceDistanceExp = QtWidgets.QLabel(self.tab1)
        self.emergenceDistanceExp.setGeometry(QtCore.QRect(260, 550, 261, 31))
        self.emergenceDistanceExp.setObjectName("emergenceDistanceExp")

        self.saveButton = QtWidgets.QPushButton(self.tab1)
        self.saveButton.setGeometry(QtCore.QRect(660, 0, 141, 81))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.clicked.connect(self.saveFieldsIntoJson)

        self.previewAnalysisButton = QtWidgets.QPushButton(self.tab1)
        self.previewAnalysisButton.setGeometry(QtCore.QRect(660, 100, 141, 81))
        self.previewAnalysisButton.setObjectName("previewAnalysisButton")
        self.previewAnalysisButton.clicked.connect(self.preview)

        self.analysisButton = QtWidgets.QPushButton(self.tab1)
        self.analysisButton.setGeometry(QtCore.QRect(660, 200, 141, 81))
        self.analysisButton.setObjectName("analysisButton")
        self.analysisButton.clicked.connect(self.analysis)

        self.PostProcessButton = QtWidgets.QPushButton(self.tab1)
        self.PostProcessButton.setGeometry(QtCore.QRect(660, 300, 141, 81))
        self.PostProcessButton.setObjectName("PostProcessButton")
        self.PostProcessButton.clicked.connect(self.PostProcess)

        self.loadConfigFileButton = QtWidgets.QPushButton(self.tab1)
        self.loadConfigFileButton.setGeometry(QtCore.QRect(660, 400, 141, 81))
        self.loadConfigFileButton.setObjectName("loadLastConfigButton")
        self.loadConfigFileButton.clicked.connect(self.read_config_from_file)

        self.loadLastConfigButton = QtWidgets.QPushButton(self.tab1)
        self.loadLastConfigButton.setGeometry(QtCore.QRect(660, 500, 141, 81))
        self.loadLastConfigButton.setObjectName("loadLastConfigButton")
        self.loadLastConfigButton.clicked.connect(self.loadJsonIntoFields)

        self.label = QtWidgets.QLabel(self.tab1)
        self.label.setGeometry(QtCore.QRect(10, 150, 161, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab1)
        self.label_2.setGeometry(QtCore.QRect(10, 200, 161, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab1)
        self.label_3.setGeometry(QtCore.QRect(10, 250, 161, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab1)
        self.label_4.setGeometry(QtCore.QRect(10, 300, 161, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab1)
        self.label_5.setGeometry(QtCore.QRect(260, 150, 261, 31))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab1)
        self.label_6.setGeometry(QtCore.QRect(260, 200, 261, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab1)
        self.label_7.setGeometry(QtCore.QRect(260, 250, 261, 31))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab1)
        self.label_8.setGeometry(QtCore.QRect(360, 300, 261, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab1)
        self.label_9.setGeometry(QtCore.QRect(10, 350, 541, 31))
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(self.tab1)
        self.label_11.setGeometry(QtCore.QRect(10, 500, 161, 31))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab1)
        self.label_12.setGeometry(QtCore.QRect(10, 450, 161, 31))
        self.label_12.setObjectName("label_12")
        self.label_26 = QtWidgets.QLabel(self.tab1)
        self.label_26.setGeometry(QtCore.QRect(260, 450, 261, 31))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.tab1)
        self.label_27.setGeometry(QtCore.QRect(260, 500, 261, 31))
        self.label_27.setObjectName("label_27")
        self.label_30 = QtWidgets.QLabel(self.tab1)
        self.label_30.setGeometry(QtCore.QRect(10, 10, 541, 31))
        self.label_30.setObjectName("label_30")
        self.line = QtWidgets.QFrame(self.tab1)
        self.line.setGeometry(QtCore.QRect(0, 340, 651, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.tab1)
        self.line_2.setGeometry(QtCore.QRect(640, -30, 20, 641))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        # Create container widget for manual calibration (hidden by default)
        self.manual_calib_widget = QtWidgets.QWidget(self.tab1)
        self.manual_calib_widget.setGeometry(QtCore.QRect(210, 375, 380, 70))
        
        
        # Write a "Manual Calibration" label
        self.manual_calib_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.manual_calib_label.setGeometry(QtCore.QRect(0, 0, 200, 31))
        self.manual_calib_label.setText("Manual Calibration Parameters:")
        
        # Known distance label and field - first row
        self.known_dist_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.known_dist_label.setGeometry(QtCore.QRect(220, 0, 100, 31))
        self.known_dist_label.setText("Known (mm):")
        
        self.knownDistanceField = QtWidgets.QLineEdit(self.manual_calib_widget)
        self.knownDistanceField.setGeometry(QtCore.QRect(320, 0, 51, 31))
        self.knownDistanceField.setPlaceholderText("10")
        self.knownDistanceField.setObjectName("knownDistanceField")
        
        # Pixel distance label and field - continue first row
        self.pixel_dist_label = QtWidgets.QLabel(self.manual_calib_widget)
        self.pixel_dist_label.setGeometry(QtCore.QRect(220, 31, 70, 31))
        self.pixel_dist_label.setText("Pixels:")
        
        self.pixelDistanceField = QtWidgets.QLineEdit(self.manual_calib_widget)
        self.pixelDistanceField.setGeometry(QtCore.QRect(320, 32, 51, 31))
        self.pixelDistanceField.setPlaceholderText("240")
        self.pixelDistanceField.setObjectName("pixelDistanceField")
        
        # Calibration helper button - second row
        self.calibrateBtn = QtWidgets.QPushButton(self.manual_calib_widget)
        self.calibrateBtn.setGeometry(QtCore.QRect(0, 30, 180, 31))
        self.calibrateBtn.setText("Open Calibration Helper")
        self.calibrateBtn.clicked.connect(self.open_calibration_helper)
        
        # Connect toggle function to checkbox
        self.videoHasQRbutton.stateChanged.connect(self.toggle_calibration_mode)
        
        # Initialize visibility
        self.toggle_calibration_mode()
    
        return
    
    def toggle_calibration_mode(self):
        """Toggle between QR and manual calibration modes"""
        has_qr = self.videoHasQRbutton.isChecked()
        self.manual_calib_widget.setVisible(not has_qr)

    def open_calibration_helper(self):
        """Opens calibration helper window"""
        if not self.videoField.text():
            QtWidgets.QMessageBox.warning(None, 'Error', 'Please select a video directory first!')
            return

        args = [
            "python",
            "calibration_helper.py",
            "--video-dir", self.videoField.text()
        ]

        try:
            subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            QtWidgets.QMessageBox.information(
                None,
                "Calibration Helper",
                "Calibration helper window has been opened.\n"
                "Measure the pixel distance between two points\n"
                "of known physical distance in your image."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Error starting calibration helper: {str(e)}")

    def setup_tab2_elements(self):
        # Create the table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Variety", "Raspberry", "Camera", "Plant Number", "Result ID", 
                                              "Error Rate", "Status", "Finish Date"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Enable sorting
        self.table.setSortingEnabled(True)

        # Create the refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_table)

        # Create the rerun analysis button
        self.rerun_analysis_button_tab2 = QPushButton("Rerun Analysis")
        self.rerun_analysis_button_tab2.clicked.connect(self.rerunAnalysis_table)

        # Create the open path button
        self.open_path_button = QPushButton("Open Path")
        self.open_path_button.clicked.connect(self.open_selected_path)

        # Create the remove path button
        self.remove_path_button = QPushButton("Remove Plant")
        self.remove_path_button.clicked.connect(self.remove_selected_path)

        # Set up the layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.open_path_button)
        buttons_layout.addWidget(self.remove_path_button)
        buttons_layout.addWidget(self.rerun_analysis_button_tab2)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(buttons_layout)

        # Create and set up the new tab
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setLayout(layout)
        self.tab_widget.addTab(self.tab2, "Tab 2")


    def setup_tab3_elements(self):
        # Create the image labels
        self.image_label1 = AspectRatioLabel()
        self.image_label2 = AspectRatioLabel()

        # Set image labels to scale contents with aspect ratio
        self.image_label1.setMaximumSize(250, 560)
        self.image_label2.setMaximumSize(400, 400)

        # Create the checkbox
        self.overlay_checkbox = QCheckBox("Overlay Image")

        # Create the dropdown menu for plant selection to the right of the checkbox
        self.plant_dropdown = QComboBox()

        # Create the refresh button
        self.refresh_button_tab3 = QPushButton("Refresh_2")
        self.refresh_button_tab3.clicked.connect(self.refresh_table)

        # Create a rerun analysis button
        self.rerun_analysis_button = QPushButton("Rerun Analysis")
        self.rerun_analysis_button.clicked.connect(self.rerunAnalysis)

        # Create the remove path button
        self.remove_path_button_tab3 = QPushButton("Remove Plant")
        self.remove_path_button_tab3.clicked.connect(self.remove_selected_plant)

        # Connect signals
        self.plant_dropdown.currentIndexChanged.connect(self.update_image_labels)
        self.overlay_checkbox.stateChanged.connect(self.update_image_labels)

        self.reviewButton = QPushButton("View full sequence")
        self.reviewButton.clicked.connect(self.reviewPlant)

        self.openPathButton2 = QPushButton("Open Folder")
        self.openPathButton2.clicked.connect(self.open_selected_path_tab3)

        # Set up the layout for the checkbox, dropdown menu, and refresh button
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.overlay_checkbox)
        controls_layout.addWidget(self.plant_dropdown)

        controls_layout2 = QHBoxLayout()
        controls_layout2.addWidget(self.refresh_button_tab3)
        controls_layout2.addWidget(self.rerun_analysis_button)
        controls_layout2.addWidget(self.remove_path_button_tab3)
        controls_layout2.addWidget(self.reviewButton)
        controls_layout2.addWidget(self.openPathButton2)

        # Set up the main layout
        layout = QHBoxLayout()
        layout.addWidget(self.image_label1)
        layout.addWidget(self.image_label2)

        bigLayout = QVBoxLayout()
        bigLayout.addLayout(layout)
        bigLayout.addLayout(controls_layout)
        bigLayout.addLayout(controls_layout2)
        
        # Create and set up the new tab
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setLayout(bigLayout)
        self.tab_widget.addTab(self.tab3, "Plant Overlay")
        
        self.update_image_labels()

    def setup_tab4_elements(self):
        self.tab4 = QtWidgets.QWidget()
        self.tab4.setObjectName("tab4")
        self.tab_widget.addTab(self.tab4, "")

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        
        self.averagePerPlantStats = QtWidgets.QCheckBox(self.tab4)
        self.averagePerPlantStats.setGeometry(QtCore.QRect(10, 70, 311, 31))
        self.averagePerPlantStats.setFont(font)
        self.averagePerPlantStats.setObjectName("averagePerPlantStats")

        self.everyXhourText = QtWidgets.QLabel(self.tab4)
        self.everyXhourText.setGeometry(QtCore.QRect(370, 70, 241, 31))
        self.everyXhourText.setObjectName("everyXhourText")

        self.everyXhourField = QtWidgets.QLineEdit(self.tab4)
        self.everyXhourField.setGeometry(QtCore.QRect(620, 70, 51, 31))
        self.everyXhourField.setObjectName("everyXhourField")

        self.reportText = QtWidgets.QLabel(self.tab4)
        self.reportText.setGeometry(QtCore.QRect(10, 110, 810, 31))
        self.reportText.setObjectName("reportText")

        self.doFPCA = QtWidgets.QCheckBox(self.tab4)
        self.doFPCA.setGeometry(QtCore.QRect(10, 150, 311, 31))
        self.doFPCA.setFont(font)
        self.doFPCA.setObjectName("doFPCA")

        self.normFPCA = QtWidgets.QCheckBox(self.tab4)
        self.normFPCA.setGeometry(QtCore.QRect(340, 150, 221, 31))
        self.normFPCA.setObjectName("normFPCA")

        self.numComponentsFPCAText = QtWidgets.QLabel(self.tab4)
        self.numComponentsFPCAText.setGeometry(QtCore.QRect(570, 150, 200, 31))
        self.numComponentsFPCAText.setObjectName("numComponentsFPCAText")

        self.numComponentsFPCAField = QtWidgets.QLineEdit(self.tab4)
        self.numComponentsFPCAField.setGeometry(QtCore.QRect(730, 150, 51, 31))
        self.numComponentsFPCAField.setObjectName("numComponentsFPCAField")

        self.line_4 = QtWidgets.QFrame(self.tab4)
        self.line_4.setGeometry(QtCore.QRect(-40, 170, 891, 41))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")

        self.doConvex = QtWidgets.QCheckBox(self.tab4)
        self.doConvex.setGeometry(QtCore.QRect(10, 200, 201, 31))
        self.doConvex.setFont(font)
        self.doConvex.setObjectName("doConvex")

        self.saveImagesConvex = QtWidgets.QCheckBox(self.tab4)
        self.saveImagesConvex.setGeometry(QtCore.QRect(370, 200, 311, 31))
        self.saveImagesConvex.setObjectName("saveImagesConvex")

        self.daysFieldConvex = QtWidgets.QLabel(self.tab4)
        self.daysFieldConvex.setGeometry(QtCore.QRect(10, 240, 131, 31))
        self.daysFieldConvex.setObjectName("daysFieldConvex")

        self.daysConvexField = QtWidgets.QLineEdit(self.tab4)
        self.daysConvexField.setGeometry(QtCore.QRect(120, 240, 221, 31))
        self.daysConvexField.setObjectName("daysConvexField")

        self.daysConvexText = QtWidgets.QLabel(self.tab4)
        self.daysConvexText.setGeometry(QtCore.QRect(350, 240, 351, 31))
        self.daysConvexText.setObjectName("daysConvexText")

        self.line_5 = QtWidgets.QFrame(self.tab4)
        self.line_5.setGeometry(QtCore.QRect(-40, 270, 891, 41))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")

        self.doFourier = QtWidgets.QCheckBox(self.tab4)
        self.doFourier.setGeometry(QtCore.QRect(10, 305, 451, 31))
        self.doFourier.setFont(font)
        self.doFourier.setObjectName("doFourier")

        self.everyXhourTextFourier = QtWidgets.QLabel(self.tab4)
        self.everyXhourTextFourier.setGeometry(QtCore.QRect(450, 305, 231, 31))
        self.everyXhourTextFourier.setObjectName("everyXhourTextFourier")

        self.everyXhourFieldFourier = QtWidgets.QLineEdit(self.tab4)
        self.everyXhourFieldFourier.setGeometry(QtCore.QRect(680, 305, 51, 31))
        self.everyXhourFieldFourier.setObjectName("everyXhourFieldFourier")

        self.line_6 = QtWidgets.QFrame(self.tab4)
        self.line_6.setGeometry(QtCore.QRect(-70, 330, 961, 41))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")

        self.doLateralAngles = QtWidgets.QCheckBox(self.tab4)
        self.doLateralAngles.setGeometry(QtCore.QRect(10, 365, 301, 31))
        self.doLateralAngles.setFont(font)
        self.doLateralAngles.setObjectName("doLateralAngles")

        self.emergenceDistanceText = QtWidgets.QLabel(self.tab4)
        self.emergenceDistanceText.setGeometry(QtCore.QRect(370, 365, 131, 31))
        self.emergenceDistanceText.setObjectName("emergenceDistanceText")

        self.emergenceDistanceField = QtWidgets.QLineEdit(self.tab4)
        self.emergenceDistanceField.setGeometry(QtCore.QRect(510, 365, 51, 31))
        self.emergenceDistanceField.setObjectName("emergenceDistanceField")

        self.emergenceDistanceTextExp = QtWidgets.QLabel(self.tab4)
        self.emergenceDistanceTextExp.setGeometry(QtCore.QRect(570, 365, 261, 31))
        self.emergenceDistanceTextExp.setObjectName("emergenceDistanceTextExp")

        self.daysAnglesText = QtWidgets.QLabel(self.tab4)
        self.daysAnglesText.setGeometry(QtCore.QRect(10, 410, 131, 31))
        self.daysAnglesText.setObjectName("daysAnglesText")

        self.daysAnglesField = QtWidgets.QLineEdit(self.tab4)
        self.daysAnglesField.setGeometry(QtCore.QRect(120, 410, 221, 31))
        self.daysAnglesField.setObjectName("daysAnglesField")

        self.everyXhourTextAngles = QtWidgets.QLabel(self.tab4)
        self.everyXhourTextAngles.setGeometry(QtCore.QRect(370, 410, 231, 31))
        self.everyXhourTextAngles.setObjectName("everyXhourTextAngles")

        self.everyXhourFieldAngles = QtWidgets.QLineEdit(self.tab4)
        self.everyXhourFieldAngles.setGeometry(QtCore.QRect(620, 410, 51, 31))
        self.everyXhourFieldAngles.setObjectName("everyXhourFieldAngles")

        self.PostProcessButton2 = QtWidgets.QPushButton(self.tab4)
        self.PostProcessButton2.setGeometry(QtCore.QRect(360, 480, 131, 81))
        self.PostProcessButton2.setObjectName("PostProcessButton2")
        self.PostProcessButton2.clicked.connect(self.PostProcess)

        self.reportButton = QtWidgets.QPushButton(self.tab4)
        self.reportButton.setGeometry(QtCore.QRect(510, 480, 131, 81))
        self.reportButton.setObjectName("re portButton")
        self.reportButton.clicked.connect(self.report)

        self.loadLastConfig2 = QtWidgets.QPushButton(self.tab4)
        self.loadLastConfig2.setGeometry(QtCore.QRect(660, 480, 141, 81))
        self.loadLastConfig2.setObjectName("loadLastConfig2")
        self.loadLastConfig2.clicked.connect(self.loadJsonIntoFields)

        self.saveButton_2 = QtWidgets.QPushButton(self.tab4)
        self.saveButton_2.setGeometry(QtCore.QRect(210, 480, 131, 81))
        self.saveButton_2.setObjectName("saveButton_2")
        self.saveButton_2.clicked.connect(self.saveFieldsIntoJson)

        self.loadProject_2 = QtWidgets.QPushButton(self.tab4)
        self.loadProject_2.setGeometry(QtCore.QRect(10, 10, 161, 31))
        self.loadProject_2.clicked.connect(lambda: self.projectField.setText(self.openFileNameDialog()))

        self.projectField_2 = QtWidgets.QLineEdit(self.tab4)
        self.projectField_2.setGeometry(QtCore.QRect(190, 10, 441, 31))
        self.projectField_2.setObjectName("projectField_2")
        self.projectField_2.textChanged.connect(self.syncProjectFolderField)

        self.captureIntervalField_3 = QtWidgets.QLineEdit(self.tab4)
        self.captureIntervalField_3.setGeometry(QtCore.QRect(140, 530, 51, 31))
        self.captureIntervalField_3.setObjectName("captureIntervalField_3")
        self.captureIntervalField_3.textChanged.connect(self.syncCaptureIntervalField)

        self.processingLimitField_3 = QtWidgets.QLineEdit(self.tab4)
        self.processingLimitField_3.setGeometry(QtCore.QRect(140, 480, 51, 31))
        self.processingLimitField_3.setObjectName("processingLimitField_3")
        self.processingLimitField_3.textChanged.connect(self.syncProcessingLimitField)

        self.line_7 = QtWidgets.QFrame(self.tab4)
        self.line_7.setGeometry(QtCore.QRect(0, 40, 891, 41))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_11 = QtWidgets.QFrame(self.tab4)
        self.line_11.setGeometry(QtCore.QRect(-30, 440, 961, 41))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")

        self.captureIntervalLabel = QtWidgets.QLabel(self.tab4)
        self.captureIntervalLabel.setGeometry(QtCore.QRect(10, 530, 111, 31))
        self.captureIntervalLabel.setObjectName("captureIntervalLabel")
        self.processingLimitLabel = QtWidgets.QLabel(self.tab4)
        self.processingLimitLabel.setGeometry(QtCore.QRect(10, 480, 121, 31))
        self.processingLimitLabel.setObjectName("processingLimitLabel")

        return

    def update_report_labels(self):
        if not hasattr(self, 'projectField') or not hasattr(self, 'report_dropdown'):
            return
                
        report_path = os.path.join(self.projectField.text(), "Report/")
        current_report = self.report_dropdown.currentText()
        
        if (self.projectField.text() == "" or not os.path.exists(self.projectField.text()) 
            or not os.path.exists(report_path) or current_report == ""):
                
            self.report_label_1.clear()
            report_path_1 = os.path.join("placeholder_figures/report_placeholder.png")
            size = QtCore.QSize(750, 550)
            pixmap_1 = QtGui.QPixmap(report_path_1)
            self.report_label_1.set_pixmap(pixmap_1, size)
            self.report_label_1.setAlignment(QtCore.Qt.AlignCenter)
            self.report_label_1.show()
            return 
        else:
            report_path_1 = os.path.join(report_path, current_report)
            size = QtCore.QSize(750, 550)
            pixmap_1 = QtGui.QPixmap(report_path_1)
            self.report_label_1.set_pixmap(pixmap_1, size)
            self.report_label_1.setAlignment(QtCore.Qt.AlignCenter)
            self.report_label_1.show()
        return
    
    def refresh_tab5(self):
        report_path = os.path.join(self.projectField.text(), "Report/")
        figures = pathlib.Path(report_path).glob("*/*.png")
        self.report_figures = [str(figure).replace(report_path, "") for figure in figures]
        self.report_figures = sorted(self.report_figures, key=natural_keys)[::-1]
        
        self.report_dropdown.clear()
        self.report_dropdown.addItems(self.report_figures)
    
    def setup_tab5_elements(self):
        self.tab5 = QtWidgets.QWidget()
        self.tab5.setObjectName("tab5")
        self.tab_widget.addTab(self.tab5, "")

        # Create the image labels
        self.report_label_1 = AspectRatioLabel()
        self.report_label_1.setMaximumSize(750, 550)
        self.report_label_1.setObjectName("report_label_1") 

        self.refresh_button_tab5 = QPushButton(self.tab5)
        self.refresh_button_tab5.setObjectName("Refresh_5")
        self.refresh_button_tab5.clicked.connect(self.refresh_tab5)
        
        # create a menu to select the report
        self.report_dropdown = QComboBox(self.tab5)
        self.report_dropdown.setGeometry(QtCore.QRect(10, 10, 161, 31))
        self.report_dropdown.setObjectName("report_dropdown")
        self.report_dropdown.currentIndexChanged.connect(self.update_report_labels)
        
        self.open_path_button = QPushButton(self.tab5)
        self.open_path_button.setObjectName("Open Path")
        self.open_path_button.clicked.connect(self.open_report_folder)

        # Set up the main layout
        layout = QVBoxLayout()
        
        horizontal_layout_top = QHBoxLayout()
        horizontal_layout_top.addWidget(self.report_label_1, 1)
        horizontal_layout_top.setAlignment(QtCore.Qt.AlignCenter)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.report_dropdown, 3)
        horizontal_layout.addWidget(self.open_path_button, 1)
        horizontal_layout.addWidget(self.refresh_button_tab5, 1)
        horizontal_layout.setAlignment(QtCore.Qt.AlignCenter)
                
        layout.addLayout(horizontal_layout_top)
        layout.addLayout(horizontal_layout)
        
        self.tab5.setLayout(layout) 
        #self.update_report_labels()
        
    def retranslate_ui(self, ChronoRootAnalysis):
        _translate = QtCore.QCoreApplication.translate
        
        def set_translation(element, text):
            element.setText(_translate("ChronoRootAnalysis", text))

        def translate_main_elements():
            ChronoRootAnalysis.setWindowTitle(_translate("ChronoRootAnalysis", "ChronoRootAnalysis"))
            # Replace WIDTH and HEIGHT with the desired width and height of the window
            fixed_size = QtCore.QSize(810, 650)

            ChronoRootAnalysis.setMinimumSize(fixed_size)
            ChronoRootAnalysis.setMaximumSize(fixed_size)

            set_translation(self.loadVideo, "Select Video Folder")
            set_translation(self.loadProject, "Select Project Folder")
            set_translation(self.saveButton, "Save")

        def translate_labels():
            set_translation(self.label, "<html><head/><body><p align=\"center\">Raspberry Module</p></body></html>")
            set_translation(self.label_2, "<html><head/><body><p align=\"center\">Camera</p></body></html>")
            set_translation(self.label_3, "<html><head/><body><p align=\"center\">Plant Number</p></body></html>")
            set_translation(self.label_4, "<html><head/><body><p align=\"center\">Identifier</p></body></html>")
            set_translation(self.label_5, "(should be the raspberry number)")
            set_translation(self.label_6, "(should be the camera number)")
            set_translation(self.label_7, "(should be a number, to identify plant)")
            set_translation(self.label_8, "(variety identifier, e.g. WT, Col0)")
            set_translation(self.label_9, "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Analysis and postprocessing parameters</span></p></body></html>")
            set_translation(self.label_11, "<html><head/><body><p>Capture interval</p></body></html>")
            set_translation(self.label_12, "<html><head/><body><p>Set processing limit</p></body></html>")
            set_translation(self.label_26, "(in days, 0 means no limit)")
            set_translation(self.label_27, "(in minutes, usually 15 minutes)")
            set_translation(self.label_30, "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Individual plant root analysis</span></p></body></html>")
            set_translation(self.daysFieldConvex, "Days to report")
            set_translation(self.daysConvexText, "(Numbers separated by commas)")
            set_translation(self.daysAnglesText, "Days to report")
            set_translation(self.emergenceDistanceText, "Emergence distance")
            set_translation(self.emergenceDistanceTextExp, "(in millimeters, default: 2 mm)")
            set_translation(self.captureIntervalLabel, "<html><head/><body><p>Capture interval</p></body></html>")
            set_translation(self.processingLimitLabel, "<html><head/><body><p>Processing limit</p></body></html>")
            set_translation(self.emergenceDistanceLabel, "<html><head/><body><p>Emergence distance</p></body></html>")
            set_translation(self.emergenceDistanceExp, "(in millimeters, default: 2 mm)")
            
        def translate_buttons():
            set_translation(self.loadVideo, "Select Video Folder")
            set_translation(self.loadProject, "Select Project Folder")
            set_translation(self.saveButton, "Save")
            set_translation(self.loadConfigFileButton, "Load\nconfig json\nfrom file")
            set_translation(self.loadLastConfigButton, "Load\nprevious\nconfiguration")
            set_translation(self.saveImagesButton, "Save Cropped Images")
            set_translation(self.videoHasQRbutton, "Video has QR codes")
            set_translation(self.analysisButton, "Analyze Plant")
            set_translation(self.previewAnalysisButton, "Preview video")
            set_translation(self.PostProcessButton, "Process\nall plants")
            set_translation(self.saveImagesConvex, "Save images for each day")
            set_translation(self.doFPCA, "Perform Functional PCA on time series")
            set_translation(self.doConvex, "Do Convex hull analysis")
            set_translation(self.doFourier, "Evaluate Growth Speeds and perform Fourier Analysis")
            set_translation(self.doLateralAngles, "Do Lateral Root Angles Analysis")
            set_translation(self.PostProcessButton2, "Process\nall plants")
            set_translation(self.reportButton, "Generate report")
            set_translation(self.loadLastConfig2, "Load\nprevious\nconfiguration")
            set_translation(self.saveButton_2, "Save")
            set_translation(self.loadProject_2, "Select Project Folder")
            set_translation(self.refresh_button, "Refresh")
            set_translation(self.refresh_button_tab3, "Refresh")
            set_translation(self.refresh_button_tab5, "Refresh")
            set_translation(self.open_path_button, "Open Path")
            set_translation(self.averagePerPlantStats, "Average intervals before testing")
            set_translation(self.everyXhourText, "Time series stats interval (dt, in hours)")
            set_translation(self.everyXhourTextFourier, "Speeds stats interval (dt, in hours)")
            set_translation(self.everyXhourTextAngles, "First LR Tip Stats interval (dt, in hours)")
            set_translation(self.reportText, "Hypothesis testing uses Mann-Whitney test every dt interval. If selected, an average value will be used, or a step (i*dt) otherwise")
            set_translation(self.normFPCA, "Normalize FPCA Boxplots")
            set_translation(self.numComponentsFPCAText, "Number of components")
            
        def translate_tab_text():
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab1), _translate("ChronoRootAnalysis", "Plant Analysis"))
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab2), _translate("ChronoRootAnalysis", "Analysis Overview"))
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab4), _translate("ChronoRootAnalysis", "Generate Report"))
            self.tab_widget.setTabText(self.tab_widget.indexOf(self.tab5), _translate("ChronoRootAnalysis", "Report"))
            
        translate_main_elements()
        translate_labels()
        translate_buttons()
        translate_tab_text()
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_ChronoRootAnalysis()
    window.show()
    
    # Defer table and UI updates until after window is shown
    QtCore.QTimer.singleShot(100, window.refresh_table)
    QtCore.QTimer.singleShot(100, window.update_image_labels)
    QtCore.QTimer.singleShot(100, window.update_report_labels)
    
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()