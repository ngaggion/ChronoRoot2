import os
import glob
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QFrame, QStyle)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

# --- UTILS ---
def loadPath(path, ext="*.png"):
    return sorted(glob.glob(os.path.join(path, ext)))

def load_plant_data(plant_path):
    """
    Helper function to load data from the plant directory.
    Returns (images, segs, bbox, conf) or raises FileNotFoundError.
    """
    json_path = os.path.join(plant_path, 'metadata.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError("metadata.json not found")

    with open(json_path, 'r') as f:
        conf = json.load(f)

    bbox = conf.get('bounding box') # .get() allows it to be None safely
    
    imagePath = conf.get('ImagePath')
    segPath = os.path.join(plant_path, "Images", "SegMulti")
    
    images = loadPath(imagePath, ext="*.png") if imagePath and os.path.exists(imagePath) else []
    segs = loadPath(segPath, ext="*.png")

    if not images:
        local_img_path = os.path.join(plant_path, "Images") 
        if os.path.exists(local_img_path):
             images = loadPath(local_img_path, ext="*.png")

    if not images:
        raise FileNotFoundError(f"No images found. Checked: {imagePath}")

    return images, segs, bbox, conf

# --- CUSTOM SLIDER FOR CLICK-TO-JUMP ---
class ClickJumpSlider(QSlider):
    def mousePressEvent(self, event):
        # Calculate the value based on the click position
        val = QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), 
            event.x(), self.width()
        )
        self.setValue(val)
        # Call the parent event to handle dragging immediately after the click
        super().mousePressEvent(event)
        
# --- CUSTOM GRAPHICS VIEW ---
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(220, 220, 220))

    def wheelEvent(self, event):
        zoomInFactor = 1.15
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

# --- MAIN WINDOW CLASS ---
class ChronoViewWindow(QMainWindow):
    def __init__(self, images, segFiles, bbox, conf, parent=None):
        super().__init__(parent)
        self.images = images
        self.segFiles = segFiles
        self.bbox = bbox
        self.conf = conf
        
        self.n = min(len(images), len(segFiles)) if segFiles else len(images)
        self.idx = 0
        self.playing = False
        self.use_seg = False
        
        # Time Calculations
        self.timeStep = conf.get('timeStep', 15)
        # Convert processingLimit to integer frames if it exists
        if conf.get('processingLimit', 0) != 0:
             limit_days = int(conf['processingLimit'])
             frames_per_day = (24 * 60) // self.timeStep
             max_frames = limit_days * frames_per_day
             self.n = min(self.n, max_frames)

        time_arr = np.arange(0, self.n * self.timeStep, self.timeStep)
        self.days = (time_arr // 1440).astype('int')
        self.hours = ((time_arr / 60) % 24).astype('int')
        self.minutes = (time_arr % 60).astype('int')

        self.setWindowTitle("ChronoRoot Viewer")
        self.resize(900, 800)

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 1. View
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        # 2. Controls
        controls = QFrame()
        controls.setFixedHeight(120) 
        controls.setStyleSheet("QFrame { background-color: #f0f0f0; border-top: 2px solid #ccc; } QLabel { color: #333; border: none; }")
        
        c_layout = QVBoxLayout(controls)
        c_layout.setContentsMargins(10, 5, 10, 5)
        
        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 16pt; font-weight: bold; color: #003366;")
        c_layout.addWidget(self.lbl_info)
        
        # Use Custom ClickJumpSlider instead of QSlider
        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.setRange(0, self.n - 1)
        self.slider.valueChanged.connect(self.set_frame)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #5c5c5c;
                width: 18px;
                height: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        c_layout.addWidget(self.slider)
        
        h_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedHeight(35)
        self.btn_play.clicked.connect(self.toggle_play)
        
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.setFixedHeight(35)
        self.btn_seg.clicked.connect(self.toggle_seg)
        
        btn_style = "QPushButton { font-size: 11pt; font-weight: bold; background-color: #e1e1e1; border: 1px solid #adadad; border-radius: 4px; } QPushButton:hover { background-color: #d4d4d4; } QPushButton:pressed { background-color: #c0c0c0; }"
        for b in [self.btn_play, self.btn_seg]:
            b.setStyleSheet(btn_style)
            h_layout.addWidget(b)
            
        c_layout.addLayout(h_layout)
        layout.addWidget(controls)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # Color Map for Segmentation (B, G, R)
        self.colors = {
            1: (0, 0, 255),     # Red
            2: (0, 255, 0),     # Green
            3: (255, 0, 0),     # Blue
            4: (0, 255, 255),   # Yellow
        }

        self.update_display()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_image)

    def fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def cv2_to_qpixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytesPerLine = 3 * w
        if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
        qImg = QImage(img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg)

    def update_display(self):
        if self.idx >= len(self.images): return
        
        # 1. Load Main Image
        img = cv2.imread(self.images[self.idx])
        if img is None: return 

        # 2. Crop Image (if bbox exists)
        if self.bbox and len(self.bbox) == 4:
            y1, y2, x1, x2 = self.bbox
            h, w = img.shape[:2]
            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                img = img[y1:y2, x1:x2]
        
        # 3. Segmentation Overlay
        if self.use_seg and self.segFiles and self.idx < len(self.segFiles):
            # IMPORTANT: Load as UNCHANGED to detect if it's grayscale (2D) or Color (3D)
            seg = cv2.imread(self.segFiles[self.idx], cv2.IMREAD_UNCHANGED)
            
            if seg is not None:                
                # Case 1: Multi-Channel (Review Mode - Already Colored)
                if len(seg.shape) == 3:
                    # Handle Alpha Channel if present
                    if seg.shape[2] == 4:
                        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)
                    
                    # Original logic used simple addition, which works best for black-background overlays
                    img = cv2.addWeighted(img, 1.0, seg, 0.7, 0)

                # Case 2: Single-Channel (Preview Mode - Integer Masks)
                elif len(seg.shape) == 2:
                    # Create an empty color overlay
                    color_mask = np.zeros_like(img)
                    
                    # Apply colors based on integer class IDs
                    for val, color in self.colors.items():
                        color_mask[seg == val] = color
                    
                    # Handle extra classes
                    color_mask[seg >= 5] = (255, 0, 255) # Purple
                    
                    # Weighted Blend
                    img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
            
        self.pixmap_item.setPixmap(self.cv2_to_qpixmap(img))
        self.lbl_info.setText(f"Day: {self.days[self.idx]}   Time: {self.hours[self.idx]:02d}:{self.minutes[self.idx]:02d}")
        
        self.slider.blockSignals(True)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

    def set_frame(self, val):
        self.idx = val
        self.update_display()

    def next_frame(self):
        self.idx = (self.idx + 1) % self.n
        self.update_display()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.timer.start(50)
            self.btn_play.setText("Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("Play")

    def toggle_seg(self):
        self.use_seg = not self.use_seg
        self.update_display()