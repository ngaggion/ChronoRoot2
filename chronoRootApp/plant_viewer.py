import os
import glob
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QFrame, QStyle,
                             QDialog, QGraphicsRectItem, QRubberBand)
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QSize, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen

from analysis.imageUtils.plot import draw_labeled_roi, draw_seed_marker, overlay_seg_mask


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

    bbox = conf.get('bounding box')

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
        val = QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(),
            event.x(), self.width()
        )
        self.setValue(val)
        super().mousePressEvent(event)


# --- CUSTOM GRAPHICS VIEW ---
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None, enable_roi=False, enable_point_pick=False):
        super().__init__(parent)
        self.enable_roi = enable_roi
        self.enable_point_pick = enable_point_pick
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(220, 220, 220))
        self._roi_origin = None
        self._rubber_band = None
        if enable_roi or enable_point_pick:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        zoomInFactor = 1.15
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        if self.enable_point_pick and event.button() == Qt.LeftButton:
            self.point_selected.emit(self.mapToScene(event.pos()))
            event.accept()
            return
        if self.enable_roi and event.button() == Qt.LeftButton:
            self._roi_origin = event.pos()
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self._rubber_band.setGeometry(QRect(self._roi_origin, QSize()))
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.enable_roi and self._roi_origin is not None and self._rubber_band is not None:
            self._rubber_band.setGeometry(QRect(self._roi_origin, event.pos()).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.enable_roi and event.button() == Qt.LeftButton and self._roi_origin is not None:
            rect = QRect(self._roi_origin, event.pos()).normalized()
            self._rubber_band.hide()
            self._roi_origin = None
            if rect.width() > 3 and rect.height() > 3:
                top_left = self.mapToScene(rect.topLeft())
                bottom_right = self.mapToScene(rect.bottomRight())
                scene_rect = QRectF(top_left, bottom_right).normalized()
                self.roi_selected.emit(scene_rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    roi_selected = pyqtSignal(QRectF)
    point_selected = pyqtSignal(QPointF)


def _cv2_to_qpixmap(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img.shape
    bytesPerLine = 3 * w
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    qImg = QImage(img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


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

        self.timeStep = conf.get('timeStep', 15)
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

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        controls = QFrame()
        controls.setFixedHeight(120)
        controls.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border-top: 2px solid #ccc; } "
            "QLabel { color: #333; border: none; }"
        )

        c_layout = QVBoxLayout(controls)
        c_layout.setContentsMargins(10, 5, 10, 5)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 16pt; font-weight: bold; color: #003366;")
        c_layout.addWidget(self.lbl_info)

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

        btn_style = (
            "QPushButton { font-size: 11pt; font-weight: bold; background-color: #e1e1e1; "
            "border: 1px solid #adadad; border-radius: 4px; } "
            "QPushButton:hover { background-color: #d4d4d4; } "
            "QPushButton:pressed { background-color: #c0c0c0; }"
        )
        for b in [self.btn_play, self.btn_seg]:
            b.setStyleSheet(btn_style)
            h_layout.addWidget(b)

        c_layout.addLayout(h_layout)
        layout.addWidget(controls)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.colors = {
            1: (0, 0, 255),
            2: (0, 255, 0),
            3: (255, 0, 0),
            4: (0, 255, 255),
        }

        self.update_display()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_image)

    def fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def cv2_to_qpixmap(self, img):
        return _cv2_to_qpixmap(img)

    def update_display(self):
        if self.idx >= len(self.images):
            return

        img = cv2.imread(self.images[self.idx])
        if img is None:
            return

        if self.bbox and len(self.bbox) == 4:
            y1, y2, x1, x2 = self.bbox
            h, w = img.shape[:2]
            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                img = img[y1:y2, x1:x2]

        if self.use_seg and self.segFiles and self.idx < len(self.segFiles):
            seg = cv2.imread(self.segFiles[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None:
                if len(seg.shape) == 3:
                    if seg.shape[2] == 4:
                        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)
                    img = cv2.addWeighted(img, 1.0, seg, 0.7, 0)
                elif len(seg.shape) == 2:
                    img = overlay_seg_mask(img, seg, self.colors)

        self.pixmap_item.setPixmap(self.cv2_to_qpixmap(img))
        self.lbl_info.setText(
            f"Day: {self.days[self.idx]}   Time: {self.hours[self.idx]:02d}:{self.minutes[self.idx]:02d}"
        )

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


class PlantROISelectorWindow(QDialog):
    """Single-plant ROI selector; prior ROIs drawn on the frame via plot.draw_labeled_roi."""

    def __init__(self, images, seg_files, previous_rois=None, time_delta=15, parent=None):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.previous_rois = previous_rois or []
        self.time_delta = time_delta
        self.pending_roi = None
        self._selected_roi = None
        self.image_width = 0
        self.image_height = 0
        self.use_seg = False
        self.idx = len(self.images) - 1 if self.images else 0

        self.setWindowTitle("Select Plant Region")
        self.resize(950, 850)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(enable_roi=True)
        self.view.setScene(self.scene)
        self.view.roi_selected.connect(self._on_roi_selected)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._set_frame)
        layout.addWidget(self.slider)

        nav_layout = QHBoxLayout()
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self._toggle_seg)
        nav_layout.addWidget(self.btn_seg)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_confirm.clicked.connect(self._confirm_current)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.setEnabled(False)
        action_layout.addWidget(self.btn_confirm)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self._update_display()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_current()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self._clear_pending_overlay()
            self.btn_confirm.setEnabled(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def get_roi(self):
        return self._selected_roi if self.result() == QDialog.Accepted else None

    def _set_frame(self, val):
        if val != self.idx:
            self._clear_pending_overlay()
            self.btn_confirm.setEnabled(False)
        self.idx = val
        self._update_display()

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display()

    def _load_frame(self):
        if not self.images:
            return None
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None
        self.image_height, self.image_width = img.shape[:2]

        for label, x1, y1, x2, y2 in self.previous_rois:
            draw_labeled_roi(img, x1, y1, x2, y2, label)

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None and len(seg.shape) == 2:
                colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
                img = overlay_seg_mask(img, seg, colors)

        return img

    def _frame_time_text(self):
        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        return f"Frame {self.idx + 1}/{len(self.images)}  |  Day {days}  Time {hours:02d}:{int(minutes):02d}"

    def _update_display(self):
        img = self._load_frame()
        if img is None:
            self.lbl_info.setText("Failed to load frame")
            return

        self.pixmap_item.setPixmap(_cv2_to_qpixmap(img))
        self.scene.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        suffix = (
            "Press Enter or Confirm to apply this selection."
            if self.pending_roi
            else "Drag a rectangle on the image. Press Enter to confirm."
        )
        self.lbl_info.setText(f"{self._frame_time_text()}  |  {suffix}")
        QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _clear_pending_overlay(self):
        if self.pending_roi is None:
            return
        _, rect_item = self.pending_roi
        self.scene.removeItem(rect_item)
        self.pending_roi = None

    def _on_roi_selected(self, scene_rect):
        self._clear_pending_overlay()
        if scene_rect.width() < 1 or scene_rect.height() < 1:
            return
        pen = QPen(QColor(255, 255, 255), 6)
        rect_item = QGraphicsRectItem(scene_rect)
        rect_item.setPen(pen)
        self.scene.addItem(rect_item)
        self.pending_roi = (scene_rect, rect_item)
        self.btn_confirm.setEnabled(True)
        self.lbl_info.setText(
            f"{self._frame_time_text()}  |  Press Enter or Confirm to apply this selection."
        )

    def _confirm_current(self):
        if not self.pending_roi:
            return
        scene_rect, rect_item = self.pending_roi
        x1 = int(max(0, min(scene_rect.left(), self.image_width - 1)))
        y1 = int(max(0, min(scene_rect.top(), self.image_height - 1)))
        x2 = int(max(0, min(scene_rect.right(), self.image_width)))
        y2 = int(max(0, min(scene_rect.bottom(), self.image_height)))
        if x2 <= x1 or y2 <= y1:
            self.lbl_info.setText("Invalid selection. Please draw a larger rectangle.")
            return
        self._selected_roi = (x1, y1, x2, y2)
        self.scene.removeItem(rect_item)
        self.pending_roi = None
        self.accept()


class SeedSelectorWindow(QDialog):
    """Pick root origin inside the selected plant ROI."""

    def __init__(self, images, seg_files, bbox, conf, parent=None):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.bbox = bbox
        self.time_delta = conf.get('timeStep', 15)
        self.use_seg = False
        self.seed_pos = None
        self.idx = len(self.images) - 1 if self.images else 0

        self.setWindowTitle("Select Root Origin")
        self.resize(700, 900)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(enable_point_pick=True)
        self.view.setScene(self.scene)
        self.view.point_selected.connect(self._on_point_selected)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)

        self.slider = ClickJumpSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._set_frame)
        layout.addWidget(self.slider)

        nav_layout = QHBoxLayout()
        self.btn_seg = QPushButton("Toggle Segmentation")
        self.btn_seg.clicked.connect(self._toggle_seg)
        nav_layout.addWidget(self.btn_seg)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        action_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_confirm.clicked.connect(self._confirm)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.setEnabled(False)
        action_layout.addWidget(self.btn_confirm)
        action_layout.addWidget(self.btn_cancel)
        layout.addLayout(action_layout)

        self._update_display()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)

    def get_seed(self):
        if self.result() == QDialog.Accepted and self.seed_pos is not None:
            return [int(self.seed_pos[0]), int(self.seed_pos[1])]
        return None

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display()

    def _set_frame(self, val):
        self.idx = val
        self._update_display()

    def _load_frame(self):
        if not self.images or len(self.bbox) != 4:
            return None
        y1, y2, x1, x2 = self.bbox
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None
        h, w = img.shape[:2]
        if not (0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w):
            return None
        img = img[y1:y2, x1:x2].copy()

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], 0)
            if seg is not None:
                seg_crop = seg[y1:y2, x1:x2]
                colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
                img = overlay_seg_mask(img, seg_crop, colors)

        if self.seed_pos is not None:
            draw_seed_marker(img, int(self.seed_pos[0]), int(self.seed_pos[1]))
        return img

    def _on_point_selected(self, scene_pos):
        h, w = self.bbox[1] - self.bbox[0], self.bbox[3] - self.bbox[2]
        x = max(0, min(scene_pos.x(), w - 1))
        y = max(0, min(scene_pos.y(), h - 1))
        self.seed_pos = (x, y)
        self.btn_confirm.setEnabled(True)
        self._update_display()

    def _update_display(self):
        img = self._load_frame()
        if img is None:
            self.lbl_info.setText("Failed to load frame")
            return

        self.pixmap_item.setPixmap(_cv2_to_qpixmap(img))
        self.scene.setSceneRect(QRectF(self.pixmap_item.pixmap().rect()))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        suffix = "Click root origin. Enter to confirm."
        if self.seed_pos is not None:
            suffix = f"Root at ({int(self.seed_pos[0])}, {int(self.seed_pos[1])}). Enter to confirm."
        self.lbl_info.setText(
            f"Frame {self.idx + 1}/{len(self.images)}  |  Day {days}  Time {hours:02d}:{int(minutes):02d}  |  {suffix}"
        )
        QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _confirm(self):
        if self.seed_pos is not None:
            self.accept()
