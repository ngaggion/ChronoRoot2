import os
import glob
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGraphicsView,
                             QGraphicsScene, QGraphicsPixmapItem, QFrame, QStyle,
                             QDialog, QGraphicsRectItem, QGraphicsTextItem,
                             QRubberBand, QDialogButtonBox)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, QRectF, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QBrush

# --- UTILS ---
def loadPath(path, ext="*.png"):
    return sorted(glob.glob(os.path.join(path, ext)))


def _find_seg_folder(segmentation_dir: str) -> str:
    for name in ('Ensemble', 'Seg'):
        path = os.path.join(segmentation_dir, name)
        if os.path.isdir(path):
            return path
    return os.path.join(segmentation_dir, 'Ensemble')


def _read_input_path_from_metadata(metadata_path: str):
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    input_path = metadata.get("input_path")
    if input_path and os.path.exists(input_path):
        return input_path
    return None


def _resolve_images_from_metadata(video_dir: str, segmentation_dir: str, user_path: str):
    """Resolve image folder via segmentation_metadata.json when needed."""
    images = loadPath(video_dir, ext="*.png")
    if images:
        return video_dir, images

    metadata_candidates = [
        os.path.join(segmentation_dir, 'segmentation_metadata.json'),
        os.path.join(user_path, 'Segmentation', 'segmentation_metadata.json'),
        os.path.join(user_path, 'segmentation_metadata.json'),
    ]
    for metadata_path in metadata_candidates:
        input_path = _read_input_path_from_metadata(metadata_path)
        if input_path:
            return input_path, loadPath(input_path, ext="*.png")
    return video_dir, images


def resolve_screening_paths(user_path: str):
    """
    Resolve image and segmentation directories from the user-selected folder.

    Supports:
    - Classic layout: user_path with Segmentation/Ensemble and optional metadata
    - Segmentation root: user_path is the segmentation output (Ensemble/ at root)
    - Co-located images with Segmentation/ subfolder

    Returns (video_dir, segmentation_dir, images, seg_files).
    """
    user_path = os.path.abspath(user_path)
    seg_sub = os.path.join(user_path, 'Segmentation')
    classic_ensemble = os.path.join(seg_sub, 'Ensemble')
    classic_meta = os.path.join(seg_sub, 'segmentation_metadata.json')
    root_ensemble = os.path.join(user_path, 'Ensemble')
    root_meta = os.path.join(user_path, 'segmentation_metadata.json')

    if os.path.isdir(classic_ensemble) or os.path.exists(classic_meta):
        segmentation_dir = seg_sub
        video_dir = user_path
    elif os.path.isdir(root_ensemble) or os.path.exists(root_meta):
        segmentation_dir = user_path
        video_dir = user_path
    else:
        segmentation_dir = seg_sub
        video_dir = user_path

    video_dir, images = _resolve_images_from_metadata(video_dir, segmentation_dir, user_path)

    seg_path = _find_seg_folder(segmentation_dir)
    seg_files = loadPath(seg_path, ext="*.png") if os.path.exists(seg_path) else []

    n = min(len(images), len(seg_files)) if seg_files else len(images)
    images = images[:n]
    seg_files = seg_files[:n]

    return video_dir, segmentation_dir, images, seg_files


def get_screening_images(video_dir: str, segmentation_dir: str):
    """Return paired image and segmentation path lists when paths are already resolved."""
    resolved_video_dir, images = _resolve_images_from_metadata(
        video_dir, segmentation_dir, video_dir
    )

    seg_path = _find_seg_folder(segmentation_dir)
    seg_files = loadPath(seg_path, ext="*.png") if os.path.exists(seg_path) else []

    n = min(len(images), len(seg_files)) if seg_files else len(images)
    images = images[:n]
    seg_files = seg_files[:n]

    return images, seg_files, resolved_video_dir


def load_screening_sequence(user_path: str, segmentation_dir: str = None, time_delta: float = 15):
    """
    Load screening preview/ROI data.
    Returns (images, seg_files, conf).

    If segmentation_dir is None, user_path is resolved via resolve_screening_paths.
    Otherwise video_dir=user_path and segmentation_dir are used directly (CLI compat).
    """
    if segmentation_dir is None:
        video_dir, segmentation_dir, images, seg_files = resolve_screening_paths(user_path)
    else:
        images, seg_files, video_dir = get_screening_images(user_path, segmentation_dir)

    if not images:
        raise FileNotFoundError(
            f"No images found for {user_path}. "
            "Check the folder or segmentation metadata input_path."
        )
    conf = {"timeStep": time_delta}
    return images, seg_files, conf


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
    def __init__(self, parent=None, enable_roi: bool = False):
        super().__init__(parent)
        self.enable_roi = enable_roi
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(220, 220, 220))
        self._roi_origin = None
        self._rubber_band = None
        if enable_roi:
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
        self.load_error = False

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
        self.slider.setRange(0, max(self.n - 1, 0))
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytesPerLine = 3 * w
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        qImg = QImage(img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg)

    def load_current_image(self):
        if self.idx >= len(self.images):
            return None
        img = cv2.imread(self.images[self.idx])
        if img is None:
            self.load_error = True
            return None
        self.load_error = False

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
                    img = cv2.addWeighted(img, 1.0, seg, 0.85, 0)
                elif len(seg.shape) == 2:
                    img = _overlay_label_segmentation(img, seg, self.colors)
        return img

    def update_display(self):
        if self.idx >= len(self.images):
            return

        img = self.load_current_image()
        if img is None:
            self.lbl_info.setText("Failed to load frame")
            return

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


ROI_GROUP_COLORS = [
    QColor(255, 80, 80),
    QColor(80, 180, 80),
    QColor(80, 120, 255),
    QColor(255, 200, 60),
    QColor(200, 80, 255),
    QColor(80, 220, 220),
]

ROI_BOX_PEN_WIDTH = 6
ROI_PENDING_PEN_WIDTH = 6

def _overlay_label_segmentation(img, seg, colors):
    """Blend saturated class colors onto segmented pixels only."""
    color_mask = np.zeros_like(img)
    for val, color in colors.items():
        color_mask[seg == val] = color
    color_mask[seg >= 5] = (255, 0, 255)
    labeled = seg > 0
    if not np.any(labeled):
        return img
    out = img.copy()
    blended = cv2.addWeighted(img, 1.0, color_mask, 0.85, 0)
    out[labeled] = blended[labeled]
    return out


class GroupROISelectorWindow(QDialog):
    """Modal ROI selector built on the plant viewer controls."""

    def __init__(self, images, seg_files, group_names, time_delta=15, parent=None):
        super().__init__(parent)
        self.images = images
        self.seg_files = seg_files
        self.group_names = group_names
        self.time_delta = time_delta
        self.confirmed_groups = {}
        self.current_group_index = 0
        self.pending_roi = None
        self.roi_overlays = []
        self.image_width = 0
        self.image_height = 0

        self.setWindowTitle("Select Group Regions")
        self.resize(950, 850)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)

        self.lbl_group = QLabel()
        self.lbl_group.setAlignment(Qt.AlignCenter)
        self.lbl_group.setStyleSheet("font-size: 14pt; font-weight: bold; color: #003366;")
        layout.addWidget(self.lbl_group)

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
        self.btn_previous = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_cancel = QPushButton("Cancel Analysis")
        self.btn_previous.clicked.connect(self._go_previous)
        self.btn_next.clicked.connect(self._go_next)
        self.btn_confirm.clicked.connect(self._confirm_current)
        self.btn_cancel.clicked.connect(self.reject)
        for btn in [self.btn_previous, self.btn_next, self.btn_confirm, self.btn_cancel]:
            action_layout.addWidget(btn)
        layout.addLayout(action_layout)

        self.use_seg = False
        self.idx = len(self.images) - 1 if self.images else 0
        self._update_group_label()
        self._update_display()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_current()
            event.accept()
            return
        if key == Qt.Key_Escape:
            self._clear_pending_overlay()
            self._update_confirm_button()
            event.accept()
            return
        if key == Qt.Key_Left and self.btn_previous.isEnabled():
            self._go_previous()
            event.accept()
            return
        if key == Qt.Key_Right and self.btn_next.isEnabled():
            self._go_next()
            event.accept()
            return
        super().keyPressEvent(event)

    def get_group_rois(self):
        if self.result() == QDialog.Accepted:
            return self.confirmed_groups
        return None

    def _all_groups_confirmed(self):
        return len(self.confirmed_groups) == len(self.group_names)

    def _current_group_name(self):
        return self.group_names[self.current_group_index]

    def _update_group_label(self):
        if self.current_group_index >= len(self.group_names):
            return
        name = self._current_group_name()
        pos = f"({self.current_group_index + 1} of {len(self.group_names)})"
        if name in self.confirmed_groups:
            self.lbl_group.setText(f"Update ROI for: {name}  {pos}")
        else:
            self.lbl_group.setText(f"Select ROI for: {name}  {pos}")
        self._update_confirm_button()
        self._update_nav_buttons()

    def _update_confirm_button(self):
        if self._all_groups_confirmed() and not self.pending_roi:
            self.btn_confirm.setText("Finish")
            self.btn_confirm.setEnabled(True)
        elif self.pending_roi:
            self.btn_confirm.setText("Confirm Selection")
            self.btn_confirm.setEnabled(True)
        else:
            self.btn_confirm.setText("Confirm Selection")
            self.btn_confirm.setEnabled(False)

    def _update_nav_buttons(self):
        self.btn_previous.setEnabled(self.current_group_index > 0)
        current_name = self._current_group_name()
        self.btn_next.setEnabled(
            current_name in self.confirmed_groups
            and self.current_group_index < len(self.group_names) - 1
        )

    def _go_previous(self):
        if self.current_group_index == 0:
            return
        self._clear_pending_overlay()
        self.current_group_index -= 1
        self._redraw_confirmed_overlays()
        self._update_group_label()
        self._refresh_info_line()

    def _go_next(self):
        current_name = self._current_group_name()
        if current_name not in self.confirmed_groups:
            return
        if self.current_group_index >= len(self.group_names) - 1:
            return
        self._clear_pending_overlay()
        self.current_group_index += 1
        self._redraw_confirmed_overlays()
        self._update_group_label()
        self._refresh_info_line()

    def _set_frame(self, val):
        if val != self.idx:
            self._clear_pending_overlay()
            self._update_confirm_button()
        self.idx = val
        self._update_display()

    def _toggle_seg(self):
        self.use_seg = not self.use_seg
        self._update_display()

    def _load_current_image(self):
        if not self.images:
            return None
        img = cv2.imread(self.images[self.idx])
        if img is None:
            return None

        self.image_height, self.image_width = img.shape[:2]

        if self.use_seg and self.seg_files and self.idx < len(self.seg_files):
            seg = cv2.imread(self.seg_files[self.idx], cv2.IMREAD_UNCHANGED)
            if seg is not None and len(seg.shape) == 2:
                colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
                img = _overlay_label_segmentation(img, seg, colors)
        return img

    def _clear_roi_overlays(self):
        for overlay in self.roi_overlays:
            for key in ('rect', 'bg', 'label'):
                item = overlay.get(key)
                if item is not None:
                    self.scene.removeItem(item)
        self.roi_overlays = []

    def _add_roi_overlay(self, group_name, x1, y1, x2, y2, color):
        w = x2 - x1
        h = y2 - y1
        rect_item = QGraphicsRectItem(QRectF(x1, y1, w, h))
        rect_item.setPen(QPen(color, ROI_BOX_PEN_WIDTH))
        self.scene.addItem(rect_item)

        label = QGraphicsTextItem(group_name)
        font = QFont()
        font.setBold(True)
        point_size = min(32, max(24, int(min(w, h) / 6)))
        font.setPointSize(point_size)
        label.setFont(font)
        label.setDefaultTextColor(QColor(255, 255, 255))

        text_rect = label.boundingRect()
        while text_rect.width() > w * 0.9 and font.pointSize() > 10:
            font.setPointSize(font.pointSize() - 1)
            label.setFont(font)
            text_rect = label.boundingRect()

        tx = x1 + (w - text_rect.width()) / 2
        ty = y1 + (h - text_rect.height()) / 2

        bg = QGraphicsRectItem(text_rect.adjusted(-4, -2, 4, 2))
        bg.setBrush(QBrush(QColor(0, 0, 0, 160)))
        bg.setPen(QPen(Qt.NoPen))
        bg.setPos(tx - 4, ty - 2)
        label.setPos(tx, ty)

        self.scene.addItem(bg)
        self.scene.addItem(label)

        return {'rect': rect_item, 'bg': bg, 'label': label}

    def _redraw_confirmed_overlays(self):
        self._clear_roi_overlays()
        for idx, group_name in enumerate(self.group_names):
            if group_name not in self.confirmed_groups:
                continue
            x1, y1, x2, y2 = self.confirmed_groups[group_name]
            color = ROI_GROUP_COLORS[idx % len(ROI_GROUP_COLORS)]
            self.roi_overlays.append(
                self._add_roi_overlay(group_name, x1, y1, x2, y2, color)
            )

    def _frame_info_suffix(self):
        if self.pending_roi:
            return "Press Enter or Confirm to apply this selection."
        if self._all_groups_confirmed():
            return "All regions set — press Enter or Finish to complete"
        if self._current_group_name() in self.confirmed_groups:
            return (
                "Region already set — draw a new rectangle to update it, "
                "or press Next to continue."
            )
        return "Drag a rectangle on the image. Use the slider to change frame. Press Enter to confirm."

    def _refresh_info_line(self):
        if not self.images:
            return
        minutes = (self.idx * self.time_delta) % 60
        hours = int((self.idx * self.time_delta / 60) % 24)
        days = int(self.idx * self.time_delta // 1440)
        self.lbl_info.setText(
            f"Frame {self.idx + 1}/{len(self.images)}  |  "
            f"Day {days}  Time {hours:02d}:{int(minutes):02d}  |  "
            f"{self._frame_info_suffix()}"
        )

    def _update_display(self):
        if not self.images:
            self.lbl_info.setText("No images available")
            return

        img = self._load_current_image()
        if img is None:
            self.lbl_info.setText("Failed to load frame")
            return

        pixmap = self._cv2_to_qpixmap(img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.images) - 1)
        self.slider.setValue(self.idx)
        self.slider.blockSignals(False)

        self._redraw_confirmed_overlays()
        self._refresh_info_line()
        self._update_nav_buttons()
        QTimer.singleShot(0, self._fit_image)

    def _fit_image(self):
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def _cv2_to_qpixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytesPerLine = 3 * w
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        qImg = QImage(img.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg)

    def _clear_pending_overlay(self):
        if self.pending_roi is None:
            return
        if isinstance(self.pending_roi, tuple):
            _, rect_item = self.pending_roi
            self.scene.removeItem(rect_item)
        elif isinstance(self.pending_roi, QGraphicsRectItem):
            self.scene.removeItem(self.pending_roi)
        self.pending_roi = None

    def _on_roi_selected(self, scene_rect: QRectF):
        self._clear_pending_overlay()
        if scene_rect.width() < 1 or scene_rect.height() < 1:
            return

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(ROI_PENDING_PEN_WIDTH)
        rect_item = QGraphicsRectItem(scene_rect)
        rect_item.setPen(pen)
        self.scene.addItem(rect_item)
        self.pending_roi = (scene_rect, rect_item)
        self._update_confirm_button()
        self._refresh_info_line()

    def _confirm_current(self):
        if self._all_groups_confirmed() and not self.pending_roi:
            self.accept()
            return
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

        group_name = self._current_group_name()
        is_update = group_name in self.confirmed_groups
        self.confirmed_groups[group_name] = (x1, y1, x2, y2)

        self.scene.removeItem(rect_item)
        self.pending_roi = None

        self._redraw_confirmed_overlays()

        if is_update:
            self._update_group_label()
            self._refresh_info_line()
            return

        if self._all_groups_confirmed():
            self.accept()
            return

        self.current_group_index += 1
        self._update_group_label()
        self._refresh_info_line()
