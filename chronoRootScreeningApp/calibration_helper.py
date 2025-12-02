import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QMessageBox,
                           QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import argparse
import PIL.Image

from typing import List, Tuple

def loadPath(path: str, ext: str = "*") -> List[str]:
    """
    Helper function to load paths with specific extensions.
    Matches the functionality expected by getImages()
    """
    import glob
    if not os.path.exists(path):
        return []
    return sorted(glob.glob(os.path.join(path, ext)))

def getImages(video_dir: str) -> Tuple[List[str], List[str]]:
    """
    Returns lists of image paths and segmentation paths
    """
    # Check if the directory exists and contains png files
    images = loadPath(video_dir, ext="*.png")
    
    # Look for segmentation files
    seg_path = os.path.join(video_dir, 'Segmentation', 'Ensemble')
    if not os.path.exists(seg_path):
        seg_path = os.path.join(video_dir, 'Seg')
    
    seg_files = loadPath(seg_path, ext="*.png")
    
    # Ensure we have matching numbers of files
    n = min(len(images), len(seg_files))
    images = images[:n]
    seg_files = seg_files[:n]
    
    return images, seg_files

class ZoomableImage(QLabel):
    point_selected = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.points = []
        self.current_frame = None
        self.zoom_factor = 1.0
        self.setMouseTracking(True)
        
        # For panning
        self.panning = False
        self.last_pos = None
        self.setAlignment(Qt.AlignCenter)

    def set_frame(self, frame):
        self.current_frame = frame
        
        # Calculate initial zoom factor to fit the window
        if frame is not None:
            scroll_area = self.parent().parent()
            if scroll_area:
                view_width = scroll_area.viewport().width() - 20
                view_height = scroll_area.viewport().height() - 20
                
                frame_height, frame_width = frame.shape[:2]
                width_ratio = view_width / frame_width
                height_ratio = view_height / frame_height
                
                # Use the smaller ratio to fit the image
                self.zoom_factor = min(width_ratio, height_ratio)
        
        self.update_display()

    def update_display(self):
        if self.current_frame is None:
            return
            
        # Convert the original frame to QImage
        height, width, _ = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create a pixmap and scale it
        pixmap = QPixmap.fromImage(q_image)
        scaled_width = int(width * self.zoom_factor)
        scaled_height = int(height * self.zoom_factor)
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Create a new pixmap for drawing
        final_pixmap = QPixmap(scaled_pixmap)
        painter = QPainter(final_pixmap)
        
        # Set up the pen for drawing
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(max(2, int(2 * self.zoom_factor)))
        painter.setPen(pen)
        
        # Draw points
        point_radius = max(5, int(5 * self.zoom_factor))
        for px, py in self.points:
            scaled_x = int(px * self.zoom_factor)
            scaled_y = int(py * self.zoom_factor)
            painter.drawEllipse(scaled_x - point_radius, scaled_y - point_radius, 
                              point_radius * 2, point_radius * 2)
        
        # Draw line and distance if we have two points
        if len(self.points) == 2:
            p1 = self.points[0]
            p2 = self.points[1]
            
            # Draw line
            p1_scaled = (int(p1[0] * self.zoom_factor), int(p1[1] * self.zoom_factor))
            p2_scaled = (int(p2[0] * self.zoom_factor), int(p2[1] * self.zoom_factor))
            painter.drawLine(p1_scaled[0], p1_scaled[1], p2_scaled[0], p2_scaled[1])
            
            # Calculate distance
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Draw distance text
            mid_x = (p1_scaled[0] + p2_scaled[0]) // 2
            mid_y = (p1_scaled[1] + p2_scaled[1]) // 2
            
            font = painter.font()
            font.setPixelSize(max(12, int(12 * self.zoom_factor)))
            painter.setFont(font)
            text = f'{distance:.1f}px'
            painter.drawText(mid_x + 5, mid_y - 5, text)
        
        painter.end()
        self.setPixmap(final_pixmap)

    def get_image_coordinates(self, pos):
        """Convert mouse coordinates to original image coordinates"""
        if not self.pixmap():
            return None

        # Get label position relative to scroll area viewport
        scroll_area = self.parent().parent()
        viewport = scroll_area.viewport()
        label_pos = self.mapTo(viewport, QPoint(0, 0))

        # Calculate image position within label (accounting for centering)
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()
        label_width = self.width()
        label_height = self.height()
        
        x_offset = (label_width - pixmap_width) // 2
        y_offset = (label_height - pixmap_height) // 2

        # Adjust position for scrolling and centering
        adjusted_x = pos.x() - x_offset + scroll_area.horizontalScrollBar().value()
        adjusted_y = pos.y() - y_offset + scroll_area.verticalScrollBar().value()

        # Convert back to original image coordinates
        original_x = int(adjusted_x / self.zoom_factor)
        original_y = int(adjusted_y / self.zoom_factor)

        # Check if point is within image bounds
        if 0 <= original_x < self.current_frame.shape[1] and \
           0 <= original_y < self.current_frame.shape[0]:
            return (original_x, original_y)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.get_image_coordinates(event.pos())
            if pos and len(self.points) < 2:
                self.points.append(pos)
                self.point_selected.emit(pos)
                self.update_display()
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            
    def mouseMoveEvent(self, event):
        if self.panning and self.last_pos:
            delta = event.pos() - self.last_pos
            scrollarea = self.parent().parent()
            vbar = scrollarea.verticalScrollBar()
            hbar = scrollarea.horizontalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            self.last_pos = event.pos()
            
    def wheelEvent(self, event):
        # Zoom in/out with mouse wheel
        old_factor = self.zoom_factor
        
        if event.angleDelta().y() > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
            
        # Limit zoom range
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor))
        
        # Update display with new zoom factor
        self.update_display()
        
    def clear_points(self):
        self.points = []
        self.update_display()



class CalibrationHelper(QMainWindow):
    def __init__(self, video_dir):
        super().__init__()
        self.video_dir = video_dir
        self.initUI()
        self.load_first_frame()
        
    def initUI(self):
        self.setWindowTitle('Calibration Helper')
        self.setGeometry(100, 100, 1000, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        instructions = QLabel(
            "Left click to select two points.\n"
            "Mouse wheel to zoom in/out.\n"
            "Right click and drag to pan when zoomed."
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.frame_widget = ZoomableImage()
        self.frame_widget.point_selected.connect(self.on_point_selected)
        scroll.setWidget(self.frame_widget)
        layout.addWidget(scroll)
        
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton('Clear Points')
        self.clear_btn.clicked.connect(self.frame_widget.clear_points)
        button_layout.addWidget(self.clear_btn)
        
        self.copy_btn = QPushButton('Copy Distance')
        self.copy_btn.clicked.connect(self.copy_distance)
        button_layout.addWidget(self.copy_btn)
        
        layout.addLayout(button_layout)
        
        self.statusBar().showMessage('Select two points to measure distance')
        
    def load_first_frame(self):
        """Load the first frame of the first video in the directory"""
        if not os.path.exists(self.video_dir):
            QMessageBox.critical(self, "Error", "Video directory does not exist!")
            return
        
        images, _ = getImages(self.video_dir)
        
        if not images:
            QMessageBox.warning(self, "Warning", "No images found in directory!")
            return
            
        # Load first frame of first video
        frame = PIL.Image.open(images[0]).convert("RGB")
        frame = np.array(frame)
        self.frame_widget.set_frame(frame)
            
    def on_point_selected(self, point):
        points = self.frame_widget.points
        if len(points) == 1:
            self.statusBar().showMessage(f'First point selected at {point}')
        elif len(points) == 2:
            distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                             (points[1][1] - points[0][1])**2)
            self.statusBar().showMessage(
                f'Distance: {distance:.1f} pixels | Point 1: {points[0]} | Point 2: {points[1]}'
            )
            
    def copy_distance(self):
        points = self.frame_widget.points
        if len(points) == 2:
            distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                             (points[1][1] - points[0][1])**2)
            QApplication.clipboard().setText(f"{distance:.1f}")
            self.statusBar().showMessage(f'Distance {distance:.1f} pixels copied to clipboard')
        else:
            self.statusBar().showMessage('Please select two points first')

def main():
    parser = argparse.ArgumentParser(description='Video Calibration Helper')
    parser.add_argument('--video-dir', required=True, help='Directory containing video files')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    ex = CalibrationHelper(args.video_dir)
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()