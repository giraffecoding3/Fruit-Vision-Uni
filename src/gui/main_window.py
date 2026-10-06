import math
from pathlib import Path
from typing import cast

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedLayout,
    QWidget,
)
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.camera.kamera import Camera
from src.gui.components.camera_overlay import CameraOverlay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH_OBJ = PROJECT_ROOT / "models" / "trained" / "Object_detection" / "weights" / "trained_object_detection_1.pt"
MODEL_PATH_CLS = PROJECT_ROOT / "models" / "trained" / "Classification_1_own_data" / "weights" / "best.pt"
MODEL_PATH_SEG = PROJECT_ROOT / "models" / "trained" / "Segmentation meine Daten" / "weights" / "best.pt"
BOX_WIDTH_RATIO = 0.60
BOX_HEIGHT_RATIO = 0.60


class Main_Window(QMainWindow):
    def __init__(self):
        super().__init__()


        self.kamera = Camera()
        self.model_obj = YOLO(MODEL_PATH_OBJ)
        self.model_cls = YOLO(MODEL_PATH_CLS)
        self.model_seg = YOLO(MODEL_PATH_SEG)
        self.detection_enabled = False
        self.segmentation_enabled = False
        self.classification_enabled = False
        self.snapshot_taken = False
        self.roi_enabled = False

        # region Main Window Settings
        self.setWindowTitle("Fruit Vision")
        self.setMinimumSize(1440, 1080)
        # endregion

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Stacked Layer - Kamera unten und Transpaerente Ebene mit Buttons oben drauf:
        self.main_layout = QStackedLayout(self.central_widget)
        self.main_layout.setStackingMode(
          QStackedLayout.StackingMode.StackAll
        )
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # region Kamera  
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.picture)
        # endregion 

        # region  Overlay
        self.overlay = CameraOverlay()
        self.main_layout.addWidget(self.overlay)
        self.main_layout.setCurrentWidget(self.overlay)

        self.terminal_window = self.overlay.terminal_window
        self.terminal_text = self.overlay.terminal_text

        self.classification_button = self.overlay.classification_button
        self.detection_button = self.overlay.detection_button
        self.segmentation_button = self.overlay.segmentation_button
        self.toggle_roi_btn = self.overlay.toggle_roi_btn
        self.seg_snapshot_btn = self.overlay.seg_snapshot
        

        self.overlay.snapshot_button.clicked.connect(self.take_snapshot)
        self.classification_button.toggled.connect(self.toggle_classification)
        self.detection_button.toggled.connect(self.toggle_detection)
        self.segmentation_button.toggled.connect(self.toggle_segmentation)
        self.toggle_roi_btn.toggled.connect(self.toggle_roi)
        self.seg_snapshot_btn.clicked.connect(self.take_snapshot)
        # endregion


        # Timer für Kamera Stream
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(30)

    def update_camera_frame(self):
        frame = self.kamera.get_picture()
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        box_width = int(width * BOX_WIDTH_RATIO)
        box_height = int(height * BOX_HEIGHT_RATIO)
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        box = frame[y1:y2, x1:x2]


            
        if self.detection_enabled:
            predictions = self.model_obj.predict(
                source=box,
                imgsz=640,
                conf=0.65,
                verbose=False,
                stream=False,
            )
            result = cast(Results, next(iter(predictions)))
            frame[y1:y2, x1:x2] = result.plot()

        if self.roi_enabled:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 60), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape

        image = QImage(cast(bytes,frame_rgb.data),width,height,channels * width, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.picture.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.picture.setPixmap(scaled_pixmap)

    def toggle_detection(self, enabled: bool):
        self.detection_enabled = enabled
        self.detection_button.setText(
            "Detection: on" if enabled else "Detection"
        )

    def class_snapshot(self):
            if self.snapshot_taken:
                predictions = self.model_cls.predict(
                    source=self.box,
                    imgsz = 640,
                    verbose = False,
                    stream = False
                )
                result = cast(Results, next(iter(predictions)))
                if result.probs is not None:
                    top1class = result.probs.top1
                    top1conf = math.floor(result.probs.top1conf.item() * 100) / 100

                    name = result.names[top1class]

                    if top1conf > 0.9:
                        self.terminal_text.setPlainText(f"I can see one {name} with a confidence of ~{top1conf}")    
                    else: 
                        self.terminal_text.setPlainText("Im not confident enough in my prediction")

    def seg_snapshot(self):
        if self.snapshot_taken:
            predictions = self.model_seg.predict(
                source = self.box,
                imgsz = 640,
                verbose = False,
                stream = False
            )
            result = cast(Results, next(iter(predictions)))

            segment = result.plot(boxes = False, labels=True) 
            rgb_img = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)

            height,width,channels = rgb_img.shape
            bpl = channels * width

            img = QImage(
                cast(bytes,rgb_img.data),
                width,
                height,
                bpl,
                QImage.Format.Format_RGB888
            ).copy()

            #Original-Img Window
            rgb_img_og = cv2.cvtColor(self.box, cv2.COLOR_BGR2RGB)
            height_og,width_og,channels_og = rgb_img_og.shape
            bpl_og = channels_og * width_og
            img_og = QImage(
                cast(bytes,rgb_img_og.data),
                width_og,
                height_og,
                bpl_og,
                QImage.Format.Format_RGB888
            ).copy()

            #Segment Window
            img_window = QDialog(self)
            img_window.setWindowTitle("Snapshot und Segmentierung")

            label_og = QLabel()
            label_og.setPixmap(QPixmap.fromImage(img_og))

            label_seg = QLabel()
            label_seg.setPixmap(QPixmap.fromImage(img))

            layout_images = QHBoxLayout(img_window)
            layout_images.addWidget(label_og)
            layout_images.addWidget(label_seg)

            img_window.adjustSize()
            img_window.exec()



            

    def take_snapshot(self):
        frame = self.kamera.get_picture()
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        box_width = int(width * BOX_WIDTH_RATIO)
        box_height = int(height * BOX_HEIGHT_RATIO)
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        self.box = frame[y1:y2,x1:x2]
        self.snapshot_taken = True

        if self.classification_enabled:
            self.class_snapshot()
        if self.segmentation_enabled:
            self.seg_snapshot()

    def toggle_roi(self, enabled: bool):
       self.roi_enabled =enabled 
        
    def toggle_classification(self, enabled: bool):
        self.classification_enabled = enabled
        self.terminal_window.setVisible(enabled)
        self.classification_button.setText(
            "Classification: on" if enabled else "Classification"
        )

    def toggle_segmentation(self, enabled: bool):
        self.segmentation_enabled = enabled
        self.seg_snapshot_btn.setVisible(enabled)
        self.segmentation_button.setText(
            "Segementation: on" if enabled else "Segmentation"
        )

