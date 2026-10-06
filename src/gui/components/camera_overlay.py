from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from src.gui.components.button_factory import ButtonFactory
from src.gui.components.terminal_widget import TerminalWidget


class CameraOverlay(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("background: transparent;")

        self.overlay_layout = QVBoxLayout(self)
        self.overlay_layout.setContentsMargins(20, 20, 20, 20)

        self.overlay_top_right = QHBoxLayout()
        self.overlay_top_right.addStretch()

        self.overlay_layout.addLayout(self.overlay_top_right)
        self.overlay_layout.addStretch()

        terminal = TerminalWidget().create_terminal_widget()
        self.terminal_window = terminal["window"]
        self.terminal_text = terminal["text"]
        self.snapshot_button = terminal["snapshot_button"]

        self.overlay_layout.addWidget(self.terminal_window)

        buttons = ButtonFactory().create_main_btns()
        self.classification_button = buttons["classification"]
        self.detection_button = buttons["detection"]
        self.segmentation_button = buttons["segmentation"]
        self.toggle_roi_btn = buttons["roi"]
        self.seg_snapshot = buttons["segsnapbtn"]

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.classification_button)
        self.button_layout.addWidget(self.detection_button)

        self.segmentation_layout = QVBoxLayout()
        self.segmentation_layout.addWidget(self.seg_snapshot)
        self.segmentation_layout.addWidget(self.segmentation_button)
        self.button_layout.addLayout(self.segmentation_layout)

        self.overlay_top_right.addWidget(self.toggle_roi_btn)
        self.overlay_layout.addLayout(self.button_layout)