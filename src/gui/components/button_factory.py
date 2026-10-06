from PyQt6.QtWidgets import QPushButton


class ButtonFactory:

    def create_main_btns(self):


        classification_button = QPushButton("Classification")
        classification_button.setCheckable(True)

        detection_button = QPushButton("Detection")
        detection_button.setCheckable(True)

        segmentation_button = QPushButton("Segmentierung")
        segmentation_button.setCheckable(True)

        toggle_roi_btn = QPushButton("Toggle \n RoI")
        toggle_roi_btn.setCheckable(True)
        toggle_roi_btn.setFixedSize(80,80)

        seg_snapshot = QPushButton("Snapshot")
        seg_snapshot.hide()

        classification_button.setObjectName("classBtn")
        classification_button.setStyleSheet("""
            QPushButton#classBtn {
                color: #8de0a1;
                background-color: #0b0f10;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton#classBtn:hover {
                background-color: #234a31;
            }
        """)

        detection_button.setObjectName("objBtn")
        detection_button.setStyleSheet("""
            QPushButton#objBtn {
                color: #8de0a1;
                background-color: #0b0f10;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton#objBtn:hover {
                background-color: #234a31;
            }
        """)

        segmentation_button.setObjectName("segBtn")
        segmentation_button.setStyleSheet("""
            QPushButton#segBtn {
                color: #8de0a1;
                background-color: #0b0f10;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton#segBtn:hover {
                background-color: #234a31;
            }
        """)

        toggle_roi_btn.setObjectName("toggleBtn")
        toggle_roi_btn.setStyleSheet("""
            QPushButton#toggleBtn {
                color: #8de0a1;
                background-color: #0b0f10;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton#toggleBtn:hover {
                background-color: #234a31;
            }
        """)
        seg_snapshot.setObjectName("segSnapBtn")
        seg_snapshot.setStyleSheet("""
            QPushButton#segSnapBtn {
                color: #8de0a1;
                background-color: #0b0f10;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
        
            QPushButton#segSnapBtn:hover {
                background-color: #234a31;
            }
        
        """)

        return {
            "classification": classification_button,
            "detection": detection_button,
            "segmentation" : segmentation_button,
            "roi" : toggle_roi_btn,
            "segsnapbtn": seg_snapshot,
        }

