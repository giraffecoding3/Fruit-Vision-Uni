from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class TerminalWidget:
    def create_terminal_widget(self):
        terminal_window = QFrame()

        terminal_header = QLabel()
        terminal_header.setText("Fruit-Vision-Classification")
        terminal_text = QPlainTextEdit()
        terminal_text.setPlainText("Take a Snapshot!")
        terminal_text.setReadOnly(True)
        terminal_snapshot = QPushButton()
        terminal_snapshot.setText("Snapshot")

        terminal_layout = QVBoxLayout(terminal_window)
        terminal_layout.addWidget(terminal_header)
        terminal_layout.addWidget(terminal_text)
        terminal_layout.addWidget(terminal_snapshot)


        #### Make Styleable Obejects
        terminal_window.setObjectName("terminalWindow")
        terminal_header.setObjectName("terminalHeader")
        terminal_text.setObjectName("terminalText")
        terminal_snapshot.setObjectName("terminalSnapshot")

        ### Styling
        terminal_window.setFixedSize(464,320)
        terminal_layout.setContentsMargins(12,10,12,12)

        terminal_window.setStyleSheet("""
            QFrame#terminalWindow {
                background-color: #0b0f10;
                border: 1px solid #344244;
                border-radius: 8px;
            }

            QLabel#terminalHeader {
                color: #8de0a1;
                font-family: Consolas;
                font-size: 12px;
                font-weight: bold;
                padding-bottom: 5px;
                border-bottom: 1px solid #344244;
            }

            QPushButton#terminalSnapshot {
                color: #8de0a1;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #39754b;
                border-radius: 4px;
                padding: 6px 10px;
            }
        
            QPushButton#terminalSnapshot:hover {
                background-color: #234a31;
            }

            QPlainTextEdit#terminalText {
                color: #8de0a1;
                font-family: Consolas;
            }
        """)

        terminal_window.hide()

        return {
        "window": terminal_window,
        "text": terminal_text,
        "snapshot_button": terminal_snapshot,
        }