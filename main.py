from src.gui.main_window import Main_Window
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = Main_Window()
window.show()
sys.exit(app.exec())
