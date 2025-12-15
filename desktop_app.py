#!/usr/bin/env python3
"""Launch script for ODRV2 Desktop Application"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

from src.desktop.main_window import MainWindow


def setup_application_style(app: QApplication):
    """Setup application-wide style and theme
    
    Args:
        app: QApplication instance
    """
    # Set application metadata
    app.setApplicationName("ODRV2")
    app.setOrganizationName("ODRV2")
    app.setApplicationVersion("1.0.0")
    
    # Modern stylesheet
    stylesheet = """
    QMainWindow {
        background-color: #f5f5f5;
    }
    
    QMenuBar {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
    }
    
    QMenuBar::item {
        padding: 8px 12px;
        background-color: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #e3f2fd;
    }
    
    QMenu {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    QMenu::item {
        padding: 8px 32px 8px 16px;
    }
    
    QMenu::item:selected {
        background-color: #e3f2fd;
    }
    
    QToolBar {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
        spacing: 5px;
        padding: 5px;
    }
    
    QToolButton {
        background-color: transparent;
        border: none;
        border-radius: 3px;
        padding: 5px;
    }
    
    QToolButton:hover {
        background-color: #e3f2fd;
    }
    
    QToolButton:pressed {
        background-color: #bbdefb;
    }
    
    QStatusBar {
        background-color: #ffffff;
        border-top: 1px solid #e0e0e0;
    }
    
    QPushButton {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #1976D2;
    }
    
    QPushButton:pressed {
        background-color: #0D47A1;
    }
    
    QPushButton:disabled {
        background-color: #BDBDBD;
        color: #757575;
    }
    
    QGroupBox {
        font-weight: bold;
        border: 2px solid #e0e0e0;
        border-radius: 5px;
        margin-top: 12px;
        padding-top: 10px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    
    QScrollBar:vertical {
        border: none;
        background: #f5f5f5;
        width: 12px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background: #bdbdbd;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #9e9e9e;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    
    QScrollBar:horizontal {
        border: none;
        background: #f5f5f5;
        height: 12px;
        margin: 0;
    }
    
    QScrollBar::handle:horizontal {
        background: #bdbdbd;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background: #9e9e9e;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    """
    
    app.setStyleSheet(stylesheet)


def main():
    """Main entry point for the application"""
    # Create application
    app = QApplication(sys.argv)
    
    # Setup styling
    setup_application_style(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
