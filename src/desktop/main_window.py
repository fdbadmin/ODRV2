"""
Main Window for ODRV2 Desktop Application.
PyQt6-based UI with drag-drop image upload, predictions, and clinical explanations.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QProgressBar, QScrollArea,
    QFrame, QSplitter, QMessageBox, QApplication, QComboBox,
    QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QPainter, QColor, QFont, QPalette, QLinearGradient

from PIL import Image

from .inference import FundusPredictor, PredictionResult, DISEASE_INFO


# Modern color palette
COLORS = {
    'primary': '#2563EB',       # Blue
    'primary_dark': '#1D4ED8',
    'primary_light': '#DBEAFE',
    'success': '#059669',       # Green
    'success_light': '#D1FAE5',
    'warning': '#D97706',       # Amber
    'warning_light': '#FEF3C7',
    'danger': '#DC2626',        # Red
    'danger_light': '#FEE2E2',
    'text': '#1F2937',
    'text_secondary': '#6B7280',
    'border': '#E5E7EB',
    'background': '#F9FAFB',
    'card': '#FFFFFF',
    'shadow': 'rgba(0, 0, 0, 0.1)',
}

# Global stylesheet for modern look
GLOBAL_STYLESHEET = """
QMainWindow {
    background-color: #F3F4F6;
}

QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QPushButton:pressed {
    background-color: #1E40AF;
}

QPushButton:disabled {
    background-color: #9CA3AF;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #F3F4F6;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #D1D5DB;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background: #9CA3AF;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QComboBox {
    padding: 8px 12px;
    font-size: 13px;
    border: 2px solid #E5E7EB;
    border-radius: 8px;
    background: white;
    min-width: 200px;
}

QComboBox:hover {
    border-color: #2563EB;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    selection-background-color: #DBEAFE;
    selection-color: #1F2937;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #E5E7EB;
    text-align: center;
}

QProgressBar::chunk {
    border-radius: 4px;
    background-color: #2563EB;
}
"""


class ModelLoaderThread(QThread):
    """Background thread for loading models."""
    finished = pyqtSignal(bool)
    progress = pyqtSignal(str)
    
    def __init__(self, predictor: FundusPredictor):
        super().__init__()
        self.predictor = predictor
    
    def run(self):
        self.progress.emit("Loading models...")
        success = self.predictor.load_models()
        self.finished.emit(success)


class PredictionThread(QThread):
    """Background thread for running predictions."""
    finished = pyqtSignal(object)  # PredictionResult or Exception
    
    def __init__(self, predictor: FundusPredictor, image: Image.Image):
        super().__init__()
        self.predictor = predictor
        self.image = image
    
    def run(self):
        try:
            result = self.predictor.predict(self.image)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


class ImageDropZone(QLabel):
    """Widget for drag-and-drop image upload."""
    
    imageDropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 400)
        self._default_style = """
            QLabel {
                border: 2px dashed #D1D5DB;
                border-radius: 16px;
                background-color: #F9FAFB;
                color: #6B7280;
                font-size: 15px;
                font-weight: 500;
            }
            QLabel:hover {
                border-color: #2563EB;
                background-color: #EFF6FF;
                color: #2563EB;
            }
        """
        self.setStyleSheet(self._default_style)
        self.setText("📷 Drop fundus image here\nor click to browse")
        self._pixmap: Optional[QPixmap] = None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #059669;
                    border-radius: 16px;
                    background-color: #D1FAE5;
                    color: #059669;
                    font-size: 15px;
                    font-weight: 500;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._default_style)
    
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self._default_style)
        
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                self.imageDropped.emit(file_path)
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Fundus Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        if file_path:
            self.imageDropped.emit(file_path)
    
    def setImage(self, pixmap: QPixmap):
        """Display an image in the drop zone."""
        self._pixmap = pixmap
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)
        super().resizeEvent(event)


class DiseaseResultWidget(QFrame):
    """Widget displaying a single disease prediction result."""
    
    def __init__(self, disease_name: str, parent=None):
        super().__init__(parent)
        self.disease_name = disease_name
        self.info = DISEASE_INFO.get(disease_name, {})
        
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self._default_style = """
            QFrame {
                background-color: #FAFAFA;
                border-radius: 10px;
                border: none;
            }
        """
        self.setStyleSheet(self._default_style)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        
        # Header row: color indicator + name + probability
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        # Color indicator dot
        color_dot = QLabel("●")
        color_dot.setStyleSheet(f"color: {self.info.get('color', '#2563EB')}; font-size: 16px;")
        header_layout.addWidget(color_dot)
        
        self.name_label = QLabel(self.info.get('name', disease_name))
        self.name_label.setStyleSheet("font-weight: 600; font-size: 14px; color: #1F2937;")
        header_layout.addWidget(self.name_label)
        
        header_layout.addStretch()
        
        self.prob_label = QLabel("--")
        self.prob_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #6B7280;")
        header_layout.addWidget(self.prob_label)
        
        layout.addLayout(header_layout)
        
        # Progress bar for probability
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setTextVisible(False)
        self.prob_bar.setMaximumHeight(6)
        self.prob_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: #E5E7EB;
            }}
            QProgressBar::chunk {{
                border-radius: 3px;
                background-color: {self.info.get('color', '#2563EB')};
            }}
        """)
        layout.addWidget(self.prob_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Clinical info (hidden by default)
        self.clinical_label = QLabel(f"📋 {self.info.get('description', '')}")
        self.clinical_label.setWordWrap(True)
        self.clinical_label.setStyleSheet("color: #4B5563; font-size: 12px; margin-top: 4px; padding: 8px; background-color: #F9FAFB; border-radius: 6px;")
        self.clinical_label.hide()
        layout.addWidget(self.clinical_label)
        
        # Recommendation (hidden by default)
        self.recommendation_label = QLabel(f"💡 {self.info.get('recommendation', '')}")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet("color: #2563EB; font-size: 12px; font-weight: 500; padding: 8px; background-color: #EFF6FF; border-radius: 6px;")
        self.recommendation_label.hide()
        layout.addWidget(self.recommendation_label)
    
    def setResult(self, probability: float, is_positive: bool, threshold: float):
        """Update the widget with prediction results."""
        pct = int(probability * 100)
        self.prob_bar.setValue(pct)
        self.prob_label.setText(f"{pct}%")
        
        if is_positive:
            self.status_label.setText(f"⚠️ DETECTED (threshold: {threshold:.0%})")
            self.status_label.setStyleSheet(f"color: {self.info.get('color', '#DC2626')}; font-size: 12px; font-weight: 600;")
            self.prob_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {self.info.get('color', '#DC2626')};")
            self.clinical_label.show()
            self.recommendation_label.show()
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: #FEF3C7;
                    border-radius: 10px;
                    border-left: 4px solid {self.info.get('color', '#D97706')};
                }}
            """)
        else:
            self.status_label.setText(f"✓ Not detected (threshold: {threshold:.0%})")
            self.status_label.setStyleSheet("color: #059669; font-size: 12px;")
            self.prob_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #6B7280;")
            self.clinical_label.hide()
            self.recommendation_label.hide()
            self.setStyleSheet(self._default_style)
    
    def clear(self):
        """Reset the widget."""
        self.prob_bar.setValue(0)
        self.prob_label.setText("--")
        self.prob_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #6B7280;")
        self.status_label.setText("")
        self.clinical_label.hide()
        self.recommendation_label.hide()
        self.setStyleSheet(self._default_style)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.predictor = FundusPredictor()
        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[str] = None
        
        self._setup_ui()
        self._load_models()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🔬 ODRV2 - Ocular Disease Recognition")
        self.setMinimumSize(1100, 800)
        
        # Apply global stylesheet
        self.setStyleSheet(GLOBAL_STYLESHEET)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #F3F4F6;")
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #E5E7EB; width: 2px; }")
        main_layout.addWidget(splitter)
        
        # Left panel: Image (in a card)
        left_card = QFrame()
        left_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 16px;
                border: none;
            }
        """)
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(24, 20, 24, 20)
        left_layout.setSpacing(16)
        
        # Original Image Section
        title_label = QLabel("📷 Fundus Image")
        title_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #1F2937; background: transparent;")
        left_layout.addWidget(title_label)
        
        # Image drop zone
        self.image_zone = ImageDropZone()
        self.image_zone.imageDropped.connect(self._on_image_dropped)
        self.image_zone.setMinimumSize(380, 320)
        self.image_zone.setMaximumHeight(380)
        left_layout.addWidget(self.image_zone)
        
        # Heatmap Section
        heatmap_header = QHBoxLayout()
        heatmap_title = QLabel("🔥 Attention Heatmap")
        heatmap_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #1F2937; background: transparent;")
        heatmap_header.addWidget(heatmap_title)
        
        # Disease selector dropdown
        self.heatmap_selector = QComboBox()
        self.heatmap_selector.setMinimumWidth(220)
        self.heatmap_selector.currentTextChanged.connect(self._on_heatmap_selection_changed)
        heatmap_header.addWidget(self.heatmap_selector)
        heatmap_header.addStretch()
        
        left_layout.addLayout(heatmap_header)
        
        # Heatmap display - same size as image zone
        self.heatmap_display = QLabel()
        self.heatmap_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_display.setMinimumSize(380, 320)
        self.heatmap_display.setMaximumHeight(380)
        self.heatmap_display.setStyleSheet("""
            QLabel {
                border: 2px dashed #D1D5DB;
                border-radius: 16px;
                background-color: #F9FAFB;
                color: #9CA3AF;
                font-size: 14px;
            }
        """)
        self.heatmap_display.setText("🔍 Heatmap will appear after analysis")
        self._heatmap_pixmap: Optional[QPixmap] = None
        left_layout.addWidget(self.heatmap_display)
        
        # Add stretch to push buttons to bottom
        left_layout.addStretch()
        
        # Store heatmaps
        self.current_heatmaps: Optional[dict] = None
        self.current_result: Optional[PredictionResult] = None
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.analyze_btn = QPushButton("  Analyze Image  ")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(50)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-size: 16px;
                font-weight: 700;
                border-radius: 12px;
                padding: 14px 32px;
            }
            QPushButton:hover {
                background-color: #047857;
            }
            QPushButton:pressed {
                background-color: #065F46;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.analyze_btn.clicked.connect(self._run_prediction)
        button_layout.addWidget(self.analyze_btn, 2)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMinimumHeight(50)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: 2px solid #E5E7EB;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #F3F4F6;
                border-color: #D1D5DB;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_all)
        button_layout.addWidget(self.clear_btn, 1)
        
        left_layout.addLayout(button_layout)
        
        splitter.addWidget(left_card)
        
        # Right panel: Results (in a card)
        right_card = QFrame()
        right_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 16px;
                border: none;
            }
        """)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(24, 20, 24, 20)
        right_layout.setSpacing(8)
        
        # Title
        results_title = QLabel("🩺 Disease Detection Results")
        results_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #1F2937; background: transparent;")
        right_layout.addWidget(results_title)
        
        # Status label
        self.status_label = QLabel("⏳ Loading models...")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 13px; background: transparent;")
        right_layout.addWidget(self.status_label)
        
        # Progress bar for loading
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)  # Indeterminate
        self.loading_bar.setMaximumHeight(4)
        right_layout.addWidget(self.loading_bar)
        
        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QWidget { background: transparent; }")
        
        results_widget = QWidget()
        results_widget.setStyleSheet("background: transparent;")
        self.results_layout = QVBoxLayout(results_widget)
        self.results_layout.setContentsMargins(0, 4, 4, 0)
        self.results_layout.setSpacing(6)
        
        # Create disease result widgets
        self.disease_widgets = {}
        for disease in ['Diabetic_Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']:
            widget = DiseaseResultWidget(disease)
            self.disease_widgets[disease] = widget
            self.results_layout.addWidget(widget)
        
        self.results_layout.addStretch()
        
        scroll.setWidget(results_widget)
        right_layout.addWidget(scroll, 1)
        
        # Disclaimer
        disclaimer = QLabel(
            "Research/screening tool only. Results must be verified by a qualified ophthalmologist."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("""
            QLabel {
                background-color: #FEF9E7;
                border: none;
                border-radius: 6px;
                padding: 10px;
                color: #92400E;
                font-size: 11px;
            }
        """)
        right_layout.addWidget(disclaimer)
        
        splitter.addWidget(right_card)
        splitter.setSizes([550, 450])
    
    def _load_models(self):
        """Load models in background thread."""
        self.loader_thread = ModelLoaderThread(self.predictor)
        self.loader_thread.progress.connect(self._on_load_progress)
        self.loader_thread.finished.connect(self._on_models_loaded)
        self.loader_thread.start()
    
    def _on_load_progress(self, message: str):
        """Update status during model loading."""
        self.status_label.setText(f"⏳ {message}")
    
    def _on_models_loaded(self, success: bool):
        """Handle model loading completion."""
        self.loading_bar.hide()
        
        if success:
            self.status_label.setText(f"✅ Ready - {len(self.predictor.models)} models loaded")
            self.status_label.setStyleSheet("color: #059669; font-size: 13px; font-weight: 500; background: transparent;")
        else:
            self.status_label.setText("❌ Failed to load models. Check model directory.")
            self.status_label.setStyleSheet("color: #DC2626; font-size: 13px; font-weight: 500; background: transparent;")
            QMessageBox.critical(
                self,
                "Model Loading Error",
                "Could not load the prediction models.\n\n"
                "Please ensure model files exist in:\n"
                "models/unified_v3_retrain/fold_*/best_model.pth"
            )
    
    def _on_image_dropped(self, file_path: str):
        """Handle image drop/selection."""
        try:
            # Load and display image
            self.current_image = Image.open(file_path).convert('RGB')
            self.current_image_path = file_path
            
            # Convert to QPixmap for display
            img = self.current_image.copy()
            img = img.convert('RGBA')
            data = img.tobytes('raw', 'RGBA')
            qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimg)
            
            self.image_zone.setImage(pixmap)
            self.analyze_btn.setEnabled(self.predictor.is_loaded)
            
            # Clear previous results and heatmaps
            self.current_heatmaps = None
            self.current_result = None
            self.heatmap_selector.clear()
            self.heatmap_display.clear()
            self.heatmap_display.setText("🔍 Heatmap will appear after analysis")
            self._heatmap_pixmap = None
            
            for widget in self.disease_widgets.values():
                widget.clear()
            
            self.status_label.setText(f"📁 Image loaded: {Path(file_path).name}")
            self.status_label.setStyleSheet("color: #2563EB; font-size: 13px; font-weight: 500; background: transparent;")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load image:\n{e}")
    
    def _run_prediction(self):
        """Run model prediction on current image."""
        if not self.current_image or not self.predictor.is_loaded:
            return
        
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("🔄 Analyzing image...")
        self.status_label.setStyleSheet("color: #6B7280; font-size: 13px; background: transparent;")
        self.loading_bar.show()
        
        self.prediction_thread = PredictionThread(self.predictor, self.current_image)
        self.prediction_thread.finished.connect(self._on_prediction_complete)
        self.prediction_thread.start()
    
    def _on_prediction_complete(self, result):
        """Handle prediction completion."""
        self.loading_bar.hide()
        self.analyze_btn.setEnabled(True)
        
        if isinstance(result, Exception):
            self.status_label.setText(f"❌ Error: {result}")
            self.status_label.setStyleSheet("color: #DC2626; font-size: 13px; font-weight: 500; background: transparent;")
            return
        
        # Store result for heatmap switching
        self.current_result = result
        self.current_heatmaps = result.heatmaps
        
        # Update disease widgets
        detected_count = 0
        for disease in result.disease_names:
            if disease in self.disease_widgets:
                prob = result.probabilities[disease]
                is_positive = result.predictions[disease]
                threshold = result.thresholds.get(disease, 0.5)
                
                self.disease_widgets[disease].setResult(prob, is_positive, threshold)
                
                if is_positive and disease != 'Other':
                    detected_count += 1
        
        # Populate heatmap selector (sorted by probability, highest first)
        self._populate_heatmap_selector(result)
        
        if detected_count > 0:
            self.status_label.setText(f"⚠️ {detected_count} condition(s) detected - see heatmap for focus regions")
            self.status_label.setStyleSheet("color: #D97706; font-size: 13px; font-weight: 600; background: transparent;")
        else:
            self.status_label.setText("✅ No pathology detected")
            self.status_label.setStyleSheet("color: #059669; font-size: 13px; font-weight: 600; background: transparent;")
    
    def _populate_heatmap_selector(self, result: PredictionResult):
        """Populate the heatmap dropdown sorted by probability."""
        self.heatmap_selector.blockSignals(True)
        self.heatmap_selector.clear()
        
        # Sort diseases by probability (highest first)
        sorted_diseases = sorted(
            result.disease_names,
            key=lambda d: result.probabilities.get(d, 0),
            reverse=True
        )
        
        for disease in sorted_diseases:
            prob = result.probabilities.get(disease, 0)
            info = DISEASE_INFO.get(disease, {})
            display_name = info.get('name', disease)
            # Add probability and detection indicator
            is_detected = result.predictions.get(disease, False)
            indicator = "⚠️ " if is_detected else ""
            self.heatmap_selector.addItem(f"{indicator}{display_name} ({prob:.0%})", disease)
        
        self.heatmap_selector.blockSignals(False)
        
        # Trigger display of highest probability heatmap
        if sorted_diseases:
            self._display_heatmap(sorted_diseases[0])
    
    def _on_heatmap_selection_changed(self, text: str):
        """Handle heatmap dropdown selection change."""
        # Get the disease key from the combo box data
        idx = self.heatmap_selector.currentIndex()
        if idx >= 0:
            disease = self.heatmap_selector.itemData(idx)
            if disease:
                self._display_heatmap(disease)
    
    def _display_heatmap(self, disease: str):
        """Display the heatmap for a specific disease overlaid on original image."""
        if self.current_image is None or self.current_heatmaps is None:
            return
        
        heatmap = self.current_heatmaps.get(disease)
        if heatmap is None:
            self.heatmap_display.setText(f"No heatmap available for {disease}")
            return
        
        # Get original image as numpy array
        original = np.array(self.current_image)
        
        # Resize heatmap to match original
        import cv2
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Blend original with heatmap (40% heatmap)
        alpha = 0.4
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap_resized, alpha, 0)
        
        # Convert to QPixmap
        overlay_img = Image.fromarray(overlay).convert('RGBA')
        data = overlay_img.tobytes('raw', 'RGBA')
        qimg = QImage(data, overlay_img.width, overlay_img.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit display
        scaled = pixmap.scaled(
            self.heatmap_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.heatmap_display.setPixmap(scaled)
        self._heatmap_pixmap = pixmap
    
    def _clear_all(self):
        """Clear image and results."""
        self.current_image = None
        self.current_image_path = None
        self.current_heatmaps = None
        self.current_result = None
        
        self.image_zone.clear()
        self.image_zone.setText("📷 Drop fundus image here\nor click to browse")
        self.image_zone._pixmap = None
        
        # Clear heatmap display
        self.heatmap_display.clear()
        self.heatmap_display.setText("🔍 Heatmap will appear after analysis")
        self._heatmap_pixmap = None
        self.heatmap_selector.clear()
        
        self.analyze_btn.setEnabled(False)
        
        for widget in self.disease_widgets.values():
            widget.clear()
        
        self.status_label.setText(f"✅ Ready - {len(self.predictor.models)} models loaded")
        self.status_label.setStyleSheet("color: #059669; font-size: 13px; font-weight: 500; background: transparent;")
