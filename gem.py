import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
                            QGroupBox, QLineEdit, QFrame)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

# Your original model architecture definition (must match the trained model)
class SimpleSteganographyModel(torch.nn.Module):
    """
    A steganography model that embeds secret information in the cover image
    and can extract it from the 3-channel stego image ALONE (no cover needed).
    """
    def __init__(self):
        super(SimpleSteganographyModel, self).__init__()

        # Embedding network - combines cover and secret to produce stego
        self.embed_net = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, kernel_size=3, padding=1),  # 6 = 3 (cover) + 3 (secret)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Output 3-channel stego
            torch.nn.Tanh()  # Keep in [-1, 1] range
        )

        # Extraction network - extracts secret from stego image ONLY (no cover needed)
        self.extract_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3 = stego image only
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Output 3-channel secret
            torch.nn.Tanh()  # Keep in [-1, 1] range
        )

    def embed(self, cover, secret):
        """
        Embed secret into cover to produce stego image
        cover: [B, 3, H, W] - normalized to [-1, 1]
        secret: [B, 3, H, W] - normalized to [-1, 1]
        returns: stego image [B, 3, H, W]
        """
        # Concatenate cover and secret
        combined = torch.cat([cover, secret], dim=1)

        # Generate stego image
        stego = self.embed_net(combined)

        # Ensure output is close to the cover but with secret embedded
        stego = cover + 0.1 * stego  # Small perturbation to maintain cover appearance

        return stego

    def extract(self, stego):
        """
        Extract secret from stego image ONLY (no cover needed)
        stego: [B, 3, H, W] - normalized to [-1, 1]
        returns: extracted secret [B, 3, H, W]
        """
        # Extract secret information from stego image alone
        extracted = self.extract_net(stego)

        return extracted

class EmbeddingWorker(QThread):
    finished = pyqtSignal(str)  # Signal to emit result path
    error = pyqtSignal(str)     # Signal to emit error message
    progress = pyqtSignal(int)  # Signal to emit progress

    def __init__(self, model, cover_path, secret_path, save_path, device):
        super().__init__()
        self.model = model
        self.cover_path = cover_path
        self.secret_path = secret_path
        self.save_path = save_path
        self.device = device

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def run(self):
        try:
            self.progress.emit(25)

            # Load and preprocess images
            cover_img = Image.open(self.cover_path).convert('RGB')
            secret_img = Image.open(self.secret_path).convert('RGB')

            cover_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)
            secret_tensor = self.transform(secret_img).unsqueeze(0).to(self.device)

            self.progress.emit(50)

            # Embed secret in cover
            with torch.no_grad():
                stego_tensor = self.model.embed(cover_tensor, secret_tensor)

            self.progress.emit(75)

            # Convert back to image format
            stego_tensor = stego_tensor.cpu().squeeze(0)
            stego_tensor = (stego_tensor + 1) / 2.0  # Denormalize
            stego_tensor = torch.clamp(stego_tensor, 0, 1)

            # Convert to PIL Image
            stego_array = (stego_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            stego_img = Image.fromarray(stego_array)

            # Save stego image
            stego_img.save(self.save_path)
            self.progress.emit(100)
            self.finished.emit(self.save_path)

        except Exception as e:
            self.error.emit(str(e))

class EmbeddingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steganography - Embed Secret in Cover Image")
        self.setGeometry(100, 100, 900, 700)

        # Model and device
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image paths
        self.cover_path = None
        self.secret_path = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("Steganography - Embed Secret in Cover Image")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)

        # Model loading section
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to model file (.pth)")
        model_layout.addWidget(self.model_path_edit)

        model_btn = QPushButton("Browse Model")
        model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(model_btn)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_direct)
        model_layout.addWidget(load_model_btn)

        main_layout.addWidget(model_group)

        # Image selection section
        images_group = QGroupBox("Select Images")
        images_layout = QVBoxLayout(images_group)

        # Cover image
        cover_layout = QHBoxLayout()
        cover_label = QLabel("Cover Image:")
        cover_layout.addWidget(cover_label)
        self.cover_path_edit = QLineEdit()
        self.cover_path_edit.setPlaceholderText("Path to cover image")
        cover_layout.addWidget(self.cover_path_edit)
        cover_btn = QPushButton("Browse")
        cover_btn.clicked.connect(self.browse_cover)
        cover_layout.addWidget(cover_btn)
        images_layout.addLayout(cover_layout)

        # Secret image
        secret_layout = QHBoxLayout()
        secret_label = QLabel("Secret Image:")
        secret_layout.addWidget(secret_label)
        self.secret_path_edit = QLineEdit()
        self.secret_path_edit.setPlaceholderText("Path to secret image")
        secret_layout.addWidget(self.secret_path_edit)
        secret_btn = QPushButton("Browse")
        secret_btn.clicked.connect(self.browse_secret)
        secret_layout.addWidget(secret_btn)
        images_layout.addLayout(secret_layout)

        main_layout.addWidget(images_group)

        # Image preview section
        preview_group = QGroupBox("Image Previews")
        preview_layout = QHBoxLayout(preview_group)

        # Cover preview
        cover_preview_layout = QVBoxLayout()
        cover_preview_label = QLabel("Cover Image")
        cover_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cover_preview_layout.addWidget(cover_preview_label)
        self.cover_preview = QLabel()
        self.cover_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_preview.setFixedSize(200, 200)
        self.cover_preview.setStyleSheet("border: 1px solid gray;")
        cover_preview_layout.addWidget(self.cover_preview)

        # Secret preview
        secret_preview_layout = QVBoxLayout()
        secret_preview_label = QLabel("Secret Image")
        secret_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        secret_preview_layout.addWidget(secret_preview_label)
        self.secret_preview = QLabel()
        self.secret_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secret_preview.setFixedSize(200, 200)
        self.secret_preview.setStyleSheet("border: 1px solid gray;")
        secret_preview_layout.addWidget(self.secret_preview)

        preview_layout.addLayout(cover_preview_layout)
        preview_layout.addLayout(secret_preview_layout)

        main_layout.addWidget(preview_group)

        # Embed button
        self.embed_btn = QPushButton("Embed Secret in Cover")
        self.embed_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        self.embed_btn.clicked.connect(self.embed_secret)
        main_layout.addWidget(self.embed_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def browse_cover(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Cover Image", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.cover_path = file_path
            self.cover_path_edit.setText(file_path)
            self.show_image_preview(file_path, self.cover_preview)

    def browse_secret(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Secret Image", "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.secret_path = file_path
            self.secret_path_edit.setText(file_path)
            self.show_image_preview(file_path, self.secret_preview)

    def show_image_preview(self, file_path, label_widget):
        try:
            img = Image.open(file_path)
            img = img.resize((190, 190), Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # Convert to QImage
            if len(img_array.shape) == 3:
                h, w, c = img_array.shape
                bytes_per_line = 3 * w
                q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                h, w = img_array.shape
                q_img = QImage(img_array.data, w, h, w, QImage.Format.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            label_widget.setPixmap(pixmap.scaled(190, 190, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "PyTorch files (*.pth *.pt);;All files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.load_model_direct()

    def load_model_direct(self):
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Please select a valid model file")
            return

        try:
            self.model = SimpleSteganographyModel()
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.status_label.setText(f"Model loaded successfully from {model_path}")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def embed_secret(self):
        if not self.model:
            QMessageBox.critical(self, "Error", "Please load the model first")
            return

        if not self.cover_path or not self.secret_path:
            QMessageBox.critical(self, "Error", "Please select both cover and secret images")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Stego Image", "",
            "PNG files (*.png);;JPEG files (*.jpg);;All files (*)"
        )

        if not save_path:
            return

        # Start embedding in a separate thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.embed_btn.setEnabled(False)

        self.worker = EmbeddingWorker(self.model, self.cover_path, self.secret_path, save_path, self.device)
        self.worker.finished.connect(self.on_embedding_finished)
        self.worker.error.connect(self.on_embedding_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def on_embedding_finished(self, result_path):
        self.progress_bar.setVisible(False)
        self.embed_btn.setEnabled(True)
        self.status_label.setText(f"Stego image saved successfully: {result_path}")
        QMessageBox.information(self, "Success", f"Secret embedded successfully!\nStego image saved as: {result_path}")

    def on_embedding_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.embed_btn.setEnabled(True)
        self.status_label.setText("Error occurred during embedding")
        QMessageBox.critical(self, "Error", f"Failed to embed secret: {error_msg}")

def main():
    app = QApplication(sys.argv)
    window = EmbeddingGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()