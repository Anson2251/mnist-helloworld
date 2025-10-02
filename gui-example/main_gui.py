#!/usr/bin/env python3
"""
MNIST GUI Application with Camera Input and Prediction Display
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from cv2.typing import MatLike
from mpmath.ctx_mp import normalize
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.registry import ModelRegistry
from src.datasets.mnist import MNISTDataset
from src.utils.device import get_device


class MNISTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Real-time Classification")
        self.root.geometry("600x480")

        # Initialize camera and model variables
        self.cap = None
        self.model = None
        self.device, _ = get_device()
        self.current_frame = None
        self.is_running = False
        self.is_detecting = False  # New flag for iterative detection mode

        # Dataset for preprocessing
        self.dataset = MNISTDataset(root='./data', download=True)

        # Create GUI elements
        self.create_widgets()

        # Start camera
        self.start_camera()

    def create_widgets(self):
        """Create all GUI widgets"""
        # Create toolbar
        self.create_toolbar()

        # Create main content area with side-by-side containers
        main_content = ttk.Frame(self.root)
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create main containers side by side
        self.create_camera_container(main_content)
        self.create_prediction_container(main_content)

        # Create control buttons
        self.create_control_buttons()

    def create_toolbar(self):
        """Create toolbar with model and checkpoint selection"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        selection_bar = ttk.Frame(toolbar)
        selection_bar.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Model selection
        ttk.Label(selection_bar, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="mynet")
        model_combo = ttk.Combobox(selection_bar, textvariable=self.model_var,
                                   values=ModelRegistry.list_available(),
                                   state="readonly", width=15)
        model_combo.pack(side=tk.LEFT, padx=5)

         # Checkpoint selection
        ttk.Label(selection_bar, text="Checkpoint:").pack(side=tk.LEFT, padx=5)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(selection_bar, textvariable=self.checkpoint_var,
                                            state="readonly", width=30)
        self.checkpoint_combo.pack(side=tk.LEFT, padx=5)

        action_bar = ttk.Frame(toolbar)
        action_bar.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Load checkpoint button
        ttk.Button(action_bar, text="Browse",
                  command=self.browse_checkpoint).pack(side=tk.LEFT, padx=5)

        # Load model button
        ttk.Button(action_bar, text="Load Model",
                  command=self.load_model).pack(side=tk.LEFT, padx=5)

        # Refresh checkpoints button
        ttk.Button(action_bar, text="Refresh",
                  command=self.refresh_checkpoints).pack(side=tk.LEFT, padx=5)

        # Initial checkpoint list refresh
        self.refresh_checkpoints()

    def create_camera_container(self, parent):
        """Create left container for camera display"""
        camera_frame = ttk.LabelFrame(parent, text="Camera Input", padding=10)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        # Camera display label
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Camera status
        self.camera_status = ttk.Label(camera_frame, text="Camera: Starting...",
                                      foreground="blue")
        self.camera_status.pack(pady=5)

    def create_prediction_container(self, parent):
        """Create right container for prediction results"""
        pred_frame = ttk.LabelFrame(parent, text="Prediction Results", padding=10)
        pred_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        # Prediction display
        self.prediction_label = ttk.Label(pred_frame, text="No prediction yet",
                                         font=("Arial", 16))
        self.prediction_label.pack(pady=20)

        # Confidence scores
        self.confidence_frame = ttk.Frame(pred_frame)
        self.confidence_frame.pack(fill=tk.BOTH, expand=True)

        # Probability bars
        self.probability_bars = []
        for i in range(10):
            bar_frame = ttk.Frame(self.confidence_frame)
            bar_frame.pack(fill=tk.X, pady=2)

            ttk.Label(bar_frame, text=f"{i}:", width=5).pack(side=tk.LEFT)
            bar = ttk.Progressbar(bar_frame, length=200, maximum=100)
            bar.pack(side=tk.LEFT, padx=5)
            value_label = ttk.Label(bar_frame, text="0%", width=5)
            value_label.pack(side=tk.LEFT)

            self.probability_bars.append((bar, value_label))

    def create_control_buttons(self):
        """Create control buttons"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Toggle detection button
        self.detect_button = ttk.Button(control_frame, text="Start Detection",
                                       command=self.toggle_detection)
        self.detect_button.pack(side=tk.LEFT, padx=5)

        # Manual capture button (always available)
        ttk.Button(control_frame, text="Manual Capture",
                  command=self.capture_and_predict).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Clear Prediction",
                  command=self.clear_prediction).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Exit",
                  command=self.on_closing).pack(side=tk.RIGHT, padx=5)

    def start_camera(self):
        """Initialize and start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")

            self.is_running = True
            self.camera_status.config(text="Camera: Running", foreground="green")
            self.update_camera()

        except Exception as e:
            self.camera_status.config(text=f"Camera: Error - {str(e)}", foreground="red")
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")

    def update_camera(self):
        """Update camera frame in GUI"""
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:

                # # Convert to RGB and resize for display
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame_resized = cv2.resize(frame_rgb, (320,180))
                # frame_resized = frame_resized[0:180, 70:250]
                #
                # # Convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(cv2.resize(frame, (320,180))[0:180, 70:250], (28, 28))
                frame = torch.from_numpy(frame).to(self.device)
                min_val = frame.min()
                max_val = frame.max()
                normalized = 1 - ((frame.to(torch.float32) - min_val) / (max_val - min_val + 1e-6))
                normalized = torch.pow(torch.pow(normalized + 0.3, 4).clip(0, 1), 4)
                normalized = (normalized / normalized.max()) * 255
                normalized = normalized.to(torch.uint8)

                self.current_frame = normalized

                # Convert to PIL Image and then to PhotoImage
                image = Image.fromarray(cv2.resize(normalized.cpu().numpy(), (224, 224), interpolation=cv2.INTER_NEAREST))
                photo = ImageTk.PhotoImage(image=image)

                # Update label
                self.camera_label.config(image=photo)
                setattr(self.camera_label, 'image', photo)  # Keep a reference

            # Schedule next update
            self.root.after(30, self.update_camera)

    def capture_and_predict(self):
        """Capture current frame and make prediction"""
        if self.current_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available")
            return

        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first")
            return

        try:
            # Preprocess the frame for MNIST prediction
            processed_image = self.preprocess_frame(self.current_frame)

            # Make prediction
            with torch.no_grad():
                output = self.model(processed_image)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            # Update prediction display
            pred_class = predicted.item()
            pred_confidence = confidence.item()

            self.prediction_label.config(
                text=f"Predicted: {pred_class} (Confidence: {pred_confidence:.2%})",
                foreground="green" if pred_confidence > 0.8 else "orange" if pred_confidence > 0.5 else "red"
            )

            # Update probability bars
            probs = probabilities.squeeze().cpu().numpy()
            for i, (bar, label) in enumerate(self.probability_bars):
                prob_percent = probs[i] * 100
                bar['value'] = prob_percent
                label.config(text=f"{prob_percent:.1f}%")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")

    def preprocess_frame(self, frame: torch.Tensor):
        """Preprocess camera frame for MNIST model"""
        frame = frame.to(torch.float32) / 255

        # Apply the same normalization as training (MNIST statistics)
        mean = 0.1307
        std = 0.3081
        frame = (frame - mean) / std

        # Add batch dimension and channel dimension
        tensor = frame.unsqueeze(0).unsqueeze(0)

        return tensor

    def browse_checkpoint(self):
        """Browse for checkpoint file"""
        filename = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            # filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.checkpoint_var.set(filename)

    def refresh_checkpoints(self):
        """Refresh list of available checkpoints"""
        checkpoint_dir = "checkpoints"
        checkpoints = []

        # Check checkpoint directories
        if os.path.exists(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for filename in files:
                    if filename.endswith(('.pth', '.pt')):
                        checkpoints.append(os.path.join(root, filename))

        # Also check root directory for model files
        for filename in os.listdir('.'):
            if filename.endswith('.pth') and 'model' in filename.lower():
                checkpoints.append(filename)

        if checkpoints:
            self.checkpoint_combo['values'] = sorted(checkpoints)
            self.checkpoint_var.set(checkpoints[0])
        else:
            self.checkpoint_combo['values'] = ["No checkpoints found"]
            self.checkpoint_var.set("No checkpoints found")

    def load_model(self):
        """Load selected model and checkpoint"""
        model_name = self.model_var.get()
        checkpoint_path = self.checkpoint_var.get()

        if not checkpoint_path or checkpoint_path == "No checkpoints found":
            messagebox.showwarning("No Checkpoint", "Please select a checkpoint file")
            return

        try:
            # Create model
            self.model = ModelRegistry.create(
                model_name,
                num_classes=10,
                input_channels=1
            )

            # Load checkpoint with error handling for different formats
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the checkpoint is just the state dict
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            messagebox.showinfo("Success", f"Model {model_name} loaded successfully!")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")
            self.model = None

    def toggle_detection(self):
        """Toggle iterative detection mode on/off"""
        if not self.is_detecting:
            # Start iterative detection
            if self.model is None:
                messagebox.showwarning("No Model", "Please load a model first")
                return

            self.is_detecting = True
            self.detect_button.config(text="Stop Detection")
            # Try to use accent style if available, otherwise use default style
            try:
                self.detect_button.config(style="Accent.TButton")
            except:
                pass
            self.iterative_detection()
        else:
            # Stop iterative detection
            self.is_detecting = False
            self.detect_button.config(text="Start Detection", style="TButton")

    def iterative_detection(self):
        """Perform iterative detection when mode is active"""
        if self.is_detecting and self.is_running:
            try:
                # Perform detection on current frame
                self.capture_and_predict()

                # Schedule next detection (every 500ms for better user experience)
                self.root.after(500, self.iterative_detection)

            except Exception as e:
                # If there's an error, stop detection and show error message
                self.is_detecting = False
                self.detect_button.config(text="Start Detection", style="TButton")
                messagebox.showerror("Detection Error", f"Iterative detection failed: {str(e)}")

    def clear_prediction(self):
        """Clear current prediction"""
        self.prediction_label.config(text="No prediction yet", foreground="black")
        for bar, label in self.probability_bars:
            bar['value'] = 0
            label.config(text="0%")

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        self.is_detecting = False  # Stop iterative detection
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = MNISTGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
