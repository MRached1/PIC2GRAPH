"""
PIC2GRAPH GUI Application
Tkinter-based interface for glasses contour detection and DAT file generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

from grid_detector import GridDetector
from perspective_corrector import PerspectiveCorrector
from contour_detector import ContourDetector
from measurements import MeasurementCalculator, FrameMeasurements
from polar_converter import PolarConverter
from dat_generator import DATGenerator, DATFileData
from image_aligner import ImageAligner


@dataclass
class AppState:
    """Application state container."""
    original_image: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    display_image: Optional[np.ndarray] = None
    left_contour: Optional[np.ndarray] = None
    right_contour: Optional[np.ndarray] = None
    pixels_per_mm: float = 0.0
    measurements: Optional[FrameMeasurements] = None
    grid_confidence: float = 0.0
    contour_confidence: float = 0.0
    rotation_applied: float = 0.0
    alignment_confidence: float = 0.0
    image_path: str = ""


class PIC2GRAPHGUI:
    """Main GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PIC2GRAPH - Glasses to Tracer File Converter")
        self.root.geometry("1400x900")

        # State
        self.state = AppState()

        # Processing modules
        self.grid_detector = GridDetector(grid_size_mm=15.0)
        self.perspective_corrector = PerspectiveCorrector()
        self.image_aligner = ImageAligner()
        self.contour_detector = ContourDetector()
        self.polar_converter = PolarConverter(num_points=1000)
        self.dat_generator = DATGenerator()

        # Display scale
        self.display_scale = 1.0
        self.max_display_size = 800

        # Zoom state
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.is_panning = False

        # Contour editing state
        self.edit_mode = False
        self.current_edit_side = "left"
        self.control_points_left: List[List[int]] = []
        self.control_points_right: List[List[int]] = []
        self.dragging_point_index: Optional[int] = None
        self.point_radius = 8

        # Store PhotoImage reference
        self.photo_image = None

        self._create_gui()

    def _create_gui(self):
        """Create the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel - Image display
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Canvas for image display
        canvas_frame = ttk.LabelFrame(left_panel, text="Image Preview (Click and drag points to edit contours)", padding="5")
        canvas_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(canvas_frame, width=900, height=700, bg="gray20")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bind mouse events for contour editing
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)  # Right-click to add point

        # Bind mouse events for zoom and pan
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)  # Windows zoom
        self.canvas.bind("<Button-2>", self._on_pan_start)  # Middle-click pan start
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)  # Middle-click pan drag
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)  # Middle-click pan end

        # Right panel - Controls and measurements (with scrollbar)
        right_container = ttk.Frame(main_frame)
        right_container.grid(row=0, column=1, sticky="nsew")

        # Create canvas and scrollbar for right panel
        right_canvas = tk.Canvas(right_container, width=320, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right_panel = ttk.Frame(right_canvas)

        right_panel.bind("<Configure>", lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.create_window((0, 0), window=right_panel, anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_canvas.pack(side="left", fill="both", expand=True)
        right_scrollbar.pack(side="right", fill="y")

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        # File operations
        file_frame = ttk.LabelFrame(right_panel, text="File Operations", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        ttk.Button(file_frame, text="Load Image", command=self._load_image).grid(
            row=0, column=0, columnspan=4, sticky="ew", pady=2)

        # Quick rotation buttons (90/180 degrees)
        ttk.Label(file_frame, text="Rotate:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Button(file_frame, text="↺90°", width=5, command=lambda: self._quick_rotate(-90)).grid(
            row=1, column=1, sticky="ew", padx=1, pady=2)
        ttk.Button(file_frame, text="180°", width=5, command=lambda: self._quick_rotate(180)).grid(
            row=1, column=2, sticky="ew", padx=1, pady=2)
        ttk.Button(file_frame, text="↻90°", width=5, command=lambda: self._quick_rotate(90)).grid(
            row=1, column=3, sticky="ew", padx=1, pady=2)

        ttk.Button(file_frame, text="Process Image", command=self._process_image).grid(
            row=2, column=0, columnspan=4, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Generate DAT File", command=self._generate_dat).grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=2)

        file_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(2, weight=1)
        file_frame.columnconfigure(3, weight=1)

        # Confidence display
        confidence_frame = ttk.LabelFrame(right_panel, text="Detection Confidence", padding="5")
        confidence_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(confidence_frame, text="Grid Detection:").grid(row=0, column=0, sticky="w")
        self.grid_confidence_var = tk.StringVar(value="--")
        ttk.Label(confidence_frame, textvariable=self.grid_confidence_var).grid(row=0, column=1, sticky="e")

        self.grid_confidence_bar = ttk.Progressbar(confidence_frame, length=150, mode='determinate')
        self.grid_confidence_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Label(confidence_frame, text="Contour Detection:").grid(row=2, column=0, sticky="w")
        self.contour_confidence_var = tk.StringVar(value="--")
        ttk.Label(confidence_frame, textvariable=self.contour_confidence_var).grid(row=2, column=1, sticky="e")

        self.contour_confidence_bar = ttk.Progressbar(confidence_frame, length=150, mode='determinate')
        self.contour_confidence_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)

        confidence_frame.columnconfigure(1, weight=1)

        # Measurements display
        measurements_frame = ttk.LabelFrame(right_panel, text="Measurements", padding="5")
        measurements_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))

        self.measurement_vars = {}
        measurements = [
            ("A (HBOX)", "a_mm"),
            ("B (VBOX)", "b_mm"),
            ("DBL (Bridge)", "dbl_mm"),
            ("Circumference", "circ_mm"),
            ("Pixels/mm", "ppm")
        ]

        for i, (label, key) in enumerate(measurements):
            ttk.Label(measurements_frame, text=f"{label}:").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="--")
            self.measurement_vars[key] = var
            ttk.Label(measurements_frame, textvariable=var, font=("Consolas", 10, "bold")).grid(
                row=i, column=1, sticky="e", pady=2)

        measurements_frame.columnconfigure(1, weight=1)

        # Manual adjustment controls
        adjust_frame = ttk.LabelFrame(right_panel, text="Manual Contour Editing", padding="5")
        adjust_frame.grid(row=3, column=0, sticky="ew", pady=(0, 5))

        self.edit_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adjust_frame, text="Enable Editing Mode",
                       variable=self.edit_mode_var,
                       command=self._toggle_edit_mode).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(adjust_frame, text="Edit Lens:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.edit_side_var = tk.StringVar(value="left")
        ttk.Radiobutton(adjust_frame, text="Left (Red)", variable=self.edit_side_var,
                       value="left", command=self._switch_edit_side).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(adjust_frame, text="Right (Blue)", variable=self.edit_side_var,
                       value="right", command=self._switch_edit_side).grid(row=2, column=1, sticky="w")

        ttk.Separator(adjust_frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(adjust_frame, text="Instructions:", font=("", 9, "bold")).grid(row=4, column=0, columnspan=2, sticky="w")
        instructions = "- Left-click + drag: Move point\n- Right-click: Add new point\n- Ctrl + click: Delete point"
        ttk.Label(adjust_frame, text=instructions, font=("", 8)).grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Button(adjust_frame, text="Reset Contours", command=self._reset_contours).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(adjust_frame, text="Apply Changes", command=self._apply_contour_changes).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        # Fine Rotation Control
        rotation_frame = ttk.LabelFrame(right_panel, text="Fine Rotation", padding="5")
        rotation_frame.grid(row=4, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(rotation_frame, text="Rotation Angle:").grid(row=0, column=0, sticky="w")
        self.rotation_var = tk.DoubleVar(value=0.0)
        self.rotation_slider = ttk.Scale(rotation_frame, from_=-45, to=45,
                                          variable=self.rotation_var,
                                          orient="horizontal",
                                          command=self._on_rotation_change)
        self.rotation_slider.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.rotation_label = ttk.Label(rotation_frame, text="0.0°", width=6)
        self.rotation_label.grid(row=0, column=2, sticky="e", padx=(5, 0))

        rotation_btn_frame = ttk.Frame(rotation_frame)
        rotation_btn_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(rotation_btn_frame, text="-1°", width=4,
                   command=lambda: self._adjust_rotation(-1)).pack(side="left", padx=2)
        ttk.Button(rotation_btn_frame, text="-0.1°", width=5,
                   command=lambda: self._adjust_rotation(-0.1)).pack(side="left", padx=2)
        ttk.Button(rotation_btn_frame, text="Reset", width=5,
                   command=lambda: self._adjust_rotation(0, reset=True)).pack(side="left", padx=2)
        ttk.Button(rotation_btn_frame, text="+0.1°", width=5,
                   command=lambda: self._adjust_rotation(0.1)).pack(side="left", padx=2)
        ttk.Button(rotation_btn_frame, text="+1°", width=4,
                   command=lambda: self._adjust_rotation(1)).pack(side="left", padx=2)

        ttk.Button(rotation_frame, text="Auto-Align Horizontal",
                   command=self._auto_align_horizontal).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        rotation_frame.columnconfigure(1, weight=1)

        # Zoom controls
        zoom_frame = ttk.LabelFrame(right_panel, text="Zoom Controls", padding="5")
        zoom_frame.grid(row=5, column=0, sticky="ew", pady=(0, 5))

        zoom_btn_frame = ttk.Frame(zoom_frame)
        zoom_btn_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(zoom_btn_frame, text="Zoom In (+)", width=10,
                   command=lambda: self._zoom(1.25)).pack(side="left", padx=2)
        ttk.Button(zoom_btn_frame, text="Zoom Out (-)", width=10,
                   command=lambda: self._zoom(0.8)).pack(side="left", padx=2)
        ttk.Button(zoom_btn_frame, text="Fit", width=5,
                   command=self._reset_zoom).pack(side="left", padx=2)

        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=8)
        self.zoom_label.grid(row=0, column=2, sticky="e", padx=(5, 0))

        ttk.Label(zoom_frame, text="Zoom:", font=("", 8)).grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.zoom_slider = ttk.Scale(zoom_frame, from_=50, to=500,
                                      orient="horizontal",
                                      command=self._on_zoom_slider_change)
        self.zoom_slider.set(100)
        self.zoom_slider.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 0))

        ttk.Label(zoom_frame, text="Mouse wheel to zoom, middle-click to pan", font=("", 7)).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))

        zoom_frame.columnconfigure(1, weight=1)

        # Detection parameters
        params_frame = ttk.LabelFrame(right_panel, text="Detection Parameters", padding="5")
        params_frame.grid(row=6, column=0, sticky="ew", pady=(0, 5))

        ttk.Label(params_frame, text="Reference Size (mm):").grid(row=0, column=0, sticky="w")
        self.grid_size_var = tk.StringVar(value="15")
        grid_entry = ttk.Entry(params_frame, textvariable=self.grid_size_var, width=10)
        grid_entry.grid(row=0, column=1, sticky="e")

        params_frame.columnconfigure(1, weight=1)

        # Status bar
        status_frame = ttk.Frame(right_panel)
        status_frame.grid(row=7, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                wraplength=300, justify="left")
        status_label.grid(row=0, column=0, sticky="w")

    def _load_image(self):
        """Load an image file."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)

        if not filepath:
            return

        try:
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Could not read image file")

            # Resize if too large
            max_dim = 2000
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            self.state.original_image = image
            self.state.processed_image = image.copy()
            self.state.image_path = filepath

            # Reset zoom and pan
            self.zoom_level = 1.0
            self.canvas_offset_x = 0
            self.canvas_offset_y = 0
            self._update_zoom_display()

            self._display_image(image)
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")

            # Reset state
            self.state.aligned_image = None
            self.state.left_contour = None
            self.state.right_contour = None
            self.state.measurements = None
            self.state.rotation_applied = 0.0
            self.state.alignment_confidence = 0.0
            self.control_points_left = []
            self.control_points_right = []
            self._update_measurements_display()
            self._reset_confidence_display()

            # Reset rotation
            self.rotation_var.set(0.0)
            self.rotation_label.config(text="0.0°")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def _zoom(self, factor: float):
        """Zoom the canvas by a factor."""
        if self.state.display_image is None and self.state.processed_image is None:
            return

        new_zoom = self.zoom_level * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            self._update_zoom_display()
            self._redraw_zoomed_image()

    def _reset_zoom(self):
        """Reset zoom to fit the image in the canvas."""
        self.zoom_level = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self._update_zoom_display()
        self._redraw_zoomed_image()

    def _on_zoom_slider_change(self, value):
        """Handle zoom slider change."""
        zoom_percent = float(value)
        self.zoom_level = zoom_percent / 100.0
        # Only update label, not slider (to avoid recursion)
        self.zoom_label.config(text=f"{int(zoom_percent)}%")
        self._redraw_zoomed_image()

    def _update_zoom_display(self):
        """Update zoom label and slider."""
        zoom_percent = int(self.zoom_level * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")
        # Temporarily unbind to avoid recursion when setting slider
        self.zoom_slider.config(command="")
        self.zoom_slider.set(zoom_percent)
        self.zoom_slider.config(command=self._on_zoom_slider_change)

    def _on_canvas_mousewheel(self, event):
        """Handle mouse wheel for zooming."""
        # Only zoom on canvas, not when scrolling controls
        if self.state.display_image is None and self.state.processed_image is None:
            return

        # Get mouse position on canvas for zoom centering
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Zoom factor based on scroll direction
        if event.delta > 0:
            factor = 1.15
        else:
            factor = 0.87

        old_zoom = self.zoom_level
        new_zoom = self.zoom_level * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))

        if new_zoom != old_zoom:
            # Adjust offset to zoom toward mouse position
            zoom_ratio = new_zoom / old_zoom
            self.canvas_offset_x = canvas_x - (canvas_x - self.canvas_offset_x) * zoom_ratio
            self.canvas_offset_y = canvas_y - (canvas_y - self.canvas_offset_y) * zoom_ratio

            self.zoom_level = new_zoom
            self._update_zoom_display()
            self._redraw_zoomed_image()

        return "break"  # Prevent default scroll behavior

    def _on_pan_start(self, event):
        """Start panning with middle mouse button."""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        """Handle pan dragging."""
        if not self.is_panning:
            return

        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        self.canvas_offset_x += dx
        self.canvas_offset_y += dy

        self.pan_start_x = event.x
        self.pan_start_y = event.y

        self._redraw_zoomed_image()

    def _on_pan_end(self, event):
        """End panning."""
        self.is_panning = False
        self.canvas.config(cursor="")

    def _redraw_zoomed_image(self):
        """Redraw the image with current zoom and pan settings."""
        # Get the image to display
        if self.state.display_image is not None:
            image = self.state.display_image.copy()
        elif self.state.processed_image is not None:
            image = self.state.processed_image.copy()
        else:
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = rgb_image.shape[:2]
        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 700

        # Calculate base scale to fit image
        base_scale = min(canvas_w / w, canvas_h / h, 1.0)
        self.display_scale = base_scale

        # Apply base scale first
        if base_scale < 1.0:
            new_w = int(w * base_scale)
            new_h = int(h * base_scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Apply zoom
        if self.zoom_level != 1.0:
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            if self.zoom_level > 1.0:
                rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(rgb_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")

        # Calculate position with offset
        x_pos = int(self.canvas_offset_x)
        y_pos = int(self.canvas_offset_y)

        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self.photo_image, tags="image")

        # Redraw control points if in edit mode
        if self.edit_mode:
            self._draw_control_points()

    def _on_rotation_change(self, value):
        """Handle rotation slider change."""
        angle = float(value)
        self.rotation_label.config(text=f"{angle:.1f}°")
        self._apply_rotation_preview()

    def _adjust_rotation(self, delta, reset=False):
        """Adjust rotation by a fixed amount."""
        if reset:
            self.rotation_var.set(0.0)
        else:
            current = self.rotation_var.get()
            new_val = max(-45, min(45, current + delta))
            self.rotation_var.set(new_val)
        self.rotation_label.config(text=f"{self.rotation_var.get():.1f}°")
        self._apply_rotation_preview()

    def _apply_rotation_preview(self):
        """Apply rotation to the image and update preview."""
        if self.state.original_image is None:
            return

        angle = self.rotation_var.get()

        if abs(angle) < 0.01:
            # No rotation needed
            rotated = self.state.original_image.copy()
        else:
            # Rotate the image
            rotated = self._rotate_image(self.state.original_image, angle)

        self.state.processed_image = rotated
        self._display_image(rotated)

        # Clear previous detection results when rotation changes
        self.state.left_contour = None
        self.state.right_contour = None
        self.state.measurements = None

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle in degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size to avoid cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix for the new center
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Apply rotation with white border
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated

    def _quick_rotate(self, degrees: int):
        """Quick rotation by 90 or 180 degrees."""
        if self.state.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Rotate the original image
        if degrees == 90:
            self.state.original_image = cv2.rotate(self.state.original_image, cv2.ROTATE_90_CLOCKWISE)
        elif degrees == -90:
            self.state.original_image = cv2.rotate(self.state.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif degrees == 180:
            self.state.original_image = cv2.rotate(self.state.original_image, cv2.ROTATE_180)

        # Reset fine rotation slider
        self.rotation_var.set(0.0)
        self.rotation_label.config(text="0.0°")

        # Update display
        self.state.processed_image = self.state.original_image.copy()
        self._display_image(self.state.original_image)

        # Clear detection results
        self.state.left_contour = None
        self.state.right_contour = None
        self.state.measurements = None

        self.status_var.set(f"Rotated {degrees}°")

    def _auto_align_horizontal(self):
        """Automatically detect and apply horizontal alignment."""
        if self.state.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.status_var.set("Auto-aligning...")
        self.root.update()

        # Use image aligner to detect rotation
        _, rotation_angle = self.image_aligner.align_image(
            self.state.original_image,
            use_glasses_detection=True
        )

        # Apply detected rotation
        self.rotation_var.set(rotation_angle)
        self.rotation_label.config(text=f"{rotation_angle:.1f}°")
        self._apply_rotation_preview()

        self.status_var.set(f"Auto-aligned: {rotation_angle:.1f}° rotation applied")

    def _process_image(self):
        """Process the loaded image."""
        if self.state.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.status_var.set("Processing image...")
        self.root.update()

        try:
            # Use the manually rotated image (from rotation slider)
            rotation_angle = self.rotation_var.get()
            if abs(rotation_angle) > 0.01:
                image = self._rotate_image(self.state.original_image, rotation_angle)
            else:
                image = self.state.original_image.copy()

            self.state.rotation_applied = rotation_angle
            self.state.alignment_confidence = 100.0 if abs(rotation_angle) > 0.01 else 0.0

            # Update grid size
            try:
                grid_size = float(self.grid_size_var.get())
                self.grid_detector.grid_size_mm = grid_size
            except ValueError:
                grid_size = 15.0

            # Grid/Reference detection on rotated image
            self.status_var.set("Detecting reference...")
            self.root.update()
            pixels_per_mm, grid_conf = self.grid_detector.detect_grid(image)
            self.state.pixels_per_mm = pixels_per_mm
            self.state.grid_confidence = grid_conf

            self.state.aligned_image = image
            self.state.processed_image = image

            # Contour detection on aligned image
            self.status_var.set("Detecting lens contours...")
            self.root.update()
            left_contour, right_contour, contour_conf = self.contour_detector.detect_contours(image)

            self.state.contour_confidence = contour_conf
            self.state.left_contour = left_contour
            self.state.right_contour = right_contour

            # Initialize control points from detected contours
            self._init_control_points()

            # Calculate measurements
            if pixels_per_mm > 0:
                calculator = MeasurementCalculator(pixels_per_mm)
                self.state.measurements = calculator.calculate_measurements(
                    left_contour, right_contour
                )

            # Update display
            self._update_display()
            self._update_confidence_display()
            self._update_measurements_display()

            rotation_info = f", Rot: {rotation_angle:.1f}deg" if abs(rotation_angle) > 0.5 else ""
            if contour_conf < 85:
                self.status_var.set(
                    f"Low confidence ({contour_conf:.1f}%){rotation_info}. Enable editing to adjust contours."
                )
            else:
                self.status_var.set(
                    f"Done. Grid: {grid_conf:.1f}%, Contour: {contour_conf:.1f}%{rotation_info}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _init_control_points(self):
        """Initialize control points from detected contours.

        Control points are stored in original image coordinates (not display coordinates)
        to allow proper handling of zoom and pan.
        """
        self.control_points_left = []
        self.control_points_right = []

        if self.state.left_contour is not None:
            points = self.state.left_contour.reshape(-1, 2)
            # Sample ~20 control points - store in original image coordinates
            step = max(1, len(points) // 20)
            self.control_points_left = [[int(p[0]), int(p[1])] for p in points[::step]]

        if self.state.right_contour is not None:
            points = self.state.right_contour.reshape(-1, 2)
            step = max(1, len(points) // 20)
            self.control_points_right = [[int(p[0]), int(p[1])] for p in points[::step]]

    def _display_image(self, image: np.ndarray):
        """Display an image on the canvas with zoom support."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = rgb_image.shape[:2]
        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 700

        # Calculate base scale to fit image
        base_scale = min(canvas_w / w, canvas_h / h, 1.0)
        self.display_scale = base_scale

        # Apply base scale first
        if base_scale < 1.0:
            new_w = int(w * base_scale)
            new_h = int(h * base_scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Apply zoom
        if self.zoom_level != 1.0:
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            if self.zoom_level > 1.0:
                rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(rgb_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")

        # Calculate position with offset
        x_pos = int(self.canvas_offset_x)
        y_pos = int(self.canvas_offset_y)

        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self.photo_image, tags="image")

    def _update_display(self):
        """Update the display with contours overlaid."""
        if self.state.processed_image is None:
            return

        display = self.state.processed_image.copy()

        # Colors for labels
        color_a = (0, 165, 255)      # Orange for A
        color_b = (255, 0, 255)      # Magenta for B
        color_bridge = (0, 255, 0)   # Green for Bridge
        color_ref = (255, 255, 0)    # Cyan for Reference

        # Draw contours (only when NOT in edit mode - in edit mode, control points show the contour)
        if not self.edit_mode:
            if self.state.left_contour is not None:
                cv2.drawContours(display, [self.state.left_contour], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(self.state.left_contour)
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 1)

            if self.state.right_contour is not None:
                cv2.drawContours(display, [self.state.right_contour], -1, (255, 0, 0), 2)
                x, y, w, h = cv2.boundingRect(self.state.right_contour)
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 1)

        # Use right contour for labeling (or left if right not available)
        label_contour = self.state.right_contour if self.state.right_contour is not None else self.state.left_contour

        # Draw labeled measurement lines on the contour (following optical measurement standards)
        if label_contour is not None:
            lx, ly, lw, lh = cv2.boundingRect(label_contour)

            # Get measurement values - calculate if not available
            if self.state.measurements is not None:
                if self.state.measurements.right_lens:
                    a_val = self.state.measurements.right_lens.a_mm
                    b_val = self.state.measurements.right_lens.b_mm
                elif self.state.measurements.left_lens:
                    a_val = self.state.measurements.left_lens.a_mm
                    b_val = self.state.measurements.left_lens.b_mm
                else:
                    a_val = b_val = 0
            elif self.state.pixels_per_mm > 0:
                # Calculate from bounding box if measurements not available
                a_val = lw / self.state.pixels_per_mm
                b_val = lh / self.state.pixels_per_mm
            else:
                a_val = b_val = 0

            # Calculate geometric center of the lens
            center_x = lx + lw // 2
            center_y = ly + lh // 2

            # Draw Datum Line (horizontal line through geometric center)
            datum_color = (128, 128, 128)  # Gray
            cv2.line(display, (lx - 30, center_y), (lx + lw + 30, center_y), datum_color, 1, cv2.LINE_AA)

            # Draw A measurement line (horizontal width) - ABOVE the lens as per standard
            a_y = ly - 20  # Above the bounding box
            cv2.line(display, (lx, a_y), (lx + lw, a_y), color_a, 3)
            # Draw end caps for A line
            cv2.line(display, (lx, a_y - 10), (lx, a_y + 10), color_a, 3)
            cv2.line(display, (lx + lw, a_y - 10), (lx + lw, a_y + 10), color_a, 3)
            # Draw arrows pointing inward
            cv2.line(display, (lx, a_y), (lx + 15, a_y - 6), color_a, 2)
            cv2.line(display, (lx, a_y), (lx + 15, a_y + 6), color_a, 2)
            cv2.line(display, (lx + lw, a_y), (lx + lw - 15, a_y - 6), color_a, 2)
            cv2.line(display, (lx + lw, a_y), (lx + lw - 15, a_y + 6), color_a, 2)
            # Draw A label with background - centered above the line
            a_label = f"A = {a_val:.1f}mm"
            (tw, th), _ = cv2.getTextSize(a_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = lx + lw // 2 - tw // 2
            label_y = a_y - 15
            cv2.rectangle(display, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(display, a_label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_a, 2)

            # Draw B measurement line (vertical height) - on RIGHT side as per standard
            b_x = lx + lw + 20  # Right of the bounding box
            cv2.line(display, (b_x, ly), (b_x, ly + lh), color_b, 3)
            # Draw end caps for B line
            cv2.line(display, (b_x - 10, ly), (b_x + 10, ly), color_b, 3)
            cv2.line(display, (b_x - 10, ly + lh), (b_x + 10, ly + lh), color_b, 3)
            # Draw arrows pointing inward
            cv2.line(display, (b_x, ly), (b_x - 6, ly + 15), color_b, 2)
            cv2.line(display, (b_x, ly), (b_x + 6, ly + 15), color_b, 2)
            cv2.line(display, (b_x, ly + lh), (b_x - 6, ly + lh - 15), color_b, 2)
            cv2.line(display, (b_x, ly + lh), (b_x + 6, ly + lh - 15), color_b, 2)
            # Draw B label with background - to the right of the line
            b_label = f"B = {b_val:.1f}mm"
            (tw, th), _ = cv2.getTextSize(b_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = b_x + 15
            label_y = ly + lh // 2 + th // 2
            cv2.rectangle(display, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(display, b_label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_b, 2)

            # Draw geometric center marker (small cross)
            cv2.line(display, (center_x - 8, center_y), (center_x + 8, center_y), (0, 0, 0), 2)
            cv2.line(display, (center_x, center_y - 8), (center_x, center_y + 8), (0, 0, 0), 2)

        # Draw DBL line with label
        if self.state.left_contour is not None and self.state.right_contour is not None:
            # Get precise bridge points
            left_points = self.state.left_contour.reshape(-1, 2)
            right_points = self.state.right_contour.reshape(-1, 2)

            # Ensure left is actually on the left
            left_center_x = np.mean(left_points[:, 0])
            right_center_x = np.mean(right_points[:, 0])

            if left_center_x > right_center_x:
                left_points, right_points = right_points, left_points

            # Calculate bridge level (y-coordinate)
            left_center_y = np.mean(left_points[:, 1])
            right_center_y = np.mean(right_points[:, 1])
            bridge_y = int((left_center_y + right_center_y) / 2)

            # Find points near bridge level
            left_y_range = left_points[:, 1].max() - left_points[:, 1].min()
            right_y_range = right_points[:, 1].max() - right_points[:, 1].min()
            y_tolerance = min(left_y_range, right_y_range) * 0.3

            left_bridge_mask = np.abs(left_points[:, 1] - bridge_y) < y_tolerance
            right_bridge_mask = np.abs(right_points[:, 1] - bridge_y) < y_tolerance

            left_bridge_points = left_points[left_bridge_mask]
            right_bridge_points = right_points[right_bridge_mask]

            if len(left_bridge_points) > 0 and len(right_bridge_points) > 0:
                # Find innermost points
                left_inner = int(left_bridge_points[:, 0].max())
                right_inner = int(right_bridge_points[:, 0].min())
            else:
                # Fallback to bounding box
                left_x, _, left_w, _ = cv2.boundingRect(self.state.left_contour)
                right_x, _, _, _ = cv2.boundingRect(self.state.right_contour)
                left_inner = left_x + left_w
                right_inner = right_x

            # Calculate bridge value
            if self.state.measurements is not None:
                bridge_val = self.state.measurements.dbl_mm
            elif self.state.pixels_per_mm > 0:
                bridge_val = abs(right_inner - left_inner) / self.state.pixels_per_mm
            else:
                bridge_val = 0

            # Draw bridge line at precise positions
            cv2.line(display, (left_inner, bridge_y), (right_inner, bridge_y), color_bridge, 3)
            # Draw end caps for bridge line
            cv2.line(display, (left_inner, bridge_y - 10), (left_inner, bridge_y + 10), color_bridge, 3)
            cv2.line(display, (right_inner, bridge_y - 10), (right_inner, bridge_y + 10), color_bridge, 3)
            # Draw arrows
            cv2.line(display, (left_inner, bridge_y), (left_inner + 15, bridge_y - 8), color_bridge, 2)
            cv2.line(display, (left_inner, bridge_y), (left_inner + 15, bridge_y + 8), color_bridge, 2)
            cv2.line(display, (right_inner, bridge_y), (right_inner - 15, bridge_y - 8), color_bridge, 2)
            cv2.line(display, (right_inner, bridge_y), (right_inner - 15, bridge_y + 8), color_bridge, 2)
            # Draw DBL/Bridge label with background
            bridge_center_x = (left_inner + right_inner) // 2
            bridge_label = f"DBL = {bridge_val:.1f}mm"
            (tw, th), _ = cv2.getTextSize(bridge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = bridge_center_x - tw // 2
            label_y = bridge_y - 15
            cv2.rectangle(display, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(display, bridge_label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bridge, 2)

            # Draw datum line through both lens centers
            if self.state.left_contour is not None and self.state.right_contour is not None:
                left_bbox = cv2.boundingRect(self.state.left_contour)
                right_bbox = cv2.boundingRect(self.state.right_contour)
                left_center_x_bbox = left_bbox[0] + left_bbox[2] // 2
                right_center_x_bbox = right_bbox[0] + right_bbox[2] // 2
                datum_color = (128, 128, 128)  # Gray
                # Extended datum line across both lenses
                cv2.line(display, (left_bbox[0] - 50, bridge_y), (right_bbox[0] + right_bbox[2] + 50, bridge_y),
                        datum_color, 1, cv2.LINE_AA)
                # Draw center markers for both lenses
                left_center_y = left_bbox[1] + left_bbox[3] // 2
                right_center_y = right_bbox[1] + right_bbox[3] // 2
                cv2.circle(display, (left_center_x_bbox, left_center_y), 5, (0, 0, 0), 2)
                cv2.circle(display, (right_center_x_bbox, right_center_y), 5, (0, 0, 0), 2)

        # Draw reference grid square
        self._draw_reference_square(display, color_ref)

        self.state.display_image = display
        self._display_image(display)

        # Draw control points if in edit mode
        if self.edit_mode:
            self._draw_control_points()

    def _draw_reference_square(self, display: np.ndarray, color: Tuple[int, int, int]) -> None:
        """Draw the detected reference square (with 'A' inside) on the display."""
        if self.state.pixels_per_mm <= 0:
            return

        # Get grid size from the entry
        try:
            grid_size_mm = float(self.grid_size_var.get())
        except ValueError:
            grid_size_mm = 15.0

        # Use the reference square detected by grid_detector
        if self.grid_detector.reference_square is not None:
            ref_x, ref_y, ref_w, ref_h = self.grid_detector.reference_square

            # Draw the reference square highlighting the detected square
            cv2.rectangle(display, (ref_x, ref_y), (ref_x + ref_w, ref_y + ref_h), color, 3)

            # Draw corner markers to highlight the corners
            marker_len = 15
            # Top-left corner
            cv2.line(display, (ref_x - marker_len, ref_y), (ref_x + marker_len, ref_y), color, 3)
            cv2.line(display, (ref_x, ref_y - marker_len), (ref_x, ref_y + marker_len), color, 3)
            # Top-right corner
            cv2.line(display, (ref_x + ref_w - marker_len, ref_y), (ref_x + ref_w + marker_len, ref_y), color, 3)
            cv2.line(display, (ref_x + ref_w, ref_y - marker_len), (ref_x + ref_w, ref_y + marker_len), color, 3)
            # Bottom-left corner
            cv2.line(display, (ref_x - marker_len, ref_y + ref_h), (ref_x + marker_len, ref_y + ref_h), color, 3)
            cv2.line(display, (ref_x, ref_y + ref_h - marker_len), (ref_x, ref_y + ref_h + marker_len), color, 3)
            # Bottom-right corner
            cv2.line(display, (ref_x + ref_w - marker_len, ref_y + ref_h), (ref_x + ref_w + marker_len, ref_y + ref_h), color, 3)
            cv2.line(display, (ref_x + ref_w, ref_y + ref_h - marker_len), (ref_x + ref_w, ref_y + ref_h + marker_len), color, 3)

            # Draw label with background
            ref_label = f"Reference: {grid_size_mm}x{grid_size_mm}mm"
            (tw, th), _ = cv2.getTextSize(ref_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_x = ref_x
            label_y = ref_y + ref_h + 30
            cv2.rectangle(display, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(display, ref_label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _image_to_canvas_coords(self, img_x: float, img_y: float) -> Tuple[float, float]:
        """Convert image coordinates to canvas coordinates (with zoom and pan)."""
        # Apply display scale and zoom
        total_scale = self.display_scale * self.zoom_level
        canvas_x = img_x * total_scale + self.canvas_offset_x
        canvas_y = img_y * total_scale + self.canvas_offset_y
        return canvas_x, canvas_y

    def _canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to image coordinates (accounting for zoom and pan)."""
        total_scale = self.display_scale * self.zoom_level
        img_x = (canvas_x - self.canvas_offset_x) / total_scale
        img_y = (canvas_y - self.canvas_offset_y) / total_scale
        return img_x, img_y

    def _draw_control_points(self):
        """Draw control points on the canvas."""
        self.canvas.delete("control_point")
        self.canvas.delete("control_line")

        # Draw left lens control points (red)
        if self.control_points_left:
            # Draw lines connecting points
            for i in range(len(self.control_points_left)):
                p1 = self.control_points_left[i]
                p2 = self.control_points_left[(i + 1) % len(self.control_points_left)]
                # Convert to canvas coordinates
                c1 = self._image_to_canvas_coords(p1[0], p1[1])
                c2 = self._image_to_canvas_coords(p2[0], p2[1])
                self.canvas.create_line(c1[0], c1[1], c2[0], c2[1],
                                        fill="red", width=2, tags="control_line")

            # Draw points
            for i, (x, y) in enumerate(self.control_points_left):
                cx, cy = self._image_to_canvas_coords(x, y)
                color = "yellow" if self.current_edit_side == "left" else "darkred"
                self.canvas.create_oval(
                    cx - self.point_radius, cy - self.point_radius,
                    cx + self.point_radius, cy + self.point_radius,
                    fill=color, outline="white", width=2,
                    tags=("control_point", f"left_{i}")
                )

        # Draw right lens control points (blue)
        if self.control_points_right:
            # Draw lines connecting points
            for i in range(len(self.control_points_right)):
                p1 = self.control_points_right[i]
                p2 = self.control_points_right[(i + 1) % len(self.control_points_right)]
                # Convert to canvas coordinates
                c1 = self._image_to_canvas_coords(p1[0], p1[1])
                c2 = self._image_to_canvas_coords(p2[0], p2[1])
                self.canvas.create_line(c1[0], c1[1], c2[0], c2[1],
                                        fill="blue", width=2, tags="control_line")

            # Draw points
            for i, (x, y) in enumerate(self.control_points_right):
                cx, cy = self._image_to_canvas_coords(x, y)
                color = "cyan" if self.current_edit_side == "right" else "darkblue"
                self.canvas.create_oval(
                    cx - self.point_radius, cy - self.point_radius,
                    cx + self.point_radius, cy + self.point_radius,
                    fill=color, outline="white", width=2,
                    tags=("control_point", f"right_{i}")
                )

    def _on_canvas_click(self, event):
        """Handle canvas click for contour editing."""
        if not self.edit_mode:
            return

        # Check if Ctrl is pressed for delete
        if event.state & 0x4:  # Ctrl key
            self._delete_nearest_point(event.x, event.y)
            return

        # Find nearest control point (compare in canvas coordinates)
        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        min_dist = float('inf')
        nearest_idx = None

        for i, (x, y) in enumerate(points):
            # Convert point to canvas coordinates for comparison
            cx, cy = self._image_to_canvas_coords(x, y)
            dist = ((event.x - cx) ** 2 + (event.y - cy) ** 2) ** 0.5
            if dist < min_dist and dist < 20:
                min_dist = dist
                nearest_idx = i

        self.dragging_point_index = nearest_idx

    def _on_canvas_drag(self, event):
        """Handle canvas drag for moving control points."""
        if not self.edit_mode or self.dragging_point_index is None:
            return

        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        if 0 <= self.dragging_point_index < len(points):
            # Convert canvas coords back to image coords
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
            points[self.dragging_point_index] = [int(img_x), int(img_y)]
            self._draw_control_points()

    def _on_canvas_release(self, event):
        """Handle canvas release."""
        self.dragging_point_index = None

    def _on_canvas_right_click(self, event):
        """Handle right-click to add a new control point."""
        if not self.edit_mode:
            return

        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        # Convert click to image coordinates
        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)

        if len(points) < 2:
            points.append([int(img_x), int(img_y)])
        else:
            # Find the edge to insert the new point (compare in canvas coordinates)
            min_dist = float('inf')
            insert_idx = 0

            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]

                # Convert points to canvas coords for distance calculation
                c1 = self._image_to_canvas_coords(p1[0], p1[1])
                c2 = self._image_to_canvas_coords(p2[0], p2[1])

                # Distance from click to line segment in canvas coords
                dist = self._point_to_segment_dist(event.x, event.y, c1[0], c1[1], c2[0], c2[1])
                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            points.insert(insert_idx, [int(img_x), int(img_y)])

        self._draw_control_points()
        self.status_var.set(f"Added point. Total: {len(points)} points")

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def _delete_nearest_point(self, x, y):
        """Delete the nearest control point."""
        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        if len(points) <= 3:
            self.status_var.set("Cannot delete: minimum 3 points required")
            return

        min_dist = float('inf')
        nearest_idx = None

        for i, (px, py) in enumerate(points):
            # Convert to canvas coordinates for comparison
            cx, cy = self._image_to_canvas_coords(px, py)
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx is not None and min_dist < 30:
            points.pop(nearest_idx)
            self._draw_control_points()
            self.status_var.set(f"Deleted point. Total: {len(points)} points")

    def _toggle_edit_mode(self):
        """Toggle contour editing mode."""
        self.edit_mode = self.edit_mode_var.get()

        if self.edit_mode:
            if not self.control_points_left and not self.control_points_right:
                # Create default control points if none exist (in image coordinates)
                if self.state.processed_image is not None:
                    h, w = self.state.processed_image.shape[:2]
                    # Create a default ellipse shape in image coordinates
                    cx, cy = int(w * 0.3), int(h * 0.5)
                    rx, ry = int(w * 0.1), int(h * 0.15)
                    self.control_points_left = self._create_ellipse_points(cx, cy, rx, ry)

                    cx2 = int(w * 0.7)
                    self.control_points_right = self._create_ellipse_points(cx2, cy, rx, ry)

            # Refresh display to hide the original contours
            self._update_display()
            self.status_var.set("Edit mode ON. Drag points to adjust, right-click to add, Ctrl+click to delete.")
        else:
            self.canvas.delete("control_point")
            self.canvas.delete("control_line")
            # Refresh display to show the original contours again
            self._update_display()
            self.status_var.set("Edit mode OFF.")

    def _create_ellipse_points(self, cx, cy, rx, ry, num_points=16):
        """Create control points in an ellipse shape."""
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = int(cx + rx * np.cos(angle))
            y = int(cy + ry * np.sin(angle))
            points.append([x, y])
        return points

    def _switch_edit_side(self):
        """Switch which lens is being edited."""
        self.current_edit_side = self.edit_side_var.get()
        if self.edit_mode:
            self._draw_control_points()
            self.status_var.set(f"Editing {'LEFT (red)' if self.current_edit_side == 'left' else 'RIGHT (blue)'} lens")

    def _apply_contour_changes(self):
        """Apply the edited control points to create new contours."""
        if not self.control_points_left and not self.control_points_right:
            messagebox.showwarning("Warning", "No control points to apply.")
            return

        try:
            from scipy.interpolate import splprep, splev

            # Convert left control points to contour (points are already in image coordinates)
            if self.control_points_left and len(self.control_points_left) >= 3:
                points = np.array(self.control_points_left, dtype=np.float64)
                points = np.vstack([points, points[0]])  # Close the contour

                tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, 500)
                x_new, y_new = splev(u_new, tck)
                self.state.left_contour = np.column_stack([x_new, y_new]).reshape(-1, 1, 2).astype(np.int32)

            # Convert right control points to contour (points are already in image coordinates)
            if self.control_points_right and len(self.control_points_right) >= 3:
                points = np.array(self.control_points_right, dtype=np.float64)
                points = np.vstack([points, points[0]])

                tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, 500)
                x_new, y_new = splev(u_new, tck)
                self.state.right_contour = np.column_stack([x_new, y_new]).reshape(-1, 1, 2).astype(np.int32)

            # Recalculate measurements
            if self.state.pixels_per_mm > 0:
                calculator = MeasurementCalculator(self.state.pixels_per_mm)
                self.state.measurements = calculator.calculate_measurements(
                    self.state.left_contour, self.state.right_contour
                )
                self._update_measurements_display()

            # Update display
            self._update_display()
            self.status_var.set("Contour changes applied successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply changes: {str(e)}")

    def _reset_contours(self):
        """Reset contours by re-processing."""
        if self.state.original_image is not None:
            self._process_image()

    def _generate_dat(self):
        """Generate the DAT file."""
        if self.state.left_contour is None and self.state.right_contour is None:
            messagebox.showwarning("Warning", "No contours detected. Process an image first.")
            return

        if self.state.pixels_per_mm <= 0:
            messagebox.showwarning("Warning", "Grid calibration failed. Cannot generate accurate file.")
            return

        try:
            calculator = MeasurementCalculator(self.state.pixels_per_mm)
            a_mm, b_mm, circ_mm = calculator.get_averaged_measurements(
                self.state.left_contour, self.state.right_contour
            )

            # Convert contours to polar format
            if self.state.left_contour is not None and self.state.right_contour is not None:
                right_radii, _ = self.polar_converter.average_contours(
                    self.state.left_contour,
                    self.state.right_contour,
                    self.state.pixels_per_mm
                )
                left_radii = self.polar_converter.mirror_radii(right_radii)
            elif self.state.right_contour is not None:
                right_radii, _ = self.polar_converter.contour_to_polar(
                    self.state.right_contour, self.state.pixels_per_mm
                )
                left_radii = self.polar_converter.mirror_radii(right_radii)
            else:
                left_radii, _ = self.polar_converter.contour_to_polar(
                    self.state.left_contour, self.state.pixels_per_mm
                )
                right_radii = self.polar_converter.mirror_radii(left_radii)

            dbl_mm = 0
            if self.state.measurements:
                dbl_mm = self.state.measurements.dbl_mm

            job_id = self.dat_generator.generate_job_id()
            data = DATFileData(
                job_id=job_id,
                dbl=dbl_mm,
                hbox_right=a_mm,
                hbox_left=a_mm,
                vbox_right=b_mm,
                vbox_left=b_mm,
                circ_right=circ_mm,
                circ_left=circ_mm,
                right_radii=right_radii,
                left_radii=left_radii
            )

            default_filename = self.dat_generator.generate_filename()
            filepath = filedialog.asksaveasfilename(
                defaultextension=".DAT",
                filetypes=[("DAT files", "*.DAT"), ("All files", "*.*")],
                initialfile=default_filename
            )

            if not filepath:
                return

            self.dat_generator.generate(data, filepath)

            self.status_var.set(f"DAT file saved: {os.path.basename(filepath)}")
            messagebox.showinfo(
                "Success",
                f"DAT file generated!\n\n"
                f"File: {filepath}\n\n"
                f"Measurements:\n"
                f"  A (HBOX): {a_mm:.2f} mm\n"
                f"  B (VBOX): {b_mm:.2f} mm\n"
                f"  DBL: {dbl_mm:.2f} mm\n"
                f"  CIRC: {circ_mm:.2f} mm"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate DAT file: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_confidence_display(self):
        """Update confidence indicators."""
        grid_conf = self.state.grid_confidence
        self.grid_confidence_var.set(f"{grid_conf:.1f}%")
        self.grid_confidence_bar['value'] = grid_conf

        contour_conf = self.state.contour_confidence
        self.contour_confidence_var.set(f"{contour_conf:.1f}%")
        self.contour_confidence_bar['value'] = contour_conf

    def _reset_confidence_display(self):
        """Reset confidence display."""
        self.grid_confidence_var.set("--")
        self.grid_confidence_bar['value'] = 0
        self.contour_confidence_var.set("--")
        self.contour_confidence_bar['value'] = 0

    def _update_measurements_display(self):
        """Update measurements display."""
        if self.state.measurements is None:
            for var in self.measurement_vars.values():
                var.set("--")
            return

        m = self.state.measurements

        a_values = []
        b_values = []
        circ_values = []

        if m.left_lens:
            a_values.append(m.left_lens.a_mm)
            b_values.append(m.left_lens.b_mm)
            circ_values.append(m.left_lens.circ_mm)
        if m.right_lens:
            a_values.append(m.right_lens.a_mm)
            b_values.append(m.right_lens.b_mm)
            circ_values.append(m.right_lens.circ_mm)

        a_avg = np.mean(a_values) if a_values else 0
        b_avg = np.mean(b_values) if b_values else 0
        circ_avg = np.mean(circ_values) if circ_values else 0

        self.measurement_vars['a_mm'].set(f"{a_avg:.2f} mm")
        self.measurement_vars['b_mm'].set(f"{b_avg:.2f} mm")
        self.measurement_vars['dbl_mm'].set(f"{m.dbl_mm:.2f} mm")
        self.measurement_vars['circ_mm'].set(f"{circ_avg:.2f} mm")
        self.measurement_vars['ppm'].set(f"{self.state.pixels_per_mm:.2f}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = PIC2GRAPHGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
