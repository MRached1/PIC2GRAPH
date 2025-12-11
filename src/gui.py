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


@dataclass
class AppState:
    """Application state container."""
    original_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    display_image: Optional[np.ndarray] = None
    left_contour: Optional[np.ndarray] = None
    right_contour: Optional[np.ndarray] = None
    pixels_per_mm: float = 0.0
    measurements: Optional[FrameMeasurements] = None
    grid_confidence: float = 0.0
    contour_confidence: float = 0.0
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
        self.grid_detector = GridDetector(grid_size_mm=8.0)
        self.perspective_corrector = PerspectiveCorrector()
        self.contour_detector = ContourDetector()
        self.polar_converter = PolarConverter(num_points=1000)
        self.dat_generator = DATGenerator()

        # Display scale
        self.display_scale = 1.0
        self.max_display_size = 800

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

        # Right panel - Controls and measurements
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # File operations
        file_frame = ttk.LabelFrame(right_panel, text="File Operations", padding="10")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(file_frame, text="Load Image", command=self._load_image).grid(
            row=0, column=0, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Process Image", command=self._process_image).grid(
            row=1, column=0, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Generate DAT File", command=self._generate_dat).grid(
            row=2, column=0, sticky="ew", pady=2)

        file_frame.columnconfigure(0, weight=1)

        # Confidence display
        confidence_frame = ttk.LabelFrame(right_panel, text="Detection Confidence", padding="10")
        confidence_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

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
        measurements_frame = ttk.LabelFrame(right_panel, text="Measurements", padding="10")
        measurements_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

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
        adjust_frame = ttk.LabelFrame(right_panel, text="Manual Contour Editing", padding="10")
        adjust_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

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

        # Detection parameters
        params_frame = ttk.LabelFrame(right_panel, text="Detection Parameters", padding="10")
        params_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params_frame, text="Grid Size (mm):").grid(row=0, column=0, sticky="w")
        self.grid_size_var = tk.StringVar(value="8")
        grid_entry = ttk.Entry(params_frame, textvariable=self.grid_size_var, width=10)
        grid_entry.grid(row=0, column=1, sticky="e")

        params_frame.columnconfigure(1, weight=1)

        # Status bar
        status_frame = ttk.Frame(right_panel)
        status_frame.grid(row=5, column=0, sticky="ew")

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

            self._display_image(image)
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")

            # Reset state
            self.state.left_contour = None
            self.state.right_contour = None
            self.state.measurements = None
            self.control_points_left = []
            self.control_points_right = []
            self._update_measurements_display()
            self._reset_confidence_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def _process_image(self):
        """Process the loaded image."""
        if self.state.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.status_var.set("Processing image...")
        self.root.update()

        try:
            image = self.state.original_image.copy()

            # Update grid size
            try:
                grid_size = float(self.grid_size_var.get())
                self.grid_detector.grid_size_mm = grid_size
            except ValueError:
                grid_size = 8.0

            # Grid detection
            self.status_var.set("Detecting grid...")
            self.root.update()
            pixels_per_mm, grid_conf = self.grid_detector.detect_grid(image)
            self.state.pixels_per_mm = pixels_per_mm
            self.state.grid_confidence = grid_conf

            self.state.processed_image = image

            # Contour detection
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

            if contour_conf < 85:
                self.status_var.set(
                    f"Low confidence ({contour_conf:.1f}%). Enable editing to adjust contours manually."
                )
            else:
                self.status_var.set(
                    f"Done. Grid: {grid_conf:.1f}%, Contour: {contour_conf:.1f}%"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _init_control_points(self):
        """Initialize control points from detected contours."""
        self.control_points_left = []
        self.control_points_right = []

        if self.state.left_contour is not None:
            points = self.state.left_contour.reshape(-1, 2)
            # Sample ~20 control points
            step = max(1, len(points) // 20)
            self.control_points_left = [[int(p[0] * self.display_scale),
                                         int(p[1] * self.display_scale)] for p in points[::step]]

        if self.state.right_contour is not None:
            points = self.state.right_contour.reshape(-1, 2)
            step = max(1, len(points) // 20)
            self.control_points_right = [[int(p[0] * self.display_scale),
                                          int(p[1] * self.display_scale)] for p in points[::step]]

    def _display_image(self, image: np.ndarray):
        """Display an image on the canvas."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = rgb_image.shape[:2]
        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 700

        scale = min(canvas_w / w, canvas_h / h, 1.0)
        self.display_scale = scale

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(rgb_image)
        self.photo_image = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image, tags="image")

    def _update_display(self):
        """Update the display with contours overlaid."""
        if self.state.processed_image is None:
            return

        display = self.state.processed_image.copy()

        # Draw contours
        if self.state.left_contour is not None:
            cv2.drawContours(display, [self.state.left_contour], -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(self.state.left_contour)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if self.state.right_contour is not None:
            cv2.drawContours(display, [self.state.right_contour], -1, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(self.state.right_contour)
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 1)

        self.state.display_image = display
        self._display_image(display)

        # Draw control points if in edit mode
        if self.edit_mode:
            self._draw_control_points()

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
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                        fill="red", width=1, tags="control_line")

            # Draw points
            for i, (x, y) in enumerate(self.control_points_left):
                color = "yellow" if self.current_edit_side == "left" else "darkred"
                self.canvas.create_oval(
                    x - self.point_radius, y - self.point_radius,
                    x + self.point_radius, y + self.point_radius,
                    fill=color, outline="white", width=2,
                    tags=("control_point", f"left_{i}")
                )

        # Draw right lens control points (blue)
        if self.control_points_right:
            # Draw lines connecting points
            for i in range(len(self.control_points_right)):
                p1 = self.control_points_right[i]
                p2 = self.control_points_right[(i + 1) % len(self.control_points_right)]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                        fill="blue", width=1, tags="control_line")

            # Draw points
            for i, (x, y) in enumerate(self.control_points_right):
                color = "cyan" if self.current_edit_side == "right" else "darkblue"
                self.canvas.create_oval(
                    x - self.point_radius, y - self.point_radius,
                    x + self.point_radius, y + self.point_radius,
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

        # Find nearest control point
        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        min_dist = float('inf')
        nearest_idx = None

        for i, (x, y) in enumerate(points):
            dist = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
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
            points[self.dragging_point_index] = [event.x, event.y]
            self._draw_control_points()

    def _on_canvas_release(self, event):
        """Handle canvas release."""
        self.dragging_point_index = None

    def _on_canvas_right_click(self, event):
        """Handle right-click to add a new control point."""
        if not self.edit_mode:
            return

        points = self.control_points_left if self.current_edit_side == "left" else self.control_points_right

        if len(points) < 2:
            points.append([event.x, event.y])
        else:
            # Find the edge to insert the new point
            min_dist = float('inf')
            insert_idx = 0

            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]

                # Distance from click to line segment
                dist = self._point_to_segment_dist(event.x, event.y, p1[0], p1[1], p2[0], p2[1])
                if dist < min_dist:
                    min_dist = dist
                    insert_idx = i + 1

            points.insert(insert_idx, [event.x, event.y])

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
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
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
                # Create default control points if none exist
                if self.state.processed_image is not None:
                    h, w = self.state.processed_image.shape[:2]
                    # Create a default ellipse shape
                    cx, cy = int(w * 0.3 * self.display_scale), int(h * 0.5 * self.display_scale)
                    rx, ry = int(w * 0.1 * self.display_scale), int(h * 0.15 * self.display_scale)
                    self.control_points_left = self._create_ellipse_points(cx, cy, rx, ry)

                    cx2 = int(w * 0.7 * self.display_scale)
                    self.control_points_right = self._create_ellipse_points(cx2, cy, rx, ry)

            self._draw_control_points()
            self.status_var.set("Edit mode ON. Drag points to adjust, right-click to add, Ctrl+click to delete.")
        else:
            self.canvas.delete("control_point")
            self.canvas.delete("control_line")
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

            # Convert left control points to contour
            if self.control_points_left and len(self.control_points_left) >= 3:
                points = np.array(self.control_points_left) / self.display_scale
                points = np.vstack([points, points[0]])  # Close the contour

                tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
                u_new = np.linspace(0, 1, 500)
                x_new, y_new = splev(u_new, tck)
                self.state.left_contour = np.column_stack([x_new, y_new]).reshape(-1, 1, 2).astype(np.int32)

            # Convert right control points to contour
            if self.control_points_right and len(self.control_points_right) >= 3:
                points = np.array(self.control_points_right) / self.display_scale
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
