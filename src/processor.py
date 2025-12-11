"""
Image Processor Module
Main processing pipeline that coordinates all detection and conversion steps
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from grid_detector import GridDetector
from perspective_corrector import PerspectiveCorrector
from contour_detector import ContourDetector
from measurements import MeasurementCalculator, FrameMeasurements
from polar_converter import PolarConverter
from dat_generator import DATGenerator, DATFileData
from image_aligner import ImageAligner


@dataclass
class ProcessingResult:
    """Results from image processing."""
    success: bool
    error_message: str = ""

    # Calibration
    pixels_per_mm: float = 0.0
    grid_confidence: float = 0.0

    # Alignment
    rotation_applied: float = 0.0
    alignment_confidence: float = 0.0

    # Detection
    left_contour: Optional[np.ndarray] = None
    right_contour: Optional[np.ndarray] = None
    contour_confidence: float = 0.0
    detection_method: str = ""

    # Measurements
    a_mm: float = 0.0
    b_mm: float = 0.0
    dbl_mm: float = 0.0
    circ_mm: float = 0.0

    # Polar data
    right_radii: list = field(default_factory=list)
    left_radii: list = field(default_factory=list)

    # Images
    original_image: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    preview_image: Optional[np.ndarray] = None

    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        return (self.grid_confidence + self.contour_confidence) / 2

    @property
    def needs_user_adjustment(self) -> bool:
        """Check if confidence is below threshold."""
        return self.overall_confidence < 85


class ImageProcessor:
    """
    Main image processing pipeline for PIC2GRAPH.

    Coordinates all processing steps:
    1. Load and preprocess image
    2. Correct perspective
    3. Detect and calibrate grid
    4. Detect lens contours
    5. Calculate measurements
    6. Convert to polar format
    7. Generate DAT file
    """

    def __init__(self, grid_size_mm: float = 15.0):
        """
        Initialize the processor.

        Args:
            grid_size_mm: Size of grid squares in mm (default 8mm)
        """
        self.grid_size_mm = grid_size_mm
        self.grid_detector = GridDetector(grid_size_mm=grid_size_mm)
        self.perspective_corrector = PerspectiveCorrector()
        self.image_aligner = ImageAligner()
        self.contour_detector = ContourDetector()
        self.polar_converter = PolarConverter(num_points=1000)
        self.dat_generator = DATGenerator()

        # Processing options
        self.max_image_size = 2000  # Resize images larger than this
        self.contour_smoothing = 0.01
        self.enable_alignment = True  # Enable automatic image alignment

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process an image of glasses on grid paper.

        Args:
            image_path: Path to the input image

        Returns:
            ProcessingResult with all detection results
        """
        result = ProcessingResult(success=False)

        try:
            # Step 1: Load image
            image = self._load_image(image_path)
            if image is None:
                result.error_message = f"Could not load image: {image_path}"
                return result

            result.original_image = image.copy()

            # Step 2: Grid detection and calibration (before alignment)
            pixels_per_mm, grid_conf = self.grid_detector.detect_grid(image)
            result.pixels_per_mm = pixels_per_mm
            result.grid_confidence = grid_conf

            if pixels_per_mm <= 0:
                result.error_message = "Failed to detect grid. Please ensure grid paper is visible."
                return result

            # Step 3: Image alignment (rotation correction)
            # This corrects for tilted photos based on grid lines and glasses orientation
            if self.enable_alignment:
                aligned_image, rotation_angle = self.image_aligner.align_image(
                    image,
                    grid_lines_h=self.grid_detector.grid_lines_horizontal,
                    grid_lines_v=self.grid_detector.grid_lines_vertical,
                    use_glasses_detection=True
                )
                result.rotation_applied = rotation_angle
                result.alignment_confidence = self.image_aligner.confidence * 100

                # If significant rotation was applied, re-detect grid on aligned image
                # for better calibration accuracy
                if abs(rotation_angle) > 1.0:
                    pixels_per_mm_new, grid_conf_new = self.grid_detector.detect_grid(aligned_image)
                    if grid_conf_new > grid_conf:
                        result.pixels_per_mm = pixels_per_mm_new
                        result.grid_confidence = grid_conf_new
                        pixels_per_mm = pixels_per_mm_new

                image = aligned_image
                result.aligned_image = aligned_image

            result.processed_image = image

            # Step 4: Contour detection
            left_contour, right_contour, contour_conf = self.contour_detector.detect_contours(image)
            result.contour_confidence = contour_conf
            result.detection_method = self.contour_detector.detection_method

            if left_contour is None and right_contour is None:
                result.error_message = "Could not detect any lens contours."
                return result

            # Smooth contours
            if left_contour is not None:
                left_contour = self.contour_detector.smooth_contour(
                    left_contour, self.contour_smoothing
                )
            if right_contour is not None:
                right_contour = self.contour_detector.smooth_contour(
                    right_contour, self.contour_smoothing
                )

            result.left_contour = left_contour
            result.right_contour = right_contour

            # Step 6: Calculate measurements
            calculator = MeasurementCalculator(pixels_per_mm)
            a_mm, b_mm, circ_mm = calculator.get_averaged_measurements(
                left_contour, right_contour
            )
            measurements = calculator.calculate_measurements(left_contour, right_contour)

            result.a_mm = a_mm
            result.b_mm = b_mm
            result.circ_mm = circ_mm
            result.dbl_mm = measurements.dbl_mm

            # Step 7: Convert to polar format
            if left_contour is not None and right_contour is not None:
                # Average both contours for symmetry
                right_radii, _ = self.polar_converter.average_contours(
                    left_contour, right_contour, pixels_per_mm
                )
                left_radii = self.polar_converter.mirror_radii(right_radii)
            elif right_contour is not None:
                right_radii, _ = self.polar_converter.contour_to_polar(
                    right_contour, pixels_per_mm
                )
                left_radii = self.polar_converter.mirror_radii(right_radii)
            else:
                left_radii, _ = self.polar_converter.contour_to_polar(
                    left_contour, pixels_per_mm
                )
                right_radii = self.polar_converter.mirror_radii(left_radii)

            result.right_radii = right_radii
            result.left_radii = left_radii

            # Create preview image
            result.preview_image = self._create_preview(
                image, left_contour, right_contour, measurements, result.rotation_applied
            )

            result.success = True

        except Exception as e:
            result.error_message = str(e)
            import traceback
            traceback.print_exc()

        return result

    def generate_dat_file(self, result: ProcessingResult,
                          output_path: Optional[str] = None) -> str:
        """
        Generate a DAT file from processing results.

        Args:
            result: ProcessingResult from process_image
            output_path: Optional output path. If None, auto-generates.

        Returns:
            Path to generated file
        """
        if not result.success:
            raise ValueError(f"Cannot generate DAT from failed result: {result.error_message}")

        # Generate job ID and filename
        job_id = self.dat_generator.generate_job_id()

        if output_path is None:
            output_dir = os.getcwd()
            filename = self.dat_generator.generate_filename()
            output_path = os.path.join(output_dir, filename)

        # Prepare data
        data = DATFileData(
            job_id=job_id,
            dbl=result.dbl_mm,
            hbox_right=result.a_mm,
            hbox_left=result.a_mm,
            vbox_right=result.b_mm,
            vbox_left=result.b_mm,
            circ_right=result.circ_mm,
            circ_left=result.circ_mm,
            right_radii=result.right_radii,
            left_radii=result.left_radii
        )

        # Validate
        is_valid, errors = self.dat_generator.validate_data(data)
        if not is_valid:
            print(f"Warning: Validation issues: {errors}")

        # Generate file
        return self.dat_generator.generate(data, output_path)

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess an image."""
        image = cv2.imread(image_path)

        if image is None:
            return None

        # Resize if too large
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return image

    def _create_preview(self, image: np.ndarray,
                        left_contour: Optional[np.ndarray],
                        right_contour: Optional[np.ndarray],
                        measurements: FrameMeasurements,
                        rotation_applied: float = 0.0) -> np.ndarray:
        """Create a preview image with contours and measurements overlaid."""
        preview = image.copy()

        # Colors for labels
        color_a = (0, 165, 255)      # Orange for A
        color_b = (255, 0, 255)      # Magenta for B
        color_bridge = (0, 255, 0)   # Green for Bridge
        color_ref = (255, 255, 0)    # Cyan for Reference

        # Use right contour for labeling (or left if right not available)
        label_contour = right_contour if right_contour is not None else left_contour

        # Draw contours
        if left_contour is not None:
            cv2.drawContours(preview, [left_contour], -1, (0, 0, 255), 2)
            x, y, w, h = cv2.boundingRect(left_contour)
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if right_contour is not None:
            cv2.drawContours(preview, [right_contour], -1, (255, 0, 0), 2)
            x, y, w, h = cv2.boundingRect(right_contour)
            cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 0, 0), 1)

        # Draw labeled measurement lines on the contour
        if label_contour is not None:
            lx, ly, lw, lh = cv2.boundingRect(label_contour)

            # Draw A measurement line (horizontal width) with label
            a_y = ly + lh + 15  # Below the bounding box
            cv2.line(preview, (lx, a_y), (lx + lw, a_y), color_a, 2)
            # Draw end caps for A line
            cv2.line(preview, (lx, a_y - 5), (lx, a_y + 5), color_a, 2)
            cv2.line(preview, (lx + lw, a_y - 5), (lx + lw, a_y + 5), color_a, 2)
            # Draw A label
            a_label = f"A: {measurements.right_lens.a_mm if measurements.right_lens else (measurements.left_lens.a_mm if measurements.left_lens else 0):.1f}mm"
            cv2.putText(preview, a_label, (lx + lw // 2 - 40, a_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_a, 2)

            # Draw B measurement line (vertical height) with label
            b_x = lx - 15  # Left of the bounding box
            cv2.line(preview, (b_x, ly), (b_x, ly + lh), color_b, 2)
            # Draw end caps for B line
            cv2.line(preview, (b_x - 5, ly), (b_x + 5, ly), color_b, 2)
            cv2.line(preview, (b_x - 5, ly + lh), (b_x + 5, ly + lh), color_b, 2)
            # Draw B label (rotated text would be ideal but let's keep it simple)
            b_label = f"B: {measurements.right_lens.b_mm if measurements.right_lens else (measurements.left_lens.b_mm if measurements.left_lens else 0):.1f}mm"
            cv2.putText(preview, b_label, (b_x - 70, ly + lh // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_b, 2)

        # Draw DBL line with label
        if left_contour is not None and right_contour is not None:
            left_x, left_y, left_w, left_h = cv2.boundingRect(left_contour)
            right_x, right_y, _, right_h = cv2.boundingRect(right_contour)
            left_inner = left_x + left_w
            right_inner = right_x
            bridge_y = right_y + right_h // 2

            # Draw bridge line
            cv2.line(preview, (left_inner, bridge_y), (right_inner, bridge_y), color_bridge, 2)
            # Draw end caps for bridge line
            cv2.line(preview, (left_inner, bridge_y - 5), (left_inner, bridge_y + 5), color_bridge, 2)
            cv2.line(preview, (right_inner, bridge_y - 5), (right_inner, bridge_y + 5), color_bridge, 2)
            # Draw Bridge label
            bridge_center_x = (left_inner + right_inner) // 2
            bridge_label = f"Bridge: {measurements.dbl_mm:.1f}mm"
            cv2.putText(preview, bridge_label, (bridge_center_x - 50, bridge_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bridge, 2)

        # Draw reference grid square
        # Find a clear area in the image for the reference square
        self._draw_reference_square(preview, image, color_ref)

        # Add measurement summary text in top-left corner
        text_y = 30
        cv2.putText(preview, f"A: {measurements.right_lens.a_mm if measurements.right_lens else 0:.2f}mm",
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(preview, f"B: {measurements.right_lens.b_mm if measurements.right_lens else 0:.2f}mm",
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(preview, f"DBL: {measurements.dbl_mm:.2f}mm",
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show rotation applied if significant
        if abs(rotation_applied) > 0.5:
            text_y += 30
            cv2.putText(preview, f"Rotation: {rotation_applied:.1f} deg",
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return preview

    def _draw_reference_square(self, preview: np.ndarray, original: np.ndarray, color: Tuple[int, int, int]) -> None:
        """Draw a reference grid square to show the calibration basis."""
        if self.grid_detector.pixels_per_mm is None or self.grid_detector.pixels_per_mm <= 0:
            return

        # Calculate grid square size in pixels
        grid_size_px = int(self.grid_size_mm * self.grid_detector.pixels_per_mm)

        if grid_size_px < 10:  # Too small to be visible
            return

        img_h, img_w = original.shape[:2]

        # Find a good location for the reference square
        # Try bottom-right corner first, with some padding
        padding = 20
        ref_x = img_w - grid_size_px - padding - 100  # Extra space for label
        ref_y = img_h - grid_size_px - padding - 30   # Extra space for label

        # Ensure we're within bounds
        ref_x = max(padding, min(ref_x, img_w - grid_size_px - padding))
        ref_y = max(padding, min(ref_y, img_h - grid_size_px - padding))

        # Draw the reference square
        cv2.rectangle(preview, (ref_x, ref_y), (ref_x + grid_size_px, ref_y + grid_size_px), color, 2)

        # Draw corner markers
        marker_len = 8
        # Top-left corner
        cv2.line(preview, (ref_x - marker_len, ref_y), (ref_x + marker_len, ref_y), color, 2)
        cv2.line(preview, (ref_x, ref_y - marker_len), (ref_x, ref_y + marker_len), color, 2)
        # Top-right corner
        cv2.line(preview, (ref_x + grid_size_px - marker_len, ref_y), (ref_x + grid_size_px + marker_len, ref_y), color, 2)
        cv2.line(preview, (ref_x + grid_size_px, ref_y - marker_len), (ref_x + grid_size_px, ref_y + marker_len), color, 2)
        # Bottom-left corner
        cv2.line(preview, (ref_x - marker_len, ref_y + grid_size_px), (ref_x + marker_len, ref_y + grid_size_px), color, 2)
        cv2.line(preview, (ref_x, ref_y + grid_size_px - marker_len), (ref_x, ref_y + grid_size_px + marker_len), color, 2)
        # Bottom-right corner
        cv2.line(preview, (ref_x + grid_size_px - marker_len, ref_y + grid_size_px), (ref_x + grid_size_px + marker_len, ref_y + grid_size_px), color, 2)
        cv2.line(preview, (ref_x + grid_size_px, ref_y + grid_size_px - marker_len), (ref_x + grid_size_px, ref_y + grid_size_px + marker_len), color, 2)

        # Draw label
        ref_label = f"Reference: {self.grid_size_mm}mm"
        cv2.putText(preview, ref_label, (ref_x, ref_y + grid_size_px + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_and_save(image_path: str, output_dir: str = None,
                     grid_size_mm: float = 15.0) -> Tuple[bool, str, Dict]:
    """
    Convenience function to process an image and save the DAT file.

    Args:
        image_path: Path to input image
        output_dir: Directory for output files (default: same as input)
        grid_size_mm: Grid size in mm

    Returns:
        Tuple of (success, message, measurements_dict)
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    processor = ImageProcessor(grid_size_mm=grid_size_mm)
    result = processor.process_image(image_path)

    if not result.success:
        return False, result.error_message, {}

    # Save DAT file
    dat_path = processor.generate_dat_file(result, output_path=None)

    # Save preview image
    if result.preview_image is not None:
        preview_path = os.path.join(output_dir, "preview.png")
        cv2.imwrite(preview_path, result.preview_image)

    measurements = {
        'a_mm': result.a_mm,
        'b_mm': result.b_mm,
        'dbl_mm': result.dbl_mm,
        'circ_mm': result.circ_mm,
        'grid_confidence': result.grid_confidence,
        'contour_confidence': result.contour_confidence,
        'rotation_applied': result.rotation_applied,
        'alignment_confidence': result.alignment_confidence,
    }

    message = f"Success! DAT file saved to: {dat_path}"
    return True, message, measurements


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python processor.py <image_path> [grid_size_mm]")
        print("\nExample: python processor.py glasses.jpg 8")
        sys.exit(1)

    image_path = sys.argv[1]
    grid_size = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    print(f"Processing: {image_path}")
    print(f"Grid size: {grid_size}mm")
    print("-" * 40)

    success, message, measurements = process_and_save(image_path, grid_size_mm=grid_size)

    if success:
        print(message)
        print("\nMeasurements:")
        for key, value in measurements.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        print(f"Error: {message}")
        sys.exit(1)
