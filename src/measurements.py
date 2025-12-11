"""
Measurements Module
Calculates A, B, DBL, and CIRC measurements from contours
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LensMeasurements:
    """Measurements for a single lens."""
    a_mm: float  # Horizontal box dimension (HBOX)
    b_mm: float  # Vertical box dimension (VBOX)
    circ_mm: float  # Circumference
    center_x_mm: float
    center_y_mm: float


@dataclass
class FrameMeasurements:
    """Complete frame measurements."""
    right_lens: Optional[LensMeasurements]
    left_lens: Optional[LensMeasurements]
    dbl_mm: float  # Distance between lenses (bridge)
    total_width_mm: float  # Total frame width

    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        return {
            'right': {
                'a': self.right_lens.a_mm if self.right_lens else 0,
                'b': self.right_lens.b_mm if self.right_lens else 0,
                'circ': self.right_lens.circ_mm if self.right_lens else 0,
            } if self.right_lens else None,
            'left': {
                'a': self.left_lens.a_mm if self.left_lens else 0,
                'b': self.left_lens.b_mm if self.left_lens else 0,
                'circ': self.left_lens.circ_mm if self.left_lens else 0,
            } if self.left_lens else None,
            'dbl': self.dbl_mm,
            'total_width': self.total_width_mm,
        }


class MeasurementCalculator:
    """Calculates optical measurements from lens contours."""

    def __init__(self, pixels_per_mm: float):
        """
        Initialize measurement calculator.

        Args:
            pixels_per_mm: Calibration factor from grid detection
        """
        self.pixels_per_mm = pixels_per_mm

    def calculate_measurements(self,
                               left_contour: Optional[np.ndarray],
                               right_contour: Optional[np.ndarray]) -> FrameMeasurements:
        """
        Calculate all measurements from lens contours.

        Args:
            left_contour: Left lens contour (can be None)
            right_contour: Right lens contour (can be None)

        Returns:
            FrameMeasurements object with all calculated values
        """
        left_measurements = None
        right_measurements = None

        if left_contour is not None:
            left_measurements = self._measure_lens(left_contour)

        if right_contour is not None:
            right_measurements = self._measure_lens(right_contour)

        # Calculate DBL (distance between lenses)
        dbl_mm = self._calculate_dbl(left_contour, right_contour)

        # Calculate total width
        total_width_mm = self._calculate_total_width(
            left_contour, right_contour, dbl_mm
        )

        return FrameMeasurements(
            right_lens=right_measurements,
            left_lens=left_measurements,
            dbl_mm=dbl_mm,
            total_width_mm=total_width_mm
        )

    def _measure_lens(self, contour: np.ndarray) -> LensMeasurements:
        """Calculate measurements for a single lens contour."""
        # Bounding box for A and B measurements
        x, y, w, h = cv2.boundingRect(contour)
        a_mm = self._pixels_to_mm(w)
        b_mm = self._pixels_to_mm(h)

        # Circumference
        circ_pixels = cv2.arcLength(contour, closed=True)
        circ_mm = self._pixels_to_mm(circ_pixels)

        # Center (centroid)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2
            cy = y + h / 2

        center_x_mm = self._pixels_to_mm(cx)
        center_y_mm = self._pixels_to_mm(cy)

        return LensMeasurements(
            a_mm=round(a_mm, 2),
            b_mm=round(b_mm, 2),
            circ_mm=round(circ_mm, 2),
            center_x_mm=center_x_mm,
            center_y_mm=center_y_mm
        )

    def _calculate_dbl(self,
                       left_contour: Optional[np.ndarray],
                       right_contour: Optional[np.ndarray]) -> float:
        """
        Calculate DBL (Distance Between Lenses) - the bridge width.
        Uses precise method to find actual inner edges at bridge level.
        """
        if left_contour is None or right_contour is None:
            return 0.0

        # Use precise calculation
        return self._calculate_dbl_precise(left_contour, right_contour)

    def _calculate_dbl_precise(self,
                               left_contour: np.ndarray,
                               right_contour: np.ndarray) -> float:
        """
        Calculate DBL precisely by finding the closest points at the bridge level.
        The bridge is typically at the nasal (inner) edge where lenses are closest.
        """
        left_points = left_contour.reshape(-1, 2)
        right_points = right_contour.reshape(-1, 2)

        # First, ensure left is actually on the left and right is on the right
        left_center_x = np.mean(left_points[:, 0])
        right_center_x = np.mean(right_points[:, 0])

        if left_center_x > right_center_x:
            # Swap them
            left_points, right_points = right_points, left_points

        # Get the y-coordinates of centers (bridge level)
        left_center_y = np.mean(left_points[:, 1])
        right_center_y = np.mean(right_points[:, 1])
        bridge_y = (left_center_y + right_center_y) / 2

        # Find points near the bridge level (within 30% of lens height for more coverage)
        left_y_range = left_points[:, 1].max() - left_points[:, 1].min()
        right_y_range = right_points[:, 1].max() - right_points[:, 1].min()
        y_tolerance = min(left_y_range, right_y_range) * 0.3

        # Filter points near bridge level
        left_bridge_mask = np.abs(left_points[:, 1] - bridge_y) < y_tolerance
        right_bridge_mask = np.abs(right_points[:, 1] - bridge_y) < y_tolerance

        left_bridge_points = left_points[left_bridge_mask]
        right_bridge_points = right_points[right_bridge_mask]

        if len(left_bridge_points) == 0 or len(right_bridge_points) == 0:
            # Fallback to bounding box method
            left_x, _, left_w, _ = cv2.boundingRect(left_contour)
            right_x, _, _, _ = cv2.boundingRect(right_contour)
            left_inner = left_x + left_w
            right_inner = right_x
            dbl_pixels = abs(right_inner - left_inner)
            return round(self._pixels_to_mm(dbl_pixels), 2)

        # Find the innermost points at bridge level
        # For left lens: find the rightmost (max x) points
        # For right lens: find the leftmost (min x) points
        left_inner_x = left_bridge_points[:, 0].max()
        right_inner_x = right_bridge_points[:, 0].min()

        # Calculate DBL
        dbl_pixels = right_inner_x - left_inner_x

        # DBL should be positive (right inner edge is to the right of left inner edge)
        if dbl_pixels < 0:
            # Something is wrong - lenses might be overlapping in detection
            # Use absolute value
            dbl_pixels = abs(dbl_pixels)

        return round(self._pixels_to_mm(dbl_pixels), 2)

    def _calculate_total_width(self,
                               left_contour: Optional[np.ndarray],
                               right_contour: Optional[np.ndarray],
                               dbl_mm: float) -> float:
        """Calculate total frame width (A + DBL + A)."""
        total_pixels = 0

        if left_contour is not None:
            _, _, w, _ = cv2.boundingRect(left_contour)
            total_pixels += w

        if right_contour is not None:
            _, _, w, _ = cv2.boundingRect(right_contour)
            total_pixels += w

        total_mm = self._pixels_to_mm(total_pixels) + dbl_mm

        return round(total_mm, 2)

    def _pixels_to_mm(self, pixels: float) -> float:
        """Convert pixels to millimeters."""
        return pixels / self.pixels_per_mm

    def get_averaged_measurements(self,
                                  left_contour: Optional[np.ndarray],
                                  right_contour: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        Get averaged A, B, and CIRC measurements from both lenses.

        Returns:
            Tuple of (a_mm, b_mm, circ_mm) averaged from both lenses
        """
        measurements = self.calculate_measurements(left_contour, right_contour)

        a_values = []
        b_values = []
        circ_values = []

        if measurements.left_lens:
            a_values.append(measurements.left_lens.a_mm)
            b_values.append(measurements.left_lens.b_mm)
            circ_values.append(measurements.left_lens.circ_mm)

        if measurements.right_lens:
            a_values.append(measurements.right_lens.a_mm)
            b_values.append(measurements.right_lens.b_mm)
            circ_values.append(measurements.right_lens.circ_mm)

        a_avg = np.mean(a_values) if a_values else 0
        b_avg = np.mean(b_values) if b_values else 0
        circ_avg = np.mean(circ_values) if circ_values else 0

        return round(a_avg, 2), round(b_avg, 2), round(circ_avg, 2)
