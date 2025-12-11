"""
Polar Coordinate Converter Module
Converts contour points to radial format for Hoya GT5000 compatibility
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2


class PolarConverter:
    """Converts Cartesian contour to polar radii format."""

    def __init__(self, num_points: int = 1000):
        """
        Initialize converter.

        Args:
            num_points: Number of radial points (Hoya GT5000 uses 1000)
        """
        self.num_points = num_points
        self.angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    def contour_to_polar(self, contour: np.ndarray,
                         pixels_per_mm: float) -> Tuple[List[int], Tuple[float, float]]:
        """
        Convert a contour to polar radii format.

        The Hoya GT5000 format stores 1000 radii measured from the geometric
        center at equidistant angles from 0 to 360 degrees.

        Coordinate system:
        - 0° = Nasal (toward nose) = positive X
        - 90° = Superior (top) = positive Y
        - 180° = Temporal (toward temple) = negative X
        - 270° = Inferior (bottom) = negative Y
        - Rotation: Counter-clockwise

        Args:
            contour: Input contour points (N x 1 x 2 or N x 2)
            pixels_per_mm: Calibration factor

        Returns:
            Tuple of (radii_centimicrons, center_mm)
            - radii_centimicrons: List of 1000 radii in 1/100 mm
            - center_mm: (x, y) center in mm
        """
        # Reshape contour if needed
        points = contour.reshape(-1, 2).astype(np.float64)

        # Calculate geometric center (boxing center)
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2
        center_y = (points[:, 1].min() + points[:, 1].max()) / 2

        # Translate points to be centered at origin
        centered_points = points - np.array([center_x, center_y])

        # For each target angle, find the radius
        radii_pixels = []

        for angle in self.angles:
            # Create a ray from center in this direction
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            # Find intersection with contour
            radius = self._ray_contour_intersection(centered_points, ray_dir)
            radii_pixels.append(radius)

        # Convert to mm, then to centimicrons (1/100 mm)
        radii_mm = np.array(radii_pixels) / pixels_per_mm
        radii_centimicrons = (radii_mm * 100).astype(int).tolist()

        # Center in mm
        center_mm = (center_x / pixels_per_mm, center_y / pixels_per_mm)

        return radii_centimicrons, center_mm

    def _ray_contour_intersection(self, points: np.ndarray,
                                  direction: np.ndarray) -> float:
        """
        Find the intersection of a ray from origin with a contour.

        Uses ray-casting: for each contour segment, check if the ray
        intersects it, and return the closest intersection distance.
        """
        num_points = len(points)
        min_distance = 0

        # For each edge of the contour
        for i in range(num_points):
            p1 = points[i]
            p2 = points[(i + 1) % num_points]

            # Ray-segment intersection
            intersection = self._ray_segment_intersection(p1, p2, direction)

            if intersection is not None:
                distance = np.linalg.norm(intersection)
                if min_distance == 0 or distance < min_distance:
                    min_distance = distance

        # If no intersection found, use nearest point method
        if min_distance == 0:
            # Find the point closest to the ray
            for point in points:
                # Project point onto ray direction
                proj = np.dot(point, direction)
                if proj > 0:  # Point is in ray direction
                    # Distance along ray to closest point
                    distance = proj
                    if min_distance == 0 or distance < min_distance:
                        min_distance = distance

        return max(min_distance, 1)  # Minimum 1 pixel

    def _ray_segment_intersection(self, p1: np.ndarray, p2: np.ndarray,
                                  direction: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate intersection of ray from origin with line segment p1-p2.

        Ray: origin + t * direction (t >= 0)
        Segment: p1 + s * (p2 - p1) (0 <= s <= 1)
        """
        # Direction of the segment
        segment_dir = p2 - p1

        # Cross product for 2D
        cross = direction[0] * segment_dir[1] - direction[1] * segment_dir[0]

        if abs(cross) < 1e-10:
            return None  # Parallel

        # Calculate parameters
        t = (p1[0] * segment_dir[1] - p1[1] * segment_dir[0]) / cross
        s = (p1[0] * direction[1] - p1[1] * direction[0]) / cross

        # Check if intersection is valid (t >= 0, 0 <= s <= 1)
        if t >= 0 and 0 <= s <= 1:
            return direction * t

        return None

    def polar_to_contour(self, radii_centimicrons: List[int],
                         pixels_per_mm: float,
                         center: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Convert polar radii back to a contour (inverse operation).

        Args:
            radii_centimicrons: List of radii in 1/100 mm
            pixels_per_mm: Calibration factor
            center: Center point in pixels

        Returns:
            Contour array
        """
        radii_mm = np.array(radii_centimicrons) / 100.0
        radii_pixels = radii_mm * pixels_per_mm

        # Generate angles
        angles = np.linspace(0, 2 * np.pi, len(radii_centimicrons), endpoint=False)

        # Convert to Cartesian
        x = radii_pixels * np.cos(angles) + center[0]
        y = radii_pixels * np.sin(angles) + center[1]

        contour = np.column_stack([x, y]).reshape(-1, 1, 2).astype(np.int32)

        return contour

    def average_contours(self, contour1: np.ndarray, contour2: np.ndarray,
                         pixels_per_mm: float) -> Tuple[List[int], Tuple[float, float]]:
        """
        Average two contours by averaging their polar representations.

        Args:
            contour1: First contour
            contour2: Second contour
            pixels_per_mm: Calibration factor

        Returns:
            Tuple of (averaged_radii_centimicrons, center_mm)
        """
        radii1, center1 = self.contour_to_polar(contour1, pixels_per_mm)
        radii2, center2 = self.contour_to_polar(contour2, pixels_per_mm)

        # Average the radii
        averaged_radii = [(r1 + r2) // 2 for r1, r2 in zip(radii1, radii2)]

        # Average the centers
        averaged_center = (
            (center1[0] + center2[0]) / 2,
            (center1[1] + center2[1]) / 2
        )

        return averaged_radii, averaged_center

    def mirror_radii(self, radii: List[int]) -> List[int]:
        """
        Mirror radii horizontally (for creating left lens from right).

        This effectively reverses the angle direction:
        - Original 0° -> 180°
        - Original 90° -> 90°
        - Original 180° -> 0°
        - Original 270° -> 270°
        """
        # Flip horizontally: angle θ becomes 180° - θ (or π - θ)
        # In terms of indices: index i becomes (n/2 - i) mod n for first half
        # and (3n/2 - i) mod n for second half

        n = len(radii)
        mirrored = [0] * n

        for i in range(n):
            # New index after horizontal flip
            new_i = (n - i) % n
            mirrored[new_i] = radii[i]

        return mirrored

    def validate_radii(self, radii: List[int]) -> Tuple[bool, str]:
        """
        Validate that radii are reasonable for an eyeglass lens.

        Returns:
            Tuple of (is_valid, message)
        """
        if len(radii) != self.num_points:
            return False, f"Expected {self.num_points} points, got {len(radii)}"

        # Convert to mm for validation
        radii_mm = np.array(radii) / 100.0

        min_radius = radii_mm.min()
        max_radius = radii_mm.max()
        mean_radius = radii_mm.mean()

        # Typical lens size validation
        if min_radius < 5:
            return False, f"Minimum radius too small: {min_radius:.2f}mm"

        if max_radius > 50:
            return False, f"Maximum radius too large: {max_radius:.2f}mm"

        # Check for unreasonable aspect ratio
        if max_radius / min_radius > 3:
            return False, f"Aspect ratio too extreme: {max_radius/min_radius:.2f}"

        # Check for sudden jumps (likely detection errors)
        diffs = np.abs(np.diff(radii_mm))
        max_diff = diffs.max()
        if max_diff > 5:  # More than 5mm jump between adjacent points
            return False, f"Suspicious discontinuity: {max_diff:.2f}mm jump"

        return True, f"Valid: radii range {min_radius:.2f}-{max_radius:.2f}mm"
