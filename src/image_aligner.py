"""
Image Alignment Module
Detects and corrects rotation/orientation of eyeglasses images before contour detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class ImageAligner:
    """
    Aligns images of eyeglasses for optimal contour detection.

    Handles:
    1. Grid-based rotation correction (using detected grid lines)
    2. Glasses frame orientation detection
    3. Automatic rotation to horizontal orientation
    """

    def __init__(self):
        self.rotation_angle: float = 0.0
        self.detected_orientation: str = "unknown"  # 'horizontal', 'vertical', 'diagonal'
        self.confidence: float = 0.0
        self.grid_angle: float = 0.0
        self.glasses_angle: float = 0.0

    def align_image(self, image: np.ndarray,
                    grid_lines_h: Optional[List] = None,
                    grid_lines_v: Optional[List] = None,
                    use_glasses_detection: bool = True) -> Tuple[np.ndarray, float]:
        """
        Align the image based on grid lines and/or glasses frame detection.

        Args:
            image: Input BGR image
            grid_lines_h: Optional pre-detected horizontal grid lines
            grid_lines_v: Optional pre-detected vertical grid lines
            use_glasses_detection: Whether to also use glasses frame detection

        Returns:
            Tuple of (aligned_image, rotation_angle_degrees)
        """
        h, w = image.shape[:2]
        rotation_candidates = []

        # Method 1: Grid-based alignment
        if grid_lines_h is not None or grid_lines_v is not None:
            grid_angle, grid_conf = self._detect_grid_rotation(
                image, grid_lines_h, grid_lines_v
            )
            if grid_conf > 0.3:
                rotation_candidates.append((grid_angle, grid_conf * 1.2, 'grid'))
                self.grid_angle = grid_angle

        # Method 2: Detect grid lines ourselves if not provided
        if not rotation_candidates:
            grid_angle, grid_conf = self._detect_grid_lines_and_angle(image)
            if grid_conf > 0.3:
                rotation_candidates.append((grid_angle, grid_conf, 'grid_auto'))
                self.grid_angle = grid_angle

        # Method 3: Glasses frame detection
        if use_glasses_detection:
            glasses_angle, glasses_conf = self._detect_glasses_orientation(image)
            if glasses_conf > 0.3:
                rotation_candidates.append((glasses_angle, glasses_conf, 'glasses'))
                self.glasses_angle = glasses_angle

        # Choose the best rotation
        if not rotation_candidates:
            self.rotation_angle = 0.0
            self.confidence = 0.0
            return image, 0.0

        # Sort by confidence and select best
        rotation_candidates.sort(key=lambda x: x[1], reverse=True)
        best_angle, best_conf, best_method = rotation_candidates[0]

        # If angles are similar, average them for better accuracy
        if len(rotation_candidates) >= 2:
            angle1, conf1, _ = rotation_candidates[0]
            angle2, conf2, _ = rotation_candidates[1]
            if abs(angle1 - angle2) < 5:  # Similar angles
                # Weighted average
                best_angle = (angle1 * conf1 + angle2 * conf2) / (conf1 + conf2)
                best_conf = (conf1 + conf2) / 2

        self.rotation_angle = best_angle
        self.confidence = best_conf

        # Only rotate if angle is significant
        if abs(best_angle) < 0.3:
            return image, 0.0

        # Apply rotation
        aligned = self._rotate_image(image, best_angle)

        return aligned, best_angle

    def _detect_grid_rotation(self, image: np.ndarray,
                              h_lines: Optional[List],
                              v_lines: Optional[List]) -> Tuple[float, float]:
        """Detect rotation angle from pre-detected grid lines."""
        angles = []

        # Analyze horizontal lines
        if h_lines and len(h_lines) >= 2:
            for line in h_lines:
                if len(line) >= 4:
                    x1, y1, x2, y2 = line[:4]
                    if abs(x2 - x1) > 20:  # Line has significant length
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        # Normalize to [-45, 45] range for horizontal lines
                        while angle > 45:
                            angle -= 90
                        while angle < -45:
                            angle += 90
                        angles.append(angle)

        # Analyze vertical lines
        if v_lines and len(v_lines) >= 2:
            for line in v_lines:
                if len(line) >= 4:
                    x1, y1, x2, y2 = line[:4]
                    if abs(y2 - y1) > 20:  # Line has significant length
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        # Convert to deviation from vertical (90 degrees)
                        angle = angle - 90
                        while angle > 45:
                            angle -= 90
                        while angle < -45:
                            angle += 90
                        angles.append(angle)

        if len(angles) < 2:
            return 0.0, 0.0

        # Remove outliers using IQR
        angles = np.array(angles)
        q1, q3 = np.percentile(angles, [25, 75])
        iqr = q3 - q1
        mask = (angles >= q1 - 1.5 * iqr) & (angles <= q3 + 1.5 * iqr)
        filtered_angles = angles[mask]

        if len(filtered_angles) < 2:
            filtered_angles = angles

        # Calculate mean and confidence based on consistency
        mean_angle = np.mean(filtered_angles)
        std_angle = np.std(filtered_angles)

        # Confidence is higher when angles are consistent
        confidence = max(0, 1.0 - std_angle / 10.0)

        return -mean_angle, confidence  # Negative to counter-rotate

    def _detect_grid_lines_and_angle(self, image: np.ndarray) -> Tuple[float, float]:
        """Detect grid lines and compute rotation angle."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=50, minLineLength=50, maxLineGap=10
        )

        if lines is None or len(lines) < 4:
            return 0.0, 0.0

        # Separate into horizontal and vertical candidates
        h_angles = []
        v_angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < 30:
                continue

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Classify as horizontal-ish or vertical-ish
            if abs(angle) < 30 or abs(angle) > 150:  # Horizontal
                # Normalize
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                h_angles.append(angle)
            elif 60 < abs(angle) < 120:  # Vertical
                # Convert to deviation from 90
                if angle > 0:
                    v_angles.append(angle - 90)
                else:
                    v_angles.append(angle + 90)

        all_angles = h_angles + v_angles

        if len(all_angles) < 3:
            return 0.0, 0.0

        # Use median for robustness
        median_angle = np.median(all_angles)
        mad = np.median(np.abs(np.array(all_angles) - median_angle))

        confidence = max(0, 1.0 - mad / 5.0)

        return -median_angle, confidence

    def _detect_glasses_orientation(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Detect the orientation of the glasses frame itself.
        Uses the overall elongated shape of the glasses to determine angle.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Apply bilateral filter and edge detection
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, 30, 100)

        # Morphological operations to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0, 0.0

        # Find large contours that might be the glasses frame or lenses
        significant_contours = []
        min_area = h * w * 0.005  # At least 0.5% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and len(contour) >= 5:
                significant_contours.append(contour)

        if not significant_contours:
            return 0.0, 0.0

        # Analyze the combined bounding region of significant contours
        # or use minimum area rectangle
        angles = []
        weights = []

        for contour in significant_contours[:10]:  # Limit to top 10
            area = cv2.contourArea(contour)

            # Fit minimum area rectangle
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect

            # Ensure width > height (standard orientation)
            if width < height:
                width, height = height, width
                angle += 90

            # Normalize angle to [-45, 45]
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90

            # Weight by area (larger contours more important)
            angles.append(angle)
            weights.append(area)

        if not angles:
            return 0.0, 0.0

        # Weighted average
        weights = np.array(weights)
        angles = np.array(angles)
        weighted_angle = np.average(angles, weights=weights)

        # Confidence based on consistency
        if len(angles) > 1:
            weighted_std = np.sqrt(np.average((angles - weighted_angle)**2, weights=weights))
            confidence = max(0, 1.0 - weighted_std / 15.0)
        else:
            confidence = 0.5

        return -weighted_angle, confidence

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by the given angle, expanding canvas to avoid cropping.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Rotated image
        """
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

    def detect_and_correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Detect if glasses are oriented correctly and rotate if needed.
        Glasses should be horizontal (temples extending left and right).

        Args:
            image: Input BGR image

        Returns:
            Tuple of (corrected_image, orientation_description, rotation_applied)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect dark regions (frames)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find the main frame contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.detected_orientation = "unknown"
            return image, "unknown", 0.0

        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < h * w * 0.01:  # Too small
            self.detected_orientation = "unknown"
            return image, "unknown", 0.0

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(largest)
        center, (rect_w, rect_h), angle = rect

        # Determine orientation
        if rect_w > rect_h:
            # Already wider than tall
            aspect = rect_w / rect_h
            if aspect > 1.5:
                self.detected_orientation = "horizontal"
                # Small angle correction if needed
                if abs(angle) > 2:
                    # Normalize angle
                    if angle > 45:
                        angle -= 90
                    elif angle < -45:
                        angle += 90
                    corrected = self._rotate_image(image, -angle)
                    return corrected, "horizontal", -angle
                return image, "horizontal", 0.0
            else:
                self.detected_orientation = "square"
                return image, "square", 0.0
        else:
            # Taller than wide - might be rotated 90 degrees
            aspect = rect_h / rect_w
            if aspect > 1.5:
                self.detected_orientation = "vertical"
                # Need to rotate 90 degrees
                # Determine which direction based on angle
                if angle > 0:
                    rotation = 90 - angle
                else:
                    rotation = -90 - angle
                corrected = self._rotate_image(image, rotation)
                return corrected, "vertical_corrected", rotation
            else:
                self.detected_orientation = "square"
                return image, "square", 0.0

    def get_alignment_info(self) -> dict:
        """Return information about the alignment performed."""
        return {
            'rotation_angle': self.rotation_angle,
            'detected_orientation': self.detected_orientation,
            'confidence': self.confidence,
            'grid_angle': self.grid_angle,
            'glasses_angle': self.glasses_angle
        }
