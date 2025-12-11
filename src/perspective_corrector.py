"""
Perspective Correction Module
Corrects for camera angle and lens distortion
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class PerspectiveCorrector:
    """Corrects perspective distortion in images of glasses on grid paper."""

    def __init__(self):
        self.transformation_matrix: Optional[np.ndarray] = None
        self.original_size: Tuple[int, int] = (0, 0)
        self.corrected_size: Tuple[int, int] = (0, 0)

    def detect_paper_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the corners of the grid paper for perspective correction.

        Args:
            image: Input image (BGR format)

        Returns:
            Array of 4 corner points or None if not detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest rectangular contour (likely the paper)
        max_area = 0
        paper_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Look for quadrilateral
                if len(approx) == 4 and area > image.shape[0] * image.shape[1] * 0.1:
                    max_area = area
                    paper_contour = approx

        if paper_contour is not None:
            return self._order_points(paper_contour.reshape(4, 2))

        return None

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in: top-left, top-right, bottom-right, bottom-left order.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference: top-right has smallest, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def correct_perspective(self, image: np.ndarray,
                           corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply perspective correction to the image.

        Args:
            image: Input image (BGR format)
            corners: Optional pre-detected corners. If None, will auto-detect.

        Returns:
            Perspective-corrected image
        """
        self.original_size = (image.shape[1], image.shape[0])

        if corners is None:
            corners = self.detect_paper_corners(image)

        if corners is None:
            # No perspective correction possible, return original
            self.transformation_matrix = None
            self.corrected_size = self.original_size
            return image

        # Calculate the dimensions of the corrected image
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        max_height = int(max(height_left, height_right))

        # Destination points for the corrected image
        dst_points = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32), dst_points
        )

        # Apply the transformation
        corrected = cv2.warpPerspective(image, self.transformation_matrix,
                                        (max_width, max_height))

        self.corrected_size = (max_width, max_height)

        return corrected

    def correct_grid_alignment(self, image: np.ndarray,
                               grid_lines_h: List,
                               grid_lines_v: List) -> np.ndarray:
        """
        Fine-tune alignment based on detected grid lines.

        Args:
            image: Input image
            grid_lines_h: Detected horizontal grid lines
            grid_lines_v: Detected vertical grid lines

        Returns:
            Aligned image
        """
        if len(grid_lines_h) < 2 or len(grid_lines_v) < 2:
            return image

        # Calculate average angle of horizontal lines
        h_angles = []
        for line in grid_lines_h:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            h_angles.append(angle)

        avg_h_angle = np.mean(h_angles)

        # Calculate average angle of vertical lines
        v_angles = []
        for line in grid_lines_v:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            v_angles.append(angle)

        avg_v_angle = np.mean(v_angles)

        # Determine rotation needed (horizontal lines should be at 0 degrees)
        rotation_angle = -avg_h_angle

        # Only rotate if angle is significant
        if abs(rotation_angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

            # Calculate new image size to avoid cropping
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)

            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2

            image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                   borderValue=(255, 255, 255))

        return image

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from original to corrected coordinates.
        """
        if self.transformation_matrix is None:
            return point

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.transformation_matrix)
        return tuple(transformed[0, 0])

    def inverse_transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a point from corrected to original coordinates.
        """
        if self.transformation_matrix is None:
            return point

        inv_matrix = np.linalg.inv(self.transformation_matrix)
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, inv_matrix)
        return tuple(transformed[0, 0])
