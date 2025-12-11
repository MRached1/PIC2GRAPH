"""
Glasses Contour Detection Module
Detects the inner edge of eyeglass frames (lens opening)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy import ndimage
from scipy.interpolate import splprep, splev


class ContourDetector:
    """Detects eyeglass lens contours from images."""

    def __init__(self):
        self.left_contour: Optional[np.ndarray] = None
        self.right_contour: Optional[np.ndarray] = None
        self.left_center: Optional[Tuple[float, float]] = None
        self.right_center: Optional[Tuple[float, float]] = None
        self.confidence: float = 0.0
        self.detection_method: str = ""

    def detect_contours(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Detect both lens contours in the image.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (left_contour, right_contour, confidence)
        """
        # Try multiple detection methods
        methods = [
            ("edge_detection", self._detect_by_edges),
            ("color_segmentation", self._detect_by_color),
            ("adaptive_threshold", self._detect_by_adaptive_threshold),
            ("lens_reflection", self._detect_by_reflection),
        ]

        best_left = None
        best_right = None
        best_confidence = 0.0
        best_method = ""

        for method_name, method_func in methods:
            try:
                left, right, conf = method_func(image)
                if conf > best_confidence:
                    best_left = left
                    best_right = right
                    best_confidence = conf
                    best_method = method_name
            except Exception as e:
                continue

        self.left_contour = best_left
        self.right_contour = best_right
        self.confidence = best_confidence
        self.detection_method = best_method

        # Calculate centers
        if self.left_contour is not None:
            self.left_center = self._calculate_center(self.left_contour)
        if self.right_contour is not None:
            self.right_center = self._calculate_center(self.right_contour)

        return self.left_contour, self.right_contour, self.confidence

    def _detect_by_edges(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Detect contours using edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-scale edge detection
        edges1 = cv2.Canny(filtered, 30, 100)
        edges2 = cv2.Canny(filtered, 50, 150)
        edges3 = cv2.Canny(filtered, 70, 200)

        # Combine edges
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

        # Morphological operations to close gaps and fill holes
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Fill enclosed regions to get solid contours
        # This helps detect the lens opening as a solid shape
        filled = edges.copy()
        h, w = filled.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, mask, (0, 0), 255)
        filled_inv = cv2.bitwise_not(filled)
        solid_regions = cv2.bitwise_or(edges, filled_inv)

        # Find contours on the solid regions
        contours, _ = cv2.findContours(solid_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Also try the original edge contours
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Combine both sets
        all_contours = list(contours) + list(edge_contours)

        return self._select_lens_contours(all_contours, image.shape)

    def _detect_by_color(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Detect contours using color-based segmentation."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Detect dark regions (frames are typically darker than background)
        _, v_channel = cv2.threshold(hsv[:, :, 2], 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Also use L channel from LAB
        _, l_channel = cv2.threshold(lab[:, :, 0], 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine masks
        combined = cv2.bitwise_or(v_channel, l_channel)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return self._select_lens_contours(contours, image.shape)

    def _detect_by_adaptive_threshold(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Detect contours using adaptive thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return self._select_lens_contours(contours, image.shape)

    def _detect_by_reflection(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Detect lens edges by their reflections (for rimless frames)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for subtle brightness changes at lens edges
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        # Threshold
        _, binary = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=2)
        binary = cv2.erode(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return self._select_lens_contours(contours, image.shape, min_confidence=0.5)

    def _select_lens_contours(self, contours: List, image_shape: Tuple,
                              min_confidence: float = 0.7) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Select the two contours most likely to be lens openings.
        Handles both horizontal (standard) and vertical (rotated) glasses orientations.
        """
        if len(contours) < 1:
            return None, None, 0.0

        h, w = image_shape[:2]
        image_area = h * w
        image_center_x = w / 2
        image_center_y = h / 2

        # Filter and score contours
        candidates = []

        # Expected lens size range based on typical glasses dimensions
        # Typical lens: 40-65mm wide, 25-50mm tall
        # At various pixels/mm ratios, this could be 0.5% to 15% of typical image area
        min_area = image_area * 0.005  # 0.5% minimum
        max_area = image_area * 0.20   # 20% maximum

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Check if contour has enough points for ellipse fitting
            if len(contour) < 5:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Filter by bounding box aspect ratio (lens-like shapes)
            bbox_aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if bbox_aspect > 2.5:  # Lenses are usually not that elongated
                continue

            # Filter out contours that span most of the image (likely paper edge)
            # But allow contours that just touch edges (lens might extend to edge)
            edge_margin = 3
            spans_width = bw > w * 0.7
            spans_height = bh > h * 0.7
            if spans_width or spans_height:
                continue

            try:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse

                # Aspect ratio check
                ellipse_aspect = max(axes) / (min(axes) + 1e-6)
                if ellipse_aspect > 2.5:
                    continue

                # Calculate how well contour matches ellipse shape
                ellipse_contour = cv2.ellipse2Poly(
                    (int(center[0]), int(center[1])),
                    (int(axes[0]/2), int(axes[1]/2)),
                    int(angle), 0, 360, 5
                )
                match_score = cv2.matchShapes(contour, ellipse_contour.reshape(-1, 1, 2),
                                              cv2.CONTOURS_MATCH_I2, 0)

                # Calculate convexity (lenses should be mostly convex)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / (hull_area + 1e-6)

                # Score based on multiple factors
                shape_score = 1.0 / (1.0 + match_score * 5)  # Ellipse match
                solidity_score = solidity  # Higher is better (more convex)

                # Prefer contours that are reasonable size (not too small, not too large)
                # Optimal is around 2-8% of image area
                optimal_area = image_area * 0.04
                size_score = 1.0 / (1.0 + abs(area - optimal_area) / optimal_area)

                # Combined score
                score = shape_score * 0.4 + solidity_score * 0.3 + size_score * 0.3

                candidates.append({
                    'contour': contour,
                    'center': center,
                    'area': area,
                    'bbox': (x, y, bw, bh),
                    'score': score,
                    'solidity': solidity
                })

            except cv2.error:
                continue

        if len(candidates) < 1:
            return None, None, 0.0

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Determine if glasses are oriented horizontally or vertically
        # by checking the positions of top candidates
        if len(candidates) >= 2:
            c1, c2 = candidates[0], candidates[1]
            x_diff = abs(c1['center'][0] - c2['center'][0])
            y_diff = abs(c1['center'][1] - c2['center'][1])
            is_vertical = y_diff > x_diff
        else:
            is_vertical = h > w  # Assume based on image orientation

        # Select best two contours
        left_contour = None
        right_contour = None
        confidence = 0.0

        if len(candidates) >= 2:
            # Find two contours that are separated appropriately
            for i, c1 in enumerate(candidates[:8]):  # Check top 8
                for c2 in candidates[i+1:10]:
                    if is_vertical:
                        # For vertical orientation, check Y separation
                        y1, y2 = c1['center'][1], c2['center'][1]
                        separation = abs(y1 - y2)
                        min_sep = min(c1['bbox'][3], c2['bbox'][3]) * 0.5  # At least half a lens height apart
                        max_sep = max(h * 0.8, c1['bbox'][3] + c2['bbox'][3] + 200)

                        if min_sep < separation < max_sep:
                            # Check they're roughly aligned horizontally
                            x_diff = abs(c1['center'][0] - c2['center'][0])
                            if x_diff < max(c1['bbox'][2], c2['bbox'][2]) * 1.5:
                                # Assign based on Y position (top = left in vertical orientation)
                                if y1 < y2:
                                    left_contour = c1['contour']
                                    right_contour = c2['contour']
                                else:
                                    left_contour = c2['contour']
                                    right_contour = c1['contour']
                                confidence = (c1['score'] + c2['score']) / 2 * 100
                                break
                    else:
                        # For horizontal orientation, check X separation
                        x1, x2 = c1['center'][0], c2['center'][0]
                        separation = abs(x1 - x2)
                        min_sep = min(c1['bbox'][2], c2['bbox'][2]) * 0.5
                        max_sep = max(w * 0.8, c1['bbox'][2] + c2['bbox'][2] + 200)

                        if min_sep < separation < max_sep:
                            # Check they're roughly aligned vertically
                            y_diff = abs(c1['center'][1] - c2['center'][1])
                            if y_diff < max(c1['bbox'][3], c2['bbox'][3]) * 1.5:
                                if x1 < x2:
                                    left_contour = c1['contour']
                                    right_contour = c2['contour']
                                else:
                                    left_contour = c2['contour']
                                    right_contour = c1['contour']
                                confidence = (c1['score'] + c2['score']) / 2 * 100
                                break

                if left_contour is not None:
                    break

        # If we couldn't find a pair, use the best single contour
        if left_contour is None and len(candidates) >= 1:
            best = candidates[0]
            # Determine position based on orientation
            if is_vertical:
                if best['center'][1] < image_center_y:
                    left_contour = best['contour']
                else:
                    right_contour = best['contour']
            else:
                if best['center'][0] < image_center_x:
                    left_contour = best['contour']
                else:
                    right_contour = best['contour']
            confidence = best['score'] * 60

        return left_contour, right_contour, max(min_confidence * 100, confidence)

    def _calculate_center(self, contour: np.ndarray) -> Tuple[float, float]:
        """Calculate the centroid of a contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = contour.mean(axis=0)[0]
        return (cx, cy)

    def smooth_contour(self, contour: np.ndarray, smoothing_factor: float = 0.01) -> np.ndarray:
        """
        Smooth a contour using spline interpolation.

        Args:
            contour: Input contour points
            smoothing_factor: Amount of smoothing (0 = no smoothing)

        Returns:
            Smoothed contour
        """
        if len(contour) < 10:
            return contour

        # Reshape contour
        points = contour.reshape(-1, 2)

        # Close the contour
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        # Fit spline
        try:
            tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing_factor * len(points), per=True)

            # Evaluate spline at more points
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)

            smoothed = np.column_stack([x_new, y_new]).reshape(-1, 1, 2).astype(np.int32)
            return smoothed

        except Exception:
            return contour

    def resample_contour(self, contour: np.ndarray, num_points: int = 1000) -> np.ndarray:
        """
        Resample contour to have exactly num_points points, evenly spaced.

        Args:
            contour: Input contour
            num_points: Desired number of points

        Returns:
            Resampled contour with num_points points
        """
        points = contour.reshape(-1, 2).astype(np.float64)

        # Calculate cumulative arc length
        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]

        # Generate evenly spaced arc lengths
        target_lengths = np.linspace(0, total_length, num_points, endpoint=False)

        # Interpolate points at target arc lengths
        resampled = np.zeros((num_points, 2))
        for i, target in enumerate(target_lengths):
            idx = np.searchsorted(cumulative, target)
            if idx == 0:
                resampled[i] = points[0]
            elif idx >= len(points):
                resampled[i] = points[-1]
            else:
                # Linear interpolation
                t = (target - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1] + 1e-10)
                resampled[i] = points[idx-1] + t * (points[idx] - points[idx-1])

        return resampled.reshape(-1, 1, 2).astype(np.int32)

    def draw_contours(self, image: np.ndarray,
                      show_centers: bool = True,
                      show_boxes: bool = True) -> np.ndarray:
        """Draw detected contours on the image."""
        result = image.copy()

        if self.left_contour is not None:
            cv2.drawContours(result, [self.left_contour], -1, (0, 0, 255), 2)
            if show_centers and self.left_center:
                cv2.circle(result, (int(self.left_center[0]), int(self.left_center[1])),
                          5, (0, 0, 255), -1)
            if show_boxes:
                x, y, w, h = cv2.boundingRect(self.left_contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if self.right_contour is not None:
            cv2.drawContours(result, [self.right_contour], -1, (255, 0, 0), 2)
            if show_centers and self.right_center:
                cv2.circle(result, (int(self.right_center[0]), int(self.right_center[1])),
                          5, (255, 0, 0), -1)
            if show_boxes:
                x, y, w, h = cv2.boundingRect(self.right_contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 1)

        return result
