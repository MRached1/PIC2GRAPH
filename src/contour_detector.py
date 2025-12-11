"""
Glasses Contour Detection Module
Detects the inner edge of eyeglass frames (lens opening)

Strategy: Detect the dark frame material, then trace its inner edges
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
        """
        methods = [
            ("frame_edge_tracing", self._detect_by_frame_edges),
            ("dark_region_inner_boundary", self._detect_dark_region_inner_boundary),
            ("color_based_frame", self._detect_by_color_segmentation),
            ("morphological_frame", self._detect_by_morphological),
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
                print(f"Method {method_name} failed: {e}")
                continue

        self.left_contour = best_left
        self.right_contour = best_right
        self.confidence = best_confidence
        self.detection_method = best_method

        if self.left_contour is not None:
            self.left_center = self._calculate_center(self.left_contour)
        if self.right_contour is not None:
            self.right_center = self._calculate_center(self.right_contour)

        return self.left_contour, self.right_contour, self.confidence

    def _detect_by_frame_edges(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        PRIMARY METHOD: Detect the dark frame, then find its inner boundary.
        The lens opening is bounded by the inner edge of the frame material.
        """
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect the dark frame material using multiple methods
        # The frame is significantly darker than the white grid paper

        # Method A: Simple threshold for dark regions
        # The frame is dark (low values), paper is white (high values)
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Method B: Adaptive threshold
        adaptive_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )

        # Method C: Otsu
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine - use the intersection (all agree it's dark)
        frame_mask = cv2.bitwise_and(dark_mask, otsu_mask)

        # Step 2: Clean up the frame mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        # Remove noise (small specks)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)

        # Fill small gaps in the frame
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)

        # Step 3: Find the frame contours
        contours, hierarchy = cv2.findContours(frame_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None or len(contours) < 1:
            return None, None, 0.0

        hierarchy = hierarchy[0]

        # Step 4: Find inner contours (holes in the frame = lens openings)
        # In RETR_CCOMP, hierarchy[i][3] is the parent, hierarchy[i][2] is first child
        # Inner contours have a parent (hierarchy[i][3] != -1)

        lens_candidates = []

        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):
            parent_idx = hier[3]

            # We want contours that are HOLES (have a parent)
            if parent_idx == -1:
                continue

            area = cv2.contourArea(contour)

            # Lens openings should be substantial
            if area < img_area * 0.01:  # At least 1% of image
                continue
            if area > img_area * 0.3:  # Not more than 30%
                continue

            # Check the shape
            if len(contour) < 5:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

            # Lens openings are roughly oval, not too elongated
            if aspect > 2.0:
                continue

            # Check solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)

            # Lens openings should be fairly convex
            if solidity < 0.75:
                continue

            # Check that parent is large enough (the frame)
            parent_contour = contours[parent_idx]
            parent_area = cv2.contourArea(parent_contour)
            if parent_area < area * 1.5:  # Parent should be bigger
                continue

            # Score
            optimal_area = img_area * 0.06
            size_score = 1.0 / (1.0 + abs(area - optimal_area) / optimal_area)

            lens_candidates.append({
                'contour': contour,
                'area': area,
                'center': self._calculate_center(contour),
                'bbox': (x, y, bw, bh),
                'solidity': solidity,
                'score': size_score * solidity * 1.2  # Boost this method
            })

        if len(lens_candidates) >= 1:
            return self._select_best_lens_pair(lens_candidates, image.shape)

        return None, None, 0.0

    def _detect_dark_region_inner_boundary(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Find dark regions (frame) and extract their inner boundaries.
        Uses distance transform to find the "skeleton" of the frame and its inner edge.
        """
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect dark frame
        _, frame_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours of the frame
        contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) < 1:
            return None, None, 0.0

        # Find the largest dark region (should be the glasses frame)
        frame_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > img_area * 0.05:  # Significant size
                frame_contours.append((contour, area))

        if len(frame_contours) < 1:
            return None, None, 0.0

        # Sort by area
        frame_contours.sort(key=lambda x: x[1], reverse=True)

        # Take the largest frame region
        main_frame = frame_contours[0][0]

        # Create a filled mask of the frame
        frame_filled = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(frame_filled, [main_frame], -1, 255, -1)

        # Now find the "holes" - areas inside the frame bounding region but not part of frame
        x, y, w, h = cv2.boundingRect(main_frame)

        # The lens openings are white regions inside the frame's bounding box
        # that are surrounded by dark frame material

        # Create ROI
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        roi_gray = gray[y1:y2, x1:x2]
        roi_frame = frame_mask[y1:y2, x1:x2]

        # Find bright regions (not frame) within the ROI
        _, bright_mask = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)

        # These bright regions that are enclosed by frame are lens openings
        # Use the frame as a boundary

        # Invert frame mask to get non-frame regions
        non_frame = cv2.bitwise_not(roi_frame)

        # Find contours of non-frame regions
        contours, _ = cv2.findContours(non_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        lens_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < img_area * 0.01:
                continue
            if area > img_area * 0.25:
                continue

            if len(contour) < 5:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 2.0:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.7:
                continue

            # Check if this region is NOT touching the ROI boundary (i.e., it's enclosed)
            if bx <= 2 or by <= 2 or bx + bw >= (x2-x1) - 2 or by + bh >= (y2-y1) - 2:
                continue  # Touches edge, not a lens opening

            # Offset contour back to original image coordinates
            contour_offset = contour.copy()
            contour_offset[:, :, 0] += x1
            contour_offset[:, :, 1] += y1

            center = self._calculate_center(contour_offset)

            optimal_area = img_area * 0.05
            size_score = 1.0 / (1.0 + abs(area - optimal_area) / optimal_area)

            lens_candidates.append({
                'contour': contour_offset,
                'area': area,
                'center': center,
                'bbox': (bx + x1, by + y1, bw, bh),
                'solidity': solidity,
                'score': size_score * solidity
            })

        if len(lens_candidates) >= 1:
            return self._select_best_lens_pair(lens_candidates, image.shape)

        return None, None, 0.0

    def _detect_by_color_segmentation(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Use color-based segmentation to find the frame and then its inner edges.
        """
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # The frame is dark - low V in HSV, low L in LAB
        v_channel = hsv[:, :, 2]
        l_channel = lab[:, :, 0]

        # Threshold both
        _, v_mask = cv2.threshold(v_channel, 100, 255, cv2.THRESH_BINARY_INV)
        _, l_mask = cv2.threshold(l_channel, 100, 255, cv2.THRESH_BINARY_INV)

        # Combine
        frame_mask = cv2.bitwise_or(v_mask, l_mask)

        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(frame_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None or len(contours) < 1:
            return None, None, 0.0

        hierarchy = hierarchy[0]

        lens_candidates = []

        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):
            parent_idx = hier[3]

            if parent_idx == -1:
                continue

            area = cv2.contourArea(contour)
            if area < img_area * 0.01 or area > img_area * 0.3:
                continue

            if len(contour) < 5:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 2.0:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.7:
                continue

            optimal_area = img_area * 0.06
            size_score = 1.0 / (1.0 + abs(area - optimal_area) / optimal_area)

            lens_candidates.append({
                'contour': contour,
                'area': area,
                'center': self._calculate_center(contour),
                'bbox': (x, y, bw, bh),
                'solidity': solidity,
                'score': size_score * solidity * 0.9
            })

        if len(lens_candidates) >= 1:
            return self._select_best_lens_pair(lens_candidates, image.shape)

        return None, None, 0.0

    def _detect_by_morphological(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Use morphological operations to isolate the frame and find lens openings.
        """
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Strong blur to remove grid lines
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Heavy morphological operations to get just the frame shape
        kernel_large = np.ones((15, 15), np.uint8)
        frame_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, kernel_large, iterations=2)

        # Find contours
        contours, hierarchy = cv2.findContours(frame_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None or len(contours) < 1:
            return None, None, 0.0

        hierarchy = hierarchy[0]

        lens_candidates = []

        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):
            parent_idx = hier[3]

            if parent_idx == -1:
                continue

            area = cv2.contourArea(contour)
            if area < img_area * 0.01 or area > img_area * 0.3:
                continue

            if len(contour) < 5:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 2.0:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < 0.7:
                continue

            optimal_area = img_area * 0.06
            size_score = 1.0 / (1.0 + abs(area - optimal_area) / optimal_area)

            lens_candidates.append({
                'contour': contour,
                'area': area,
                'center': self._calculate_center(contour),
                'bbox': (x, y, bw, bh),
                'solidity': solidity,
                'score': size_score * solidity * 0.85
            })

        if len(lens_candidates) >= 1:
            return self._select_best_lens_pair(lens_candidates, image.shape)

        return None, None, 0.0

    def _select_best_lens_pair(self, candidates: List[dict], image_shape: Tuple) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Select the best pair of lens contours."""
        if len(candidates) < 1:
            return None, None, 0.0

        img_h, img_w = image_shape[:2]
        image_center_x = img_w / 2
        image_center_y = img_h / 2

        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Determine orientation
        if len(candidates) >= 2:
            c1, c2 = candidates[0], candidates[1]
            x_diff = abs(c1['center'][0] - c2['center'][0])
            y_diff = abs(c1['center'][1] - c2['center'][1])
            is_vertical = y_diff > x_diff
        else:
            is_vertical = img_h > img_w

        left_contour = None
        right_contour = None
        confidence = 0.0

        if len(candidates) >= 2:
            best_pair_score = 0

            for i, c1 in enumerate(candidates[:10]):
                for c2 in candidates[i+1:15]:
                    pair_score = c1['score'] + c2['score']

                    if is_vertical:
                        separation = abs(c1['center'][1] - c2['center'][1])
                        alignment = abs(c1['center'][0] - c2['center'][0])
                        size_ref = max(c1['bbox'][3], c2['bbox'][3])
                    else:
                        separation = abs(c1['center'][0] - c2['center'][0])
                        alignment = abs(c1['center'][1] - c2['center'][1])
                        size_ref = max(c1['bbox'][2], c2['bbox'][2])

                    if separation < size_ref * 0.3:
                        continue
                    if separation > (img_w if not is_vertical else img_h) * 0.9:
                        continue
                    if alignment > size_ref * 2:
                        continue

                    size_ratio = min(c1['area'], c2['area']) / (max(c1['area'], c2['area']) + 1e-6)
                    if size_ratio < 0.4:
                        continue

                    pair_score *= (1 + size_ratio)
                    pair_score *= (1 + min(1, separation / size_ref))

                    if pair_score > best_pair_score:
                        best_pair_score = pair_score

                        if is_vertical:
                            if c1['center'][1] < c2['center'][1]:
                                left_contour = c1['contour']
                                right_contour = c2['contour']
                            else:
                                left_contour = c2['contour']
                                right_contour = c1['contour']
                        else:
                            if c1['center'][0] < c2['center'][0]:
                                left_contour = c1['contour']
                                right_contour = c2['contour']
                            else:
                                left_contour = c2['contour']
                                right_contour = c1['contour']

                        confidence = min(95, (c1['score'] + c2['score']) / 2 * 100 * 1.2)

        if left_contour is None and len(candidates) >= 1:
            best = candidates[0]
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
            confidence = best['score'] * 50

        return left_contour, right_contour, confidence

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
        """Smooth a contour using spline interpolation."""
        if len(contour) < 10:
            return contour

        points = contour.reshape(-1, 2)

        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        try:
            tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing_factor * len(points), per=True)
            u_new = np.linspace(0, 1, 500)
            x_new, y_new = splev(u_new, tck)
            smoothed = np.column_stack([x_new, y_new]).reshape(-1, 1, 2).astype(np.int32)
            return smoothed
        except Exception:
            return contour

    def resample_contour(self, contour: np.ndarray, num_points: int = 1000) -> np.ndarray:
        """Resample contour to have exactly num_points points, evenly spaced."""
        points = contour.reshape(-1, 2).astype(np.float64)

        diffs = np.diff(points, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative[-1]

        target_lengths = np.linspace(0, total_length, num_points, endpoint=False)

        resampled = np.zeros((num_points, 2))
        for i, target in enumerate(target_lengths):
            idx = np.searchsorted(cumulative, target)
            if idx == 0:
                resampled[i] = points[0]
            elif idx >= len(points):
                resampled[i] = points[-1]
            else:
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
