"""
Grid Detection and Calibration Module
Detects 8mm grid paper and calculates pixels-per-mm ratio
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class GridDetector:
    """Detects reference square and provides calibration for measurements."""

    def __init__(self, grid_size_mm: float = 15.0):
        """
        Initialize grid detector.

        Args:
            grid_size_mm: Size of reference square in millimeters (default 15mm)
        """
        self.grid_size_mm = grid_size_mm
        self.pixels_per_mm: Optional[float] = None
        self.grid_lines_horizontal: List[np.ndarray] = []
        self.grid_lines_vertical: List[np.ndarray] = []
        self.confidence: float = 0.0
        # Reference square detection results
        self.reference_square: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

    def detect_grid(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Detect reference square and calculate pixels per mm.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (pixels_per_mm, confidence)
        """
        # Primary method: Detect the reference square with "A" inside
        result_ref = self._detect_reference_square(image)
        if result_ref[0] > 0 and result_ref[1] > 70:
            self.pixels_per_mm = result_ref[0]
            self.confidence = result_ref[1]
            return result_ref

        # Fallback to other methods if reference square not found
        results = []

        # Method 1: Standard Hough line detection
        result1 = self._detect_grid_hough(image)
        if result1[0] > 0:
            results.append(('hough', result1))

        # Method 2: Enhanced line detection with preprocessing
        result2 = self._detect_grid_enhanced(image)
        if result2[0] > 0:
            results.append(('enhanced', result2))

        # Method 3: Autocorrelation-based detection
        result3 = self._detect_grid_autocorrelation(image)
        if result3[0] > 0:
            results.append(('autocorr', result3))

        # Method 4: Morphological approach
        result4 = self._detect_grid_morphological(image)
        if result4[0] > 0:
            results.append(('morpho', result4))

        # Select best result based on confidence, but validate ppm is reasonable
        valid_results = []
        for method, (ppm, conf) in results:
            # Validate ppm is in reasonable range
            if 0.5 <= ppm <= 50:  # Reasonable range for most photos
                valid_results.append((method, (ppm, conf)))
            else:
                # Heavily penalize unreasonable values
                valid_results.append((method, (ppm, conf * 0.1)))

        if valid_results:
            best_method, (ppm, conf) = max(valid_results, key=lambda x: x[1][1])
            self.pixels_per_mm = ppm
            self.confidence = conf
            return ppm, conf

        # Fallback
        return self._detect_grid_fallback(image)

    def _detect_reference_square(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Detect the reference square (with 'A' inside) for calibration.
        The reference square is 15x15mm.

        Returns:
            Tuple of (pixels_per_mm, confidence)
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply multiple thresholding methods to find the square
        # Method 1: Adaptive threshold
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 5)

        # Method 2: Otsu threshold
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Method 3: Simple threshold for dark lines on white paper
        _, binary3 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Try each binary image
        square_candidates = []

        for binary in [binary1, binary2, binary3]:
            # Clean up the binary image
            kernel = np.ones((3, 3), np.uint8)
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours
            contours, hierarchy = cv2.findContours(binary_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Filter by area - reference square should be reasonable size
                min_area = (min(img_w, img_h) * 0.02) ** 2
                max_area = (min(img_w, img_h) * 0.20) ** 2

                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Check if it's roughly square (aspect ratio close to 1)
                aspect_ratio = w / h if h > 0 else 0
                if not (0.7 < aspect_ratio < 1.4):
                    continue

                # Check solidity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # For hand-drawn squares, solidity might be lower
                if solidity < 0.5:
                    continue

                # Check if there's content inside (the "A" letter)
                inner_margin = int(min(w, h) * 0.15)
                if inner_margin < 5:
                    continue

                inner_y1 = max(0, y + inner_margin)
                inner_y2 = min(img_h, y + h - inner_margin)
                inner_x1 = max(0, x + inner_margin)
                inner_x2 = min(img_w, x + w - inner_margin)

                if inner_y2 <= inner_y1 or inner_x2 <= inner_x1:
                    continue

                inner_roi = gray[inner_y1:inner_y2, inner_x1:inner_x2]

                if inner_roi.size == 0:
                    continue

                # Look for dark pixels inside (the letter "A")
                _, inner_binary = cv2.threshold(inner_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                dark_ratio = np.sum(inner_binary > 0) / inner_binary.size

                # There should be some dark content (the "A") but not too much
                # Relaxed constraints for hand-drawn content
                if 0.02 < dark_ratio < 0.6:
                    # Calculate score - prefer squares with content inside
                    squareness = 1.0 - abs(1.0 - aspect_ratio)
                    # Boost score for having content inside
                    content_score = 1.0 + (dark_ratio * 2) if 0.05 < dark_ratio < 0.4 else 1.0
                    score = squareness * solidity * content_score

                    square_candidates.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'has_content': dark_ratio,
                        'score': score
                    })

        if not square_candidates:
            return 0, 0

        # Remove duplicates (same bbox appearing from different threshold methods)
        unique_candidates = []
        seen_bboxes = set()
        for c in square_candidates:
            bbox_key = (c['bbox'][0] // 10, c['bbox'][1] // 10, c['bbox'][2] // 10, c['bbox'][3] // 10)
            if bbox_key not in seen_bboxes:
                seen_bboxes.add(bbox_key)
                unique_candidates.append(c)

        if not unique_candidates:
            return 0, 0

        # Sort by score and pick the best
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        best = unique_candidates[0]

        x, y, w, h = best['bbox']

        # Store the reference square location
        self.reference_square = (x, y, w, h)

        # Calculate pixels per mm
        # The reference square is 15x15mm, use average of width and height
        avg_size_px = (w + h) / 2
        pixels_per_mm = avg_size_px / self.grid_size_mm

        # Calculate confidence based on squareness and detection quality
        confidence = min(95, best['score'] * 80 + 20)

        return pixels_per_mm, confidence

    def _detect_grid_hough(self, image: np.ndarray) -> Tuple[float, float]:
        """Standard Hough line detection method."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            binary, rho=1, theta=np.pi/180,
            threshold=50, minLineLength=50, maxLineGap=10
        )

        if lines is None or len(lines) < 4:
            return 0, 0

        return self._process_lines(lines, image.shape)

    def _detect_grid_enhanced(self, image: np.ndarray) -> Tuple[float, float]:
        """Enhanced detection with better preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Use Canny edge detection
        edges = cv2.Canny(filtered, 30, 100)

        # Dilate to connect broken lines
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=30, minLineLength=30, maxLineGap=15
        )

        if lines is None or len(lines) < 4:
            return 0, 0

        return self._process_lines(lines, image.shape)

    def _detect_grid_autocorrelation(self, image: np.ndarray) -> Tuple[float, float]:
        """Detect grid using autocorrelation to find periodic spacing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # Sample multiple horizontal and vertical strips for robustness
        h_spacings = []
        v_spacings = []

        # Sample at multiple positions
        for offset in [-h//4, -h//8, 0, h//8, h//4]:
            y = h//2 + offset
            if 0 <= y < h:
                strip = gray[y, :]
                strip = (strip - strip.mean()) / (strip.std() + 1e-10)
                autocorr = np.correlate(strip, strip, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                spacing = self._find_autocorr_peak(autocorr, min_lag=15)
                if spacing > 15:
                    h_spacings.append(spacing)

        for offset in [-w//4, -w//8, 0, w//8, w//4]:
            x = w//2 + offset
            if 0 <= x < w:
                strip = gray[:, x]
                strip = (strip - strip.mean()) / (strip.std() + 1e-10)
                autocorr = np.correlate(strip, strip, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                spacing = self._find_autocorr_peak(autocorr, min_lag=15)
                if spacing > 15:
                    v_spacings.append(spacing)

        # Use median to be robust to outliers
        h_spacing = np.median(h_spacings) if h_spacings else 0
        v_spacing = np.median(v_spacings) if v_spacings else 0

        if h_spacing > 15 and v_spacing > 15:
            # If spacings are very different, one might be detecting every 2nd line
            # Use the smaller spacing as the true grid spacing
            if h_spacing > 0 and v_spacing > 0:
                ratio = max(h_spacing, v_spacing) / min(h_spacing, v_spacing)
                if ratio > 1.7:  # One is roughly double the other
                    # Use the smaller spacing
                    spacing = min(h_spacing, v_spacing)
                    # Check if the larger is roughly 2x (confirming grid detection)
                    if 1.7 <= ratio <= 2.3:
                        conf = 80.0  # Confirmed by ratio
                    else:
                        conf = 60.0
                    return spacing / self.grid_size_mm, conf
                else:
                    # Both spacings are similar, average them
                    avg_spacing = (h_spacing + v_spacing) / 2
                    diff = abs(h_spacing - v_spacing) / max(h_spacing, v_spacing)
                    conf = (1 - diff) * 90
                    return avg_spacing / self.grid_size_mm, conf
        elif h_spacing > 15:
            return h_spacing / self.grid_size_mm, 65.0
        elif v_spacing > 15:
            return v_spacing / self.grid_size_mm, 65.0

        return 0, 0

    def _detect_grid_morphological(self, image: np.ndarray) -> Tuple[float, float]:
        """Detect grid using morphological operations."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to isolate grid lines (blue/gray lines on white)
        # Grid lines are usually darker than paper
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Extract horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # Extract vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Find horizontal line positions
        h_proj = np.sum(h_lines, axis=1)
        h_positions = self._find_line_positions(h_proj)

        # Find vertical line positions
        v_proj = np.sum(v_lines, axis=0)
        v_positions = self._find_line_positions(v_proj)

        if len(h_positions) >= 2 and len(v_positions) >= 2:
            h_spacing = self._calculate_spacing_from_positions(h_positions)
            v_spacing = self._calculate_spacing_from_positions(v_positions)

            if h_spacing > 10 and v_spacing > 10:
                avg_spacing = (h_spacing + v_spacing) / 2
                ppm = avg_spacing / self.grid_size_mm
                diff = abs(h_spacing - v_spacing) / max(h_spacing, v_spacing)
                conf = (1 - diff) * 85
                return ppm, conf

        return 0, 0

    def _detect_grid_fallback(self, image: np.ndarray) -> Tuple[float, float]:
        """Fallback estimation based on typical photo properties."""
        h, w = image.shape[:2]

        # Assume typical setup: glasses span roughly 60-80% of image width
        # Typical glasses are 120-140mm total width
        # Estimate based on this assumption
        estimated_frame_width_mm = 130  # Average frame width
        estimated_frame_width_pixels = w * 0.7  # Assume frame is 70% of image width

        self.pixels_per_mm = estimated_frame_width_pixels / estimated_frame_width_mm
        self.confidence = 25.0  # Low confidence for fallback

        return self.pixels_per_mm, self.confidence

    def _process_lines(self, lines: np.ndarray, image_shape: Tuple) -> Tuple[float, float]:
        """Process detected lines to calculate grid spacing."""
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 20 or angle > 160:  # Horizontal
                horizontal_lines.append(line[0])
            elif 70 < angle < 110:  # Vertical
                vertical_lines.append(line[0])

        h_spacing = self._calculate_line_spacing(horizontal_lines, axis='y')
        v_spacing = self._calculate_line_spacing(vertical_lines, axis='x')

        self.grid_lines_horizontal = horizontal_lines
        self.grid_lines_vertical = vertical_lines

        if h_spacing > 0 and v_spacing > 0:
            avg_spacing = (h_spacing + v_spacing) / 2
            ppm = avg_spacing / self.grid_size_mm
            diff = abs(h_spacing - v_spacing) / max(h_spacing, v_spacing)
            conf = (1 - diff) * 100
            return ppm, conf
        elif h_spacing > 0:
            return h_spacing / self.grid_size_mm, 60.0
        elif v_spacing > 0:
            return v_spacing / self.grid_size_mm, 60.0

        return 0, 0

    def _find_autocorr_peak(self, autocorr: np.ndarray, min_lag: int = 15) -> int:
        """Find the first significant peak in autocorrelation after min_lag."""
        if len(autocorr) < min_lag * 2:
            return 0

        # Start searching after min_lag to skip the main peak at lag 0
        search_region = autocorr[min_lag:min(500, len(autocorr))]

        if len(search_region) < 10:
            return 0

        # Find peaks
        peaks = []
        for i in range(1, len(search_region) - 1):
            if search_region[i] > search_region[i-1] and search_region[i] > search_region[i+1]:
                if search_region[i] > np.mean(search_region):
                    peaks.append((i + min_lag, search_region[i]))

        if peaks:
            # Return the first significant peak
            return peaks[0][0]

        return 0

    def _find_line_positions(self, projection: np.ndarray, threshold_ratio: float = 0.3) -> List[int]:
        """Find line positions from a projection profile."""
        threshold = projection.max() * threshold_ratio
        positions = []

        in_line = False
        line_start = 0

        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                in_line = True
                line_start = i
            elif val <= threshold and in_line:
                in_line = False
                positions.append((line_start + i) // 2)

        return positions

    def _calculate_spacing_from_positions(self, positions: List[int]) -> float:
        """Calculate average spacing from line positions."""
        if len(positions) < 2:
            return 0

        diffs = np.diff(sorted(positions))
        if len(diffs) == 0:
            return 0

        # Use median to be robust to outliers
        median_diff = np.median(diffs)

        # Filter outliers
        valid_diffs = diffs[(diffs > median_diff * 0.5) & (diffs < median_diff * 1.5)]

        if len(valid_diffs) > 0:
            return np.mean(valid_diffs)

        return median_diff

    def _calculate_line_spacing(self, lines: List, axis: str = 'y') -> float:
        """Calculate average spacing between parallel lines."""
        if len(lines) < 2:
            return 0

        # Get the relevant coordinate for each line
        if axis == 'y':
            positions = sorted([(l[1] + l[3]) / 2 for l in lines])
        else:
            positions = sorted([(l[0] + l[2]) / 2 for l in lines])

        # Calculate differences between adjacent lines
        diffs = np.diff(positions)

        # Filter out outliers (non-grid lines)
        if len(diffs) > 0:
            median_diff = np.median(diffs)
            valid_diffs = diffs[(diffs > median_diff * 0.5) & (diffs < median_diff * 1.5)]

            if len(valid_diffs) > 0:
                return np.mean(valid_diffs)

        return 0

    def _find_first_peak(self, profile: np.ndarray, min_distance: int = 10) -> int:
        """Find the first significant peak in a 1D profile."""
        if len(profile) < min_distance * 2:
            return 0

        # Smooth the profile
        kernel_size = 5
        smoothed = np.convolve(profile, np.ones(kernel_size)/kernel_size, mode='valid')

        # Find peaks
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > np.mean(smoothed) * 2:  # Significant peak
                    peaks.append(i + kernel_size // 2)

        if peaks and peaks[0] > min_distance:
            return peaks[0]
        elif len(peaks) > 1:
            return peaks[1]

        return 0

    def pixels_to_mm(self, pixels: float) -> float:
        """Convert pixel measurement to millimeters."""
        if self.pixels_per_mm is None or self.pixels_per_mm == 0:
            raise ValueError("Grid not calibrated. Run detect_grid first.")
        return pixels / self.pixels_per_mm

    def mm_to_pixels(self, mm: float) -> float:
        """Convert millimeter measurement to pixels."""
        if self.pixels_per_mm is None or self.pixels_per_mm == 0:
            raise ValueError("Grid not calibrated. Run detect_grid first.")
        return mm * self.pixels_per_mm

    def draw_detected_grid(self, image: np.ndarray) -> np.ndarray:
        """Draw detected grid lines on the image for visualization."""
        result = image.copy()

        # Draw horizontal lines in blue
        for line in self.grid_lines_horizontal[:20]:  # Limit for clarity
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw vertical lines in green
        for line in self.grid_lines_vertical[:20]:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return result
