# HOYA GT5000 Tracer File Integration Guide

## Overview

This document describes how to parse and visualize eyeglass frame shape data from HOYA GT5000 lens tracers. The tracer outputs files in **OMA/VCA (Optical Manufacturers Association / Vision Council of America)** format.

---

## File Format Specification

### File Extension
- `.DAT` (plain text, despite binary-looking extension)

### Encoding
- ASCII text with CRLF (`\r\n`) line endings

---

## Data Structure

### Header Parameters

| Parameter | Description | Example | Unit |
|-----------|-------------|---------|------|
| `ANS` | Answer/Response code | `9901` | - |
| `JOB` | Job identifier | `2025-12-11_12-00-39` | - |
| `STATUS` | Tracing status (0=success) | `0` | - |
| `DBL` | Distance Between Lenses (bridge width) | `20` | mm |
| `HBOX` | Horizontal box dimension (width) | `52.32;52.32` | mm |
| `VBOX` | Vertical box dimension (height) | `47.27;47.27` | mm |
| `CIRC` | Circumference of lens | `167.40;167.40` | mm |
| `CIRC3D` | 3D circumference | `167.40;167.40` | mm |
| `ZTILT` | Z-axis tilt angle | `2.74` | degrees |
| `FCRV` | Front curve | `3.85` | diopters |
| `DO` | Dominant eye indicator | `L` | L/R |
| `ETYP` | Edge type | `1;1` | - |

### TRCFMT (Trace Format) Parameter

Format: `TRCFMT=<version>;<num_points>;<spacing>;<side>;<format>`

| Field | Value | Description |
|-------|-------|-------------|
| Version | `1` | Format version |
| Num Points | `1000` | Number of radial measurements |
| Spacing | `E` | Equidistant angular spacing |
| Side | `R` or `L` | Right or Left lens |
| Format | `F` | Format type (Full) |

### Radial Data (R= lines)

- Each `R=` line contains 10 radius values separated by semicolons
- Values are in **1/100 mm (centimicrons)**
- 100 lines × 10 values = 1000 points per lens
- Points are measured at equidistant angles from 0° to 360°
- Angular step = 360° / 1000 = 0.36° per point

**Example:**
```
R=2533;2536;2539;2542;2545;2548;2551;2555;2558;2562
```
This represents radii of 25.33mm, 25.36mm, 25.39mm, etc.

---

## Coordinate System

- **Origin**: Geometric center of the lens (Boxing Center)
- **0°**: Nasal side (toward nose)
- **90°**: Superior (top)
- **180°**: Temporal side (toward temple)
- **270°**: Inferior (bottom)
- **Rotation**: Counter-clockwise

---

## Parsing Algorithm

### Python Implementation

```python
def parse_oma_file(filepath: str) -> dict:
    """
    Parse HOYA GT5000 OMA/VCA tracer file.

    Args:
        filepath: Path to the .DAT file

    Returns:
        Dictionary containing metadata and radial data for both lenses
    """
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    result = {
        'metadata': {},
        'right_radii': [],  # in 1/100 mm
        'left_radii': [],   # in 1/100 mm
    }

    current_side = None

    for line in lines:
        line = line.strip()

        # Detect which lens we're parsing
        if line.startswith('TRCFMT='):
            parts = line.split(';')
            if 'R' in parts:
                current_side = 'R'
            elif 'L' in parts:
                current_side = 'L'

        # Parse radial data
        elif line.startswith('R='):
            values = [int(v) for v in line[2:].split(';')]
            if current_side == 'R':
                result['right_radii'].extend(values)
            elif current_side == 'L':
                result['left_radii'].extend(values)

        # Parse metadata
        elif '=' in line and not line.startswith('R='):
            key, _, value = line.partition('=')
            result['metadata'][key] = value

    return result
```

### Converting to Cartesian Coordinates

```python
import numpy as np

def polar_to_cartesian(radii_centimicrons: list) -> tuple:
    """
    Convert polar radii to Cartesian coordinates.

    Args:
        radii_centimicrons: List of radii in 1/100 mm

    Returns:
        Tuple of (x_coords, y_coords) in mm
    """
    # Convert to mm
    radii_mm = np.array(radii_centimicrons) / 100.0

    # Generate angles (0 to 2π, equidistant)
    num_points = len(radii_mm)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Convert to Cartesian
    x = radii_mm * np.cos(angles)
    y = radii_mm * np.sin(angles)

    return x, y
```

### Extracting Box Dimensions

```python
def get_box_dimensions(metadata: dict) -> dict:
    """
    Extract lens dimensions from metadata.

    Returns:
        Dictionary with parsed dimensions in mm
    """
    def parse_pair(value: str) -> tuple:
        parts = value.split(';')
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else float(parts[0])

    return {
        'dbl': float(metadata.get('DBL', 0)),
        'hbox_right': parse_pair(metadata.get('HBOX', '0'))[0],
        'hbox_left': parse_pair(metadata.get('HBOX', '0'))[1],
        'vbox_right': parse_pair(metadata.get('VBOX', '0'))[0],
        'vbox_left': parse_pair(metadata.get('VBOX', '0'))[1],
        'circ_right': parse_pair(metadata.get('CIRC', '0'))[0],
        'circ_left': parse_pair(metadata.get('CIRC', '0'))[1],
    }
```

---

## Visualization

### Basic Plot with Matplotlib

```python
import matplotlib.pyplot as plt

def plot_frame(right_x, right_y, left_x, left_y, dbl_mm):
    """
    Plot complete eyeglass frame as worn.

    Args:
        right_x, right_y: Right lens coordinates in mm
        left_x, left_y: Left lens coordinates in mm
        dbl_mm: Distance between lenses in mm
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate offset for frame positioning
    half_dbl = dbl_mm / 2
    right_offset = half_dbl + np.max(right_x)
    left_offset = half_dbl + np.max(left_x)

    # Plot right lens (shifted right)
    ax.plot(right_x + right_offset, right_y, 'b-', linewidth=2)
    ax.fill(right_x + right_offset, right_y, alpha=0.3, color='blue')

    # Plot left lens (mirrored and shifted left)
    ax.plot(-left_x - left_offset, left_y, 'b-', linewidth=2)
    ax.fill(-left_x - left_offset, left_y, alpha=0.3, color='blue')

    ax.set_aspect('equal')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.grid(True, linestyle='--', alpha=0.5)

    return fig, ax
```

---

## Complete Integration Example

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class HoyaTracerParser:
    """Parser for HOYA GT5000 tracer files."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.metadata = {}
        self.right_radii = []
        self.left_radii = []
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            content = f.read()

        current_side = None

        for line in content.strip().split('\n'):
            line = line.strip()

            if line.startswith('TRCFMT='):
                parts = line.split(';')
                current_side = 'R' if 'R' in parts else 'L' if 'L' in parts else None
            elif line.startswith('R='):
                values = [int(v) for v in line[2:].split(';')]
                if current_side == 'R':
                    self.right_radii.extend(values)
                elif current_side == 'L':
                    self.left_radii.extend(values)
            elif '=' in line:
                key, _, value = line.partition('=')
                self.metadata[key] = value

    @property
    def dbl(self) -> float:
        """Distance between lenses in mm."""
        return float(self.metadata.get('DBL', 0))

    @property
    def hbox(self) -> tuple:
        """Horizontal box (width) for right and left lens in mm."""
        val = self.metadata.get('HBOX', '0;0')
        parts = val.split(';')
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else float(parts[0])

    @property
    def vbox(self) -> tuple:
        """Vertical box (height) for right and left lens in mm."""
        val = self.metadata.get('VBOX', '0;0')
        parts = val.split(';')
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else float(parts[0])

    @property
    def circumference(self) -> tuple:
        """Circumference for right and left lens in mm."""
        val = self.metadata.get('CIRC', '0;0')
        parts = val.split(';')
        return float(parts[0]), float(parts[1]) if len(parts) > 1 else float(parts[0])

    def get_cartesian(self, side: str = 'R') -> tuple:
        """
        Get Cartesian coordinates for a lens.

        Args:
            side: 'R' for right, 'L' for left

        Returns:
            Tuple of (x, y) numpy arrays in mm
        """
        radii = self.right_radii if side == 'R' else self.left_radii
        radii_mm = np.array(radii) / 100.0
        angles = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)

        return radii_mm * np.cos(angles), radii_mm * np.sin(angles)

    def plot(self, save_path: str = None):
        """Generate and optionally save frame visualization."""
        rx, ry = self.get_cartesian('R')
        lx, ly = self.get_cartesian('L')

        fig, ax = plt.subplots(figsize=(12, 6))

        half_dbl = self.dbl / 2
        right_offset = half_dbl + np.max(rx)
        left_offset = half_dbl + np.max(lx)

        ax.plot(rx + right_offset, ry, 'b-', lw=2, label='Right')
        ax.fill(rx + right_offset, ry, alpha=0.3, color='blue')
        ax.plot(-lx - left_offset, ly, 'r-', lw=2, label='Left')
        ax.fill(-lx - left_offset, ly, alpha=0.3, color='red')

        ax.set_aspect('equal')
        ax.set_title(f'Frame: {self.metadata.get("JOB", "Unknown")}')
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def to_dict(self) -> dict:
        """Export all data as dictionary."""
        return {
            'metadata': self.metadata,
            'dimensions': {
                'dbl': self.dbl,
                'hbox': self.hbox,
                'vbox': self.vbox,
                'circumference': self.circumference,
            },
            'right_lens': {
                'radii_centimicrons': self.right_radii,
                'num_points': len(self.right_radii),
            },
            'left_lens': {
                'radii_centimicrons': self.left_radii,
                'num_points': len(self.left_radii),
            },
        }


# Usage example
if __name__ == '__main__':
    parser = HoyaTracerParser('path/to/file.DAT')

    # Access properties
    print(f"DBL: {parser.dbl} mm")
    print(f"HBOX: {parser.hbox} mm")
    print(f"VBOX: {parser.vbox} mm")

    # Get coordinates
    x, y = parser.get_cartesian('R')

    # Plot and save
    parser.plot('output.png')

    # Export as dict (for JSON serialization, etc.)
    data = parser.to_dict()
```

---

## Data Export Formats

### JSON Export

```python
import json

def export_to_json(parser: HoyaTracerParser, output_path: str):
    """Export parsed data to JSON."""
    data = parser.to_dict()
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
```

### SVG Export

```python
def export_to_svg(parser: HoyaTracerParser, output_path: str):
    """Export frame shape to SVG."""
    rx, ry = parser.get_cartesian('R')
    lx, ly = parser.get_cartesian('L')

    # Create SVG path
    def coords_to_path(x, y):
        points = [f"{xi:.2f},{yi:.2f}" for xi, yi in zip(x, y)]
        return f"M {points[0]} L " + " L ".join(points[1:]) + " Z"

    half_dbl = parser.dbl / 2
    right_offset = half_dbl + max(rx)
    left_offset = half_dbl + max(lx)

    # Offset coordinates
    rx_offset = rx + right_offset
    lx_offset = -lx - left_offset

    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-80 -30 160 60">
  <path d="{coords_to_path(rx_offset, ry)}" fill="none" stroke="blue" stroke-width="0.5"/>
  <path d="{coords_to_path(lx_offset, ly)}" fill="none" stroke="red" stroke-width="0.5"/>
</svg>'''

    with open(output_path, 'w') as f:
        f.write(svg_content)
```

---

## Error Handling

```python
class TracerParseError(Exception):
    """Exception raised for tracer file parsing errors."""
    pass

def validate_tracer_file(filepath: str) -> bool:
    """
    Validate that file is a valid HOYA tracer file.

    Returns:
        True if valid, raises TracerParseError otherwise
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Check for required fields
    required_fields = ['TRCFMT', 'DBL', 'HBOX', 'VBOX']
    for field in required_fields:
        if f'{field}=' not in content:
            raise TracerParseError(f"Missing required field: {field}")

    # Check for radial data
    if 'R=' not in content:
        raise TracerParseError("No radial data found")

    return True
```

---

## Dependencies

```
numpy>=1.20.0
matplotlib>=3.4.0
```

Install with:
```bash
pip install numpy matplotlib
```

---

## References

- OMA Communication Standard for Frame Tracers
- VCA (Vision Council of America) Data Format Specification
- HOYA GT5000 Technical Documentation

---

## Notes

- The file format is compatible with most lens edging systems
- Both lenses are stored in the same file with separate `TRCFMT` headers
- The coordinate system follows optical industry conventions
- Radius values with higher precision may be available in newer tracer models
