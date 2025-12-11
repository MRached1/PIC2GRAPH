# PIC2GRAPH

Convert photographs of eyeglasses on grid paper to Hoya GT5000 compatible tracer files (.DAT).

## Overview

This tool performs the **reverse** of what frame tracers do:
- **Tracer**: Physically scans frame → outputs .DAT file
- **PIC2GRAPH**: Photo of frame on grid paper → outputs .DAT file

## Requirements

- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python run.py
```

## Usage Instructions

### Taking the Photo

1. Place the eyeglass frame on **8mm grid paper** (white paper with blue/grey lines)
2. Position the camera **directly above** (bird's eye view)
3. Ensure both lenses are visible
4. Good lighting, minimal shadows
5. Frame can be at any angle on the paper

### Using the Application

1. **Load Image**: Click "Load Image" and select your photo
2. **Process**: Click "Process Image" to detect the grid and lens contours
3. **Review**: Check the preview - contours shown in red (left) and blue (right)
4. **Adjust** (if needed): Enable "Contour Editing" to manually adjust control points
5. **Generate**: Click "Generate DAT File" to create the tracer file

### Measurements Output

- **A (HBOX)**: Horizontal box dimension (lens width)
- **B (VBOX)**: Vertical box dimension (lens height)
- **DBL**: Distance Between Lenses (bridge width)
- **CIRC**: Circumference of lens shape

## Output Format

Generates `.DAT` files compatible with:
- Hoya GT5000 tracer format
- OMA/VCA standard
- 1000 points per lens (equidistant angular spacing)

## Project Structure

```
PIC2GRAPH/
├── run.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── gui.py            # Tkinter GUI application
│   ├── processor.py      # Main processing pipeline
│   ├── grid_detector.py  # Grid detection & calibration
│   ├── contour_detector.py # Lens contour detection
│   ├── measurements.py   # A, B, DBL calculations
│   ├── polar_converter.py # Cartesian to polar conversion
│   └── dat_generator.py  # .DAT file generation
└── DOC/
    └── HOYA_GT5000_INTEGRATION.md  # Format specification
```

## Configuration

Default grid size is 8mm. Can be changed in the GUI under "Detection Parameters".

## Limitations

- Automatic detection works best with full-rim frames
- Semi-rimless/rimless frames may require manual adjustment
- Transparent frames are challenging - use manual editing
- Photo quality affects accuracy

## Accuracy

Target accuracy: 0.5mm
Output precision: 1/100mm (centimicrons)

## Integration

The `ImageProcessor` class can be used as a standalone module:

```python
from src.processor import ImageProcessor

processor = ImageProcessor(grid_size_mm=8.0)
result = processor.process_image("path/to/photo.jpg")

if result.success:
    dat_path = processor.generate_dat_file(result)
    print(f"Generated: {dat_path}")
    print(f"A: {result.a_mm}mm, B: {result.b_mm}mm")
```
