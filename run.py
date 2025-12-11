#!/usr/bin/env python3
"""
PIC2GRAPH - Image to Hoya GT5000 Tracer File Converter
Main entry point for the application.

This program converts photographs of eyeglasses on 8mm grid paper
to Hoya GT5000 compatible .DAT tracer files.

Usage:
    python run.py

Requirements:
    - Python 3.8+
    - opencv-python
    - numpy
    - Pillow
    - scipy
"""

import sys
import os

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

from gui import main

if __name__ == "__main__":
    main()
