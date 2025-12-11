"""
DAT File Generator Module
Generates Hoya GT5000 compatible .DAT tracer files
"""

import os
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class DATFileData:
    """Data structure for DAT file generation."""
    job_id: str
    dbl: float  # Distance between lenses in mm
    hbox_right: float  # A measurement right lens
    hbox_left: float  # A measurement left lens
    vbox_right: float  # B measurement right lens
    vbox_left: float  # B measurement left lens
    circ_right: float  # Circumference right lens
    circ_left: float  # Circumference left lens
    right_radii: List[int]  # 1000 radii in centimicrons
    left_radii: List[int]  # 1000 radii in centimicrons


class DATGenerator:
    """Generates Hoya GT5000 compatible .DAT files."""

    def __init__(self):
        self.version = "1"
        self.num_points = 1000
        self.spacing = "E"  # Equidistant
        self.format_type = "F"  # Full format

    def generate(self, data: DATFileData, output_path: str) -> str:
        """
        Generate a .DAT file in Hoya GT5000 OMA/VCA format.

        Args:
            data: DATFileData containing all measurements and radii
            output_path: Path to write the .DAT file

        Returns:
            Path to the generated file
        """
        lines = []

        # Header
        lines.append(f"ANS=9901")
        lines.append(f"JOB={data.job_id}")
        lines.append(f"STATUS=0")

        # Frame measurements
        lines.append(f"DBL={data.dbl:.0f}")
        lines.append(f"HBOX={data.hbox_right:.2f};{data.hbox_left:.2f}")
        lines.append(f"VBOX={data.vbox_right:.2f};{data.vbox_left:.2f}")
        lines.append(f"CIRC={data.circ_right:.2f};{data.circ_left:.2f}")
        lines.append(f"CIRC3D={data.circ_right:.2f};{data.circ_left:.2f}")

        # Default values for 3D parameters (cannot be determined from 2D photo)
        lines.append(f"ZTILT=0.00")
        lines.append(f"FCRV=0.00")
        lines.append(f"DO=R")  # Default dominant eye
        lines.append(f"ETYP=1;1")  # Default edge type

        # Right lens trace data
        lines.append(f"TRCFMT={self.version};{self.num_points};{self.spacing};R;{self.format_type}")
        lines.extend(self._format_radii(data.right_radii))

        # Left lens trace data
        lines.append(f"TRCFMT={self.version};{self.num_points};{self.spacing};L;{self.format_type}")
        lines.extend(self._format_radii(data.left_radii))

        # Write file with CRLF line endings (as per OMA spec)
        content = "\r\n".join(lines) + "\r\n"

        with open(output_path, 'w', newline='') as f:
            f.write(content)

        return output_path

    def _format_radii(self, radii: List[int]) -> List[str]:
        """
        Format radii into R= lines with 10 values each.

        Args:
            radii: List of radii in centimicrons

        Returns:
            List of formatted R= lines
        """
        lines = []

        for i in range(0, len(radii), 10):
            chunk = radii[i:i+10]
            values = ";".join(str(r) for r in chunk)
            lines.append(f"R={values}")

        return lines

    def generate_filename(self, prefix: str = "PIC2GRAPH") -> str:
        """
        Generate a filename with timestamp.

        Args:
            prefix: Filename prefix

        Returns:
            Filename like "PIC2GRAPH_2025-12-11_14-30-45.DAT"
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{prefix}_{timestamp}.DAT"

    def generate_job_id(self, prefix: str = "PIC2GRAPH") -> str:
        """
        Generate a job ID with timestamp.

        Args:
            prefix: Job ID prefix

        Returns:
            Job ID like "PIC2GRAPH_2025-12-11_14-30-45"
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{prefix}_{timestamp}"

    def validate_data(self, data: DATFileData) -> tuple:
        """
        Validate data before generating file.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check radii count
        if len(data.right_radii) != self.num_points:
            errors.append(f"Right lens needs {self.num_points} radii, got {len(data.right_radii)}")

        if len(data.left_radii) != self.num_points:
            errors.append(f"Left lens needs {self.num_points} radii, got {len(data.left_radii)}")

        # Check measurements are positive
        if data.dbl < 0:
            errors.append(f"DBL must be positive, got {data.dbl}")

        if data.hbox_right <= 0 or data.hbox_left <= 0:
            errors.append(f"HBOX must be positive")

        if data.vbox_right <= 0 or data.vbox_left <= 0:
            errors.append(f"VBOX must be positive")

        # Check reasonable ranges (typical eyeglass dimensions)
        if data.dbl > 30:
            errors.append(f"DBL unusually large: {data.dbl}mm (typical: 10-25mm)")

        if data.hbox_right > 70 or data.hbox_left > 70:
            errors.append(f"HBOX unusually large (typical: 40-65mm)")

        if data.vbox_right > 60 or data.vbox_left > 60:
            errors.append(f"VBOX unusually large (typical: 25-50mm)")

        # Check radii values
        for i, r in enumerate(data.right_radii):
            if r <= 0:
                errors.append(f"Right lens radius at index {i} is non-positive: {r}")
                break
            if r > 5000:  # 50mm max
                errors.append(f"Right lens radius at index {i} too large: {r/100}mm")
                break

        for i, r in enumerate(data.left_radii):
            if r <= 0:
                errors.append(f"Left lens radius at index {i} is non-positive: {r}")
                break
            if r > 5000:  # 50mm max
                errors.append(f"Left lens radius at index {i} too large: {r/100}mm")
                break

        return len(errors) == 0, errors

    def parse_dat_file(self, filepath: str) -> Dict:
        """
        Parse an existing .DAT file (for verification/comparison).

        Args:
            filepath: Path to .DAT file

        Returns:
            Dictionary with parsed data
        """
        with open(filepath, 'r') as f:
            content = f.read()

        result = {
            'metadata': {},
            'right_radii': [],
            'left_radii': [],
        }

        current_side = None

        for line in content.strip().split('\n'):
            line = line.strip()

            if line.startswith('TRCFMT='):
                parts = line.split(';')
                if 'R' in parts:
                    current_side = 'R'
                elif 'L' in parts:
                    current_side = 'L'

            elif line.startswith('R='):
                values = [int(v) for v in line[2:].split(';')]
                if current_side == 'R':
                    result['right_radii'].extend(values)
                elif current_side == 'L':
                    result['left_radii'].extend(values)

            elif '=' in line and not line.startswith('R='):
                key, _, value = line.partition('=')
                result['metadata'][key] = value

        return result


def create_dat_file(job_id: str,
                    dbl: float,
                    a_mm: float,
                    b_mm: float,
                    circ_mm: float,
                    right_radii: List[int],
                    left_radii: List[int],
                    output_dir: str) -> str:
    """
    Convenience function to create a .DAT file.

    Args:
        job_id: Job identifier
        dbl: Distance between lenses in mm
        a_mm: Horizontal box dimension (same for both lenses)
        b_mm: Vertical box dimension (same for both lenses)
        circ_mm: Circumference (same for both lenses)
        right_radii: 1000 radii for right lens in centimicrons
        left_radii: 1000 radii for left lens in centimicrons
        output_dir: Directory to save the file

    Returns:
        Path to generated file
    """
    generator = DATGenerator()

    data = DATFileData(
        job_id=job_id,
        dbl=dbl,
        hbox_right=a_mm,
        hbox_left=a_mm,
        vbox_right=b_mm,
        vbox_left=b_mm,
        circ_right=circ_mm,
        circ_left=circ_mm,
        right_radii=right_radii,
        left_radii=left_radii
    )

    # Validate
    is_valid, errors = generator.validate_data(data)
    if not is_valid:
        raise ValueError(f"Invalid data: {'; '.join(errors)}")

    # Generate filename and path
    filename = generator.generate_filename()
    output_path = os.path.join(output_dir, filename)

    return generator.generate(data, output_path)
