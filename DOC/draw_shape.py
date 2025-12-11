import matplotlib.pyplot as plt
import numpy as np

# Parse the tracer file
def parse_tracer_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    right_radii = []
    left_radii = []
    current_side = None
    metadata = {}

    for line in lines:
        line = line.strip()
        if line.startswith('TRCFMT='):
            parts = line.split(';')
            if 'R' in parts:
                current_side = 'R'
            elif 'L' in parts:
                current_side = 'L'
        elif line.startswith('R='):
            values = line[2:].split(';')
            radii = [int(v) for v in values]
            if current_side == 'R':
                right_radii.extend(radii)
            elif current_side == 'L':
                left_radii.extend(radii)
        elif '=' in line and not line.startswith('R='):
            key, _, value = line.partition('=')
            metadata[key] = value

    return right_radii, left_radii, metadata

# Convert polar to cartesian
def polar_to_cartesian(radii, num_points=1000):
    # Radii are in 1/100 mm, convert to mm
    radii_mm = np.array(radii) / 100.0
    # Angles from 0 to 360 degrees (equidistant)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    x = radii_mm * np.cos(angles)
    y = radii_mm * np.sin(angles)

    return x, y

# Parse file
filepath = r'C:\Users\User\Desktop\shapes\2025-12-11_12-00-39.DAT'
right_radii, left_radii, metadata = parse_tracer_file(filepath)

print("=== HOYA GT5000 Tracer Data Analysis ===")
print(f"Job: {metadata.get('JOB', 'N/A')}")
print(f"DBL (Bridge): {metadata.get('DBL', 'N/A')} mm")
print(f"HBOX (Width): {metadata.get('HBOX', 'N/A')} mm")
print(f"VBOX (Height): {metadata.get('VBOX', 'N/A')} mm")
print(f"Circumference: {metadata.get('CIRC', 'N/A')} mm")
print(f"Right lens points: {len(right_radii)}")
print(f"Left lens points: {len(left_radii)}")

# Convert to cartesian coordinates
rx, ry = polar_to_cartesian(right_radii, len(right_radii))
lx, ly = polar_to_cartesian(left_radii, len(left_radii))

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Plot Right Lens
ax1 = axes[0]
ax1.plot(rx, ry, 'b-', linewidth=1.5)
ax1.fill(rx, ry, alpha=0.3, color='blue')
ax1.set_aspect('equal')
ax1.set_title('Right Lens (R)', fontsize=14, fontweight='bold')
ax1.set_xlabel('mm')
ax1.set_ylabel('mm')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Plot Left Lens
ax2 = axes[1]
ax2.plot(lx, ly, 'r-', linewidth=1.5)
ax2.fill(lx, ly, alpha=0.3, color='red')
ax2.set_aspect('equal')
ax2.set_title('Left Lens (L)', fontsize=14, fontweight='bold')
ax2.set_xlabel('mm')
ax2.set_ylabel('mm')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Plot Both Lenses Together (as worn - mirror left lens)
ax3 = axes[2]
dbl = float(metadata.get('DBL', '20'))
half_dbl = dbl / 2

# Right lens shifted right
ax3.plot(rx + half_dbl + np.max(rx), ry, 'b-', linewidth=1.5, label='Right')
ax3.fill(rx + half_dbl + np.max(rx), ry, alpha=0.3, color='blue')

# Left lens mirrored and shifted left
ax3.plot(-lx - half_dbl - np.max(lx), ly, 'r-', linewidth=1.5, label='Left')
ax3.fill(-lx - half_dbl - np.max(lx), ly, alpha=0.3, color='red')

ax3.set_aspect('equal')
ax3.set_title('Complete Frame (As Worn)', fontsize=14, fontweight='bold')
ax3.set_xlabel('mm')
ax3.set_ylabel('mm')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax3.legend()

# Add metadata text box
info_text = f"DBL: {metadata.get('DBL', 'N/A')} mm\nHBOX: {metadata.get('HBOX', 'N/A')} mm\nVBOX: {metadata.get('VBOX', 'N/A')} mm\nCIRC: {metadata.get('CIRC', 'N/A')} mm"
fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('HOYA GT5000 Eyeglass Frame Shape', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\User\Desktop\shapes\eyeglass_shape.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nShape saved to: eyeglass_shape.png")
