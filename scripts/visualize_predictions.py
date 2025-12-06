import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser(
    description="Visualize inpainting predictions: erased vs original vs inpainted"
)
parser.add_argument(
    "--test-dir",
    required=True,
    help="Path to test set directory (contains erased/, and optionally orig/ subdirectories)",
)
parser.add_argument(
    "--pred-dir",
    required=True,
    help="Path to predictions directory",
)
parser.add_argument(
    "--output",
    default=None,
    help="Output path for saved figure (default: {pred-dir}/comparison.png)",
)
parser.add_argument(
    "--triplet-cols",
    type=int,
    default=2,
    help="Number of triplet columns (default: 2)",
)
args = parser.parse_args()

test_dir = Path(args.test_dir)
pred_dir = Path(args.pred_dir)
output_path = args.output if args.output else pred_dir / "comparison.png"

# Check if orig/ directory exists
has_orig = (test_dir / "orig").exists()

# Find all prediction files
pred_files = sorted(pred_dir.glob("*_out_*.png"))

if not pred_files:
    print(f"No prediction files found in {pred_dir}")
    exit(1)

# Extract base names and collect image pairs/triplets
image_sets = []
for pred_path in pred_files:
    # Extract base name by removing _out_{model_name} suffix
    # Pattern: {base_name}_out_{model_name}.png
    match = re.match(r"(.+)_out_.+\.png$", pred_path.name)
    if not match:
        continue

    base_name = match.group(1)
    erased_path = test_dir / "erased" / f"{base_name}_erased.png"

    if not erased_path.exists():
        print(f"Warning: Missing erased file for {base_name}")
        continue

    image_set = {
        "name": base_name,
        "erased": erased_path,
        "inpainted": pred_path,
    }

    if has_orig:
        orig_path = test_dir / "orig" / f"{base_name}.png"
        if orig_path.exists():
            image_set["original"] = orig_path

    image_sets.append(image_set)

if not image_sets:
    print("No complete image sets found")
    exit(1)

print(f"Found {len(image_sets)} image sets")
print(f"Mode: {'Erased | Original | Inpainted' if has_orig else 'Erased | Inpainted'}")

# Determine number of images per set
images_per_set = 3 if has_orig else 2

# Layout: triplet_cols sets per row
n_images = len(image_sets)
set_cols = args.triplet_cols
n_rows = (n_images + set_cols - 1) // set_cols
n_cols = set_cols * images_per_set

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(2.5 * n_cols, 2.5 * n_rows),
    squeeze=False,
)

# Column headers
if has_orig:
    col_labels = ["Erased", "Original", "Inpainted"]
else:
    col_labels = ["Erased", "Inpainted"]

for idx, image_set in enumerate(image_sets):
    row_idx = idx // set_cols
    set_col = idx % set_cols
    col_base = set_col * images_per_set

    # Load images
    erased_img = Image.open(image_set["erased"])
    inpainted_img = Image.open(image_set["inpainted"])

    if has_orig and "original" in image_set:
        orig_img = Image.open(image_set["original"])
        images = [erased_img, orig_img, inpainted_img]
    else:
        images = [erased_img, inpainted_img]

    for col_offset, img in enumerate(images):
        col_idx = col_base + col_offset
        ax = axes[row_idx, col_idx]
        ax.imshow(img)
        ax.axis("off")

        # Add column headers on first row
        if row_idx == 0:
            ax.set_title(col_labels[col_offset], fontsize=12, fontweight="bold")

    # Add image name below the set's first image
    axes[row_idx, col_base].set_xlabel(image_set["name"], fontsize=8)

# Hide empty axes
for idx in range(n_images, n_rows * set_cols):
    row_idx = idx // set_cols
    set_col = idx % set_cols
    col_base = set_col * images_per_set
    for col_offset in range(images_per_set):
        axes[row_idx, col_base + col_offset].axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved comparison figure to: {output_path}")
