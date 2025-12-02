import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description="Compute L1 metric between prediction and ground truth"
)
parser.add_argument("--pred", required=True, help="Path to prediction image")
parser.add_argument("--orig", required=True, help="Path to original/ground truth image")
parser.add_argument("--mask", required=True, help="Path to mask image")
args = parser.parse_args()

pred = np.array(Image.open(args.pred)).astype(float)
orig = np.array(Image.open(args.orig)).astype(float)
mask = np.array(Image.open(args.mask).convert("L")).astype(float) / 255.0

# Full image L1 (mean absolute error)
l1_full = np.mean(np.abs(pred - orig))

# Masked region L1 (where mask > 0.5 = white = inpainted area)
mask_binary = mask > 0.5
if mask_binary.ndim == 2 and pred.ndim == 3:
    mask_binary = np.stack([mask_binary] * pred.shape[2], axis=-1)

masked_pixels_pred = pred[mask_binary]
masked_pixels_orig = orig[mask_binary]
l1_masked = np.mean(np.abs(masked_pixels_pred - masked_pixels_orig))

print(f"full:{l1_full:.6f}")
print(f"masked:{l1_masked:.6f}")
