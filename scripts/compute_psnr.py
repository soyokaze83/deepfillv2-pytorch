import argparse

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description="Compute PSNR metric between prediction and ground truth"
)
parser.add_argument("--pred", required=True, help="Path to prediction image")
parser.add_argument("--orig", required=True, help="Path to original/ground truth image")
parser.add_argument("--mask", required=True, help="Path to mask image")
args = parser.parse_args()

pred = np.array(Image.open(args.pred)).astype(float)
orig = np.array(Image.open(args.orig)).astype(float)
mask = np.array(Image.open(args.mask).convert("L")).astype(float) / 255.0

MAX_PIXEL = 255.0

# Full image PSNR
mse_full = np.mean((pred - orig) ** 2)
if mse_full == 0:
    psnr_full = float("inf")
else:
    psnr_full = 10 * np.log10((MAX_PIXEL ** 2) / mse_full)

# Masked region PSNR (where mask > 0.5 = white = inpainted area)
mask_binary = mask > 0.5
if mask_binary.ndim == 2 and pred.ndim == 3:
    mask_binary = np.stack([mask_binary] * pred.shape[2], axis=-1)

masked_pixels_pred = pred[mask_binary]
masked_pixels_orig = orig[mask_binary]
mse_masked = np.mean((masked_pixels_pred - masked_pixels_orig) ** 2)
if mse_masked == 0:
    psnr_masked = float("inf")
else:
    psnr_masked = 10 * np.log10((MAX_PIXEL ** 2) / mse_masked)

print(f"full:{psnr_full:.6f}")
print(f"masked:{psnr_masked:.6f}")
